# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cluster-selection trainer.

Subclasses :class:`torchtitan.trainer.Trainer` with the minimum surface area
needed to drive PMP cluster-weight updates.  Everything that makes torchtitan
robust at scale (FSDP2 / HSDP gather-schedule, DCP checkpointing, compile,
metrics, profiler, gradient accumulation, grad-norm clipping, lr scheduler,
etc.) is inherited verbatim.

Design rules
------------
* We do **not** modify torchtitan core.  Only ``trainer.py`` is subclassed.
* Supported parallelism: FSDP2 / HSDP only.  TP / PP / CP / EP all refuse to
  start, because PMP's ``autograd.grad`` path hasn't been validated on
  those meshes.  This is enforced in ``__post_init__`` of :class:`Config`.
* PMP runs only on the batch mesh (data-parallel).  Sketches are
  all-reduced on this mesh so HSDP "replica × shard" also works.
* Reclustering is explicitly unsupported online (the experiment uses
  offline bucketing); changing data mid-run requires re-running the
  ``prepare_clusters`` script.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from torchtitan.experiments.cluster_data_selection.config.job_config import (
    ClusterConfig,
)
from torchtitan.experiments.cluster_data_selection.data.dataloader import (
    ClusterDataLoader,
)
from torchtitan.experiments.cluster_data_selection.data.dev_dataset import (
    DevBatchCache,
)
from torchtitan.experiments.cluster_data_selection.pmp.count_sketch import (
    CountSketchProjector,
)
from torchtitan.experiments.cluster_data_selection.pmp.grad_utils_sketch import (
    compute_cluster_contributions_sketch,
)
from torchtitan.experiments.cluster_data_selection.pmp.weight_state import (
    ClusterWeightState,
)


class ClusterSelectionTrainer(Trainer):
    """Trainer that drives PMP cluster-weight updates during pre-training."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        cluster: ClusterConfig = field(default_factory=ClusterConfig)

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            p = self.parallelism
            # First-version contract: only FSDP2 / HSDP data-parallelism
            # configurations are supported.  Guard against silent breakage
            # on meshes we have not validated.
            if p.tensor_parallel_degree > 1:
                raise ValueError(
                    "cluster_data_selection: tensor_parallel_degree>1 is "
                    "not supported yet. PMP's autograd.grad path has only "
                    "been validated for FSDP2 / HSDP."
                )
            if p.pipeline_parallel_degree > 1:
                raise ValueError(
                    "cluster_data_selection: pipeline_parallel_degree>1 is "
                    "not supported yet (PMP needs full-model gradients on "
                    "every rank)."
                )
            if p.context_parallel_degree > 1:
                raise ValueError(
                    "cluster_data_selection: context_parallel_degree>1 is "
                    "not supported yet (dev / PMP forward would need to "
                    "shard sequences)."
                )
            if p.expert_parallel_degree > 1:
                raise ValueError(
                    "cluster_data_selection: expert_parallel_degree>1 is "
                    "not supported yet."
                )
            if not isinstance(self.dataloader, ClusterDataLoader.Config):
                raise ValueError(
                    "cluster_data_selection requires ClusterDataLoader.Config"
                    f" as dataloader, got {type(self.dataloader).__name__}."
                )

    # Additional state kept per trainer instance.
    cluster_config: ClusterConfig
    cluster_weight_state: ClusterWeightState
    count_sketch: CountSketchProjector
    dev_cache: DevBatchCache
    _cluster_dataset_ref: Any  # BucketedClusterDataset, avoid circular import

    @record
    def __init__(self, config: Config) -> None:
        # Defer all heavy lifting to the parent; this builds model, FSDP2,
        # optimizer, dataloader, checkpointer, etc.
        super().__init__(config)

        self.cluster_config = config.cluster
        pmp_cfg = config.cluster.pmp

        # Reach into the dataloader to grab the BucketedClusterDataset; we
        # validated the type in Config.__post_init__.
        cluster_loader = self.dataloader
        assert isinstance(cluster_loader, ClusterDataLoader)
        self._cluster_dataset_ref = cluster_loader.cluster_dataset
        num_clusters = self._cluster_dataset_ref.num_clusters

        self.cluster_weight_state = ClusterWeightState(
            num_clusters=num_clusters,
            temperature=pmp_cfg.temperature,
            min_weight=pmp_cfg.min_weight,
            accumulate=pmp_cfg.accumulate_grad_gamma,
            drop_bad_clusters=pmp_cfg.drop_bad_clusters,
            drop_patience=pmp_cfg.drop_patience,
        )
        # Ensure the dataset starts sampling from the canonical uniform
        # distribution (redundant but explicit).
        self._cluster_dataset_ref.update_weights(self.cluster_weight_state.weights)

        self.count_sketch = CountSketchProjector(
            sketch_dim=pmp_cfg.sketch_dim,
            seed=pmp_cfg.sketch_seed,
        )

        # Dev cache requires a tokenizer.  torchtitan has already built one.
        self.dev_cache = DevBatchCache(
            dev_dir=config.cluster.dev.dev_dir,
            tokenizer=self.tokenizer,
            text_field=config.cluster.dev.text_field,
            max_length=config.cluster.dev.max_length,
            max_samples=config.cluster.dev.max_samples,
            batch_size=pmp_cfg.dev_batch_size,
        )

        logger.info(
            "[ClusterSelectionTrainer] ready: num_clusters=%d, PMP enabled=%s, "
            "update_interval=%d",
            num_clusters,
            pmp_cfg.enabled,
            pmp_cfg.update_interval,
        )

    # ------------------------------------------------------------------
    # PMP hook
    # ------------------------------------------------------------------
    def _should_run_pmp(self) -> bool:
        pmp = self.cluster_config.pmp
        if not pmp.enabled:
            return False
        if pmp.update_interval <= 0:
            return False
        return (self.step % pmp.update_interval) == 0

    def _run_pmp(self) -> None:
        """Compute ``grad_gamma_delta`` on the current parameters, update
        cluster weights, and propagate them into the dataloader.

        Called right after a normal training step — the optimizer has
        already advanced ``θ_{t-1} → θ_t`` and ``zero_grad`` was executed
        at the top of :meth:`Trainer.train_step`, so ``param.grad`` slots
        are safe to reuse here (``autograd.grad`` doesn't touch them
        anyway).
        """
        pmp = self.cluster_config.pmp
        if len(self.model_parts) != 1:
            raise RuntimeError(
                "cluster_data_selection expects a single model_part "
                "(PP is disallowed by Config.__post_init__)."
            )
        model = self.model_parts[0]

        # Select the PMP mesh: FSDP2 / HSDP use the batch mesh; pure single
        # process uses None.
        dp_mesh = (
            self.parallel_dims.get_mesh("batch")
            if self.parallel_dims.dp_enabled
            else None
        )

        # Free training activations/gradients; PMP forward is memory-heavy.
        # Zero gradients explicitly to avoid stale state in case any tool
        # inspects them during PMP.  The helper below uses torch.autograd.grad
        # (not .backward()), so FSDP2's reduce-scatter hook stays untouched.
        self.optimizers.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        grad_gamma_delta = compute_cluster_contributions_sketch(
            model=model,
            dev_batches=self.dev_cache.iter_on_device(self.device),
            train_dataset=self._cluster_dataset_ref,
            sketcher=self.count_sketch,
            num_clusters=self.cluster_weight_state.num_clusters,
            pmp_lr=pmp.lr,
            n_samples_per_cluster=pmp.n_samples_per_cluster,
            dp_mesh=dp_mesh,
            device=self.device,
        )

        self.cluster_weight_state.update(grad_gamma_delta)
        # Broadcast into the on-disk sampler.
        self._cluster_dataset_ref.update_weights(
            self.cluster_weight_state.weights
        )
        # Drop any graph the sketcher may have pinned.
        del grad_gamma_delta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Override train_step to splice in PMP
    # ------------------------------------------------------------------
    def train_step(
        self,
        data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ) -> None:
        # Run the vanilla torchtitan step first.  It advances self.step
        # upstream in Trainer.train(), not inside train_step, so here we
        # just do the real work.
        super().train_step(data_iterator)

        if self._should_run_pmp():
            logger.info("[PMP] triggering at step=%d", self.step)
            self._run_pmp()

    # ------------------------------------------------------------------
    # Checkpoint plumbing: include PMP state so DCP persists weights.
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        parent = super().state_dict()
        parent["cluster_weight_state"] = self.cluster_weight_state.state_dict()
        return parent

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        cws_state = state_dict.pop("cluster_weight_state", None)
        super().load_state_dict(state_dict)
        if cws_state is not None:
            self.cluster_weight_state.load_state_dict(cws_state)
            self._cluster_dataset_ref.update_weights(
                self.cluster_weight_state.weights
            )
