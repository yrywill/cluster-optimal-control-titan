# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cluster-contribution computation for the PMP backward pass.

The algorithm is the Hessian-free variant described in the original
``cluster_data_selection`` paper:

    q       = sketch( ∇L_dev(θ) )                      # device shard-safe
    v_k     = sketch( ∇L_{C_k}(θ) )                    # per cluster k
    ct_k    = pmp_lr · <q, v_k>
    grad_gamma_delta[k] += ct_k

We compute gradients via :func:`torch.autograd.grad` (NOT ``loss.backward``)
so FSDP2's reduce-scatter / optimizer hooks stay untouched.

FSDP2 parameter-sharding consideration
--------------------------------------
Because we bypass ``.backward()`` we also bypass FSDP2's forward-hook that
all-gathers parameters.  That is safe: FSDP2's ``fully_shard`` applied
modules perform the gather *inside* their own forward — the forward pass
we call here is the same PyTorch forward that FSDP2 has instrumented, so
the gathers fire on-demand.  The resulting gradients, however, live as
DTensors (sharded over the batch mesh) exactly as if ``.backward()`` were
used.  CountSketch materialises them via ``full_tensor`` before sketching,
so we consume the same shape as in vanilla training.

Distribution of work across DP ranks
------------------------------------
* Dev batches are sharded across DP ranks (batch-mesh): rank ``r``
  processes the batches ``dev_batches[r::dp_world_size]``.  A final
  all-reduce over the batch mesh reconstructs ``q``.
* Cluster list is sharded across DP ranks the same way: rank ``r`` is
  responsible for clusters ``cluster_ids[r::dp_world_size]``, and a final
  all-reduce over the batch mesh sums ``grad_gamma_delta``.

With both reductions on the batch mesh (not world) we preserve support
for HSDP (replicas × shards) if/when the experiment grows into it.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh

from torchtitan.tools.logging import logger

from torchtitan.experiments.cluster_data_selection.data.bucketed_dataset import (
    BucketedClusterDataset,
)
from torchtitan.experiments.cluster_data_selection.pmp.count_sketch import (
    CountSketchProjector,
)


def _cross_entropy_sketch_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    trainable: list[tuple[str, nn.Parameter]],
    sketcher: CountSketchProjector,
) -> torch.Tensor:
    """Forward + backward + sketch, returning an ``[m]`` vector.

    Uses ``loss.backward()`` instead of ``torch.autograd.grad`` so that
    FSDP2's reduce-scatter hooks fire correctly and ``.grad`` is populated
    on the DTensor parameters.  We zero grads before and collect them after.
    """
    # Zero existing grads to isolate this forward/backward.
    for _, p in trainable:
        if p.grad is not None:
            p.grad.zero_()

    logits = model(input_ids)
    losses = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
    ).view(labels.shape)
    loss_sum = (losses * loss_mask).sum()
    denom = loss_mask.sum().clamp(min=1.0)
    loss = loss_sum / denom

    loss.backward()

    # Free graph refs as aggressively as possible.
    del logits, losses, loss_sum, loss

    # Collect .grad from parameters (populated by FSDP2's hooks).
    named_grads = [(n, p.grad) for n, p in trainable]
    try:
        return sketcher.sketch_named_grads(named_grads, device=input_ids.device)
    finally:
        # Clear grads to avoid accumulation across PMP iterations.
        for _, p in trainable:
            if p.grad is not None:
                p.grad.zero_()
        del named_grads


def compute_cluster_contributions_sketch(
    *,
    model: nn.Module,
    dev_batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    train_dataset: BucketedClusterDataset,
    sketcher: CountSketchProjector,
    num_clusters: int,
    pmp_lr: float,
    n_samples_per_cluster: int,
    dp_mesh: DeviceMesh | None,
    device: torch.device,
    skip_eval_mode: bool = False,
) -> torch.Tensor:
    """Return ``grad_gamma_delta`` of shape ``[num_clusters]``.

    Args:
        model: A plain :class:`torch.nn.Module` (an individual
            ``model_part``).  Under FSDP2 it should be the already-sharded
            module so forward triggers FSDP's gather hooks.
        dev_batches: Iterable of ``(input_ids, labels, loss_mask)`` already
            on ``device``.  Use :meth:`DevBatchCache.iter_on_device`.
        train_dataset: The :class:`BucketedClusterDataset` used for
            training; we call ``sample_from_cluster`` on it.
        sketcher: A shared :class:`CountSketchProjector`.
        num_clusters: Total K.
        pmp_lr: Scalar multiplier on each ``<q, v_k>``.
        n_samples_per_cluster: How many samples per cluster to sketch.
        dp_mesh: DP (batch) mesh used for all reductions.  ``None`` means
            single-process debug mode; we skip collectives then.
        device: Target device for all PMP tensors.
        skip_eval_mode: If True, keep the model in train mode during PMP.
            This avoids torch.compile recompilation when the model has no
            train/eval-dependent behavior (e.g. Llama3 has no dropout).
    """
    was_training = model.training
    if not skip_eval_mode:
        model.eval()
    try:
        dp_rank = dp_mesh.get_local_rank() if dp_mesh is not None else 0
        dp_world_size = dp_mesh.size() if dp_mesh is not None else 1

        trainable: list[tuple[str, nn.Parameter]] = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        ]
        if not trainable:
            raise RuntimeError("Model has no trainable parameters for PMP.")

        # ------------------------------------------------------------------
        # 1) Sketch the dev gradient  q = sketch(∇L_dev)
        # ------------------------------------------------------------------
        q = torch.zeros(sketcher.m, device=device, dtype=torch.float32)
        dev_batches_list = list(dev_batches)
        n_dev_local = 0
        # Pad so every rank runs the same number of forwards (FSDP2
        # all-gather is collective).
        max_dev = (len(dev_batches_list) + dp_world_size - 1) // dp_world_size
        for idx in range(max_dev):
            i = dp_rank + idx * dp_world_size
            if i < len(dev_batches_list):
                input_ids, labels, loss_mask = dev_batches_list[i]
                q = q + _cross_entropy_sketch_loss(
                    model,
                    input_ids,
                    labels,
                    loss_mask,
                    trainable,
                    sketcher,
                )
                n_dev_local += 1
            else:
                # Dummy forward to keep FSDP2 symmetric.
                if dev_batches_list:
                    d_ids, d_lab, d_mask = dev_batches_list[0]
                    _cross_entropy_sketch_loss(
                        model, d_ids, d_lab, d_mask, trainable, sketcher,
                    )
        if dp_mesh is not None and dp_world_size > 1:
            dist.all_reduce(q, op=dist.ReduceOp.SUM, group=dp_mesh.get_group())
        total_dev = max(len(dev_batches_list), 1)
        q = q / float(total_dev)
        logger.info(
            "[PMP] dev sketch: norm=%.4f, dev_total=%d, dev_local_rank%d=%d",
            float(q.norm()),
            total_dev,
            dp_rank,
            n_dev_local,
        )

        # ------------------------------------------------------------------
        # 2) For each cluster k, sketch v_k and accumulate ct_k = lr·<q,v_k>
        # ------------------------------------------------------------------
        grad_gamma_delta = torch.zeros(
            num_clusters, device=device, dtype=torch.float32
        )
        my_clusters = list(range(dp_rank, num_clusters, dp_world_size))
        # Pad so every rank runs the same number of model forwards (FSDP2
        # all-gather is collective and requires symmetric execution).
        max_local = (num_clusters + dp_world_size - 1) // dp_world_size
        n_local = 0
        for idx in range(max_local):
            k = my_clusters[idx] if idx < len(my_clusters) else None
            if k is not None:
                sample = train_dataset.sample_from_cluster(k, n_samples_per_cluster)
            else:
                sample = None
            if sample is not None:
                input_ids, labels, loss_mask = sample
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                loss_mask = loss_mask.to(device, non_blocking=True)
                v_k = _cross_entropy_sketch_loss(
                    model,
                    input_ids,
                    labels,
                    loss_mask,
                    trainable,
                    sketcher,
                )
                ct_k = torch.dot(q, v_k)
                grad_gamma_delta[k] = pmp_lr * ct_k
                n_local += 1
            else:
                # Dummy forward to keep FSDP2 all-gathers symmetric across
                # ranks.  Use the first dev batch as a throwaway input.
                if dev_batches_list:
                    d_ids, d_lab, d_mask = dev_batches_list[0]
                    _cross_entropy_sketch_loss(
                        model, d_ids, d_lab, d_mask, trainable, sketcher,
                    )
                    # Result is discarded — grad_gamma_delta is not touched.

        if dp_mesh is not None and dp_world_size > 1:
            dist.all_reduce(
                grad_gamma_delta,
                op=dist.ReduceOp.SUM,
                group=dp_mesh.get_group(),
            )

        logger.info(
            "[PMP] cluster sketch done: local_rank%d=%d, total_norm=%.4f",
            dp_rank,
            n_local,
            float(grad_gamma_delta.norm()),
        )
        return grad_gamma_delta
    finally:
        if not skip_eval_mode and was_training:
            model.train()
