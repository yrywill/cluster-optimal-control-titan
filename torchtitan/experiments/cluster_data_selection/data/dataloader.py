# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cluster-aware dataloader with centralized sampling.

Rank 0 performs global weighted sampling to decide how many samples each
cluster contributes per step, then broadcasts the allocation.  All ranks
then read their assigned portion from each cluster **in sequential order**,
guaranteeing no data duplication across ranks or across steps.

This replaces the previous design where each rank independently sampled
clusters — which could cause the same line to be read by multiple ranks
(especially for small clusters).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.experiments.cluster_data_selection.data.bin_dataset import (
    BinClusterDataset,
)
from torchtitan.experiments.cluster_data_selection.data.bucketed_dataset import (
    BucketedClusterDataset,
)
from torchtitan.tools.logging import logger


def _detect_bin_format(bucket_dir: str) -> bool:
    """Check if the bucket_dir uses bin format (from meta.json)."""
    import json
    import os

    meta_path = os.path.join(bucket_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return False
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta.get("format") == "bin"


class ClusterDataLoader(BaseDataLoader, Stateful):
    """DataLoader with centralized cluster sampling + parallel reading.

    Sampling contract (per ``next()`` call):
      1. Rank 0 draws ``local_batch_size * dp_world_size`` cluster IDs
         using weighted sampling (respecting current PMP weights).
      2. The full cluster-ID tensor is broadcast to all ranks.
      3. Each rank computes which lines from which clusters it should read
         (deterministic assignment based on rank), then reads + tokenizes
         in parallel.
      4. Global per-cluster cursors advance by the exact count used,
         ensuring sequential reads and zero duplication.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        bucket_dir: str = ""
        """Directory containing ``meta.json`` and ``bucket_XXXX.jsonl``."""

        text_field: str = "text"
        """JSON key for the training text in each bucket record."""

        seed: int = 42
        """Base RNG seed for cluster sampling on rank 0."""

        infinite: bool = True
        """Loop buckets forever (wrap cursors on EOF)."""

        within_bucket_order: str = "sequential"
        """Ignored in centralized mode (always sequential), kept for compat."""

    # Expose the dataset for PMP access.
    cluster_dataset: BucketedClusterDataset

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        dp_pg: dist.ProcessGroup | None = None,
        **kwargs,
    ) -> None:
        if not config.bucket_dir:
            raise ValueError(
                "ClusterDataLoader.Config.bucket_dir must be set "
                "(point it at the output of scripts.prepare_clusters)."
            )

        self._dp_world_size = dp_world_size
        self._dp_rank = dp_rank
        self._local_batch_size = local_batch_size
        self._global_batch_size = local_batch_size * dp_world_size
        self._seq_len = seq_len
        self._dp_pg = dp_pg
        self._infinite = config.infinite

        # Auto-detect format: bin (mmap) vs jsonl (legacy)
        self._use_bin = _detect_bin_format(config.bucket_dir)

        if self._use_bin:
            # Binary mmap mode: zero tokenize, zero parse
            self._bin_dataset = BinClusterDataset(
                bucket_dir=config.bucket_dir,
                seq_len=seq_len,
                infinite=config.infinite,
            )
            self._num_clusters = self._bin_dataset.num_clusters
            # Expose as cluster_dataset for trainer compatibility
            self.cluster_dataset = self._bin_dataset
            self._tokenizer = None
            logger.info(
                "[ClusterDataLoader] Using BIN format (mmap, zero-tokenize)"
            )
        else:
            # Legacy JSONL mode with on-the-fly tokenize
            self._bin_dataset = None
            self.cluster_dataset = BucketedClusterDataset(
                bucket_dir=config.bucket_dir,
                tokenizer=tokenizer,
                seq_len=seq_len,
                dp_rank=0,
                dp_world_size=1,
                text_field=config.text_field,
                seed=config.seed,
                infinite=config.infinite,
                within_bucket_order="sequential",
            )
            self._num_clusters = self.cluster_dataset.num_clusters
            self._tokenizer = tokenizer
            logger.info(
                "[ClusterDataLoader] Using JSONL format (on-the-fly tokenize)"
            )

        # Centralized RNG (only rank 0's matters, but all ranks maintain
        # the same state for determinism after broadcast).
        self._cluster_rng = random.Random(config.seed)
        self._step = 0

        # Per-cluster global cursor: how many lines have been consumed
        # across all ranks combined.  All ranks maintain the same cursor
        # (ensured by seeing the same broadcast each step).
        self._cursors = np.zeros(self._num_clusters, dtype=np.int64)

        # Packing buffers per rank (to handle seq packing locally)
        self._token_buffer: list[int] = []

        logger.info(
            "[ClusterDataLoader] centralized sampling: "
            "num_clusters=%d, dp_rank=%d/%d, local_batch=%d, global_batch=%d",
            self._num_clusters,
            dp_rank,
            dp_world_size,
            local_batch_size,
            self._global_batch_size,
        )

    # ------------------------------------------------------------------
    # Weight update API (called by PMP / trainer)
    # ------------------------------------------------------------------
    @property
    def weights(self) -> np.ndarray:
        return self.cluster_dataset._weights

    def update_weights(self, weights: np.ndarray | torch.Tensor) -> None:
        self.cluster_dataset.update_weights(weights)

    # ------------------------------------------------------------------
    # Centralized sampling + broadcast
    # ------------------------------------------------------------------
    def _sample_and_broadcast(self) -> torch.Tensor:
        """Rank 0 samples global_batch_size cluster IDs, broadcasts to all.

        Returns:
            int32 tensor of shape [global_batch_size] with cluster IDs (on CPU).
        """
        if self._dp_pg is None or self._dp_world_size == 1:
            # Single-GPU fallback: no communication needed
            cluster_ids = self._cluster_rng.choices(
                range(self._num_clusters),
                weights=self.cluster_dataset._weights.tolist(),
                k=self._global_batch_size,
            )
            return torch.tensor(cluster_ids, dtype=torch.int32)

        # Must use CUDA tensor for NCCL broadcast
        device = torch.cuda.current_device()
        if self._dp_rank == 0:
            cluster_ids = self._cluster_rng.choices(
                range(self._num_clusters),
                weights=self.cluster_dataset._weights.tolist(),
                k=self._global_batch_size,
            )
            cluster_ids_tensor = torch.tensor(
                cluster_ids, dtype=torch.int32, device=device
            )
        else:
            cluster_ids_tensor = torch.empty(
                self._global_batch_size, dtype=torch.int32, device=device
            )

        dist.broadcast(cluster_ids_tensor, src=0, group=self._dp_pg)
        return cluster_ids_tensor.cpu()

    def _assign_and_read(
        self, cluster_ids_tensor: torch.Tensor
    ) -> list[torch.Tensor]:
        """Given broadcast cluster IDs, each rank reads its assigned portion.

        The assignment is deterministic: sample i goes to rank (i % dp_world).
        Within each cluster, samples are read sequentially from the global
        cursor position, guaranteeing no overlap.

        Returns:
            List of token tensors (each [seq_len+1]), length = local_batch_size.
        """
        cluster_ids = cluster_ids_tensor.tolist()

        # Count how many samples each cluster contributes globally
        cluster_counts = np.zeros(self._num_clusters, dtype=np.int64)
        for cid in cluster_ids:
            cluster_counts[cid] += 1

        # Determine this rank's assignment
        per_cluster_counter = np.zeros(self._num_clusters, dtype=np.int64)
        my_assignments: list[tuple[int, int]] = []  # (cluster_id, offset_within_step)

        for i, cid in enumerate(cluster_ids):
            if i % self._dp_world_size == self._dp_rank:
                my_assignments.append((cid, int(per_cluster_counter[cid])))
            per_cluster_counter[cid] += 1

        # Read samples
        results: list[torch.Tensor] = []

        if self._use_bin:
            # === BIN MODE: mmap read, zero tokenize ===
            for cid, offset_in_step in my_assignments:
                # Each sample in bin mode consumes seq_len+1 tokens from cursor
                token_offset = int(self._cursors[cid]) * (self._seq_len + 1) + offset_in_step * (self._seq_len + 1)
                window = self._bin_dataset.read_window(cid, token_offset)
                if window is None:
                    results.append(torch.zeros(self._seq_len + 1, dtype=torch.long))
                else:
                    results.append(window)
        else:
            # === JSONL MODE: read + tokenize ===
            for cid, offset_in_step in my_assignments:
                global_line_idx = int(self._cursors[cid]) + offset_in_step
                text = self.cluster_dataset.read_line_at(cid, global_line_idx)
                if text is None:
                    results.append(torch.zeros(self._seq_len + 1, dtype=torch.long))
                    continue
                token_ids = self._tokenizer.encode(text, add_bos=True, add_eos=True)
                if len(token_ids) < 2:
                    results.append(torch.zeros(self._seq_len + 1, dtype=torch.long))
                    continue
                token_ids = token_ids[: self._seq_len + 1]
                if len(token_ids) < self._seq_len + 1:
                    pad_id = (
                        self._tokenizer.eos_id
                        if self._tokenizer.eos_id is not None
                        else 0
                    )
                    token_ids = token_ids + [pad_id] * (
                        self._seq_len + 1 - len(token_ids)
                    )
                results.append(torch.tensor(token_ids, dtype=torch.long))

        # Advance global cursors (all ranks do this identically)
        self._cursors += cluster_counts
        self._step += 1

        return results

    # ------------------------------------------------------------------
    # Iterator interface
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        return self

    def __next__(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Produce one local batch [local_batch_size, seq_len]."""
        cluster_ids_tensor = self._sample_and_broadcast()
        token_tensors = self._assign_and_read(cluster_ids_tensor)

        # Stack into batch: each tensor is [seq_len + 1]
        batch = torch.stack(token_tensors)  # [local_batch_size, seq_len + 1]
        inputs = batch[:, :-1]  # [local_batch_size, seq_len]
        labels = batch[:, 1:]  # [local_batch_size, seq_len]

        return {"input": inputs}, labels

    # ------------------------------------------------------------------
    # Stateful interface (DCP checkpointing)
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "cursors": self._cursors.tolist(),
            "weights": self.cluster_dataset._weights.tolist(),
            "rng_state": self._cluster_rng.getstate(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        self._step = int(state_dict.get("step", 0))
        cursors = state_dict.get("cursors")
        if cursors is not None:
            self._cursors = np.asarray(cursors, dtype=np.int64)
        w = state_dict.get("weights")
        if w is not None:
            self.cluster_dataset.update_weights(np.asarray(w, dtype=np.float64))
        rng_state = state_dict.get("rng_state")
        if rng_state is not None:
            self._cluster_rng.setstate(rng_state)

