"""Memory-mapped binary cluster dataset.

Each cluster is stored as a flat int32 .bin file containing concatenated
tokenized text (with BOS/EOS markers between documents).  At training time,
we mmap each file and read fixed-size windows directly — no tokenization,
no JSON parsing, no seeking.

This is a drop-in replacement for BucketedClusterDataset when the bucket
directory contains .bin files (indicated by meta.json "format": "bin").
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.tools.logging import logger


class BinClusterDataset(Stateful):
    """Memory-mapped binary dataset for cluster-based training.

    Each cluster has a `.bin` file (int32 tokens, concatenated).
    Reading a window of `seq_len+1` tokens is a single numpy slice —
    zero I/O overhead once OS page cache is warm.

    Args:
        bucket_dir: Directory with bucket_XXXX.bin + meta.json.
        seq_len: Training sequence length (will read seq_len+1 tokens per sample).
        infinite: Wrap around when cursor exceeds file length.
    """

    def __init__(
        self,
        bucket_dir: str,
        *,
        seq_len: int,
        infinite: bool = True,
    ) -> None:
        self._bucket_dir = bucket_dir
        self._seq_len = seq_len
        self._infinite = infinite

        # Load metadata
        meta_path = os.path.join(bucket_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta.get("format") == "bin", (
            f"BinClusterDataset requires format='bin' in meta.json, "
            f"got {meta.get('format')!r}. Run prepare_bins.py first."
        )

        self._num_clusters = int(meta["num_clusters"])
        self._cluster_sizes = np.asarray(meta["cluster_sizes"], dtype=np.int64)
        self._cluster_token_counts = np.asarray(
            meta["cluster_token_counts"], dtype=np.int64
        )

        # Initial weights proportional to TOKEN count (not doc count)
        # so clusters with more text get proportionally more sampling
        token_weights = self._cluster_token_counts.astype(np.float64)
        total = token_weights.sum()
        if total > 0:
            self._weights = token_weights / total
        else:
            self._weights = np.ones(self._num_clusters, dtype=np.float64) / max(
                self._num_clusters, 1
            )

        # Open mmap for each cluster
        self._mmaps: list[np.ndarray | None] = []
        for k in range(self._num_clusters):
            bin_path = os.path.join(bucket_dir, f"bucket_{k:04d}.bin")
            if os.path.exists(bin_path) and self._cluster_token_counts[k] > 0:
                mmap = np.memmap(bin_path, dtype=np.int32, mode="r")
                self._mmaps.append(mmap)
            else:
                self._mmaps.append(None)

        logger.info(
            "[BinClusterDataset] Loaded %d clusters, total %.1fB tokens, "
            "seq_len=%d",
            self._num_clusters,
            total / 1e9,
            seq_len,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_clusters(self) -> int:
        return self._num_clusters

    @property
    def cluster_sizes(self) -> np.ndarray:
        return self._cluster_sizes

    @property
    def cluster_token_counts(self) -> np.ndarray:
        return self._cluster_token_counts

    # ------------------------------------------------------------------
    # Weight API (for PMP)
    # ------------------------------------------------------------------
    def update_weights(self, weights: np.ndarray | torch.Tensor) -> None:
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        weights = np.asarray(weights, dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            weights = np.ones_like(weights) / self._num_clusters
        else:
            weights = weights / total
        self._weights = weights

    # ------------------------------------------------------------------
    # Core reading: zero-copy window from mmap
    # ------------------------------------------------------------------
    def read_window(self, cluster_id: int, offset: int) -> torch.Tensor | None:
        """Read seq_len+1 tokens from cluster at the given token offset.

        Wraps around if offset exceeds the cluster's token count.
        Returns int64 tensor of shape [seq_len+1], or None if cluster is empty.
        """
        mmap = self._mmaps[cluster_id]
        if mmap is None:
            return None

        length = self._seq_len + 1
        total_tokens = len(mmap)

        if total_tokens < length:
            # Cluster too small — tile it
            repeats = (length // total_tokens) + 1
            tiled = np.tile(mmap, repeats)[:length]
            return torch.from_numpy(tiled.astype(np.int64))

        # Wrap offset
        start = offset % total_tokens

        if start + length <= total_tokens:
            # Fast path: contiguous read
            window = mmap[start : start + length]
        else:
            # Wrap around file boundary
            tail = mmap[start:]
            head = mmap[: length - len(tail)]
            window = np.concatenate([tail, head])

        return torch.from_numpy(window.astype(np.int64))

    # ------------------------------------------------------------------
    # PMP helper: sample_from_cluster
    # ------------------------------------------------------------------
    def sample_from_cluster(
        self, cluster_id: int, n_samples: int, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Read n_samples windows from cluster for PMP sketch computation.

        Returns:
            (input_ids [B, L], labels [B, L], loss_mask [B, L]) or None.
        """
        mmap = self._mmaps[cluster_id]
        if mmap is None:
            return None

        total_tokens = len(mmap)
        if total_tokens < self._seq_len + 1:
            return None

        L = self._seq_len
        input_ids = torch.zeros(n_samples, L, dtype=torch.long)
        labels = torch.zeros(n_samples, L, dtype=torch.long)
        loss_mask = torch.ones(n_samples, L, dtype=torch.float32)

        for i in range(n_samples):
            start = (offset + i * (L + 1)) % total_tokens
            window = self.read_window(cluster_id, start)
            if window is None:
                loss_mask[i] = 0
                continue
            input_ids[i] = window[:-1]
            labels[i] = window[1:]

        return input_ids, labels, loss_mask

    # ------------------------------------------------------------------
    # Stateful (DCP checkpointing)
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "weights": self._weights.tolist(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        w = state_dict.get("weights")
        if w is not None:
            self.update_weights(np.asarray(w, dtype=np.float64))
