# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bucketed-cluster iterable dataset.

The offline ``prepare_clusters`` script groups raw training samples into one
JSONL file per cluster.  At training time this dataset opens *all* buckets
simultaneously and, at each yield, picks a cluster proportionally to the
current sampling weights, then pulls the next line from that bucket.

Because weights change during training (PMP updates them), we expose
:meth:`BucketedClusterDataset.update_weights` — callable from the trainer
after each PMP step.  The dataset is :class:`Stateful` so torchtitan's DCP
checkpointing can resume mid-epoch.

Design notes / constraints:
  * Each bucket file is opened once per epoch; we round-robin within a
    bucket and re-open it when exhausted.  No extra shuffling: buckets are
    already shuffled by the offline script.
  * Data-parallel sharding happens via ``dp_rank`` / ``dp_world_size`` —
    each rank consumes a disjoint slice of GLOBAL samples by only yielding
    when ``global_step_in_epoch % dp_world_size == dp_rank``.
  * ``random.Random`` with a deterministic seed is used for cluster draws
    so the sequence is reproducible given the same weights history.
  * PMP also needs to sample arbitrary indices from an arbitrary cluster
    (see :meth:`sample_from_cluster`).  That path tokenises on demand and
    returns the raw model / label tensors — NOT the packed-sequence form
    used during training.

This module has NO dependency on HuggingFace Transformers or sklearn; all of
that lives in the offline script.
"""

from __future__ import annotations

import glob
import json
import os
import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger


_META_FILE = "meta.json"
_BUCKET_GLOB = "bucket_*.jsonl"


def _read_meta(bucket_dir: str) -> dict[str, Any]:
    meta_path = os.path.join(bucket_dir, _META_FILE)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Bucket directory {bucket_dir!r} is missing '{_META_FILE}'. "
            "Run 'python -m torchtitan.experiments.cluster_data_selection."
            "scripts.prepare_clusters' first."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _discover_bucket_files(bucket_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(bucket_dir, _BUCKET_GLOB)))
    if not files:
        raise FileNotFoundError(
            f"No {_BUCKET_GLOB!r} files found in {bucket_dir!r}."
        )
    return files


class BucketedClusterDataset(IterableDataset, Stateful):
    """Stream samples drawn by cluster weight from on-disk buckets.

    Args:
        bucket_dir: Directory produced by the offline clustering script.
            Must contain ``meta.json`` + ``bucket_XXXX.jsonl`` files.
        tokenizer: A torchtitan :class:`BaseTokenizer`.
        seq_len: Packed-sequence length for training.  The packer emits
            samples of exactly ``seq_len`` tokens each.
        dp_rank: Data-parallel rank; each rank yields only every
            ``dp_world_size``-th draw.
        dp_world_size: Total DP world size.
        text_field: Key in each JSON record whose value is the text.
        seed: Base seed for the cluster-draw RNG (shifted by epoch).
        infinite: If True, re-open buckets forever.

    The dataset is designed so every rank arrives at the same *cluster
    sequence* given the same weights and seed; only the yielded slice
    differs between ranks.  This lets PMP weight updates broadcast
    trivially — every rank recomputes the same cluster-draw order.
    """

    def __init__(
        self,
        bucket_dir: str,
        *,
        tokenizer: BaseTokenizer,
        seq_len: int,
        dp_rank: int,
        dp_world_size: int,
        text_field: str = "text",
        seed: int = 42,
        infinite: bool = True,
        within_bucket_order: str = "sequential",
    ) -> None:
        super().__init__()
        if within_bucket_order not in ("sequential", "random"):
            raise ValueError(
                f"within_bucket_order must be 'sequential' or 'random', "
                f"got {within_bucket_order!r}"
            )
        self._bucket_dir = bucket_dir
        self._bucket_files = _discover_bucket_files(bucket_dir)
        self._tokenizer = tokenizer
        self._seq_len = seq_len
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size
        self._text_field = text_field
        self._seed = seed
        self._infinite = infinite
        self._within_bucket_order = within_bucket_order

        meta = _read_meta(bucket_dir)
        self._num_clusters = int(meta["num_clusters"])
        if self._num_clusters != len(self._bucket_files):
            raise ValueError(
                f"meta.json reports num_clusters={self._num_clusters} but "
                f"{len(self._bucket_files)} bucket files are on disk."
            )
        self._cluster_sizes = np.asarray(meta["cluster_sizes"], dtype=np.int64)

        # Initial weights proportional to cluster size (avoids over-sampling
        # tiny clusters). Trainer calls :meth:`update_weights` after each PMP
        # backward pass to override these.
        size_weights = self._cluster_sizes.astype(np.float64)
        total = size_weights.sum()
        if total > 0:
            self._weights = size_weights / total
        else:
            self._weights = np.ones(self._num_clusters, dtype=np.float64) / max(
                self._num_clusters, 1
            )

        # RNG state — seed is offset by DP rank for data diversity across
        # ranks, BUT cluster-draw RNG uses the shared seed so every rank
        # samples the same cluster sequence (only the rank's own slice is
        # yielded).  The two RNGs are independent.
        self._cluster_rng = random.Random(seed)
        self._within_cluster_rng = random.Random(seed + 10_000 + dp_rank)

        self._step = 0  # global draw index (shared across ranks)
        self._epoch = 0

        # Cached text files (open lazily per worker process).
        self._file_handles: list[Any] = []

        # For random within-bucket order: lazily built byte offsets per
        # bucket file.  Each entry is a list of byte positions where
        # non-empty lines start, or None if not yet scanned.
        self._line_offsets: list[list[int] | None] = [None] * self._num_clusters

        # Packing buffer — appends sample tokens until we hit `seq_len+1`.
        self._token_buffer: list[int] = []

        # Per-sample index tracking is handled only implicitly via file
        # positions; on resume we re-open all files and fast-forward by
        # re-seeding the RNG with ``(seed, epoch)``.  That means long
        # mid-epoch jobs lose at most one epoch's worth of progress, which
        # is acceptable for PMP-style training where the curriculum is the
        # meaningful quantity, not exact sample ordering.

        logger.info(
            "[BucketedClusterDataset] num_clusters=%d, dp_rank=%d/%d, "
            "seq_len=%d, within_bucket_order=%s",
            self._num_clusters,
            dp_rank,
            dp_world_size,
            seq_len,
            within_bucket_order,
        )

    # ------------------------------------------------------------------
    # Cluster-weight API (called from trainer after PMP backward)
    # ------------------------------------------------------------------
    def update_weights(self, weights: np.ndarray | torch.Tensor) -> None:
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self._num_clusters,):
            raise ValueError(
                f"weights shape {weights.shape} != ({self._num_clusters},)"
            )
        total = float(weights.sum())
        if total <= 0.0:
            logger.warning(
                "[BucketedClusterDataset] weights sum to 0; falling back to uniform."
            )
            weights = np.ones_like(weights) / self._num_clusters
        else:
            weights = weights / total
        self._weights = weights

    @property
    def num_clusters(self) -> int:
        return self._num_clusters

    @property
    def cluster_sizes(self) -> np.ndarray:
        return self._cluster_sizes

    # ------------------------------------------------------------------
    # IterableDataset
    # ------------------------------------------------------------------
    def _open_all(self) -> None:
        self._close_all()
        self._file_handles = [
            open(f, "r", encoding="utf-8") for f in self._bucket_files
        ]

    def _close_all(self) -> None:
        for fh in self._file_handles:
            try:
                fh.close()
            except Exception:
                pass
        self._file_handles = []

    def _build_line_offsets(self, cluster_id: int) -> list[int]:
        """Scan a bucket file and record the byte offset of every non-empty line.

        The resulting index is cached in ``self._line_offsets[cluster_id]``
        so the cost is paid at most once per cluster per process.
        """
        offsets: list[int] = []
        path = self._bucket_files[cluster_id]
        with open(path, "r", encoding="utf-8") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(pos)
        self._line_offsets[cluster_id] = offsets
        return offsets

    def _read_random_line(self, cluster_id: int) -> str | None:
        """Read a uniformly random line from the given bucket file.

        Uses pre-built byte offsets to seek directly to the chosen line,
        avoiding the need to scan the entire file on every call.
        """
        offsets = self._line_offsets[cluster_id]
        if offsets is None:
            offsets = self._build_line_offsets(cluster_id)
        if not offsets:
            return None

        idx = self._within_cluster_rng.randrange(len(offsets))
        fh = self._file_handles[cluster_id]
        fh.seek(offsets[idx])
        line = fh.readline()
        if not line:
            return None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and self._text_field in obj:
            return obj[self._text_field]
        return None

    def _read_next_line_sequential(self, cluster_id: int) -> str | None:
        """Read the next line sequentially from a cluster, reopening on EOF."""
        fh = self._file_handles[cluster_id]
        line = fh.readline()
        if not line:
            # Bucket exhausted — reopen from top.
            fh.close()
            self._file_handles[cluster_id] = open(
                self._bucket_files[cluster_id], "r", encoding="utf-8"
            )
            line = self._file_handles[cluster_id].readline()
            if not line:
                return None  # truly empty bucket
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and self._text_field in obj:
            return obj[self._text_field]
        return None

    def _read_next_line(self, cluster_id: int) -> str | None:
        """Read the next text sample from a cluster.

        Dispatches to random or sequential reading based on the
        ``within_bucket_order`` setting.
        """
        if self._within_bucket_order == "random":
            return self._read_random_line(cluster_id)
        return self._read_next_line_sequential(cluster_id)

    def _draw_cluster(self) -> int:
        """Weighted categorical draw using the shared cluster RNG."""
        # random.choices is O(K) per draw which is fine for K up to ~10k.
        # For larger K we could switch to numpy's vectorised multinomial.
        return self._cluster_rng.choices(
            range(self._num_clusters), weights=self._weights.tolist(), k=1
        )[0]

    def __iter__(
        self,
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        if not self._file_handles:
            self._open_all()

        max_buffer = self._seq_len + 1

        while True:
            # Produce exactly one packed window per outer loop iteration.
            while len(self._token_buffer) < max_buffer:
                cluster_id = self._draw_cluster()
                step = self._step
                self._step += 1

                # Only THIS rank's slice is actually tokenised & yielded;
                # other ranks advance the same RNG in lock-step and will
                # process their own slice on their side.
                if step % self._dp_world_size != self._dp_rank:
                    # Still need to advance the corresponding bucket file so
                    # the workload is balanced across ranks; skipping file
                    # reads is fine because each rank owns an independent
                    # view of the buckets (multi-open is OK in POSIX).
                    continue

                text = self._read_next_line(cluster_id)
                if text is None:
                    continue

                token_ids = self._tokenizer.encode(
                    text, add_bos=True, add_eos=True
                )
                if len(token_ids) < 2:
                    continue
                self._token_buffer.extend(token_ids)

            # Yield one packed window.
            window = self._token_buffer[:max_buffer]
            self._token_buffer = self._token_buffer[max_buffer:]
            window_t = torch.tensor(window, dtype=torch.long)
            inputs = window_t[:-1]
            labels = window_t[1:]
            yield {"input": inputs}, labels

            if not self._infinite and self._step > 10 * max(
                sum(self._cluster_sizes), 1
            ):
                # Safety break for non-infinite smoke tests.
                break

    # ------------------------------------------------------------------
    # Stateful interface (DCP checkpointing)
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "epoch": self._epoch,
            "weights": self._weights.tolist(),
            "token_buffer": list(self._token_buffer),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        self._step = int(state_dict.get("step", 0))
        self._epoch = int(state_dict.get("epoch", 0))
        w = state_dict.get("weights")
        if w is not None:
            self.update_weights(np.asarray(w, dtype=np.float64))
        self._token_buffer = list(state_dict.get("token_buffer", []))
        # Re-seed cluster RNG so we deterministically resume the draw stream
        # from the saved ``step``.  We simulate by jumping the RNG forward.
        self._cluster_rng = random.Random(self._seed)
        for _ in range(self._step):
            self._cluster_rng.choices(
                range(self._num_clusters),
                weights=self._weights.tolist(),
                k=1,
            )

    # ------------------------------------------------------------------
    # PMP helper: draw a mini-batch from ONE specific cluster
    # ------------------------------------------------------------------
    def sample_from_cluster(
        self, cluster_id: int, n_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Read ``n_samples`` lines from the given bucket and tokenise them.

        Returns a triple of CPU tensors:
            input_ids [B, L], labels [B, L], loss_mask [B, L]
        or ``None`` if the bucket is empty / unreadable.

        Both shapes are padded to the maximum sequence length in the batch
        (capped at ``seq_len``).  This is the format expected by PMP's
        sketch-loss forward pass.
        """
        if not (0 <= cluster_id < self._num_clusters):
            return None

        # Open a short-lived file handle so we don't disturb the main
        # streaming iterator's position.
        path = self._bucket_files[cluster_id]
        texts: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                return None
            # Draw without replacement if bucket is big enough, else with.
            n_take = min(n_samples, len(lines))
            picked = self._within_cluster_rng.sample(lines, n_take)
            for line in picked:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and self._text_field in obj:
                    texts.append(obj[self._text_field])
        if not texts:
            return None

        # Tokenise.  Dev-like format: per-sample input/label shift, padded.
        encoded = [
            self._tokenizer.encode(t, add_bos=True, add_eos=True)[
                : self._seq_len + 1
            ]
            for t in texts
        ]
        encoded = [e for e in encoded if len(e) >= 2]
        if not encoded:
            return None

        max_len = min(max(len(e) for e in encoded) - 1, self._seq_len)
        B = len(encoded)
        pad_id = self._tokenizer.eos_id if self._tokenizer.eos_id is not None else 0

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
        loss_mask = torch.zeros((B, max_len), dtype=torch.float32)
        for i, ids in enumerate(encoded):
            ids = ids[: max_len + 1]
            seq_len = len(ids) - 1
            input_ids[i, :seq_len] = torch.tensor(ids[:-1], dtype=torch.long)
            labels[i, :seq_len] = torch.tensor(ids[1:], dtype=torch.long)
            loss_mask[i, :seq_len] = 1.0
        return input_ids, labels, loss_mask
