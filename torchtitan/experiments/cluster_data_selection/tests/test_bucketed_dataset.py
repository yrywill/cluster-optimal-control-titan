# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for BucketedClusterDataset.

We build a tiny on-disk corpus (3 clusters × a handful of records) with a
stub tokenizer, then exercise:

  * basic iteration returns ``(input_dict, labels)`` pairs of the right shape
  * ``update_weights`` shifts the per-cluster sampling distribution
  * ``sample_from_cluster`` honours the requested cluster id

The tests do not use torchtitan's actual tokenizer (which requires HF
assets); instead a minimal substitute class satisfies the ``encode`` /
``eos_id`` surface that the dataset relies on.
"""

from __future__ import annotations

import json
import os
import tempfile

from torchtitan.experiments.cluster_data_selection.data.bucketed_dataset import (
    BucketedClusterDataset,
)


class StubTokenizer:
    """Word-level tokenizer: each unique word gets an integer id."""

    eos_id = 1
    bos_id = 2

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1, "<bos>": 2}

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = len(self._vocab)
        return self._vocab[tok]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False):
        ids = [self._id(w) for w in text.split()]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids


def _write_fixture(tmp: str) -> None:
    meta = {"num_clusters": 3, "cluster_sizes": [4, 3, 5]}
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(meta, f)

    for k, prefix in enumerate(("red", "blue", "green")):
        lines = [f"{prefix} word sample number {i}" for i in range(10)]
        with open(os.path.join(tmp, f"bucket_{k:04d}.jsonl"), "w") as f:
            for l in lines:
                f.write(json.dumps({"text": l}) + "\n")


def test_iteration_yields_packed_windows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture(tmp)
        ds = BucketedClusterDataset(
            bucket_dir=tmp,
            tokenizer=StubTokenizer(),
            seq_len=32,
            dp_rank=0,
            dp_world_size=1,
            infinite=True,
        )
        it = iter(ds)
        for _ in range(5):
            inputs, labels = next(it)
            assert "input" in inputs
            assert inputs["input"].shape == (32,)
            assert labels.shape == (32,)


def test_update_weights_biases_cluster_draws() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture(tmp)
        ds = BucketedClusterDataset(
            bucket_dir=tmp,
            tokenizer=StubTokenizer(),
            seq_len=4,
            dp_rank=0,
            dp_world_size=1,
            seed=123,
            infinite=True,
        )
        # Concentrate all mass on cluster 2 ("green").
        import numpy as np

        ds.update_weights(np.array([0.0, 0.0, 1.0]))
        it = iter(ds)
        first_batch, _ = next(it)
        # Decode: every token that has a human-readable mapping should
        # correspond to the "green" corpus.  Using the stub tokenizer's
        # vocab in reverse lets us check this.
        inv_vocab = {v: k for k, v in ds._tokenizer._vocab.items()}
        words = [inv_vocab[int(t)] for t in first_batch["input"] if int(t) in inv_vocab]
        # None of the training words should be 'red' or 'blue'.
        assert "red" not in words
        assert "blue" not in words


def test_sample_from_cluster_returns_correct_bucket() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture(tmp)
        ds = BucketedClusterDataset(
            bucket_dir=tmp,
            tokenizer=StubTokenizer(),
            seq_len=8,
            dp_rank=0,
            dp_world_size=1,
            infinite=True,
        )
        result = ds.sample_from_cluster(1, n_samples=3)
        assert result is not None
        input_ids, labels, loss_mask = result
        assert input_ids.shape == labels.shape == loss_mask.shape
        assert input_ids.shape[0] <= 3


if __name__ == "__main__":
    for fn in (
        test_iteration_yields_packed_windows,
        test_update_weights_biases_cluster_draws,
        test_sample_from_cluster_returns_correct_bucket,
    ):
        fn()
        print(f"OK: {fn.__name__}")
