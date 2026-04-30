# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KMeans wrappers used by the offline ``prepare_clusters`` script.

We keep sklearn / faiss imports local so that the core online trainer
remains free of heavy offline dependencies.
"""

from __future__ import annotations

import numpy as np


def run_minibatch_kmeans(
    features: np.ndarray,
    *,
    n_clusters: int,
    n_init: int = 5,
    max_iter: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """sklearn MiniBatchKMeans — fast, memory-efficient, default choice."""
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        verbose=0,
    )
    return km.fit_predict(features).astype(np.int32)


def run_full_kmeans(
    features: np.ndarray,
    *,
    n_clusters: int,
    n_init: int = 5,
    max_iter: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """Full sklearn KMeans — more accurate, slower."""
    from sklearn.cluster import KMeans

    km = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        verbose=0,
    )
    return km.fit_predict(features).astype(np.int32)


def run_faiss_kmeans(
    features: np.ndarray,
    *,
    n_clusters: int,
    max_iter: int = 300,
    n_init: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Faiss GPU KMeans — fastest for very large datasets."""
    import faiss

    D = features.shape[1]
    features = np.ascontiguousarray(features, dtype=np.float32)
    kmeans = faiss.Kmeans(
        d=D,
        k=n_clusters,
        niter=max_iter,
        nredo=n_init,
        seed=seed,
        verbose=False,
        gpu=faiss.get_num_gpus() > 0,
    )
    kmeans.train(features)
    _, ids = kmeans.index.search(features, 1)
    return ids.flatten().astype(np.int32)


def run_kmeans(
    method: str,
    features: np.ndarray,
    *,
    n_clusters: int,
    n_init: int = 5,
    max_iter: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """Dispatch based on ``method``.

    ``random`` skips clustering entirely (uniform random assignment) and is
    useful as a baseline when comparing PMP against no clustering.
    """
    if method == "minibatch":
        return run_minibatch_kmeans(
            features,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            seed=seed,
        )
    if method == "kmeans":
        return run_full_kmeans(
            features,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            seed=seed,
        )
    if method == "faiss":
        return run_faiss_kmeans(
            features,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            seed=seed,
        )
    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.integers(0, n_clusters, size=features.shape[0], dtype=np.int32)
    raise ValueError(
        f"Unknown clustering method {method!r}. "
        "Expected one of: minibatch, kmeans, faiss, random."
    )
