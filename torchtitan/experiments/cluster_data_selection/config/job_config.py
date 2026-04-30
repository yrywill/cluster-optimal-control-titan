# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config schema for the cluster-data-selection experiment.

The entire experiment contributes a single nested block under
``Trainer.Config``.  All fields are plain dataclasses so torchtitan's ``tyro``
CLI can override them (e.g. ``--cluster.pmp.update_interval=100``).
"""

from dataclasses import dataclass, field


@dataclass
class ClusteringConfig:
    """Parameters for the OFFLINE clustering + bucketing stage.

    Online training reads pre-bucketed ``bucket_XXXX.jsonl`` files from
    ``bucket_dir``; no clustering happens during training itself.  Keep these
    fields for bookkeeping and for the ``prepare_clusters`` script.
    """

    method: str = "minibatch"
    """Clustering backend: ``minibatch`` | ``kmeans`` | ``faiss``."""

    cluster_size: int = 1000
    """Target samples per cluster; ``num_clusters = max(1, N // cluster_size)``."""

    feature: str = "intermediate"
    """Feature source: only ``intermediate`` (early-exit hidden states) is
    supported online; ``embedding`` is also supported offline."""

    embed_layer: int = -1
    """Transformer layer index for ``intermediate`` features; ``-1`` = middle layer."""

    feature_batch_size: int = 2048
    """Batch size used while extracting features in the offline script."""

    embed_model_path: str = ""
    """Optional lightweight embedding model (e.g. qwen2.5-0.5B) used ONLY by
    the offline script.  Empty string disables it and uses the main model."""

    n_init: int = 5
    """KMeans ``n_init`` (number of restarts)."""

    max_iter: int = 300
    """KMeans maximum iterations."""

    seed: int = 42
    """Random seed for clustering (also used by MiniBatchKMeans)."""


@dataclass
class PMPConfig:
    """PMP (Perturbation-based Meta-Policy) cluster-weight update settings."""

    enabled: bool = True
    """If False the trainer reduces to plain torchtitan (no re-weighting)."""

    update_interval: int = 50
    """Run PMP backward and update cluster weights every N gradient steps."""

    lr: float = 0.1
    """Learning-rate factor applied to ``<sketch(∇L_dev), sketch(∇L_k)>``."""

    temperature: float = 0.5
    """Softmax temperature used when converting ``grad_gamma`` → weights.

    Lower temperature concentrates sampling on high-scoring clusters."""

    min_weight: float = 0.01
    """Floor per cluster to prevent starvation before re-normalisation."""

    accumulate_grad_gamma: bool = True
    """If True, accumulate ``grad_gamma`` across PMP calls; else reset every call."""

    drop_bad_clusters: bool = True
    """Permanently drop clusters with consecutive negative contributions."""

    drop_patience: int = 5
    """Consecutive negative updates before a cluster is dropped."""

    sketch_dim: int = 8192
    """CountSketch output dimension.  8192 is the paper's recommendation."""

    sketch_seed: int = 42
    """Random seed for CountSketch hash/sign tables."""

    n_samples_per_cluster: int = 4
    """How many random samples per cluster are used to compute ``∇L_k``."""

    dev_batch_size: int = 4
    """Batch size used when computing ``∇L_dev`` during PMP."""


@dataclass
class DevDataConfig:
    """Location and tokenisation details for the held-out validation set
    that drives the PMP objective.  Completely independent from torchtitan's
    training dataloader and from torchtitan's ``Validator``."""

    dev_dir: str = ""
    """Folder containing ``*.json`` / ``*.jsonl`` dev files (e.g. MMLU)."""

    text_field: str = "text"
    """Key name to read from each JSON record."""

    max_samples: int = 100
    """Cap on dev samples pulled for PMP (negative = no cap)."""

    max_length: int = 1024
    """Max token length per dev sample."""


@dataclass
class ClusterConfig:
    """Top-level block attached to ``ClusterSelectionTrainer.Config``."""

    bucket_dir: str = ""
    """Directory produced by ``scripts.prepare_clusters``.  Must contain
    ``meta.json`` and ``bucket_XXXX.jsonl`` files."""

    pmp: PMPConfig = field(default_factory=PMPConfig)
    dev: DevDataConfig = field(default_factory=DevDataConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
