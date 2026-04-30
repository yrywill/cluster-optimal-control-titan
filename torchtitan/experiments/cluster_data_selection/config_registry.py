# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ready-made configs for the cluster-data-selection experiment.

Invoke via torchtitan's normal launcher:

    MODULE=cluster_data_selection CONFIG=llama3_debug_cluster ./run_train.sh

Pick the flavor that matches your Llama3 scale, then override
``--cluster.bucket_dir`` / ``--cluster.dev.dev_dir`` on the CLI.
"""

from __future__ import annotations

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.models.llama3 import model_registry

from torchtitan.experiments.cluster_data_selection.config.job_config import (
    ClusterConfig,
    DevDataConfig,
    PMPConfig,
)
from torchtitan.experiments.cluster_data_selection.data.dataloader import (
    ClusterDataLoader,
)
from torchtitan.experiments.cluster_data_selection.trainer import (
    ClusterSelectionTrainer,
)


def _base_cluster_config() -> ClusterConfig:
    """Neutral defaults; each model flavor overrides the bits it cares about."""
    return ClusterConfig(
        bucket_dir="",  # MUST be overridden on CLI
        pmp=PMPConfig(
            enabled=True,
            update_interval=500,
            lr=0.01,
            temperature=1,
            min_weight=0.01,
            sketch_dim=8192,
            sketch_seed=42,
            n_samples_per_cluster=3,
            dev_batch_size=4,
            drop_bad_clusters=True,
            drop_patience=5,
        ),
        dev=DevDataConfig(
            dev_dir="",  # MUST be overridden on CLI
            text_field="text",
            max_samples=100,
            max_length=1024,
        ),
    )


def llama3_debug_cluster() -> ClusterSelectionTrainer.Config:
    """Tiny configuration suitable for smoke tests and CI.

    Use with ``COMM_MODE=fake_backend`` to validate the code path without a
    GPU.  Point the two required paths at any non-empty local folders
    (even if they only hold a handful of JSON files) before running for
    real.
    """
    return ClusterSelectionTrainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=1024,
            steps=20,
        ),
        dataloader=ClusterDataLoader.Config(
            bucket_dir="",  # override via --cluster.bucket_dir + --dataloader.bucket_dir
            text_field="text",
            seed=42,
            infinite=True,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(),
        checkpoint=CheckpointManager.Config(
            interval=20,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        validator=Validator.Config(enable=False),
        cluster=_base_cluster_config(),
    )


def llama3_3b_cluster() -> ClusterSelectionTrainer.Config:
    """3B training configuration derived from torchtitan's llama3 3B settings."""
    cfg = llama3_debug_cluster()
    cfg.model_spec = model_registry("3B")
    cfg.training = TrainingConfig(
        local_batch_size=2,
        seq_len=2048,
        steps=1000,
    )
    cfg.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=200,
        decay_ratio=0.8,
        decay_type="cosine",
        min_lr_factor=0.1,
    )
    cfg.optimizer = OptimizersContainer.Config(lr=3e-5)
    return cfg


def llama3_8b_cluster() -> ClusterSelectionTrainer.Config:
    cfg = llama3_3b_cluster()
    cfg.model_spec = model_registry("8B")
    cfg.training.local_batch_size = 1
    return cfg
