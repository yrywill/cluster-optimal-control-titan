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


def _default_parallelism() -> ParallelismConfig:
    """Default HSDP parallelism: fixed 8-GPU FSDP shard groups.

    Layout by world size:
      -  8 GPUs: dp_shard=8, dp_replicate=1  (pure FSDP)
      - 16 GPUs: dp_shard=8, dp_replicate=2  (2 HSDP groups)
      - 128 GPUs: dp_shard=8, dp_replicate=16 (16 HSDP groups)

    The ``dp_replicate`` degree is resolved at runtime via the ``-1``
    convention: torchtitan sets ``dp_replicate = world_size / dp_shard``
    when dp_replicate is ``-1``.  However, since torchtitan only allows
    dp_shard to be ``-1`` (not dp_replicate), we fix dp_shard=8 and leave
    dp_replicate to be inferred from the relationship:
        dp_replicate * dp_shard * cp * tp * pp == world_size
    Since dp_shard is fixed at 8 and other dims default to 1,
    dp_replicate is effectively world_size // 8.

    For this to work we set dp_shard=8 (fixed) and dp_replicate=-1
    (auto-infer from world_size).  Note: torchtitan's ParallelDims only
    supports dp_shard=-1 natively; dp_replicate must be explicit.
    So we hardcode dp_shard=8 and set dp_replicate=1 here as the safe
    default.  For multi-node runs, override via CLI:
        --parallelism.data_parallel_replicate_degree=2   (16 GPUs)
        --parallelism.data_parallel_replicate_degree=16  (128 GPUs)
    """
    return ParallelismConfig(
        data_parallel_shard_degree=8,
        data_parallel_replicate_degree=1,
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
        parallelism=_default_parallelism(),
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


def llama3_3b_cluster_16gpu() -> ClusterSelectionTrainer.Config:
    """3B config for 16 GPUs (2 nodes x 8 GPUs), HSDP mode.

    dp_shard=8 (within-node FSDP), dp_replicate=2 (cross-node replication).
    Global batch size = local_batch_size(8) * dp_shard(8) * dp_replicate(2) = 128.
    Global tokens/step = 128 * 2048 = 262,144.
    """
    cfg = llama3_debug_cluster()
    cfg.hf_assets_path = "/apdcephfs_jn4/share_304380933/rongyiyu/code/llama-3.2-3B"
    cfg.model_spec = model_registry("3B")
    cfg.dump_folder = "/apdcephfs_jn5/share_304380933/rongyiyu/output"
    cfg.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=1000,
    )
    cfg.optimizer = OptimizersContainer.Config(lr=1e-5)
    cfg.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=200,
        decay_ratio=0.8,
        decay_type="cosine",
        min_lr_factor=0.1,
    )
    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        data_parallel_replicate_degree=1,  # overridden at runtime by launch script
    )
    cfg.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=200,
        last_save_model_only=True,
        last_save_in_hf=False,
    )
    cfg.cluster = ClusterConfig(
        bucket_dir="",  # override via CLI
        pmp=PMPConfig(
            enabled=True,
            update_interval=100,
            lr=0.01,
            temperature=1,
            min_weight=0.01,
            sketch_dim=8192,
            sketch_seed=42,
            n_samples_per_cluster=4,
            dev_batch_size=8,
            drop_bad_clusters=True,
            drop_patience=5,
        ),
        dev=DevDataConfig(
            dev_dir="",  # override via CLI
            text_field="text",
            max_samples=1000,
            max_length=1024,
        ),
    )
    return cfg


def llama3_3b_cluster_128gpu() -> ClusterSelectionTrainer.Config:
    """3B config for 128 GPUs (16 nodes x 8 GPUs), HSDP mode.

    dp_shard=8 (within-node FSDP), dp_replicate=16 (cross-node replication).
    Global batch size = local_batch_size(8) * dp_shard(8) * dp_replicate(16) = 1024.
    Global tokens/step = 1024 * 2048 = 2,097,152.
    """
    cfg = llama3_debug_cluster()
    cfg.model_spec = model_registry("3B")
    cfg.dump_folder = "/apdcephfs_jn5/share_304380933/rongyiyu/output"
    cfg.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=1000,
    )
    cfg.optimizer = OptimizersContainer.Config(lr=1e-5)
    cfg.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=200,
        decay_ratio=0.8,
        decay_type="cosine",
        min_lr_factor=0.1,
    )
    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        data_parallel_replicate_degree=16,
    )
    cfg.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=200,
        last_save_model_only=True,
        last_save_in_hf=False,
    )
    cfg.cluster = ClusterConfig(
        bucket_dir="",  # override via CLI
        pmp=PMPConfig(
            enabled=True,
            update_interval=100,
            lr=0.01,
            temperature=1,
            min_weight=0.01,
            sketch_dim=8192,
            sketch_seed=42,
            n_samples_per_cluster=4,
            dev_batch_size=8,
            drop_bad_clusters=True,
            drop_patience=5,
        ),
        dev=DevDataConfig(
            dev_dir="",  # override via CLI
            text_field="text",
            max_samples=1000,
            max_length=1024,
        ),
    )
    return cfg