<div align="center">

# Cluster-Optimal-Control-Titan

#### Cluster-Based Optimal Data Selection for LLM Pre-training

Built on top of [torchtitan](https://github.com/pytorch/torchtitan) — a PyTorch native platform for training generative AI models.

</div>

## Overview

**Cluster-Optimal-Control-Titan** integrates **Cluster-Based Optimal Data Selection** into the torchtitan training framework. Instead of uniformly sampling from the training corpus, this system:

1. **Groups training data into K clusters** via offline KMeans clustering
2. **Dynamically reweights** the sampling distribution during training using a **Hessian-free PMP** (Perturbation-based Meta-Policy) update driven by a held-out validation set

This approach leverages torchtitan's native multi-node multi-GPU training machinery (FSDP2/HSDP, DCP checkpointing, `torch.compile`, Float8, metrics, profiler, etc.) while adding intelligent data selection on top.

### Core Algorithm

```
every N grad steps:
    q       = sketch(nabla L_dev(theta))              # dev gradient sketch
    for each cluster k:
        v_k     = sketch(nabla L_{C_k}(theta))       # cluster gradient sketch
        ct_k    = pmp_lr * <q, v_k>                   # contribution score
        grad_gamma[k] += ct_k
    w_k  ~ exp(-grad_gamma[k] / temperature)          # new sampling weights
```

`CountSketch` keeps the gradient projection memory-bounded (~60 MB vs tens-of-GB for an explicit projection matrix) and is linear, so it stays correct when gradients are sharded under FSDP2.

## Project Structure

```
cluster-optimal-control-titan/
├── torchtitan/                         # Core training framework
│   ├── train.py                        # Main training loop entry point
│   ├── config/                         # Configuration system
│   ├── components/                     # Reusable training components
│   │   ├── checkpoint.py               #   Distributed checkpointing
│   │   └── quantization/               #   Float8 / MXFP8 support
│   ├── distributed/                    # Parallelism utilities
│   │   ├── pipeline_parallel.py        #   Pipeline Parallel helpers
│   │   └── deepep/                     #   DeepEP integration
│   ├── models/                         # Supported model architectures
│   │   ├── llama3/                     #   Llama 3 / 3.1 (8B, 70B, 405B)
│   │   ├── llama4/                     #   Llama 4
│   │   ├── qwen3/                      #   Qwen3
│   │   ├── qwen3_vl/                   #   Qwen3-VL (Vision-Language)
│   │   ├── deepseek_v3/                #   DeepSeek V3 (MoE)
│   │   ├── flux/                       #   Flux (Diffusion)
│   │   ├── gpt_oss/                    #   GPT-OSS
│   │   └── common/                     #   Shared components (attention, MoE, RoPE, etc.)
│   ├── hf_datasets/                    # HuggingFace dataset integrations
│   ├── experiments/                    # Experimental features
│   │   ├── cluster_data_selection/     #   ** Core: Cluster-based data selection (PMP) **
│   │   │   ├── trainer.py              #     ClusterSelectionTrainer
│   │   │   ├── config/                 #     ClusterConfig, PMPConfig, DevDataConfig
│   │   │   ├── data/                   #     Bucketed dataset & cluster dataloader
│   │   │   ├── pmp/                    #     CountSketch projector & weight state
│   │   │   ├── clustering/             #     KMeans wrappers (offline)
│   │   │   ├── scripts/                #     Data preparation scripts
│   │   │   └── tests/                  #     Unit tests
│   │   ├── autoparallel/              #   Auto-parallelism experiments
│   │   ├── graph_trainer/             #   CUDA graph-based training
│   │   ├── ft/                        #   Fine-tuning & DiLoCo
│   │   ├── rl/                        #   Reinforcement learning
│   │   ├── vlm/                       #   Vision-Language model training
│   │   ├── forge/                     #   Forge experiments
│   │   └── transformers_modeling_backend/
│   ├── ops/                           # Custom operators
│   ├── protocols/                     # Protocol definitions
│   └── tools/                         # Utility tools
├── scripts/                           # Helper scripts
│   ├── checkpoint_conversion/         #   Checkpoint format conversion
│   ├── generate/                      #   Distributed inference
│   └── ci/                            #   CI utilities
├── tests/                             # Integration & unit tests
│   ├── integration_tests/
│   └── unit_tests/
├── docs/                              # Documentation
├── benchmarks/                        # Performance benchmarks
├── run_train.sh                       # Training launch script
├── multinode_trainer.slurm            # Multi-node SLURM job script
├── requirements.txt                   # Dependencies
└── pyproject.toml                     # Package configuration
```

## Key Features

### Cluster Data Selection (PMP)
- **Offline clustering & bucketing**: Pre-process corpus into K clusters via KMeans
- **Binary mmap format**: Pre-tokenize clusters → `.bin` files for zero I/O overhead (2.8x speedup vs JSONL)
- **Centralized sampling**: Rank 0 does global weighted draw → broadcast → each rank reads non-overlapping data (zero duplication)
- **Online PMP reweighting**: Dynamically adjust cluster sampling weights during training
- **CountSketch projector**: Memory-efficient gradient projection (~60 MB), FSDP2 DTensor-aware
- **Checkpointable state**: PMP weights, grad_gamma, cursors, and dead-cluster flags survive DCP saves/loads

### Inherited from torchtitan
- **Multi-dimensional parallelism**: FSDP2, Tensor Parallel (async TP), Pipeline Parallel, Context Parallel
- **Multiple model architectures**: Llama 3/4, Qwen3, DeepSeek V3, Flux, and more
- **Distributed checkpointing** with async support
- **`torch.compile`** for performance optimization
- **Float8 / MXFP8** training support
- **Flexible learning rate scheduler** (warmup-stable-decay)
- **Metrics logging** via TensorBoard or Weights & Biases

## Quick Start

### Installation

```bash
git clone https://github.com/yrywill/cluster-optimal-control-titan.git
cd cluster-optimal-control-titan
pip install -r requirements.txt
pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Step 1: Prepare Data (Offline Clustering)

```bash
# (Optional) Download a public corpus
HF_ENDPOINT=https://hf-mirror.com \
python3 -m torchtitan.experiments.cluster_data_selection.scripts.download_dclm \
    --output_dir /path/to/dclm_raw \
    --num_files 50 \
    --num_workers 4 \
    --min_tokens 64 \
    --keep_url

# Cluster and bucket the corpus
python -m torchtitan.experiments.cluster_data_selection.scripts.prepare_clusters \
    --input_dir  /path/to/raw_jsonl \
    --output_dir /path/to/buckets \
    --embed_model_path /path/to/qwen2.5-0.5B \
    --method minibatch \
    --cluster_size 500 \
    --shuffle_within_bucket
```

### Step 1.5: Convert to Binary Format (Recommended)

Pre-tokenize all clusters into binary `.bin` files for **zero-overhead mmap** reading at training time (no JSON parse, no tokenize, ~2.8x faster):

```bash
python -m torchtitan.experiments.cluster_data_selection.scripts.prepare_bins \
    --bucket_dir /path/to/buckets \
    --tokenizer_path /path/to/llama-3.2-3B \
    --num_workers 32
```

This creates `bucket_XXXX.bin` files alongside existing JSONL and updates `meta.json` with `"format": "bin"`. The dataloader auto-detects the format at runtime.

### Step 2: Pre-training with PMP Reweighting

#### Taiji Platform — 64 GPUs (8 nodes)

```bash
# PMP enabled, bin format auto-detected
HOST_NUM=8 \
TRAIN_LOCAL_BATCH_SIZE=4 \
GRAD_ACCUM_STEPS=4 \
BUCKET_DIR=/path/to/buckets \
DUMP_FOLDER=/path/to/output/train_pmp \
bash torchtitan/experiments/cluster_data_selection/start_train_pmp.sh
```

Global batch = 4 × 4 × 8 × 8 = 1024. PMP updates every 1000 steps.

#### Without PMP (baseline, size-proportional sampling)

```bash
HOST_NUM=8 \
TRAIN_LOCAL_BATCH_SIZE=4 \
GRAD_ACCUM_STEPS=4 \
BUCKET_DIR=/path/to/buckets \
DUMP_FOLDER=/path/to/output/train_no_pmp \
bash torchtitan/experiments/cluster_data_selection/start_train_no_pmp_fix.sh
```

#### Single Node — 8 GPUs

```bash
NGPU=8 MODULE=cluster_data_selection CONFIG=llama3_3b_cluster \
    ./run_train.sh \
    --dataloader.bucket_dir=/path/to/buckets \
    --cluster.dev.dev_dir=/path/to/mmlu_validation \
    --cluster.pmp.enabled \
    --cluster.pmp.update_interval=1000 \
    --cluster.pmp.lr=0.01 \
    --cluster.pmp.temperature=1
```

#### 2 Nodes — 16 GPUs

```bash
# Run on EACH of the 2 nodes (set MASTER_ADDR to node 0's IP):
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=8 --nnodes=2 \
    --node_rank=$NODE_RANK \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
    --module cluster_data_selection --config llama3_3b_cluster \
    --dataloader.bucket_dir=/path/to/buckets \
    --cluster.dev.dev_dir=/path/to/mmlu_validation \
    --cluster.pmp.update_interval=50 \
    --cluster.pmp.lr=0.1 \
    --cluster.pmp.temperature=0.5 \
    --parallelism.data_parallel_shard_degree=-1
```

#### 16 Nodes — 128 GPUs

```bash
# Run on EACH of the 16 nodes (set MASTER_ADDR to node 0's IP):
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=8 --nnodes=16 \
    --node_rank=$NODE_RANK \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
    --module cluster_data_selection --config llama3_8b_cluster \
    --dataloader.bucket_dir=/path/to/buckets \
    --cluster.dev.dev_dir=/path/to/mmlu_validation \
    --cluster.pmp.update_interval=100 \
    --cluster.pmp.lr=0.1 \
    --cluster.pmp.temperature=0.5 \
    --parallelism.data_parallel_shard_degree=-1
```

> **Note**: `--parallelism.data_parallel_shard_degree=-1` auto-computes the FSDP shard degree from `world_size / (dp_replicate * cp * tp * pp)`. For HSDP (replicate + shard), set both explicitly, e.g. `--parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=64` for 128 GPUs.

#### SLURM Multi-Node Launch

For clusters managed by SLURM, create a job script:

```bash
#!/bin/bash
#SBATCH --job-name=pmp-train
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

srun torchrun --nproc_per_node=8 --nnodes=16 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m torchtitan.train \
    --module cluster_data_selection --config llama3_8b_cluster \
    --dataloader.bucket_dir=/path/to/buckets \
    --cluster.dev.dev_dir=/path/to/mmlu_validation \
    --parallelism.data_parallel_shard_degree=-1
```

### Step 3: Standard torchtitan Training (without PMP)

```bash
# Llama 3 8B on 8 GPUs (single node)
NGPU=8 MODULE=llama3 CONFIG=llama3_8b ./run_train.sh

# Llama 3 8B on 16 GPUs (2 nodes)
torchrun --nproc_per_node=8 --nnodes=2 \
    --node_rank=$NODE_RANK \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    -m torchtitan.train --module llama3 --config llama3_8b
```

### Debug without GPUs

```bash
NGPU=8 COMM_MODE=fake_backend \
    MODULE=cluster_data_selection CONFIG=llama3_debug_cluster \
    ./run_train.sh \
    --dataloader.bucket_dir=/path/to/small_buckets \
    --cluster.dev.dev_dir=/path/to/tiny_dev
```

## Supported Parallelism for PMP

| Parallelism | Status |
| --- | --- |
| FSDP2 / HSDP | Supported |
| Tensor Parallel | Not yet supported |
| Pipeline Parallel | Not yet supported |
| Context Parallel | Not yet supported |
| Expert Parallel | Not yet supported |

> The PMP path relies on `torch.autograd.grad` over the whole model on every rank. Non-DP meshes need further validation. Guard rails refuse unsupported configurations at config time.

## Running Tests

```bash
# Unit tests (CPU only)
pytest torchtitan/experiments/cluster_data_selection/tests/ -x

# Full test suite
pytest tests/ -x
```

## Documentation

- [Checkpointing](docs/checkpoint.md)
- [FSDP2](docs/fsdp.md)
- [Float8 Training](docs/float8.md)
- [MXFP8 Training](docs/mxfp8.md)
- [Metrics & Logging](docs/metrics.md)
- [Extension Points](docs/extension.md)
- [Debugging Tools](docs/debugging.md)
- [Loss Convergence](docs/converging.md)

## Acknowledgements

This project is built on top of [torchtitan](https://github.com/pytorch/torchtitan) by Meta/PyTorch.

## Citation

```bibtex
@inproceedings{
   liang2025torchtitan,
   title={TorchTitan: One-stop PyTorch native solution for production ready {LLM} pretraining},
   author={Wanchao Liang and Tianyu Liu and Less Wright and Will Constable and Andrew Gu and Chien-Chin Huang and Iris Zhang and Wei Feng and Howard Huang and Junjie Wang and Sanket Purandare and Gokul Nadathur and Stratos Idreos},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=SFN6Wm7YBI}
}
```

## License

Source code is made available under a [BSD 3 license](./LICENSE). You may have other legal obligations that govern your use of other content linked in this repository, such as the license or terms of service for third-party data and models.
