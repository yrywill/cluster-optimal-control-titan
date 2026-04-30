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
- **Online PMP reweighting**: Dynamically adjust cluster sampling weights during training
- **CountSketch projector**: Memory-efficient gradient projection (~60 MB), FSDP2 DTensor-aware
- **Checkpointable state**: PMP weights, grad_gamma, and dead-cluster flags survive DCP saves/loads

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

### Step 2: Pre-training with PMP Reweighting

```bash
MODULE=cluster_data_selection CONFIG=llama3_3b_cluster \
    ./run_train.sh \
    --dataloader.bucket_dir=/path/to/buckets \
    --cluster.dev.dev_dir=/path/to/mmlu_validation \
    --cluster.pmp.update_interval=50 \
    --cluster.pmp.lr=0.1 \
    --cluster.pmp.temperature=0.5
```

### Step 3: Standard torchtitan Training (without PMP)

```bash
# Llama 3 8B on 8 GPUs
MODULE=llama3 CONFIG=llama3_8b ./run_train.sh
```

### Debug without GPUs

```bash
NGPU=8 COMM_MODE=fake_backend \
    MODULE=cluster_data_selection CONFIG=llama3_debug_cluster \
    ./run_train.sh \
    --dataloader.bucket_dir=/path/to/small_buckets \
    --cluster.dev.dev_dir=/path/to/tiny_dev
```

### Multi-Node Training (SLURM)

Adjust node/GPU count in `multinode_trainer.slurm`, then:

```bash
sbatch multinode_trainer.slurm
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
