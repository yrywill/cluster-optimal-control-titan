# Cluster Data Selection (PMP) for torchtitan

This experiment ports the **Cluster-Based Optimal Data Selection** training
loop from the original `cluster_data_selection` repository on top of
torchtitan, preserving torchtitan's native multi-node multi-GPU training
machinery (FSDP2 / HSDP, DCP checkpointing, `torch.compile`, Float8,
metrics, profiler, validator, ...).

## Core idea

Traditional pre-training draws samples uniformly from the training corpus.
Here we first group training samples into **K clusters** (offline KMeans),
then *dynamically* reweight the sampling distribution during training using
a **Hessian-free PMP** (Perturbation-based Meta-Policy) update driven by a
held-out validation set:

```
every N grad steps:
    q     = sketch(∇L_dev(θ))                     # dev gradient sketch
    for each cluster k:
        v_k     = sketch(∇L_{C_k}(θ))            # cluster gradient sketch
        ct_k    = pmp_lr · <q, v_k>              # contribution
        grad_gamma[k] += ct_k
    w_k  ∝ exp(-grad_gamma[k] / temperature)     # new sampling weights
```

`CountSketch` keeps the gradient projection memory-bounded (~60 MB vs
tens-of-GB for an explicit projection matrix) and is linear — so it stays
correct when gradients are sharded under FSDP2.

## Quick start

### A. Baseline 预训练（无 PMP，纯随机采样）

从零预训练 Llama3-3B，数据均匀随机采样：

```bash
bash torchtitan/experiments/cluster_data_selection/start_train_no_pmp.sh
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TRAIN_LR` | 3e-4 | 学习率（从零预训标准值） |
| `TRAIN_STEPS` | 19000 | 训练步数（~160B tokens） |
| `TRAIN_SEQ_LEN` | 4096 | 序列长度（自动 packing，不浪费） |
| `TRAIN_WARMUP_STEPS` | 2000 | Warmup 步数 |
| `TRAIN_LOCAL_BATCH_SIZE` | 16 | 每 GPU batch size |
| `CKPT_INTERVAL` | 1000 | DCP checkpoint 保存间隔 |
| `BUCKET_DIR` | .../data_bucketed_by_semddp/final | 数据集路径 |

特性：
- 模型架构 = Llama3-3B（dim=3072, 28层, 24头），权重随机初始化
- 采样权重按 cluster 大小成比例（等价于全数据集均匀随机）
- Checkpoint 只存 DCP 格式（快速，不会 hang）
- 支持 SwanLab 实时 loss 画图
- 环境变量覆盖：`TRAIN_LR=1e-4 TRAIN_STEPS=5000 bash start_train_no_pmp.sh`

### B. PMP 数据选择训练

使用 PMP 动态调整 cluster 采样权重：

```bash
bash torchtitan/experiments/cluster_data_selection/start_cluster_train.sh
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TRAIN_LR` | 3e-4 | 学习率 |
| `TRAIN_STEPS` | 19000 | 训练步数 |
| `TRAIN_LOCAL_BATCH_SIZE` | 4 | 每 GPU batch size |
| `GRAD_ACC_STEPS` | 2 | 梯度累积步数 |
| `PMP_UPDATE_INTERVAL` | 100 | PMP 权重更新频率（steps） |
| `PMP_LR` | 0.01 | PMP 学习率 |
| `PMP_TEMPERATURE` | 1 | Softmax 温度 |
| `PMP_SKETCH_DIM` | 8192 | CountSketch 维度 |
| `BUCKET_DIR` | ... | 聚类数据目录 |
| `DEV_DIR` | ... | Dev 验证集目录 |

训练结束后 node 0 自动将 DCP checkpoint 转换为 HF 格式。

### C. DCP → HF 模型转换

训练完成后（或训练中断后），独立执行 checkpoint 格式转换：

```bash
# 在任意机器上执行（不需要 GPU，需要 ~12GB CPU 内存）
bash torchtitan/experiments/cluster_data_selection/convert_checkpoint.sh \
    /path/to/output/checkpoint
```

或者用 Python 脚本（更多选项）：

```bash
python3 scripts/convert_all_dcp_to_hf.py \
    /path/to/output/checkpoint \
    --model_flavor 3B \
    --export_dtype bfloat16 \
    --hf_assets_path /path/to/llama-3.2-3B \
    --tokenizer_path /path/to/llama-3.2-3B \
    --skip_existing \
    --validate
```

转换后的目录结构（和官方 Llama-3.2-3B 发布一致）：

```
checkpoint/global_step1000/hf/
├── config.json                         # 模型配置
├── generation_config.json              # 生成配置
├── model-00001-of-00002.safetensors    # 权重 shard 1
├── model-00002-of-00002.safetensors    # 权重 shard 2
├── model.safetensors.index.json        # 权重索引
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json
```

加载转换后的模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/path/to/global_step1000/hf")
tokenizer = AutoTokenizer.from_pretrained("/path/to/global_step1000/hf")
```

---

## Supported parallelism

| Parallelism | Status |
| --- | --- |
| FSDP2 / HSDP (`data_parallel_shard_degree` ≥ 1, `data_parallel_replicate_degree` ≥ 1) | ✅ Supported |
| Tensor Parallel (`tp > 1`) | ❌ Rejected at config time |
| Pipeline Parallel (`pp > 1`) | ❌ Rejected at config time |
| Context Parallel (`cp > 1`) | ❌ Rejected at config time |
| Expert Parallel (`ep > 1`) | ❌ Rejected at config time |

## Files

```
cluster_data_selection/
├── trainer.py                  # ClusterSelectionTrainer (subclass of Trainer)
├── config_registry.py          # llama3_debug_cluster / llama3_3b_cluster / ...
├── config/job_config.py        # ClusterConfig, PMPConfig, DevDataConfig
├── data/
│   ├── bucketed_dataset.py     # IterableDataset over pre-bucketed JSONL (token packing)
│   ├── dataloader.py           # ClusterDataLoader
│   └── dev_dataset.py          # DevBatchCache for PMP's dev gradient
├── pmp/
│   ├── count_sketch.py         # CountSketchProjector (FSDP2 DTensor-aware)
│   ├── grad_utils_sketch.py    # compute_cluster_contributions_sketch
│   └── weight_state.py         # ClusterWeightState (softmax + bad-cluster drop)
├── scripts/prepare_clusters.py # Offline: raw JSONL → bucket_XXXX.jsonl + meta.json
├── start_train_no_pmp.sh       # 启动脚本：baseline 训练（无 PMP）
├── start_cluster_train.sh      # 启动脚本：PMP 训练 + 自动转换
├── convert_checkpoint.sh       # 启动脚本：独立 DCP→HF 转换
└── tests/
    ├── test_count_sketch.py
    └── test_bucketed_dataset.py
```

## End-to-end workflow

### 0. (Optional) Download a public corpus

```bash
HF_ENDPOINT=https://hf-mirror.com \
python3 -m torchtitan.experiments.cluster_data_selection.scripts.download_dclm \
    --output_dir /path/to/dclm_raw \
    --num_files 50 \
    --num_workers 4 \
    --min_tokens 64 \
    --keep_url
```

### 1. Offline clustering & bucketing

Run once per corpus:

```bash
python -m torchtitan.experiments.cluster_data_selection.scripts.prepare_clusters \
    --input_dir  /path/to/raw_jsonl \
    --output_dir /path/to/buckets \
    --embed_model_path /path/to/qwen2.5-0.5B \
    --method minibatch \
    --cluster_size 500 \
    --shuffle_within_bucket
```

Output:

```
/path/to/buckets/
├── meta.json              # num_clusters, cluster_sizes, ...
├── bucket_0001.jsonl
├── bucket_0002.jsonl
└── ...
```

### 2. Training

Choose one:

```bash
# A. Baseline（无 PMP，纯随机）
bash torchtitan/experiments/cluster_data_selection/start_train_no_pmp.sh

# B. PMP 数据选择
bash torchtitan/experiments/cluster_data_selection/start_cluster_train.sh
```

### 3. Convert checkpoints to HF format

```bash
bash torchtitan/experiments/cluster_data_selection/convert_checkpoint.sh \
    /path/to/output/checkpoint
```

### 4. Debug without GPUs

```bash
NGPU=8 COMM_MODE=fake_backend \
    MODULE=cluster_data_selection CONFIG=llama3_debug_cluster \
    ./run_train.sh \
    --dataloader.bucket_dir=/path/to/small_buckets \
    --cluster.dev.dev_dir=/path/to/tiny_dev
```

### 5. Unit tests

```bash
pytest torchtitan/experiments/cluster_data_selection/tests/ -x
```

## Implementation notes

### Token packing

The dataloader uses greedy concatenation packing: short samples are
concatenated in a token buffer until `seq_len` is reached, then a window
is emitted. No padding, no wasted positions. Long samples are split across
multiple windows.

### Sampling weights

- **Without PMP**: weights are proportional to cluster size (equivalent to
  uniform random sampling from the full dataset, prevents over-fitting
  on tiny clusters).
- **With PMP**: weights are dynamically updated every `update_interval`
  steps based on gradient alignment with the dev set.

### PMP under FSDP2

* Parameters held by `fully_shard`-wrapped modules are `DTensor`s sharded
  over the batch mesh.  `CountSketchProjector._materialize` handles this
  transparently via `DTensor.full_tensor(grad_placements=[Replicate()])`.
* We call `torch.autograd.grad(loss, params)` instead of `loss.backward()`
  to avoid double-firing FSDP2's reduce-scatter hooks during PMP.
* Work distribution across DP ranks uses both dev-batch sharding and
  cluster sharding, each followed by an all-reduce on the batch mesh.

### Checkpointing

* Training saves DCP format only (fast, avoids NCCL hang on shared FS).
* HF conversion is done post-training on a single node (CPU only).
* `ClusterSelectionTrainer.state_dict` includes `cluster_weight_state` so
  PMP's accumulated `grad_gamma`, current weights, and dead-cluster flags
  all survive DCP saves/loads.

### SwanLab integration

Loss curves are automatically logged to SwanLab when enabled via
`--metrics.enable-swanlab`. Configure via environment variables:

```bash
export SWANLAB_API_KEY="your_key"
export SWANLAB_PROJECT="llama3-3b-pretrain"
export SWANLAB_EXPERIMENT_NAME="run_name"
```
