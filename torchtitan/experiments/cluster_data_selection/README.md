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

## Supported parallelism (first version)

| Parallelism | Status |
| --- | --- |
| FSDP2 / HSDP (`data_parallel_shard_degree` ≥ 1, `data_parallel_replicate_degree` ≥ 1) | ✅ Supported |
| Tensor Parallel (`tp > 1`) | ❌ Rejected at config time |
| Pipeline Parallel (`pp > 1`) | ❌ Rejected at config time |
| Context Parallel (`cp > 1`) | ❌ Rejected at config time |
| Expert Parallel (`ep > 1`) | ❌ Rejected at config time |

The PMP path relies on `torch.autograd.grad` over the whole model on every
rank; the non-DP meshes need further work before they can be validated
numerically.  Guard rails in `ClusterSelectionTrainer.Config.__post_init__`
refuse unsupported configurations instead of silently mis-computing.

## Files

```
cluster_data_selection/
├── trainer.py                  # ClusterSelectionTrainer (subclass of Trainer)
├── config_registry.py          # llama3_debug_cluster / llama3_3b_cluster / ...
├── config/job_config.py        # ClusterConfig, PMPConfig, DevDataConfig, ClusteringConfig
├── data/
│   ├── bucketed_dataset.py     # IterableDataset over pre-bucketed JSONL
│   ├── dataloader.py           # ClusterDataLoader (wraps ParallelAwareDataloader)
│   └── dev_dataset.py          # DevBatchCache for PMP's dev gradient
├── pmp/
│   ├── count_sketch.py         # CountSketchProjector (FSDP2 DTensor-aware)
│   ├── grad_utils_sketch.py    # compute_cluster_contributions_sketch
│   └── weight_state.py         # ClusterWeightState (softmax + bad-cluster drop)
├── clustering/kmeans.py        # MiniBatch / Full / Faiss wrappers (offline only)
├── scripts/prepare_clusters.py # Offline: raw JSONL → bucket_XXXX.jsonl + meta.json
└── tests/
    ├── test_count_sketch.py
    └── test_bucketed_dataset.py
```

## End-to-end usage

### 0. (Optional) Download a public corpus

If you don't have a JSONL corpus handy, ``scripts.download_dclm`` pulls
[DCLM-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)
and slims each record to ``{"text": ..., "url": ...}`` ready for clustering:

```bash
# Behind a firewall, set HF_ENDPOINT to a mirror before running.
HF_ENDPOINT=https://hf-mirror.com \
python3 -m torchtitan.experiments.cluster_data_selection.scripts.download_dclm \
    --output_dir /path/to/dclm_raw \
    --num_files 50 \
    --num_workers 4 \
    --min_tokens 64 \
    --keep_url
```

Each upstream shard yields ~61k records and ~5 MB of slimmed JSONL.  The
script is resumable: interrupted files are left as ``*.part`` and skipped
unless you pass ``--overwrite``.

### 1. Offline clustering & bucketing

Run once per corpus.  Touches HuggingFace Transformers / sklearn; the
online trainer does not.

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
├── cluster_ids.npy        # int32 [N]
├── bucket_0000.jsonl
├── bucket_0001.jsonl
└── ...
```

### 2. Pre-training with PMP reweighting

Use torchtitan's standard launcher:

```bash
MODULE=cluster_data_selection CONFIG=llama3_3b_cluster \
    ./run_train.sh \
    --dataloader.bucket_dir=/path/to/buckets \
    --cluster.dev.dev_dir=/path/to/mmlu_validation \
    --cluster.pmp.update_interval=50 \
    --cluster.pmp.lr=0.1 \
    --cluster.pmp.temperature=0.5
```

The trainer:

1. Builds the torchtitan model + FSDP2 shards + optimizer as usual.
2. Opens all bucket files, initialises cluster weights uniformly.
3. Runs normal training steps (fully delegates to `Trainer.train_step`).
4. Every `pmp.update_interval` steps, computes a new `grad_gamma_delta`
   via CountSketch and pushes fresh cluster weights into the dataset.

### 3. Debug without GPUs

torchtitan's fake backend works out-of-the-box:

```bash
NGPU=8 COMM_MODE=fake_backend \
    MODULE=cluster_data_selection CONFIG=llama3_debug_cluster \
    ./run_train.sh \
    --dataloader.bucket_dir=/path/to/small_buckets \
    --cluster.dev.dev_dir=/path/to/tiny_dev
```

### 4. Unit tests

```bash
pytest torchtitan/experiments/cluster_data_selection/tests/ -x
```

The CountSketch and bucketed-dataset tests run entirely on CPU and require
only numpy / torch.

## Implementation notes

### PMP under FSDP2

* Parameters held by `fully_shard`-wrapped modules are `DTensor`s sharded
  over the batch mesh.  `CountSketchProjector._materialize` handles this
  transparently via `DTensor.full_tensor(grad_placements=[Replicate()])`.
* We call `torch.autograd.grad(loss, params)` instead of `loss.backward()`
  to avoid double-firing FSDP2's reduce-scatter hooks during PMP.  Forward
  still fires FSDP2's gather hooks as designed.
* Work distribution across DP ranks uses both the dev-batch sharding
  (`dev_batches[r::dp_world_size]`) and the cluster sharding
  (`clusters[r::dp_world_size]`), each followed by an all-reduce on the
  batch mesh.  No double counting because each cluster is visited by
  exactly one rank.

### Checkpointing

`ClusterSelectionTrainer.state_dict` includes `cluster_weight_state` so
PMP's accumulated `grad_gamma`, current weights, negative-streak counters,
and dead-cluster flags all survive DCP saves/loads.  The bucketed dataset
itself is `Stateful` and persists its draw index + packing buffer through
`ParallelAwareDataloader`'s normal mechanism.

### Why offline bucketing?

Training-time in-memory clustering (the original repo's default for small
corpora) is not scalable: 10M+ samples would dominate startup.  Offline
bucketing:

* Makes the online trainer **completely independent** of sklearn, faiss,
  and HuggingFace Transformers — keeping torchtitan's experiment folder
  reproducible and lean.
* Lets multiple training runs share the same clustering artefact.
* Allows the offline script to be run on arbitrarily large hardware
  without blocking the training launch.

## Things intentionally left out

* **Ghost projection / ring-buffer JVP paths**: the original repo's
  `CountSketch` fast path supersedes them and the JVP path drags in a lot
  of complexity.  We keep only the fast path for the first version.
* **Online re-clustering (`recluster_interval`)**: skipped; rerun
  `prepare_clusters` and restart training if you need a fresh bucketing.
* **Multi-domain dev weighting (`DevDomainManager`)**: the dev set is a
  single folder.  Re-introduce if a specific experiment requires domain
  reweighting of the PMP objective.
* **DeepSpeed**: torchtitan uses FSDP2 natively; the DeepSpeed branch of
  the original repo is obsolete here.
