# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline clustering & bucketing script.

This script is the *only* place in the experiment that touches
HuggingFace Transformers, sklearn, and faiss.  The online training trainer
has no dependency on any of these modules.

Usage
-----

    # Single GPU (or CPU debug):
    python -m torchtitan.experiments.cluster_data_selection.scripts.prepare_clusters \
        --input_dir /path/to/raw_jsonl \
        --output_dir /path/to/buckets \
        --embed_model_path /path/to/qwen2.5-0.5B \
        --cluster_size 500 --method minibatch

    # Multi-GPU (recommended for >1M samples — each rank embeds a shard,
    # rank 0 gathers features, runs MiniBatchKMeans and writes buckets):
    torchrun --nproc_per_node=8 \
        -m torchtitan.experiments.cluster_data_selection.scripts.prepare_clusters \
        --input_dir /path/to/raw_jsonl \
        --output_dir /path/to/buckets \
        --embed_model_path /path/to/qwen2.5-0.5B \
        --cluster_size 500 --method minibatch

Output layout
-------------

    output_dir/
    ├── meta.json                 # num_clusters, cluster_sizes, config
    ├── cluster_ids.npy           # int32 [N] — assignment per input sample
    └── bucket_0000.jsonl ... bucket_XXXX.jsonl

Each bucket file contains one JSON object per line with the original text
in a ``"text"`` field.  The online dataloader reads these files directly
and never needs the feature matrix or the embedding model.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import sys

import numpy as np
import torch

from torchtitan.experiments.cluster_data_selection.clustering.kmeans import (
    run_kmeans,
)


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Text loader — deliberately duplicated from dev_dataset.py because this
# script is meant to run standalone without importing torchtitan core.
# ----------------------------------------------------------------------
def _load_texts(input_dir: str, text_field: str) -> list[str]:
    patterns = [
        os.path.join(input_dir, "*.json"),
        os.path.join(input_dir, "*.jsonl"),
        os.path.join(input_dir, "**", "*.json"),
        os.path.join(input_dir, "**", "*.jsonl"),
    ]
    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No .json / .jsonl under {input_dir!r}")

    texts: list[str] = []
    for path in files:
        # Try JSONL first, line by line.  If *any* line parses, treat the
        # file as JSONL even if a handful of records are mangled — large
        # web-scraped corpora routinely contain a small fraction of
        # truncated / malformed rows, and dropping the entire file because
        # of 0.4% bad lines would discard millions of samples.
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            per_line: list[str] = []
            jsonl_ok_any = False
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                jsonl_ok_any = True
                if isinstance(obj, dict) and text_field in obj:
                    val = obj[text_field]
                    if isinstance(val, str) and val:
                        per_line.append(val)
        if jsonl_ok_any:
            texts.extend(per_line)
            continue

        # Fallback: not JSONL — try parsing the whole file as a single
        # JSON value (array of records / dict with a records field /
        # single-record dict).
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and text_field in item:
                    val = item[text_field]
                    if isinstance(val, str) and val:
                        texts.append(val)
        elif isinstance(data, dict):
            for key in ("data", "items", "records", "samples"):
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict) and text_field in item:
                            val = item[text_field]
                            if isinstance(val, str) and val:
                                texts.append(val)
                    break
            else:
                if text_field in data:
                    val = data[text_field]
                    if isinstance(val, str) and val:
                        texts.append(val)
    return texts


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def _extract_embeddings(
    texts: list[str],
    *,
    embed_model_path: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    embed_layer: int,
    rank: int = 0,
    world_size: int = 1,
) -> np.ndarray:
    """Run the embedding model once per text, take the mean of an
    intermediate layer's hidden states.

    Uses output_hidden_states=True for portability (works on any HF arch,
    at the cost of one extra pass through the upper layers).

    When ``world_size > 1`` each rank only computes features for its own
    shard (``texts[rank::world_size]`` ordering);  the caller is
    responsible for gathering shards back into the global [N, H] matrix on
    rank 0.  Returned shape is ``[ceil(N/world_size) or less, H]``.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModel.from_pretrained(
        embed_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError(
            "Embedding model config is missing num_hidden_layers — "
            "cannot choose an intermediate layer."
        )
    target_layer = embed_layer if embed_layer >= 0 else num_layers // 2
    if not (0 <= target_layer <= num_layers):
        raise ValueError(
            f"embed_layer={target_layer} out of range [0, {num_layers}]"
        )
    if rank == 0:
        logger.info(
            "[prepare_clusters] Using hidden layer %d / %d for features",
            target_layer,
            num_layers,
        )

    # Contiguous shard: rank r gets texts[r*chunk : (r+1)*chunk].  We use a
    # contiguous layout (rather than strided) because the downstream gather
    # can then reconstruct the full-N ordering by simple concatenation of
    # the first ``total_shard_len`` entries per rank.
    n_total = len(texts)
    per_rank = (n_total + world_size - 1) // world_size
    start_idx = rank * per_rank
    end_idx = min(start_idx + per_rank, n_total)
    local_texts = texts[start_idx:end_idx]
    if rank == 0:
        logger.info(
            "[prepare_clusters] sharding N=%d across world_size=%d "
            "(per_rank≈%d, this rank=%d: %d texts)",
            n_total,
            world_size,
            per_rank,
            rank,
            len(local_texts),
        )

    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(local_texts), batch_size):
            chunk = local_texts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[target_layer]  # [B, L, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            features.append(pooled.cpu().float().numpy())
            if rank == 0 and (start // batch_size) % 20 == 0:
                logger.info(
                    "[prepare_clusters][rank0] %d / %d local texts embedded",
                    start + len(chunk),
                    len(local_texts),
                )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not features:
        # Rank with no samples (e.g. world_size > N): return zero-row matrix
        # with H inferred from tokenizer's hidden size — unlikely in practice.
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(features, axis=0)


def _gather_features(
    local_features: np.ndarray,
    *,
    world_size: int,
    device: torch.device,
) -> np.ndarray | None:
    """All-gather per-rank feature shards into a full [N, H] matrix on
    rank 0.  Other ranks return ``None``.

    Assumes contiguous sharding (rank 0 owns the first chunk, rank 1 the
    second, ...).  Uses ``dist.gather_object`` so shards can have
    different row counts without manual padding.
    """
    import torch.distributed as dist

    if world_size == 1:
        return local_features

    rank = dist.get_rank()
    # Convert to a picklable object.  For a 100k × ~896 fp32 matrix each
    # shard is ~40 MB — well within gather_object's comfort zone.
    obj = local_features
    gathered: list[np.ndarray] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(
        obj=obj,
        object_gather_list=gathered,
        dst=0,
    )
    if rank != 0:
        return None
    assert gathered is not None
    # Concatenate in rank order (matches the contiguous shard layout
    # produced by ``_extract_embeddings``).
    return np.concatenate(gathered, axis=0)


# ----------------------------------------------------------------------
# Bucket writer
# ----------------------------------------------------------------------
def _write_buckets(
    texts: list[str],
    cluster_ids: np.ndarray,
    output_dir: str,
    text_field: str,
    shuffle_within_bucket: bool,
    seed: int,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    num_clusters = int(cluster_ids.max()) + 1
    by_cluster: list[list[int]] = [[] for _ in range(num_clusters)]
    for idx, cid in enumerate(cluster_ids):
        by_cluster[int(cid)].append(idx)

    rng = random.Random(seed)
    cluster_sizes: list[int] = []
    for k, idxs in enumerate(by_cluster):
        if shuffle_within_bucket:
            rng.shuffle(idxs)
        path = os.path.join(output_dir, f"bucket_{k:04d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in idxs:
                f.write(json.dumps({text_field: texts[i]}, ensure_ascii=False))
                f.write("\n")
        cluster_sizes.append(len(idxs))
        if len(idxs) == 0:
            logger.warning("Cluster %d is empty.", k)

    return {
        "num_clusters": num_clusters,
        "cluster_sizes": cluster_sizes,
        "text_field": text_field,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _init_distributed() -> tuple[int, int, torch.device]:
    """Initialise torch.distributed if launched under torchrun; otherwise
    return (rank=0, world_size=1, cpu_or_cuda:0).

    Returns (rank, world_size, device).  Every rank is pinned to its own
    CUDA device based on ``LOCAL_RANK`` so feature extraction runs fully
    in parallel without stepping on each other's memory.
    """
    import torch.distributed as dist

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return rank, world_size, device

    # Non-distributed fallback
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return 0, 1, device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline KMeans clustering of a JSON(L) corpus, writing "
        "one bucket file per cluster for online training."
    )
    parser.add_argument("--input_dir", required=True, help="Raw JSONL corpus folder.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination for bucket_XXXX.jsonl + meta.json.",
    )
    parser.add_argument(
        "--embed_model_path",
        required=True,
        help="HF model used for feature extraction (e.g. qwen2.5-0.5B).",
    )
    parser.add_argument("--text_field", default="text")
    parser.add_argument(
        "--method",
        default="minibatch",
        choices=("minibatch", "kmeans", "faiss", "random"),
    )
    parser.add_argument("--cluster_size", type=int, default=500)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument(
        "--embed_layer",
        type=int,
        default=-1,
        help="Intermediate hidden-state layer index; -1 = middle layer.",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Cap on the number of input texts; -1 = no cap.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle_within_bucket",
        action="store_true",
        help="Shuffle each bucket's sample order on disk (recommended).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Override the device for single-process runs (e.g. 'cpu' for "
            "debugging).  Under torchrun each rank is pinned to its own "
            "CUDA device and this flag is ignored."
        ),
    )
    args = parser.parse_args()

    rank, world_size, device = _init_distributed()
    if args.device is not None and world_size == 1:
        device = torch.device(args.device)

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    if rank == 0:
        logger.info("[prepare_clusters] args=%s", vars(args))
        logger.info(
            "[prepare_clusters] world_size=%d device=%s",
            world_size,
            device,
        )

    texts = _load_texts(args.input_dir, args.text_field)
    if args.max_samples > 0 and len(texts) > args.max_samples:
        rng = random.Random(args.seed)
        texts = rng.sample(texts, args.max_samples)
    if rank == 0:
        logger.info("[prepare_clusters] loaded %d texts", len(texts))
    if not texts:
        raise SystemExit("No texts loaded — check --input_dir / --text_field.")

    n_clusters = max(1, len(texts) // args.cluster_size)
    if rank == 0:
        logger.info(
            "[prepare_clusters] target n_clusters=%d (N=%d, cluster_size=%d)",
            n_clusters,
            len(texts),
            args.cluster_size,
        )

    if args.method == "random":
        if rank != 0:
            return
        # Skip the embedding step entirely for the random baseline.
        cluster_ids = np.array(
            [i % n_clusters for i in range(len(texts))], dtype=np.int32
        )
        rng = random.Random(args.seed)
        rng.shuffle(cluster_ids.tolist())  # no-op on numpy, reorder below
        cluster_ids = np.random.default_rng(args.seed).permutation(cluster_ids)
    else:
        local_features = _extract_embeddings(
            texts,
            embed_model_path=args.embed_model_path,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            embed_layer=args.embed_layer,
            rank=rank,
            world_size=world_size,
        )

        if world_size > 1:
            import torch.distributed as dist

            # Ensure every rank is done before gathering.
            dist.barrier()
            features = _gather_features(
                local_features,
                world_size=world_size,
                device=device,
            )
            # Free non-rank-0 memory early — other ranks have no further work.
            del local_features
            if rank != 0:
                dist.barrier()  # rank 0 does kmeans + write + then releases us
                return
        else:
            features = local_features

        assert features is not None  # rank 0 only past this point
        logger.info(
            "[prepare_clusters] features shape=%s, running %s",
            features.shape,
            args.method,
        )
        cluster_ids = run_kmeans(
            args.method,
            features,
            n_clusters=n_clusters,
            n_init=args.n_init,
            max_iter=args.max_iter,
            seed=args.seed,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "cluster_ids.npy"), cluster_ids)

    meta = _write_buckets(
        texts=texts,
        cluster_ids=cluster_ids,
        output_dir=args.output_dir,
        text_field=args.text_field,
        shuffle_within_bucket=args.shuffle_within_bucket,
        seed=args.seed,
    )
    meta["method"] = args.method
    meta["cluster_size_target"] = args.cluster_size
    meta["source_dir"] = args.input_dir
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("[prepare_clusters] wrote %s", meta_path)

    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()  # release non-zero ranks
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
