#!/usr/bin/env python3
"""Convert existing JSONL bucket files to binary token format.

Reads bucket_XXXX.jsonl files, tokenizes every line, concatenates all tokens
per cluster into a flat int32 .bin file for zero-overhead mmap reading at
training time.

Usage:
    python -m torchtitan.experiments.cluster_data_selection.scripts.prepare_bins \
        --bucket_dir /path/to/data_sampled_300M \
        --tokenizer_path /path/to/llama-3.2-3B \
        --num_workers 16

Output (in-place alongside existing JSONL):
    bucket_0000.bin, bucket_0001.bin, ...
    meta.json updated with "format": "bin", "cluster_token_counts": [...]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


def tokenize_one_bucket(args: tuple) -> tuple[int, int, str]:
    """Tokenize a single bucket JSONL → .bin file.

    Args:
        args: (cluster_id, jsonl_path, bin_path, tokenizer_path, text_field)

    Returns:
        (cluster_id, token_count, status_message)
    """
    cluster_id, jsonl_path, bin_path, tokenizer_path, text_field = args

    # Each worker loads its own tokenizer (not picklable across processes)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    all_tokens = []
    num_docs = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(text_field)
            if not text:
                continue

            # Tokenize with BOS/EOS
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if bos_id is not None:
                token_ids = [bos_id] + token_ids
            if eos_id is not None:
                token_ids = token_ids + [eos_id]

            if len(token_ids) < 2:
                continue

            all_tokens.extend(token_ids)
            num_docs += 1

    # Write binary
    if all_tokens:
        arr = np.array(all_tokens, dtype=np.int32)
        arr.tofile(bin_path)
    else:
        # Write empty file
        np.array([], dtype=np.int32).tofile(bin_path)

    return (cluster_id, len(all_tokens), f"cluster {cluster_id}: {num_docs} docs, {len(all_tokens)} tokens")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL buckets to binary token files"
    )
    parser.add_argument(
        "--bucket_dir",
        type=str,
        required=True,
        help="Directory containing bucket_XXXX.jsonl and meta.json",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to HuggingFace tokenizer (e.g. llama-3.2-3B)",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="JSON field containing the text (default: text)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers for tokenization",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .bin files",
    )
    args = parser.parse_args()

    bucket_dir = Path(args.bucket_dir)
    meta_path = bucket_dir / "meta.json"

    if not meta_path.exists():
        print(f"Error: {meta_path} not found")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    num_clusters = meta["num_clusters"]
    print(f"Converting {num_clusters} clusters from JSONL → bin")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Workers: {args.num_workers}")
    print()

    # Build task list
    tasks = []
    skipped = 0
    for cluster_id in range(num_clusters):
        jsonl_path = bucket_dir / f"bucket_{cluster_id:04d}.jsonl"
        bin_path = bucket_dir / f"bucket_{cluster_id:04d}.bin"

        if not jsonl_path.exists():
            print(f"Warning: {jsonl_path} not found, skipping")
            skipped += 1
            continue

        if bin_path.exists() and not args.overwrite:
            skipped += 1
            continue

        tasks.append((
            cluster_id,
            str(jsonl_path),
            str(bin_path),
            args.tokenizer_path,
            args.text_field,
        ))

    if skipped > 0:
        print(f"Skipped {skipped} clusters (already converted or missing JSONL)")

    if not tasks:
        print("Nothing to convert!")
        # Still update meta if needed
        if "format" not in meta:
            _update_meta(bucket_dir, meta, num_clusters)
        return

    print(f"Converting {len(tasks)} clusters...")

    # Process in parallel
    cluster_token_counts = [0] * num_clusters
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(tokenize_one_bucket, t): t[0] for t in tasks}
        with tqdm(total=len(tasks), desc="Tokenizing") as pbar:
            for future in as_completed(futures):
                cluster_id, token_count, msg = future.result()
                cluster_token_counts[cluster_id] = token_count
                pbar.update(1)
                pbar.set_postfix_str(msg[-40:])

    # Fill in counts for skipped (already existing) bins
    for cluster_id in range(num_clusters):
        if cluster_token_counts[cluster_id] == 0:
            bin_path = bucket_dir / f"bucket_{cluster_id:04d}.bin"
            if bin_path.exists():
                size_bytes = bin_path.stat().st_size
                cluster_token_counts[cluster_id] = size_bytes // 4  # int32

    _update_meta(bucket_dir, meta, num_clusters, cluster_token_counts)


def _update_meta(bucket_dir, meta, num_clusters, cluster_token_counts=None):
    """Update meta.json with bin format info."""
    if cluster_token_counts is None:
        cluster_token_counts = []
        for cluster_id in range(num_clusters):
            bin_path = bucket_dir / f"bucket_{cluster_id:04d}.bin"
            if bin_path.exists():
                cluster_token_counts.append(bin_path.stat().st_size // 4)
            else:
                cluster_token_counts.append(0)

    meta["format"] = "bin"
    meta["token_dtype"] = "int32"
    meta["cluster_token_counts"] = cluster_token_counts

    total_tokens = sum(cluster_token_counts)
    meta_path = bucket_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Total tokens: {total_tokens:,} ({total_tokens*4/1e9:.1f} GB)")
    print(f"Updated {meta_path}")


if __name__ == "__main__":
    main()
