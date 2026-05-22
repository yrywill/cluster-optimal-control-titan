"""Build visualization data for cluster explorer website.

Reads all cluster JSONL files, samples 20 texts per cluster, extracts
TF-IDF keywords for cluster naming, and outputs a single JSON file
for the web visualization.

Usage:
    python torchtitan/experiments/cluster_data_selection/scripts/build_cluster_viz_data.py \
        --bucket_dir /path/to/data_sampled_300M \
        --output torchtitan/experiments/cluster_data_selection/viz/cluster_viz_data.json \
        --num_samples 20 \
        --workers 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


# ============================================================
# TF-IDF keyword extraction (no external deps needed)
# ============================================================

# Common English stopwords
STOPWORDS = set(
    "a about above after again against all am an and any are aren't as at be "
    "because been before being below between both but by can't cannot could "
    "couldn't did didn't do does doesn't doing don't down during each few for "
    "from further get got had hadn't has hasn't have haven't having he he'd "
    "he'll he's her here here's hers herself him himself his how how's i i'd "
    "i'll i'm i've if in into is isn't it it's its itself let's me more most "
    "mustn't my myself no nor not of off on once only or other ought our ours "
    "ourselves out over own same shan't she she'd she'll she's should "
    "shouldn't so some such than that that's the their theirs them themselves "
    "then there there's these they they'd they'll they're they've this those "
    "through to too under until up very was wasn't we we'd we'll we're we've "
    "were weren't what what's when when's where where's which while who who's "
    "whom why why's will with won't would wouldn't you you'd you'll you're "
    "you've your yours yourself yourselves also just like one two three new "
    "first use used using many well may much even still however since said "
    "will can make way get go see come take know think want look find give "
    "tell work call try need feel become leave put mean keep let begin seem "
    "help show hear play run move live believe hold bring happen write provide "
    "sit stand lose pay meet include continue set learn change lead understand "
    "watch follow stop create speak read allow add spend grow open walk win "
    "offer remember love consider appear buy wait serve die send expect build "
    "stay fall cut reach kill remain suggest raise pass sell require report "
    "decide pull".split()
)


def tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase, keep alpha tokens > 2 chars."""
    return [
        w.lower()
        for w in re.findall(r"[a-zA-Z]+", text)
        if len(w) > 2 and w.lower() not in STOPWORDS
    ]


def extract_keywords(texts: list[str], top_k: int = 5) -> list[str]:
    """Extract top-k keywords from a collection of texts using TF-IDF-like scoring."""
    if not texts:
        return []

    # Term frequency across all texts in this cluster
    tf = Counter()
    doc_freq = Counter()
    n_docs = len(texts)

    for text in texts:
        words = tokenize(text)
        tf.update(words)
        # Document frequency: count unique words per doc
        doc_freq.update(set(words))

    if not tf:
        return []

    # TF-IDF score: tf * log(N / df)
    scores = {}
    for word, count in tf.items():
        df = doc_freq.get(word, 1)
        idf = math.log(n_docs / df + 1)
        scores[word] = count * idf

    # Return top-k by score
    top_words = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [w for w, _ in top_words]


# ============================================================
# Cluster processing
# ============================================================


def process_cluster(args: tuple) -> dict[str, Any] | None:
    """Process a single cluster: sample texts and extract keywords."""
    cluster_id, bucket_dir, cluster_size, num_samples, seed = args

    jsonl_path = os.path.join(bucket_dir, f"bucket_{cluster_id:04d}.jsonl")
    if not os.path.isfile(jsonl_path):
        return None

    if cluster_size == 0:
        return None

    # Sample lines
    rng = random.Random(seed + cluster_id)

    # For large files, read only what we need
    texts = []
    try:
        if cluster_size <= num_samples:
            # Read all
            with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        if text:
                            texts.append(text[:500])  # Truncate for viz
                    except json.JSONDecodeError:
                        continue
        else:
            # Random reservoir sampling - read first N lines then sample
            # For efficiency, just read the first 200 lines and sample from those
            candidates = []
            max_read = min(cluster_size, max(200, num_samples * 5))
            with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= max_read:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        if text:
                            candidates.append(text[:500])
                    except json.JSONDecodeError:
                        continue

            if len(candidates) <= num_samples:
                texts = candidates
            else:
                texts = rng.sample(candidates, num_samples)
    except Exception as e:
        print(f"  [WARN] cluster {cluster_id}: {e}", file=sys.stderr)
        return None

    if not texts:
        return None

    # Extract keywords → name
    keywords = extract_keywords(texts, top_k=5)
    name = " / ".join(keywords[:4]) if keywords else f"Cluster {cluster_id}"

    return {
        "id": cluster_id,
        "size": cluster_size,
        "name": name,
        "keywords": keywords,
        "samples": texts[:num_samples],
    }


def main():
    parser = argparse.ArgumentParser(description="Build cluster visualization data")
    parser.add_argument(
        "--bucket_dir",
        default="/apdcephfs_jn5/share_304380933/rongyiyu/data_sampled_300M",
        help="Directory with bucket_XXXX.jsonl + meta.json",
    )
    parser.add_argument(
        "--output",
        default="torchtitan/experiments/cluster_data_selection/viz/cluster_viz_data.json",
        help="Output JSON file for the visualization",
    )
    parser.add_argument("--num_samples", type=int, default=20, help="Texts per cluster")
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load metadata
    meta_path = os.path.join(args.bucket_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_clusters = int(meta["num_clusters"])
    cluster_sizes = meta["cluster_sizes"]
    total_samples = sum(cluster_sizes)

    print(f"Clusters: {num_clusters}, Total samples: {total_samples:,}")
    print(f"Sampling {args.num_samples} texts per cluster with {args.workers} workers...")

    # Build task list
    tasks = [
        (i + 1, args.bucket_dir, cluster_sizes[i], args.num_samples, args.seed)
        for i in range(num_clusters)
        if cluster_sizes[i] > 0
    ]

    # Process in parallel
    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_cluster, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            if done % 500 == 0:
                print(f"  Processed {done}/{len(tasks)} clusters...")
            result = future.result()
            if result is not None:
                results.append(result)

    # Sort by size descending
    results.sort(key=lambda x: -x["size"])

    print(f"Done: {len(results)} clusters with data (out of {num_clusters})")

    # Compute size distribution for the overview
    import numpy as np

    sizes_arr = np.array(cluster_sizes)
    size_brackets = [
        {"label": "1-10", "count": int(((sizes_arr >= 1) & (sizes_arr <= 10)).sum())},
        {"label": "11-100", "count": int(((sizes_arr > 10) & (sizes_arr <= 100)).sum())},
        {"label": "101-1K", "count": int(((sizes_arr > 100) & (sizes_arr <= 1000)).sum())},
        {"label": "1K-10K", "count": int(((sizes_arr > 1000) & (sizes_arr <= 10000)).sum())},
        {"label": "10K-50K", "count": int(((sizes_arr > 10000) & (sizes_arr <= 50000)).sum())},
        {"label": "50K+", "count": int((sizes_arr > 50000).sum())},
    ]

    output = {
        "total_clusters": num_clusters,
        "total_samples": total_samples,
        "clusters_with_data": len(results),
        "size_distribution": size_brackets,
        "clusters": results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    file_size_mb = os.path.getsize(args.output) / 1e6
    print(f"Output: {args.output} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
