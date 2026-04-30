#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Download & convert DCLM-baseline-1.0 shards for cluster-data-selection.

Upstream dataset:
    https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0

Upstream layout::

    dclm-baseline-1.0/
      global-shard_01_of_10/
        local-shard_0_of_10/
          shard_00000000_processed.jsonl.zst   # ~150 MB → 61k records
          shard_00000001_processed.jsonl.zst
          ...
        local-shard_1_of_10/ ...
      global-shard_02_of_10/ ...
      ...

Each record has many fields (metadata, WARC headers, quality scores, ...).
The clustering + PMP pipeline only reads the ``text`` field, so this script
slims each record down to::

    {"text": "<article content>", "url": "<optional>"}

Output layout (ready for
``torchtitan.experiments.cluster_data_selection.scripts.prepare_clusters``)::

    <output_dir>/
      dclm_g01_l0_s0000.jsonl
      dclm_g01_l0_s0001.jsonl
      ...

Network: the script reads ``HF_ENDPOINT`` from the environment (set to
``https://hf-mirror.com`` if you're behind a firewall that blocks
huggingface.co).  Per-file downloads are atomic: we stream into ``*.part``
then rename, so Ctrl-C / crash is safe and re-runs skip completed files.

Examples
--------

Dry run — just print the first 10 URLs the script would fetch::

    python3 -m torchtitan.experiments.cluster_data_selection.scripts.download_dclm \\
        --output_dir /tmp/dclm_raw \\
        --num_files 10 \\
        --dry_run

Full run — 50 shards (~3 M records), capped to English, minimum 64 tokens::

    HF_ENDPOINT=https://hf-mirror.com \\
    python3 -m torchtitan.experiments.cluster_data_selection.scripts.download_dclm \\
        --output_dir /path/to/dclm_raw \\
        --num_files 50 \\
        --num_workers 4 \\
        --min_tokens 64 \\
        --keep_url

Quick 100k-sample smoke set (one shard, stop early)::

    python3 -m torchtitan.experiments.cluster_data_selection.scripts.download_dclm \\
        --output_dir /tmp/dclm_mini \\
        --num_files 1 \\
        --max_samples_per_file 100000
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import io
import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Iterator


logger = logging.getLogger("download_dclm")


# ----------------------------------------------------------------------
# Upstream layout constants
# ----------------------------------------------------------------------
DATASET_REPO = "mlfoundations/dclm-baseline-1.0"
NUM_GLOBAL_SHARDS = 10  # global-shard_01_of_10 .. global-shard_10_of_10
NUM_LOCAL_SHARDS = 10   # local-shard_0_of_10 .. local-shard_9_of_10
# Each local shard contains many shard_XXXXXXXX_processed.jsonl.zst files —
# exact count varies by local shard.  We fetch the actual file list via the
# HF tree API so we don't have to hard-code it.


def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _api_url(path: str) -> str:
    return f"{_endpoint()}/api/datasets/{DATASET_REPO}/tree/main/{path}"


def _file_url(path: str) -> str:
    return f"{_endpoint()}/datasets/{DATASET_REPO}/resolve/main/{path}"


# ----------------------------------------------------------------------
# Shard discovery
# ----------------------------------------------------------------------
def _http_get_json(url: str, timeout: int = 60) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "torchtitan-dclm/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _list_local_shard(global_idx: int, local_idx: int) -> list[str]:
    """Return the list of *.jsonl.zst file paths inside one local shard."""
    prefix = f"global-shard_{global_idx:02d}_of_10/local-shard_{local_idx}_of_10"
    entries = _http_get_json(_api_url(prefix))
    return [
        e["path"]
        for e in entries
        if e.get("type") == "file" and e["path"].endswith(".jsonl.zst")
    ]


def enumerate_all_shards(
    *, global_shards: list[int] | None = None,
    local_shards: list[int] | None = None,
    seed: int = 42,
    shuffle: bool = True,
) -> list[str]:
    """Return the full list of shard file paths to choose from.

    Args:
        global_shards: Restrict to these global shard indices (1..10).
        local_shards:  Restrict to these local shard indices (0..9).
        seed:          RNG seed for shuffling.
        shuffle:       Whether to randomise file order so that a subset is
                       a roughly-uniform sample of the full corpus.
    """
    global_shards = global_shards or list(range(1, NUM_GLOBAL_SHARDS + 1))
    local_shards = local_shards or list(range(NUM_LOCAL_SHARDS))
    all_paths: list[str] = []
    for g in global_shards:
        for l in local_shards:
            paths = _list_local_shard(g, l)
            logger.info("Discovered %3d files in g%02d/l%d", len(paths), g, l)
            all_paths.extend(paths)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_paths)
    return all_paths


# ----------------------------------------------------------------------
# Streaming decompression: avoid an extra module dependency (zstandard).
# We pipe through `zstd -dc` using subprocess + stdin/stdout.
# ----------------------------------------------------------------------
def _iter_jsonl_from_zst(raw_stream: io.IOBase) -> Iterator[dict]:
    """Yield JSON-decoded records from a zstd-compressed JSONL stream.

    We prefer the ``zstandard`` Python module if available because it avoids
    spawning a subprocess per file (matters for num_workers > 1).  Fall
    back to ``zstd -dc`` otherwise.
    """
    try:
        import zstandard as zstd  # type: ignore
    except ImportError:
        zstd = None

    if zstd is not None:
        dctx = zstd.ZstdDecompressor()
        # stream_reader wraps the raw byte stream and emits decompressed bytes
        with dctx.stream_reader(raw_stream) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_reader:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.debug("JSONDecodeError: %s", e)
                    continue
        return

    # Fallback: subprocess pipe.  urllib response doesn't expose a usable
    # fd to subprocess, so we pump bytes manually.
    import subprocess
    import threading

    proc = subprocess.Popen(
        ["zstd", "-dc"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    assert proc.stdin is not None and proc.stdout is not None

    def _pump() -> None:
        try:
            while True:
                chunk = raw_stream.read(1 << 20)
                if not chunk:
                    break
                proc.stdin.write(chunk)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="ignore").rstrip("\n")
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
    t.join(timeout=5)
    proc.wait()


# ----------------------------------------------------------------------
# Per-shard download + slim transform
# ----------------------------------------------------------------------
def _output_name_for(shard_path: str) -> str:
    """``global-shard_01_of_10/local-shard_3_of_10/shard_00000017_processed.jsonl.zst``
    → ``dclm_g01_l3_s00000017.jsonl``.
    """
    parts = shard_path.split("/")
    g = parts[0].replace("global-shard_", "").split("_")[0]
    l = parts[1].replace("local-shard_", "").split("_")[0]
    basename = parts[-1]
    # shard_00000017_processed.jsonl.zst → s00000017
    stem = basename.split("_")[1]
    return f"dclm_g{g}_l{l}_s{stem}.jsonl"


def _passes_filters(
    record: dict,
    *,
    min_tokens: int,
    max_tokens: int | None,
    lang: str,
    min_lang_prob: float,
    min_quality: float | None,
) -> bool:
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        return False

    # Quick token proxy via whitespace split — avoids tokenizer dependency.
    n_tokens = len(text.split())
    if n_tokens < min_tokens:
        return False
    if max_tokens is not None and n_tokens > max_tokens:
        return False

    lang_field = record.get("language_id_whole_page_fasttext")
    if lang and isinstance(lang_field, dict):
        p = float(lang_field.get(lang, 0.0))
        if p < min_lang_prob:
            return False

    if min_quality is not None:
        q = record.get(
            "fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob"
        )
        if q is not None and float(q) < min_quality:
            return False

    return True


def _process_one_shard(
    shard_path: str,
    output_dir: str,
    *,
    keep_url: bool,
    max_samples_per_file: int,
    filters: dict,
    overwrite: bool,
) -> tuple[str, int, int]:
    """Stream one upstream shard, write slim JSONL locally.

    Returns (shard_path, records_written, records_skipped).
    """
    target_name = _output_name_for(shard_path)
    final_path = os.path.join(output_dir, target_name)
    part_path = final_path + ".part"

    if os.path.exists(final_path) and not overwrite:
        # Already done.
        return (shard_path, 0, 0)

    url = _file_url(shard_path)
    req = urllib.request.Request(url, headers={"User-Agent": "torchtitan-dclm/1.0"})
    n_written = 0
    n_skipped = 0
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(
            part_path, "w", encoding="utf-8"
        ) as out:
            for rec in _iter_jsonl_from_zst(resp):
                if not _passes_filters(rec, **filters):
                    n_skipped += 1
                    continue

                slim = {"text": rec["text"]}
                if keep_url and "url" in rec:
                    slim["url"] = rec["url"]
                out.write(json.dumps(slim, ensure_ascii=False))
                out.write("\n")
                n_written += 1

                if (
                    max_samples_per_file > 0
                    and n_written >= max_samples_per_file
                ):
                    break

        os.replace(part_path, final_path)
    except Exception as e:
        # Leave .part for visibility; callers can retry.
        logger.error("shard %s failed: %s", shard_path, e)
        raise
    finally:
        if os.path.exists(part_path) and not os.path.exists(final_path):
            # Keep partials so the user can inspect — but log clearly.
            logger.warning(
                "shard %s left a partial file at %s", shard_path, part_path
            )

    elapsed = time.time() - start
    logger.info(
        "✓ %s -> %s (%d rec, %d skipped, %.1fs)",
        shard_path,
        target_name,
        n_written,
        n_skipped,
        elapsed,
    )
    return (shard_path, n_written, n_skipped)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DCLM-baseline-1.0 shards and convert to the "
        "cluster-data-selection JSONL format ({'text': ...}).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Folder for dclm_g*_l*_s*.jsonl output files.",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=10,
        help="How many upstream shards to download (each yields ~61k records).",
    )
    parser.add_argument(
        "--global_shards",
        default="",
        help="Comma-separated subset of global shard indices (1-10). Empty=all.",
    )
    parser.add_argument(
        "--local_shards",
        default="",
        help="Comma-separated subset of local shard indices (0-9). Empty=all.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=-1,
        help="Cap records per output file (<=0 means no cap).",
    )
    parser.add_argument(
        "--keep_url",
        action="store_true",
        help="Also keep the 'url' field in each output record (else text-only).",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=32,
        help="Drop records with fewer than this many whitespace tokens.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=0,
        help="Drop records with more than this many whitespace tokens "
        "(0 disables the upper cap).",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Keep only records whose fastText language id matches "
        "(empty string disables the filter).",
    )
    parser.add_argument(
        "--min_lang_prob",
        type=float,
        default=0.5,
        help="Minimum fastText language probability for --lang match.",
    )
    parser.add_argument(
        "--min_quality",
        type=float,
        default=None,
        help="Optional lower bound on the upstream quality score "
        "(fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist locally.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the list of URLs that would be fetched and exit.",
    )
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("HF endpoint: %s", _endpoint())

    gl_list = (
        [int(x) for x in args.global_shards.split(",") if x.strip()]
        if args.global_shards
        else None
    )
    lo_list = (
        [int(x) for x in args.local_shards.split(",") if x.strip()]
        if args.local_shards
        else None
    )

    all_paths = enumerate_all_shards(
        global_shards=gl_list, local_shards=lo_list, seed=args.seed
    )
    logger.info("Total discoverable shards: %d", len(all_paths))
    if args.num_files > 0:
        all_paths = all_paths[: args.num_files]
    logger.info("Will fetch %d shards", len(all_paths))

    if args.dry_run:
        for p in all_paths:
            print(_file_url(p))
        return

    filters = dict(
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
        lang=args.lang,
        min_lang_prob=args.min_lang_prob,
        min_quality=args.min_quality,
    )

    total_written = 0
    total_skipped = 0
    t0 = time.time()
    with futures.ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futs = [
            pool.submit(
                _process_one_shard,
                path,
                args.output_dir,
                keep_url=args.keep_url,
                max_samples_per_file=args.max_samples_per_file,
                filters=filters,
                overwrite=args.overwrite,
            )
            for path in all_paths
        ]
        for fut in futures.as_completed(futs):
            try:
                _, w, s = fut.result()
                total_written += w
                total_skipped += s
            except Exception as e:  # noqa: BLE001
                logger.error("shard failed: %s", e)

    elapsed = time.time() - t0
    logger.info(
        "Done. shards=%d, records_written=%d, skipped=%d, %.1fs (%.1f rec/s)",
        len(all_paths),
        total_written,
        total_skipped,
        elapsed,
        total_written / max(elapsed, 1e-6),
    )


if __name__ == "__main__":
    main()
