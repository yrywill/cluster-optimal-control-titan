# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dev-set loader used ONLY by PMP.

Completely decoupled from torchtitan's main training dataloader and from
``torchtitan.components.validate.Validator``.  The dev set is expected to be
a small JSON/JSONL folder (e.g. MMLU) that fits comfortably in CPU memory.

The batches returned here mirror the shape that PMP's sketch-loss forward
expects:

    input_ids [B, L],  labels [B, L],  loss_mask [B, L]

They are stored on CPU and moved to the GPU only while PMP runs.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger


def _load_texts(dev_dir: str, text_field: str, max_samples: int) -> list[str]:
    patterns = [
        os.path.join(dev_dir, "*.json"),
        os.path.join(dev_dir, "*.jsonl"),
        os.path.join(dev_dir, "**", "*.json"),
        os.path.join(dev_dir, "**", "*.jsonl"),
    ]
    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No .json or .jsonl files found under {dev_dir!r}"
        )

    texts: list[str] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Try JSONL first: parse each line independently, skip bad lines.
        per_line: list[str] = []
        num_bad = 0
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                num_bad += 1
                continue
            if isinstance(obj, dict) and text_field in obj:
                per_line.append(obj[text_field])
        if per_line:
            if num_bad > 0:
                logger.warning(
                    "Skipped %d malformed lines in %s", num_bad, path,
                )
            texts.extend(per_line)
            continue
        # Fall back to plain JSON.
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning("Could not parse %s: %s", path, e)
            continue
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and text_field in item:
                    texts.append(item[text_field])
        elif isinstance(data, dict):
            for key in ("data", "items", "records", "samples"):
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict) and text_field in item:
                            texts.append(item[text_field])
                    break
            else:
                if text_field in data:
                    texts.append(data[text_field])

    if max_samples > 0 and len(texts) > max_samples:
        texts = texts[:max_samples]
    return texts


class DevBatchCache:
    """Pre-tokenise the dev set once and cache batches on CPU.

    Designed for the PMP backward: we want to recompute ``∇L_dev`` many
    times without paying tokenisation cost.  The batches are kept on CPU
    and shipped to the GPU on demand via :meth:`iter_on_device`.
    """

    def __init__(
        self,
        *,
        dev_dir: str,
        tokenizer: BaseTokenizer,
        text_field: str,
        max_length: int,
        max_samples: int,
        batch_size: int,
    ) -> None:
        if not dev_dir:
            raise ValueError(
                "DevBatchCache requires a non-empty dev_dir (set "
                "cluster.dev.dev_dir in your config)."
            )
        texts = _load_texts(dev_dir, text_field, max_samples)
        if not texts:
            raise ValueError(
                f"Loaded 0 dev samples from {dev_dir!r}; check text_field={text_field!r}."
            )
        logger.info(
            "[DevBatchCache] Loaded %d dev texts from %s", len(texts), dev_dir
        )

        pad_id = tokenizer.eos_id if tokenizer.eos_id is not None else 0

        all_tokens: list[list[int]] = []
        for t in texts:
            ids = tokenizer.encode(t, add_bos=True, add_eos=True)[: max_length + 1]
            if len(ids) >= 2:
                all_tokens.append(ids)

        self._batches: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        for start in range(0, len(all_tokens), batch_size):
            chunk = all_tokens[start : start + batch_size]
            max_len = min(max(len(ids) for ids in chunk) - 1, max_length)
            B = len(chunk)
            input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
            labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
            loss_mask = torch.zeros((B, max_len), dtype=torch.float32)
            for i, ids in enumerate(chunk):
                ids = ids[: max_len + 1]
                seq_len = len(ids) - 1
                input_ids[i, :seq_len] = torch.tensor(ids[:-1], dtype=torch.long)
                labels[i, :seq_len] = torch.tensor(ids[1:], dtype=torch.long)
                loss_mask[i, :seq_len] = 1.0
            self._batches.append((input_ids, labels, loss_mask))

        logger.info(
            "[DevBatchCache] Prepared %d dev batches (batch_size=%d, max_length=%d)",
            len(self._batches),
            batch_size,
            max_length,
        )

    def __len__(self) -> int:
        return len(self._batches)

    def iter_on_device(
        self, device: torch.device
    ):
        """Yield batches already moved to ``device``."""
        for input_ids, labels, loss_mask in self._batches:
            yield (
                input_ids.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
                loss_mask.to(device, non_blocking=True),
            )
