#!/usr/bin/env python3
"""批量将 DCP checkpoint 转换为完整的 HF 格式（含 config.json + tokenizer + safetensors）。

用法:
    python scripts/convert_all_dcp_to_hf.py <checkpoint_root> [options]

示例:
    # 转换某次训练产出的所有 DCP checkpoint
    python scripts/convert_all_dcp_to_hf.py \
        /apdcephfs_jn5/share_304380933/rongyiyu/output/train_XXXXX/checkpoint

    # 指定模型和精度
    python scripts/convert_all_dcp_to_hf.py /path/to/checkpoint \
        --model_flavor 3B --export_dtype bfloat16

    # 跳过已经转换过的
    python scripts/convert_all_dcp_to_hf.py /path/to/checkpoint --skip_existing

    # 转换完自动验证
    python scripts/convert_all_dcp_to_hf.py /path/to/checkpoint --validate

转换后目录结构:
    global_step200/
    ├── __0_0.distcp, ...      (DCP 原始文件)
    └── hf/
        ├── config.json             ← 模型配置 (HF AutoModel 必需)
        ├── model-00001-of-00001.safetensors  ← 权重
        ├── model.safetensors.index.json
        ├── tokenizer.json          ← tokenizer
        └── tokenizer_config.json
"""

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

try:
    from torch.distributed.checkpoint import HuggingFaceStorageWriter
except ImportError:
    from torch.distributed.checkpoint import _HuggingFaceStorageWriter as HuggingFaceStorageWriter

# Add repo root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP


# ============================================================
# Llama3 模型配置 → HF config.json 的映射
# ============================================================
# HF config.json 内容，和官方 Llama-3.2 发布保持一致
# 参考: /apdcephfs_jn4/share_304380933/rongyiyu/code/llama-3.2-3B/config.json
LLAMA3_HF_CONFIGS = {
    "1B": {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "tie_word_embeddings": True,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "use_cache": True,
        "initializer_range": 0.02,
        "pretraining_tp": 1,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
    },
    "3B": {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 28,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "tie_word_embeddings": True,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "use_cache": True,
        "initializer_range": 0.02,
        "pretraining_tp": 1,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
    },
    "8B": {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "use_cache": True,
        "initializer_range": 0.02,
        "pretraining_tp": 1,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
    },
    "70B": {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "use_cache": True,
        "initializer_range": 0.02,
        "pretraining_tp": 1,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
    },
}

# generation_config.json 和 special_tokens_map.json 的标准内容
GENERATION_CONFIG = {
    "_from_model_config": True,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.9,
}

SPECIAL_TOKENS_MAP = {
    "bos_token": {
        "content": "<|begin_of_text|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "eos_token": {
        "content": "<|end_of_text|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
}


def generate_hf_config(model_flavor: str, export_dtype: str) -> dict:
    """Generate HF-compatible config.json content."""
    if model_flavor not in LLAMA3_HF_CONFIGS:
        raise ValueError(
            f"Unknown model_flavor '{model_flavor}'. "
            f"Available: {list(LLAMA3_HF_CONFIGS.keys())}"
        )
    config = LLAMA3_HF_CONFIGS[model_flavor].copy()
    config["torch_dtype"] = export_dtype
    return config


@torch.inference_mode()
def convert_single_checkpoint(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    model_flavor: str,
    hf_assets_path: str | None,
    export_dtype: str,
    tokenizer_path: Path | None,
) -> None:
    """Convert one DCP checkpoint to HF format."""
    # Load model architecture
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model

    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)
    assert sd_adapter is not None, (
        "Cannot convert: sd_adapter is None. "
        "Make sure the model has a state_dict_adapter defined."
    )

    # Load DCP state dict
    state_dict = model._get_state_dict()
    dcp.load(state_dict, checkpoint_id=str(input_dir))

    # Convert torchtitan → HF key naming
    hf_state_dict = sd_adapter.to_hf(state_dict)

    # Apply export dtype
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    # Save as HF safetensors
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = HuggingFaceStorageWriter(
        path=str(output_dir),
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )
    dcp.save(hf_state_dict, storage_writer=storage_writer)

    # Generate config.json (和官方 Llama-3.2 发布格式一致)
    hf_config = generate_hf_config(model_flavor, export_dtype)
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # Generate generation_config.json
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(GENERATION_CONFIG, f, indent=2)

    # Generate special_tokens_map.json
    with open(output_dir / "special_tokens_map.json", "w") as f:
        json.dump(SPECIAL_TOKENS_MAP, f, indent=2)

    # Copy tokenizer files
    if tokenizer_path and tokenizer_path.is_dir():
        for tok_file in tokenizer_path.glob("tokenizer*.json"):
            dest = output_dir / tok_file.name
            if not dest.exists():
                shutil.copy2(tok_file, dest)
        # Also copy special_tokens_map.json if exists in source
        stm = tokenizer_path / "special_tokens_map.json"
        if stm.exists():
            shutil.copy2(stm, output_dir / "special_tokens_map.json")

    # Clean up sharded intermediate files
    sharded_dir = output_dir / "sharded"
    if sharded_dir.exists():
        shutil.rmtree(sharded_dir)


def validate_checkpoint(hf_dir: Path, model_flavor: str) -> bool:
    """Validate that a converted HF checkpoint is loadable."""
    try:
        from safetensors import safe_open

        # Check required files exist
        required_files = ["config.json", "model.safetensors.index.json"]
        for f in required_files:
            if not (hf_dir / f).exists():
                print(f"    MISSING: {f}")
                return False

        # Check config.json
        with open(hf_dir / "config.json") as f:
            config = json.load(f)
        if "model_type" not in config:
            print("    config.json missing model_type")
            return False

        # Check weight file
        safetensor_files = list(hf_dir.glob("model-*.safetensors"))
        if not safetensor_files:
            print("    No .safetensors files found")
            return False

        # Count keys
        total_keys = 0
        for sf in safetensor_files:
            with safe_open(str(sf), framework="pt") as f:
                total_keys += len(f.keys())

        expected = LLAMA3_HF_CONFIGS[model_flavor]
        n_layers = expected["num_hidden_layers"]
        # 9 keys per layer + embed_tokens + norm + (lm_head if not tied)
        expected_keys = n_layers * 9 + 2
        if not expected.get("tie_word_embeddings", False):
            expected_keys += 1

        if total_keys != expected_keys:
            print(f"    Key count mismatch: got {total_keys}, expected {expected_keys}")
            return False

        return True
    except Exception as e:
        print(f"    Validation error: {e}")
        return False


def find_dcp_checkpoints(root: Path) -> list[Path]:
    """Find all directories containing .distcp files."""
    results = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("global_step"):
            if any(d.glob("*.distcp")):
                results.append(d)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert DCP checkpoints to complete HF format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "checkpoint_root",
        type=Path,
        help="Root directory containing global_step* subdirectories",
    )
    parser.add_argument(
        "--model_name", default="llama3", help="Model module name (default: llama3)"
    )
    parser.add_argument(
        "--model_flavor", default="3B", help="Model size/flavor (default: 3B)"
    )
    parser.add_argument(
        "--export_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Export precision (default: bfloat16)",
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        default=None,
        help="HF assets with model.safetensors.index.json (optional, for multi-shard output)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        default=REPO_ROOT / "tests" / "assets" / "tokenizer",
        help="Directory with tokenizer files to copy",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip checkpoints that already have hf/ with config.json",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each checkpoint after conversion",
    )
    args = parser.parse_args()

    if not args.checkpoint_root.exists():
        print(f"Error: {args.checkpoint_root} does not exist")
        sys.exit(1)

    # Find all DCP checkpoints
    dcp_dirs = find_dcp_checkpoints(args.checkpoint_root)

    print("=" * 60)
    print(" DCP -> HF Batch Converter")
    print("=" * 60)
    print(f" Root:         {args.checkpoint_root}")
    print(f" Model:        {args.model_name}/{args.model_flavor}")
    print(f" Export dtype:  {args.export_dtype}")
    print(f" Tokenizer:    {args.tokenizer_path}")
    print(f" Skip existing: {args.skip_existing}")
    print(f" Validate:     {args.validate}")
    print("=" * 60)

    if not dcp_dirs:
        print("No DCP checkpoints (global_step* with .distcp) found.")
        sys.exit(0)

    print(f"\nFound {len(dcp_dirs)} DCP checkpoint(s):")
    for d in dcp_dirs:
        has_hf = (d / "hf" / "config.json").exists()
        status = "[HF: done]" if has_hf else "[DCP only]"
        print(f"  {d.name}  {status}")
    print()

    # Convert each
    success = 0
    failed = 0
    skipped = 0

    for dcp_dir in dcp_dirs:
        step_name = dcp_dir.name
        hf_dir = dcp_dir / "hf"

        if args.skip_existing and (hf_dir / "config.json").exists():
            print(f"[{step_name}] Already converted, skipping.")
            skipped += 1
            continue

        print(f"[{step_name}] Converting...")
        try:
            convert_single_checkpoint(
                input_dir=dcp_dir,
                output_dir=hf_dir,
                model_name=args.model_name,
                model_flavor=args.model_flavor,
                hf_assets_path=str(args.hf_assets_path) if args.hf_assets_path else None,
                export_dtype=args.export_dtype,
                tokenizer_path=args.tokenizer_path,
            )

            if args.validate:
                print(f"[{step_name}] Validating...")
                if validate_checkpoint(hf_dir, args.model_flavor):
                    print(f"[{step_name}] Validation passed!")
                else:
                    print(f"[{step_name}] Validation FAILED!")
                    failed += 1
                    continue

            print(f"[{step_name}] Done -> {hf_dir}")
            success += 1

        except Exception as e:
            print(f"[{step_name}] FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f" Results: {success} converted, {skipped} skipped, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
