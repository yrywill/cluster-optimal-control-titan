#!/usr/bin/env python3
"""Standalone DCP → HF safetensors converter for Llama3-3B.

Does NOT require matching PyTorch version or full torchtitan imports.
Works with PyTorch 2.6+ nightly (needs dcp.load with no_dist=True).

Usage:
    conda activate convert
    python -u scripts/convert_dcp_to_hf_standalone.py \
        /path/to/checkpoint \
        --tokenizer_path /path/to/llama-3.2-3B
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file


# ============================================================
# Llama3-3B HF config
# ============================================================
LLAMA3_3B_CONFIG = {
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
    "torch_dtype": "bfloat16",
    "bos_token_id": 128000,
    "eos_token_id": 128001,
}

N_HEADS = LLAMA3_3B_CONFIG["num_attention_heads"]      # 24
N_KV_HEADS = LLAMA3_3B_CONFIG["num_key_value_heads"]   # 8
DIM = LLAMA3_3B_CONFIG["hidden_size"]                  # 3072
HEAD_DIM = LLAMA3_3B_CONFIG["head_dim"]                # 128

GENERATION_CONFIG = {
    "_from_model_config": True,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.9,
}


# ============================================================
# HF permutation for RoPE compatibility
# (exact logic from Llama3StateDictAdapter.to_hf)
# ============================================================
def permute_for_hf(w: torch.Tensor, n_heads: int, dim1: int = None, dim2: int = None) -> torch.Tensor:
    """Apply HuggingFace's RoPE permutation to Q/K weight matrices."""
    if dim1 is None:
        dim1 = w.shape[0]
    if dim2 is None:
        dim2 = w.shape[1]
    return (
        w.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
        .clone()
    )


# ============================================================
# torchtitan → HF key mapping for Llama3
# ============================================================
def convert_state_dict_to_hf(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert torchtitan state_dict to HF LlamaForCausalLM format.

    Handles:
    - Key renaming
    - HF RoPE permutation on Q and K weights
    - Weight tying (remove lm_head if tied)
    - dtype conversion to bfloat16
    """
    hf_sd = {}

    for key, value in state_dict.items():
        value = value.to(torch.bfloat16)

        # Apply permutation before key rename (easier to detect)
        if ".attention.qkv_linear.wq.weight" in key:
            value = permute_for_hf(value, N_HEADS)
        elif ".attention.qkv_linear.wk.weight" in key:
            kv_dim = HEAD_DIM * N_KV_HEADS
            value = permute_for_hf(value, N_KV_HEADS, kv_dim, DIM)

        # Key mapping
        hf_key = key
        hf_key = hf_key.replace("tok_embeddings.weight", "model.embed_tokens.weight")
        hf_key = re.sub(r"^norm\.weight$", "model.norm.weight", hf_key)
        hf_key = hf_key.replace("output.weight", "lm_head.weight")
        hf_key = re.sub(r"^layers\.(\d+)\.", r"model.layers.\1.", hf_key)
        hf_key = hf_key.replace(".attention.qkv_linear.wq.", ".self_attn.q_proj.")
        hf_key = hf_key.replace(".attention.qkv_linear.wk.", ".self_attn.k_proj.")
        hf_key = hf_key.replace(".attention.qkv_linear.wv.", ".self_attn.v_proj.")
        hf_key = hf_key.replace(".attention.wo.", ".self_attn.o_proj.")
        hf_key = hf_key.replace(".feed_forward.w1.", ".mlp.gate_proj.")
        hf_key = hf_key.replace(".feed_forward.w2.", ".mlp.down_proj.")
        hf_key = hf_key.replace(".feed_forward.w3.", ".mlp.up_proj.")
        hf_key = hf_key.replace(".attention_norm.", ".input_layernorm.")
        hf_key = hf_key.replace(".ffn_norm.", ".post_attention_layernorm.")

        hf_sd[hf_key] = value

    # Llama3-3B: if tie_word_embeddings=False, keep lm_head
    if LLAMA3_3B_CONFIG["tie_word_embeddings"]:
        if "lm_head.weight" in hf_sd:
            del hf_sd["lm_head.weight"]
            print("    [tie] Removed lm_head.weight (tied with embed_tokens)", flush=True)

    return hf_sd


def build_empty_state_dict() -> dict[str, torch.Tensor]:
    """Build an empty state dict with correct shapes for Llama3-3B."""
    cfg = LLAMA3_3B_CONFIG
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    vocab = cfg["vocab_size"]
    n_layers = cfg["num_hidden_layers"]

    sd = {}
    sd["tok_embeddings.weight"] = torch.empty(vocab, hidden)
    sd["norm.weight"] = torch.empty(hidden)
    sd["output.weight"] = torch.empty(vocab, hidden)

    for i in range(n_layers):
        prefix = f"layers.{i}"
        sd[f"{prefix}.attention.qkv_linear.wq.weight"] = torch.empty(n_heads * head_dim, hidden)
        sd[f"{prefix}.attention.qkv_linear.wk.weight"] = torch.empty(n_kv_heads * head_dim, hidden)
        sd[f"{prefix}.attention.qkv_linear.wv.weight"] = torch.empty(n_kv_heads * head_dim, hidden)
        sd[f"{prefix}.attention.wo.weight"] = torch.empty(hidden, n_heads * head_dim)
        sd[f"{prefix}.feed_forward.w1.weight"] = torch.empty(inter, hidden)
        sd[f"{prefix}.feed_forward.w2.weight"] = torch.empty(hidden, inter)
        sd[f"{prefix}.feed_forward.w3.weight"] = torch.empty(inter, hidden)
        sd[f"{prefix}.attention_norm.weight"] = torch.empty(hidden)
        sd[f"{prefix}.ffn_norm.weight"] = torch.empty(hidden)

    return sd


@torch.inference_mode()
def convert_single(input_dir: Path, output_dir: Path, tokenizer_path: Path | None):
    """Convert one DCP checkpoint to HF safetensors."""
    print(f"  Loading DCP from {input_dir} ...", flush=True)
    state_dict = build_empty_state_dict()
    dcp.load(state_dict, checkpoint_id=str(input_dir), no_dist=True)

    print("  Converting keys + permuting Q/K for HF ...", flush=True)
    hf_sd = convert_state_dict_to_hf(state_dict)

    # Validate key count
    n_layers = LLAMA3_3B_CONFIG["num_hidden_layers"]
    # 9 keys/layer + embed_tokens + norm + (lm_head if not tied)
    expected = n_layers * 9 + 2
    if not LLAMA3_3B_CONFIG["tie_word_embeddings"]:
        expected += 1  # lm_head.weight
    actual = len(hf_sd)
    print(f"    Keys: {actual} (expected {expected})", flush=True)
    assert actual == expected, f"Key count mismatch: {actual} != {expected}"

    # Save in 2-shard format matching original llama-3.2-3B layout
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read original index to get exact shard assignment
    ref_index_path = Path("/apdcephfs_jn4/share_304380933/rongyiyu/code/llama-3.2-3B/model.safetensors.index.json")
    with open(ref_index_path) as f:
        ref_index = json.load(f)
    ref_weight_map = ref_index["weight_map"]

    shard1_keys = {k for k, v in ref_weight_map.items() if v == "model-00001-of-00002.safetensors"}
    shard2_keys = {k for k, v in ref_weight_map.items() if v == "model-00002-of-00002.safetensors"}

    shard1_sd = {k: v for k, v in hf_sd.items() if k in shard1_keys}
    shard2_sd = {k: v for k, v in hf_sd.items() if k in shard2_keys}

    # Sanity check
    assert len(shard1_sd) + len(shard2_sd) == len(hf_sd), (
        f"Shard split mismatch: {len(shard1_sd)}+{len(shard2_sd)} != {len(hf_sd)}"
    )

    print(f"  Saving shard 1 ({len(shard1_sd)} keys) ...", flush=True)
    save_file(shard1_sd, str(output_dir / "model-00001-of-00002.safetensors"))
    print(f"  Saving shard 2 ({len(shard2_sd)} keys) ...", flush=True)
    save_file(shard2_sd, str(output_dir / "model-00002-of-00002.safetensors"))

    # Generate model.safetensors.index.json (same structure as original)
    weight_map = {}
    for k in sorted(shard1_sd.keys()):
        weight_map[k] = "model-00001-of-00002.safetensors"
    for k in sorted(shard2_sd.keys()):
        weight_map[k] = "model-00002-of-00002.safetensors"

    total_size = sum(v.numel() * v.element_size() for v in hf_sd.values())
    index_data = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=2)

    # Remove single-file safetensors if exists from previous run
    single_file = output_dir / "model.safetensors"
    if single_file.exists():
        single_file.unlink()

    # config.json - copy from original to be exact
    shutil.copy2(
        "/apdcephfs_jn4/share_304380933/rongyiyu/code/llama-3.2-3B/config.json",
        output_dir / "config.json",
    )

    # generation_config.json
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(GENERATION_CONFIG, f, indent=2)

    # Copy tokenizer and all config files from reference
    if tokenizer_path and tokenizer_path.is_dir():
        for f_name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            src = tokenizer_path / f_name
            if src.exists():
                shutil.copy2(src, output_dir / f_name)
        print("  Copied tokenizer files", flush=True)

    print(f"  Done: {output_dir}", flush=True)


def find_dcp_checkpoints(root: Path) -> list[Path]:
    results = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("global_step"):
            if any(d.glob("*.distcp")):
                results.append(d)
    return results


def main():
    parser = argparse.ArgumentParser(description="Standalone DCP → HF converter for Llama3-3B")
    parser.add_argument("checkpoint_root", type=Path, help="Root with global_step* dirs")
    parser.add_argument("--tokenizer_path", type=Path, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint_root.exists():
        print(f"Error: {args.checkpoint_root} does not exist")
        sys.exit(1)

    dcp_dirs = find_dcp_checkpoints(args.checkpoint_root)
    print(f"Found {len(dcp_dirs)} DCP checkpoint(s)", flush=True)
    for d in dcp_dirs:
        has_hf = (d / "hf" / "config.json").exists()
        print(f"  {d.name}  {'[done]' if has_hf else '[todo]'}", flush=True)
    print(flush=True)

    success = 0
    for dcp_dir in dcp_dirs:
        hf_dir = dcp_dir / "hf"
        if args.skip_existing and (hf_dir / "config.json").exists():
            print(f"[{dcp_dir.name}] Skipping (already exists)", flush=True)
            success += 1
            continue
        print(f"[{dcp_dir.name}] Converting ...", flush=True)
        try:
            convert_single(dcp_dir, hf_dir, args.tokenizer_path)
            success += 1
        except Exception as e:
            import traceback
            print(f"[{dcp_dir.name}] FAILED: {e}", flush=True)
            traceback.print_exc()

    print(f"\nDone: {success}/{len(dcp_dirs)} converted", flush=True)


if __name__ == "__main__":
    main()
