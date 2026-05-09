#!/bin/bash
# =============================================================================
# 批量将 DCP checkpoint 转换为 HF safetensors 格式
#
# 用法:
#   bash scripts/convert_all_dcp_to_hf.sh <checkpoint_root_dir> [options]
#
# 示例:
#   # 转换某次训练的所有 checkpoint
#   bash scripts/convert_all_dcp_to_hf.sh /apdcephfs_jn5/share_304380933/rongyiyu/output/train_05071529/checkpoint
#
#   # 指定模型和精度
#   bash scripts/convert_all_dcp_to_hf.sh /path/to/checkpoint --model_flavor 3B --export_dtype bfloat16
#
#   # 跳过已有 HF 转换的 checkpoint
#   bash scripts/convert_all_dcp_to_hf.sh /path/to/checkpoint --skip_existing
#
# 转换后的 HF 文件保存在每个 global_step*/hf/ 目录下，同时复制 tokenizer 文件。
# =============================================================================

set -euo pipefail

# ============== 默认参数 ==============
MODEL_NAME="llama3"
MODEL_FLAVOR="3B"
EXPORT_DTYPE="bfloat16"
HF_ASSETS_PATH=""  # 留空则自动探测
TOKENIZER_PATH="./tests/assets/tokenizer"
SKIP_EXISTING=0
PARALLEL_JOBS=1

# ============== 解析参数 ==============
CHECKPOINT_ROOT="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)
            MODEL_NAME="$2"; shift 2 ;;
        --model_flavor)
            MODEL_FLAVOR="$2"; shift 2 ;;
        --export_dtype)
            EXPORT_DTYPE="$2"; shift 2 ;;
        --hf_assets_path)
            HF_ASSETS_PATH="$2"; shift 2 ;;
        --tokenizer_path)
            TOKENIZER_PATH="$2"; shift 2 ;;
        --skip_existing)
            SKIP_EXISTING=1; shift ;;
        --parallel|-j)
            PARALLEL_JOBS="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT_ROOT" ]]; then
    echo "Usage: $0 <checkpoint_root_dir> [options]"
    echo ""
    echo "Options:"
    echo "  --model_name NAME        Model module name (default: llama3)"
    echo "  --model_flavor FLAVOR    Model size/flavor (default: 3B)"
    echo "  --export_dtype DTYPE     Export precision: float16/bfloat16/float32 (default: bfloat16)"
    echo "  --hf_assets_path PATH    HF assets with index.json (auto-detect if empty)"
    echo "  --tokenizer_path PATH    Tokenizer files to copy (default: ./tests/assets/tokenizer)"
    echo "  --skip_existing          Skip checkpoints that already have hf/ directory"
    echo "  --parallel|-j N          Number of parallel conversions (default: 1)"
    exit 1
fi

# ============== 定位脚本目录 ==============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONVERT_SCRIPT="$SCRIPT_DIR/checkpoint_conversion/convert_to_hf.py"

if [[ ! -f "$CONVERT_SCRIPT" ]]; then
    echo "Error: convert_to_hf.py not found at $CONVERT_SCRIPT"
    exit 1
fi

# ============== 查找所有 DCP checkpoint ==============
echo "============================================================"
echo " DCP → HF Batch Converter"
echo "============================================================"
echo " Checkpoint root: $CHECKPOINT_ROOT"
echo " Model: ${MODEL_NAME}/${MODEL_FLAVOR}"
echo " Export dtype: $EXPORT_DTYPE"
echo " Skip existing: $SKIP_EXISTING"
echo "============================================================"

# 找到所有 global_step* 目录（含 .distcp 文件的才是 DCP）
STEP_DIRS=()
for dir in "$CHECKPOINT_ROOT"/global_step*; do
    if [[ -d "$dir" ]] && ls "$dir"/*.distcp &>/dev/null; then
        STEP_DIRS+=("$dir")
    fi
done

if [[ ${#STEP_DIRS[@]} -eq 0 ]]; then
    echo "No DCP checkpoints (global_step* with .distcp files) found in $CHECKPOINT_ROOT"
    exit 0
fi

echo "Found ${#STEP_DIRS[@]} DCP checkpoint(s):"
for dir in "${STEP_DIRS[@]}"; do
    step_name=$(basename "$dir")
    has_hf="✗"
    if [[ -f "$dir/hf/model.safetensors.index.json" ]]; then
        has_hf="✓"
    fi
    echo "  $step_name  [HF: $has_hf]"
done
echo ""

# ============== 转换函数 ==============
convert_one() {
    local dcp_dir="$1"
    local step_name
    step_name=$(basename "$dcp_dir")
    local hf_dir="$dcp_dir/hf"

    # 跳过已有的
    if [[ $SKIP_EXISTING -eq 1 ]] && [[ -f "$hf_dir/model.safetensors.index.json" ]]; then
        echo "[$step_name] Already has HF format, skipping."
        return 0
    fi

    echo "[$step_name] Converting DCP → HF (${EXPORT_DTYPE})..."

    # 构建 hf_assets_path 参数
    local assets_arg=""
    if [[ -n "$HF_ASSETS_PATH" ]]; then
        assets_arg="--hf_assets_path $HF_ASSETS_PATH"
    fi

    # 执行转换
    cd "$REPO_ROOT"
    python "$CONVERT_SCRIPT" \
        "$dcp_dir" \
        "$hf_dir" \
        --model_name "$MODEL_NAME" \
        --model_flavor "$MODEL_FLAVOR" \
        --export_dtype "$EXPORT_DTYPE" \
        $assets_arg

    # 复制 tokenizer 文件
    if [[ -d "$TOKENIZER_PATH" ]]; then
        cp -n "$TOKENIZER_PATH"/tokenizer*.json "$hf_dir/" 2>/dev/null || true
        echo "[$step_name] Copied tokenizer files."
    fi

    # 清理 sharded 中间文件（可选，节省空间）
    if [[ -d "$hf_dir/sharded" ]]; then
        rm -rf "$hf_dir/sharded"
        echo "[$step_name] Cleaned up sharded intermediate files."
    fi

    echo "[$step_name] Done! Output: $hf_dir"
    echo ""
}

# ============== 执行转换 ==============
SUCCESS=0
FAILED=0

for dir in "${STEP_DIRS[@]}"; do
    if convert_one "$dir"; then
        ((SUCCESS++))
    else
        ((FAILED++))
        echo "[$(basename "$dir")] FAILED!"
    fi
done

echo "============================================================"
echo " Conversion complete!"
echo " Success: $SUCCESS / ${#STEP_DIRS[@]}"
if [[ $FAILED -gt 0 ]]; then
    echo " Failed: $FAILED"
fi
echo "============================================================"
