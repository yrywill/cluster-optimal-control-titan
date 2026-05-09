#!/bin/bash
set -euo pipefail

# ============================================================
# DCP Checkpoint → HF 格式转换
# 独立脚本，训练结束后在任意机器上执行（不需要 GPU）
#
# 用法:
#   bash convert_checkpoint.sh <checkpoint目录>
#
# 示例:
#   # 转换某次训练的所有 checkpoint
#   bash convert_checkpoint.sh /apdcephfs_jn5/share_304380933/rongyiyu/output/train_nopmp_05081200/checkpoint
#
#   # 只转换最新的，跳过已有的
#   bash convert_checkpoint.sh /path/to/checkpoint
#
# 产出 (每个 global_step*/hf/ 下):
#   ├── config.json
#   ├── generation_config.json
#   ├── model-00001-of-00002.safetensors
#   ├── model-00002-of-00002.safetensors
#   ├── model.safetensors.index.json
#   ├── special_tokens_map.json
#   ├── tokenizer.json
#   └── tokenizer_config.json
#
# 转换后直接加载:
#   model = AutoModelForCausalLM.from_pretrained("<path>/hf")
# ============================================================

CHECKPOINT_DIR="${1:-}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
    echo "Usage: $0 <checkpoint_directory>"
    echo ""
    echo "Example:"
    echo "  $0 /apdcephfs_jn5/share_304380933/rongyiyu/output/train_nopmp_05081200/checkpoint"
    exit 1
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Error: Directory not found: ${CHECKPOINT_DIR}"
    exit 1
fi

# ============================================================
# 配置（根据你的模型修改）
# ============================================================
MODEL_FLAVOR=${MODEL_FLAVOR:-"3B"}
EXPORT_DTYPE=${EXPORT_DTYPE:-"bfloat16"}
HF_ASSETS_PATH=${HF_ASSETS_PATH:-"/apdcephfs_jn4/share_304380933/rongyiyu/code/llama-3.2-3B"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/apdcephfs_jn4/share_304380933/rongyiyu/code/llama-3.2-3B"}

# ============================================================
# 环境
# ============================================================
CONDA_BASE=/apdcephfs_jn5/share_304380933/rongyiyu/miniconda
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate rongyiyu

REPO_ROOT=/apdcephfs_jn4/share_304380933/rongyiyu/code/torchtitan
cd "${REPO_ROOT}"

# ============================================================
# 执行转换
# ============================================================
echo "============================================================"
echo " DCP -> HF Checkpoint Conversion"
echo "============================================================"
echo " Input:    ${CHECKPOINT_DIR}"
echo " Model:    Llama3-${MODEL_FLAVOR}"
echo " Dtype:    ${EXPORT_DTYPE}"
echo " HF ref:   ${HF_ASSETS_PATH}"
echo " Tokenizer: ${TOKENIZER_PATH}"
echo "============================================================"
echo ""

python3 scripts/convert_all_dcp_to_hf.py \
    "${CHECKPOINT_DIR}" \
    --model_flavor "${MODEL_FLAVOR}" \
    --export_dtype "${EXPORT_DTYPE}" \
    --hf_assets_path "${HF_ASSETS_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --skip_existing \
    --validate

echo ""
echo "============================================================"
echo " Done! HF checkpoints:"
for d in "${CHECKPOINT_DIR}"/global_step*/hf; do
    if [[ -f "$d/config.json" ]]; then
        echo "   $d"
    fi
done
echo ""
echo " Load: AutoModelForCausalLM.from_pretrained('<path>/hf')"
echo "============================================================"
