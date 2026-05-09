#!/bin/bash
set -ex

# ============================================================
# TorchTitan Cluster Data Selection - 128 GPU (16 nodes x 8 GPUs)
# 训练 + 训后自动转换 DCP → HF 格式
#
# 用法 (太极平台 start_cmd):
#   bash torchtitan/experiments/cluster_data_selection/start_128gpu.sh
#
# 环境变量覆盖:
#   TRAIN_LR=1e-4 TRAIN_STEPS=2000 bash start_128gpu.sh
# ============================================================

# ===================== NCCL & Network =====================
export CUDA_DEVICE_MAX_CONNECTIONS="4"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_SL="3"
export NCCL_CHECK_DISABLE="1"
export NCCL_P2P_DISABLE="0"
export NCCL_IB_DISABLE="0"
export NCCL_LL_THRESHOLD="16384"
export NCCL_IB_CUDA_SUPPORT="1"
export NCCL_COLLNET_ENABLE="0"
export SHARP_COLL_ENABLE_SAT="0"
export NCCL_NET_GDR_LEVEL="2"
export NCCL_IB_QPS_PER_CONNECTION="4"
export NCCL_PXN_DISABLE="0"
export NCCL_NVLS_ENABLE="0"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NCCL_ALGO=^NVLS
export NCCL_TOPO_AFFINITY=6
export NCCL_CUMEM_HOST_ENABLE=0

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"bond1"}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-"bond1"}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"bond1"}
export NCCL_IB_TC=${NCCL_IB_TC:-"160"}
export NCCL_IB_HCA=${NCCL_IB_HCA:-"mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6"}
export NCCL_IB_TIMEOUT="22"
export NCCL_NET_GDR_READ="1"

export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-"bond1"}
export NVSHMEM_HCA_LIST=${NVSHMEM_HCA_LIST:-"mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1"}
export NVSHMEM_IB_TRAFFIC_CLASS=${NVSHMEM_IB_TRAFFIC_CLASS:-"160"}

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=100

# ===================== Training Config =====================
MODULE_NAME="torchtitan.train"
CONFIG_NAME="llama3_3b_cluster_128gpu"
MODEL_FLAVOR="3B"
EXPORT_DTYPE="bfloat16"

# Data paths
BUCKET_DIR=${BUCKET_DIR:-"/apdcephfs_jn4/share_304380933/rongyiyu/code/torchtitan/outputs/buckets_dataset100k_cs500"}
DEV_DIR=${DEV_DIR:-"/apdcephfs_jn4/share_304380933/rongyiyu/code/cluster_data_selection/valid_3shot"}

# Training hyperparameters
TRAIN_LR=${TRAIN_LR:-"3e-5"}
TRAIN_STEPS=${TRAIN_STEPS:-"1000"}
TRAIN_WARMUP_STEPS=${TRAIN_WARMUP_STEPS:-"200"}
TRAIN_LOCAL_BATCH_SIZE=${TRAIN_LOCAL_BATCH_SIZE:-"8"}
TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-"2048"}
DP_REPLICATE=${DP_REPLICATE:-"16"}
CKPT_INTERVAL=${CKPT_INTERVAL:-"200"}

# PMP hyperparameters
PMP_UPDATE_INTERVAL=${PMP_UPDATE_INTERVAL:-"100"}
PMP_LR=${PMP_LR:-"0.01"}
PMP_TEMPERATURE=${PMP_TEMPERATURE:-"1"}
PMP_N_SAMPLES=${PMP_N_SAMPLES:-"4"}
PMP_DEV_BATCH_SIZE=${PMP_DEV_BATCH_SIZE:-"8"}
PMP_SKETCH_DIM=${PMP_SKETCH_DIM:-"8192"}
PMP_DROP_PATIENCE=${PMP_DROP_PATIENCE:-"5"}
PMP_MAX_DEV_SAMPLES=${PMP_MAX_DEV_SAMPLES:-"1000"}

# ===================== Multi-node Setup =====================
HOST_GPU_NUM=8
HOST_NUM=${HOST_NUM:-16}
HOSTFILE=${HOSTFILE:-"/etc/taiji/hostfile"}
LOCAL_IP=${LOCAL_IP:-"127.0.0.1"}
BACKGROUND_MODE=${BACKGROUND_MODE:-0}

if [[ ${BACKGROUND_MODE} -eq 1 ]]; then
    IP_INDEX=$(grep -n "${LOCAL_IP} slots=" ${HOSTFILE} | awk -F: '{ print $1 }')
    if [[ -z ${IP_INDEX} || "${IP_INDEX}" -gt ${HOST_NUM} ]]; then
        exit 0
    fi
    if [[ -f ${HOSTFILE} ]]; then
        MASTER_ADDR=$(head -n1 ${HOSTFILE} | awk '{ print $1 }')
    fi
    NODE_RANK=$((IP_INDEX-1))
else
    MASTER_ADDR=${CHIEF_IP:-"${LOCAL_IP}"}
    NODE_RANK=${INDEX:-0}
fi

export WORLD_SIZE=${WORLD_SIZE:-$((HOST_NUM*HOST_GPU_NUM))}
export MASTER_ADDR=${MASTER_ADDR:-${CHIEF_IP:-"127.0.0.1"}}
export MASTER_PORT=${MASTER_PORT:-29600}

hostname "node${NODE_RANK}"
for i in $(seq 0 $((HOST_NUM-1))); do
    grep -q "node${i}" /etc/hosts 2>/dev/null || true
done

# ===================== Output Directory =====================
DUMP_FOLDER=${DUMP_FOLDER:-"/apdcephfs_jn5/share_304380933/rongyiyu/output/train_$(date +%m%d%H%M)"}
mkdir -p "${DUMP_FOLDER}/logs"
mkdir -p "${DUMP_FOLDER}/scripts"
cp "$0" "${DUMP_FOLDER}/scripts/$(basename $0)" 2>/dev/null || true

# ===================== Conda Environment =====================
CONDA_BASE=${CONDA_BASE:-"/apdcephfs_jn5/share_304380933/rongyiyu/miniconda"}
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate rongyiyu

# Enter project root
cd /apdcephfs_jn4/share_304380933/rongyiyu/code/torchtitan

# ===================== Print Config Summary =====================
echo "============================================================"
echo " Cluster Data Selection Training"
echo "============================================================"
echo " Nodes: ${HOST_NUM}, GPUs: $((HOST_NUM*HOST_GPU_NUM))"
echo " Node rank: ${NODE_RANK}, Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo " dp_shard=8, dp_replicate=${DP_REPLICATE}"
echo " Global batch = ${TRAIN_LOCAL_BATCH_SIZE} * 8 * ${DP_REPLICATE} = $((TRAIN_LOCAL_BATCH_SIZE*8*DP_REPLICATE))"
echo " LR: ${TRAIN_LR}, Steps: ${TRAIN_STEPS}, Warmup: ${TRAIN_WARMUP_STEPS}"
echo " PMP: interval=${PMP_UPDATE_INTERVAL}, lr=${PMP_LR}, temp=${PMP_TEMPERATURE}"
echo " Data: ${BUCKET_DIR}"
echo " Dev: ${DEV_DIR} (max_samples=${PMP_MAX_DEV_SAMPLES})"
echo " Save: ${DUMP_FOLDER} (every ${CKPT_INTERVAL} steps, DCP only)"
echo " Post-train: auto convert DCP -> HF (${EXPORT_DTYPE})"
echo "============================================================"

# ===================== Launch Training =====================
DISTRIBUTED_ARGS=(
    --nproc_per_node $HOST_GPU_NUM
    --nnodes $HOST_NUM
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

TRAIN_ARGS=(
    --module cluster_data_selection
    --config ${CONFIG_NAME}
    --parallelism.data_parallel_replicate_degree=${DP_REPLICATE}
    --training.local_batch_size=${TRAIN_LOCAL_BATCH_SIZE}
    --training.seq_len=${TRAIN_SEQ_LEN}
    --training.steps=${TRAIN_STEPS}
    --optimizer.lr=${TRAIN_LR}
    --lr_scheduler.warmup_steps=${TRAIN_WARMUP_STEPS}
    --dump_folder=${DUMP_FOLDER}
    --checkpoint.interval=${CKPT_INTERVAL}
    --checkpoint.no-last-save-in-hf
    --dataloader.bucket_dir=${BUCKET_DIR}
    --cluster.dev.dev_dir=${DEV_DIR}
    --cluster.dev.max_samples=${PMP_MAX_DEV_SAMPLES}
    --cluster.pmp.update_interval=${PMP_UPDATE_INTERVAL}
    --cluster.pmp.lr=${PMP_LR}
    --cluster.pmp.temperature=${PMP_TEMPERATURE}
    --cluster.pmp.n_samples_per_cluster=${PMP_N_SAMPLES}
    --cluster.pmp.dev_batch_size=${PMP_DEV_BATCH_SIZE}
    --cluster.pmp.sketch_dim=${PMP_SKETCH_DIM}
    --cluster.pmp.drop_patience=${PMP_DROP_PATIENCE}
)

torchrun ${DISTRIBUTED_ARGS[@]} -m ${MODULE_NAME} ${TRAIN_ARGS[@]} "$@"
TRAIN_EXIT_CODE=$?

# ===================== Post-Training: DCP → HF Conversion =====================
# 只在 rank 0 上执行转换，避免多节点重复操作
if [[ ${NODE_RANK} -eq 0 ]] && [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
    echo ""
    echo "============================================================"
    echo " Training complete! Converting DCP checkpoints to HF format..."
    echo "============================================================"

    CKPT_DIR="${DUMP_FOLDER}/checkpoint"
    if [[ -d "${CKPT_DIR}" ]]; then
        python3 scripts/convert_all_dcp_to_hf.py \
            "${CKPT_DIR}" \
            --model_flavor ${MODEL_FLAVOR} \
            --export_dtype ${EXPORT_DTYPE} \
            --tokenizer_path ./tests/assets/tokenizer \
            --skip_existing \
            --validate

        echo ""
        echo "============================================================"
        echo " All checkpoints converted! HF models at:"
        for d in "${CKPT_DIR}"/global_step*/hf; do
            if [[ -f "$d/config.json" ]]; then
                echo "   $d"
            fi
        done
        echo "============================================================"
    else
        echo "WARNING: Checkpoint directory not found at ${CKPT_DIR}"
    fi
elif [[ ${TRAIN_EXIT_CODE} -ne 0 ]]; then
    echo "Training failed with exit code ${TRAIN_EXIT_CODE}, skipping HF conversion."
fi
