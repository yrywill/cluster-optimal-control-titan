#!/bin/bash
set -ex

# ============================================================
# TorchTitan Llama3-3B Training (WITH PMP, cluster-based optimal data selection)
# Multi-node multi-GPU, Taiji platform compatible
#
# Usage:
#   bash torchtitan/experiments/cluster_data_selection/start_train_pmp.sh
#
# Env override:
#   TRAIN_LR=1e-4 TRAIN_STEPS=2000 bash start_train_pmp.sh
# ============================================================

# ============================================================
# Training hyperparameters
# ============================================================
TRAIN_LR=${TRAIN_LR:-"1e-4"}
TRAIN_STEPS=${TRAIN_STEPS:-"38000"}
TRAIN_LOCAL_BATCH_SIZE=${TRAIN_LOCAL_BATCH_SIZE:-"8"}
TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-"4096"}
TRAIN_WARMUP_STEPS=${TRAIN_WARMUP_STEPS:-"2000"}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-"2"}
CKPT_INTERVAL=${CKPT_INTERVAL:-"1000"}
DUMP_FOLDER=${DUMP_FOLDER:-"/apdcephfs_jn5/share_304380933/rongyiyu/output/train_pmp"}

# --- Dataset path ---
BUCKET_DIR=${BUCKET_DIR:-"/apdcephfs_jn5/share_304380933/rongyiyu/data_sampled_300M"}

# --- Dev data for PMP ---
DEV_DIR=${DEV_DIR:-"/apdcephfs_jn4/share_304380933/rongyiyu/code/cluster_data_selection/valid_3shot"}

# --- PMP hyperparameters ---
PMP_UPDATE_INTERVAL=${PMP_UPDATE_INTERVAL:-"1000"}
PMP_LR=${PMP_LR:-"0.01"}
PMP_TEMPERATURE=${PMP_TEMPERATURE:-"1"}
PMP_SKETCH_DIM=${PMP_SKETCH_DIM:-"8192"}
PMP_N_SAMPLES=${PMP_N_SAMPLES:-"4"}
PMP_DEV_BATCH_SIZE=${PMP_DEV_BATCH_SIZE:-"8"}
PMP_MAX_DEV_SAMPLES=${PMP_MAX_DEV_SAMPLES:-"1000"}

# ============================================================
# Below: generally no need to modify
# ============================================================

# NCCL / IB communication
export CUDA_DEVICE_MAX_CONNECTIONS="4"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_SL="3"
export NCCL_CHECK_DISABLE="1"
export NCCL_P2P_DISABLE="0"
export NCCL_IB_DISABLE="0"
export NCCL_LL_THRESHOLD="16384"
export NCCL_IB_CUDA_SUPPORT="1"
export NCCL_TOPO_AFFINITY="0"
export NCCL_COLLNET_ENABLE="0"
export SHARP_COLL_ENABLE_SAT="0"
export NCCL_NET_GDR_LEVEL="2"
export NCCL_IB_QPS_PER_CONNECTION="4"
export NCCL_PXN_DISABLE="0"
export NCCL_NVLS_ENABLE="0"

# Determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NCCL_ALGO=^NVLS
export NCCL_TOPO_AFFINITY=6
export NCCL_CUMEM_HOST_ENABLE=0

# Network interface
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

# ============================================================
# Multi-node setup (Taiji platform compatible)
# ============================================================
HOST_GPU_NUM=8
HOST_NUM=${HOST_NUM:-1}

HOSTFILE=${HOSTFILE:-"/etc/taiji/hostfile"}
_BOND1_IP=$(ifconfig bond1 2>/dev/null | grep -oP 'inet \K[0-9.]+' || hostname -I | awk '{print $1}')
LOCAL_IP=${LOCAL_IP:-"${_BOND1_IP}"}

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
export MASTER_ADDR=${MASTER_ADDR:-${CHIEF_IP:-"${LOCAL_IP}"}}
export MASTER_PORT=${MASTER_PORT:-29600}

# Fix hostname resolution
hostname "node${NODE_RANK}"
grep -q "node${NODE_RANK}" /etc/hosts 2>/dev/null || echo "${LOCAL_IP} node${NODE_RANK}" >> /etc/hosts 2>/dev/null || true

# dp_replicate = num_nodes (each node has 8 GPUs for FSDP shard)
DP_REPLICATE=${HOST_NUM}

# global_batch_size = local_batch_size * grad_accum * dp_shard * dp_replicate
GLOBAL_BATCH_SIZE=$((TRAIN_LOCAL_BATCH_SIZE * GRAD_ACCUM_STEPS * 8 * DP_REPLICATE))

echo "============================================================"
echo " Llama3-3B Training (WITH PMP, cluster-based optimal selection)"
echo "============================================================"
echo " Nodes: ${HOST_NUM}, GPUs: ${WORLD_SIZE}"
echo " dp_shard=8, dp_replicate=${DP_REPLICATE}"
echo " Local batch: ${TRAIN_LOCAL_BATCH_SIZE}, Grad accum: ${GRAD_ACCUM_STEPS}"
echo " Global batch = ${TRAIN_LOCAL_BATCH_SIZE} * ${GRAD_ACCUM_STEPS} * 8 * ${DP_REPLICATE} = ${GLOBAL_BATCH_SIZE}"
echo " LR: ${TRAIN_LR}, Steps: ${TRAIN_STEPS}, Warmup: ${TRAIN_WARMUP_STEPS}"
echo " Data: ${BUCKET_DIR}"
echo " Dev: ${DEV_DIR}"
echo " PMP: interval=${PMP_UPDATE_INTERVAL}, lr=${PMP_LR}, temp=${PMP_TEMPERATURE}"
echo " Save: ${DUMP_FOLDER} (DCP every ${CKPT_INTERVAL} steps)"
echo " Node rank: ${NODE_RANK}, Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================================"

# Output directory
mkdir -p "${DUMP_FOLDER}/logs"
mkdir -p "${DUMP_FOLDER}/scripts"
cp "$0" "${DUMP_FOLDER}/scripts/$(basename $0)" 2>/dev/null || true

# ============================================================
# Conda environment
# ============================================================
CONDA_BASE=/apdcephfs_jn5/share_304380933/rongyiyu/miniconda
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate rongyiyu

# Enter project root
cd /apdcephfs_jn4/share_304380933/rongyiyu/code/torchtitan

# ============================================================
# Launch training
# ============================================================
DISTRIBUTED_ARGS=(
    --nproc_per_node $HOST_GPU_NUM
    --nnodes $HOST_NUM
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

TRAIN_ARGS=(
    --module cluster_data_selection
    --config llama3_3b_cluster_16gpu
    # Tokenizer
    --hf_assets_path=/apdcephfs_jn4/share_304380933/rongyiyu/code/llama-3.2-3B
    # Parallelism
    --parallelism.data_parallel_shard_degree=8
    --parallelism.data_parallel_replicate_degree=${DP_REPLICATE}
    # Training
    --training.local_batch_size=${TRAIN_LOCAL_BATCH_SIZE}
    --training.global_batch_size=${GLOBAL_BATCH_SIZE}
    --training.seq_len=${TRAIN_SEQ_LEN}
    --training.steps=${TRAIN_STEPS}
    # Optimizer & LR scheduler
    --optimizer.lr=${TRAIN_LR}
    --lr_scheduler.warmup_steps=${TRAIN_WARMUP_STEPS}
    --lr_scheduler.decay_ratio=0.8
    --lr_scheduler.decay_type=cosine
    --lr_scheduler.min_lr_factor=0.1
    # Activation checkpoint
    --activation_checkpoint.mode=selective
    # Checkpoint (DCP, save full state for resume)
    --dump_folder=${DUMP_FOLDER}
    --checkpoint.interval=${CKPT_INTERVAL}
    --checkpoint.keep_latest_k=0
    --checkpoint.no-last-save-in-hf
    # Dataset - random within-bucket sampling
    --dataloader.bucket_dir=${BUCKET_DIR}
    --dataloader.within_bucket_order=random
    # PMP enabled
    --cluster.pmp.enabled
    --cluster.pmp.update_interval=${PMP_UPDATE_INTERVAL}
    --cluster.pmp.lr=${PMP_LR}
    --cluster.pmp.temperature=${PMP_TEMPERATURE}
    --cluster.pmp.sketch_dim=${PMP_SKETCH_DIM}
    --cluster.pmp.n_samples_per_cluster=${PMP_N_SAMPLES}
    --cluster.pmp.dev_batch_size=${PMP_DEV_BATCH_SIZE}
    # Dev data
    --cluster.dev.dev_dir=${DEV_DIR}
    --cluster.dev.max_samples=${PMP_MAX_DEV_SAMPLES}
)

if [[ ${BACKGROUND_MODE} -eq 1 ]]; then
    LOG_FILE="node-${NODE_RANK}-of-${HOST_NUM}_$(date +%y%m%d-%H-%M-%S).log"
    nohup torchrun ${DISTRIBUTED_ARGS[@]} -m torchtitan.experiments.cluster_data_selection.train_compat ${TRAIN_ARGS[@]} "$@" > ${DUMP_FOLDER}/logs/${LOG_FILE} 2>&1 &
else
    torchrun ${DISTRIBUTED_ARGS[@]} -m torchtitan.experiments.cluster_data_selection.train_compat ${TRAIN_ARGS[@]} "$@"
fi
