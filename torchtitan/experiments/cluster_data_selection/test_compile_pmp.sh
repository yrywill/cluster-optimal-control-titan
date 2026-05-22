#!/bin/bash
set -ex

# ============================================================
# Verify: torch.compile + PMP compatibility
#
# Quick test: 200 steps, PMP every 50 steps (4 triggers).
# local_batch=4, grad_accum=4, compile=true.
#
# Usage:
#   bash torchtitan/experiments/cluster_data_selection/test_compile_pmp.sh
# ============================================================

TRAIN_STEPS=${TRAIN_STEPS:-"200"}
TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-"4096"}
TRAIN_LOCAL_BATCH_SIZE=${TRAIN_LOCAL_BATCH_SIZE:-"4"}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-"4"}
PMP_UPDATE_INTERVAL=${PMP_UPDATE_INTERVAL:-"50"}
DUMP_FOLDER=${DUMP_FOLDER:-"/apdcephfs_jn5/share_304380933/rongyiyu/output/test_compile_pmp"}

BUCKET_DIR=${BUCKET_DIR:-"/apdcephfs_jn5/share_304380933/rongyiyu/data_sampled_300M"}
DEV_DIR=${DEV_DIR:-"/apdcephfs_jn4/share_304380933/rongyiyu/code/cluster_data_selection/valid_3shot"}

# ============================================================
# Environment
# ============================================================
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

# ============================================================
# Multi-node setup
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

hostname "node${NODE_RANK}"
grep -q "node${NODE_RANK}" /etc/hosts 2>/dev/null || echo "${LOCAL_IP} node${NODE_RANK}" >> /etc/hosts 2>/dev/null || true

DP_REPLICATE=${HOST_NUM}
GLOBAL_BATCH_SIZE=$((TRAIN_LOCAL_BATCH_SIZE * GRAD_ACCUM_STEPS * 8 * DP_REPLICATE))

echo "============================================================"
echo " TEST: torch.compile + PMP"
echo "============================================================"
echo " Nodes: ${HOST_NUM}, GPUs: ${WORLD_SIZE}"
echo " local_batch=${TRAIN_LOCAL_BATCH_SIZE}, grad_accum=${GRAD_ACCUM_STEPS}"
echo " global_batch=${GLOBAL_BATCH_SIZE}"
echo " Steps: ${TRAIN_STEPS}, PMP interval: ${PMP_UPDATE_INTERVAL}"
echo " compile.enable=true"
echo " Dump: ${DUMP_FOLDER}"
echo "============================================================"

mkdir -p "${DUMP_FOLDER}/logs"
mkdir -p "${DUMP_FOLDER}/scripts"
cp "$0" "${DUMP_FOLDER}/scripts/$(basename $0)" 2>/dev/null || true

# ============================================================
# Conda
# ============================================================
CONDA_BASE=/apdcephfs_jn5/share_304380933/rongyiyu/miniconda
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate rongyiyu
cd /apdcephfs_jn4/share_304380933/rongyiyu/code/torchtitan

# ============================================================
# Launch
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
    --hf_assets_path=/apdcephfs_jn4/share_304380933/rongyiyu/code/llama-3.2-3B
    # Parallelism
    --parallelism.data_parallel_shard_degree=8
    --parallelism.data_parallel_replicate_degree=${DP_REPLICATE}
    # Training
    --training.local_batch_size=${TRAIN_LOCAL_BATCH_SIZE}
    --training.global_batch_size=${GLOBAL_BATCH_SIZE}
    --training.seq_len=${TRAIN_SEQ_LEN}
    --training.steps=${TRAIN_STEPS}
    # Optimizer & LR
    --optimizer.lr=1e-4
    --lr_scheduler.warmup_steps=50
    --lr_scheduler.decay_ratio=0.8
    --lr_scheduler.decay_type=cosine
    --lr_scheduler.min_lr_factor=0.1
    # AC
    --activation_checkpoint.mode=selective
    # torch.compile
    --compile.enable=true
    # Checkpoint (disabled)
    --dump_folder=${DUMP_FOLDER}
    --checkpoint.interval=10000
    --checkpoint.keep_latest_k=0
    --checkpoint.no-last-save-in-hf
    # Dataset
    --dataloader.bucket_dir=${BUCKET_DIR}
    --dataloader.within_bucket_order=sequential
    # PMP
    --cluster.pmp.enabled
    --cluster.pmp.update_interval=${PMP_UPDATE_INTERVAL}
    --cluster.pmp.lr=0.01
    --cluster.pmp.temperature=1
    --cluster.pmp.sketch_dim=8192
    --cluster.pmp.n_samples_per_cluster=4
    --cluster.pmp.dev_batch_size=8
    # Dev data
    --cluster.dev.dev_dir=${DEV_DIR}
    --cluster.dev.max_samples=1000
)

LOG_FILE="compile_pmp_node${NODE_RANK}_$(date +%y%m%d-%H%M%S).log"

torchrun ${DISTRIBUTED_ARGS[@]} \
    -m torchtitan.experiments.cluster_data_selection.train_compat \
    ${TRAIN_ARGS[@]} "$@" 2>&1 | tee "${DUMP_FOLDER}/logs/${LOG_FILE}"

EXIT_CODE=$?

# ============================================================
# Post-run check (rank 0 only)
# ============================================================
if [[ ${NODE_RANK} -eq 0 ]]; then
    echo ""
    echo "============================================================"
    echo " Results"
    echo "============================================================"

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "[PASS] compile + PMP completed without errors"
    else
        echo "[FAIL] Training crashed (exit code ${EXIT_CODE})"
    fi

    # Weight history
    HIST="${DUMP_FOLDER}/cluster_weight_history.jsonl"
    if [[ -f "${HIST}" ]]; then
        echo ""
        echo "Weight update analysis:"
        python3 << 'PYEOF'
import json, statistics, os

hist_path = os.environ.get("HIST", "/apdcephfs_jn5/share_304380933/rongyiyu/output/test_compile_pmp/cluster_weight_history.jsonl")
records = []
with open(hist_path) as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

pmp = [r for r in records if r['event'] == 'pmp_update']
print(f"  PMP updates: {len(pmp)} (expected 4)")
print(f"  Clusters: {len(records[0]['weights'])}")

if pmp:
    init_w = records[0]['weights']
    prev = init_w
    for r in pmp:
        w = r['weights']
        delta_l2 = sum((a-b)**2 for a,b in zip(w, prev)) ** 0.5
        gg_norm = sum(x**2 for x in r['grad_gamma']) ** 0.5
        dead = sum(r['dead'])
        print(f"    step {r['step']:>4d}: Δw_L2={delta_l2:.6f}, "
              f"w=[{min(w):.3e},{max(w):.3e}], gg_norm={gg_norm:.4f}, dead={dead}")
        prev = w

    # Estimate for batch=1024 scenario
    # PMP grad magnitude is independent of training batch (fixed dev_batch=8, n_samples=4)
    # But model evolves 4x faster with batch=1024 vs 256
    # => at same interval, model is at a later training point => potentially different grad signal
    total_drift = sum((a-b)**2 for a,b in zip(pmp[-1]['weights'], init_w)) ** 0.5
    avg_delta = statistics.mean(
        sum((a-b)**2 for a,b in zip(pmp[i]['weights'],
            (records[0]['weights'] if i==0 else pmp[i-1]['weights'])))**0.5
        for i in range(len(pmp))
    )
    print(f"\n  Total drift after {len(pmp)} PMP steps: {total_drift:.6f}")
    print(f"  Avg Δw per PMP step: {avg_delta:.6f}")
    print(f"\n  --- Estimate for batch=1024 (interval=1000) ---")
    print(f"  PMP grad is normalized by dev_batch & n_samples (FIXED),")
    print(f"  so per-step Δw ≈ same magnitude regardless of training batch.")
    print(f"  Over 38000 steps: ~38 PMP triggers.")
    print(f"  Estimated total drift ≈ {avg_delta * 38:.4f}")
    print(f"  With temperature=1: weight ratio ≈ exp(drift/temp)")
    final_ratio = max(pmp[-1]['weights']) / max(min(pmp[-1]['weights']), 1e-10)
    print(f"  Current ratio after {len(pmp)} steps: {final_ratio:.1f}x")
    proj_ratio = final_ratio ** (38.0 / len(pmp))
    print(f"  Projected ratio after 38 PMP steps: ~{proj_ratio:.1f}x")
    if proj_ratio > 20:
        print(f"\n  [WARNING] Projected ratio > 20x. For batch=1024 + interval=1000:")
        print(f"  Suggest: increase temperature to 2.0, or reduce pmp_lr to 0.005")
    elif proj_ratio > 10:
        print(f"\n  [CAUTION] Projected ratio > 10x. Monitor weight distribution.")
    else:
        print(f"\n  [OK] Weight update speed looks reasonable.")
PYEOF
    fi

    # MFU
    echo ""
    echo "MFU (skip first 80 steps for compile warmup):"
    grep "mfu:" ${DUMP_FOLDER}/logs/${LOG_FILE} | tail -100 | \
        awk -F'mfu:' '{print $2}' | awk -F'%' '{sum+=$1; n++} END {if(n>0) printf "  Average MFU: %.2f%% (%d samples)\n", sum/n, n}'

    echo ""
fi
