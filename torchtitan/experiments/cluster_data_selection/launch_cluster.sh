#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Convenience wrapper around torchtitan's run_train.sh for the
# cluster_data_selection experiment.  Identical behaviour and flags, just
# with the right MODULE / CONFIG defaults and some sanity checks.
#
# Usage:
#   # Standard multi-GPU run:
#   NGPU=8 BUCKET_DIR=/path/to/buckets DEV_DIR=/path/to/dev ./launch_cluster.sh
#
#   # Fake backend (no GPU, config validation only):
#   NGPU=8 COMM_MODE=fake_backend BUCKET_DIR=/path/to/buckets DEV_DIR=/path/to/dev ./launch_cluster.sh
#
#   # Passing extra CLI overrides:
#   ./launch_cluster.sh --cluster.pmp.lr=0.05 --training.steps=500

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODULE=${MODULE:-cluster_data_selection}
CONFIG=${CONFIG:-llama3_debug_cluster}
BUCKET_DIR=${BUCKET_DIR:-""}
DEV_DIR=${DEV_DIR:-""}

if [ -z "${BUCKET_DIR}" ]; then
    echo "ERROR: set BUCKET_DIR to the output of scripts.prepare_clusters" >&2
    exit 1
fi
if [ -z "${DEV_DIR}" ]; then
    echo "ERROR: set DEV_DIR to a folder of dev JSON files (e.g. MMLU)" >&2
    exit 1
fi

cd "${TORCHTITAN_ROOT}"

export MODULE CONFIG
exec ./run_train.sh \
    --dataloader.bucket_dir="${BUCKET_DIR}" \
    --cluster.dev.dev_dir="${DEV_DIR}" \
    "$@"
