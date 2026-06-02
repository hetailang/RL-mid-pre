#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/afs/L202500188/CCAC"

RUN_DIR="${RUN_DIR:-$REPO_ROOT/OSRL/examples/train/logs/OfflineBallRun-v0-cost-5/CCAC_eval_episodes5_eval_every5000_num_workers4_update_steps10000-3dff/CCAC_eval_episodes5_eval_every5000_num_workers4_update_steps10000-3dff}"
TARGET_COSTS="${TARGET_COSTS:-[1,2,5,10]}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$REPO_ROOT/OSRL/examples/eval"
source ../../../.venv/bin/activate

export PYTHONPATH="$REPO_ROOT/OSRL:$REPO_ROOT/DSRL:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export DSRL_DATASET_DIR="$REPO_ROOT/datasets/dsrl"

echo "RUN_DIR=$RUN_DIR"
echo "TARGET_COSTS=$TARGET_COSTS"
echo "EVAL_EPISODES=$EVAL_EPISODES"
echo "DEVICE=$DEVICE"

python eval_ccac.py \
  --path "$RUN_DIR" \
  --target_costs "$TARGET_COSTS" \
  --eval_episodes "$EVAL_EPISODES" \
  --device "$DEVICE"
