#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/afs/L202500188/CCAC"

TASK="${TASK:-OfflineBallRun-v0}"
SEED="${SEED:-0}"
UPDATE_STEPS="${UPDATE_STEPS:-50000}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SUFFIX="${SUFFIX:-sanity50k}"

cd "$REPO_ROOT/OSRL/examples/train"
source ../../../.venv/bin/activate

export PYTHONPATH="$REPO_ROOT/OSRL:$REPO_ROOT/DSRL:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES
export DSRL_DATASET_DIR="$REPO_ROOT/datasets/dsrl"

echo "TASK=$TASK"
echo "SEED=$SEED"
echo "UPDATE_STEPS=$UPDATE_STEPS"
echo "EVAL_EVERY=$EVAL_EVERY"
echo "EVAL_EPISODES=$EVAL_EPISODES"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "DEVICE=$DEVICE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DSRL_DATASET_DIR=$DSRL_DATASET_DIR"

python train_ccac.py \
  --task "$TASK" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --update_steps "$UPDATE_STEPS" \
  --eval_every "$EVAL_EVERY" \
  --eval_episodes "$EVAL_EPISODES" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --suffix "$SUFFIX"
