#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/afs/L202500188/CCAC"

TASK="${TASK:-OfflineBallRun-v0}"
SEED="${SEED:-0}"
STEPS="${STEPS:-10000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_SAMPLES="${EVAL_SAMPLES:-20000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/validation_experiments/outputs/test_b}"
VARIANTS="${VARIANTS:-original focal_hard focal_soft no_classifier}"
OOD_MARGIN="${OOD_MARGIN:-5.0}"

cd "$REPO_ROOT"
source .venv/bin/activate

export PYTHONPATH="$REPO_ROOT/OSRL:$REPO_ROOT/DSRL:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export DSRL_DATASET_DIR="$REPO_ROOT/datasets/dsrl"
export OUTPUT_DIR

echo "TASK=$TASK"
echo "SEED=$SEED"
echo "STEPS=$STEPS"
echo "EVAL_EVERY=$EVAL_EVERY"
echo "EVAL_SAMPLES=$EVAL_SAMPLES"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "DEVICE=$DEVICE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "VARIANTS=$VARIANTS"
echo "OOD_MARGIN=$OOD_MARGIN"

for variant in $VARIANTS; do
  case "$variant" in
    original)
      classifier_loss="bce"
      ood_mode="hard"
      ;;
    focal_hard)
      classifier_loss="focal"
      ood_mode="hard"
      ;;
    focal_soft)
      classifier_loss="focal"
      ood_mode="soft"
      ;;
    no_classifier)
      classifier_loss="bce"
      ood_mode="none"
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 1
      ;;
  esac

  run_name="${variant}_${TASK}_seed${SEED}_steps${STEPS}"
  echo "Running $variant -> $run_name"
  python validation_experiments/scripts/train_cost_critic_separation.py \
    --task "$TASK" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --eval-every "$EVAL_EVERY" \
    --eval-samples "$EVAL_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --classifier-loss "$classifier_loss" \
    --ood-mode "$ood_mode" \
    --ood-margin "$OOD_MARGIN" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$run_name"
done

python validation_experiments/scripts/summarize_test_b.py --output-dir "$OUTPUT_DIR"
