# Experiment Commands

Run commands from `/mnt/afs/L202500188/CCAC` unless otherwise noted.

## 0. Environment And GPU Check

```bash
cd /mnt/afs/L202500188/CCAC
source .venv/bin/activate

export PYTHONPATH=$PWD/OSRL:$PWD/DSRL:$PYTHONPATH
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
export DSRL_DATASET_DIR=$PWD/datasets/dsrl

python -c "import torch, dsrl, osrl; print('python ok'); print('cuda', torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected:

- `python ok`
- `cuda True`
- at least one CUDA device

If the GPU is H100, also run a minimal CUDA kernel test:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(y.mean().item())
PY
```

If this hangs or prints a warning like `sm_90 is not compatible`, the PyTorch
build is not usable on H100. The current OSRL package pins `torch~=1.13.0`, which
does not support H100. Fix the environment before running training:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(y.mean().item())
PY
```

Do this after installing `OSRL`/`DSRL`, because reinstalling OSRL dependencies
may pull `torch~=1.13.0` back into the environment.

## 1. Sanity Baseline: Original CCAC Short Run

Purpose: verify that the original CCAC code can train, save checkpoints, and
evaluate on a small task. This is not intended to reproduce paper numbers.

If the dataset server is unstable, download the dataset first. Training will use
the local file when it exists under `DSRL_DATASET_DIR`.

```bash
cd /mnt/afs/L202500188/CCAC
mkdir -p datasets/dsrl

source .venv/bin/activate
export DSRL_DATASET_DIR=/mnt/afs/L202500188/CCAC/datasets/dsrl

python validation_experiments/scripts/download_dsrl_dataset.py --task OfflineBallRun-v0 --retries 5 --sleep-seconds 20
```

Alternative first-task datasets:

```bash
python validation_experiments/scripts/download_dsrl_dataset.py --task OfflineCarRun-v0 --retries 5 --sleep-seconds 20
python validation_experiments/scripts/download_dsrl_dataset.py --task OfflineBallCircle-v0 --retries 5 --sleep-seconds 20
```

After a download finishes, verify that h5py can open the file:

```bash
python -c "import h5py; p='datasets/dsrl/SafetyBallRun-v0-80-940.hdf5'; f=h5py.File(p, 'r'); print(p, list(f.keys())[:10]); f.close()"
```

```bash
cd /mnt/afs/L202500188/CCAC/OSRL/examples/train
source ../../../.venv/bin/activate

export PYTHONPATH=/mnt/afs/L202500188/CCAC/OSRL:/mnt/afs/L202500188/CCAC/DSRL:$PYTHONPATH
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
export DSRL_DATASET_DIR=/mnt/afs/L202500188/CCAC/datasets/dsrl

python train_ccac.py \
  --task OfflineBallRun-v0 \
  --device cuda:0 \
  --seed 0 \
  --update_steps 10000 \
  --eval_every 5000 \
  --eval_episodes 5 \
  --batch_size 512 \
  --num_workers 4
```

Note: `train_ccac.py` runs CVAE warmup for `update_steps`, then RL training for
another `update_steps`. With `update_steps=10000`, total loop count is about
20000.

## 2. Locate Latest Run Directory

Run this after training finishes:

```bash
cd /mnt/afs/L202500188/CCAC/OSRL/examples/train

RUN_DIR=$(dirname "$(find logs -name config.yaml -print | sort | tail -1)")
realpath "$RUN_DIR"
find "$RUN_DIR" -maxdepth 2 -type f | sort
```

Record the absolute `RUN_DIR` in `validation_experiments/results.md`.

## 2.5 Original CCAC 50k Baseline

After the 10k sanity run succeeds, run a longer original CCAC baseline. This is
still not a paper reproduction, but it is a better anchor for later small
full-policy comparison.

```bash
cd /mnt/afs/L202500188/CCAC
bash validation_experiments/scripts/train_original_ccac_50k.sh
```

Optional overrides:

```bash
SEED=1 CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 bash validation_experiments/scripts/train_original_ccac_50k.sh
```

The script defaults to:

- task: `OfflineBallRun-v0`
- update steps: `50000`
- eval every: `10000`
- eval episodes: `10`
- suffix: `sanity50k`

## 3. Evaluate Sanity Baseline

Replace `<ABS_RUN_DIR>` with the absolute path printed above.

For the completed `OfflineBallRun-v0` sanity run, use:

```bash
cd /mnt/afs/L202500188/CCAC
bash validation_experiments/scripts/eval_sanity_baseline.sh
```

Optional overrides:

```bash
TARGET_COSTS="[1,2,5,10]" EVAL_EPISODES=20 DEVICE=cuda:0 bash validation_experiments/scripts/eval_sanity_baseline.sh
```

```bash
cd /mnt/afs/L202500188/CCAC/OSRL/examples/eval
source ../../../.venv/bin/activate

export PYTHONPATH=/mnt/afs/L202500188/CCAC/OSRL:/mnt/afs/L202500188/CCAC/DSRL:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export DSRL_DATASET_DIR=/mnt/afs/L202500188/CCAC/datasets/dsrl

python eval_ccac.py \
  --path <ABS_RUN_DIR> \
  --target_costs 1 2 5 10 \
  --eval_episodes 10 \
  --device cuda:0
```

If `--target_costs 1 2 5 10` fails to parse, use:

```bash
python eval_ccac.py \
  --path <ABS_RUN_DIR> \
  --target_costs '[1,2,5,10]' \
  --eval_episodes 10 \
  --device cuda:0
```

## 4. Test A: Classifier-Only Evaluation

Run all four classifier variants:

```bash
cd /mnt/afs/L202500188/CCAC
bash validation_experiments/scripts/run_test_a_classifier.sh
```

Defaults:

- task: `OfflineBallRun-v0`
- seed: `0`
- steps per variant: `5000`
- eval samples: `20000`
- variants: `original ours_weighted ours_focal ours_focal_soft`
- output dir: `validation_experiments/outputs/test_a`

Useful overrides:

```bash
STEPS=10000 EVAL_EVERY=2000 EVAL_SAMPLES=50000 bash validation_experiments/scripts/run_test_a_classifier.sh
```

Run only one variant for debugging:

```bash
VARIANTS="original" STEPS=200 bash validation_experiments/scripts/run_test_a_classifier.sh
```

Each run writes:

- `metrics.json`
- `history.json`
- `ood_score_histogram.png`
- `model.pt`

If training completed but the final summary failed, summarize existing outputs:

```bash
cd /mnt/afs/L202500188/CCAC
python validation_experiments/scripts/summarize_test_a.py --output-dir validation_experiments/outputs/test_a
```

## 5. Test B: Cost-Critic Separation

Debug one variant first:

```bash
cd /mnt/afs/L202500188/CCAC
VARIANTS="original" STEPS=200 EVAL_EVERY=100 EVAL_SAMPLES=2000 bash validation_experiments/scripts/run_test_b_cost_critic.sh
```

Run the default Test B comparison:

```bash
cd /mnt/afs/L202500188/CCAC
bash validation_experiments/scripts/run_test_b_cost_critic.sh
```

Defaults:

- task: `OfflineBallRun-v0`
- seed: `0`
- steps per variant: `10000`
- eval samples: `20000`
- variants: `original focal_hard focal_soft no_classifier`
- OOD margin: `5.0`
- output dir: `validation_experiments/outputs/test_b`

The current Test B implementation is matched-budget: it compares real IND and
generated OOD samples under the same query cost budget and pushes OOD `Q_c`
toward `matched IND Q_c + OOD_MARGIN`.

Each run writes:

- `metrics.json`
- `history.json`
- `qc_gap_curve.png`
- `qc_histogram.png`
- `model.pt`

If training completed but the final summary failed, summarize existing outputs:

```bash
cd /mnt/afs/L202500188/CCAC
python validation_experiments/scripts/summarize_test_b.py --output-dir validation_experiments/outputs/test_b
```

## 6. Test C: Small Full-Policy Variant

Test C compares the 50k original CCAC baseline with the best small-experiment
variant from Test A/B. The first selected variant is `focal_soft`:

```bash
cd /mnt/afs/L202500188/CCAC
source .venv/bin/activate
export WANDB_MODE=offline
export PYTHONPATH="$PWD/OSRL:$PWD/DSRL:${PYTHONPATH:-}"
export DSRL_DATASET_DIR="$PWD/datasets/dsrl"

cd OSRL/examples/train
python train_ccac.py \
  --task OfflineBallRun-v0 \
  --device cuda:0 \
  --seed 0 \
  --update_steps 50000 \
  --eval_every 10000 \
  --eval_episodes 10 \
  --batch_size 512 \
  --num_workers 4 \
  --classifier_loss focal \
  --ood_mode soft \
  --suffix test_c_focal_soft_50k
```

Evaluate the saved checkpoint across target costs:

```bash
cd /mnt/afs/L202500188/CCAC
source .venv/bin/activate
export PYTHONPATH="$PWD/OSRL:$PWD/DSRL:${PYTHONPATH:-}"
export DSRL_DATASET_DIR="$PWD/datasets/dsrl"

cd OSRL/examples/eval
python eval_ccac.py \
  --path /mnt/afs/L202500188/CCAC/OSRL/examples/train/logs/OfflineBallRun-v0-cost-5/CCAC_classifier_lossfocal_num_workers4_ood_modesoft_update_steps50000_test_c_focal_soft_50k-bb72/CCAC_classifier_lossfocal_num_workers4_ood_modesoft_update_steps50000_test_c_focal_soft_50k-bb72 \
  --target_costs "[1,2,5,10]" \
  --eval_episodes 20 \
  --device cuda:0
```

Record the run directory and evaluation output in
`validation_experiments/results.md`.
