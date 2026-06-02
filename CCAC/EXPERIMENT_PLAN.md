# CCAC Small Experiment Plan

This document defines a small-scale validation plan for improving the OOD
classifier and OOD usage in CCAC. The goal is not to fully reproduce the paper
benchmark first. Instead, we first validate the hypothesis with cheap,
controlled experiments, then run a small policy benchmark only after the
classifier and cost-critic evidence is positive.

## Main Question

Can a better OOD classifier objective and softer OOD usage reduce unsafe
misclassification, improve cost-critic separation, and make CCAC safer under
small cost budgets without significantly hurting reward?

## Recommended Order

1. Run one short original CCAC sanity baseline.
2. Run Test A: classifier-only comparison.
3. Run Test B: cost-critic separation comparison.
4. Run Test C: small full-policy comparison on 1-2 tasks.

The full paper benchmark should be deferred until the small experiments show a
clear signal.

## Sanity Baseline

Before adding new experiment code, run a short original CCAC training job to
confirm that the environment, dataset, logging, checkpointing, and evaluation
work in this checkout.

Recommended first task:

- `OfflineBallRun-v0`
- Optional second task: `OfflineCarRun-v0`

Suggested short run:

```bash
cd OSRL/examples/train
python train_ccac.py --task OfflineBallRun-v0 --device cuda:0 --update_steps 10000 --eval_every 5000 --eval_episodes 5
```

Expected output:

- Training finishes without environment or dataset errors.
- A checkpoint and config are written under `logs/`.
- Evaluation prints reward, cost, and length during training.

This run is only a sanity check. It is not expected to match the paper numbers.

## Test A: Classifier-Only Evaluation

### Purpose

Train only the CVAE/OOD classifier part and compare classifier losses and OOD
score usage. This removes RL noise and directly tests whether the proposed
classifier changes reduce unsafe false negatives.

### Variants

| Variant | Classifier loss | OOD usage |
| --- | --- | --- |
| Original | BCE | hard threshold |
| Ours-1 | weighted BCE | hard threshold |
| Ours-2 | focal loss | hard threshold |
| Ours-3 | focal loss | soft OOD score |

### Required script

Add a classifier-only script, for example:

```text
OSRL/examples/analysis/train_ood_classifier.py
```

The script should:

- Load one DSRL offline dataset.
- Run `set_cost_thresholds(data)`.
- Construct `(s, a, kappa, y)` samples using the same basic label idea as
  `CCAC.vae_loss`.
- Train the CVAE and classifier only.
- Evaluate OOD scores on a held-out split.
- Save metrics and plots.

### Metrics

- AUROC
- AUPRC
- False negative rate
- False positive rate
- Expected calibration error
- OOD score histogram

### Main evidence to show

The improved classifier should produce fewer OOD/unsafe samples incorrectly
classified as IND. For safe RL, false negatives are more important than false
positives because missed OOD unsafe samples will not be penalized by the cost
critic regularizer.

### Expected runtime

On one GPU:

- About 10-40 minutes per task and variant group.
- No full RL training required.

## Test B: Cost-Critic Separation

### Purpose

Reproduce the idea of the paper's cost-critic separation analysis: OOD samples
should receive higher `Q_c` than IND samples, while IND `Q_c` should remain in a
reasonable range.

### Variants

| Method | Expected behavior |
| --- | --- |
| Original CCAC | OOD `Q_c` higher than IND `Q_c` |
| Soft OOD-CCAC | OOD `Q_c` is more stable and smoothly higher than IND `Q_c` |
| No classifier | OOD/IND separation is weak |

### Required script

Add a small cost-critic analysis script, for example:

```text
OSRL/examples/analysis/train_cost_critic_separation.py
```

The script should:

- Train CVAE/classifier and cost critic for a limited number of steps.
- Avoid full actor training at first.
- Generate candidate OOD `(s, a)` samples with the CVAE.
- Compare `Q_c` values for IND dataset samples and generated OOD samples.
- Save line plots or histograms of IND `Q_c` and OOD `Q_c`.

### Implementation needs

The current `cost_critic_loss_cvae` uses a hard classifier mask. For this test,
add a controlled option for:

- hard OOD mask
- soft OOD score weighting
- no classifier filtering

This can be implemented either by adding flags to CCAC or by writing a separate
analysis-only training loop.

### Metrics and plots

- Mean and standard deviation of IND `Q_c`
- Mean and standard deviation of OOD `Q_c`
- OOD-minus-IND `Q_c` gap
- `Q_c` histograms
- Training-step curves for the `Q_c` gap

### Expected runtime

On one GPU:

- About 0.5-2 hours per task, depending on step count.
- Start with 10k-50k critic update steps.

## Test C: Small Full-Policy Benchmark

### Purpose

After Test A and Test B show positive evidence, run a small full-policy check to
make sure the classifier/OOD changes do not damage policy quality.

### Tasks

Start with one task:

- `OfflineBallRun-v0`

Optional second task:

- `OfflineCarRun-v0`

Optional later task:

- `OfflineBallCircle-v0`

### Variants

At minimum:

- Original CCAC
- Best variant from Test A/B

Do not run all four classifier variants in the first full-policy pass unless
Test A/B strongly justify it.

### Metrics

- Normalized reward
- Normalized cost
- Violation rate
- Cost-budget alignment: absolute difference between actual cost and target cost
- Small-budget safety performance

### Evaluation budgets

Use several target costs in `eval_ccac.py`, for example:

```bash
cd OSRL/examples/eval
python eval_ccac.py --path <log_path> --target_costs 1 2 5 10 --eval_episodes 20 --device cuda:0
```

### Expected runtime

On one GPU:

- Short sanity policy run: 1-4 hours per variant/seed.
- Default-length training across multiple seeds will take much longer.

Start with:

- `update_steps=20000` or `50000`
- 1 seed
- 1 task

Then expand only if the result is promising.

## What Does Not Need To Be Done First

Do not start with:

- Full paper table reproduction.
- All 38 DSRL tasks.
- All baselines from the paper.
- Three or more seeds for every variant.
- Dynamic-budget experiments.
- Figure-level plotting for every paper result.

These are useful later but too expensive for the first validation cycle.

## Minimal Deliverables For A Midterm-Style Report

1. A short original CCAC sanity run on one task.
2. Test A table with AUROC, AUPRC, FNR, FPR, and ECE.
3. Test A OOD score histogram.
4. Test B IND/OOD `Q_c` separation plot.
5. One small Test C table comparing original CCAC and the best variant.

## Success Criteria

The small experiment package is considered successful if:

- The improved classifier lowers false negatives on OOD/unsafe samples.
- Soft OOD usage keeps or improves OOD-vs-IND `Q_c` separation.
- The best variant does not clearly reduce normalized reward in the small full
  policy run.
- The best variant reduces cost violation rate or improves small-budget safety.

## Risks

- Classifier-only gains may not transfer to policy performance.
- Soft OOD weighting may weaken the cost-critic penalty if scores are poorly
  calibrated.
- Very short full-policy runs can be noisy.
- Some DSRL tasks may require longer training before policy-level effects are
  visible.

Mitigation: use Test A and Test B as cheap filters before running full policy
training, and keep the first policy benchmark small.
