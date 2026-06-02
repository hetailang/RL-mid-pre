# Validation Experiment Scripts

This file records the implemented and planned validation experiment scripts.

## Test A: Classifier-Only Script

Implemented path:

```text
validation_experiments/scripts/train_ood_classifier.py
```

Purpose:

- Train only the CVAE and OOD classifier.
- Compare BCE, weighted BCE, focal loss, and focal loss with soft OOD score.
- Report classifier metrics without running full RL.

Inputs:

- `--task`
- `--device`
- `--seed`
- `--steps`
- `--batch_size`
- `--classifier_loss {bce,weighted_bce,focal}`
- `--ood_usage {hard,soft}`
- `--output_dir`

Launcher:

```text
validation_experiments/scripts/run_test_a_classifier.sh
```

Outputs:

- `metrics.json`
- `ood_score_histogram.png`
- `history.json`
- `model.pt`
- optional raw prediction table for later analysis

Metrics:

- AUROC
- AUPRC
- False negative rate
- False positive rate
- Expected calibration error

Implementation notes:

- Reuse `set_cost_thresholds(data)`.
- Reuse `CVAE` and `Classifier` from `OSRL/osrl/common/net.py`.
- Construct labels consistently with the existing `CCAC.vae_loss` logic first,
  then refine if needed.
- Keep the script independent from actor/critic training.

## Test B: Cost-Critic Separation Script

Implemented path:

```text
validation_experiments/scripts/train_cost_critic_separation.py
```

Purpose:

- Train CVAE/classifier/cost critic for a limited number of steps.
- Compare IND and OOD `Q_c` values under matched query budgets.
- Test hard OOD mask, soft OOD score weighting, and no-classifier variants.

Inputs:

- `--task`
- `--device`
- `--seed`
- `--steps`
- `--batch_size`
- `--ood_mode {hard,soft,none}`
- `--output_dir`

Launcher:

```text
validation_experiments/scripts/run_test_b_cost_critic.sh
```

Outputs:

- `metrics.json`
- `qc_gap_curve.png`
- `qc_histogram.png`
- `history.json`
- `model.pt`

Metrics:

- IND `Q_c` mean/std
- OOD `Q_c` mean/std
- OOD-minus-IND `Q_c` gap
- matched-budget OOD-minus-IND `Q_c` gap

Implementation notes:

- Start with analysis-only code instead of modifying full CCAC training.
- Generate candidate OOD `(s, a)` with CVAE.
- For `soft`, weight generated samples by classifier probability instead of
  selecting only scores above a threshold.
- The current script pushes generated OOD samples toward `matched IND Q_c +
  margin`, so `qc_matched_gap` is the primary separation metric.

## Test C: Small Full-Policy Variant

Planned changes:

- Add controlled config flags to CCAC or a variant training script.
- Compare original CCAC with the best variant from Test A/B.

Initial tasks:

- `OfflineBallRun-v0`
- optional `OfflineCarRun-v0`

Initial training budget:

- `update_steps=20000` or `50000`
- one seed

Evaluation target costs:

- `1`
- `2`
- `5`
- `10`
