# Experiment Results

Use this file to record command outputs, run directories, metrics, and short
conclusions.

## 0. Environment And GPU Check

Date:

Command:

```bash
python -c "import torch, dsrl, osrl; print('python ok'); print('cuda', torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Output:

```text
torch 2.4.1+cu121
CUDA runtime 12.1
NVIDIA H100 80GB HBM3
device capability (9, 0)
minimal CUDA matmul completed successfully
```

Conclusion:

- GPU environment is usable after upgrading PyTorch from 1.13.1+cu117 to 2.4.1+cu121.
- H100 training should use the upgraded environment because PyTorch 1.13 does not support sm_90.

## 1. Sanity Baseline: Original CCAC Short Run

Task:

- `OfflineBallRun-v0`

Training command:

```bash
python train_ccac.py --task OfflineBallRun-v0 --device cuda:0 --seed 0 --update_steps 10000 --eval_every 5000 --eval_episodes 5 --batch_size 512 --num_workers 4
```

Run directory:

```text
/mnt/afs/L202500188/CCAC/OSRL/examples/train/logs/OfflineBallRun-v0-cost-5/CCAC_eval_episodes5_eval_every5000_num_workers4_update_steps10000-3dff/CCAC_eval_episodes5_eval_every5000_num_workers4_update_steps10000-3dff
```

Final training/eval output:

```text
CVAE warmup: 10000/10000 in 00:39, 252.74 it/s
Training: 10000/10000 in 03:28, 48.01 it/s

wandb run summary:
eval/Reward 373.32395
eval/Cost 11.0
eval/Length 100.0
loss/pred_acc 0.90786
loss/loss_class 0.20838
loss/loss_vae 0.03879
loss/recon_loss_sa 0.01436
loss/kl_loss 0.04887
train/best_idx 9999.0
```

Notes:

- Original CCAC sanity baseline completed successfully on `OfflineBallRun-v0`.
- This is a short run only and is not expected to reproduce the paper benchmark.
- Final cost is above the default cost limit 5, so multi-budget evaluation is needed before drawing any policy conclusions.

## 2. Sanity Baseline Evaluation

Target costs:

- `1`
- `2`
- `5`
- `10`

Evaluation output:

```text
Eval reward 390.46926645364874, normalized reward: 0.2798615697811066; target cost 1.0, real cost 1.0, normalized cost: 1.0
Eval reward 365.4602146041528, normalized reward: 0.26064018763005453; target cost 2.0, real cost 5.0, normalized cost: 2.5
Eval reward 373.3175058101646, normalized reward: 0.26667912096714136; target cost 5.0, real cost 11.0, normalized cost: 2.2
Eval reward 397.50222850618235, normalized reward: 0.28526694268684993; target cost 10.0, real cost 20.0, normalized cost: 2.0
```

Short conclusion:

- Evaluation runs successfully from the saved checkpoint.
- The short original CCAC sanity run exactly satisfies target cost 1, but violates larger budgets by roughly 2.0x-2.5x.
- This confirms the pipeline is usable, but the short run is not a reliable policy-quality baseline.
- Next step should be Test A classifier-only evaluation before spending more time on full-policy training.

## 2.5 Original CCAC 50k Baseline

Task:

- `OfflineBallRun-v0`

Training command:

```bash
bash validation_experiments/scripts/train_original_ccac_50k.sh
```

Run directory:

```text
/mnt/afs/L202500188/CCAC/OSRL/examples/train/logs/OfflineBallRun-v0-cost-5/CCAC_num_workers4_update_steps50000_sanity50k-1806/CCAC_num_workers4_update_steps50000_sanity50k-1806
```

Final training/eval output:

```text
CVAE warmup: 50000/50000 in 03:18, 251.67 it/s
Training: 50000/50000 in 17:42, 47.04 it/s

wandb run summary:
eval/Reward 353.54429
eval/Cost 0.0
eval/Length 100.0
loss/pred_acc 0.93046
loss/loss_class 0.15999
loss/loss_vae 0.03735
loss/recon_loss_sa 0.01318
loss/kl_loss 0.04834
train/best_idx 49999.0
```

Notes:

- 50k original CCAC baseline completed successfully.
- Final in-training evaluation has cost 0 at the default target cost 5.
- This run is a better original CCAC anchor for later Test C than the 10k sanity run.
- Multi-budget evaluation shows zero realized cost for all tested target costs.

Multi-budget evaluation:

```text
Eval reward 327.7773295376515, normalized reward: 0.2316779887119859; target cost 1.0, real cost 0.0, normalized cost: 0.0
Eval reward 344.68497169347756, normalized reward: 0.24467281367001087; target cost 2.0, real cost 0.0, normalized cost: 0.0
Eval reward 353.72738639135133, normalized reward: 0.25162260567020717; target cost 5.0, real cost 0.0, normalized cost: 0.0
Eval reward 362.9074458166811, normalized reward: 0.25867818824269034; target cost 10.0, real cost 0.0, normalized cost: 0.0
```

Short conclusion:

- The 50k original CCAC run is very conservative on `OfflineBallRun-v0`.
- It satisfies all tested budgets with zero cost, but reward is lower than the 10k short run.
- Later Test C should compare whether variants can preserve this safety while improving or at least not degrading reward.

## 3. Test A: Classifier-Only Evaluation

Status:

- Implemented in `validation_experiments/scripts/train_ood_classifier.py`.
- Launcher: `validation_experiments/scripts/run_test_a_classifier.sh`.

Metrics to record:

- AUROC
- AUPRC
- False negative rate
- False positive rate
- Expected calibration error
- OOD score histogram path

Results:

```text
run_name                                                classifier_loss  ood_usage  auroc              auprc              false_negative_rate  false_positive_rate  ece                  accuracy  positive_rate
original_OfflineBallRun-v0_seed0_steps5000              bce              hard       0.9426979002340196 0.9377758240664587 0.10379849685151331  0.17205042347843214  0.022266196687949193 0.86155   0.4923
ours_weighted_OfflineBallRun-v0_seed0_steps5000         weighted_bce     hard       0.9421872891374918 0.9367967798833445 0.1043063172862076   0.17037620642111484  0.02245371591806578  0.86215   0.4923
ours_focal_OfflineBallRun-v0_seed0_steps5000            focal            hard       0.9394261543067554 0.9332628011268744 0.0356489945155393   0.28166239905455975  0.10002276495615597  0.83945   0.4923
ours_focal_soft_OfflineBallRun-v0_seed0_steps5000       focal            soft       0.9394261543067554 0.9332628011268744 0.0356489945155393   0.28166239905455975  0.10002276495615597  0.83945   0.4923
```

Short conclusion:

- Focal loss greatly reduces false negative rate: about `0.104` to `0.036`.
- This comes with a higher false positive rate: about `0.17` to `0.28`.
- AUROC/AUPRC remain close but slightly lower for focal variants.
- Calibration is worse for focal variants in this first run.
- In Test A, focal loss gives the safety-relevant behavior we wanted: fewer OOD samples are missed as IND.

## 4. Test B: Cost-Critic Separation

Status:

- Implemented in `validation_experiments/scripts/train_cost_critic_separation.py`.
- Launcher: `validation_experiments/scripts/run_test_b_cost_critic.sh`.

Metrics to record:

- IND `Q_c` mean/std
- OOD `Q_c` mean/std
- OOD-minus-IND `Q_c` gap
- matched-budget OOD-minus-IND `Q_c` gap
- Histogram/curve paths

Results:

```text
Initial 10k results below are from the first, pre-matched Test B implementation.
They are kept as a negative control showing why the matched-budget revision was needed.
```

Test B 10k summary:

```text
run_name                                           classifier_loss  ood_mode  qc_ind_mean        qc_ood_mean        qc_gap              qc_selected_ood_mean  qc_selected_gap      ood_score_mean       ood_selected_rate
focal_hard_OfflineBallRun-v0_seed0_steps10000      focal            hard      33.23070526123047  33.43223571777344  0.20153045654296875 27.511934280395508   -5.718770980834961   0.5988857746124268   0.75545
focal_soft_OfflineBallRun-v0_seed0_steps10000      focal            soft      32.70475769042969  32.90208053588867  0.19732284545898438 27.072284698486328   -5.632472991943359   0.5988857746124268   0.75545
no_classifier_OfflineBallRun-v0_seed0_steps10000   bce              none      33.27153778076172  33.46988296508789  0.19834518432617188 19.247058868408203   -14.024478912353516  0.4521072804927826   0.4186
original_OfflineBallRun-v0_seed0_steps10000        bce              hard      33.27287673950195  33.470672607421875 0.19779586791992188 19.24806785583496    -14.024808883666992  0.4521072804927826   0.4186
```

Short conclusion:

- This first Test B implementation does not produce a strong OOD-vs-IND `Q_c` gap.
- Overall OOD-minus-IND gap is only about `0.20` for all variants.
- Focal variants select more generated samples as OOD (`0.755` vs `0.419`) and selected OOD `Q_c` is higher than original/no-classifier selected OOD (`27.1-27.5` vs `19.2`), but selected OOD `Q_c` is still below IND `Q_c`.
- This suggests the current analysis loss/evaluation construction is too weak or mismatched: it pushes generated samples only toward their query budget and does not force selected OOD samples above comparable IND samples.
- Next revision should evaluate matched-budget IND/OOD pairs and/or use a stronger OOD margin target before using Test B as evidence.

Matched-budget Test B 10k rerun:

```text
run_name                                           classifier_loss  ood_mode  qc_ind_mean        qc_matched_ind_mean  qc_ood_mean        qc_gap              qc_matched_gap     qc_selected_ood_mean  qc_selected_gap     qc_selected_matched_gap  ood_score_mean       ood_selected_rate
focal_hard_OfflineBallRun-v0_seed0_steps10000      focal            hard      33.87271499633789  71.43616485595703   77.96253204345703 44.08981704711914  6.5263671875      79.51565551757812   45.642940521240234 8.079490661621094       0.5988857746124268   0.75545
focal_soft_OfflineBallRun-v0_seed0_steps10000      focal            soft      33.70148849487305  49.101688385009766  57.81245040893555 24.1109619140625   8.710762023925781 55.23283386230469   21.53134536743164  6.131145477294922       0.5988857746124268   0.75545
no_classifier_OfflineBallRun-v0_seed0_steps10000   bce              none      32.84892654418945  39.477176666259766  45.99302673339844 13.144100189208984  6.515850067138672 40.20764923095703   7.358722686767578  0.7304725646972656      0.4521072804927826   0.4186
original_OfflineBallRun-v0_seed0_steps10000        bce              hard      50.14657974243164  186.58396911621094  159.7550506591797 109.60847473144531  -26.82891845703125 274.9081115722656  224.76153564453125 88.32414245605469       0.4521072804927826   0.4186
```

Matched-budget conclusion:

- The revised Test B now produces positive matched-budget OOD-vs-IND separation for `focal_hard`, `focal_soft`, and `no_classifier`.
- `focal_soft` is the cleanest candidate for Test C: `qc_matched_gap` is `8.71`, and selected OOD also stays above matched IND with `qc_selected_matched_gap` `6.13`.
- `focal_hard` also separates selected OOD well, but its matched IND and OOD means are more inflated than `focal_soft`.
- `original` has a positive selected matched gap, but its overall `qc_matched_gap` is negative and the Q scale is much less stable. It should not be selected as the improved variant.
- Proceed to Test C with `classifier_loss=focal` and `ood_mode=soft`.

## 5. Test C: Small Full-Policy Benchmark

Status:

- Implemented through CCAC training flags:
  - `classifier_loss`
  - `ood_mode`
  - `focal_alpha`
  - `focal_gamma`
- Ran one 50k full-policy variant on `OfflineBallRun-v0`, seed `0`.

Metrics to record:

- Normalized reward
- Normalized cost
- Violation rate
- Cost-budget alignment
- Small-budget safety result

Results:

```text
Variant: focal_soft
Command:
bash-equivalent:
cd OSRL/examples/train
python train_ccac.py --task OfflineBallRun-v0 --device cuda:0 --seed 0 --update_steps 50000 --eval_every 10000 --eval_episodes 10 --batch_size 512 --num_workers 4 --classifier_loss focal --ood_mode soft --suffix test_c_focal_soft_50k

Run directory:
/mnt/afs/L202500188/CCAC/OSRL/examples/train/logs/OfflineBallRun-v0-cost-5/CCAC_classifier_lossfocal_num_workers4_ood_modesoft_update_steps50000_test_c_focal_soft_50k-bb72/CCAC_classifier_lossfocal_num_workers4_ood_modesoft_update_steps50000_test_c_focal_soft_50k-bb72

In-training evaluation:
step    eval/Reward         eval/Cost
9999    408.092080289383    9.1
19999   416.9696128437551   2.0
29999   417.5955968841801   0.0
39999   412.87628277740953  0.0
49999   420.95643552709595  0.0
```

Multi-budget evaluation:

```text
Eval reward 418.0535022872787, normalized reward: 0.3010621791242246; target cost 1.0, real cost 0.0, normalized cost: 0.0
Eval reward 400.5586442512687, normalized reward: 0.2876160335441261; target cost 2.0, real cost 0.0, normalized cost: 0.0
Eval reward 421.16955686347677, normalized reward: 0.30345710701565703; target cost 5.0, real cost 0.0, normalized cost: 0.0
Eval reward 451.30088341050293, normalized reward: 0.32661535170897604; target cost 10.0, real cost 0.0, normalized cost: 0.0
```

Short conclusion:

- Test C gives a strong positive first signal on `OfflineBallRun-v0`.
- Compared with the 50k original CCAC baseline, `focal_soft` keeps zero realized cost across target costs `1/2/5/10`.
- It also substantially improves reward: original 50k multi-budget reward was about `328/345/354/363`; `focal_soft` is about `418/401/421/451`.
- The first 10k checkpoint is still unsafe at default target cost 5 (`cost=9.1`), but from 20k onward the run satisfies the default budget and reaches zero cost from 30k onward.
- Next validation should repeat `focal_soft` on at least one more seed or one additional task before claiming a robust policy-level improvement.
