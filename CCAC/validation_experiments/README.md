# CCAC Validation Experiments

This folder records the small validation experiments for improving the CCAC OOD
classifier and OOD usage. The goal is to validate the idea with cheap,
controlled experiments before running a full paper-style benchmark.

## Folder Contents

- `commands.md`: runnable commands for sanity checks, training, and evaluation.
- `results.md`: place to record observed outputs and conclusions.
- `scripts.md`: planned experiment scripts and implementation notes.
- `scripts/`: runnable experiment scripts.

## Experiment Order

1. Sanity baseline: short original CCAC run on `OfflineBallRun-v0`.
2. Test A: classifier-only comparison.
3. Test B: cost-critic IND/OOD separation.
4. Test C: small full-policy comparison.

Do not start with the full paper benchmark. First confirm that the original code
runs, then validate the proposed classifier and OOD-use changes.

## Current Status

- Original CCAC sanity and 50k baseline scripts are implemented.
- Test A classifier-only scripts are implemented and initial results are recorded.
- Test B cost-critic separation scripts are implemented; the current version uses
  matched-budget IND/OOD comparison.
- Test C full-policy variant scripts are not implemented yet.
