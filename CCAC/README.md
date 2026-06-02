<h1 align="center">
<br>
CCAC: Constraint-Conditioned Actor-Critic for Offline Safe Reinforcement Learning
</h1>

<p align="center">
Repo for "<a href="https://openreview.net/forum?id=nrRkAAAufl" target="_blank">Constraint-Conditioned Actor-Critic for Offline Safe Reinforcement Learning</a>" [ICLR 2025]
</p>

<!-- The official implementation of OASIS, a **Data-centric** approach for offline safe RL. -->

## Method

**CCAC** is an offline safe RL method that models the relationship between state-action distributions and safety constraints in offline datasets. It leverages this relationship to regularize both critic and policy learning, enabling zero-shot adaptation to varying constraint thresholds. These thresholds can differ across rollouts or change dynamically over time during deployment.

## Installation
To install the packages, please first create a python environment with python==3.8, then run:

``` bash
cd OSRL
pip install -e .
cd ../DSRL
pip install -e .
```

## Training (optional)

To train a CCAC agent, simply run:
``` bash
cd OSRL/examples/train
python train_ccac.py --task <env_name> --param1 <args1> ... 
```
By default, the config file and the logs during training will be written to `logs\` folder and the training plots can be viewed online using Wandb.
The default parameters can be found in `OSRL/examples/configs/ccac_configs.py`.

## Evaluation

**The pre-trained models are available [here](https://drive.google.com/drive/folders/1cM7tr5My-wkzl0uxepyb99G_XESTm3TJ?usp=sharing).**
To evaluate a trained CCAC agent, simply run:
``` bash
cd OSRL/examples/eval
python eval_ccac.py --path <path_to_model> --eval_episodes <number_of_episodes> --target_costs <list_of_target_cost_thresholds>
```
It will load config file from `path_to_model/config.yaml` and model file from `path_to_model/checkpoints/model.pt`, run the number of episodes for each target cost threshold, and print the average normalized reward and cost.

## Validation Experiments

Small validation experiments for OOD classifier and cost-critic behavior are
tracked under `validation_experiments/`. Start with:

```bash
cd /mnt/afs/L202500188/CCAC
bash validation_experiments/scripts/train_original_ccac_50k.sh
bash validation_experiments/scripts/run_test_a_classifier.sh
bash validation_experiments/scripts/run_test_b_cost_critic.sh
```

See `validation_experiments/commands.md` for exact environment setup,
H100/PyTorch notes, dataset download fallback, and evaluation commands. Results
and current conclusions are in `validation_experiments/results.md`.

## Github Reference
- OSRL: https://github.com/liuzuxin/osrl
- DSRL: https://github.com/liuzuxin/dsrl

## Bibtex

If you find our code and paper can help, please consider citing our paper as:
```
@inproceedings{guoconstraint,
  title={Constraint-Conditioned Actor-Critic for Offline Safe Reinforcement Learning},
  author={Guo, Zijian and Zhou, Weichao and Wang, Shengao and Li, Wenchao},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```
