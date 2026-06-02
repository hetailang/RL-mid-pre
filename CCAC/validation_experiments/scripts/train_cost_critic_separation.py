#!/usr/bin/env python3
"""Cost-critic IND/OOD separation validation for CCAC."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import trange

import dsrl  # noqa: F401
from osrl.common.dataset import set_cost_thresholds
from osrl.common.exp_util import seed_all
from osrl.common.net import CVAE, Classifier, EnsembleQCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="OfflineBallRun-v0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=20000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--critic-hidden-size", type=int, default=256)
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--num-qc", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-loss", choices=["bce", "weighted_bce", "focal"], default="bce")
    parser.add_argument("--ood-mode", choices=["hard", "soft", "none"], default="hard")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--ood-weight", type=float, default=1.0)
    parser.add_argument("--ood-margin", type=float, default=5.0)
    parser.add_argument("--output-dir", default="validation_experiments/outputs/test_b")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def sample_batch(
    observations: torch.Tensor,
    actions: torch.Tensor,
    cost_thresholds: torch.Tensor,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = observations.shape[0]
    idx = torch.randint(0, n, (batch_size,), device=device)
    query_idx = torch.randint(0, n, (batch_size,), device=device)
    obs = observations[idx]
    act = actions[idx]
    original_kappa = cost_thresholds[idx]
    query_kappa = cost_thresholds[query_idx]
    labels = (original_kappa > query_kappa).float()
    return obs, act, original_kappa, query_kappa, labels


def classifier_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    loss_name: str,
    focal_alpha: float,
    focal_gamma: float,
) -> torch.Tensor:
    probs = probs.clamp(1e-6, 1.0 - 1e-6)
    if loss_name == "bce":
        return F.binary_cross_entropy(probs, labels)
    if loss_name == "weighted_bce":
        positives = labels.sum().clamp_min(1.0)
        negatives = (labels.numel() - labels.sum()).clamp_min(1.0)
        pos_weight = (negatives / positives).detach()
        weights = torch.where(labels > 0.5, pos_weight, torch.ones_like(labels))
        return F.binary_cross_entropy(probs, labels, weight=weights)
    bce = F.binary_cross_entropy(probs, labels, reduction="none")
    p_t = torch.where(labels > 0.5, probs, 1.0 - probs)
    alpha_t = torch.where(labels > 0.5, focal_alpha, 1.0 - focal_alpha)
    return (alpha_t * (1.0 - p_t).pow(focal_gamma) * bce).mean()


def plot_curve(history, output_path: Path) -> None:
    steps = [row["step"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, [row["qc_ind_mean"] for row in history], label="IND Qc")
    plt.plot(steps, [row["qc_ood_mean"] for row in history], label="OOD Qc")
    plt.plot(steps, [row["qc_gap"] for row in history], label="OOD-IND gap")
    plt.xlabel("Training step")
    plt.ylabel("Q_c")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_hist(ind_qc: np.ndarray, ood_qc: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(ind_qc, bins=50, alpha=0.65, label="IND Qc")
    plt.hist(ood_qc, bins=50, alpha=0.65, label="OOD Qc")
    plt.xlabel("Q_c")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


@torch.no_grad()
def evaluate(
    cvae: CVAE,
    classifier: Classifier,
    cost_critic: EnsembleQCritic,
    observations: torch.Tensor,
    actions: torch.Tensor,
    cost_thresholds: torch.Tensor,
    batch_size: int,
    eval_samples: int,
    threshold: float,
    device: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    cvae.eval()
    classifier.eval()
    cost_critic.eval()
    ind_values, ind_matched_values, ood_values, ood_scores = [], [], [], []
    remaining = eval_samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        obs, act, original_kappa, query_kappa, _ = sample_batch(
            observations, actions, cost_thresholds, current_batch, device
        )

        ind_sc = torch.cat([obs, original_kappa[:, None]], dim=1)
        ind_qc, _ = cost_critic.predict(ind_sc, act)
        matched_ind_sc = torch.cat([obs, query_kappa[:, None]], dim=1)
        matched_ind_qc, _ = cost_critic.predict(matched_ind_sc, act)

        noise = torch.randn(current_batch, cvae.latent_dim, device=device).clamp(-1.0, 1.0)
        gen_sa = cvae.decode(obs, query_kappa[:, None], noise)
        gen_s = gen_sa[:, : observations.shape[1]]
        gen_a = gen_sa[:, observations.shape[1] :]
        gen_sc = torch.cat([gen_s, query_kappa[:, None]], dim=1)
        gen_sac = torch.cat([gen_sa, query_kappa[:, None]], dim=1)
        scores = classifier(gen_sac)
        ood_qc, _ = cost_critic.predict(gen_sc, gen_a)

        ind_values.append(ind_qc.detach().cpu().numpy())
        ind_matched_values.append(matched_ind_qc.detach().cpu().numpy())
        ood_values.append(ood_qc.detach().cpu().numpy())
        ood_scores.append(scores.detach().cpu().numpy())
        remaining -= current_batch

    ind_np = np.concatenate(ind_values)
    matched_ind_np = np.concatenate(ind_matched_values)
    ood_np = np.concatenate(ood_values)
    score_np = np.concatenate(ood_scores)
    selected = score_np >= threshold
    selected_rate = float(selected.mean())
    selected_ood = ood_np[selected] if selected.any() else ood_np
    metrics = {
        "qc_ind_mean": float(ind_np.mean()),
        "qc_ind_std": float(ind_np.std()),
        "qc_matched_ind_mean": float(matched_ind_np.mean()),
        "qc_matched_ind_std": float(matched_ind_np.std()),
        "qc_ood_mean": float(ood_np.mean()),
        "qc_ood_std": float(ood_np.std()),
        "qc_selected_ood_mean": float(selected_ood.mean()),
        "qc_selected_ood_std": float(selected_ood.std()),
        "qc_gap": float(ood_np.mean() - ind_np.mean()),
        "qc_matched_gap": float(ood_np.mean() - matched_ind_np.mean()),
        "qc_selected_gap": float(selected_ood.mean() - ind_np.mean()),
        "qc_selected_matched_gap": float(selected_ood.mean() - matched_ind_np.mean()),
        "ood_score_mean": float(score_np.mean()),
        "ood_selected_rate": selected_rate,
    }
    cvae.train()
    classifier.train()
    cost_critic.train()
    return metrics, ind_np, ood_np


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    random.seed(args.seed)

    run_name = args.run_name or f"{args.task}_{args.classifier_loss}_{args.ood_mode}_seed{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.task)
    data = env.get_dataset()
    state_min, state_max = set_cost_thresholds(data)
    observations_np = data["observations"].astype(np.float32)
    actions_np = np.clip(data["actions"].astype(np.float32), -1.0, 1.0)
    cost_thresholds_np = data["cost_thresholds"].astype(np.float32)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(observations_np.shape[0])
    val_size = int(observations_np.shape[0] * args.val_ratio)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    device = args.device
    train_obs = torch.tensor(observations_np[train_idx], device=device)
    train_act = torch.tensor(actions_np[train_idx], device=device)
    train_kappa = torch.tensor(cost_thresholds_np[train_idx], device=device)
    val_obs = torch.tensor(observations_np[val_idx], device=device)
    val_act = torch.tensor(actions_np[val_idx], device=device)
    val_kappa = torch.tensor(cost_thresholds_np[val_idx], device=device)

    state_dim = train_obs.shape[1]
    action_dim = train_act.shape[1]
    max_action = float(env.action_space.high[0])

    cvae = CVAE(state_dim, action_dim, args.hidden_size, args.latent_size,
                max_action, state_min, state_max, device).to(device)
    classifier = Classifier(state_dim, action_dim, args.hidden_size).to(device)
    cost_critic = EnsembleQCritic(
        state_dim, action_dim, [args.critic_hidden_size, args.critic_hidden_size],
        torch.nn.ReLU, cost_conditioned=True, num_q=args.num_qc
    ).to(device)

    cvae_optim = torch.optim.Adam(cvae.parameters(), lr=args.vae_lr)
    classifier_optim = torch.optim.Adam(classifier.parameters(), lr=args.vae_lr)
    critic_optim = torch.optim.Adam(cost_critic.parameters(), lr=args.critic_lr)

    history = []
    for step in trange(args.steps, desc=f"Test B {run_name}"):
        obs, act, original_kappa, query_kappa, labels = sample_batch(
            train_obs, train_act, train_kappa, args.batch_size, device
        )
        true_sa = torch.cat([obs, act], dim=1)

        recon_sa, mean, std = cvae(obs, act, original_kappa)
        recon_loss = F.mse_loss(recon_sa, true_sa)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + args.beta * kl_loss
        cvae_optim.zero_grad()
        vae_loss.backward()
        cvae_optim.step()

        true_sac = torch.cat([true_sa, query_kappa[:, None]], dim=1)
        recon_sac = torch.cat([recon_sa.detach(), query_kappa[:, None]], dim=1)
        sac = torch.cat([true_sac, recon_sac], dim=0)
        class_labels = torch.cat([labels, labels], dim=0)
        class_scores = classifier(sac)
        class_loss = classifier_loss(
            class_scores, class_labels, args.classifier_loss,
            args.focal_alpha, args.focal_gamma
        )
        classifier_optim.zero_grad()
        class_loss.backward()
        classifier_optim.step()

        ind_sc = torch.cat([obs, original_kappa[:, None]], dim=1)
        _, qc_list = cost_critic.predict(ind_sc, act)
        ind_loss = cost_critic.loss(original_kappa, qc_list)

        matched_ind_sc = torch.cat([obs, query_kappa[:, None]], dim=1)
        matched_ind_qc, _ = cost_critic.predict(matched_ind_sc, act)

        with torch.no_grad():
            noise = torch.randn(args.batch_size, cvae.latent_dim, device=device).clamp(-1.0, 1.0)
            gen_sa = cvae.decode(obs, query_kappa[:, None], noise)
            gen_sac = torch.cat([gen_sa, query_kappa[:, None]], dim=1)
            ood_scores = classifier(gen_sac)

        gen_s = gen_sa[:, :state_dim]
        gen_a = gen_sa[:, state_dim:]
        gen_sc = torch.cat([gen_s, query_kappa[:, None]], dim=1)
        qc_ood, _ = cost_critic.predict(gen_sc, gen_a)

        if args.ood_mode == "none":
            weights = torch.ones_like(qc_ood)
        elif args.ood_mode == "soft":
            weights = ood_scores.detach().clamp(0.0, 1.0)
        else:
            weights = (ood_scores.detach() >= args.threshold).float()

        if weights.sum() > 0:
            ood_target = matched_ind_qc.detach() + args.ood_margin
            ood_margin = F.relu(ood_target - qc_ood).pow(2)
            ood_loss = (weights * ood_margin).sum() / weights.sum().clamp_min(1.0)
        else:
            ood_loss = torch.zeros((), device=device)

        critic_loss = ind_loss + args.ood_weight * ood_loss
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            metrics, _, _ = evaluate(
                cvae, classifier, cost_critic, val_obs, val_act, val_kappa,
                args.batch_size, args.eval_samples, args.threshold, device
            )
            row = {
                "step": step + 1,
                "loss_vae": float(vae_loss.item()),
                "loss_classifier": float(class_loss.item()),
                "loss_ind_critic": float(ind_loss.item()),
                "loss_ood": float(ood_loss.item()),
                "loss_critic": float(critic_loss.item()),
                "train_ood_weight_mean": float(weights.mean().item()),
                **metrics,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True))

    final_metrics, ind_qc, ood_qc = evaluate(
        cvae, classifier, cost_critic, val_obs, val_act, val_kappa,
        args.batch_size, args.eval_samples, args.threshold, device
    )
    final_metrics.update({
        "task": args.task,
        "seed": args.seed,
        "steps": args.steps,
        "classifier_loss": args.classifier_loss,
        "ood_mode": args.ood_mode,
        "threshold": args.threshold,
        "run_name": run_name,
    })

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2, sort_keys=True)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, sort_keys=True)
    plot_curve(history, output_dir / "qc_gap_curve.png")
    plot_hist(ind_qc, ood_qc, output_dir / "qc_histogram.png")
    torch.save(
        {
            "cvae": cvae.state_dict(),
            "classifier": classifier.state_dict(),
            "cost_critic": cost_critic.state_dict(),
            "args": vars(args),
            "metrics": final_metrics,
        },
        output_dir / "model.pt",
    )
    print(f"saved outputs to {output_dir}")
    print(json.dumps(final_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
