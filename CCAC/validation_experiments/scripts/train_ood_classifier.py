#!/usr/bin/env python3
"""Classifier-only validation for CCAC OOD detection."""

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
from osrl.common.net import CVAE, Classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="OfflineBallRun-v0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=20000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--vae-hidden-size", type=int, default=512)
    parser.add_argument("--vae-latent-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--classifier-loss", choices=["bce", "weighted_bce", "focal"], default="bce")
    parser.add_argument("--ood-usage", choices=["hard", "soft"], default="hard")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--output-dir", default="validation_experiments/outputs/test_a")
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


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    preds = (scores >= threshold).astype(np.int64)

    positives = max(int(labels.sum()), 1)
    negatives = max(int((1 - labels).sum()), 1)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp_curve = np.cumsum(sorted_labels == 1)
    fp_curve = np.cumsum(sorted_labels == 0)

    tpr = np.concatenate([[0.0], tp_curve / positives, [1.0]])
    fpr = np.concatenate([[0.0], fp_curve / negatives, [1.0]])
    auroc = float(np.trapz(tpr, fpr))

    precision = tp_curve / np.maximum(np.arange(1, labels.size + 1), 1)
    recall = tp_curve / positives
    recall_with_zero = np.concatenate([[0.0], recall])
    precision_with_start = np.concatenate([[labels.mean()], precision])
    auprc = float(np.sum((recall_with_zero[1:] - recall_with_zero[:-1]) * precision_with_start[1:]))

    ece = expected_calibration_error(labels, scores)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "false_negative_rate": float(fn / positives),
        "false_positive_rate": float(fp / negatives),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": float((tp + tn) / labels.size),
        "positive_rate": float(labels.mean()),
        "mean_score_positive": float(scores[labels == 1].mean()) if (labels == 1).any() else 0.0,
        "mean_score_negative": float(scores[labels == 0].mean()) if (labels == 0).any() else 0.0,
        "ece": ece,
    }


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    total = labels.size
    ece = 0.0
    for start in np.linspace(0.0, 1.0, bins, endpoint=False):
        end = start + 1.0 / bins
        if end >= 1.0:
            mask = (scores >= start) & (scores <= end)
        else:
            mask = (scores >= start) & (scores < end)
        if not mask.any():
            continue
        bin_acc = labels[mask].mean()
        bin_conf = scores[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


@torch.no_grad()
def evaluate(
    classifier: Classifier,
    observations: torch.Tensor,
    actions: torch.Tensor,
    cost_thresholds: torch.Tensor,
    batch_size: int,
    eval_samples: int,
    threshold: float,
    device: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    classifier.eval()
    all_scores, all_labels = [], []
    remaining = eval_samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        obs, act, _, query_kappa, labels = sample_batch(
            observations, actions, cost_thresholds, current_batch, device
        )
        sac = torch.cat([obs, act, query_kappa[:, None]], dim=1)
        scores = classifier(sac)
        all_scores.append(scores.detach().cpu().numpy().reshape(-1))
        all_labels.append(labels.detach().cpu().numpy().reshape(-1))
        remaining -= current_batch

    scores_np = np.concatenate(all_scores)
    labels_np = np.concatenate(all_labels)
    metrics = binary_metrics(labels_np, scores_np, threshold)
    classifier.train()
    return metrics, labels_np, scores_np


def save_histogram(labels: np.ndarray, scores: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(scores[labels == 0], bins=40, range=(0, 1), alpha=0.65, label="IND label 0")
    plt.hist(scores[labels == 1], bins=40, range=(0, 1), alpha=0.65, label="OOD label 1")
    plt.xlabel("OOD score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    random.seed(args.seed)

    run_name = args.run_name or f"{args.task}_{args.classifier_loss}_{args.ood_usage}_seed{args.seed}"
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

    cvae = CVAE(
        state_dim,
        action_dim,
        args.vae_hidden_size,
        args.vae_latent_size,
        max_action,
        state_min,
        state_max,
        device,
    ).to(device)
    classifier = Classifier(state_dim, action_dim, args.vae_hidden_size).to(device)

    cvae_optim = torch.optim.Adam(cvae.parameters(), lr=args.lr)
    classifier_optim = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    history = []
    for step in trange(args.steps, desc=f"Test A {run_name}"):
        obs, act, original_kappa, query_kappa, labels = sample_batch(
            train_obs, train_act, train_kappa, args.batch_size, device
        )

        recon_sa, mean, std = cvae(obs, act, original_kappa)
        true_sa = torch.cat([obs, act], dim=1)
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

        scores = classifier(sac)
        class_loss = classifier_loss(
            scores, class_labels, args.classifier_loss, args.focal_alpha, args.focal_gamma
        )
        classifier_optim.zero_grad()
        class_loss.backward()
        classifier_optim.step()

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            metrics, _, _ = evaluate(
                classifier,
                val_obs,
                val_act,
                val_kappa,
                args.batch_size,
                args.eval_samples,
                args.threshold,
                device,
            )
            row = {
                "step": step + 1,
                "loss_vae": float(vae_loss.item()),
                "recon_loss": float(recon_loss.item()),
                "kl_loss": float(kl_loss.item()),
                "loss_classifier": float(class_loss.item()),
                **metrics,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True))

    final_metrics, labels_np, scores_np = evaluate(
        classifier,
        val_obs,
        val_act,
        val_kappa,
        args.batch_size,
        args.eval_samples,
        args.threshold,
        device,
    )
    final_metrics.update({
        "task": args.task,
        "seed": args.seed,
        "steps": args.steps,
        "classifier_loss": args.classifier_loss,
        "ood_usage": args.ood_usage,
        "threshold": args.threshold,
        "run_name": run_name,
    })

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2, sort_keys=True)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, sort_keys=True)

    save_histogram(labels_np, scores_np, output_dir / "ood_score_histogram.png")
    torch.save(
        {
            "cvae": cvae.state_dict(),
            "classifier": classifier.state_dict(),
            "args": vars(args),
            "metrics": final_metrics,
        },
        output_dir / "model.pt",
    )
    print(f"saved outputs to {output_dir}")
    print(json.dumps(final_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
