---
theme: seriph
title: Robust Offline Safe RL under Dynamic Safety Budgets
info: RL Mid-term Presentation
class: text-slate-950
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Robust Offline Safe RL under Dynamic Safety Budgets

## From CCAC to Soft & Calibrated OOD Regularization

RL Mid-term Presentation  
2026.06.03

<style>
.slidev-layout.cover {
  background: #ffffff !important;
  color: #0f172a !important;
}

.slidev-layout.cover h1,
.slidev-layout.cover h2,
.slidev-layout.cover p {
  color: #0f172a !important;
  text-shadow: none !important;
}
</style>

---
layout: center
---

# Background: Offline Safe RL

Traditional RL learns by repeatedly interacting with the environment.

In safety-critical domains, online exploration can be too expensive or risky, so we want to learn from an offline dataset.

<div class="mt-10 grid grid-cols-3 gap-4 text-center">
  <div class="rounded border border-slate-200 p-4">
    <div class="text-xl font-semibold">Robotics</div>
    <div class="mt-2 text-sm text-slate-700">Avoid collisions, falls, and unsafe motions</div>
  </div>
  <div class="rounded border border-slate-200 p-4">
    <div class="text-xl font-semibold">Autonomous Driving</div>
    <div class="mt-2 text-sm text-slate-700">Unsafe exploration cannot be used to collect data</div>
  </div>
  <div class="rounded border border-slate-200 p-4">
    <div class="text-xl font-semibold">Healthcare</div>
    <div class="mt-2 text-sm text-slate-700">Policies must satisfy strict risk budgets</div>
  </div>
</div>

---

# Problem Setup

Offline Safe RL optimizes reward while constraining cumulative safety cost:

$$
\max_\pi \ \mathbb{E}_{\pi}[R(\tau)]
\quad
\text{s.t.}
\quad
\mathbb{E}_{\pi}[C(\tau)] \le \epsilon
$$

It must solve three problems at the same time:

1. OOD state-action pairs caused by offline data
2. A policy that satisfies the cost constraint
3. A safety budget that may change during deployment

<div class="mt-6 rounded border border-slate-200 p-4">
Goal: learn a high-reward and safe policy that can adapt to different remaining budgets.
</div>

---

# CCAC: Existing Backbone

CCAC addresses this setting with a budget-conditioned actor-critic framework.

$$
\pi_\theta(a_t \mid s_t,\kappa_t),
\qquad
\kappa_{t+1}=\kappa_t-c_t
$$

The remaining budget $\kappa_t$ becomes part of the policy input, so the same policy can behave differently under different safety budgets.

$$
\underbrace{s_t}_{\text{state}}
\ ,\
\underbrace{\kappa_t}_{\text{remaining budget}}
\quad \longrightarrow \quad
\underbrace{\pi_\theta(a_t \mid s_t,\kappa_t)}_{\text{budget-conditioned actor}}
\quad \longrightarrow \quad
\underbrace{a_t}_{\text{action}}
$$

Take-away: CCAC turns a fixed-threshold safe RL problem into a dynamic-budget policy learning problem.

---

# CCAC: How Safety Is Enforced

The key mechanism is to identify generated OOD / unsafe state-action pairs and penalize them through the cost critic.

| Module | Role | Notation |
|---|---|---|
| Offline data | Build budget-annotated transitions | $D=\{(s_t,a_t,r_t,c_t,s_{t+1},\kappa_t)\}$ |
| Constraint-conditioned CVAE | Generate state-action pairs under a budget | $(\hat{s},\hat{a})\sim p_\phi(s,a\mid z,\kappa)$ |
| OOD classifier | Decide whether $(s,a,\kappa)$ is OOD / unsafe | $h_\psi(s,a\mid\kappa)>0.5$ |
| Cost critic | Overestimate cost for unsafe samples | $Q_c(s,a\mid\kappa)$ |
| Actor | Maximize reward while avoiding high cost | $\pi_\theta(a\mid s,\kappa)$ |

$$
p_\phi(s,a\mid z,\kappa)
\rightarrow
h_\psi(s,a\mid\kappa)
\rightarrow
Q_c(s,a\mid\kappa)
\rightarrow
\pi_\theta(a\mid s,\kappa)
$$

---

# Our Observation: A Brittle OOD Gate

CCAC relies heavily on the OOD classifier. If classifier decisions are inaccurate, cost overestimation can become unstable.

The original hard OOD rule is:

$$
\mathbb{I}_{\text{OOD}}(s,a,\kappa)
=
\mathbb{I}\left[h_\psi(s,a\mid\kappa)>0.5\right]
$$

This can cause two issues:

1. Classifier scores may not be calibrated; $0.49$ and $0.51$ are treated completely differently
2. Under imbalanced data, the classifier may miss truly dangerous OOD samples

<div class="mt-6 rounded border border-slate-200 p-4">
Question: can we make OOD regularization softer and more calibrated?
</div>

---

# Our Small Improvement

## Soft & Calibrated OOD Regularization

Instead of hard filtering, use a soft OOD weight:

$$
w_{\text{OOD}}(s,a,\kappa)
=
\sigma\left(\alpha\left(h_\psi(s,a\mid\kappa)-\tau\right)\right)
$$

Then use $w_{\text{OOD}}$ to weight the cost overestimation penalty, rather than making a binary decision.

We also improve classifier training with:

- weighted BCE or focal loss for class imbalance
- calibration analysis on classifier scores
- false negative rate analysis for unsafe OOD samples

---

# Expected Benefit and Evaluation

| Question | What we will measure |
|---|---|
| Does soft weighting reduce threshold brittleness? | Reward-cost trade-off near the safety boundary |
| Does calibration improve safety detection? | ECE / reliability curve of OOD scores |
| Does it catch dangerous OOD samples? | OOD false negative rate |
| Does budget adaptation remain stable? | Performance under different initial budgets $\kappa_1$ |

Expected outcome: similar or better reward, lower constraint violation, and more stable behavior when the OOD classifier is uncertain.

---

# Feasibility

We plan to build on CCAC and evaluate the modification on DSRL-style benchmark tasks.

| Component | Plan |
|---|---|
| Environments | Run / Circle / Velocity-style safety control tasks |
| Baseline | Original CCAC with hard OOD threshold |
| Variant | CCAC + soft OOD weight + calibrated classifier loss |
| Metrics | Reward, cost, constraint violation, OOD FNR, calibration, budget adaptation |
| Ablations | hard vs. soft OOD, BCE vs. focal loss, fixed vs. varying budgets |

<div class="mt-6 rounded border border-slate-200 p-4">
The change is localized to classifier training and cost regularization, so it is feasible without redesigning the whole algorithm.
</div>

---

# Related Work

<table class="text-[0.82rem] leading-tight">
  <thead>
    <tr>
      <th>Direction</th>
      <th>Methods</th>
      <th>What they address</th>
      <th>Remaining gap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Offline RL</td>
      <td>CQL, IQL, BCQ, BEAR</td>
      <td>Distribution shift</td>
      <td>No explicit safety constraints</td>
    </tr>
    <tr>
      <td>Safe RL</td>
      <td>Lagrangian, CPO, PID-Lagrangian</td>
      <td>Constrained optimization</td>
      <td>Often needs online interaction</td>
    </tr>
    <tr>
      <td>Offline Safe RL</td>
      <td>CPQ, COptiDICE, VOCE, FISOR</td>
      <td>Reward-cost trade-off from offline data</td>
      <td>Hard to adapt to changing budgets</td>
    </tr>
    <tr>
      <td>Budget-conditioned OSRL</td>
      <td>CCAC</td>
      <td>Dynamic budget adaptation + OOD detection</td>
      <td>Classifier threshold may be brittle</td>
    </tr>
  </tbody>
</table>

<div class="mt-4 text-xl font-semibold">
Our angle: make the OOD safety signal softer, calibrated, and easier to evaluate.
</div>

---
layout: center
---

# Q&A

Thank you!

---

# References

- Guo, Z., Zhou, W., Wang, S., & Li, W. Constraint-Conditioned Actor-Critic for Offline Safe Reinforcement Learning. ICLR 2025.
- Levine, S., Kumar, A., Tucker, G., & Fu, J. Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. 2020.
- Kumar, A., Zhou, A., Tucker, G., & Levine, S. Conservative Q-Learning for Offline Reinforcement Learning. NeurIPS 2020.
- Liu, Z. et al. Datasets and Benchmarks for Offline Safe Reinforcement Learning. 2023.
- Liu, Z. et al. Constrained Decision Transformer for Offline Safe Reinforcement Learning. ICML 2023.
