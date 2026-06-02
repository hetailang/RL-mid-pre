# Robust Offline Safe RL under Dynamic Safety Budgets

> Concise bilingual script based on the updated PPT.  
> Target time: about 5-6 minutes.  
> Suggested use: present with the English script; use Chinese for rehearsal.

## Slide 1: Title

**English script:**  
Hello everyone. Our project is about **Robust Offline Safe RL under Dynamic Safety Budgets**. We focus on learning a safe policy from offline data, and our small improvement is to make OOD regularization softer and more calibrated.

**中文翻译：**  
大家好，我们的项目是 **动态安全预算下的鲁棒离线安全强化学习**。我们关注如何从离线数据中学习安全策略，并尝试通过更 soft、更 calibrated 的 OOD regularization 做一个小改进。

## Slide 2: Background

**English script:**  
In many safety-critical domains, online exploration is too risky. For example, robots may collide or fall, autonomous vehicles cannot explore dangerous actions, and healthcare policies must satisfy strict risk limits. This motivates offline RL: learning only from a fixed dataset without new environment interaction.

**中文翻译：**  
在很多安全关键场景中，在线探索风险太高。比如机器人可能碰撞或摔倒，自动驾驶不能尝试危险动作，医疗策略也必须满足严格风险限制。因此，我们希望使用 offline RL，只从固定离线数据中学习，不进行新的环境交互。

## Slide 3: Problem Setup

**English script:**  
Offline safe RL adds a safety constraint to offline RL. The policy should maximize reward while keeping cumulative cost below a threshold. In this project, we consider three challenges: OOD state-action pairs from offline data, cost constraint satisfaction, and safety budgets that may change during deployment. Our goal is a high-reward and safe policy that adapts to different remaining budgets.

**中文翻译：**  
Offline safe RL 在 offline RL 基础上加入了安全约束。策略需要最大化 reward，同时让 cumulative cost 不超过阈值。我们关注三个挑战：离线数据带来的 OOD state-action pairs，cost constraint 的满足，以及部署时可能变化的 safety budget。我们的目标是学习一个高 reward、安全，并能适应不同剩余预算的策略。

## Slide 4: CCAC Backbone

**English script:**  
We use CCAC as the existing backbone. Its key idea is budget conditioning: the remaining budget kappa is part of the policy input. After each step, the budget is updated by subtracting the incurred cost. So the same policy can behave differently under different budgets: more aggressive when the budget is large, and more conservative when the budget is small.

**中文翻译：**  
我们使用 CCAC 作为 backbone。它的核心思想是 budget conditioning：把剩余安全预算 kappa 作为 policy input 的一部分。每一步后，budget 会减去当前 cost。因此，同一个 policy 可以在不同预算下表现不同：预算大时更积极，预算小时更保守。

## Slide 5: How Safety Is Enforced

**English script:**  
CCAC enforces safety through an OOD classifier and a cost critic. A CVAE generates state-action pairs under a given budget. The classifier decides whether a generated pair is OOD or unsafe. Then the cost critic overestimates the cost of unsafe samples, so the actor learns to avoid them while maximizing reward.

**中文翻译：**  
CCAC 通过 OOD classifier 和 cost critic 实现安全约束。CVAE 在给定 budget 下生成 state-action pairs。classifier 判断这些样本是否 OOD 或 unsafe。然后 cost critic 对 unsafe samples 高估 cost，使 actor 在最大化 reward 的同时避开这些高风险动作。

## Slide 6: Our Observation

**English script:**  
Our observation is that the OOD classifier uses a hard threshold. If the score is above 0.5, the sample is OOD; otherwise, it is in-distribution. This can be brittle. Scores like 0.49 and 0.51 are almost the same, but they are treated completely differently. Also, with imbalanced data, the classifier may miss truly unsafe OOD samples. This motivates us to ask: can we make OOD regularization softer and more calibrated?

**中文翻译：**  
我们的观察是，OOD classifier 使用 hard threshold。score 大于 0.5 就是 OOD，否则就是 in-distribution。这个机制可能比较脆弱。比如 0.49 和 0.51 很接近，但会被完全不同地处理。另外，在数据不平衡时，classifier 可能漏掉真正 unsafe 的 OOD 样本。因此我们想问：能不能让 OOD regularization 更 soft、更 calibrated？

## Slide 7: Our Improvement

**English script:**  
Our improvement is **Soft and Calibrated OOD Regularization**. Instead of a binary OOD decision, we use a soft OOD weight computed from the classifier score. This weight controls the cost overestimation penalty. Higher OOD confidence means stronger penalty, while boundary samples are treated more smoothly. We also test weighted BCE and focal loss to reduce class imbalance, and evaluate calibration and false negative rate.

**中文翻译：**  
我们的小改进是 **Soft and Calibrated OOD Regularization**。我们不再使用二值 OOD 判断，而是根据 classifier score 计算 soft OOD weight。这个 weight 用来控制 cost overestimation penalty。OOD confidence 越高，penalty 越强；边界样本则被更平滑地处理。我们还测试 weighted BCE 和 focal loss 来缓解类别不平衡，并评估 calibration 和 false negative rate。

## Slide 8: Preliminary Validation

**English script:**  
We ran small-scale validation on OfflineBallRun-v0. First, in classifier-only testing, focal loss reduces OOD false negative rate from 10.4 percent to 3.6 percent. This means fewer dangerous OOD samples are missed. Second, in cost critic analysis, focal-soft gives a matched gap of 8.71, showing that OOD samples receive higher cost values than matched IND samples. Third, in a 50k-step policy run, realized cost stays zero under budgets 1, 2, 5, and 10.

**中文翻译：**  
我们在 OfflineBallRun-v0 上做了小规模验证。第一，在 classifier-only 测试中，focal loss 将 OOD false negative rate 从 10.4% 降到 3.6%，说明更少危险 OOD 样本被漏掉。第二，在 cost critic 分析中，focal-soft 的 matched gap 是 8.71，说明 OOD 样本获得了更高 cost value。第三，在 50k-step policy run 中，budgets 为 1、2、5、10 时 realized cost 都保持为 0。

## Slide 9: Feasibility Evidence

**English script:**  
Compared with the 50k-step original CCAC baseline, focal-soft keeps zero realized cost and improves reward across all tested budgets. The reward gain ranges from about 16 to 27 percent. This is still preliminary, but it suggests that the modification is small, trainable, and has a positive reward-cost signal.

**中文翻译：**  
与 50k-step original CCAC baseline 相比，focal-soft 在所有测试预算下都保持 zero realized cost，同时提升了 reward。reward gain 大约在 16% 到 27% 之间。虽然这仍然是 preliminary result，但说明这个修改很小、可训练，并且有正向的 reward-cost signal。

## Slide 10: Next Steps

**English script:**  
For the final presentation, we need to test robustness. We will repeat experiments with more seeds and at least one additional task. We will report calibration metrics such as ECE and reliability curves. We will also run ablations: hard versus soft OOD, BCE versus focal loss, and fixed versus varying budgets. Finally, we will evaluate reward, realized cost, violation rate, OOD false negative rate, and budget adaptation.

**中文翻译：**  
期末展示前，我们需要验证结果是否稳定。我们会用更多 seeds，并至少增加一个任务。我们会报告 calibration metrics，比如 ECE 和 reliability curves。我们还会做 ablation：hard versus soft OOD、BCE versus focal loss、fixed versus varying budgets。最后评估 reward、realized cost、violation rate、OOD false negative rate 和 budget adaptation。

## Slide 11: Related Work

**English script:**  
Related work can be grouped into four directions. Offline RL methods handle distribution shift but usually ignore safety constraints. Safe RL methods optimize constraints but often require online interaction. Offline safe RL methods learn reward-cost trade-offs from offline data, but adapting to changing budgets is still hard. Budget-conditioned OSRL addresses dynamic budgets, but the hard OOD threshold may be brittle. Our angle is to make this OOD safety signal softer, more calibrated, and easier to evaluate.

**中文翻译：**  
相关工作可以分成四类。Offline RL 主要处理 distribution shift，但通常没有显式 safety constraints。Safe RL 可以处理约束优化，但通常需要 online interaction。Offline safe RL 能从离线数据中学习 reward-cost trade-off，但适应变化预算仍然困难。Budget-conditioned OSRL 处理 dynamic budgets，但 hard OOD threshold 可能比较脆弱。我们的切入点是让 OOD safety signal 更 soft、更 calibrated，并且更容易评估。

## Slide 12: Closing

**English script:**  
To summarize, we study robust offline safe RL under dynamic safety budgets. Starting from a budget-conditioned actor-critic backbone, we identify a brittle hard OOD gate and propose soft OOD regularization. Preliminary results show lower OOD false negative rate, clearer cost critic separation, and better reward while keeping zero realized cost. Next, we will verify these findings with more seeds, tasks, and ablations. Thank you.

**中文翻译：**  
总结一下，我们研究的是 dynamic safety budgets 下的鲁棒 offline safe RL。基于 budget-conditioned actor-critic backbone，我们发现 hard OOD gate 可能比较脆弱，并提出 soft OOD regularization。初步结果显示，它降低了 OOD false negative rate，增强了 cost critic separation，并在保持 zero realized cost 的同时提升 reward。接下来我们会用更多 seeds、任务和 ablations 验证这些结果。谢谢大家。

