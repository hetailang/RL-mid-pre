# Q&A Preparation for Updated Mid-term PPT

> Concise bilingual Q&A for the updated presentation.  
> Each answer is designed for about 15-30 seconds.

## Q1. What is the key idea of your project?

**问题中文翻译：你们项目的核心思路是什么？**

**English answer:**  
The key idea is to study offline safe RL under dynamic safety budgets. We use a budget-conditioned actor-critic backbone, where the remaining budget is part of the policy input. Our small improvement is to replace hard OOD filtering with soft OOD regularization, so that cost critic penalties become smoother and less sensitive to classifier uncertainty.

**中文回答：**  
核心思路是研究 dynamic safety budgets 下的 offline safe RL。我们使用 budget-conditioned actor-critic，把剩余安全预算作为 policy input。我们的小改进是用 soft OOD regularization 替代 hard OOD filtering，让 cost critic 的惩罚更平滑，也降低对 classifier 不确定性的敏感性。

## Q2. Why is the hard OOD threshold a problem?

**问题中文翻译：为什么 hard OOD threshold 是一个问题？**

**English answer:**  
The hard threshold makes a binary decision at 0.5. Samples with scores 0.49 and 0.51 are very close, but they are treated completely differently. If the classifier is not well calibrated or the data is imbalanced, this can make cost overestimation unstable and may miss truly unsafe OOD samples.

**中文回答：**  
hard threshold 在 0.5 处做二值判断。0.49 和 0.51 的样本其实很接近，但会被完全不同地处理。如果 classifier 没有校准好，或者数据不平衡，就可能导致 cost overestimation 不稳定，也可能漏掉真正 unsafe 的 OOD 样本。

## Q3. What exactly is your modification?

**问题中文翻译：你们具体改了什么？**

**English answer:**  
Instead of using a hard OOD indicator, we compute a soft OOD weight from the classifier score. This weight is used to scale the cost overestimation penalty. Higher OOD confidence leads to a stronger penalty, while uncertain boundary samples receive intermediate penalties instead of an abrupt binary decision.

**中文回答：**  
我们不再使用 hard OOD indicator，而是根据 classifier score 计算一个 soft OOD weight。这个 weight 用来缩放 cost overestimation penalty。OOD confidence 越高，penalty 越强；边界附近的不确定样本会得到中间强度的 penalty，而不是突然变成 0 或 1。

## Q4. Why use focal loss?

**问题中文翻译：为什么使用 focal loss？**

**English answer:**  
OOD and IND samples can be imbalanced, and standard BCE may under-focus on hard or minority OOD examples. Focal loss gives more attention to hard samples, which is useful because missing unsafe OOD samples is dangerous in safe RL. In our preliminary classifier test, focal loss reduced OOD false negative rate from 10.4 percent to 3.6 percent.

**中文回答：**  
OOD 和 IND 样本可能不平衡，普通 BCE 可能对难分类样本或少数类 OOD 样本关注不够。focal loss 会更关注 hard samples，而在 safe RL 中漏掉 unsafe OOD 样本是很危险的。我们的初步 classifier 测试中，focal loss 把 OOD false negative rate 从 10.4% 降到了 3.6%。

## Q5. Does focal loss guarantee better calibration?

**问题中文翻译：focal loss 一定能带来更好的 calibration 吗？**

**English answer:**  
No, focal loss does not guarantee better calibration. That is why we do not claim calibration improvement yet. For the final presentation, we plan to report ECE and reliability curves. If focal loss improves detection but hurts calibration, we may combine it with temperature scaling or present calibration as a separate analysis.

**中文回答：**  
不一定。focal loss 不保证 probability calibration 一定更好。所以我们现在不会过早声称 calibration 已经提升。期末我们计划报告 ECE 和 reliability curves。如果 focal loss 提升了 detection 但损害 calibration，我们可能结合 temperature scaling，或者把 calibration 作为单独分析。

## Q6. Why is false negative rate important?

**问题中文翻译：为什么 false negative rate 很重要？**

**English answer:**  
In this setting, a false negative means an unsafe OOD sample is classified as in-distribution. Then it may not receive cost overestimation, and the actor may learn to choose it. This can directly lead to constraint violations, so reducing OOD false negatives is especially important for safe RL.

**中文回答：**  
在这个 setting 里，false negative 指的是 unsafe OOD 样本被误判成 in-distribution。这样它可能不会受到 cost overestimation，actor 之后也可能学会选择它，从而直接导致 constraint violation。因此降低 OOD false negative 对 safe RL 特别重要。

## Q7. How do you know the cost critic is improved?

**问题中文翻译：你们怎么知道 cost critic 有改进？**

**English answer:**  
We check cost critic separation between matched OOD and IND samples. Ideally, OOD samples should have higher Q_c values than similar IND samples. In our preliminary analysis, the focal-soft variant produced a matched gap of 8.71, which suggests that the cost critic assigns higher cost to risky OOD samples.

**中文回答：**  
我们检查 matched OOD 和 IND 样本之间的 cost critic separation。理想情况下，OOD 样本的 Q_c 应该高于相似的 IND 样本。我们的初步分析中，focal-soft variant 得到 matched gap 8.71，说明 cost critic 确实给 risky OOD 样本分配了更高 cost。

## Q8. Are the preliminary results enough to prove the method works?

**问题中文翻译：这些 preliminary results 足以证明方法有效吗？**

**English answer:**  
Not yet. They are feasibility signals, not final evidence. They show that the modification is trainable and promising on OfflineBallRun-v0. For the final presentation, we need more seeds, at least one additional task, and ablations such as hard versus soft OOD and BCE versus focal loss.

**中文回答：**  
还不够。这些结果是 feasibility signals，不是最终证明。它们说明这个修改可以训练，并且在 OfflineBallRun-v0 上有希望。期末我们还需要更多 seeds、至少一个额外任务，以及 hard versus soft OOD、BCE versus focal loss 等 ablation。

## Q9. Why only 50k steps in the preliminary run?

**问题中文翻译：为什么 preliminary run 只跑 50k steps？**

**English answer:**  
The 50k-step run is only for early feasibility checking. Full offline RL training is expensive, so before running longer experiments, we first want to see whether the modification can be integrated into policy training and whether it gives a positive reward-cost signal. Longer and repeated runs are part of our next steps.

**中文回答：**  
50k-step run 只是早期 feasibility checking。完整 offline RL 训练成本较高，所以在跑更长实验前，我们先确认这个修改能否整合进 policy training，以及是否有正向 reward-cost signal。更长、更重复的 runs 是我们下一步要做的。

## Q10. Why does reward improve while cost stays zero?

**问题中文翻译：为什么 reward 提升了但 cost 仍然保持为 0？**

**English answer:**  
One possible explanation is that soft OOD regularization avoids over-penalizing uncertain but useful samples, while still strongly penalizing high-confidence unsafe samples. This may allow the actor to explore safer high-reward actions within the offline training distribution. However, we still need more seeds and tasks to confirm this pattern.

**中文回答：**  
一种可能解释是，soft OOD regularization 不会过度惩罚不确定但有用的样本，同时仍然强烈惩罚高置信度 unsafe 样本。这可能让 actor 在离线训练分布内找到更安全的高 reward 动作。不过这个现象还需要更多 seeds 和任务来确认。

## Q11. What baselines or ablations will you compare?

**问题中文翻译：你们会比较哪些 baseline 或 ablation？**

**English answer:**  
The main comparisons are original hard-threshold CCAC versus our soft OOD version. We also plan to compare BCE, weighted BCE, and focal loss for classifier training. In addition, we will compare hard versus soft OOD usage and fixed versus varying budget evaluation.

**中文回答：**  
主要比较 original hard-threshold CCAC 和我们的 soft OOD version。我们还计划比较 BCE、weighted BCE 和 focal loss 对 classifier training 的影响。另外会比较 hard versus soft OOD usage，以及 fixed versus varying budget evaluation。

## Q12. What metrics will you report?

**问题中文翻译：你们会报告哪些指标？**

**English answer:**  
At the classifier level, we will report OOD false negative rate, AUROC, AUPRC, and calibration error. At the critic level, we will report OOD-IND Q_c separation. At the policy level, we will report reward, realized cost, violation rate, and budget adaptation under different target budgets.

**中文回答：**  
classifier 层面，我们会报告 OOD false negative rate、AUROC、AUPRC 和 calibration error。critic 层面，我们会报告 OOD-IND Q_c separation。policy 层面，我们会报告 reward、realized cost、violation rate，以及不同 target budgets 下的 budget adaptation。

## Q13. What if soft OOD regularization makes the policy too conservative?

**问题中文翻译：如果 soft OOD regularization 让策略太保守怎么办？**

**English answer:**  
That is possible, because stronger safety regularization can reduce reward. We will tune the strength of the soft penalty and analyze the reward-cost trade-off. Our goal is not only higher reward, but higher reward while keeping cost and violation rate stable.

**中文回答：**  
这是有可能的，因为更强的 safety regularization 可能降低 reward。我们会调整 soft penalty 的强度，并分析 reward-cost trade-off。我们的目标不是单纯提高 reward，而是在 cost 和 violation rate 稳定的情况下提升 reward。

## Q14. What is the main risk of the project?

**问题中文翻译：这个项目最大的风险是什么？**

**English answer:**  
The main risk is that the preliminary result may not hold across more seeds or tasks. To handle this, our final plan includes repeated runs, one additional task, and multiple ablations. Even if the full policy result is mixed, the classifier and critic-level analyses can still explain when the soft OOD idea helps or fails.

**中文回答：**  
主要风险是 preliminary result 可能在更多 seeds 或任务上不稳定。为了解决这个问题，我们期末计划做 repeated runs、增加一个任务，并做多个 ablations。即使 full policy 结果不完全稳定，classifier 和 critic-level 分析也能解释 soft OOD 在什么时候有效或失败。

## Q15. What is the expected contribution?

**问题中文翻译：你们预期的贡献是什么？**

**English answer:**  
Our expected contribution is a small but focused improvement to budget-conditioned offline safe RL. We study whether hard OOD filtering can be replaced by soft OOD regularization, and evaluate it at three levels: classifier detection, cost critic separation, and policy reward-cost performance.

**中文回答：**  
我们的预期贡献是对 budget-conditioned offline safe RL 做一个小而明确的改进。我们研究 hard OOD filtering 是否可以被 soft OOD regularization 替代，并从三个层面评估它：classifier detection、cost critic separation 和 policy reward-cost performance。

## Short Emergency Answers

### If asked: What is the one-sentence novelty?

**问题中文翻译：一句话说你们的创新点是什么？**

**English answer:**  
We replace hard OOD filtering with soft OOD regularization, using classifier confidence to weight cost critic penalties instead of making a brittle binary decision.

**中文回答：**  
我们用 soft OOD regularization 替代 hard OOD filtering，用 classifier confidence 加权 cost critic penalty，而不是做脆弱的二值判断。

### If asked: Is this already proven?

**问题中文翻译：这个方法已经被证明有效了吗？**

**English answer:**  
Not fully yet. Current results are preliminary feasibility evidence. The final project will focus on robustness checks with more seeds, tasks, and ablations.

**中文回答：**  
还没有完全证明。目前结果只是初步 feasibility evidence。期末项目会重点做更多 seeds、任务和 ablation 来验证鲁棒性。

### If asked: Why should we care about calibration?

**问题中文翻译：为什么我们要关心 calibration？**

**English answer:**  
Because the classifier score is used as a safety signal. If the score is poorly calibrated, the cost critic may under-penalize unsafe samples or over-penalize useful ones.

**中文回答：**  
因为 classifier score 被用作安全信号。如果 score 校准不好，cost critic 可能低估 unsafe 样本，也可能过度惩罚有用样本。

