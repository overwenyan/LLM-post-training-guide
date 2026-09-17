# 第 8 讲　DPO 家族：把奖励模型折叠进策略

> **本讲在主线中的位置**：信号仍然是人类偏好（轴一不变），但数据回到完全离线（轴二左移）。这一讲要回答两个问题：**为什么可以不训 RM**，以及**这样做到底失去了什么**。
>
> **学完你应该能做到**
> 1. 从 KL 正则化目标完整推出 DPO，并说明配分函数为什么会抵消。
> 2. 解释似然位移（likelihood displacement）现象——为什么 chosen 的对数概率下降是常态，什么时候它变成问题。
> 3. 说清 IPO、KTO、ORPO、SimPO 各自修了 DPO 的哪一处（含原稿漏掉的 KTO 参考点）。
> 4. 在给定数据与算力条件下，判断该走 DPO 还是 PPO/GRPO。


## 0. 新增符号

$\hat r_\theta$（隐式奖励）、$Z(x)$（配分函数）、$\tau$（IPO 的目标间隔参数）、$\lambda_{\text{OR}}$（ORPO 权重）、$\delta$（SimPO margin）、$z_{\text{ref}}$（KTO 参考点）。见附录 A。


## 1. 动机：RLHF 的四项成本

第 7 讲的流水线要付四笔账：标注偏好数据、训练 RM、在线 rollout、四模型显存。DPO 的问题很直接：

> 能不能跳过 RM 和 PPO，直接用偏好对更新策略？

答案是可以——只要你接受**放弃在线探索**。


## 2. 推导


### 2.1 起点

第 7 讲的 RL 目标：

$$
\max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi(\cdot\mid x)}\big[r(x,y)\big]-\beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\Vert\,\pi_{\text{ref}}(\cdot\mid x)\big)
$$

这个问题有**闭式最优解**（对每个 $x$ 单独求解带 KL 正则的线性目标，是标准的 Gibbs 变分结果）：

$$
\pi^\star(y\mid x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\Big(\frac{1}{\beta}r(x,y)\Big),\qquad Z(x)=\sum_{y}\pi_{\text{ref}}(y\mid x)\exp\!\Big(\frac{1}{\beta}r(x,y)\Big)
$$

直觉：最优策略就是把参考模型按奖励做指数重加权；$\beta$ 控制重加权的锐度。


### 2.2 反解奖励

取对数整理：

$$
r(x,y)=\beta\log\frac{\pi^\star(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)
$$

**关键观察**：$\beta\log Z(x)$ 只依赖 $x$，与 $y$ 无关。$Z(x)$ 本身是对整个词表空间求和，根本算不出来——但下一步它会消失。


### 2.3 代入 Bradley–Terry

$$
P(y_w\succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)=\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)
$$

同一个 $x$ 下两个回答相减，$\beta\log Z(x)$ 抵消。**这正是第 7 讲 BT 不可辨识性的正面用途**：既然偏好只由奖励差决定，那个算不出来的常数就永远不需要算。


### 2.4 损失

定义隐式奖励 $\hat r_\theta(x,y)=\beta\log\dfrac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$，则

$$
\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{\text{pref}}}\Big[\log\sigma\big(\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l)\big)\Big]
$$

形式上就是一个逻辑回归。**RM 没有消失，它被折叠进了策略与参考模型的对数比里。**


## 3. 梯度：它在推谁

$$
\nabla_\theta\mathcal{L}_{\text{DPO}}=-\beta\,\mathbb{E}\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)}_{\text{判反得越厉害，权重越大}}\Big(\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x)\Big)\Big]
$$

与第 7 讲 RM 损失的梯度权重形式完全一致——因为它们是同一个逻辑回归，只是把 $r_\phi$ 换成了 $\hat r_\theta$。

用附录 C 的统一视角：DPO 的 $w_t=\pm\beta\sigma(\hat r_l-\hat r_w)$，**chosen 侧为正、rejected 侧为负**。这是 SFT 做不到的事——SFT 只有正权重。


## 4. 似然位移：DPO 最常被误解的现象

损失只管**差值** $\hat r_w-\hat r_l$ 变大。差值变大有三种方式：

| 方式 | $\log\pi_\theta(y_w)$ | $\log\pi_\theta(y_l)$ | 是否是我们想要的 |
|---|---|---|---|
| 理想 | 上升 | 下降 | 是 |
| 常见 | 略降 | 大降 | 可接受 |
| 危险 | **大降** | 更大降 | 否 |

训练日志里 chosen 的对数概率下降是**常态**，不是 bug。真正的问题是概率质量去了哪里：既然 $y_w$ 和 $y_l$ 的概率都在降，剩下的质量必然流向**第三类没有出现在数据里的回答**，而没有任何机制保证那些回答更好。极端情况下，模型会被推向既不像 chosen 也不像 rejected 的奇怪分布——这就是被称作**似然位移**的失效模式，它在 $y_w$ 与 $y_l$ 很相似时尤其严重。

**缓解**：

- 加一个 SFT 正则项（在 chosen 上的 NLL），把 $\log\pi_\theta(y_w)$ 拽住。[Llama 3](https://arxiv.org/abs/2407.21783) 的做法即属此类，同时还把格式类 token 从 DPO 损失里屏蔽掉。
- 只加在 chosen 概率低于参考模型时生效的惩罚项（DPO-Positive 一类的做法）。
- 监控指标不要只看 accuracy 和 margin，要**同时看 chosen/rejected 两条 logprob 曲线**。

```python
def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp,
             beta=0.1, length_norm=None, sft_coef=0.0, chosen_tokens=None):
    """所有 logp 均为整条回答的 token logprob 之和（[B]）。"""
    pi_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp
    logits = beta * (pi_logratio - ref_logratio)         # = r̂_w − r̂_l
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    if sft_coef > 0.0:                                    # 抑制似然位移
        # chosen_tokens：chosen 的 token 数，用于归一化
        assert chosen_tokens is not None
        loss = loss - sft_coef * (policy_chosen_logp / chosen_tokens).mean()

    # 监控项：这两条曲线比 loss 更能说明训练是否健康
    metrics = {
        "implicit_reward_margin": logits.mean().item(),
        "chosen_logp": policy_chosen_logp.mean().item(),
        "rejected_logp": policy_rejected_logp.mean().item(),
        "acc": (logits > 0).float().mean().item(),
    }
    return loss, metrics
```

工程上还有一个省钱要点：$\pi_{\text{ref}}$ 的 logprob 可以**离线预计算一次**存下来，训练时不必常驻第二份模型权重。


## 5. DPO 与 PPO 的本质差别：不是“有没有 RM”

常见的错误答案是“DPO 没有 RM，PPO 有”。更准确的答案是**数据覆盖**：

- PPO 在**当前策略采样出的样本**上获得反馈，因此能纠正模型此刻正在犯的错。
- DPO 只在**固定偏好数据**上优化。数据没覆盖的行为，它既看不见也纠正不了；而训练过程中策略会逐渐偏离数据分布，反馈却不会更新。

这就是离线 RL 的经典困境。三个可观察的后果：

1. DPO 训得越久，策略与偏好数据的分布差距越大，隐式奖励的可信度越低 → 需要早停。
2. 偏好数据里的系统性偏差（长度、风格、某类话题的标注习惯）会被直接吸收。
3. 在可验证推理任务上，DPO 无法像 RLVR 那样通过采样发现新解法。

**在线/迭代 DPO** 就是为了打这个补丁：每一轮用当前模型采样若干回答，用 RM 或验证器（verifier）排出 $(y_w,y_l)$，再跑一轮 DPO。它牺牲了“完全离线”的简洁，换回了部分在线性——Llama 3 的多轮“拒绝采样（rejection sampling） → SFT → DPO”循环正是这种形态。


## 6. 家族成员：各修一处


### 6.1 IPO（2023.10）

**问题**：当偏好几乎确定（$P\to1$）时，BT 的 logit 会被推向无穷，DPO 倾向于把差值拉到尽可能大，导致过拟合，$\beta$ 的约束形同虚设。

**做法**：把偏好学习改写成回归到固定间隔：

$$
\mathcal{L}_{\text{IPO}}=\mathbb{E}\Big[\Big(h_\theta(x,y_w,y_l)-\frac{1}{2\tau}\Big)^2\Big],\qquad h_\theta=\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
$$

不再要求差值无限大，只要求它到达某个目标值。

> **译名订正**：IPO 的 I 来自把通用目标里的映射 $\Psi$ 取成**恒等映射**（identity mapping），原稿译成“身份偏好优化（preference optimization）”是错的。


### 6.2 KTO（2024.2）

**问题**：成对偏好数据昂贵，实际生产中更容易拿到的是单边信号（用户点赞/点踩、工单是否解决）。

**做法**：借前景理论，把好样本与坏样本非对称加权：

$$
\mathcal{L}_{\text{KTO}}=\mathbb{E}\big[\lambda_D\big(1-v(x,y)\big)\big],\qquad
v(x,y)=\begin{cases}\sigma\big(\beta(\hat r_\theta(x,y)-z_{\text{ref}})\big) & y\ \text{desirable}\\[2pt] \sigma\big(\beta(z_{\text{ref}}-\hat r_\theta(x,y))\big) & y\ \text{undesirable}\end{cases}
$$

**$z_{\text{ref}}$ 是 KTO 的核心，原稿完全漏掉了它**。它是当前批次内估计的 KL 参考点（用不匹配的 prompt-response 配对估计 $\mathrm{KL}(\pi_\theta\Vert\pi_{\text{ref}})$），充当“好坏”的分界基准，且通常不回传梯度。没有它，损失就失去了参照系。


### 6.3 ORPO（2024.3）

**问题**：SFT 和偏好对齐要跑两段，还要常驻一个 $\pi_{\text{ref}}$。

**做法**：合成一步，用赔率比（odds ratio）：

$$
\mathcal{L}_{\text{ORPO}}=\mathcal{L}_{\text{SFT}}+\lambda_{\text{OR}}\,\mathcal{L}_{\text{OR}},\qquad
\mathcal{L}_{\text{OR}}=-\log\sigma\Big(\log\frac{\mathrm{odds}_\theta(y_w\mid x)}{\mathrm{odds}_\theta(y_l\mid x)}\Big),\qquad \mathrm{odds}_\theta(y\mid x)=\frac{\pi_\theta(y\mid x)}{1-\pi_\theta(y\mid x)}
$$

不需要 reference，显存省一份；代价是对 $\lambda_{\text{OR}}$ 敏感。


### 6.4 SimPO（2024.5）

**问题**：DPO 的隐式奖励与生成时用的打分（平均对数概率）不一致，且长回答天然占优。

**做法**：直接用**长度归一化的平均对数概率**当隐式奖励，并加一个 margin：

$$
\hat r_{\text{SimPO}}(x,y)=\frac{\beta}{|y|}\log\pi_\theta(y\mid x),\qquad
\mathcal{L}_{\text{SimPO}}=-\mathbb{E}\big[\log\sigma\big(\hat r_w-\hat r_l-\delta\big)\big]
$$

不需要 reference，长度偏差（length bias）被压制；代价是失去锚点，策略可以自由漂移，$\delta$ 也要调。


### 6.5 CPO（2024.1）

把 $\pi_{\text{ref}}$ 近似成均匀分布（等价于把它从损失里去掉），再加一个 SFT 项保底。省一次前向，代价是近似有偏。


### 6.6 对照表

| 方法 | 需要 reference | 需要成对偏好 | 是否含 SFT 项 | 修了什么 |
|---|---|---|---|---|
| DPO | 是 | 是 | 否 | —— |
| IPO | 是 | 是 | 否 | 确定性偏好下的过拟合 |
| KTO | 是（经 $z_{\text{ref}}$） | **否** | 否 | 只有单边标签的场景 |
| ORPO | **否** | 是 | **是** | 两段流程合一 |
| SimPO | **否** | 是 | 否 | 长度偏差 + reference 显存 |
| CPO | 近似去掉 | 是 | **是** | reference 前向成本 |


## 7. 工程坑

| 坑 | 症状 | 处理 |
|---|---|---|
| $\beta$ 过小 | 偏离 reference 太远，语言退化 | 提高 $\beta$；监控隐式奖励绝对值 |
| $\beta$ 过大 | 几乎学不动，margin 长期接近 0 | 降低 $\beta$ |
| reference 选错 | 隐式奖励整体失真 | 必须用**产出这批数据的那个 SFT 检查点** |
| 数据噪声 | DPO 直接把噪声学成偏好 | 标注一致性检查；过滤 margin 极小的对 |
| 长度偏差 | 回答持续变长 | SimPO 式长度归一化；成对长度匹配 |
| 过拟合 | 验证集 margin 还在涨，真实能力在降 | 早停；用能力基准而非偏好胜率做早停信号 |
| 只看胜率 | 偏好胜率涨、通用能力降 | 胜率 + 能力 + 安全三线同时看（第 14 讲） |
| 似然位移 | chosen logprob 大幅下降 | 加 SFT 正则；见 §4 |


## 8. 什么时候用哪个

| 条件 | 选择 |
|---|---|
| 有高质量成对偏好，算力有限，无 rollout 基础设施 | DPO / IPO |
| 只有单边好坏标签（点赞、工单结果） | KTO |
| 想省 reference 显存 | ORPO / SimPO |
| 回答长度分布严重不均 | SimPO 或长度归一化 DPO |
| 任务可验证（数学、代码） | GRPO / RLVR（第 9 讲） |
| 有可靠 RM + 工程能力，需要在线探索 | PPO（第 7 讲） |
| 想要离线的简洁 + 部分在线性 | 迭代 DPO：采样 → 排序 → DPO，循环多轮 |

实践中它们常常是**阶段关系**而非互斥：`SFT → 拒绝采样 → DPO → RLVR/GRPO` 是 2024–2025 年很常见的一条链。


## 9. 案例

> **案例｜Hugging Face [Zephyr-7B-$\beta$](https://arxiv.org/abs/2310.16944)（2023 年 10 月）**
> 第一个被广泛复现的 DPO 配方：在 Mistral-7B 上先做蒸馏式 SFT，再用 UltraFeedback 偏好数据做 DPO（团队称之为 dDPO，因为偏好标签也来自更强模型而非人类）。它的意义是把“对齐”的门槛降到了几百美元级别，DPO 从此成为开源社区的默认选项。

> **案例｜Meta Llama 3（2024 年 7 月）**
> 工业级流水线里 DPO 的样子：多轮迭代，每轮先用当前模型做拒绝采样生成 SFT 数据，再做 DPO。两个细节值得抄——在 DPO 损失里**屏蔽格式类特殊 token**（否则模型会把梯度浪费在模板上，甚至学到用格式区分好坏），以及**加一个 NLL 正则项**抑制似然位移。

> **案例｜AI2 [Tülu 3](https://arxiv.org/abs/2411.15124)（2024 年 11 月）**
> 把 DPO 放进了 `SFT → DPO → RLVR` 三段式，并使用长度归一化的 DPO 变体。它的价值在于**全流程开源**：数据、代码、评测、超参都可复现，是做后训练（post-training）研究最实用的基线。


## 10. 自测题

**1. 为什么 DPO 不需要训练奖励模型？**

因为 KL 正则化目标的最优解可以反解出奖励与策略的关系 $r=\beta\log(\pi^\star/\pi_{\text{ref}})+\beta\log Z(x)$，把它代入 BT 模型后，只依赖 $x$ 的配分项在同一 prompt 的两个回答相减时抵消。于是偏好概率可以直接用策略表示，RM 被折叠进了策略本身。

**2. $\beta$ 在 DPO 里控制什么？和 RLHF 的 KL 系数是什么关系？**

数学角色相同：都控制策略偏离参考模型的强度。DPO 里它是隐式奖励的温度——$\beta$ 越大，同样的对数比差值对应的奖励差越大，模型越不需要大幅改变概率就能满足偏好，因此更新更保守。

**3. chosen 的 logprob 在训练中下降，说明训练坏了吗？**

不一定。损失只要求差值变大，所以 chosen 略降、rejected 大降是常见且可接受的。危险信号是 chosen 大幅下降——此时概率质量流向了数据里没有的第三类回答，没有任何机制保证它们更好。要同时监控两条 logprob 曲线，必要时加 SFT 正则。

**4. DPO 和 PPO 最本质的差别是什么？**

不是“有没有 RM”，而是数据覆盖：PPO 在当前策略采样的样本上获得反馈，能纠正模型此刻的错误；DPO 只在固定数据上优化，策略漂移后反馈不会跟着更新。这是离线 RL 的经典困境，也是迭代 DPO 存在的理由。

**5. KTO 的参考点 $z_{\text{ref}}$ 是什么？为什么不能省？**

它是批次内估计的 KL 参考点，充当“好/坏”的分界基准。KTO 的价值函数（value function）是关于 $\hat r_\theta-z_{\text{ref}}$ 的非对称函数，没有这个基准就没有参照系，单边标签无法转化为有意义的梯度方向。

**6. ORPO 和 SimPO 的共同点与差别？**

共同点是都不需要 reference 模型，省一份显存。差别：ORPO 把 SFT 损失和赔率比偏好项相加，一步完成微调与对齐；SimPO 用长度归一化的平均对数概率当隐式奖励并加 margin，重点是压制长度偏差。

**7. 你只有 3000 条偏好对，DPO 训了 3 个 epoch，验证 margin 一直在涨但人工评估变差。怎么办？**

典型过拟合加似然位移。先把早停信号从 margin 换成能力/安全基准；检查 chosen logprob 是否大幅下降，若是则加 SFT 正则；提高 $\beta$ 收紧约束；数据层面过滤 margin 极小或标注不一致的对。3000 条规模下 1 个 epoch 通常就够。


## 11. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Full derivation — derive DPO on the whiteboard.**
*Anchored to: Rafailov et al., NeurIPS 2023 best paper. Loops: post-training research scientist (OpenAI, Anthropic, Google DeepMind, Meta GenAI). This is probably the single most-asked derivation in post-training interviews.*

*Answer key*: (1) closed-form optimum of the KL-regularized objective, `π*(y|x) ∝ π_ref(y|x)·exp(r/β)`; (2) invert to get `r = β log(π*/π_ref) + β log Z(x)`; (3) substitute into Bradley–Terry — `β log Z(x)` depends only on `x` and cancels in the difference; (4) the result is logistic regression on the implicit reward margin. Be ready for "why can't we just compute Z(x)?" — it sums over the whole response space.

**2. Diagnosis — the chosen log-probability is going down during DPO training. Is that a bug?**
*Anchored to: likelihood-displacement analyses (2024). Loops: post-training engineer (Meta, Cohere, Mistral, Hugging Face-style OSS roles).*

*Answer key*: no — the loss only constrains the margin, so both log-probs commonly fall. It becomes a problem when the chosen log-prob falls sharply, because the freed probability mass goes to responses absent from the data with no guarantee they are better; this is worst when chosen and rejected are near-duplicates. Fixes: add an NLL term on the chosen response, mask formatting tokens, monitor both log-prob curves rather than just accuracy and margin.

**3. Conceptual — what does DPO give up relative to PPO? Answer in terms of data coverage, not model count.**
*Loops: research scientist (Anthropic, OpenAI); a screen for whether you understand offline RL.*

*Answer key*: PPO gets feedback on samples drawn from the current policy, so it corrects the errors the model is making *now*; DPO optimizes a frozen dataset while the policy drifts away from it. Consequences: early stopping matters, dataset biases are absorbed wholesale, and no new solutions can be discovered by exploration. Iterative/online DPO recovers part of this by regenerating and re-ranking each round.

**4. Design — you have thumbs-up/thumbs-down telemetry, not pairwise preferences. What do you use?**
*Anchored to: KTO (2024). Loops: applied AI / product ML (OpenAI applied, Scale AI, enterprise ML teams).*

*Answer key*: KTO, because it consumes single-sided labels via an asymmetric value function around a batch-estimated KL reference point. Then raise the real concern: thumbs-up is a proxy for immediate user satisfaction and directly incentivizes sycophancy — cite the GPT-4o rollback (Lecture 7) and insist on pairing it with a behavioral eval.

**5. Coding — implement the DPO loss, including reference log-probs and an optional length normalization.**
*Loops: RL/post-training engineer (Hugging Face TRL-adjacent, NVIDIA NeMo, startups).*

*Answer key*: sequence-level sums of token log-probs, `logits = β((π_c − π_r) − (ref_c − ref_r))`, `loss = −logsigmoid(logits)`. Mention that reference log-probs can be precomputed offline to avoid holding a second model in memory, and that length normalization turns this into SimPO's reward. Follow-up to expect: which tokens go into the sum — the response only, with formatting tokens optionally masked.

**6. Comparison — when would you pick DPO over GRPO, and vice versa?**
*Loops: applied post-training (most companies with a fine-tuning product).*

*Answer key*: DPO when you have pairwise preferences, limited compute, no rollout infrastructure, and a subjective task. GRPO when the task is verifiable, you can afford sampling `G` responses per prompt, and you need the policy to discover solutions rather than imitate ranked ones. In practice they compose: SFT → DPO for general alignment, then RLVR for reasoning.

**7. Judgment — a candidate architecture proposes running DPO for 5 epochs on 2k preference pairs. What do you say?**
*Loops: senior/staff review-style question (frontier labs, well-funded startups).*

*Answer key*: too many epochs for that data size — expect overfitting and likelihood displacement. Ask for the early-stopping signal (should be capability/safety evals, not preference win-rate), the $\beta$ value, whether the reference is the exact SFT checkpoint that produced the data, and the length distributions of chosen vs rejected. Recommend one epoch, a held-out eval built beforehand, and iterative rounds instead of more epochs.


## 12. 本讲相对原稿的修订

| 位置 | 原稿 | 本讲 |
|---|---|---|
| 推导 | 给了主要步骤 | 补最优解的来源、$Z(x)$ 为什么算不出又为什么不用算 |
| 似然位移 | 未提 | 新增一节：现象、三种差值变大的方式、缓解手段、监控指标 |
| DPO vs PPO | “离线 vs 在线、有无 RM” | 改为数据覆盖视角，并补迭代/在线 DPO |
| IPO | 译作“身份偏好优化”，公式用隐式奖励 | 订正译名（恒等映射），给出回归形式与 $\tau$ |
| KTO | 缺少参考点 $z_{\text{ref}}$ | 补齐，并说明它为什么是核心 |
| SimPO | 公式缺 $\beta$ | 补齐，margin 符号统一为 $\delta$ |
| 符号 | ORPO 权重用 $\gamma$，与折扣因子（discount factor）冲突 | 改为 $\lambda_{\text{OR}}$（附录 A 保留字母） |
| 代码 | 无 | 补 DPO loss 实现，含 SFT 正则与健康度监控项 |
| 案例 | 无 | 补 Zephyr 2023.10、Llama 3 2024.7、Tülu 3 2024.11 |
| 讨论题 | 10 题带简答 | 7 题直接给答案 + 7 道英文面试题（含公司标注） |


## 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。2026 年的条目超出我的训练数据，来自本次检索。

- [Rafailov et al., 2023 — Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Azar et al., 2023 — A General Theoretical Paradigm（IPO）](https://arxiv.org/abs/2310.12036)
- [Ethayarajh et al., 2024 — KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- [Hong et al., 2024 — ORPO](https://arxiv.org/abs/2403.07691)
- [Meng et al., 2024 — SimPO](https://arxiv.org/abs/2405.14734)
- [Xu et al., 2024 — CPO](https://arxiv.org/abs/2401.08417)
- [Pal et al., 2024 — Smaug / DPO-Positive（似然位移的早期修补）](https://arxiv.org/abs/2402.13228)
- [Razin et al., 2024 — Unintentional Unalignment: Likelihood Displacement in DPO](https://arxiv.org/abs/2410.08847)
- [Tunstall et al., 2023 — Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944)
- [Dubey et al., 2024 — Llama 3（迭代 DPO、屏蔽格式 token、NLL 正则）](https://arxiv.org/abs/2407.21783)
- [Lambert et al., 2024 — Tülu 3（长度归一化 DPO + RLVR）](https://arxiv.org/abs/2411.15124)

---

**下一讲**：第 9 讲　RLVR 与 GRPO 家族（已完成）。之后进入第 10 讲　推理模型。
