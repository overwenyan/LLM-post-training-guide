# 第 5 讲　SFT、拒绝采样与蒸馏：三种 KL 方向

> **本讲在主线中的位置**：轴一的第二格（信号来自示范），轴二覆盖了从离线到在线的整段。这一讲把三种“模仿”放在一起讲——它们的区别只是 KL 的方向和样本的来源。
>
> **学完你应该能做到**
> 1. 说清 SFT 在优化什么，以及它的五条上限分别卡在哪里。
> 2. 把 chat template、loss mask、packing 这三个工程命门的常见错误和症状对上号。
> 3. 区分硬标签蒸馏、软标签蒸馏、on-policy 蒸馏三者的 KL 方向与后果。
> 4. 解释为什么 2026 年的旗舰模型用 on-policy 蒸馏来**合并**多个 RL 专家。

**新增符号**：$\mathcal{M}$（loss mask 位置集合）、$m_t$、$\mathrm{chat}(\cdot)$、$\pi_{\mathcal{T}}$（教师）、$\mathcal{D}_{\text{sft}}$、$\mathrm{ver}(x,y)$。见附录 A。

---


## 1. SFT 到底在优化什么

**定义 5.1（监督微调）**　给定示范数据 $\mathcal{D}_{\text{sft}}=\{(x^{(i)},y^{(i)})\}$，

$$
\mathcal{L}_{\text{SFT}}(\theta)=-\mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{sft}}}\Big[\sum_{t\in\mathcal{M}}\log\pi_\theta(y_t\mid x,y_{<t})\Big]
$$

**等价视角**：这等于最小化 $\mathbb{E}_x\,\mathrm{KL}\big(p_{\text{data}}(\cdot\mid x)\,\Vert\,\pi_\theta(\cdot\mid x)\big)$ 加一个常数。注意 KL 的方向——**数据在前，模型在后，是前向 KL**。前向 KL 的性质是 *mode covering*：凡是数据里出现过的模式，模型都必须给非零概率。

三个直接后果：

1. 数据里的错误、噪声、风格瑕疵，模型会照单全收，而且**无法通过多训来消除**。
2. 数据里没有的解法，SFT 学不出来——它没有采样、没有探索、没有比较。
3. 把两个矛盾的好答案都放进数据，模型会学成两者的混合，而不是择优。

用附录 C 的统一视角看，SFT 就是权重恒为 1 的策略梯度（policy gradient）：**每一条数据都被同等地推高，与它好不好无关**。后面所有方法，本质上都是在给这个权重加上判断力。


## 2. 数据：上限主要在这里


### 2.1 来源与特点

| 来源 | 优点 | 风险 |
|---|---|---|
| 人工标注 | 质量最高 | 贵、规模有限 |
| 强模型蒸馏 | 便宜、规模大 | 教师偏差；许可与合规问题 |
| [Self-Instruct](https://arxiv.org/abs/2212.10560) 式自举 | 几乎零成本扩规模 | 多样性塌缩、事实错误累积 |
| 拒绝采样（见 §5） | 分布来自模型自身，缓解暴露偏差（exposure bias） | 依赖验证器（verifier）；多样性下降 |
| 多轮对话 | 训练真实交互形态 | 角色边界与 mask 容易出错 |
| 工具 / agent 轨迹（trajectory） | 教会格式与错误恢复 | observation token 必须 mask 掉 |
| 安全与拒答数据 | 建立边界 | 配比过高 → 过度拒答（over-refusal） |

> **案例｜Stanford [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)（2023 年 3 月）与 Meta [LIMA](https://arxiv.org/abs/2305.11206)（2023 年 5 月）**
> Alpaca 用 52K 条由 GPT 生成的指令数据，把 LLaMA-7B 调成了像样的对话模型，成本仅数百美元——它证明了“指令跟随”这件事非常便宜，也开启了蒸馏数据的时代。两个月后 LIMA 用 **1000 条**精选数据达到了有竞争力的对齐效果，论点更进一步：**对齐主要是在激活预训练（pre-training）已有的能力，而不是灌输新知识**。这个论点和第 10 讲的“激发 vs 扩展”是同一条线。


### 2.2 质量、数量与配比

一句可操作的原则：**SFT 数据不是越多越好，而是越接近你想要的目标行为分布越好。**

但要注意 LIMA 式结论的适用边界。推理模型的冷启动（cold start）阶段需要的是另一种量级：[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 的冷启动只用了数千条长 CoT（因为它只需要解决格式与可读性），但第三阶段的拒绝采样 SFT 用了约 60 万条推理数据加约 20 万条通用数据（因为它要把 RL 得到的能力固化回一个通用模型）。**"多少条够"取决于这一步在流水线里承担什么职责**：教格式用千条，固化能力用十万条量级。

典型配比维度：通用对话、数学、代码、推理、多语言、安全、工具调用、长上下文。常见失衡症状：

- 安全数据过多 → 过度拒答，无害问题也不答。
- 代码占比过高 → 自然语言风格退化，回答变得干硬、爱列点。
- 只训单一领域 → 通用能力明显下降（§7）。


## 3. Chat template 与 loss mask

**定义 5.2（chat template）**　把角色与内容序列化成 token 序列的函数：$x=\mathrm{chat}(\{(\text{role}_1,c_1),\dots\})$。

**铁律：训练与推理必须使用完全一致的 template。** 不一致的典型症状：模型不知道谁在说话、自问自答、把特殊 token 当普通文本输出、BOS/EOS 位置错乱。

**定义 5.3（loss masking）**　只对 $\mathcal{M}$ 内的 token 计损失，通常是 assistant 生成的内容。

| 常见错误 | 症状 |
|---|---|
| 把 user token 计入 loss | 模型学会替用户提问，回答完继续编下一轮 |
| 把 padding 计入 loss | 模型学会输出 padding token |
| EOS 未计入 loss | **模型不会停**，一直写到 max_len |
| 工具返回的 observation 计入 loss | 模型开始编造工具返回值（agent 场景高发，第 11 讲） |
| mask 与自回归偏移错位一位 | 训练 loss 看着正常，生成质量莫名其妙地差 |

最后一条最难查：预测第 $t$ 个 token 用的是前 $t-1$ 个，`labels` 必须相对 `input_ids` 左移一位。写完 SFT 的 dataloader 之后，**把一条样本的 (token, label, mask) 三元组打印出来逐行对齐看一遍**，这是最省时间的做法。

```python
# 一个最小的、带 mask 的 SFT loss
def sft_loss(logits, labels, mask):
    # logits: [B, L, V]；labels: [B, L]（已左移对齐）；mask: [B, L]，1 表示计损失
    logp = torch.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)   # [B, L]
    return -(token_logp * mask).sum() / mask.sum().clamp(min=1)
```


## 4. Packing：吞吐与正确性的权衡

**定义 5.4（packing）**　把多条短样本拼成一条长序列，减少 padding 浪费。

三个必须处理的细节：

1. **块对角（block-diagonal）注意力**：样本 $i$ 的 token 不能注意到样本 $j$。$A_{ij}=1$ 当且仅当 $\mathrm{doc}(i)=\mathrm{doc}(j)$ 且 $j\le i$。漏掉这条会造成跨样本信息泄露。
2. **position id 重置**：每个文档从 0 重新开始（或全局连续，但必须与推理侧一致）。
3. **mask 与边界对齐**：每个 token 属于哪个文档、哪个角色必须记录清楚；文档间要加 EOS；拼接后的截断不能把回答截断在中间。


## 5. 拒绝采样：把 RL 的“筛选”借过来

**定义 5.5（拒绝采样微调 RFT）**

1. 采样：$\{y^{(1)},\dots,y^{(k)}\}\sim\pi_{\theta_{\text{old}}}(\cdot\mid x)$
2. 打分：$\mathrm{ver}(x,y)$ 或 $r_\phi(x,y)$
3. 筛选：$\mathcal{D}'=\{(x,y):\mathrm{ver}(x,y)=1\}$
4. 回炉 SFT：$\theta\leftarrow\arg\min\mathcal{L}_{\text{SFT}}(\theta;\mathcal{D}\cup\mathcal{D}')$

**为什么有效**：训练样本来自模型自己的分布，缓解了暴露偏差；同时借助外部信号做了择优。用统一视角看，它是权重为 $\mathbb{I}[\text{通过}]$ 的策略梯度——**只有正样本、没有负向梯度的 RL**。

**代表方法**：[STaR](https://arxiv.org/abs/2203.14465)（模型生成推理链，答对则保留，答错则给出正确答案让它反推理由，再筛选回炉）、RAFT（奖励排序后取 top-k）、Best-of-N 蒸馏。

**局限**：依赖验证器质量；只用正样本意味着模型学不到“不要这样答”；多轮迭代后多样性单调下降；采样成本随 $k$ 线性增长。

> **案例｜Meta [Llama 2-Chat](https://arxiv.org/abs/2307.09288)（2023.7）与 [Llama 3](https://arxiv.org/abs/2407.21783)（2024.7）**
> 两代都把拒绝采样放在 RL/DPO 之前：先用当前模型采样、用 RM 选出最好的一条、回炉做 SFT，再做偏好优化（preference optimization）。Llama 3 更是把“拒绝采样 → SFT → DPO”循环了多轮。这说明在工业流水线里，**拒绝采样不是 RL 的替代品，而是 RL 的前置增效步骤**——它把策略推到一个更好的起点，后续 RL 的每一次 rollout 就更值钱。


## 6. 蒸馏：三种 KL 方向

这是本讲的核心，也是原稿完全没有覆盖的部分。

| 方式 | 目标 | 样本来自 | KL 方向 | 性质 |
|---|---|---|---|---|
| 硬标签（序列级） | $-\sum_t\log\pi_\theta(y_t)$ | 教师 | 前向（隐式） | 黑盒可用，最简单 |
| 软标签 | $\sum_t\mathrm{KL}(\pi_{\mathcal{T}}\Vert\pi_\theta)$ | 教师 | 前向 | mode covering，需要 logits |
| **On-policy 蒸馏** | $\sum_t\mathrm{KL}(\pi_\theta\Vert\pi_{\mathcal{T}})$ | **学生自己** | **反向** | mode seeking，逐 token 纠错 |


### 6.1 为什么 on-policy 是关键区别

离线蒸馏的问题是**分布不匹配**：教师的轨迹往往是学生自己走不到的路径。学生在训练时学的是“教师会怎么走”，推理时却走在自己的路径上，一旦偏离就没有任何指导——这就是暴露偏差在蒸馏里的版本。

On-policy 蒸馏让学生先自己生成，再让教师对**学生实际写出的每一个 token** 打分。写成策略梯度形式，逐 token 的权重就是

$$
w_t=\log\pi_{\mathcal{T}}(y_t\mid s_t)-\log\pi_\theta(y_t\mid s_t)
$$

教师比学生更看好的 token 被推高，学生过度自信的 token 被压低。

```python
def on_policy_distill_loss(student_logits, teacher_logits, labels, mask,
                           full_vocab=True):
    """反向 KL：学生在自己的 rollout 上贴近教师。"""
    s_logp = torch.log_softmax(student_logits.float(), dim=-1)
    t_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
    if full_vocab:
        # 全词表反向 KL：sum_v p_s(v) [log p_s(v) - log p_t(v)]
        p_s = s_logp.exp()
        kl = (p_s * (s_logp - t_logp)).sum(-1)                   # [B, L]
    else:
        # 采样 token 上的单点估计，便宜但方差大
        s = s_logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        t = t_logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        kl = s - t
    return (kl * mask).sum() / mask.sum().clamp(min=1)
```

前提是师生**同词表、同 tokenizer**；全词表版本的通信与显存成本高得多，但梯度稳定得多（教师之间意见不一致时尤其明显）。


### 6.2 三个真实用法

**用法一：大教师 → 小学生（经典压缩）**
DeepSeek-R1-Distill（2025.1）把 R1 的推理轨迹用**硬标签**蒸进 Qwen/Llama 小模型，效果好过在小模型上直接跑 RL。[Qwen3](https://arxiv.org/abs/2505.09388)（2025.4）用的是 logits 对齐版本，报告称这种蒸馏只要 RL 约十分之一的 GPU 小时，效果还更好。

**用法二：多个 RL 专家 → 一个统一模型（2026 年的主流）**
[DeepSeek-V4](https://arxiv.org/abs/2606.19348)（2026.4）先按领域各训一个专家（SFT → GRPO），再用 on-policy 蒸馏把它们合并成一个学生，学生在自己的 rollout 上优化对教师的反向 KL。小米 [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780) 把这种多教师形式命名为 [MOPD](https://arxiv.org/abs/2606.30406)，NVIDIA [Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007) 用十余个专家教师做同样的事。

这里有个反直觉的点：**这些教师通常不比学生大**，它们是同一基座、同样大小、只在单一领域被 RL 推得更远的检查点。**让它们成为好教师的是专精，不是规模。**

**用法三：自己教自己**
[GLM-5](https://arxiv.org/abs/2602.15763)（2026.2）在多个 RL 阶段之后用一次蒸馏，把前序阶段被侵蚀的能力从更早的检查点恢复回来。Thinking Machines 的做法类似：在新领域微调之后，从**微调前的检查点**蒸馏，既保住新知识又找回被抹掉的旧行为——这是持续学习（continual learning）的一条实用路径（第 12 讲）。


### 6.3 为什么会形成这个趋势

回到附录 C 的信息密度论证：

| 反馈 | 一条轨迹带回的监督量 |
|---|---|
| 结果奖励（outcome reward） RL | 1 个标量 |
| On-policy 蒸馏 | $\lvert y\rvert\times\lvert\mathcal{V}\rvert$ 个数 |

RL 的不可替代之处是**探索**——它能发现教师和数据里都没有的解法。蒸馏不能。所以现在的分工是：**用 RL 去探索并造出专家，用蒸馏去合并与传递**。说“蒸馏取代了 RL”是不准确的；准确的说法是 RL 换了位置。

**共同局限**：学生的上限被教师锁死；反向 KL 是 mode seeking，多样性会收缩（pass@k 可能下降）；教师之间冲突时需要加权，权重本身是超参。


## 7. 灾难性遗忘

**定义 5.6**　微调后目标能力上升，但通用/旧能力下降：$\mathcal{L}_{\text{gen}}(\theta_{\text{SFT}})>\mathcal{L}_{\text{gen}}(\theta_0)$。

成因：数据分布单一、学习率过大、epoch 过多、全参更新、格式过拟合。

缓解手段（细节见第 12 讲）：

| 手段 | 形式 | 备注 |
|---|---|---|
| 数据回放 | $\mathcal{L}=\mathcal{L}_{\text{SFT}}+\lambda\mathcal{L}_{\text{replay}}$ | 最有效也最简单 |
| KL 正则 | $+\beta\,\mathrm{KL}(\pi_\theta\Vert\pi_{\text{ref}})$ | 与 RL 阶段同一个工具 |
| LoRA | $W=W_0+BA$ | “学得少也忘得少” |
| 低学习率 / 早停 / 少 epoch | —— | SFT 通常 1–3 个 epoch 就够 |
| 从旧检查点蒸馏 | §6.2 用法三 | 2026 年的新做法 |


## 8. SFT 的五条上限

1. **只能模仿，不能超越示范**。目标是逼近数据分布，数据里没有的推理方式学不到。
2. **偏好不可表达**。把 $y_w$ 和 $y_l$ 都放进数据只会让模型同时提高两者的似然。要表达“谁更好”，需要成对目标 → 第 7、8 讲。
3. **暴露偏差**。训练看真前缀，推理看自己的前缀，一旦前面错了后面会越错越远 → 拒绝采样与 RL 缓解。
4. **不能直接优化不可导奖励**。答案正确性、代码通过率、安全性都是黑盒判定 → 第 9 讲。
5. **长 CoT 难以靠纯 SFT 扩展**。搜索、回溯、自我验证这些行为需要试错才能学到分寸。


### 关于“SFT 记忆、RL 泛化”

2025 年 1 月有一篇影响很大的工作，结论是在规则类与视觉导航任务上，SFT 倾向于记住训练分布，RL 的分布外泛化更好。这个结论常被简化成“SFT 没用”，但要注意三点反证：

- 同一篇工作也指出 **SFT 是 RL 的必要初始化**——不先 SFT，RL 的起点太差。
- DeepSeek-R1 用 R1-Zero 证明了可以不用 SFT，但正式版仍然加回了冷启动 SFT，因为可读性与格式必须靠示范教。
- 到 2026 年，DeepSeek-V4 的每一个领域专家**依然是从 SFT 开始**再做 GRPO。

**正确的说法**：SFT 决定策略的起点与行为形态，RL 决定它能被推到多远。两者不是竞争关系。


## 9. 工程检查清单

- [ ] 训练与推理 chat template 完全一致（用同一份代码生成）
- [ ] 打印过一条样本的 token / label / mask 三元组，逐行核对偏移
- [ ] EOS 计入 loss；padding、user、system、observation 全部 mask
- [ ] packing 用块对角 attention，position id 已重置，文档间有 EOS
- [ ] 数据去重、去污染（与评测集）
- [ ] 训练前测一遍通用能力基线，训练后复测（遗忘检测）
- [ ] 蒸馏数据的师生 tokenizer 一致；许可条款已确认
- [ ] 保留 base 与 SFT 两个检查点（RL 的 reference 用 SFT，消融用 base）


## 10. 自测题

**1. 为什么说 SFT 优化的是前向 KL？这带来什么后果？**

$\mathcal{L}_{\text{SFT}}$ 等价于 $\mathrm{KL}(p_{\text{data}}\Vert\pi_\theta)$ 加常数，数据分布在前。前向 KL 是 mode covering：数据里的每个模式都要被覆盖，因此噪声也会被学到，且模型无法通过训练更久来"筛掉"坏数据。

**2. 为什么 EOS 必须计入 loss？**

EOS 是模型要学会预测的一个动作。不计损失，模型就学不到"什么时候该停"，推理时会一直写到 max_len。在 RL 阶段这会进一步表现为大量截断样本，污染奖励（第 9 讲）。

**3. 拒绝采样和 RL 的本质区别？**

拒绝采样只保留通过验证的样本、权重为 0/1，没有负向梯度，且每轮要重新采样筛选后回炉 SFT；RL 用可正可负的优势直接更新策略，能利用失败轨迹的信息，且可以在 token 级分配信用。拒绝采样的好处是稳定、实现简单，常作为 RL 的前置步骤。

**4. on-policy 蒸馏和普通蒸馏的核心差别是什么？为什么它能替代混合 RL 阶段？**

样本来源和 KL 方向：学生自己采样，优化反向 KL，教师对学生实际写出的每个 token 给信号，因此没有分布不匹配问题。它能替代混合 RL 阶段，是因为"把多个已有能力合并到一个模型"这件事不需要探索，只需要高密度的模仿信号，而蒸馏的信息密度远高于标量奖励。但它不能替代造专家那一步的 RL。

**5. 多教师蒸馏时，教师为什么不必比学生大？**

教师的价值来自领域专精而非规模。它们通常是同一基座、同样大小、只在某个领域被 RL 推得更远的检查点。学生需要的是"在这个领域里更好的下一 token 分布"，而不是"更大的模型"。

**6. 你的 SFT 模型训完后数学变强但闲聊变差，怎么办？**

先确认是遗忘还是配比问题：用训练前的检查点在通用集上复测，对比下降幅度。处理顺序是——降学习率与 epoch 数、混入通用数据回放、考虑 LoRA 而非全参、必要时从旧检查点做一次蒸馏恢复。同时检查是不是格式过拟合（回答变得过度结构化）。


## 11. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Whiteboard code — masked SFT loss.**
*Loops: post-training engineer (Meta GenAI, NVIDIA, Hugging Face-style OSS roles).*

Implement the loss and explain how `labels` align with `input_ids`. Follow-up: which tokens must be masked in a multi-turn, tool-calling sample?

*Answer key*: shift labels left by one; gather log-probs at label indices; average over the mask. Mask out system, user, padding, and tool-`observation` tokens; keep assistant tokens **and EOS**. The tool-observation case is the one most candidates miss, and it is exactly what makes agent models hallucinate tool outputs.

**2. Comparison — hard-label vs soft-label vs on-policy distillation.**
*Anchored to: DeepSeek-R1-Distill (2025), Qwen3 (2025), DeepSeek-V4 (2026). Loops: research engineer (most frontier labs; increasingly a standard question).*

*Answer key*: hard-label = train on teacher text, black-box, works across tokenizers. Soft-label = match teacher's next-token distribution, forward KL, mode-covering, needs logits. On-policy = student samples, teacher scores every token, reverse KL, mode-seeking, removes the distribution mismatch. Use on-policy when merging multiple specialists or repairing forgetting; use hard-label when the teacher is closed-weight.

**3. Debugging — the model never stops generating.**
*Loops: applied engineer (every company shipping fine-tuned models).*

*Answer key*: three usual causes — EOS excluded from the loss mask; EOS placed wrong in the chat template; packing without EOS at document boundaries. Diagnose by printing the `(token, label, mask)` triple for one sample and by decoding a raw training sequence.

**4. Design — 5k high-quality human examples plus an API to a strong closed model. Build the SFT set.**
*Loops: applied AI / data (Scale AI, Surge, OpenAI solutions, Anthropic applied).*

*Answer key*: use the human data to define the target behavior distribution and to build the eval set, not to bulk up training. Scale with model-generated data but filter hard (verifier, or human spot-check on a random sample). Check the provider's terms on training competing models. Finally, run rejection sampling with your own model so the final distribution sits near your policy, which reduces exposure bias.

**5. Pushback — "on-policy distillation exists now, so RL is obsolete."**
*Anchored to: DeepSeek-V4, MiMo-V2-Flash MOPD, GLM-5 (2026). Loops: research scientist (frontier labs).*

*Answer key*: distillation cannot produce capability the teacher lacks; the student's ceiling is the teacher. Without RL there are no expert teachers. The accurate statement is that RL moved: out of the merge stage, into the specialist-creation stage. Bonus: note that teachers are typically same-size checkpoints specialized by RL, not larger models.

**6. Concept — forward vs reverse KL, and why it shows up in pass@k.**
*Loops: research scientist (Google DeepMind, Anthropic).*

*Answer key*: forward KL (SFT, soft-label distillation) is mode-covering — it keeps mass on everything in the data, noise included. Reverse KL (on-policy distillation, and implicitly KL-regularized RL) is mode-seeking — it concentrates mass, which sharpens pass@1 and can shrink diversity, showing up as flat or falling pass@k at large k.


## 12. 本讲相对原稿的修订

| 位置 | 原稿 | 本讲 |
|---|---|---|
| SFT 的 KL 方向 | 提到"最小化交叉熵/KL"但未区分方向 | 明确前向 KL 与 mode covering 的后果 |
| 蒸馏 | 只有硬标签 / 软标签两种 | 补 on-policy 蒸馏与反向 KL，给出公式与代码 |
| 蒸馏的用途 | 仅"大模型 → 小模型" | 补"合并 RL 专家""从旧检查点恢复"两种 2026 年主流用法 |
| 冷启动数据规模 | 未提 | 给出"教格式用千条、固化能力用十万条量级"的判断依据 |
| SFT vs RL | "SFT 是模仿学习，不是 RL" | 保留，但补上 2025 年之争与三条反证 |
| observation mask | 未提 | 补 agent 轨迹里最常见的 mask 错误 |
| 案例 | 无 | 补 Alpaca、LIMA、Llama 2/3、R1-Distill、Qwen3、DeepSeek-V4、GLM-5 |


## 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。2026 年的条目超出我的训练数据，来自本次检索。

- [Wang et al., 2022 — Self-Instruct](https://arxiv.org/abs/2212.10560)
- [Stanford Alpaca, 2023 — 代码与数据](https://github.com/tatsu-lab/stanford_alpaca)
- [Zhou et al., 2023 — LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [Zelikman et al., 2022 — STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465)
- [Dong et al., 2023 — RAFT: Reward rAnked FineTuning](https://arxiv.org/abs/2304.06767)
- [Dubey et al., 2024 — Llama 3（多轮拒绝采样 + DPO）](https://arxiv.org/abs/2407.21783)
- [DeepSeek-AI, 2025 — DeepSeek-R1（含 R1-Distill）](https://arxiv.org/abs/2501.12948)
- [Chu et al., 2025 — SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2501.17161)
- [Qwen Team, 2025 — Qwen3（强到弱蒸馏）](https://arxiv.org/abs/2505.09388)
- [Thinking Machines, 2025 — On-Policy Distillation（实践者视角最清晰的一篇）](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Xiaomi, 2026 — MiMo-V2-Flash（MOPD）](https://arxiv.org/abs/2601.02780)
- [MOPD, 2026 — Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2606.30406)
- [Z.AI, 2026 — GLM-5（跨阶段蒸馏恢复能力）](https://arxiv.org/abs/2602.15763)
- [DeepSeek-AI, 2026 — DeepSeek-V4（专家 RL + OPD 合并）](https://arxiv.org/abs/2606.19348)

---

**下一讲**：第 6 讲　RL 基础——MDP、策略梯度、baseline、GAE、PPO clip，以及 KL 的三种估计器。
