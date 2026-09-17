# 第 12 讲　Adaptation：把通用模型变成你的模型

> **本讲在主线中的位置**：前面十一讲讲的是**实验室怎么造模型**。这一讲换一个视角——**你拿到一个已经很强的模型，怎么让它适配你的领域、语言、用户、知识和环境**。绝大多数人的实际工作在这一层，而原稿几乎没有覆盖。
>
> **学完你应该能做到**
> 1. 把所有适配手段放进“改哪里 × 适配到什么”这张二维表，并快速定位方案。
> 2. 用一棵决策树回答“该上 RAG、prompt、SFT、LoRA、RFT 还是继续预训练（continued pre-training, CPT）”。
> 3. 写出 LoRA 的公式，解释 $\alpha/r$ 缩放，并说明它在 RL 里为什么格外好用。
> 4. 设计一次适配的评估：目标任务、通用能力、安全三条线同时看。


## 0. 新增符号

$W_0$（原权重）、$B,A$（LoRA 低秩矩阵）、$r$（秩）、$\alpha$（LoRA 缩放）、$\Omega_i$（参数重要性）、$\theta^\ast$（旧任务最优参数）。见附录 A。


## 1. 什么是 adaptation

**定义 12.1**　在不重新预训练（pre-training）的前提下，改变模型在特定分布上的行为或知识，使其满足具体场景需求。

它和后训练（post-training）的关系：后训练是**模型提供方**为所有人做的通用对齐与能力激发；adaptation 是**模型使用方**为自己做的定向改造。两者用的技术高度重叠（SFT、偏好优化（preference optimization）、RL 都会出现），但约束完全不同——你通常没有百万条标注、没有千卡集群，也不能承受把通用能力训坏。


## 2. 二维分类


### 2.1 改哪里

| 层次 | 手段 | 训练成本 | 知识时效 | 改变能力吗 |
|---|---|---|---|---|
| 输入侧 | prompt、few-shot、RAG、上下文工程（context engineering） | 零 | **实时可更新** | 否，只调用已有能力 |
| 参数侧 | 继续预训练、全参微调、LoRA/QLoRA 等 PEFT | 中到高 | 训练时固化 | 是 |
| 表示侧 | activation steering、表示微调 | 低 | 固化 | 局部行为可控 |
| 组合侧 | 模型合并（model merging）、任务向量（task arithmetic）、在线策略蒸馏（on-policy distillation） | 低到中 | 取决于来源 | 合并已有能力 |
| 输出侧 | 约束解码（constrained decoding）、语法/schema 约束、验证器（verifier）重排 | 零 | 实时 | 否，但能保证形式正确 |


### 2.2 适配到什么

领域知识、语言、长上下文、模态、工具与环境、用户与组织偏好、持续到来的新知识。下面 §5 逐个给方案。

**一个常被忽视的判断**：如果你的需求是“模型不知道某些事实”，答案基本是 RAG 而不是微调；如果需求是“模型知道但不会用、格式不对、风格不符、推理不到位”，才轮到训练。**微调教行为，检索给知识**——这句话能挡掉一大半错误立项。


## 3. 决策树

```
需求是什么？
├─ 缺事实/知识时效性强 ──────────────→ RAG（+ 结构化检索、引用）
├─ 只是格式/风格/角色不对 ──────────→ 先试 system prompt 与 few-shot
│                                      └─ 稳定性不够 → 小规模 SFT / LoRA
├─ 输出必须结构合法（JSON、SQL、DSL）→ 约束解码 + schema 校验（训练是次选）
├─ 领域语料极大、术语与分布差异大 ──→ 继续预训练（CPT）+ 回放通用数据
├─ 任务可自动验证、样本少但要推理 ──→ 强化微调（RFT / RLVR，见第 9 讲）
├─ 有偏好对，想让模型更合口味 ──────→ DPO 家族（第 8 讲）
├─ 要同时具备多种已有能力 ──────────→ 模型合并或在线策略蒸馏（第 5 讲）
└─ 知识持续更新、不能忘旧 ──────────→ 持续学习：回放 + 从旧检查点蒸馏
```

成本量级（同一任务的粗略对比）：prompt/RAG 约等于零训练成本；LoRA 微调通常是数十到数百美元级；全参微调高一到两个数量级；继续预训练再高一到两个数量级；RFT 介于 LoRA 与全参之间，但要额外付 rollout 与验证器的钱。


## 4. 参数侧的主力：LoRA


### 4.1 公式

$$
W=W_0+\Delta W=W_0+\frac{\alpha}{r}BA,\qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll\min(d,k)
$$

- $W_0$ 冻结，只训练 $B,A$，可训练参数量从 $dk$ 降到 $r(d+k)$。
- $B$ 初始化为零、$A$ 随机初始化，保证训练开始时 $\Delta W=0$，模型行为与基座完全一致。
- $\alpha/r$ 是缩放因子。**注意一个常见陷阱**：固定 $\alpha$ 去扫不同的 $r$ 时，有效学习率会随 $r$ 增大而变小，于是高秩看起来“学得更慢”，这是缩放导致的假象，不是容量问题。
- 推理时可以把 $\frac{\alpha}{r}BA$ 合并回 $W_0$，零额外延迟；也可以保留为可插拔适配器，一个基座服务多个租户。

**QLoRA**：把 $W_0$ 量化到 4-bit 冻结，只用 BF16 训练适配器，单卡即可微调数十 B 模型。代价是量化带来的少量质量损失与反量化开销。


### 4.2 两条经验规律

**规律一：LoRA 学得少，也忘得少。** [Biderman et al., 2024](https://arxiv.org/abs/2405.09673) 的结论是——在数据量很大、接近预训练规模的场景，LoRA 的容量不够，会明显落后于全参；但它对基座能力的破坏也小得多，是一种自带正则的适配方式。

**规律二：配置正确时，LoRA 在后训练规模上可以追平全参。** Thinking Machines 的 [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)（2025.9）给了三条可直接抄的实践：

1. **加到所有权重矩阵上，尤其是 MLP 与 MoE 层**；只加在注意力上表现更差，即使把秩调高、可训练参数量对齐也一样。
2. **学习率要明显高于全参**（经验上高一个量级左右，具体倍数随模型规模变化）。
3. **RL 场景下，即使很低的秩（低到 1）也能匹配全参**。

第三条的解释非常值得记住，因为它把第 9 讲和这一讲连起来了：

> RL 每条轨迹（trajectory）只回来一个标量奖励，信息量以 bit 计；一整轮训练注入模型的新信息总量极小，所以低秩容量就足够。而 SFT 每个 token 都在注入信息，容量需求高得多。

**推论**：如果你在做 RLVR 或 RLHF 的适配，LoRA 几乎总是值得先试的方案——省显存、省时间、容易回滚、可插拔部署。如果你在灌大量新知识，LoRA 会先撞上容量墙。


### 4.3 其他 PEFT

Adapter（插入小模块）、Prefix/Prompt tuning（学一段虚拟前缀）、DoRA（把更新拆成方向与幅度）等。今天的默认选择仍是 LoRA 及其变体，其余多在特定约束下使用。


## 5. 按“适配到什么”给方案


### 5.1 领域

三条路线，成本递增：

| 路线 | 代表 | 适用 |
|---|---|---|
| RAG + prompt | 大多数企业落地 | 知识密集、变化快、需要引用 |
| 微调（SFT/LoRA） | 行为与格式对齐 | 有领域任务样本，要固化行为 |
| 继续预训练 | [Code Llama](https://arxiv.org/abs/2308.12950)（2023.8，在 Llama 2 上追加约 500B 代码 token） | 领域语料极大、分布差异显著 |

> **案例｜[BloombergGPT](https://arxiv.org/abs/2303.17564)（2023.3）与它的启示**
> 彭博用一半金融语料、一半通用语料从头训了一个 50B 模型。当时这是领域模型的标准做法；但随后两年通用模型能力上升得太快，"为领域从头训练"的性价比迅速下滑，主流转向了"强通用基座 + 检索 + 轻量微调"。
> **教训**：做适配方案选型时，要把**基座的进步速度**算进去——你花六个月训的领域模型，可能比不过六个月后的通用模型加一层 RAG。


### 5.2 语言

优先级通常是：tokenizer 是否高效（影响成本）→ 是否有高质量指令数据 → 是否需要继续预训练补语料。只做 SFT 往往能解决“会说但不肯说”的问题，解决不了“语料里就没有”的问题。


### 5.3 长上下文

回到第 2 讲：调整 RoPE 基频或位置插值（[YaRN](https://arxiv.org/abs/2309.00071) 一类），在少量长文档上继续训练。注意评估要用“大海捞针”之外的多跳（multi-hop）检索任务，否则很容易得到虚高的结论。


### 5.4 模态

[LLaVA](https://arxiv.org/abs/2304.08485)（2023.4）的范式至今通用：冻结视觉编码器与语言模型，先训一个连接器把视觉特征投影到语言空间，再做视觉指令微调（instruction tuning）。**这是 adaptation 的一个纯粹例子——新增能力靠的是一个小模块加少量数据，而不是重训基座。**


### 5.5 工具与环境

见第 11 讲。要点是工具调用格式先用 schema 约束解码保证合法，再考虑用 RL 提升选择与编排能力。


### 5.6 用户与组织偏好

三层递进：system prompt（零成本、可随时改）→ 偏好数据 + DPO（第 8 讲）→ 组织专属 RM/rubric（第 7 讲）。多租户场景优先用可插拔 LoRA，避免为每个客户维护一套全参权重。


### 5.7 新知识与持续学习

核心矛盾是**学新不忘旧**：

$$
\min_\theta\ \mathcal{L}_{\text{new}}(\theta)+\lambda\sum_i\Omega_i\big(\theta_i-\theta_i^\ast\big)^2
$$

这是 EWC 式的正则，按参数重要性 $\Omega_i$ 惩罚偏离旧任务最优解。实践中更常用的是三件事：

1. **数据回放**：混入通用数据与历史任务数据。
2. **LoRA**：更新受限，天然少忘（§4.2 规律一）。
3. **从旧检查点蒸馏**：微调之后，用**微调前的自己**当教师做一次在线策略蒸馏，把被抹掉的行为找回来，同时保留新知识（第 5 讲 §6.2）。这是 2025–2026 年出现的新解法，也是目前最贴近“持续学习（continual learning）”的可用路径。


### 5.8 测试时适配

在推理阶段临时更新（对单个任务做少量梯度步，或用检索到的样本做 in-context 适配）。学术上活跃，工程上少见，因为它把训练的复杂度带进了服务链路。


## 6. 强化微调：把 RL 产品化的适配

OpenAI 在 2024 年 12 月发布的强化微调（Reinforcement Fine-Tuning）把第 9 讲的技术做成了产品形态：用户提供任务样本与一个**评分器**，平台用 RL 训练一个专属模型。

**它适合什么**：任务有明确对错判据、样本量小（几十到几千）、且需要模型**推理**而不只是模仿格式。典型场景是专业领域的分类判定、结构化抽取、规则密集的合规检查。

**它不适合什么**：没有可靠评分器的主观任务；只是缺知识的任务（应该上 RAG）；只是格式问题的任务（SFT 或约束解码更便宜）。

注意这里的 RFT 与第 5 讲的拒绝采样微调（Rejection sampling Fine-Tuning）缩写相同、含义不同（附录 B 有提示）。面试里遇到先反问是哪一个。


## 7. 评估一次适配

三条线必须同时看，缺一条就会出事：

| 线 | 测什么 | 常见失误 |
|---|---|---|
| 目标任务 | 适配想要提升的指标 | 只看这条，于是通用能力悄悄塌了 |
| 通用能力 | 适配前的基线能力集合 | 没有留适配前的对照数字 |
| 安全与合规 | 拒答率、越狱（jailbreak）、泄露、口径 | 微调会削弱基座的安全训练 |

**最容易被忽略的一条**：微调会削弱模型原有的安全对齐。哪怕你的数据完全无害，格式与分布的改变也可能让安全行为退化。所以任何要上线的适配，都应该把安全评测放进回归测试。

另外要留**适配前的检查点与它的评测数字**。没有对照，你无法区分“模型变差了”和“评测变严了”。


## 8. 工程检查清单

- [ ] 先确认需求是缺知识还是缺行为（前者 RAG，后者训练）
- [ ] 评测集在动手前就建好，包含目标、通用、安全三线
- [ ] 保留并测过适配前检查点的基线数字
- [ ] LoRA 加在所有线性层（含 MLP/MoE），不是只加注意力
- [ ] LoRA 学习率按经验显著高于全参；扫秩时注意 $\alpha/r$ 缩放的影响
- [ ] 训练数据混入通用数据回放，防遗忘
- [ ] 结构化输出优先用约束解码，而不是靠训练“教会格式”
- [ ] 多租户场景用可插拔适配器，而非多份全参权重
- [ ] 上线前跑安全回归；有回滚方案


## 9. 自测题

**1. 什么时候用 RAG，什么时候该微调？**

缺事实、知识更新快、需要引用来源时用 RAG；模型知道但不会用、格式不对、风格不符、推理不到位时才训练。一句话是"微调教行为，检索给知识"。两者经常同时需要：用 RAG 供知识，用轻量微调让模型稳定地按你要的方式使用这些知识。

**2. 写出 LoRA 的公式，并解释为什么 $B$ 初始化为零。**

$W=W_0+\frac{\alpha}{r}BA$。$B=0$ 保证训练起点 $\Delta W=0$，模型行为与基座完全一致，不会因为随机初始化的适配器一上来就破坏已有能力；同时 $A$ 随机初始化保证梯度不为零、能够开始学习。

**3. 为什么 RL 场景下很低的 LoRA 秩就够用？**

因为 RL 的信息注入率极低——每条轨迹只回来一个标量奖励，整轮训练注入模型的新信息以 bit 计，低秩容量足以承载。SFT 则每个 token 都在注入信息，容量需求高得多。这也解释了为什么"LoRA 学得少"在 SFT 上是缺点、在 RL 上几乎不构成限制。

**4. 扫 LoRA 秩时为什么高秩看起来学得更慢？**

标准 LoRA 按 $\alpha/r$ 缩放，固定 $\alpha$ 增大 $r$ 会让有效学习率变小。这是缩放导致的假象，不是容量不足。正确做法是随秩调整 $\alpha$ 或直接调学习率，让不同秩在可比的有效学习率下对照。

**5. BloombergGPT 的教训是什么？**

它代表"为领域从头训练"的路线。随后通用基座能力快速上升，这条路线的性价比迅速下滑。教训是选型时必须把基座的进步速度算进去——耗时半年训出的领域模型，可能不如半年后的通用模型加一层检索。

**6. 适配之后模型在目标任务上提升明显，你还要测什么？**

通用能力与安全。微调会削弱基座的安全对齐，即使训练数据完全无害也可能因分布与格式变化而退化。必须有适配前检查点的基线数字做对照，否则无法区分模型变差与评测变严。

**7. 持续学习里"从旧检查点蒸馏"是怎么工作的？**

先在新数据上微调获得新知识，然后把**微调前的自己**当教师，在学生自己的 rollout 上做在线策略蒸馏，把被抹掉的旧行为拉回来。因为蒸馏只在学生实际会走到的状态上纠正，它能在保留新知识的同时恢复旧能力，比单纯的数据回放更精准。


## 10. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Scoping — a customer says "the model doesn't know our internal processes." Fine-tune or retrieve?**
*Loops: forward-deployed / applied AI (OpenAI, Anthropic, Scale AI, Databricks, most enterprise AI teams). This is the single most common applied-AI screening question.*

*Answer key*: almost always retrieval first — missing facts that change over time are a knowledge problem, and fine-tuning bakes them in with no update path and no citations. Fine-tune when the gap is behavioral: format, tone, tool-use patterns, domain reasoning style. Say explicitly that you would build the eval set before choosing, and that the two compose.

**2. Mechanics — write the LoRA update and explain the $\alpha$/r scaling.**
*Anchored to: Hu et al. (2021). Loops: ML engineer / applied research (nearly universal).*

*Answer key*: `W = W_0 + (α/r)·BA`, `B` zero-initialized so training starts at the base model. `α/r` sets the effective update scale; a classic pitfall is sweeping `r` at fixed `α`, which silently lowers the effective learning rate at higher ranks and makes high rank look worse than it is.

**3. Depth — why does LoRA match full fine-tuning for RL even at very low rank?**
*Anchored to: "LoRA Without Regret" (Thinking Machines, Sept 2025). Loops: post-training research (frontier labs, RL-focused startups).*

*Answer key*: RL injects very little information per episode — a scalar reward, on the order of bits — so total information absorbed over a run is small and low-rank capacity suffices. Contrast with SFT, where every token carries signal. Also cite the practical configuration: apply LoRA to all matrices including MLP/MoE rather than attention only, and use a substantially higher learning rate than full fine-tuning.

**4. Design — multi-tenant product, 200 customers each wanting a tuned model. Architecture?**
*Loops: applied ML / platform (Databricks, Together AI, Fireworks, enterprise platform teams).*

*Answer key*: one shared base plus per-tenant LoRA adapters served from a pool, batching across adapters; keep prompts and retrieval per tenant. Full-parameter copies do not scale in memory or in ops. Discuss adapter versioning, rollback, and the isolation/eval story — each tenant needs its own regression suite including safety.

**5. Risk — what breaks when you fine-tune a well-aligned model on harmless domain data?**
*Loops: safety-adjacent applied roles (Anthropic, OpenAI, Google).*

*Answer key*: safety behavior can degrade even with benign data, because distribution and format shift move the model away from the alignment training. Mitigations: mix in general and safety data as replay, prefer LoRA to constrain the update, run a safety regression before shipping, and keep the pre-adaptation checkpoint plus its eval numbers as the control.

**6. Trade-off — RFT (reinforcement fine-tuning) versus SFT for a niche classification task with 500 labeled examples.**
*Anchored to: OpenAI's RFT product (Dec 2024). Loops: applied AI / solutions engineering.*

*Answer key*: RFT fits when there is a reliable grader and the task needs reasoning rather than pattern matching, and it is more sample-efficient at small data sizes. SFT is the cheaper default when the mapping is largely surface-level, and constrained decoding may remove the need entirely if the real issue is output format. Ask what the grader looks like before recommending RFT — no grader, no RFT.

**7. Continual learning — how do you ship monthly model updates without regressing last month's behavior?**
*Anchored to: distillation-from-earlier-checkpoint practice (2025–2026). Loops: staff/senior applied (frontier labs, platform teams).*

*Answer key*: maintain a growing regression suite; replay historical data; prefer parameter-efficient updates; and after fine-tuning, distill from the previous checkpoint on the student's own rollouts to restore eroded behavior while keeping the new knowledge. Note that this is the same mechanism GLM-5 used across RL stages — recovery by distillation rather than by regularization alone.


## 11. 本讲相对原稿的修订

| 位置 | 原稿 | 本讲 |
|---|---|---|
| Adaptation | 仅在 SFT 讲里提了一句 LoRA | 独立成讲 |
| 分类 | 无 | “改哪里 × 适配到什么”二维表 + 决策树 |
| LoRA | 只有 $W=W_0+BA$ | 补 $\alpha/r$ 缩放与扫秩陷阱、全层放置、学习率、RL 低秩结论 |
| 持续学习 | EWC 公式一条 | 补回放、LoRA、从旧检查点蒸馏三条实用路径 |
| 领域适配 | 无 | 补 RAG / 微调 / 继续预训练三条路线与成本对照，含 BloombergGPT 的教训 |
| 评估 | 无 | 补目标、通用、安全三线与“微调会削弱安全对齐”的警告 |
| RFT 歧义 | 无 | 明确两个 RFT 的区别 |


## 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。

- [Hu et al., 2021 — LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Dettmers et al., 2023 — QLoRA](https://arxiv.org/abs/2305.14314)
- [Biderman et al., 2024 — LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)
- [Thinking Machines, 2025 — LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- [TRL 文档 — LoRA Without Regret 的复现指南](https://huggingface.co/docs/trl/main/en/lora_without_regret)
- [Houlsby et al., 2019 — Adapter](https://arxiv.org/abs/1902.00751)
- [Li & Liang, 2021 — Prefix-Tuning](https://arxiv.org/abs/2101.00190)
- [Lewis et al., 2020 — RAG](https://arxiv.org/abs/2005.11401)
- [Rozière et al., 2023 — Code Llama（继续预训练的典型案例）](https://arxiv.org/abs/2308.12950)
- [Wu et al., 2023 — BloombergGPT](https://arxiv.org/abs/2303.17564)
- [Liu et al., 2023 — LLaVA（视觉指令微调）](https://arxiv.org/abs/2304.08485)
- [Peng et al., 2023 — YaRN（长上下文扩展）](https://arxiv.org/abs/2309.00071)
- [Ilharco et al., 2022 — Task Arithmetic（任务向量与模型合并）](https://arxiv.org/abs/2212.04089)
- [Wortsman et al., 2022 — Model Soups](https://arxiv.org/abs/2203.05482)
- [Kirkpatrick et al., 2016 — EWC（弹性权重巩固）](https://arxiv.org/abs/1612.00796)
- [Thinking Machines, 2025 — On-Policy Distillation（含用旧检查点恢复能力的做法）](https://thinkingmachines.ai/blog/on-policy-distillation/)

---

**下一讲**：第 13 讲　训练系统与稳定性——rollout 引擎、异步 RL，以及那个最隐蔽的崩溃源：训练与推理算出来的 logprob 不一样。
