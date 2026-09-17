# 第 15 讲　前沿问题与实战项目：把这十五讲变成你的东西

> **本讲在主线中的位置**：收尾。前十四讲给的是地图，这一讲给两样东西——**还没有答案的问题**（研究入口），以及**能放进简历的项目**（求职入口）。
>
> **学完你应该能做到**
> 1. 说清 RL 算力的 scaling 形态，以及它对"该研究什么"的直接含义。
> 2. 复述五个未决问题各自的现状与可行的切入点。
> 3. 选一条实战项目线，知道基线、评测、失败模式与交付物长什么样。
> 4. 把这些内容组织成面试里能讲的东西。


## 0. 一个总框架：三种资源，三种信号密度

整套讲义可以压成一张表。后训练（post-training）的一切选择，都是在**信号密度**与**获取成本**之间做交换：

| 信号来源 | 每条轨迹（trajectory）的信息量 | 获取成本 | 能否产生新能力 | 讲次 |
|---|---|---|---|---|
| 人类示范 | 每 token 一个标签 | 标注贵 | 否（上限是示范者） | 5 |
| 人类偏好 | 每对一个 bit | 标注贵 + RM 会被 hack | 有限 | 7、8 |
| 验证器（verifier） | 每条一个标量 | 环境/验证器工程 | **是**（可探索） | 9 |
| 环境反馈（environment feedback） | 每条一个标量（但状态可读） | 环境是主要成本 | **是** | 11 |
| 教师分布 | 每 token 整个词表 | 需要已有更强的教师 | 否 | 5 |

**两条推论**：

1. 只有带**探索**的信号源（验证器、环境）能越过现有分布的边界；其余都是在传递已有能力。
2. 信息密度最高的（教师分布）恰恰不产生新能力。所以 2026 年的主流配方是——**用 RL 探索出专家，用蒸馏廉价地合并与传递**（第 5 讲）。


## 1. 未决问题一：RL 算力怎么 scale

[ScaleRL](https://arxiv.org/abs/2510.13786)（2025.10，超过 40 万 GPU 小时的系统研究）给出的拟合形式：

$$
r_C=r_0+(A-r_0)\cdot\frac{1}{1+\left(\dfrac{C_{\text{mid}}}{C}\right)^{B}}
$$

- $A$：**渐近上限**，这个配方最终能到哪。
- $B$：**算力效率**，收敛有多快。
- $C_{\text{mid}}$：达到一半增益所需的算力。

最有价值的结论是它把两件长期被混为一谈的事分开了：**不是所有配方的上限都一样**，而 loss 聚合方式、归一化、课程、off-policy 算法这些细节**主要改变效率（$B$），不太改变上限（$A$）**。

**对研究选题的含义**：

- 如果你的新方法只是让曲线更早上升，那它的价值是省钱，要按省钱来论证（同等性能下的算力比），而不是按分数论证。
- 想抬高 $A$，通常要动数据、环境、奖励信号的**种类**，而不是优化器层面的细节。
- 小规模实验可以外推：用低算力段拟合曲线，预测大算力段，这是小团队做 RL 研究的现实路径。


## 2. 未决问题二：激发还是扩展

第 10 讲给了三方证据。这里只补**可执行的判定协议**：

1. 在基座 pass@1024 为 0 的题集上评估 RL 后模型。
2. 评测集时间切分（time split），排除污染。
3. 至少两个基座族复现。
4. 对照同等算力的"拒绝采样（rejection sampling） + SFT"。
5. 报告多个 $k$ 的 pass@k 与 pass^k。

**为什么这是个好切入点**：它需要的算力不大（可以在 1–7B 上做），但结论对整个领域有意义，且现有工作在方法学上普遍不严格。


## 3. 未决问题三：RL 与蒸馏的分工

2026 年的事实是：多家实验室把混合 RL 阶段换成了"领域专家 RL + 多教师在线策略蒸馏（on-policy distillation）"（第 5 讲）。由此产生的开放问题：

- **递归自举的上限在哪**？专家由 RL 造出，学生由蒸馏合并，下一代专家又从学生出发——这个循环能走多远，会不会收敛到一个不动点？
- **教师之间冲突怎么处理**？当前做法是加权求和反向 KL，权重是超参。有没有原则性的做法（例如按领域置信度动态加权）？
- **蒸馏是否会系统性压缩多样性**？反向 KL 是 mode seeking，多轮之后 pass@k 会怎样？这一点目前几乎没有公开数据。


## 4. 未决问题四：不可验证任务的奖励

可验证任务的路线已经相对成熟；写作、咨询、研究报告这类没有唯一答案的任务仍然开放。当前三条路线：

| 路线 | 现状 | 风险 |
|---|---|---|
| 标量 RM | 成熟但易被 hack，过优化曲线明确 | 第 7 讲的转折点问题 |
| Rubric + 生成式 RM | 2026 年主流方向，可解释可审计 | rubric 的覆盖面与维护成本 |
| 自评判（actor 兼任评委） | DeepSeek-V4 的做法 | 自我偏好（self-preference）与共谋：策略与评委共享失败模式 |

**一个清晰的研究问题**：当评委本身也在被 RL 优化时，reward hacking 是被消除了，还是只是转移到了评委身上？这个问题目前没有好的实证答案，而它可以在中小规模上研究。


## 5. 未决问题五：持续学习

模型上线后知识会过时、需求会变化，但每次重训代价高昂。当前可用的三条路径（第 12 讲）——回放、参数高效更新、从旧检查点蒸馏——都属于缓解而非解决。更深的问题：

- **知识该放在权重里还是上下文里**？百万上下文与检索让"放外面"变得便宜，那么还有多少知识值得写进权重？
- **能否做到无遗忘的增量更新**，且不需要保留全部历史数据？
- **在线学习的安全性**：一个持续从用户交互中学习的模型，如何避免第 7 讲那个谄媚事故的放大版？


## 6. 三条实战项目线

下面每条都按**能写进简历**的标准设计：有基线、有评测、有可交付物。模型规模控制在单卡或少量卡可跑。


### 项目 A｜GRPO 的逐项消融复现

**问题**：DAPO 报告的四件套各自贡献多少？在更小的模型与更少的算力下还成立吗？

| 项 | 建议 |
|---|---|
| 模型 | 1.5B–4B 的开源小模型，**两个不同族**（关键：抗基座特异性） |
| 数据 | 公开数学 RL 数据集（如 DAPO 放出的题集） |
| 框架 | verl 或 TRL |
| 基线 | 朴素 GRPO |
| 变量 | 去 std 归一化、token 级聚合、clip-higher、动态采样、超长软惩罚，逐项加 |
| 评测 | 时间切分的数学题；报告 avg@k、pass@1/8/64、熵、长度曲线 |
| 交付 | 一张消融表 + 曲线图 + 一页结论；代码可一键复现 |

**预期失败**：小模型上部分技术收益不显著；这本身是有价值的结论，**只要你把算力与基座都报清楚**。


### 项目 B｜不可验证任务的 rubric 奖励

**问题**：在一个没有标准答案的任务（例如给定材料写结构化摘要、或客服回复）上，rubric + 生成式评委能不能带来真实提升，而不只是评委分数的提升？

| 项 | 建议 |
|---|---|
| 任务 | 选一个你能判断质量的小领域 |
| 奖励 | 写 5–8 条可核查的 rubric 条目，用一个模型当评委 |
| 关键设计 | 评委与被训模型**不同族**；双向评测消位置偏好（position bias）；长度对齐 |
| 评测 | 独立评委 + 人工盲评 50 条；同时测通用能力是否退化 |
| 交付 | rubric 全文、评委与人类一致率、训练前后的人工盲评胜率 |

**预期失败**：评委分数涨、人工盲评不涨。这正是最值得写进报告的发现——它复现了第 7 讲的过优化。


### 项目 C｜在线策略蒸馏 vs RL 的成本对比

**问题**：把两个领域专家合并成一个模型，OPD 与混合 RL 哪个更省、效果差多少？

| 项 | 建议 |
|---|---|
| 步骤 | 先各训两个小专家（数学、代码），再用两种方式合并 |
| 对照 | (1) 混合数据 RL；(2) 多教师 OPD；(3) 简单的权重平均 |
| 指标 | 达到相同合并效果所需的 GPU 小时；两个领域的能力保持率；pass@k 变化 |
| 交付 | 成本-效果曲线 + 多样性分析 |

**预期失败**：小规模上专家之间冲突不明显，OPD 优势体现不出来。可以人为制造冲突（让两个专家的输出风格差异更大）来放大效应。


### 项目 D（可选）｜最小可用 agent 环境

做一个可自动判定、可重置、可并发的小环境（例如受限 shell 任务或 SQL 任务），跑一轮 RL，报告 **pass^k** 而不是 pass@k，并主动展示你发现的 reward hacking 手法。**"我构造了环境并抓到了模型作弊"这句话在面试里比分数有说服力得多。**


## 7. 怎么把项目变成实习竞争力

**交付物规范**（这几条本身就是能力信号）：

- README 写清：模型、数据、算力、超参、采样次数、温度、随机种子。
- 一键复现脚本，以及一个跑得动的小规模冒烟配置。
- 报告里必须有**失败分析**，不只有成功曲线。
- 明确写出你**没有**验证的东西与结论的边界。

**面试怎么讲**：用"问题 → 假设 → 实验 → 发现 → 我改变了什么判断"的结构，而不是流水账。最加分的一句话往往是：**"我原本以为 X，数据显示不是，所以我改成了 Y。"**

**岗位地图**（不同岗位看重的讲次不同）：

| 岗位 | 核心考察 | 重点讲次 |
|---|---|---|
| Post-training research | 算法理解、实验设计、研究品味 | 4、6、7、8、9、10、15 |
| RL infra / training systems | 吞吐、稳定性、分布式 | 1、2、13 |
| Applied AI / forward-deployed | 需求拆解、适配选型、评测 | 12、14、11 |
| Evaluation / data | 测量学、污染、judge 设计 | 14、5 |
| Alignment / safety | reward hacking、CoT 监控、红队（red teaming） | 7、10、14 |


## 8. 八周学习与项目计划

| 周 | 读 | 做 |
|---|---|---|
| 1 | 第 1–3 讲 + Transformer/scaling 经典 | 跑通一次小模型 SFT，手写 attention 与 KV cache 估算 |
| 2 | 第 4–5 讲 + InstructGPT、LIMA | 做一次 SFT + 拒绝采样，建立评测脚本 |
| 3 | 第 6–7 讲 + PPO、GAE、Gao et al. | 训一个小 RM，画出 reward 与独立评测的分叉 |
| 4 | 第 8 讲 + DPO、SimPO | 跑 DPO，监控 chosen/rejected logprob，复现似然位移（likelihood displacement） |
| 5 | 第 9 讲 + DeepSeekMath、DAPO、Dr. GRPO | 启动项目 A 的第一组消融 |
| 6 | 第 10、13 讲 + R1、TIS/批不变 | 加上 $\Delta_t$ 监控，检查你的训推一致性 |
| 7 | 第 11、12 讲 + LoRA Without Regret | 用 LoRA 重跑一遍 RL，对比成本 |
| 8 | 第 14、15 讲 | 整理报告、写 README、准备面试讲法 |


## 9. 整套讲义回顾

**十五讲地图**：

```
Part I  模型从哪来      1 NLP与Transformer → 2 预训练与中期训练 → 3 模型谱系
Part II 后训练核心      4 全景（两轴+统一梯度） → 5 SFT与蒸馏 → 6 RL基础
                        → 7 奖励建模与RLHF → 8 DPO家族 → 9 RLVR与GRPO
                        → 10 推理模型 → 11 Agent与多轮RL → 12 Adaptation
Part III 系统与评估     13 训练系统与稳定性 → 14 评估、安全与对齐
Part IV  收尾           15 前沿与实战项目
```

**核心决策树**：

```
你的任务能自动判定对错吗？
├─ 能 ────────────────→ RLVR / GRPO 族（第 9 讲）
└─ 不能
   ├─ 有成对偏好数据 ─→ DPO 族（第 8 讲）；资源充足且要探索 → RLHF/PPO（第 7 讲）
   ├─ 只有单边标签 ───→ KTO
   ├─ 有更强的教师 ───→ 在线策略蒸馏（第 5 讲）
   └─ 只缺知识 ───────→ RAG，不要训练（第 12 讲）
```

**三句话**：

1. 后训练的主线是**学习信号从哪来**，以及**数据是谁生成的**。所有方法都是这两条轴上的一个点。
2. 所有 loss 都是"加权的对数似然梯度"，区别只在样本来源与权重（附录 C）。
3. 算法只是一半，另一半是**测量**与**系统**——被推翻的结论里，绝大多数死于污染、mask 错误或训推不一致，而不是算法选错。


## 10. 自测题

**1. ScaleRL 把哪两件事分开了？这对选题有什么含义？**

把渐近上限 $A$ 与算力效率 $B$ 分开。多数实现细节（loss 聚合、归一化、课程、off-policy 处理）主要改变 $B$。含义是：如果你的方法只提效率，就按"同等性能下省多少算力"来论证；想抬高上限，通常要动数据、环境或奖励信号的种类。

**2. 为什么说只有带探索的信号源才能越过现有分布的边界？**

示范、偏好、教师分布都是在传递已有的行为——上限是提供者。验证器与环境允许模型采样出没人示范过的解法并得到反馈，这是唯一能产生分布外新行为的机制。代价是信息密度极低，一条轨迹只回一个标量。

**3. "专家 RL + 多教师蒸馏"这条路线有哪些未解问题？**

递归自举的上限（学生变教师能走多远）；教师冲突时的加权原则（目前是超参）；以及反向 KL 的 mode seeking 是否会在多轮后系统性压缩多样性——最后一点几乎没有公开数据，适合作为研究切入。

**4. 当评委本身在被 RL 优化时，会发生什么？**

理论上有两种可能：评判能力随策略一起提升，或者策略与评委共享失败模式并互相迁就（自我偏好与共谋）。目前缺少实证结论，而这个问题在中小规模上可以研究，是很好的选题。

**5. 项目 B 里"评委分数涨、人工盲评不涨"为什么是个好结果？**

它复现了奖励模型过优化：代理指标上升而真实质量没有跟上。这说明你的实验设计里有独立评估，能够抓到这种背离——而这正是招聘方想看到的能力。

**6. 为什么做小模型实验也要在两个基座族上跑？**

因为单一基座的结论可能来自它的预训练（pre-training）记忆而非你的方法，伪奖励现象就是例子。跨族复现是最低成本的抗污染手段。

**7. 简历项目里最值得写的一句话是什么类型？**

"我原本以为 X，数据显示不是，所以我改成了 Y。"它同时展示了假设、实验、以及根据证据修正判断的能力，比任何分数都更能说明你真的在做研究。


## 11. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Research taste — if you had 64 GPUs for one month, what post-training question would you answer?**
*Loops: research scientist (OpenAI, Anthropic, Google DeepMind); the standard "do you have taste" question.*

*Answer key*: pick something decidable at that scale and state the decision rule up front. Good shapes: an ablation that separates ceiling from efficiency; a clean test of elicitation vs extension with the five-point protocol; a study of whether reward hacking migrates to the judge when the judge is itself trained. Bad shapes: "train a better model" or anything that needs frontier-scale compute to show an effect.

**2. Scaling — how would you decide whether a new RL recipe is worth adopting without running it to convergence?**
*Anchored to: ScaleRL (2025). Loops: research scientist / staff engineer (frontier labs).*

*Answer key*: fit the sigmoidal compute–performance curve on low-compute runs and compare the asymptote A and the efficiency exponent B against your current recipe; extrapolate rather than eyeballing early reward curves, because early-fast recipes often plateau lower. State explicitly which of the two parameters the new method is supposed to improve.

**3. Project walkthrough — tell me about an experiment of yours that failed.**
*Loops: universal; usually the highest-signal question in the loop.*

*Answer key*: structure as hypothesis → experiment → contradicting evidence → changed belief → what you did next. Name the specific measurement that caught it (entropy curve, log-prob gap, human blind eval). Avoid framing failure as bad luck; frame it as information. The sentence interviewers remember is "I expected X, the data said otherwise, so I changed to Y."

**4. Scoping — a PM asks for "a model that writes better reports." How do you turn that into a training problem?**
*Loops: applied AI / forward-deployed (OpenAI, Anthropic, Scale AI).*

*Answer key*: first make quality checkable — decompose into a rubric with 5–8 verifiable clauses and build a human-blind eval before touching training. Then pick the cheapest sufficient intervention: prompt and retrieval, then SFT on curated examples, then preference optimization or rubric-based RL. Set the judge up with a different base model, length control, and periodic human recalibration.

**5. Frontier view — is RL going away now that labs distill experts instead of running mixed RL?**
*Anchored to: DeepSeek-V4, MOPD, GLM-5 (2026). Loops: research scientist and senior applied.*

*Answer key*: no — RL moved rather than disappeared. Distillation cannot exceed its teachers, and the teachers are produced by RL. What changed is the merge stage, where dense per-token signal beats scalar rewards. Open questions worth naming: recursive bootstrapping limits, teacher-conflict weighting, and diversity loss from repeated reverse-KL training.

**6. Prioritization — you join a team whose RL runs are unstable and whose evals are noisy. What do you fix first?**
*Loops: senior/staff (frontier labs, startups).*

*Answer key*: evals first. Without a trustworthy measurement you cannot tell whether a stability fix worked. Then measurement of the training loop itself — mask correctness, log-prob gap, verifier reliability, zero-variance group fraction. Algorithms last. Justify it by the observation that most "the algorithm doesn't work" reports resolve into data or measurement problems.

**7. Ethics/judgment — leadership wants to add a thumbs-up signal to the reward mixture to boost engagement. Your response?**
*Anchored to: the GPT-4o sycophancy episode (2025). Loops: applied and safety-adjacent roles.*

*Answer key*: name the failure mode explicitly — user satisfaction rewards feeling good rather than being useful, and the known outcome is sycophancy. Counter-proposal: keep the signal but cap its weight, pair it with a sycophancy eval in the pre-launch gate, and monitor for the specific behaviors (agreeing with false premises, excessive praise) both in training and post-launch. Insist the eval exists before the reward ships.


## 12. 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。

- [Khatri et al., 2025 — The Art of Scaling RL Compute（ScaleRL）](https://arxiv.org/abs/2510.13786)
- [Yue et al., 2025 — Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?](https://arxiv.org/abs/2504.13837)
- [Liu et al., 2025 — ProRL](https://arxiv.org/abs/2505.24864)
- [Shao et al., 2025 — Spurious Rewards](https://arxiv.org/abs/2506.10947)
- [Yu et al., 2025 — DAPO（项目 A 的复现对象与数据集）](https://arxiv.org/abs/2503.14476)
- [Liu et al., 2025 — Understanding R1-Zero-Like Training（Dr. GRPO）](https://arxiv.org/abs/2503.20783)
- [Thinking Machines, 2025 — On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Thinking Machines, 2025 — LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- [MOPD, 2026 — Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2606.30406)
- [DeepSeek-AI, 2026 — DeepSeek-V4（专家 RL + OPD、GRM 自评判）](https://arxiv.org/abs/2606.19348)
- [verl — 训练框架与 DAPO 配方](https://verl.readthedocs.io/en/latest/algo/dapo.html)
- [TRL — 文档与配方](https://huggingface.co/docs/trl)

---

**全篇完。** 附录 A（符号表与缩写表）与附录 C（Loss 速查）已单独成文；附录 D（面试题库汇总）、E（论文时间线与阅读顺序）、F（勘误记录）可在需要时再整理。
