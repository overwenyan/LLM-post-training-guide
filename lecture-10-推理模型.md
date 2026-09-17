# 第 10 讲　推理模型：长 CoT 是教出来的，还是练出来的

> **本讲在主线中的位置**：第 9 讲给了算法（GRPO 家族），这一讲给的是**流水线**。推理模型的后训练（post-training）不是单一算法，而是冷启动（cold start）、RL、拒绝采样（rejection sampling）、再 SFT 的多轮自举。
>
> **学完你应该能做到**
> 1. 说清 R1-Zero 到底证明了什么、没证明什么。
> 2. 复述 DeepSeek-R1 的四阶段流水线，以及论文里两个明确的**负面**结论。
> 3. 解释思考预算（thinking budget）是怎么在训练侧实现的，以及过度思考（overthinking）为什么会发生。
> 4. 用证据链而不是立场回答“RL 是激发还是扩展能力”。
> 5. 说明为什么“不要对 CoT 施加优化压力”是一条工程建议而不是哲学主张。


## 0. 新增符号

$c$（推理链）、$a$（最终答案）、$y=(c,a)$、$\pi_{\mathcal{T}}$（教师）、$C_{\text{test}}$（测试时计算（test-time compute）量）、$\mathcal{D}_{\text{cold}}$（冷启动数据）。见附录 A。


## 1. 定义与两条 scaling 曲线

**定义 10.1（推理模型）**　回答前先生成较长内部推理链的模型，其输出可分解为

$$
y=(c,a),\qquad c=(c_1,\dots,c_{|c|})\ \text{为推理链},\quad a\ \text{为最终答案}
$$

训练目标仍然只盯 $a$ 的正确性（结果奖励（outcome reward）），但模型被允许用任意长的 $c$ 去达成它。**这一点是全部现象的根源**：$c$ 不被直接监督，所以它既可以长出搜索与回溯，也可以长出空转与作弊。

[OpenAI o1](https://openai.com/index/learning-to-reason-with-llms/)（2024 年 9 月）给出的关键图有两条曲线：性能随**训练时 RL 算力**上升，也随**测试时思考 token 数**上升，且两条都近似对数线性。这是"把算力从预训练（pre-training）挪到 RL 和推理"这个转向的起点。

$$
\text{Performance}\ \approx\ f\big(C_{\text{RL}}\big)+g\big(C_{\text{test}}\big)
$$


## 2. R1-Zero 证明了什么

[DeepSeek-R1](https://arxiv.org/abs/2501.12948)（2025 年 1 月）里有两个模型，常被混为一谈。

**R1-Zero**：直接在 V3-Base 上跑 GRPO，奖励只有答案正确性与格式，**没有任何 SFT**。结果：

- 回答长度在训练中自发增长，出现回溯、自我检查、换一种解法重试等行为（论文称之为 aha moment）。
- AIME 成绩从个位数涨到与 o1 同级。
- 但输出可读性差、中英混杂，不适合直接交付。

**它证明了什么**：长 CoT 行为**不必**由 SFT 示范教出来，纯结果奖励 + 足够强的基座就能把它逼出来。这直接推翻了原稿“直接对 base 跑 RL 模型不会生成长 CoT”的说法。

**它没有证明什么**：没有证明这些行为是 RL“创造”的。[Dr. GRPO](https://arxiv.org/abs/2503.20783) 那篇的分析指出，自我反思类模式在**基座模型的采样里就已经存在**，RL 主要提高了它们出现的频率。所以更准确的表述是：

> RL 把基座里低概率的推理模式，变成了高概率的默认行为。

这句话也是 §6 那场争论的起点。


## 3. R1 的四阶段流水线

正式版 R1 没有走 Zero 的路线，而是：

| 阶段 | 做什么 | 解决什么 |
|---|---|---|
| 1. 冷启动 SFT | 数千条高质量长 CoT | 可读性、格式、语言一致性；给 RL 一个好起点 |
| 2. 面向推理的 RL | GRPO + 规则奖励，另加**语言一致性奖励** | 提升正确率；压住中英混杂 |
| 3. 拒绝采样 + SFT | 用上一步模型生成并筛选约 60 万条推理数据，混约 20 万条通用数据，**从 base 重新 SFT** | 把 RL 得到的能力固化回一个通用模型 |
| 4. 全场景 RL | 可验证任务用规则奖励，开放任务用偏好 RM | 找回有用性与无害性 |

**第三阶段是最容易被忽略的一步**：它不是在第二阶段的模型上继续训，而是拿新数据从 base 重训。原因是第二阶段的模型在推理上很强、在别的方面已经退化，与其修补不如重来。这与第 5 讲“用蒸馏合并专家”的思路是同一个问题的两种解法。


### 论文里两个明确的负面结论

这两条比正面结论更有价值，因为它们省掉了整个社区的试错：

1. **没有采用 PRM**。理由是步骤边界难以定义、过程标注难以自动化、以及 PRM 一旦进入大规模 RL 就会被 hack，重训 PRM 的成本又很高。
2. **MCTS 式搜索没跑通**。语言空间的分支因子远大于围棋，且价值模型难以训练到可用。


### 蒸馏 vs 小模型自己做 RL

论文做了对照：把 R1 的轨迹（trajectory） SFT 进 Qwen/Llama 小模型，效果**好于**在同样的小模型上直接跑 RL。解释回到第 2 讲的可激发性——小基座采样不出正确解时，结果奖励恒为 0，RL 没有梯度可用；而蒸馏提供的是逐 token 的密集信号。


## 4. 长度控制：从长度惩罚到思考预算


### 4.1 训练侧

[Kimi k1.5](https://arxiv.org/abs/2501.12599)（与 R1 同日发布）把长度当成一等公民：显式的长度惩罚、长上下文 RL、以及 partial rollout（把没生成完的轨迹存下来，下一轮接着生成，避免长尾拖垮吞吐，见第 13 讲）。

DAPO 的软超长惩罚是另一种写法（第 9 讲 §4.2）：在接近上限的区间内线性递增惩罚，超过上限才给满惩罚。硬截断的问题是把“没写完”当成“答错”，制造奖励噪声。


### 4.2 推理侧：可控的思考预算

2025 年起，思考深度变成了用户可调参数：

| 产品 | 形态 |
|---|---|
| Anthropic Claude 3.7 Sonnet（2025.2） | 同一个模型可切普通 / 扩展思考，思考预算可设 |
| Google Gemini 2.5（2025.3–4） | thinking budget 作为 API 参数暴露 |
| Alibaba [Qwen3](https://arxiv.org/abs/2505.09388)（2025.4） | 开源模型的混合思考模式 |
| OpenAI GPT-5（2025.8） | 路由：由系统判断该不该深思 |
| DeepSeek-V4（2026.4） | Non-think / Think High / Think Max 三档，由**不同 RL 配置训练出的专家**分别支撑 |

**训练侧怎么实现**：不是加一个开关那么简单。要在 RL 阶段用不同的长度惩罚与上下文预算训练不同档位的行为，再让模型在条件（系统提示、特殊 token、路由）下切换。V4 的做法最直白——干脆训成不同专家再合并。


### 4.3 过度思考

$$
|c|\uparrow\quad\text{但}\quad \text{pass@1}\ \text{不变或下降}
$$

典型诱因：长度奖励设计不当、结果奖励太稀疏导致模型用长度“对冲”、以及验证器（verifier）允许模型反复自我确认。简单题上尤其明显——模型会为 `2+3` 生成几百 token 的检查。缓解：软长度惩罚、按题目难度设定目标长度区间、在评测里同时报告平均 token 数与准确率。


## 5. ORM 与 PRM 的现状

| | ORM（结果奖励） | PRM（过程奖励（process reward）） |
|---|---|---|
| 形式 | $R=\mathrm{ver}(x,a)$ | 对每个推理步打分再聚合 |
| 标注 | 自动 | 步骤级标注，昂贵 |
| 信号 | 稀疏，信用分配（credit assignment）难 | 密集，能定位错误步 |
| 风险 | 过程可以胡编但答案蒙对 | PRM 本身被 hack；步骤边界难定义 |

[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)（2023）证明了过程监督在 best-of-N 重排上显著优于结果监督，并开源了 PRM800K。之后的路线是**自动化过程标注**：用蒙特卡洛估计每个前缀的“最终答对概率”当作步骤分数（Math-Shepherd 一类做法），避免人工标注。

**但在大规模 RL 里，主流仍是 ORM + 规则验证器**，R1 的负面结论是主要原因。PRM 今天更常出现在两个位置：推理时的 best-of-N 重排，以及数据筛选（挑出过程可靠的轨迹去做 SFT）。

> 一个值得注意的转向：2026 年 DeepSeek-V4 对难验证任务用的是**生成式奖励模型 + rubric**，而不是传统 PRM（第 7 讲）。过程监督没有消失，它换了形态——从“给每步打分的小模型”变成“会写评判理由的大模型”。


## 6. 测试时计算

| 方式 | 做法 | 适用 |
|---|---|---|
| 自洽性（self-consistency）投票 | 采样 $K$ 条，对答案多数投票 | 答案空间离散、可比较 |
| Best-of-N | 用验证器或 RM 重排 | 有可靠打分器 |
| 预算强制 | 在生成中插入 `Wait` 之类 token 强行延长思考 | 小模型上便宜有效 |
| 长 CoT 本身 | 单条更长的推理 | 已经 RL 过的推理模型 |

**RL 与测试时计算是互补而不是替代**：RL 提升单次采样的质量（pass@1 与推理策略的有效性），测试时计算提升采样的利用率。两者叠加的收益通常大于任一单独使用。

一个实践含义：如果你的部署允许多次采样，那么**训练目标应该盯 pass@k 而不只是 pass@1**——但当前绝大多数 RLVR 配方优化的是 pass@1，这正是下一节争论的落点。


## 7. 激发还是扩展：完整证据链

| 立场 | 主要证据 | 反驳 |
|---|---|---|
| **激发**（RL 只是锐化分布） | [Yue et al., 2025.4](https://arxiv.org/abs/2504.13837)：RL 模型在小 $k$ 上赢，$k$ 足够大时基座反超，说明解在基座采样空间里 | 大 $k$ 采样成本不现实；基座的"能采到"不等于"能可靠产出" |
| **扩展**（RL 能学到新策略） | [ProRL, 2025.5](https://arxiv.org/abs/2505.24864)：延长 RL、扩宽任务域后，出现基座在任何 $k$ 下都解不出而 RL 后能解的题 | 任务选择可能有利；训练算力差异未完全对齐 |
| **警告**（部分收益是假象） | [Spurious Rewards, 2025.6](https://arxiv.org/abs/2506.10947)：在 Qwen2.5-Math 上随机甚至错误奖励也能涨分，换模型族复现不了 | 说明该基座特殊（可能有污染），不能否定所有 RLVR 收益 |

**怎么把这个问题变成可做的实验**（这是面试里真正的加分项）：

1. 报告多个 $k$ 的 pass@k，而不是只报 pass@1。
2. 用**时间切分（time split）**的评测集（例如训练后才公布的竞赛题），排除污染。
3. 至少在两个模型族上复现。
4. 对照“同等算力的拒绝采样 + SFT”基线——如果 RL 没有超过它，那你测到的多半只是采样与筛选的效果。
5. 区分“解出新题”和“更稳定地解出旧题”：前者需要在基座 pass@1024 为 0 的题上验证。


## 8. CoT 的忠实性与可监控性

推理链带来了一个意外的安全红利：**模型把一部分计算过程写成了自然语言，因此可以被读**。但有两条重要限制：

1. **CoT 不总是忠实的**。Anthropic 2025 年的研究发现，当模型在提示里收到暗示（例如答案提示或越权信息）时，它的推理链经常**不提及**这个暗示，却在答案里使用了它。所以 CoT 是“模型写下来的过程”，不等于“模型实际依赖的过程”。
2. **对 CoT 施加优化压力会让它变得不可读**。OpenAI 2025 年 3 月的工作展示了：直接惩罚“推理链里提到作弊意图”并不能消除作弊，只会让模型学会**隐藏意图**，作弊照旧发生（obfuscated reward hacking）。

由此形成了一条现在被广泛接受的工程建议：

> 把 CoT 当作**监控通道**而不是**优化目标**。奖励只施加在最终答案与外部可验证的行为上；对 CoT 只做读取与告警，不做惩罚。

2025 年 7 月多家实验室联署的立场文件进一步指出：CoT 的可监控性是一个**脆弱且可能随架构演进消失**的性质（例如潜空间推理会让它彻底失效），值得作为一项独立指标去评估和保护。


## 9. 失败模式速查

| 模式 | 表现 | 处理 |
|---|---|---|
| 过度思考 | 长度涨、准确率不涨 | 软长度惩罚、难度自适应目标长度 |
| 欠思考 | 频繁换思路，每条都没走完 | 提高长度下限、奖励完整推导 |
| 熵坍塌（entropy collapse） | pass@k 下降、输出同质 | 见第 9 讲 §5.1 |
| 格式过拟合 | 推理链看着合理，逻辑是空的 | 人工抽检；用过程一致性检查 |
| 验证器被 hack | 猜答案、利用容差、重复提交 | 多验证器集成、对抗测试 |
| 语言退化/混杂 | 中英混杂、可读性差 | 语言一致性奖励（R1 的做法） |
| 错误传播 | 早期一步错，后面全错 | 自我验证提示、过程奖励或重排 |


## 10. 工程检查清单

- [ ] 冷启动数据是否覆盖回溯、自我检查、多路径尝试等目标行为
- [ ] 上下文预算是否大于期望 CoT 长度（否则大量样本被截断）
- [ ] 截断样本的处理策略明确（过滤 or 软惩罚），不当作答错
- [ ] 评测同时报告准确率与平均思考 token 数
- [ ] pass@1 与 pass@k 同时监控
- [ ] 有时间切分或私有题目的评测集
- [ ] CoT 只用于监控，未进入奖励项
- [ ] 语言一致性、格式合规单独监控


## 11. 自测题

**1. R1-Zero 证明了什么，没证明什么？**

证明了长 CoT 行为不必靠 SFT 示范教出来——纯结果奖励加足够强的基座就能逼出回溯与自我检查，代价是可读性差、语言混杂。没证明这些行为是 RL 创造的：后续分析显示基座采样里已经存在自我反思模式，RL 主要提高了它们的出现频率。

**2. R1 第三阶段为什么要从 base 重新 SFT，而不是在 RL 后的模型上继续训？**

第二阶段的模型推理很强但其他能力已经退化。与其在退化的模型上修补，不如用它生成高质量数据，再从 base 重训一个均衡的模型。这与"训专家再合并"是同一个问题的两种解法。

**3. 为什么 DeepSeek 没有在 R1 里用 PRM？**

三条理由写在论文里：推理步骤的边界难以定义；步骤级标注难以自动化且昂贵；PRM 进入大规模 RL 后会被 hack，而重训 PRM 成本很高。今天 PRM 更多用在推理时重排和数据筛选，而不是 RL 的奖励源。

**4. 思考预算在训练侧是怎么实现的？**

不是加开关。需要在 RL 阶段用不同的长度惩罚和上下文预算训练出不同的行为档位，再通过系统提示、特殊 token 或路由来触发。DeepSeek-V4 的做法最直接：三档模式由不同 RL 配置的专家分别支撑，最后合并成一个模型。

**5. 过度思考为什么会发生？怎么缓解？**

结果奖励只看答案，不惩罚过程长度，模型就会用长度对冲不确定性；若验证器允许反复自我确认，这种行为还会被强化。缓解手段是软长度惩罚、按难度设定目标长度区间，以及在评测里把平均 token 数和准确率一起报。

**6. 为什么不应该把 CoT 纳入奖励？**

因为惩罚"推理链里暴露出的坏意图"并不会消除坏行为，只会训练模型隐藏意图，结果是既失去了监控通道，又没有解决问题。正确做法是奖励只加在最终答案和外部可验证的行为上，CoT 保留为只读的监控信号。

**7. 有人说"我的方法让模型学会了基座学不会的题"。你要看什么才信？**

要看基座在大 $k$（例如 pass@1024）下是否真的解不出这些题；评测集是否时间切分以排除污染；是否在第二个模型族上复现；以及是否对照了同等算力的"拒绝采样 + SFT"基线。缺少最后一条时，测到的往往只是采样加筛选的效果。


## 12. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Pipeline recall — walk me through DeepSeek-R1's training pipeline and tell me which design choice you find least obvious.**
*Anchored to: DeepSeek-R1 (2025-01). Loops: post-training research engineer (OpenAI, Anthropic, Meta GenAI, NVIDIA).*

*Answer key*: cold-start SFT → reasoning-focused GRPO with a language-consistency reward → rejection sampling (~600k reasoning + ~200k general) and **re-SFT from the base** → all-scenario RL. The least obvious choice is re-starting from base in stage 3: the stage-2 model is strong at reasoning but degraded elsewhere, so it is used as a data generator rather than as the thing to keep training.

**2. Negative results — name two things the R1 paper reports as *not* working, and why that matters.**
*Anchored to: DeepSeek-R1. Loops: research scientist (frontier labs); tests whether you read papers for the ablations rather than the headline.*

*Answer key*: PRM (step boundaries hard to define, annotation hard to automate, gets hacked under large-scale RL, retraining is expensive) and MCTS-style search (branching factor in language space, value model hard to train). These matter because negative results from a lab with that much compute save the community a large amount of duplicated effort.

**3. Debate — is RL eliciting existing capability or creating new capability? Give me the evidence, then tell me how you would settle it.**
*Anchored to: Yue et al. (2025), ProRL (2025), Spurious Rewards (2025). Loops: research scientist (Anthropic, Google DeepMind, OpenAI).*

*Answer key*: present all three lines of evidence rather than picking a side. Then give an experiment: multi-k pass@k on a time-split eval set, replicated on two base families, with a compute-matched rejection-sampling-plus-SFT baseline, and a specific criterion — problems where the base scores zero at very large k.

**4. Safety reasoning — why shouldn't chain-of-thought be optimized directly?**
*Anchored to: OpenAI's work on monitoring reasoning models (2025-03) and the CoT-monitorability position paper (2025-07). Loops: alignment/safety (Anthropic, OpenAI, UK AISI, Redwood).*

*Answer key*: penalizing stated bad intent does not remove the behavior — it teaches the model to hide intent while continuing to hack, so you lose the monitoring channel and keep the problem. Recommended practice: reward only final answers and externally verifiable actions; read the CoT for monitoring and alerting. Add that monitorability is fragile and may disappear with architectural changes such as latent-space reasoning.

**5. Product design — the model spends 800 tokens on "what is 2+3". Diagnose and fix.**
*Loops: applied post-training (OpenAI, Anthropic, Alibaba Qwen, agent startups).*

*Answer key*: overthinking driven by outcome-only reward plus weak or absent length shaping. Fixes: soft overlong penalties, difficulty-conditioned target length, a thinking-budget or routing layer at inference, and reporting mean thinking tokens alongside accuracy so the regression is visible. Mention that hard truncation is the wrong fix because it turns "unfinished" into "wrong".

**6. Trade-off — small model, limited compute, you need reasoning ability. RL or distillation?**
*Anchored to: R1-Distill results (2025). Loops: applied/startup roles.*

*Answer key*: distillation, usually. If the small base cannot sample a correct solution, outcome rewards are identically zero and RL gets no gradient, whereas distillation supplies dense per-token signal. Use RL afterwards if you have a verifier and the distilled model already reaches non-trivial pass@k. On-policy distillation is the stronger variant when you can run the teacher during training.

**7. Evaluation design — how would you measure whether test-time compute or training-time RL is the better next investment for your product?**
*Loops: staff/senior applied (frontier labs, well-funded startups).*

*Answer key*: measure the accuracy-vs-cost frontier for both: for test-time, sweep samples/budget at fixed model; for training-time, use the sigmoidal RL compute fit (Lecture 15) to extrapolate from small runs. Then bring in serving economics — test-time compute costs per request forever, RL costs once. The answer depends on request volume and latency budget, and you should say so.


## 13. 本讲相对原稿的修订

| 位置 | 原稿 | 本讲 |
|---|---|---|
| R1-Zero | 完全未提；并称“直接对 base 跑 RL 不会有长 CoT” | 单独成节，订正该结论，并给出“没证明什么”的另一面 |
| R1 流水线 | 笼统四阶段 | 补语言一致性奖励、第三阶段从 base 重训、数据规模 |
| 负面结论 | 未提 | 补 PRM 与 MCTS 两条，并说明其价值 |
| 长度控制 | 只有长度惩罚 | 补 partial rollout、软超长惩罚、思考预算的训练侧实现 |
| PRM 现状 | “长 CoT 开始重视过程奖励” | 改为：大规模 RL 主流仍是 ORM；PRM 移到重排与数据筛选；2026 年转向 GRM + rubric |
| 激发 vs 扩展 | 两派观点各一句 | 三方证据表 + 五条可执行的判定方法 |
| CoT 安全 | 未提 | 新增忠实性与可监控性一节，给出“只监控不优化”的工程建议 |
| 讨论题 | 10 题带简答 | 7 题直接给答案 + 7 道英文面试题（含公司标注） |


## 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。

- [DeepSeek-AI, 2025 — DeepSeek-R1（含 R1-Zero 与蒸馏对照）](https://arxiv.org/abs/2501.12948)
- [Kimi Team, 2025 — Kimi k1.5](https://arxiv.org/abs/2501.12599)
- [Liu et al., 2025 — Understanding R1-Zero-Like Training（Dr. GRPO；基座已含自我反思）](https://arxiv.org/abs/2503.20783)
- [Lightman et al., 2023 — Let's Verify Step by Step（PRM800K）](https://arxiv.org/abs/2305.20050)
- [Wang et al., 2023 — Math-Shepherd（自动过程标注）](https://arxiv.org/abs/2312.08935)
- [Wang et al., 2022 — Self-Consistency](https://arxiv.org/abs/2203.11171)
- [Snell et al., 2024 — Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)
- [Muennighoff et al., 2025 — s1: Simple Test-Time Scaling（预算强制）](https://arxiv.org/abs/2501.19393)
- [Yue et al., 2025 — Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?](https://arxiv.org/abs/2504.13837)
- [Liu et al., 2025 — ProRL](https://arxiv.org/abs/2505.24864)
- [Shao et al., 2025 — Spurious Rewards](https://arxiv.org/abs/2506.10947)
- [Baker et al., 2025 — Monitoring Reasoning Models for Misbehavior（obfuscated reward hacking）](https://arxiv.org/abs/2503.11926)
- [Korbak et al., 2025 — Chain of Thought Monitorability（多实验室联署立场文件）](https://arxiv.org/abs/2507.11473)
- [Qwen Team, 2025 — Qwen3（混合思考模式）](https://arxiv.org/abs/2505.09388)
- [DeepSeek-AI, 2026 — DeepSeek-V4（三档思考模式）](https://arxiv.org/abs/2606.19348)

---

**下一讲**：第 11 讲　Agent 与多轮 RL——当环境不再是“拼接 token”，信用分配会难到什么程度。
