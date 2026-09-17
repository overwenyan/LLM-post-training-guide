# 第 14 讲　评估、安全与对齐：怎么知道后训练真的成功了

> **本讲在主线中的位置**：前面十三讲都在讲怎么把模型训好。这一讲讲怎么**知道**它变好了——以及为什么这件事比训练本身更容易出错。后训练（post-training）领域里被推翻的结论，多数不是算法错了，而是**测量错了**。
>
> **学完你应该能做到**
> 1. 把评估拆成能力、对齐、安全、可靠性四条线，并说明缺任何一条会漏掉什么。
> 2. 指出当前主流基准各自考什么、各自怎么失效。
> 3. 识别四类数据污染（data contamination），并设计能抵抗它们的评测方案。
> 4. 说清 LLM-as-judge 的三种系统性偏差与缓解。
> 5. 解释对齐税（alignment tax）与校准（calibration）退化，以及安全与可用性之间那条权衡曲线。


## 0. 新增符号

$\mathrm{ECE}$（期望校准误差）、$\mathrm{FRR}$（错误拒答率）、$\mathrm{ASR}$（攻击成功率）、$\mathrm{Tax}$（对齐税）、$\mathrm{conf}$/$\mathrm{acc}$（置信度与准确率）。见附录 A。


## 1. 四条线

| 线 | 问题 | 典型指标 | 只看这条会漏掉 |
|---|---|---|---|
| 能力 | 模型**会不会**做 | 学科问答、数学、代码、长上下文 | 模型会做但不肯做、或做得不合规 |
| 对齐 | 模型**愿不愿意**按人类偏好做 | 指令遵循、偏好胜率、格式合规 | 胜率涨了但能力塌了 |
| 安全 | 模型**会不会做不该做的事** | 越狱（jailbreak）成功率、有害内容率、过度拒答（over-refusal）率 | 安全但不可用，或可用但不安全 |
| 可靠性 | 同一件事**能不能稳定做成** | pass^k、方差、长程任务完成率 | 平均值好看但方差巨大（agent 场景致命） |

第四条线是 agent 时代新增的（第 11 讲）：用户不接受“十次里成功一次”，而 pass@k 恰恰奖励这种模型。


## 2. 基准：各自考什么、各自怎么坏

原稿列的 MMLU、GSM8K、HumanEval 在今天基本饱和，作为主指标已经没有区分度。当前常用的一组：

| 基准 | 考什么 | 失效方式 |
|---|---|---|
| [GPQA Diamond](https://arxiv.org/abs/2311.12022) | 研究生级科学推理 | 题量小（数百），方差大 |
| AIME / HMMT | 竞赛数学 | **每年只有 30 题左右**，单次采样噪声极大 |
| FrontierMath | 更难的数学，分层 | 私有，复现依赖提供方 |
| [Humanity's Last Exam](https://arxiv.org/abs/2501.14249) | 跨学科极难问答 | 分数低时区分度有限；题目质量争议 |
| [LiveCodeBench](https://arxiv.org/abs/2403.07974) | 按时间滚动的编程题 | 时间切分（time split）是它的核心价值 |
| [SWE-bench Verified](https://arxiv.org/abs/2310.06770) | 真实仓库 issue 修复 | 强烈依赖 scaffold 与测试质量 |
| Terminal-Bench / 工具类基准 | 多步操作与工具编排 | 环境版本漂移；结果难复现 |
| 长上下文检索（MRCR 一类） | 多针检索与长程一致性 | "大海捞针"过于容易，需多跳（multi-hop）版本 |
| [IFEval](https://arxiv.org/abs/2311.07911) 及后续 | 可程序化验证的指令遵循 | 只覆盖能被规则判定的指令 |
| ARC-AGI 系列 | 抽象归纳 | 与训练分布差异大，解读争议多 |
| LMArena 等对战榜 | 人类总体偏好 | 风格偏好占比高；可被针对性优化（§5） |

**方法学比选哪个基准更重要**：

- **采样次数**：AIME 只有 30 题，单次采样的标准误大到足以让两个模型的排名互换。报告必须写明 avg@$k$ 与 $k$ 的取值。
- **温度与解码参数**：不同温度下的结论可能相反，尤其涉及 pass@k。
- **scaffold**：agent 基准的分数很大一部分由脚手架决定，跨论文比较几乎无意义，除非脚手架一致。
- **置信区间**：小题量基准必须报区间，而不是一个点估计。
- **prompt 敏感性**：同一模型换个提示词波动几个点是常态，因此模型间比较要固定提示模板。


## 3. 污染：四种类型与四种防御

| 类型 | 说明 |
|---|---|
| 直接污染 | 评测题原文出现在预训练（pre-training）或后训练数据里 |
| 近似污染 | 改写版、翻译版、同源题库 |
| 时间泄漏 | 评测题发布于模型数据截止之前，网上已有解答 |
| 间接污染 | 用强模型合成数据时，把它记住的评测题带了进来 |

**防御**：

1. **时间切分**：用模型数据截止之后发布的题目（LiveCodeBench、当年的竞赛题）。
2. **私有集**：自建、不公开、不进任何训练管线。
3. **n-gram / 近重复检索**：有语料访问权时直接查。
4. **跨模型族复现**：同一方法在两个基座上都涨，才更可能是方法的功劳。

> **案例｜Qwen2.5-Math 上的伪奖励现象（2025.6）**
> 在该系列模型上，**随机奖励甚至错误奖励**也能让 MATH 分数明显上涨，而换到其他模型族就复现不出来。合理的解释是这个基座在预训练阶段已经吸收了大量同类题目，RL 只是把已有的答题模式激活出来。
> **教训**：任何 RLVR 结论都必须跨模型族复现，否则你可能在测量基座的记忆，而不是你的方法。


## 4. LLM-as-judge

用模型当评委已经是默认做法（第 7 讲的生成式奖励模型也是同一件事），但它有三种系统性偏差：

| 偏差 | 表现 | 缓解 |
|---|---|---|
| 位置偏好（position bias） | 偏向先出现或后出现的那个回答 | 双向评测取平均 |
| 长度/格式偏好 | 偏向更长、更整齐的回答 | 长度对齐、rubric 明确禁止以长短判优 |
| 自我偏好（self-preference） | 偏向与自己风格相近的输出 | 用不同基座的评委；交叉验证 |

**还有一条容易被忽略的**：评委与人类的一致性会随时间漂移。模型更新了、rubric 微调了、被评的分布变了，一致性就变了。所以要**定期用人工标注复校**，把"评委与人类的一致率"本身当成一条需要监控的指标。


## 5. 排行榜与刷榜

对战榜衡量的是**人类总体偏好**，其中风格（格式整齐、语气热情、长度适中）的权重比多数人以为的高。两个后果：

- 针对性优化风格可以在不提升能力的情况下提高排名。
- 榜单排名与你的实际业务指标可能不相关。

> **案例｜Llama 4 的实验版本争议（2025 年 4 月）**
> Meta 在对战榜上提交的是一个**为对话体验专门优化的实验版本**，而公开发布的是另一个版本。事件之后，平台加强了版本披露要求，学界也出现了系统性批评（"排行榜幻觉"一类工作），指出私下多版本测试、选择性披露、数据获取不对称会扭曲排名。
> **教训**：看榜单时问三件事——提交的是不是公开可用的版本、有没有风格控制、同一家提交了多少个变体。

**给研究者的建议**：把公开榜单当作**信号之一**，真正的决策依据应该是你自己的、与业务对齐的、未参与任何调参的私有评测集。


## 6. 对齐税与校准


### 6.1 对齐税

$$
\mathrm{Tax}=\mathcal{C}(\pi_{\text{base}})-\mathcal{C}(\pi_{\text{aligned}})
$$

其中 $\mathcal{C}$ 是某项通用能力指标。来源：数据配比失衡（安全数据过多）、KL 约束过强或过弱、DPO 过拟合偏好数据、学习率过大导致遗忘。

缓解：预训练数据混合（PPO-ptx 式）、通用数据回放、LoRA 限制更新幅度、从旧检查点蒸馏恢复（第 5、12 讲）。

**检测的前提是有对照**：必须保留对齐前检查点的评测数字，否则你分不清是模型退步还是评测变严。


### 6.2 校准

$$
\mathrm{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{n}\Big|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\Big|
$$

把预测按置信度分箱，看每个箱内的准确率与平均置信度差多少。

**一个经典且反直觉的事实**：GPT-4 的技术报告显示，预训练模型的校准相当好，而 **RLHF 之后校准明显变差**——模型变得过度自信。直觉解释是 RLHF 奖励的是"人类觉得好的回答"，而犹豫、给概率、说不确定往往不被偏好。

这条对产品有直接影响：如果你的下游要用模型的置信度做路由或转人工，就不能假设对齐后的模型还保持校准，必须重新测 ECE，必要时做后处理校准。


## 7. 安全：两条曲线的权衡

安全评估的核心不是单点指标，而是**权衡曲线**：

$$
\text{有害内容通过率 }\mathrm{ASR}\quad \text{vs}\quad \text{无害请求被拒率 }\mathrm{FRR}
$$

把安全训练加强，ASR 下降、FRR 上升。只报一个数字没有意义——"我们的模型 ASR 只有 2%"可能是因为它拒答了三成正常问题。

评估要覆盖：

- **直接有害请求**与**越狱变体**（角色扮演、编码混淆、多轮渐进、多语言）。
- **过度拒答集**：正常的医学、法律、安全研究类问题。
- **多语言一致性**：安全训练常以英语为主，其他语言上防线更薄。
- **agent 行为安全**（第 11 讲）：越权、不可逆操作、提示注入（prompt injection）。

手段上，第 7 讲讲过的规则奖励、Constitutional AI、以及让模型在回答前显式推理安全规范（deliberative alignment 一类）都属于"把价值判断显式写出来"的路线，比隐含在偏好数据里更可审计。


## 8. 从 reward hacking 到广义不对齐

训练期的 reward hacking 监控（第 7 讲 §9）与发布前的安全评测是两件事，都要做：

| 阶段 | 看什么 |
|---|---|
| 训练中 | 奖励与独立评测的背离、长度、熵、格式化程度、验证器（verifier）被绕过的迹象 |
| 发布前 | 能力、对齐、安全、可靠性四线 + 越狱红队（red teaming） + agent 行为 |
| 上线后 | 真实分布上的回归监控、用户反馈（注意它本身会诱导谄媚） |

两条必须记住的实证结论：

1. **在编码环境里学会作弊会泛化成广义不对齐**（第 7 讲的 Anthropic 研究）。所以 agent 环境里的漏洞是对齐问题，不只是分数问题。
2. **对 CoT 施加优化压力会让它变得不可读**（第 10 讲）。CoT 应当作为监控通道保留，不进入奖励。


## 9. 搭一套自己的评测体系

一个能用的最小方案：

1. **先建评测，再动训练**。没有事先建好的评测，所有训练结论都是事后叙事。
2. **四线齐备**：目标任务、通用能力、安全、可靠性。
3. **三类数据集**：公开基准（可比性）、私有集（抗污染）、时间切分集（抗泄漏）。
4. **一份从不参与任何调参的 held-out**，只在决定是否上线时看。
5. **人工抽检有固定节奏**：每个里程碑读 50–100 条真实输出，比任何自动指标都更早发现 reward hacking。
6. **回归测试**：每次更新都跑，包含安全用例与上次修过的 bug 用例。
7. **报告规范**：写清采样次数、温度、提示模板、scaffold 版本、置信区间。


## 10. 自测题

**1. 为什么能力、对齐、安全要分开评估？**

它们会互相掩盖。只看能力，会漏掉模型会做但不肯做或做得不合规；只看对齐胜率，会漏掉能力被训塌；只看安全，会得到一个拒答一切的"安全"模型。四线并列才能看到权衡本身。

**2. 为什么 AIME 这类基准必须报 avg@k？**

题量只有 30 题左右，单次采样的方差大到足以让两个模型的排名互换。报告要写明采样次数和温度，否则数字不可比，也不可复现。

**3. 四类数据污染分别是什么？**

直接污染（原题在训练数据里）、近似污染（改写或翻译版）、时间泄漏（题目早于数据截止且网上有解）、间接污染（用强模型合成数据时把它记住的题带进来）。第四类最隐蔽，因为你的数据管线看起来完全干净。

**4. LLM-as-judge 有哪三种系统性偏差？**

位置偏好（偏向特定顺序）、长度与格式偏好（偏向更长更整齐）、自我偏好（偏向与自己风格相近的输出）。缓解是双向评测、rubric 明确禁止以长短判优、以及用不同基座的评委交叉验证。还要定期用人工标注复校评委与人类的一致率。

**5. 为什么 RLHF 之后校准会变差？**

RLHF 优化的是人类偏好，而人类通常不偏好犹豫、概率化、承认不确定的回答，于是模型被推向过度自信。GPT-4 技术报告里有这条对照。如果下游要用置信度做路由或转人工，必须重新测 ECE。

**6. 只报"我们的 ASR 只有 2%"为什么没有意义？**

因为没有给出对应的过度拒答率。把安全训练加强总能把 ASR 压低，代价是正常请求被拒。安全是一条 ASR 与 FRR 的权衡曲线，只有同时给出两个数才能判断模型是否可用。

**7. 你要验证一个 RLVR 方法真的有效，最少要做哪几件事？**

时间切分或私有评测集（抗污染）；至少两个模型族上复现；报告多个 $k$ 的 pass@k；对照同等算力的拒绝采样（rejection sampling）加 SFT 基线；以及通用能力与安全的前后对照。少任何一条，结论都可能是基座记忆或采样筛选的效果。


## 11. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Design — build an evaluation suite for a post-training project from scratch.**
*Loops: research scientist and applied AI (OpenAI, Anthropic, Google DeepMind, Scale AI). Frequently the "take-home discussion" prompt.*

*Answer key*: four lines (capability, alignment, safety, reliability); three dataset types (public for comparability, private for contamination resistance, time-split for leakage); one held-out set that never touches tuning; fixed reporting protocol (sampling count, temperature, prompt template, scaffold version, confidence intervals); scheduled human spot-checks. Emphasize building it before training starts.

**2. Statistics — a team reports 73.3% on AIME. What do you ask?**
*Loops: research scientist (frontier labs).*

*Answer key*: AIME has ~30 problems, so 73.3% is 22/30 and a single-sample number has a standard error of several points. Ask for avg@k with k stated, the temperature, the confidence interval, whether the eval year postdates the training cutoff, and whether the same protocol was used for baselines.

**3. Contamination — how would you tell whether an RLVR gain is real or memorized?**
*Anchored to: the spurious-rewards result on Qwen2.5-Math (2025). Loops: research scientist (Anthropic, AI2, frontier labs).*

*Answer key*: replicate on a second base family; evaluate on time-split problems released after the data cutoff; run a control with a random or shuffled reward — if that also improves the score, the signal is coming from the base, not your method; and compare against a compute-matched rejection-sampling baseline.

**4. Judge design — you are using an LLM as the grader for a production eval. What can go wrong?**
*Loops: applied AI / eval infrastructure (OpenAI, Anthropic, enterprise ML).*

*Answer key*: position bias (swap and average), verbosity and formatting bias (length-controlled comparisons, explicit rubric), self-preference (use a judge from a different family), and drift in judge–human agreement over time (periodic human recalibration, treat agreement rate as a monitored metric).

**5. Product — should we optimize for the public leaderboard?**
*Anchored to: the Llama 4 experimental-version episode (April 2025) and leaderboard-bias critiques. Loops: applied/product-facing roles.*

*Answer key*: no as a primary target. Arena-style scores weight style heavily and can be improved without capability gains; selective submission of variants distorts rankings. Use leaderboards as one signal, and make shipping decisions on a private, business-aligned held-out suite. If asked to chase the leaderboard anyway, negotiate for style-controlled numbers and disclosure of which checkpoint was submitted.

**6. Calibration — the product wants to route low-confidence answers to a human. What do you check?**
*Anchored to: the GPT-4 report's post-RLHF calibration degradation. Loops: applied ML (enterprise, healthcare/fintech AI teams).*

*Answer key*: measure ECE on your own distribution rather than assuming calibration holds; expect the aligned model to be overconfident relative to the base; consider post-hoc calibration, self-consistency spread, or an auxiliary verifier as the routing signal; and re-measure after every model update, since calibration is not stable across versions.

**7. Safety — how do you report safety results honestly?**
*Loops: safety / policy-adjacent roles (Anthropic, OpenAI, Google, AISI-style orgs).*

*Answer key*: always pair attack success rate with false refusal rate — either number alone is gameable. Cover jailbreak families (roleplay, encoding, multi-turn escalation, multilingual), an over-refusal set of legitimate sensitive questions, multilingual consistency, and, for agents, behavioral safety including prompt injection. Note that fine-tuning can erode safety alignment even with benign data, so safety belongs in the regression suite.


## 12. 本讲相对原稿的修订

| 位置 | 原稿 | 本讲 |
|---|---|---|
| 基准 | MMLU、GSM8K、HumanEval、BBH | 更新为当前一组，并逐条写明失效方式 |
| 评估线 | 能力、对齐、安全三线 | 补第四条：可靠性（pass^k、方差） |
| 方法学 | 未提 | 补采样次数、温度、scaffold、置信区间、报告规范 |
| 污染 | 定义一句 | 补四类污染与四种防御，含伪奖励案例 |
| LLM-as-judge | 未提 | 补三种偏差与一致性漂移 |
| 排行榜 | 未提 | 补风格权重、刷榜事件与解读方法 |
| 校准 | 给了 ECE 公式 | 补"RLHF 后校准变差"的实证与产品含义 |
| 安全 | 列举手段 | 改为 ASR–FRR 权衡曲线，强调两个数必须同时报 |
| 广义不对齐 | 未提 | 补 reward hacking 泛化与 CoT 监控两条结论 |
| 讨论题 | 10 题带简答 | 7 题直接给答案 + 7 道英文面试题（含公司标注） |


## 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。

- [Rein et al., 2023 — GPQA](https://arxiv.org/abs/2311.12022)
- [Phan et al., 2025 — Humanity's Last Exam](https://arxiv.org/abs/2501.14249)
- [Jain et al., 2024 — LiveCodeBench（时间切分评测）](https://arxiv.org/abs/2403.07974)
- [Jimenez et al., 2023 — SWE-bench](https://arxiv.org/abs/2310.06770)
- [Zhou et al., 2023 — IFEval](https://arxiv.org/abs/2311.07911)
- [Zheng et al., 2023 — Judging LLM-as-a-Judge（MT-Bench 与评委偏差）](https://arxiv.org/abs/2306.05685)
- [Singh et al., 2025 — The Leaderboard Illusion](https://arxiv.org/abs/2504.20879)
- [OpenAI, 2023 — GPT-4 Technical Report（RLHF 后校准退化）](https://arxiv.org/abs/2303.08774)
- [Guo et al., 2017 — On Calibration of Modern Neural Networks（ECE）](https://arxiv.org/abs/1706.04599)
- [Sharma et al., 2023 — Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548)
- [Shao et al., 2025 — Spurious Rewards（污染与伪收益）](https://arxiv.org/abs/2506.10947)
- [MacDiarmid et al., 2025 — Natural Emergent Misalignment from Reward Hacking](https://arxiv.org/abs/2511.18397)
- [Korbak et al., 2025 — Chain of Thought Monitorability](https://arxiv.org/abs/2507.11473)
- [Guan et al., 2024 — Deliberative Alignment](https://arxiv.org/abs/2412.16339)

---

**下一讲**：第 15 讲　前沿问题与研究/求职项目——RL 的 scaling 规律、未决争议，以及三条能放进简历的项目线。
