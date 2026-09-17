# 第 4 讲　Post-training 全景：两条轴、一个公式、四条真实流水线

> **本讲在主线中的位置**：第 1–3 讲讲清了模型从哪来（Transformer、预训练（pre-training）、各代模型谱系）。从这一讲开始进入后训练（post-training）。本讲只做一件事——建立地图，后面每一讲都挂在这张图上。
>
> **学完你应该能做到**
> 1. 看到任何一个后训练方法，能立刻说出它在两条轴上的位置。
> 2. 用同一个梯度公式解释 SFT、RFT、PPO、GRPO、DPO、蒸馏的区别。
> 3. 复述并比较四家实验室的真实后训练流水线（2022 / 2023 / 2025 / 2026 各一条）。
> 4. 回答“为什么 LLM 的 RL 和教科书 RL 不一样”这个必考题。

**新增符号**：本讲首次使用 $\pi_\theta,\pi_{\text{ref}},\mathcal{D},R(x,y),\beta$。完整定义见附录 A，后续各讲沿用，不再重复。

---


## 1. 两条轴：后训练的坐标系

原稿的主线是“奖励信号从哪来”。这条线是对的，但只有它不够——它解释不了 DPO 和 PPO 用同一种信号（人类偏好）却是两套完全不同的工程。所以本讲义用**两条轴**。

**轴一：学习信号从哪来（信号的密度与可靠性）**

$$
\underbrace{\text{下一个 token}}_{\text{预训练}}\;\to\;\underbrace{\text{人类/教师示范}}_{\text{SFT}}\;\to\;\underbrace{\text{人类偏好}}_{\text{RM / DPO}}\;\to\;\underbrace{\text{规则与验证器}}_{\text{RLVR}}\;\to\;\underbrace{\text{环境反馈}}_{\text{Agent RL}}\;\to\;\underbrace{\text{教师逐 token 分布}}_{\text{On-policy 蒸馏}}
$$

**轴二：训练数据是谁生成的（离线 → 在线）**

$$
\underbrace{\text{固定语料}}_{\text{SFT, DPO}}\;\to\;\underbrace{\text{旧策略采样}}_{\text{PPO, 异步 RL}}\;\to\;\underbrace{\text{当前策略采样}}_{\text{on-policy RL, OPD}}
$$

把两条轴画成平面，主流方法各就各位：

| | 离线数据 | 旧策略采样 | 当前策略采样 |
|---|---|---|---|
| **示范** | SFT、硬标签蒸馏 | —— | RFT / [STaR](https://arxiv.org/abs/2203.14465)、on-policy 蒸馏 |
| **人类偏好** | DPO 家族 | PPO（RLHF） | 在线/迭代 DPO |
| **规则验证器（verifier）** | 拒绝采样（rejection sampling）数据集 | GRPO、[DAPO](https://arxiv.org/abs/2503.14476) | RLVR（同步 on-policy） |
| **环境反馈（environment feedback）** | 离线轨迹（trajectory）模仿 | Agent RL（异步） | 多轮 on-policy RL |

> **为什么第二条轴重要**：轴二决定了你的**工程形态**。离线方法只需要一个训练引擎；在线方法需要推理引擎、参数同步、重要性采样（importance sampling）修正，整套系统复杂度翻倍（第 13 讲）。很多时候选 DPO 还是 GRPO，不是算法问题，是你有没有 rollout 基础设施的问题。


## 2. 一个公式：所有方法都是“加权的对数似然梯度”

$$
\boxed{\;\nabla_\theta \mathcal{L} \;=\; -\,\mathbb{E}_{(x,y)\sim \text{来源}}\Big[\sum_{t} w_t\,\nabla_\theta \log \pi_\theta(y_t\mid x,y_{<t})\Big]\;}
$$

| 方法 | $y$ 来自 | $w_t$ |
|---|---|---|
| SFT | 数据集 | $1$ |
| 拒绝采样微调 | 自己采样 | $\mathbb{I}[\mathrm{ver}(x,y)=1]$ |
| REINFORCE / GRPO | 自己采样 | $\hat A$（可正可负） |
| PPO | 旧策略 | $\rho_t\hat A_t$（被 clip） |
| DPO | 离线偏好对 | $\pm\beta\,\sigma(\hat r_l-\hat r_w)$ |
| On-policy 蒸馏 | 自己采样 | $\log\pi_{\mathcal{T}}(y_t)-\log\pi_\theta(y_t)$ |

这张表是全书的骨架。往后每见到一个新 loss，先问：**样本从哪来？权重是什么？** 完整推导和逐条“三问”在附录 C。

一个直接的推论，也是 2025–2026 年路线变化的根因——**信息密度**：

| 反馈方式 | 一条轨迹带回来的监督量 |
|---|---|
| RLVR（结果奖励（outcome reward）） | 1 个标量 |
| 过程奖励（process reward） PRM | $O(\text{步数})$ 个标量 |
| On-policy 蒸馏 | $\lvert y\rvert\times\lvert\mathcal{V}\rvert$ 个数 |

RL 的优势是能**探索**出数据里没有的解法，代价是每次探索只换回 1 bit 左右的信息。这个权衡贯穿第 9、10、15 讲。


## 3. 各阶段到底解决什么

| 阶段 | 信号 | 学习方式 | 解决的问题 | 解决不了的问题 |
|---|---|---|---|---|
| 预训练 | 下一个 token | 自监督 | 知识、语言能力 | 不听指令，会续写而不是回答 |
| SFT | 示范 | 模仿学习（前向 KL） | 指令跟随、chat 格式、基本能力调用 | 无法表达“两个回答哪个更好”；暴露偏差（exposure bias） |
| RLHF | 人类偏好 → RM | 在线 RL | 有用性、无害性、风格偏好 | 贵、不稳定、RM 可被 hack |
| DPO 家族 | 人类偏好对 | 离线偏好优化（preference optimization） | 以极低成本拿到大部分对齐收益 | 离线、探索弱、覆盖不到的行为纠正不了 |
| RLVR | 规则 / 验证器 | 在线 RL | 数学、代码、逻辑的正确率 | 只适用于可自动验证的任务 |
| Agent RL | 环境 | 多轮在线 RL | 工具使用、长程任务 | 环境构建是瓶颈，信用分配（credit assignment）更难 |
| On-policy 蒸馏 | 教师分布 | 在线模仿 | 把多个专家能力合成一个模型、修复遗忘 | 上限受教师限制，不产生新能力 |

**一句话**：SFT 教“怎么答”，偏好优化教“答哪个更好”，RLVR 教“答对没有”，Agent RL 教“怎么把事做成”，蒸馏教“像那个更强的自己一样答”。


## 4. 为什么 LLM 的 RL 和教科书 RL 不一样

这是面试高频题，按下面七条答，基本满分。

1. **动作空间巨大但离散且结构化**：每步从 $10^4\sim10^5$ 个 token 里选一个，但 token 之间有语言结构。
2. **状态转移是确定性的**：$s_{t+1}=s_t\oplus a_t$，就是拼接。没有环境随机性——**除非**接入工具、代码执行、浏览器（第 11 讲，那时才有真正的环境）。
3. **回合长、奖励稀疏**：一条轨迹几百到几万 token，奖励往往只在最后给一个标量。信用分配极难。
4. **初始策略已经很强**：不是从随机策略学起，而是从预训练 + SFT 出发。所以 RL 更像“校准（calibration）与激发”，也因此 KL 锚点和小学习率是标配。
5. **采样成本占大头**：rollout 通常吃掉整个 RL 训练的大部分算力，这决定了系统设计（第 13 讲）。
6. **奖励是代理指标，必然可被 hack**：长度、格式、谄媚、重复安全话术，都是模型能找到的捷径。
7. **评估本身是开放问题**：多数任务没有唯一正确答案，所以“RL 有没有真提升”本身就需要单独一讲（第 14 讲）。

> **常见追问**：既然状态转移确定、初始策略又强，为什么不直接用 best-of-N + SFT（即拒绝采样）就够了？
> **答题要点**：拒绝采样只用正样本、没有负向梯度，且每轮都要重新采样与筛选；RL 能利用失败轨迹的信息、能在 token 级分配信用、且可以持续在线改进。但在小规模、验证器可靠的场景，拒绝采样常常是性价比更高的第一步——[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 的流水线里两者都用（见 §5.3）。


## 5. 四条真实流水线

> **怎么读这一节**：注意每条流水线的**阶段数**在变多、**信号种类**在变杂、**谁生成数据**在往在线端移动。这正是两条轴的历史轨迹。


### 5.1 OpenAI InstructGPT（2022 年 1 月上线，3 月论文）——三阶段范式的原型

- **流程**：SFT（约 1.3 万条人工示范）→ RM（6B，约 3.3 万条人类比较）→ PPO。
- **关键设计**：逐 token KL 惩罚锚定 SFT 模型；PPO-ptx 在 RL 损失里混入预训练梯度，用来抵消对齐税（alignment tax）。
- **标志性结论**：标注者更偏好 1.3B 的 [InstructGPT](https://arxiv.org/abs/2203.02155)，而不是 175B 的 GPT-3。**后训练可以顶得上两个数量级的参数**——这是整个后训练领域立项的依据。
- **留给后面的问题**：四模型（Actor / Critic / RM / Reference）太贵 → 第 7、8 讲。


### 5.2 Meta Llama 2-Chat（2023 年 7 月）——迭代式、多奖励模型

- **流程**：SFT（27,540 条高质量标注，Meta 明确指出“质量压倒数量”）→ **两个独立 RM**（有用性、安全性）→ 五轮迭代（RLHF-V1…V5），每轮先**拒绝采样微调**再 **PPO**。
- **关键设计**：把“有用”和“安全”拆成两个 RM 再组合，避免单一 RM 内部目标打架；RM 训练用带 margin 的损失（偏好强度进入 loss）。
- **教训**：对齐不是一次性的，是**迭代**的——每轮新模型采样出的数据分布不同，RM 需要跟着更新，否则 RM 很快就在给分布外样本打分。
- **对应讲次**：第 7 讲（RM 与 PPO）、第 5 讲（拒绝采样）。


### 5.3 DeepSeek-R1（2025 年 1 月）——可验证奖励 + 多阶段自举

- **R1-Zero**：直接在 V3-Base 上跑 GRPO，只用规则奖励（答案正确性 + 格式）。**没有任何 SFT**，长 CoT 与自我检查行为自己涌现了，但输出可读性差、中英混杂。
- **R1 正式流水线（四阶段）**：
  1. 冷启动（cold start） SFT：数千条高质量长 CoT，主要解决可读性和格式；
  2. 面向推理的 RL：GRPO + 规则奖励，另加**语言一致性奖励**；
  3. 拒绝采样 + SFT：用上一步的模型生成并筛选约 60 万条推理数据，混合约 20 万条通用数据，**从 base 重新 SFT**；
  4. 全场景 RL：可验证任务用规则奖励，开放任务用偏好 RM。
- **两个明确的负面结论**（论文里写了）：没有采用 PRM（步骤难定义、易被 hack、重训成本高）；MCTS 式搜索在语言空间里没跑通。
- **蒸馏结论**：把 R1 的轨迹 SFT 进 Qwen / Llama 小模型，效果好过在小模型上直接跑 RL。
- **对应讲次**：第 9、10 讲。**这条流水线推翻了原稿“直接对 base 跑 RL 不会有长 CoT”的说法。**


### 5.4 DeepSeek-V4（2026 年 4 月 24 日）——RL 专家 + 多教师在线策略蒸馏

- **流程**：先按领域各训一个专家（数学、代码、agent、指令跟随等），每个专家都走 **SFT → GRPO**；然后用**在线策略蒸馏（OPD）**把十几个专家合并成一个统一模型，学生在自己的 rollout 上优化对教师的反向 KL。
- **关键变化**：V3.2 里那个“混合 RL 阶段”被整体替换掉了——RL 仍在，但退到了“造专家”的位置，最终合并靠蒸馏。
- **难验证任务怎么办**：不再训练独立的标量 RM，而是用**生成式奖励模型（GRM）**评判 rubric 引导的轨迹，且 actor 自身就充当 GRM。
- **另外两个工程点**：三档思考模式（Non-think / Think High / Think Max）由不同 RL 配置的专家分别支撑；为 RL rollout 专门建了可并发数十万沙箱的执行平台。
- **为什么这是趋势而不是个例**：小米 [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780) 把多教师形式命名为 [MOPD](https://arxiv.org/abs/2606.30406)，[GLM-5](https://arxiv.org/abs/2602.15763) 把同一机制用在跨训练阶段恢复能力，NVIDIA [Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007) 用十余个专家教师，[Qwen3](https://arxiv.org/abs/2505.09388) 报告称这种蒸馏只要 RL 约十分之一的 GPU 小时且效果更好。
- **对应讲次**：第 5 讲（蒸馏三种 KL）、第 15 讲（它是否会取代混合 RL）。


### 5.5 四条流水线对照

| | InstructGPT 2022 | [Llama 2-Chat](https://arxiv.org/abs/2307.09288) 2023 | DeepSeek-R1 2025 | [DeepSeek-V4](https://arxiv.org/abs/2606.19348) 2026 |
|---|---|---|---|---|
| 阶段数 | 3 | 3 + 5 轮迭代 | 4（另有 Zero 分支） | 2 段（N 个专家 + 1 次合并） |
| 主信号 | 人类偏好 | 人类偏好 ×2 | 规则验证器为主 | 规则 + GRM，最后是教师分布 |
| RL 算法 | PPO | PPO | GRPO | GRPO（仅用于造专家） |
| 数据来源 | 旧策略 | 旧策略 + 拒绝采样 | 在线 + 自举 SFT | 在线（学生自采样） |
| 主要成本 | 人工标注 | 人工标注 + 迭代 | rollout + 验证 | 专家 RL + 全词表蒸馏 |


## 6. 方法速查（订正版）

| 方法 | 奖励来源 | 在线/离线 | 需要 RM | 需要 critic | 需要 reference |
|---|---|---|---|---|---|
| PPO | RM | 在线 | 是 | 是 | 是 |
| DPO | 偏好对（隐式） | 离线 | 否 | 否 | 是 |
| ORPO / SimPO | 偏好对 | 离线 | 否 | 否 | **否** |
| GRPO | 验证器 / RM | 在线 | 可不要 | 否 | **可不要**（DAPO 等直接去掉 KL） |
| [RLOO](https://arxiv.org/abs/2402.14740) | 验证器 / RM | 在线 | 可不要 | 否 | 视实现 |
| On-policy 蒸馏 | 教师分布 | 在线 | 否 | 否 | 教师即锚点 |

> **勘误**：原稿写“GRPO 通常需要 reference 模型做 KL 惩罚”。这在 2024 年成立，但 2025 年起的主流 RLVR 配方（DAPO、[Dr. GRPO](https://arxiv.org/abs/2503.20783)、多数开源复现）常直接去掉 KL 项，此时不需要常驻 reference，显存少一份。是否保留 KL 取决于你更怕跑飞还是更怕学不动。


## 7. 工程坑预览（第 13 讲详讲）

- 训练引擎与推理引擎的 logprob 不一致 → 重要性比失真（最隐蔽的崩溃源）。
- KL 系数 $\beta$ 难调：$\beta$ **太小**→ 偏离参考太远、语言退化；$\beta$ **太大**→ 学不动。（原稿此处把“KL 大小”与“$\beta$ 大小”说反了。）
- reward hacking：奖励曲线好看，评测不动甚至下降。
- 长度爆炸与熵坍塌（entropy collapse）。
- 评测污染与 benchmark 过拟合。
- 异步 RL 的陈旧度（staleness）失控。


## 8. 代表工作与时间线

| 年份 | 工作 | 为什么重要 |
|---|---|---|
| 2017.6 | Christiano et al., Deep RL from Human Preferences | 偏好学习的起点 |
| 2020.9 | Stiennon et al., Learning to Summarize from Human Feedback | 第一次在语言任务上跑通 RLHF 全流程 |
| 2022.3 | Ouyang et al., InstructGPT | 三阶段范式定型 |
| 2022.12 | Bai et al., [Constitutional AI](https://arxiv.org/abs/2212.08073) | 用 AI 反馈替代部分人类标注 |
| 2023.5 | Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | PRM 与过程监督 |
| 2023.5 | Rafailov et al., DPO | 把 RLHF 压成一个分类损失 |
| 2024.2 | Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300)（GRPO） | 去掉 critic 的组相对优势 |
| 2024.2 | Ahmadian et al., RLOO | “回到基础”：REINFORCE 就够了 |
| 2024.11 | Lambert et al., [Tülu 3](https://arxiv.org/abs/2411.15124) | 提出并系统化 RLVR |
| 2025.1 | DeepSeek-R1 | 纯 RL 涌现推理 + 四阶段流水线 |
| 2025.3 | DAPO / Dr. GRPO | GRPO 的偏差与稳定性修补 |
| 2025.10 | Meta, The Art of Scaling RL Compute | RL 算力与性能的 sigmoid 拟合 |
| 2026.4 | DeepSeek-V4 | RL 专家 + 多教师 OPD |

阅读顺序建议：InstructGPT → DPO → DeepSeekMath → R1 → DAPO → V4。前三篇打地基，后三篇看现状。


## 9. 自测题

**1. 为什么说 SFT 是“奖励恒为 1 的策略梯度（policy gradient）”？这个视角有什么用？**

把 $\mathcal{L}_{\text{SFT}}$ 的梯度写成 $-\mathbb{E}[\sum_t w_t\nabla\log\pi_\theta(y_t)]$ 并令 $w_t\equiv1$ 即可。用处：一旦把 $w_t$ 换成 $\mathbb{I}[\text{验证通过}]$ 就是拒绝采样微调，换成 $\hat A$ 就是 REINFORCE。所有后训练方法在这个框架下只差“样本来源 + 权重”。

**2. 同样是人类偏好，DPO 和 PPO 的本质差别在哪？**

信号相同（轴一位置相同），数据来源不同（轴二不同）。PPO 在当前策略采样的样本上优化，能纠正模型**现在**会犯的错；DPO 只在固定偏好数据上优化，数据没覆盖的行为它看不见。工程上，前者需要完整的 rollout 基础设施，后者只需要一个训练循环。

**3. 为什么 LLM RL 里几乎总取 $\gamma=1$？**

折扣因子（discount factor）的作用是处理无限期任务的收敛性，并表达“早拿到奖励更好”。LLM 的回合是有限的，且奖励通常只在结尾给；若 $\gamma<1$，相当于系统性偏好更短的回答，人为引入长度偏差（length bias）。所以取 $\gamma=1$，长度控制交给显式的长度惩罚。

**4. RLVR 的奖励来自验证器，为什么还需要人类偏好？**

验证器只覆盖可自动判定的任务。开放写作、对话语气、拒答边界、价值观权衡都没有验证器。而且即使在数学任务上，验证器只管答案对不对，不管过程是否胡编——所以现代流水线通常是“可验证部分用规则 + 不可验证部分用 RM/GRM/rubric”的混合（见 §5.3 第四阶段、§5.4）。

**5. 为什么 2026 年的旗舰模型会把“混合 RL”换成“专家 RL + 在线策略蒸馏”？**

两个原因。其一是**能力打架**：一个模型同时对数学、代码、agent、指令跟随做 RL，后一个阶段常常侵蚀前一个阶段的收益；分开训各自最优，再合并。其二是**信息密度**：RL 一条轨迹只回 1 个标量，教师分布每个 token 都给信号，合并阶段用蒸馏收敛更快、更便宜。注意这不等于“RL 没用了”——专家本身仍然是 RL 训出来的。

**6. 一个团队只有 8 张卡和一批人工标注的偏好数据，你建议走哪条路线？**

SFT → DPO（或 SimPO 以省掉 reference 显存）。理由：没有 rollout 基础设施时在线 RL 的工程成本远超收益；偏好数据是离线方法的天然输入。如果任务是数学/代码且有验证器，再考虑用小组大小的 GRPO 起步，并优先接 vLLM 做采样。


## 10. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Whiteboard — draw the RLHF pipeline and account for memory.**
*Anchored to: InstructGPT (2022). Loops: post-training research engineer (OpenAI, Anthropic, Meta GenAI, NVIDIA NeMo).*

Write the three stages, then state how many models sit in GPU memory during PPO and which of them need gradients.

*Answer key*: SFT stage = 1 model. RM stage = 1 model. PPO stage = 4 (actor, critic, reward, reference); only actor and critic carry gradients and optimizer state. For a rough memory figure use ~16 bytes/param for a trainable model under BF16 mixed precision (params + grads + two Adam moments), and ~2 bytes/param for the inference-only ones. Strong answers mention sharing a backbone, offloading, or quantizing the frozen models.

**2. Derivation — get DPO from the KL-regularized RL objective.**
*Anchored to: Rafailov et al. (2023). Loops: research scientist (Anthropic, Google DeepMind, OpenAI).*

*Answer key*: the objective `max_π E[r] − β KL(π‖π_ref)` has closed-form optimum `π*(y|x) = π_ref(y|x) exp(r(x,y)/β) / Z(x)`. Invert it: `r(x,y) = β log(π*/π_ref) + β log Z(x)`. Substitute into Bradley–Terry; `β log Z(x)` is shared by both responses to the same prompt and cancels. You are left with a logistic loss on the implicit reward difference.

**3. Debugging — reward curve rising, benchmark flat.**
*Loops: applied post-training (every lab); very common as a "tell me how you'd investigate" prompt.*

*Answer key*: state an ordered plan. (1) Check response-length and entropy curves. (2) Check the train/inference logprob gap and the importance-ratio distribution. (3) Sample outputs and read them — is the model format-gaming or repeating? (4) Stress the verifier/RM with adversarial cases. (5) Only then touch hyperparameters. The signal the interviewer wants: you suspect measurement and reward before you suspect the optimizer.

**4. Design — customer-support logs, no preference labels, goal is "resolution rate."**
*Loops: applied AI / forward-deployed (OpenAI, Anthropic, Scale AI).*

*Answer key*: first define an automatable proxy (escalation to human, repeat contact within N days, ticket closure) and validate that the proxy correlates with human judgment on a sample. Start with rejection sampling on the proxy, then rubric-based judging with a generative RM if needed. Build the held-out eval **before** training, and keep a slice that never touches any tuning.

**5. Research judgment — is RL eliciting existing capability or extending it?**
*Loops: research scientist (Anthropic, Google DeepMind, OpenAI).*

*Answer key*: do not pick a side. Cite the pass@k reversal at large k, the counter-evidence from prolonged RL runs, and the spurious-reward result showing gains on one base family that do not replicate elsewhere. Then answer the real question: what experiment would settle it for *your* task — multi-k pass@k on held-out, time-split data, replicated across two base families.

**6. Trade-off — you have 8 GPUs and a batch of human preference data. Which route?**
*Loops: startup / applied roles (Mistral, Cohere, Together, and most series-A labs).*

*Answer key*: SFT → DPO (or SimPO to drop the reference model and save memory). Justify with the second axis: online RL requires rollout infrastructure whose engineering cost exceeds its benefit at this scale. If the task is verifiable, a small-group GRPO run with vLLM for sampling is the next step up.


## 11. 本讲相对原稿的修订

| 位置 | 原稿 | 本讲 |
|---|---|---|
| 符号 | 无符号表 | 引入全局符号表（附录 A），本讲起全书统一 |
| 主线 | 单轴“奖励信号从哪来” | 双轴 + 统一梯度公式 |
| 方法表 | “GRPO 通常需要 reference” | 改为“可不要”，并说明取舍 |
| 案例 | 无 | 补四条真实流水线（2022/2023/2025/2026） |
| 讨论题 | 6 题无答案 | 6 题直接给答案 + 6 道英文面试题（含公司标注） |
| 覆盖面 | 止于 RLVR / 推理模型 | 增加 Agent RL 与 on-policy 蒸馏两列 |


## 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。2026 年的条目超出我的训练数据，来自本次检索。

- [Christiano et al., 2017 — Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)
- [Stiennon et al., 2020 — Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)
- [Ouyang et al., 2022 — InstructGPT](https://arxiv.org/abs/2203.02155)
- [Bai et al., 2022 — Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Lightman et al., 2023 — Let's Verify Step by Step（PRM）](https://arxiv.org/abs/2305.20050)
- [Rafailov et al., 2023 — Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Shao et al., 2024 — DeepSeekMath（GRPO）](https://arxiv.org/abs/2402.03300)
- [Ahmadian et al., 2024 — Back to Basics: RLOO](https://arxiv.org/abs/2402.14740)
- [Lambert et al., 2024 — Tülu 3（RLVR）](https://arxiv.org/abs/2411.15124)
- [DeepSeek-AI, 2025 — DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Yu et al., 2025 — DAPO](https://arxiv.org/abs/2503.14476)
- [Liu et al., 2025 — Understanding R1-Zero-Like Training（Dr. GRPO）](https://arxiv.org/abs/2503.20783)
- [Khatri et al., 2025 — The Art of Scaling RL Compute（ScaleRL）](https://arxiv.org/abs/2510.13786)
- [DeepSeek-AI, 2026 — DeepSeek-V4](https://arxiv.org/abs/2606.19348)

---

**下一讲**：第 5 讲　SFT、拒绝采样与蒸馏——三种 KL 方向，以及“SFT 记忆、RL 泛化”之争。
