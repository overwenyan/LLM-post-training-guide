# 附录 A　全局符号表

> **使用约定**
> 1. 本讲义的符号是**全局保留**的：一个字母在全书只承担一个含义。原稿中 `T`（模板函数 / 序列长度）、`γ`（折扣因子（discount factor） / ORPO 权重 / 长度惩罚）、`λ`（GAE 参数 / 正则权重）、`A`（优势 / assistant 位置集合）的冲突，在这里全部拆开。
> 2. 每讲开头只列**新增符号**，不重复整张表。
> 3. 论文原文与本表不一致时，正文会用脚注注明“原文记为 X”。


## A.1 序列与数据

| 符号 | 含义 | 首次出现 |
|---|---|---|
| $\mathcal{V}$ | 词表（vocabulary），$\lvert \mathcal{V}\rvert$ 通常 $3\times10^4 \sim 2\times10^5$ | 第 1 讲 |
| $x$ | prompt / 输入上下文 | 第 1 讲 |
| $y=(y_1,\dots,y_{\lvert y\rvert})$ | 一段完整回答，$y_t\in\mathcal{V}$ | 第 1 讲 |
| $y_{<t}$ | 回答的前 $t-1$ 个 token | 第 1 讲 |
| $\lvert y\rvert$ | 回答长度（token 数）。**不再用 $T$ 表示长度** | 第 1 讲 |
| $c,\ a$ | 推理链（CoT）与最终答案，$y=(c,a)$ | 第 10 讲 |
| $\mathrm{chat}(\cdot)$ | chat template 函数：把 (角色, 内容) 序列化成 token 序列 | 第 5 讲 |
| $\mathcal{M}$ | loss mask 覆盖的 token 位置集合（通常是 assistant token） | 第 5 讲 |
| $m_t\in\{0,1\}$ | 第 $t$ 个 token 是否计入损失，$m_t=\mathbb{I}[t\in\mathcal{M}]$ | 第 5 讲 |
| $\mathcal{D}$ | 数据集，下标区分用途：$\mathcal{D}_{\text{sft}},\ \mathcal{D}_{\text{pref}},\ \mathcal{D}_{\text{rl}}$ | 第 4 讲 |
| $(x,y_w,y_l)$ | 偏好三元组：$y_w$ 被偏好（winning），$y_l$ 不被偏好（losing） | 第 7 讲 |


## A.2 模型与策略

| 符号 | 含义 | 首次出现 |
|---|---|---|
| $\theta$ | 待优化的模型参数 | 第 2 讲 |
| $\pi_\theta(y\mid x)$ | 当前策略，即正在训练的 LLM；$\pi_\theta(y\mid x)=\prod_t \pi_\theta(y_t\mid x,y_{<t})$ | 第 4 讲 |
| $\pi_{\text{ref}}$ | 参考模型（KL 锚点），通常是 SFT 后的检查点 | 第 4 讲 |
| $\pi_{\text{old}}$ | 采样时用的旧策略（rollout 产生数据的那个版本） | 第 6 讲 |
| $\pi_{\text{infer}}$ | 推理引擎实际运行的策略（与 $\pi_\theta$ 数值上可能不等，见第 13 讲） | 第 13 讲 |
| $\pi_{\mathcal{T}}$ | 教师模型（蒸馏用） | 第 5 讲 |
| $V_\psi(s)$ | 价值网络 / critic，参数 $\psi$ | 第 6 讲 |
| $r_\phi(x,y)$ | 奖励模型 RM，参数 $\phi$，输出标量 | 第 7 讲 |
| $\hat r_\theta(x,y)$ | DPO 的**隐式奖励** $\beta\log\dfrac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ | 第 8 讲 |
| $r^\star(x,y)$ | 真实（不可观测）奖励 | 第 7 讲 |


## A.3 RL 量

| 符号 | 含义 | 首次出现 |
|---|---|---|
| $s_t$ | 状态；LLM 中 $s_t=(x,y_{<t})$ | 第 6 讲 |
| $a_t$ | 动作；LLM 中 $a_t=y_t$ | 第 6 讲 |
| $\tau$ | 一条轨迹（LLM 中就是一整段回答；agent 中是多轮交互序列） | 第 6 讲 |
| $R(x,y)$ | 任务奖励（可验证奖励或 RM 打分），**序列级** | 第 4 讲 |
| $\tilde r_t$ | 经过奖励塑形（reward shaping）后的**逐 token** 奖励（含 KL 项） | 第 7 讲 |
| $G_t$ | 从第 $t$ 步起的回报 | 第 6 讲 |
| $V^\pi,\ Q^\pi,\ A^\pi$ | 状态价值、动作价值、优势函数（advantage function） | 第 6 讲 |
| $\hat A_t$ | 优势估计（GAE 或组内归一化） | 第 6 讲 |
| $\hat A_i$ | 第 $i$ 个采样回答的组内优势（GRPO 族） | 第 9 讲 |
| $\rho_t(\theta)$ | token 级重要性采样（importance sampling）比 $\dfrac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\text{old}}(y_t\mid x,y_{<t})}$ | 第 6 讲 |
| $\rho^{\text{seq}}(\theta)$ | 序列级重要性比（[GSPO](https://arxiv.org/abs/2507.18071) 用，含长度归一化） | 第 9 讲 |
| $H(\pi_\theta)$ | 策略熵 | 第 9 讲 |
| $\mathrm{ver}(x,y)$ | 验证器（verifier），输出 $\{0,1\}$ 或 $[0,1]$。**与价值函数（value function） $V$ 区分** | 第 9 讲 |
| $\mathcal{E}$ | 环境（agent RL） | 第 11 讲 |
| $o_t$ | 观察（工具返回、环境反馈（environment feedback）） | 第 11 讲 |


## A.4 超参数（保留字母，勿复用）

| 符号 | 含义 | 典型取值 |
|---|---|---|
| $\beta$ | 偏离参考模型的强度。**统一含义**：RLHF 中的 KL 系数、DPO 中的温度，二者数学角色相同 | RLHF $10^{-3}\sim10^{-1}$；DPO $0.01\sim0.5$ |
| $\varepsilon$ | PPO/GRPO clip 范围（[DAPO](https://arxiv.org/abs/2503.14476) 中拆成 $\varepsilon_{\text{low}},\varepsilon_{\text{high}}$） | $0.2$（DAPO: $0.2/0.28$） |
| $\gamma$ | **折扣因子，仅此一个含义**。LLM RL 中几乎总取 $\gamma=1$ | $1$ |
| $\lambda$ | **GAE 参数，仅此一个含义** | $0.95\sim1.0$ |
| $G$ | 组大小（每个 prompt 采样几个回答） | $8\sim64$ |
| $\alpha_{\text{len}}$ | 长度惩罚系数（原稿用 $\gamma$，已改） | 任务相关 |
| $w_{\text{fmt}}$ | 格式奖励权重（原稿用 $\lambda$，已改） | $0.1\sim1$ |
| $\lambda_{\text{OR}}$ | ORPO 中 odds-ratio 项的权重（原稿用 $\gamma$，已改） | $0.1\sim1$ |
| $\delta$ | SimPO 的目标奖励间隔 margin（原稿用 $\Delta$/$\gamma$，已改） | $0.5\sim1.6$ |
| $\eta$ | 学习率 | SFT $10^{-6}\sim10^{-5}$；RL $10^{-7}\sim10^{-6}$ |
| $T_{\text{sync}}$ | 异步 RL 的参数同步周期 | 第 13 讲 |
| $C$ | 计算量（scaling law 中） | 第 2、15 讲 |


## A.5 通用记号

| 符号 | 含义 |
|---|---|
| $\mathbb{I}[\cdot]$ | 指示函数 |
| $\sigma(\cdot)$ | sigmoid，$\sigma(z)=1/(1+e^{-z})$ |
| $\mathrm{KL}(p\Vert q)$ | KL 散度；$\hat{k}_1,\hat{k}_2,\hat{k}_3$ 表示三种采样估计器（第 6 讲） |
| $\mathbb{E}$ | 期望；下标标明采样分布 |
| $\mathcal{L}$ | 损失（**最小化**）；$\mathcal{J}$ 表示目标（**最大化**） |
| $\nabla_\theta$ | 对参数求梯度 |

---


# 附录 B　缩写表

按首次出现的讲次排序。正文中每个缩写首次出现时仍会给出中英文全称。

| 缩写 | 全称 | 中文 | 讲次 |
|---|---|---|---|
| LLM | Large Language Model | 大语言模型 | 1 |
| MoE | Mixture of Experts | 混合专家 | 1 |
| [RoPE](https://arxiv.org/abs/2104.09864) | Rotary Position Embedding | 旋转位置编码 | 1 |
| GQA / MLA | Grouped-Query / Multi-head Latent Attention | 分组查询注意力 / 多头潜在注意力 | 1 |
| MTP | Multi-Token Prediction | 多 token 预测 | 2 |
| CPT | Continued Pre-Training | 继续预训练（continued pre-training, CPT） | 2、12 |
| SFT | Supervised Fine-Tuning | 监督微调 | 4 |
| RM | Reward Model | 奖励模型 | 4 |
| GRM | Generative Reward Model | 生成式奖励模型 | 7 |
| RLHF | RL from Human Feedback | 基于人类反馈的强化学习 | 4 |
| RLAIF | RL from AI Feedback | 基于 AI 反馈的强化学习 | 7 |
| CAI | [Constitutional AI](https://arxiv.org/abs/2212.08073) | 宪法式 AI | 7 |
| DPO | Direct Preference Optimization | 直接偏好优化（preference optimization） | 4 |
| IPO / KTO / ORPO / SimPO / CPO | 见第 8 讲 | DPO 家族变体 | 8 |
| RLVR | RL with Verifiable Rewards | 可验证奖励强化学习 | 4 |
| PPO | Proximal Policy Optimization | 近端策略优化 | 6 |
| GAE | Generalized Advantage Estimation | 广义优势估计 | 6 |
| GRPO | Group Relative Policy Optimization | 组相对策略优化 | 9 |
| DAPO | Decoupled Clip and Dynamic sAmpling Policy Optimization | —— | 9 |
| GSPO | Group Sequence Policy Optimization | 组序列策略优化 | 9 |
| [CISPO](https://arxiv.org/abs/2506.13585) | Clipped IS-weight Policy Optimization | 裁剪重要性权重策略优化 | 9 |
| [RLOO](https://arxiv.org/abs/2402.14740) | REINFORCE Leave-One-Out | 留一法 REINFORCE | 9 |
| ORM / PRM | Outcome / Process Reward Model | 结果 / 过程奖励（process reward）模型 | 9 |
| RFT | Rejection sampling Fine-Tuning | 拒绝采样（rejection sampling）微调 | 5 |
| RFT（另一含义） | Reinforcement Fine-Tuning | 强化微调（OpenAI 产品名，2024.12） | 12 |
| CoT | Chain-of-Thought | 思维链 | 10 |
| KD | Knowledge Distillation | 知识蒸馏 | 5 |
| OPD / [MOPD](https://arxiv.org/abs/2606.30406) | (Multi-teacher) On-Policy Distillation | （多教师）在线策略蒸馏（on-policy distillation） | 5、15 |
| TIS | Truncated Importance Sampling | 截断重要性采样 | 13 |
| PEFT | Parameter-Efficient Fine-Tuning | 参数高效微调 | 12 |
| LoRA / QLoRA | Low-Rank Adaptation / Quantized LoRA | 低秩适配 | 12 |
| RAG | Retrieval-Augmented Generation | 检索增强生成 | 12 |
| OOD | Out-of-Distribution | 分布外 | 14 |
| ECE | Expected Calibration Error | 期望校准（calibration）误差 | 14 |

> **注意 RFT 的歧义**：学术文献里 RFT 通常指 Rejection sampling Fine-Tuning（拒绝采样微调，第 5 讲）；OpenAI 2024 年 12 月发布的产品 Reinforcement Fine-Tuning 也简称 RFT（第 12 讲）。面试里被问到 RFT 时，先反问是哪一个。
