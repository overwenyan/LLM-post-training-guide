# 第 6 讲　RL 基础：策略梯度、baseline、GAE、clip，以及 KL 的三种估计器

> **本讲在主线中的位置**：从这一讲起，权重 $w_t$ 不再是常数，而是“这个动作比平均水平好多少”。本讲只讲机制，不讲具体配方——RLHF 在第 7 讲，RLVR 在第 9 讲。
>
> **学完你应该能做到**
> 1. 推导策略梯度（policy gradient）定理，并证明减去 baseline 不引入偏差。
> 2. 说清 GAE 的 $\lambda$ 在权衡什么，以及为什么 LLM 里 $\gamma=1$。
> 3. 解释 PPO 的 clip 为什么能稳住训练，以及 clip 生效时梯度发生了什么。
> 4. 区分 KL 的三种采样估计器，以及“KL 放 reward 里”和“放 loss 里”的差别。

**新增符号**：$s_t,a_t,\tau$、$G_t$、$V^\pi,Q^\pi,A^\pi$、$\hat A_t$、$\rho_t$、$\gamma,\lambda,\varepsilon$、$\hat k_1,\hat k_2,\hat k_3$。见附录 A。

---


## 1. MDP 与 LLM 的对应

**定义 6.1（马尔可夫决策过程）**　$\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma)$，策略 $\pi(a\mid s)$ 给出状态 $s$ 下选动作 $a$ 的概率。

| MDP 概念 | LLM 中是什么 |
|---|---|
| 状态 $s_t$ | $(x,y_{<t})$：prompt 加已生成的 token |
| 动作 $a_t$ | 下一个 token $y_t$ |
| 动作空间 $\mathcal{A}$ | 词表 $\mathcal{V}$，$10^4\sim10^5$ |
| 转移 $P$ | **确定性拼接** $s_{t+1}=s_t\oplus a_t$ |
| 奖励 $r_t$ | 中间通常为 0，末尾给 $R(x,y)$ |
| 终止 | 生成 EOS 或达到最大长度 |
| 策略 $\pi_\theta$ | 语言模型本身 |
| 轨迹（trajectory） $\tau$ | 一整段回答 |

**回报与价值**：

$$
G_t=\sum_{k=t}^{|y|}\gamma^{k-t}r_k,\quad V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s],\quad Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]
$$

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

优势 $A^\pi$ 的含义：**在状态 $s$ 下选 $a$，比按当前策略随便走好多少**。这是后面所有算法的核心量。


### 为什么 LLM 里 $\gamma=1$

折扣因子（discount factor）有两个作用：保证无限期任务的回报收敛，以及表达“早拿到奖励更好”。LLM 的回合是有限的，奖励通常只在末尾给。若取 $\gamma<1$，末尾奖励折回前面的 token 时会被指数衰减，等价于**系统性地偏好短回答**——人为引入长度偏差（length bias）。所以标准做法是 $\gamma=1$，长度控制交给显式的长度惩罚（第 9 讲）。


## 2. 目标函数

$$
\mathcal{J}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\big[R(x,y)\big]-\beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\Vert\,\pi_{\text{ref}}(\cdot\mid x)\big)
$$

两项各司其职：第一项要奖励高，第二项要别跑太远。$\beta$ 控制这个权衡，它在 DPO 里会以“温度”的身份再次出现（第 8 讲）——两者数学角色相同，这也是附录 A 把它们统一成一个符号的原因。


## 3. 策略梯度定理

目标对参数求梯度时，麻烦在于**采样分布本身依赖 $\theta$**。用 score function 技巧（也叫 log-derivative trick）：

$$
\nabla_\theta p_\theta(\tau)=p_\theta(\tau)\,\nabla_\theta\log p_\theta(\tau)
$$

于是

$$
\nabla_\theta\mathcal{J}(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_{t}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A^\pi(s_t,a_t)\Big]
$$

**关键一步**：轨迹概率 $p_\theta(\tau)=p(s_0)\prod_t\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$ 取对数后，环境转移项 $P$ 与 $\theta$ 无关，求梯度时直接消失——**所以我们不需要知道环境模型**。在 LLM 里 $P$ 本来就是确定性拼接，这一项更是平凡。

**直觉**：$\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ 指向“让这个 token 更可能被生成”的方向；$A^\pi$ 决定往这个方向走多远、往哪边走。这正是附录 C 统一公式里的 $w_t$。


## 4. REINFORCE 与 baseline


### 4.1 朴素版本方差极大

$$
\nabla_\theta\mathcal{J}=\mathbb{E}\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\Big]
$$

$G_t$ 在长序列上波动巨大，且量级依赖奖励的绝对尺度（回忆第 7 讲：BT 模型只约束奖励差，绝对尺度不可辨识）。


### 4.2 baseline 为什么无偏

引入任意只依赖状态的 $b(s_t)$：

$$
\mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)\big]=b(s)\sum_a\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s)=b(s)\,\nabla_\theta\underbrace{\sum_a\pi_\theta(a\mid s)}_{=1}=0
$$

**这个三行推导是面试必考**。结论：减去任何与动作无关的量都不改变梯度期望，但能显著降低方差。

常用 baseline：
- 学习一个价值函数（value function） $V_\psi(s)$ → actor-critic（§5）
- 同一 prompt 下的组内平均奖励 → GRPO（第 9 讲）
- 留一法均值 → [RLOO](https://arxiv.org/abs/2402.14740)（第 9 讲）
- 全局 batch 移动平均 → [REINFORCE++](https://arxiv.org/abs/2501.03262)（第 9 讲）

**注意**：$b$ 必须与动作无关。用“同一条轨迹自己的奖励”当 baseline 会引入偏差，这也是 RLOO 要做“留一”的原因。


## 5. Actor-Critic 与 GAE

用 critic 估计价值，TD 误差为

$$
\delta_t=r_t+\gamma V_\psi(s_{t+1})-V_\psi(s_t)
$$

**定义 6.2（GAE）**

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l\ge 0}(\gamma\lambda)^l\,\delta_{t+l}
$$

$\lambda$ 在偏差与方差之间插值：

| $\lambda$ | 等价于 | 偏差 | 方差 |
|---|---|---|---|
| 0 | 单步 TD：$\delta_t$ | 大（完全信任 critic） | 小 |
| 1 | 蒙特卡洛：$G_t-V_\psi(s_t)$ | 小 | 大 |
| 0.95 左右 | 折中 | —— | —— |

在 LLM 里的特殊性：奖励只在末尾，$\lambda$ 决定这个末尾信号如何沿着几千个 token 往回传播。$\lambda$ 偏小时，信用几乎传不到开头的 token；$\lambda=1$ 时整条回答的每个 token 拿到相同的优势（因为中间奖励全是 0），退化成 REINFORCE 加 baseline。

**critic 在 LLM 上为什么难训**：它要为每个 token 位置预测未来回报，但训练信号只有末尾一个标量；网络规模与 actor 相当，显存翻倍；策略一直在变，价值目标也跟着漂。这是 GRPO 去掉 critic 的直接动机。

> **案例｜critic 并没有死：VAPO / VC-PPO（字节 Seed，2025 年 4 月）**
> 在大家纷纷转向无 critic 方法时，字节的一组工作反其道而行：通过价值网络预训练（先用固定策略的回报把 critic 训到位再开始 RL）、actor 与 critic 解耦的 $\lambda$、长度自适应的 GAE 等手段，让基于价值的方法在 32B 规模的数学推理上超过了无 critic 基线。**结论不是“critic 没用”，而是“critic 很难训，难到大多数团队选择绕开”**——这是工程判断与算法优劣的区别，面试里值得说清。


## 6. PPO：为什么要 clip


### 6.1 重要性采样

策略梯度是 on-policy 的：采一批数据只能更新一次。为了复用数据，引入重要性比

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)},\qquad \mathcal{J}=\mathbb{E}_{\pi_{\text{old}}}[\rho_t\hat A_t]
$$

问题：当 $\pi_\theta$ 偏离 $\pi_{\text{old}}$ 较多时，$\rho_t$ 的方差爆炸，单个样本就能主导整个梯度。


### 6.2 Clip 目标

$$
\mathcal{L}^{\text{clip}}=-\mathbb{E}\Big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\hat A_t\big)\Big]
$$

**逐情形理解**（面试常追问）：

| 情形 | 行为 |
|---|---|
| $\hat A_t>0$，$\rho_t<1+\varepsilon$ | 正常推高该 token 概率 |
| $\hat A_t>0$，$\rho_t>1+\varepsilon$ | 取到裁剪项，**梯度为零**——已经推够了 |
| $\hat A_t<0$，$\rho_t>1-\varepsilon$ | 正常压低 |
| $\hat A_t<0$，$\rho_t<1-\varepsilon$ | 梯度为零——已经压够了 |

$\min$ 让目标成为真实目标的**悲观下界**：优化一个下界，保证不会因为重要性比失真而走得太远。

**两个容易被忽略的推论**：

1. 如果一批 rollout 只做一次更新，$\rho_t\equiv1$，clip 完全不生效。此时 PPO 退化为带 baseline 的策略梯度。clip 只在多 epoch 复用或异步 RL 造成数据陈旧时才有意义。
2. clip 生效时梯度归零，意味着**这个 token 这一步不再学习**。第 9 讲里 [DAPO](https://arxiv.org/abs/2503.14476) 的 clip-higher（把上界单独调大）和 [CISPO](https://arxiv.org/abs/2506.13585)（裁权重而不裁项）都是在处理这个“梯度被丢掉”的副作用。

完整的 PPO 损失：

$$
\mathcal{L}=\mathcal{L}^{\text{clip}}+c_1\,\mathbb{E}\big[(V_\psi(s_t)-\hat R_t)^2\big]-c_2\,H(\pi_\theta)
$$


## 7. KL 的三种估计器（原稿缺失）

我们要的是 $\mathrm{KL}(\pi_\theta\Vert\pi_{\text{ref}})=\mathbb{E}_{y\sim\pi_\theta}\big[\log\frac{\pi_\theta}{\pi_{\text{ref}}}\big]$，但在训练里只有采样的 token。记 $u=\dfrac{\pi_{\text{ref}}(y_t\mid s_t)}{\pi_\theta(y_t\mid s_t)}$：

| 估计器 | 式子 | 无偏 | 非负 | 方差 |
|---|---|---|---|---|
| $\hat k_1$ | $-\log u$ | 是 | **否**（单样本可能为负） | 大 |
| $\hat k_2$ | $\tfrac12(\log u)^2$ | 否 | 是 | 小 |
| $\hat k_3$ | $u-\log u-1$ | 是 | 是 | 中 |

$\hat k_3$ 兼顾无偏与非负，是 GRPO 的默认选择。

**一个容易踩的坑**：$\hat k_3$ 对 **KL 的值**无偏，但常见实现中它对**梯度**并不无偏（因为 $u$ 里含 $\pi_\theta$，自动求导会产生额外项）。如果你发现 KL 曲线与预期不符，先检查这里。

```python
def kl_estimators(logp_theta, logp_ref):
    log_u = logp_ref - logp_theta        # log(pi_ref / pi_theta)
    k1 = -log_u                          # 无偏，可能为负
    k2 = 0.5 * log_u.pow(2)              # 有偏，非负
    k3 = log_u.exp() - log_u - 1.0       # 无偏且非负（GRPO 用）
    return k1, k2, k3
```


### KL 放 reward 里，还是放 loss 里？

| 做法 | 形式 | 谁在用 | 效果差别 |
|---|---|---|---|
| 塑形进逐 token 奖励 | $\tilde r_t=-\beta\,\hat k_1$，末尾再加 $R(x,y)$ | PPO-RLHF | KL 会经过 GAE 传播，与任务奖励一起参与优势计算 |
| 直接加在损失上 | $\mathcal{L}+\beta\,\hat k_3$ | GRPO | 不进入优势，作用更直接、更容易解释 |

两者不等价。前者中 KL 的影响会被 critic 与 GAE“加工”一遍；后者是纯粹的正则项。知道这个区别，才能读懂不同代码库里 KL 曲线为什么长得不一样。


## 8. 回到统一视角

把本讲的结果代回附录 C 的公式：

$$
\nabla_\theta\mathcal{L}=-\mathbb{E}\Big[\sum_t w_t\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\Big]
$$

| 方法 | $w_t$ |
|---|---|
| SFT | $1$ |
| 拒绝采样（rejection sampling） | $\mathbb{I}[\text{通过验证}]$ |
| REINFORCE | $G_t$ |
| REINFORCE + baseline | $G_t-b(s_t)$ |
| Actor-Critic / PPO | $\hat A_t$（GAE），再乘被 clip 的 $\rho_t$ |
| GRPO | 组内归一化的 $\hat A_i$ |

**一句话**：整条技术路线的演进，就是在不断改进 $w_t$ 的**估计方式**（更低方差、更少偏差）和**获取成本**（要不要 critic、要不要 RM）。


## 9. 从 PPO 到 GRPO 的动机

| PPO 的负担 | GRPO 的处理 |
|---|---|
| critic 与 actor 同量级，显存翻倍 | 去掉 critic |
| 稀疏奖励下 critic 难拟合 | 用组内多个样本的奖励做蒙特卡洛 baseline |
| GAE 的 $\lambda$ 要调 | 不需要 GAE，整条回答共享一个优势 |
| 需要 RM | 可验证任务直接用验证器（verifier） |

代价：每个 prompt 要采样 $G$ 次；失去 token 级信用分配（credit assignment）。完整讨论在第 9 讲。


## 10. 自测题

**1. 写出策略梯度定理的推导，并说明为什么不需要环境模型。**

用 $\nabla p_\theta(\tau)=p_\theta(\tau)\nabla\log p_\theta(\tau)$。展开 $\log p_\theta(\tau)=\log p(s_0)+\sum_t[\log\pi_\theta(a_t|s_t)+\log P(s_{t+1}|s_t,a_t)]$，其中初始分布与转移项都与 $\theta$ 无关，求梯度后消失，只剩策略项。所以不需要知道 $P$。

**2. 证明减去 baseline 不引入偏差，并说明它必须满足什么条件。**

$\mathbb{E}_a[\nabla\log\pi_\theta(a|s)b(s)]=b(s)\nabla\sum_a\pi_\theta(a|s)=b(s)\nabla 1=0$。条件是 $b$ 只能依赖状态、不能依赖当前动作。用同一条轨迹自己的奖励当 baseline 会破坏这个条件，所以 RLOO 要做留一。

**3. GAE 的 $\lambda$ 取 0 和取 1 分别对应什么？LLM 里怎么选？**

$\lambda=0$ 是单步 TD，完全信任 critic，偏差大方差小；$\lambda=1$ 是蒙特卡洛，偏差小方差大。LLM 中奖励在末尾，$\lambda$ 小会导致信用传不到开头；实践中常取接近 1 的值，而一旦取 1 且中间奖励为 0，每个 token 的优势就相同，与无 critic 方法的差别只剩 baseline 的来源。

**4. PPO 的 clip 在什么情况下完全不生效？**

一批 rollout 只做一次梯度更新时，$\rho_t\equiv1$，落在 $[1-\varepsilon,1+\varepsilon]$ 内，裁剪不触发。此时 PPO 等价于带 baseline 的策略梯度。clip 的意义在于多 epoch 复用数据或异步 RL 导致 $\pi_\theta$ 与 $\pi_{\text{old}}$ 拉开时。

**5. $\hat k_3$ 相比 $\hat k_1$ 好在哪？有什么陷阱？**

$\hat k_1$ 虽无偏但单样本可能为负，作为正则项会出现"负 KL"的怪异曲线；$\hat k_3$ 同时无偏且非负，数值更稳。陷阱是它对值无偏，但常见实现对梯度不无偏。

**6. 为什么 LLM 的 RL 不从随机策略开始就一定要加 KL 约束？**

因为初始策略已经很强，RL 的作用是校准（calibration）而非从零学习。没有锚点时，模型会为了奖励快速牺牲语言质量、安全行为和通用能力（reward hacking 的温床）。但在可验证奖励且需要大幅改变行为分布的场景（长 CoT），KL 反而是阻力，所以 RLVR 常把它去掉（第 9 讲）。


## 11. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Whiteboard derivation — policy gradient theorem and baseline unbiasedness.**
*Loops: RL research (OpenAI, Anthropic, Google DeepMind, NVIDIA); this is the single most common RL whiteboard item.*

*Answer key*: use `∇p_θ(τ) = p_θ(τ)∇log p_θ(τ)`; expand `log p_θ(τ)` and note that the initial-state and transition terms are $\theta$-independent and vanish — hence no environment model is needed. For the baseline: `E_a[∇log π(a|s) b(s)] = b(s)∇Σ_a π(a|s) = b(s)∇1 = 0`. State the condition explicitly: `b` may depend on the state only, never on the sampled action — which is why RLOO leaves the sample out of its own baseline.

**2. Concept — $\gamma$ versus $\lambda$.**
*Loops: RL research engineer (most labs).*

*Answer key*: $\gamma$ is the discount in the return; $\lambda$ interpolates GAE between one-step TD ($\lambda$=0, biased, low variance) and Monte Carlo ($\lambda$=1, unbiased, high variance). In LLM RL $\gamma$=1, because with a terminal-only reward any $\gamma$<1 systematically favors shorter responses — an artificial length bias. $\lambda$ is usually near 1 because the only reward signal has to travel thousands of tokens backwards.

**3. Coding — implement GAE.**
*Loops: RL infra (NVIDIA NeMo-RL, Meta, Anthropic).*

Given `rewards`, `values`, `mask`, return advantages and returns.

*Answer key*: iterate backwards, `delta_t = r_t + γ·V_{t+1}·mask_{t+1} − V_t`, `A_t = delta_t + γλ·A_{t+1}·mask_{t+1}`, `returns = A + V`. Watch the terminal position and padded positions. Mention whitening advantages before the policy loss and why (scale invariance of the BT-derived reward).

**4. Depth probe — PPO's clip zeroes the gradient. What breaks, and how has the field responded?**
*Anchored to: DAPO and CISPO (2025). Loops: post-training research (frontier labs).*

*Answer key*: once the ratio leaves the clip range, that token contributes no gradient. Low-probability tokens hit the upper bound first, so exploratory tokens are suppressed preferentially → entropy collapse. Responses: decoupled clip bounds (clip-higher), or clipping the importance *weight* instead of dropping the term (CISPO), which preserves gradients on reflective tokens like "However" / "Wait".

**5. Pushback — "critics are useless for LLM RL, always use GRPO."**
*Anchored to: VAPO / VC-PPO (ByteDance Seed, 2025). Loops: research scientist (frontier labs); tests whether you distinguish engineering difficulty from algorithmic merit.*

*Answer key*: the critic is hard to train (terminal-only signal, actor-scale memory, moving target), not useless — value-based recipes with value pretraining and decoupled $\lambda$ have beaten critic-free baselines at 32B scale on math. The real decision variables are token-level credit assignment needs, memory budget, and team capacity.

**6. Concept — the three KL estimators.**
*Anchored to: GRPO's use of k3. Loops: post-training engineer (DeepSeek-style open labs, Hugging Face TRL, NVIDIA).*

*Answer key*: `k1 = −log u` unbiased but can be negative; `k2 = ½(log u)²` non-negative but biased; `k3 = u − log u − 1` unbiased *and* non-negative, hence GRPO's choice. Caveat worth volunteering: k3 is unbiased for the value, but in the usual autograd implementation it is not an unbiased estimator of the gradient. Also distinguish putting KL into the per-token reward (PPO-RLHF, passes through GAE) from adding it to the loss (GRPO, pure regularizer).


## 12. 本讲相对原稿的修订

| 位置 | 原稿 | 本讲 |
|---|---|---|
| baseline 无偏性 | 给出结论 | 补三行推导，并强调"只能依赖状态"这一条件 |
| 策略梯度定理 | 给出式子 | 补 score function 推导与"为什么不需要环境模型" |
| $\gamma=1$ | 讨论题里问了 | 正文给出理由（避免人为长度偏差） |
| KL 估计器 | 未提 | 补 $\hat k_1/\hat k_2/\hat k_3$ 对照、代码与梯度陷阱 |
| KL 的位置 | 未区分 | 补"放 reward"与"放 loss"的差别 |
| clip | 只说"限制更新幅度" | 补四种情形表、悲观下界解释、以及"梯度归零"的副作用 |
| critic | "难训，所以 GRPO 去掉" | 补 VAPO 一类反例，区分"难训"与"无用" |
| 讨论题 | 10 题无答案 | 6 题直接给答案 + 6 道英文面试题（含公司标注） |


## 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。2026 年的条目超出我的训练数据，来自本次检索。

- [Schulman et al., 2015 — TRPO](https://arxiv.org/abs/1502.05477)
- [Schulman et al., 2015 — GAE](https://arxiv.org/abs/1506.02438)
- [Schulman et al., 2017 — PPO](https://arxiv.org/abs/1707.06347)
- [Schulman — Approximating KL Divergence（k1/k2/k3 的出处）](http://joschu.net/blog/kl-approx.html)
- [Shao et al., 2024 — DeepSeekMath（GRPO，KL 作为损失项 + k3）](https://arxiv.org/abs/2402.03300)
- [Ouyang et al., 2022 — InstructGPT（PPO-ptx 与逐 token KL 塑形）](https://arxiv.org/abs/2203.02155)

---

**下一讲**：第 7 讲　奖励建模与 RLHF/PPO——Bradley-Terry、RM 过优化、生成式奖励模型，以及 GPT-4o 谄媚事件的教训。
