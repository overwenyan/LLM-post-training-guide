# 第 9 讲　RLVR 与 GRPO 家族：按“修了哪个毛病”来读变体

> **本讲在主线中的位置**：轴一从“人类偏好”换成“规则与验证器（verifier）”，轴二从离线换回在线。这是 2025 年以来推理模型后训练（post-training）的主力路线，也是原稿改动最大的一讲。
>
> **学完你应该能做到**
> 1. 手写 GRPO 目标，并指出式子里每一项分别在防什么问题。
> 2. 说清 [Dr. GRPO](https://arxiv.org/abs/2503.20783)、[DAPO](https://arxiv.org/abs/2503.14476)、[GSPO](https://arxiv.org/abs/2507.18071)、[CISPO](https://arxiv.org/abs/2506.13585) 各修了 GRPO 的哪一处，以及代价是什么。
> 3. 正确区分 [RLOO](https://arxiv.org/abs/2402.14740) 与 [REINFORCE++](https://arxiv.org/abs/2501.03262)（原稿把两者说反了）。
> 4. 用 PyTorch 写出带 mask 和两种聚合方式的 GRPO loss。
> 5. 面对“reward 涨、评测不动”的日志，按优先级排查。

**新增符号**：$G$（组大小）、$\hat A_i$（组内优势）、$\rho_{i,t}$（token 级重要性比）、$\rho^{\text{seq}}_i$（序列级重要性比）、$\mathrm{ver}(x,y)$（验证器）、$H(\pi_\theta)$（策略熵）、$\varepsilon_{\text{low}},\varepsilon_{\text{high}}$、$w_{\text{fmt}}$、$\alpha_{\text{len}}$。完整定义见附录 A。

---


## 1. RLVR：把奖励从“模型”换成“程序”

**定义 9.1（RLVR）**　奖励由一段可自动执行的程序给出：

$$
R(x,y)=\mathrm{ver}(x,y)\in\{0,1\}\ \text{或}\ [0,1]
$$

与 RLHF 的根本区别只有一条：**RM 是学出来的近似，验证器是写出来的规则**。这带来三个后果。

| | RM（RLHF） | 验证器（RLVR） |
|---|---|---|
| 分布外行为 | 外推，打分不可靠 | 规则照常执行，行为可预期 |
| 被 hack 的方式 | 学到 RM 的统计捷径（长度、格式、谄媚） | 找规则漏洞（数值容差、测试覆盖不足、猜答案） |
| 能否无限加压 | 不能，过优化后真实质量下降 | 可以加到验证器本身的边界为止 |
| 覆盖范围 | 任何任务 | 只覆盖可自动判定的任务 |

**适用任务**：数学（答案匹配、符号等价、数值容差）、代码（单测、编译、运行结果）、逻辑与约束满足、形式化证明（证明助手检查）、工具调用（返回状态）、结构化输出（JSON 可解析）、部分安全规则。

**不适用任务**：开放写作、对话语气、价值观权衡、主观偏好。这些仍然要回到 RM / 生成式奖励模型 / rubric（第 7、15 讲）。

> **验证器的工程细节比算法更容易出事**。至少要处理：数学答案的等价判定（`1/2` vs `0.5` vs `\frac{1}{2}`，用 sympy 做符号等价而不是字符串比较）；代码执行的超时、内存上限与沙箱隔离；单测覆盖不足导致 hard-code 就能过；截断样本（模型没写完就到 max_len）到底算错还是丢弃。最后一条尤其重要，见 §4.4 的 overlong 处理。


## 2. 奖励设计

典型的组合奖励：

$$
R(x,y)=\underbrace{\mathrm{ver}(x,y)}_{\text{正确性}}+\underbrace{w_{\text{fmt}}\,\mathbb{I}[\text{格式正确}]}_{\text{可解析}}-\underbrace{\alpha_{\text{len}}\,\mathrm{penalty}(|y|)}_{\text{控制长度}}
$$

三条经验：

1. **正确性权重必须显著大于格式权重**。否则模型会先学会“把 `\boxed{}` 写对”，再也不进步——这是最常见的 reward hacking 入门款。
2. **长度惩罚不要用线性硬惩罚**。DAPO 的软超长惩罚是更好的默认：在 `[L_max - L_cache, L_max]` 区间内线性递增惩罚，超过 `L_max` 才给满惩罚，避免模型因为“怕长”而不敢展开推理。
3. **零和一之外尽量别加中间分**。部分分（比如“步骤格式对给 0.3”）会被模型精确地套利。


## 3. GRPO：为什么去掉 critic，以及式子每一项在干什么

PPO 的 critic 在 LLM 上有三个问题：与 actor 同量级的显存、稀疏奖励下难以拟合、价值估计误差直接污染优势。GRPO 的思路是：**同一道题采样 $G$ 个答案，用组内均值当 baseline**。

对每个 prompt $x$，从 $\pi_{\text{old}}$ 采样 $\{y_1,\dots,y_G\}$，计算 $R_i=R(x,y_i)$：

$$
\hat A_i=\frac{R_i-\mathrm{mean}(R)}{\mathrm{std}(R)+\epsilon}
$$

$$
\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\min\Big(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_i\Big)\Bigg]\;-\;\beta\,\mathbb{E}\big[\hat k_3\big]
$$

逐项拆：

| 项 | 作用 | 去掉会怎样 |
|---|---|---|
| $R_i-\mathrm{mean}(R)$ | 组内 baseline，降方差且不引入偏差 | 方差爆炸，几乎训不动 |
| $\div(\mathrm{std}(R)+\epsilon)$ | 把不同难度题的梯度尺度拉平 | 简单题和难题梯度量级差很多（但见 §4.1，它本身是偏差源） |
| $\min(\cdot,\mathrm{clip}(\cdot))$ | 限制单步更新幅度（悲观下界） | 一次更新步子过大，策略崩 |
| $\frac{1}{\lvert y_i\rvert}$ | 按长度归一化每条回答的贡献 | 长回答的 token 数多，会主导梯度（但见 §4.1） |
| $\beta\hat k_3$ | 锚住参考模型 | 语言可能退化（但 RLVR 中常被直接去掉，见下） |

**三个原稿没写对/没写的实现细节**：

1. **KL 加在 loss 上，不在 reward 里**。[DeepSeekMath](https://arxiv.org/abs/2402.03300) 原文用的是 $\hat k_3=\frac{\pi_{\text{ref}}}{\pi_\theta}-\log\frac{\pi_{\text{ref}}}{\pi_\theta}-1$ 这个非负估计器，直接作为损失项。这与 PPO-RLHF 把 KL 塑形进逐 token 奖励是两种不同做法，梯度形式也不同。
2. **2025 年后的主流 RLVR 配方常常直接令 $\beta=0$**。DAPO、Dr. GRPO 及多数开源复现都去掉了 KL。理由：可验证奖励本身约束性强，而推理模型本来就要大幅偏离 SFT 分布（长 CoT 的分布离 SFT 很远），KL 反而是阻力。代价是语言退化和安全行为漂移需要靠后续阶段兜回来。
3. **$\pi_{\text{old}}$ 的 logprob 必须由训练引擎重算**，不能直接用推理引擎返回的值，否则 $\rho_{i,t}$ 从一开始就是错的（第 13 讲）。

> **案例｜DeepSeek DeepSeekMath-7B（2024 年 2 月）**
> GRPO 的出处。在 7B 规模上，用 GSM8K + MATH 的规则奖励做 RL，把 MATH 准确率从 46.8% 推到 51.7%（当时 7B 开源模型的 SOTA）。论文里明确写了“把 KL 直接加到损失里，而不是加进奖励”，以及用组内相对奖励替代价值模型的动机——这是后来整条 GRPO 家族的起点。


## 4. 变体家族：每个都在修 GRPO 的某一处


### 4.1 Dr. GRPO（2025 年 3 月，Sea AI Lab）——去掉两个偏差项

发现 GRPO 的两处归一化都会引入系统性偏差：

- **除以 $\mathrm{std}(R)$ → 难度偏差**：全组几乎都对或都错时 $\mathrm{std}$ 很小，优势被放大，等于给这些题过大的权重。
- **除以 $|y_i|$ → 长度偏差（length bias）**：同样是错误答案，写得越长，每个 token 分到的惩罚越小。于是模型学会“错就错得长一点”——这正是很多复现里“回答越训越长但正确率不动”的来源。

Dr. GRPO 的做法很直接：**两个都去掉**，改用常数归一化。同一篇论文还给出另一个结论：R1-Zero 式训练中的“自我反思”“aha”行为，在**基座模型里就已经能采样出来**，RL 主要是提高了它出现的频率。这条结论会在第 10 讲的“激发 vs 扩展”里再用。


### 4.2 DAPO（2025 年 3 月，字节 Seed × 清华 AIR）——四个工程补丁

<div>

| 技术 | 修的问题 | 做法 |
|---|---|---|
| **Clip-Higher** | 熵坍塌（entropy collapse） | 解耦上下界，$\varepsilon_{\text{low}}=0.2,\ \varepsilon_{\text{high}}=0.28$。原因：低概率 token 往上推时最先撞到 $1+\varepsilon$ 被切断，探索被系统性压制 |
| **Dynamic Sampling** | 零梯度 | 丢弃全对或全错的组（$\hat A\equiv0$），过采样后补齐 batch |
| **Token-Level Loss** | 长度偏差 | 聚合改成 $\frac{1}{\sum_i\lvert y_i\rvert}\sum_i\sum_t$，不再按回答平均 |
| **Overlong Reward Shaping** | 截断噪声 | 超长样本过滤 + 软惩罚，避免“没写完”被当成“答错” |

</div>

论文的消融非常干净，值得背下来：在 Qwen2.5-32B base 上，AIME 2024 从朴素 GRPO 的 30 分，依次加 overlong filtering（36）、clip-higher（38）、软超长惩罚（41）、token 级 loss（42）、动态采样（50），最终以一半的训练步数超过 DeepSeek-R1-Zero-Qwen-32B 的 47 分。

> **面试用法**：被问“GRPO 有什么问题”时，这张消融表是最好的答案——它把每个问题和收益都量化了。注意 token 级 loss 单独看只涨 1 分，但论文说明它的价值在于**训练稳定性和长度增长更健康**，这类“不涨分但保命”的技术是工程判断力的体现。


### 4.3 GSPO（2025 年 7 月，Qwen 团队）——把重要性比搬到序列级

问题：奖励是**序列级**给的，重要性比却是**token 级**算的，粒度不匹配；在长序列上，token 级比值的噪声会逐步累积。在 MoE 上尤其致命——一次梯度更新后，同一个输入的激活专家会有相当比例发生变化，token 级比值随之剧烈波动。

GSPO 把比值改成长度归一化的序列似然比，并在序列级做 clip：

$$
\rho^{\text{seq}}_i(\theta)=\left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\text{old}}(y_i\mid x)}\right)^{1/|y_i|},\qquad \mathcal{J}_{\text{GSPO}}=\mathbb{E}\Big[\frac{1}{G}\sum_i\min\big(\rho^{\text{seq}}_i\hat A_i,\ \mathrm{clip}(\rho^{\text{seq}}_i,1\!-\!\varepsilon,1\!+\!\varepsilon)\hat A_i\big)\Big]
$$

**工程收益**：此前 Qwen 团队为了让 MoE 的 GRPO 收敛，要用 **Routing Replay**——缓存旧策略激活的专家，在新策略上“重放”同样的路由，保证分子分母走同一套子网络。这招有效但费显存、费通信，还限制了 MoE 的实际容量。GSPO 把这套机制整个去掉了，因为序列似然对个别 token 的路由抖动不敏感。论文把它归为 [Qwen3](https://arxiv.org/abs/2505.09388) 系列模型提升的原因之一。

> 注意：clip 的**粒度**从 token 变成序列，意味着一条回答要么整体参与更新、要么整体被裁掉。后续也有工作认为 GSPO 的收益主要来自序列级 clip 这个正则效果，而非序列级重要性采样（importance sampling）本身——这是一个开放问题，适合作为研究切入点。


### 4.4 CISPO（2025 年 6 月，MiniMax-M1）——不要丢梯度，要压权重

观察：被 clip 掉的往往恰恰是**最关键的 token**。像 `However`、`Wait`、`Recheck` 这类引发反思与回溯的词，在旧策略下概率很低，一旦更新后概率上升，比值立刻超出 $1+\varepsilon$ 而被裁掉，**梯度直接归零**。长 CoT 的反思能力恰恰就长在这些 token 上。

CISPO 的做法：回到 REINFORCE 形式，把**重要性权重裁剪后用 stop-gradient 当系数**，而不是裁掉整项：

$$
\mathcal{J}_{\text{CISPO}}=\mathbb{E}\Big[\frac{1}{\sum_i |y_i|}\sum_{i,t}\mathrm{sg}\big[\mathrm{clip}(\rho_{i,t},1-\varepsilon_{\text{low}},1+\varepsilon_{\text{high}})\big]\,\hat A_i\,\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})\Big]
$$

这样每个 token 都保留梯度，只是权重被限幅。同一份技术报告还给出另一个重要的工程发现：训练与推理的概率不一致主要来自 **LM head 的精度**，把它提到 FP32 后两者的相关性大幅回升（第 13 讲会展开）。


### 4.5 RLOO 与 REINFORCE++（订正原稿）

| | RLOO（Cohere, 2024.2） | REINFORCE++（2025.1） |
|---|---|---|
| baseline | **留一法**：$b_i=\frac{1}{G-1}\sum_{j\ne i}R_j$，优势 $\hat A_i=\frac{G}{G-1}(R_i-\mathrm{mean}(R))$ | **全局 batch** 归一化（组外 baseline） |
| clip | **不带 clip**，就是 REINFORCE（作者论证 LLM RLHF 中 clip 很少生效） | **保留 PPO clip** |
| KL | 序列级 | token 级，塑形进奖励 |
| 适合场景 | 组大小适中、单轮 RLHF | 奖励噪声大、prompt 难度分布很宽 |

> **原稿错误**：原稿写“RLOO 目标类似 PPO（带 clip）”“Reinforce++ 可能去掉 PPO clip”。两者正好相反。这是面试里很容易被抓的细节。


### 4.6 家族全景

| 方法 | 比值粒度 | baseline | clip 方式 | KL | 最适合 |
|---|---|---|---|---|---|
| PPO | token | critic + GAE | token 级裁项 | 奖励塑形（reward shaping） | RM 信号、需要精细信用分配（credit assignment） |
| GRPO | token | 组内均值/标准差 | token 级裁项 | loss 中 $\hat k_3$ | 可验证奖励的通用起点 |
| Dr. GRPO | token | 组内均值（无 std） | token 级裁项 | 常去掉 | 消除长度/难度偏差 |
| DAPO | token | 组内均值 | 上下界解耦 | 去掉 | 长 CoT、大规模稳定训练 |
| GSPO | **序列** | 组内均值 | 序列级裁项 | 可选 | **MoE**、超长序列 |
| CISPO | token | 组内均值 | **裁权重不裁项** | 可选 | 保住反思类低概率 token |
| RLOO | 序列 | 留一法 | 无 | 序列级 | 简洁、单轮 RLHF |
| REINFORCE++ | token | 全局 batch | 有 | token 级 | 奖励噪声大的混合任务 |

**选型建议**：dense 模型从 GRPO + Dr. GRPO 的修正（去 std、去 $1/|y|$）起步；训练不稳或熵掉得快就加 DAPO 的四件套；模型是 MoE 就直接上 GSPO；发现反思类行为被压制再考虑 CISPO。


## 5. 熵、pass@k 与“RL 到底提升了什么”


### 5.1 熵是第一健康指标

训练早期熵快速下降是正常的（策略在收敛）；但**熵掉到接近零**意味着探索停止，之后奖励曲线会平掉。可观测的连带现象：不同采样几乎一样、pass@k 随 k 增大不再上升、输出模板化。

常见干预手段，按副作用从小到大：动态采样（丢掉零优势组）→ clip-higher → 提高采样温度 / 增大 $G$ → 数据难度重配比 → 加熵奖励（最后手段，系数稍大就会发散）。


### 5.2 pass@k 的证据要小心讲

$$
\text{pass@}k=\mathbb{E}_x\left[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
$$

- 一方（2025 年 4 月，Yue et al.）：RLVR 提升小 $k$ 的表现，但 $k$ 足够大时**基座模型反超**——说明 RL 在锐化分布，而不是扩边界。
- 另一方（2025 年 5 月，NVIDIA [ProRL](https://arxiv.org/abs/2505.24864) 等）：把 RL 训练拉长、任务域拉宽之后，出现了基座在任何 $k$ 下都解不出而 RL 后模型能解的题。
- 第三方警告（2025 年 6 月，[Spurious Rewards](https://arxiv.org/abs/2506.10947)）：在 Qwen2.5-Math 系列上，**随机奖励甚至错误奖励**都能让 MATH 分数明显上涨，但换成其他模型族就复现不出来。这说明部分“RL 收益”其实是在激活基座预训练（pre-training）里已经存在的答题模式，也提示这些基座可能有数据污染（data contamination）。

**正确的讲法**：这是一个未决问题，结论取决于任务、基座覆盖度和 RL 算力规模。做研究时，任何“RL 扩展了能力”的主张都必须报告多个 $k$ 下的 pass@k，并且至少在两个模型族上复现。

> **案例｜AI2 [Tülu 3](https://arxiv.org/abs/2411.15124)（2024 年 11 月）**
> RLVR 这个名字的出处。它把“可验证奖励”从推理专用技巧推广成流水线里的一个标准阶段：SFT → DPO → RLVR，并且开源了全部数据、代码与配方。对做研究的人，这是唯一一条能完整复现的工业级流水线，非常适合作为项目基线。


## 6. 动手：手写 GRPO loss

面试高频白板题。要点：mask、聚合方式、优势归一化开关、KL 项位置。

```python
import torch
import torch.nn.functional as F

def grpo_loss(
    logp,          # [B, L] 当前策略在已采样 token 上的 logprob（需要梯度）
    old_logp,      # [B, L] 采样时策略的 logprob（训练引擎重算，不要用推理引擎的）
    ref_logp,      # [B, L] 参考模型 logprob，beta=0 时可传 None
    rewards,       # [B]    序列级奖励，B = num_prompts * G，同一 prompt 的样本连续排列
    mask,          # [B, L] 1 表示该位置计入 loss（回答 token，不含 prompt 与 padding）
    group_size,    # G
    eps_low=0.2,
    eps_high=0.28, # DAPO 的 clip-higher；退回 GRPO 则设为 0.2
    beta=0.0,      # KL 系数，RLVR 常取 0
    scale_by_std=False,   # GRPO 原版为 True；Dr. GRPO 建议 False
    # agg="token": DAPO 式，按全 batch token 平均
    # agg="seq"  : GRPO 原版，先按句内平均
    agg="token",
):
    B, L = logp.shape
    # ---- 1. 组内优势 ----
    R = rewards.view(-1, group_size)                      # [P, G]
    adv = R - R.mean(dim=1, keepdim=True)
    if scale_by_std:
        adv = adv / (R.std(dim=1, keepdim=True) + 1e-6)   # 难度偏差来源
    # 组内优势对同一条回答的所有 token 相同
    adv = adv.reshape(B, 1)

    # ---- 2. 重要性比与裁剪（PPO 式悲观下界）----
    # 单次更新时 ratio 恒为 1，多 epoch 复用数据才会偏离
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
    pg_loss = -torch.min(unclipped, clipped)              # [B, L]

    # ---- 3. KL 惩罚：k3 估计器，直接加在 loss 上（不是加进 reward）----
    if beta > 0.0 and ref_logp is not None:
        log_u = ref_logp - logp                           # log(pi_ref / pi_theta)
        kl = torch.exp(log_u) - log_u - 1.0                # >= 0
        pg_loss = pg_loss + beta * kl

    # ---- 4. 聚合方式：这一步决定了有没有长度偏差 ----
    if agg == "token":
        loss = (pg_loss * mask).sum() / mask.sum().clamp(min=1)
    elif agg == "seq":
        per_seq = (pg_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        loss = per_seq.mean()
    else:
        raise ValueError(agg)
    return loss


def filter_zero_variance_groups(rewards, group_size):
    """DAPO 的动态采样：全对或全错的组优势恒为 0，梯度为零，应丢弃并补采。"""
    R = rewards.view(-1, group_size)
    keep = (R.max(dim=1).values - R.min(dim=1).values) > 0    # [P]
    return keep
```

**代码里最容易被追问的三点**：

1. `ratio` 在单次更新（每批数据只走一个 optimizer step）时恒等于 1，clip 完全不生效——这时 GRPO 退化成带组内 baseline 的 REINFORCE。clip 只有在一批 rollout 做多次更新，或异步 RL 导致数据陈旧时才有意义。
2. `mask` 必须排除 prompt、padding 和工具返回的 observation token。多轮场景漏掉这一条，模型会开始学着生成工具的返回值。
3. `agg` 的两种写法数值上差别不大，但长度分布偏斜时梯度分配完全不同，这就是 §4.2 里 token 级 loss 要解决的问题。


## 7. 工程检查清单

- [ ] $\pi_{\text{old}}$ 的 logprob 由训练引擎重算，与推理引擎的偏差被监控
- [ ] 零方差组被过滤（否则有效 batch 会悄悄变小）
- [ ] 熵、回答长度、重要性比分布、clip 触发比例都在看板上
- [ ] 截断样本的处理策略明确（过滤还是软惩罚），不要混进“答错”
- [ ] 验证器有超时、沙箱与资源上限；抽样人工复核它的判定
- [ ] 评测集与训练 prompt 严格隔离，另留一份未参与任何调参的集合
- [ ] 至少在两个模型族上验证结论（避免 Qwen2.5-Math 式的伪收益）


## 8. 代表工作

| 时间 | 工作 | 一句话 |
|---|---|---|
| 2024.2 | DeepSeekMath（GRPO） | 组内相对优势替代 critic |
| 2024.2 | RLOO | 留一法 baseline，无 clip |
| 2024.11 | Tülu 3 | 提出 RLVR，开源全流程 |
| 2025.1 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) / [Kimi k1.5](https://arxiv.org/abs/2501.12599) | 规则奖励驱动的大规模推理 RL |
| 2025.1 | REINFORCE++ | 全局 batch 归一化 + clip |
| 2025.3 | Dr. GRPO | 指出 std 与长度归一化的偏差 |
| 2025.3 | DAPO | 四件套 + 完整消融 + 开源系统 |
| 2025.6 | [MiniMax-M1](https://arxiv.org/abs/2506.13585)（CISPO） | 裁权重而非裁项；FP32 LM head |
| 2025.7 | GSPO | 序列级比值，免除 Routing Replay |


## 9. 自测题

**1. GRPO 为什么能去掉 critic？代价是什么？**

用同一 prompt 下 $G$ 个样本的奖励均值作为 baseline，是价值函数（value function）的蒙特卡洛替代。代价有三：每个 prompt 必须采样 $G$ 次（rollout 成本上升 $G$ 倍）；优势在整条回答上是常数，失去 token 级信用分配；组内奖励全同时梯度为零，需要动态采样兜底。

**2. 为什么说除以组内标准差会引入偏差？**

$\mathrm{std}(R)$ 反映题目难度：全组几乎都对或都错时 std 很小，优势被放大，相当于给这些“信息量最少”的题最大的梯度权重。Dr. GRPO 的建议是去掉它，改用常数归一化。

**3. Clip-Higher 为什么能缓解熵坍塌？**

优势为正时，token 概率被推高直到 $\rho>1+\varepsilon$ 就停止贡献梯度。低概率 token 的 $\rho$ 增长更快，最先撞上限，于是“探索性 token”被系统性地限制上升，而高概率 token 还能继续加强，分布越来越尖。把上界单独调大（0.28）给低概率 token 更多上升空间。

**4. 同样是可验证奖励，什么时候该用 GSPO 而不是 GRPO？**

模型是 MoE、或序列很长时。MoE 每次更新后激活专家会变，token 级比值剧烈波动，原本要靠 Routing Replay 缓存并重放路由才能收敛；GSPO 用长度归一化的序列似然比，对单 token 的路由抖动不敏感，可以省掉这套机制。dense 小模型上收益通常不明显。

**5. RLVR 训练里回答越来越长但正确率不动，你怎么查？**

按顺序：(1) 看聚合方式，$1/|y_i|$ 会让长的错误答案受罚更轻，换成 token 级聚合；(2) 看截断样本是否被当成答错（噪声奖励会鼓励模型写满上下文）；(3) 看长度惩罚是否缺失或过弱；(4) 看熵与 pass@k，区分“长但多样”和“长但重复”；(5) 抽样人工读，确认是不是在复读或自我检查空转。

**6. 为什么 RLVR 配方普遍敢把 KL 去掉，而 RLHF 不敢？**

RLVR 的奖励是程序给的，不会像 RM 那样被策略分布漂移“骗到”，所以不需要 KL 来限制模型走远；而长 CoT 本身就要求策略大幅偏离 SFT 分布，KL 是阻力。RLHF 里 RM 在分布外不可靠，去掉 KL 会迅速走向 reward hacking 与语言退化。代价是 RLVR 后模型的通用与安全行为可能漂移，需要后续阶段兜回来。


## 10. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Whiteboard — write the GRPO objective and name three differences from PPO.**
*Anchored to: DeepSeekMath (2024-02). Loops: post-training research engineer (every frontier lab; effectively table stakes in 2026).*

*Answer key*: group sampling, group-normalized advantage, clipped ratio, KL as a loss term. Three differences: baseline source (group statistics vs learned critic); advantage granularity (one constant per response vs per-token GAE); KL placement (loss term with the k3 estimator vs reward shaping). A strong answer also flags that dividing by the group std and by response length are both bias sources (Dr. GRPO).

**2. Coding — implement the GRPO loss.**
*Loops: RL infra (NVIDIA, ByteDance Seed-style teams, verl/TRL-adjacent roles).*

Support masking, both aggregation modes, and an advantage-normalization switch.

*Answer key*: see §6. Expected follow-ups: (i) "what is the ratio on the first update?" — exactly 1, so clipping does nothing unless you reuse rollouts for multiple epochs or run asynchronously; (ii) "which tokens go in the mask?" — assistant tokens only, excluding prompt, padding, and tool observations; (iii) "why filter zero-variance groups?" — their advantage is identically zero, so they silently shrink the effective batch.

**3. Diagnosis — 300 steps in: reward 0.3 → 0.75, AIME flat, entropy 0.8 → 0.05.**
*Loops: post-training research (OpenAI, Anthropic, DeepMind); a favorite because it separates people who have actually run RL from people who have read about it.*

*Answer key*: hypothesis one is entropy collapse plus verifier gaming. Verify by sampling outputs, computing pass@16 (if it is flat while pass@1 rises, the policy is sharpening rather than improving), and cross-checking the verifier with a second implementation. Fix order: dynamic sampling and clip-higher first; entropy bonus last, because it is the easiest way to blow up the run.

**4. Design — build an RLVR setup for SQL generation.**
*Loops: applied AI / enterprise (Databricks, Snowflake, Google Cloud, OpenAI solutions).*

*Answer key*: the verifier executes the query against a read-only replica and compares result sets, not SQL strings. Handle execution timeouts, row-order insensitivity, and empty results. Anticipate hacking: a model can emit `LIMIT 0`-style queries that match empty ground truths, so you need multiple test databases and negative cases. Mention the safety boundary — no writes, no DDL, sandboxed credentials.

**5. Paper review — "our method beats GRPO by 6 points on AIME." What do you push on?**
*Loops: research scientist (frontier labs); often used as a live paper-discussion round.*

*Answer key*: same base model? matched RL compute, not just matched steps? how many k values of pass@k are reported? AIME has 30 problems — was it avg@32 or a single sample, and at what temperature? is the training data disjoint from the eval? does the gain replicate on a second base family?

**6. Concept — why can RLVR recipes drop the KL term when RLHF cannot?**
*Anchored to: DAPO (2025) vs [InstructGPT](https://arxiv.org/abs/2203.02155) (2022). Loops: post-training research.*

*Answer key*: a program-based reward does not degrade as the policy drifts, whereas a learned RM becomes unreliable off-distribution, so RLHF needs the anchor to prevent over-optimization. Long-CoT training also *requires* moving far from the SFT distribution, which makes the KL an obstacle. The cost of dropping it is drift in general and safety behavior, which must be recovered in a later stage.


## 11. 本讲相对原稿的修订

| 位置 | 原稿 | 本讲 |
|---|---|---|
| GRPO 的 KL | “加在奖励或损失里”，用 log 比值 | 明确为 loss 项 + $\hat k_3$ 估计器；补充“2025 年后常直接去掉” |
| “GRPO 需要 reference 吗” | “通常需要” | 改为可选，并说明取舍 |
| GRPO 归一化 | 未提偏差 | 补 Dr. GRPO 的 std 与长度双偏差 |
| 变体覆盖 | 仅 RLOO、Reinforce++ | 补 Dr. GRPO、DAPO、GSPO、CISPO，并按“修了什么”组织 |
| RLOO / REINFORCE++ | 描述互换（说反） | 订正：RLOO 无 clip，REINFORCE++ 有 clip + 全局归一化 |
| pass@k 结论 | “证据更支持采样效率” | 改为未决争议，给出三方证据与研究方法建议 |
| 代码 | 无 | 补可运行的 GRPO loss 与动态采样过滤 |
| 案例 | 无 | 补 DeepSeekMath 2024.2、Tülu 3 2024.11、DAPO 2025.3、CISPO 2025.6、GSPO 2025.7 |


## 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。2026 年的条目超出我的训练数据，来自本次检索。

- [Shao et al., 2024 — DeepSeekMath（GRPO 出处）](https://arxiv.org/abs/2402.03300)
- [Ahmadian et al., 2024 — Back to Basics: RLOO](https://arxiv.org/abs/2402.14740)
- [Lambert et al., 2024 — Tülu 3（RLVR）](https://arxiv.org/abs/2411.15124)
- [Hu, 2025 — REINFORCE++](https://arxiv.org/abs/2501.03262)
- [DeepSeek-AI, 2025 — DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Kimi Team, 2025 — Kimi k1.5](https://arxiv.org/abs/2501.12599)
- [Yu et al., 2025 — DAPO（四件套与完整消融）](https://arxiv.org/abs/2503.14476)
- [Liu et al., 2025 — Understanding R1-Zero-Like Training（Dr. GRPO）](https://arxiv.org/abs/2503.20783)
- [Yue et al., 2025 — Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?](https://arxiv.org/abs/2504.13837)
- [Liu et al., 2025 — ProRL](https://arxiv.org/abs/2505.24864)
- [MiniMax, 2025 — MiniMax-M1（CISPO、FP32 LM head）](https://arxiv.org/abs/2506.13585)
- [Shao et al., 2025 — Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947)
- [Zheng et al., 2025 — GSPO](https://arxiv.org/abs/2507.18071)
- [Khatri et al., 2025 — The Art of Scaling RL Compute（ScaleRL）](https://arxiv.org/abs/2510.13786)
- [verl — DAPO 复现配方与配置](https://verl.readthedocs.io/en/latest/algo/dapo.html)

---

**下一讲**：第 10 讲　推理模型——R1-Zero 到底证明了什么、思考预算（thinking budget）怎么控制、以及“激发还是扩展”的完整证据链。
