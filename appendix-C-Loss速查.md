# 附录 C　Loss 速查：每个损失到底在优化什么

> 这份附录是全书的“一页纸”。每个 loss 都按同一套**三问**拆解：
> 1. **优化什么**：目标函数、最优解长什么样。
> 2. **梯度在推谁**：哪些 token 的概率被推高、哪些被压低，权重多大。
> 3. **什么时候会坏**：退化解、被 hack 的方式、常见故障。
>
> 符号见附录 A。

---


## C.0 统一视角：所有后训练 loss 都是“加权的对数似然梯度”

绝大多数后训练（post-training）方法的梯度都能写成同一个形式：

$$
\nabla_\theta \mathcal{L} \;=\; -\,\mathbb{E}_{(x,y)\sim \text{某个来源}}\left[\sum_{t} w_t\,\nabla_\theta \log \pi_\theta(y_t\mid x,y_{<t})\right]
$$

方法之间只差两件事：**$y$ 从哪来**（谁生成的训练样本），和**权重 $w_t$ 是多少**（这个 token 该被推高还是压低、推多狠）。

| 方法 | $y$ 的来源 | 权重 $w_t$ | 数据新鲜度 |
|---|---|---|---|
| 预训练（pre-training） / SFT | 固定语料、人类或教师示范 | $1$（mask 内） | 离线 |
| 硬标签蒸馏 | 教师采样 | $1$ | 离线 |
| 拒绝采样（rejection sampling）微调 RFT | **自己**采样 | $\mathbb{I}[\text{通过验证}]$ | 半在线 |
| REINFORCE / [RLOO](https://arxiv.org/abs/2402.14740) / GRPO | 自己采样 | $\hat A$（可正可负） | 在线 |
| PPO | 旧策略采样 | $\hat A_t\cdot\rho_t$（被 clip） | 近在线 |
| DPO 家族 | 离线偏好对 | $\pm\beta\,\sigma(\hat r_l-\hat r_w)$ | 离线 |
| On-policy 蒸馏 | 自己采样 | $\log \pi_{\mathcal{T}}(y_t)-\log\pi_\theta(y_t)$ | 在线 |

**一句话记忆：SFT 是“奖励恒为 1 的离策略梯度（policy gradient）”，RL 是“奖励可正可负的在线梯度”，蒸馏是“奖励逐 token 给的在线梯度”。**

三者的信息密度依次上升：RL 一条轨迹（trajectory）只回来一个标量，蒸馏一条轨迹回来 $|y|\times|\mathcal{V}|$ 个数。这解释了为什么 2025–2026 年多家实验室用 on-policy 蒸馏替代混合 RL 阶段（第 15 讲）。

---


## C.1 SFT（监督微调）

$$
\mathcal{L}_{\text{SFT}}(\theta)=-\mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{sft}}}\left[\sum_{t\in\mathcal{M}} \log\pi_\theta(y_t\mid x,y_{<t})\right]
$$

- **优化什么**：最小化 $\mathbb{E}_x\,\mathrm{KL}\big(p_{\text{data}}(\cdot\mid x)\,\Vert\,\pi_\theta(\cdot\mid x)\big)$ + 常数。这是**前向 KL**，性质是 *mode covering*：数据里出现过的模式，模型都要给非零概率，因此数据里的噪声也会被照单全收。
- **梯度在推谁**：只推高专家 token 的概率，never 压低任何东西（softmax 的归一化间接压低其他 token）。权重恒为 1，**与这条数据好坏无关**。
- **什么时候会坏**：
  - 数据里有错，模型照学（前向 KL 的必然结果）。
  - 暴露偏差（exposure bias）：训练看真前缀，推理看自己的前缀。
  - mask 错位 → 学会生成用户话术 / 不会停（EOS 未计入 loss）。
  - 数据分布单一 → 灾难性遗忘（catastrophic forgetting）。


## C.2 拒绝采样微调 RFT / STaR

$$
\mathcal{L}_{\text{RFT}}=-\mathbb{E}_{x,\;y\sim\pi_{\theta_{\text{old}}}(\cdot\mid x)}\Big[\mathbb{I}\big[\mathrm{ver}(x,y)=1\big]\sum_{t}\log\pi_\theta(y_t\mid x,y_{<t})\Big]
$$

- **优化什么**：在“自己能生成且验证通过”的分布上做最大似然。等价于**只保留正优势样本的策略梯度**（把 $\hat A$ 换成 0/1）。
- **梯度在推谁**：只推高通过验证的轨迹，失败样本被直接丢弃、不提供任何负向信号。
- **什么时候会坏**：只有正样本 → 无法学会“不要这样答”；多轮迭代后多样性下降、过拟合自身输出；验证器（verifier）有漏洞就直接学漏洞。


## C.3 蒸馏：三种 KL 方向

| 方式 | 目标 | $y$ 来自 | 性质 |
|---|---|---|---|
| 硬标签（序列级） | $-\sum_t\log\pi_\theta(y_t)$，$y\sim\pi_{\mathcal{T}}$ | 教师 | 最简单，黑盒可用（R1-Distill 就是这一种） |
| 软标签（前向 KL） | $\sum_t\mathrm{KL}\big(\pi_{\mathcal{T}}(\cdot\mid s_t)\Vert\pi_\theta(\cdot\mid s_t)\big)$ | 教师 | mode covering，需要 logits |
| **On-policy 蒸馏（反向 KL）** | $\sum_t\mathrm{KL}\big(\pi_\theta(\cdot\mid s_t)\Vert\pi_{\mathcal{T}}(\cdot\mid s_t)\big)$ | **学生自己** | mode seeking，学生在自己的分布上被逐 token 纠错 |

反向 KL 写成策略梯度形式，逐 token 的“奖励”就是

$$
w_t \;=\; \log\pi_{\mathcal{T}}(y_t\mid s_t)-\log\pi_\theta(y_t\mid s_t)
$$

- **优化什么**：让学生在**自己会走到的状态上**贴近教师。它避开了离线蒸馏的分布不匹配（教师的轨迹学生走不到）。
- **梯度在推谁**：教师比学生更喜欢的 token 被推高，学生比教师更自信的 token 被压低。每个 token 都有信号，不需要等到序列结束。
- **什么时候会坏**：学生上限被教师锁死；反向 KL 是 mode seeking，多样性会收缩；需要教师 logits（同词表、同 tokenizer）。


## C.4 奖励模型（Bradley–Terry）

$$
\mathcal{L}_{\text{RM}}(\phi)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{\text{pref}}}\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big]
$$

- **优化什么**：把“哪个更好”的**序**拟合出来。注意 BT 模型只约束**奖励差**，$r_\phi$ 的绝对尺度和平移不可辨识——这就是 RL 阶段必须对奖励做归一化的根本原因。
- **梯度在推谁**：权重是 $\sigma(r_\phi(y_w)-r_\phi(y_l))$ 反过来的量，即模型判反的样本梯度更大（天然难例挖掘）。
- **什么时候会坏**：
  - 长度是最容易学到的捷径特征 → 长度偏差（length bias）。
  - 分布外打分不可靠：策略跑远了，RM 就在外推。
  - 过优化：优化压力越大，RM 与真实偏好的差距越被放大（Gao et al., 2022 的经典曲线，第 7 讲）。


## C.5 PPO（RLHF 主力）

逐 token 奖励塑形（把 KL 放进 reward）：

$$
\tilde r_t=-\beta\log\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\text{ref}}(y_t\mid s_t)},\qquad \tilde r_{|y|}\mathrel{+}= r_\phi(x,y)
$$

优势用 GAE（LLM 中 $\gamma=1$）：$\hat A_t=\sum_{l\ge0}\lambda^{l}\,\delta_{t+l}$，$\delta_t=\tilde r_t+V_\psi(s_{t+1})-V_\psi(s_t)$。

$$
\mathcal{L}_{\text{PPO}}=-\mathbb{E}\Big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\hat A_t\big)\Big]+c_1\mathcal{L}_{V}-c_2 H(\pi_\theta)
$$

- **优化什么**：在“别离 $\pi_{\text{ref}}$ 太远”的约束下最大化 RM 分数。$\min$ 与 clip 的组合构造了原目标的**悲观下界**，把单步更新幅度限制住。
- **梯度在推谁**：$\hat A_t>0$ 的 token 被推高，但一旦 $\rho_t>1+\varepsilon$ 梯度被切断（已经推够了）；$\hat A_t<0$ 时对称。**clip 生效时该 token 的梯度为零**，这是理解 [DAPO](https://arxiv.org/abs/2503.14476) “clip-higher” 的前提。
- **什么时候会坏**：critic 拟合差 → 优势噪声大；$\beta$ 太小 → 语言退化；$\beta$ 太大 → 学不动；$\rho_t$ 因训推 logprob 不一致而失真（第 13 讲）。


## C.6 GRPO 及其变体家族

**GRPO（[DeepSeekMath](https://arxiv.org/abs/2402.03300), 2024.2）**：对每个 $x$ 采样 $G$ 个回答，组内归一化当优势：

$$
\hat A_i=\frac{R(x,y_i)-\mathrm{mean}(R)}{\mathrm{std}(R)+\epsilon}
$$

$$
\mathcal{J}_{\text{GRPO}}=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\min\big(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1\!-\!\varepsilon,1\!+\!\varepsilon)\hat A_i\big)\right]-\beta\,\hat{k}_3
$$

- **优化什么**：用同组样本互相当 baseline，省掉 critic。$\hat A_i$ 的含义是“这个答案比同题的平均水平好多少”。
- **注意两个实现细节**（原稿缺失）：
  - KL **直接加在 loss 上**（不是塞进 reward），用 $\hat k_3=\frac{\pi_{\text{ref}}}{\pi_\theta}-\log\frac{\pi_{\text{ref}}}{\pi_\theta}-1$ 这个非负估计器。
  - 除以 $\mathrm{std}(R)$ 会放大“全组几乎都对/都错”的简单题与难题的梯度；除以 $|y_i|$ 会让长的错误回答每 token 受罚更轻。

| 变体 | 改了 GRPO 的哪一处 | 效果 |
|---|---|---|
| **[Dr. GRPO](https://arxiv.org/abs/2503.20783)**（2025.3） | 去掉 $\mathrm{std}$ 归一化与 $1/\lvert y_i\rvert$ | 消除难度偏差与长度偏差 |
| **DAPO**（2025.3） | clip-higher（$\varepsilon_{\text{low}}\ne\varepsilon_{\text{high}}$）、动态采样（丢弃全对/全错的组）、token 级 loss 聚合、超长软惩罚、去掉 KL | 抑制熵坍塌（entropy collapse）、提高样本效率 |
| **[GSPO](https://arxiv.org/abs/2507.18071)**（2025.7） | 重要性比改成**序列级**并做长度归一化 | MoE 训练稳定性显著改善 |
| **[CISPO](https://arxiv.org/abs/2506.13585)**（2025.6） | 裁剪重要性**权重**而不是丢弃 token 梯度 | 保住被 clip 掉的关键 token（如“Wait”“However”） |
| **RLOO**（2024.2） | baseline 用留一法均值，原版**不带 clip** | 更接近纯 REINFORCE |
| **[REINFORCE++](https://arxiv.org/abs/2501.03262)**（2025.1） | 保留 PPO clip + token 级 KL 惩罚 + **全局 batch** 优势归一化 | 组外 baseline，抗奖励噪声 |

> **常见误解订正**：RLOO 不是“带 clip 的 PPO 式目标”，REINFORCE++ 也不是“去掉 clip 的 REINFORCE”。原稿把两者说反了。


## C.7 DPO

$$
\mathcal{L}_{\text{DPO}}=-\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\big(\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l)\big)\Big],\qquad \hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
$$

- **优化什么**：把 KL 正则化奖励最大化问题的闭式最优解 $\pi^\star\propto\pi_{\text{ref}}\exp(r/\beta)$ 反解出来，让**隐式奖励差**去拟合 BT 偏好。配分函数 $Z(x)$ 在同一 $x$ 的两个回答相减时抵消，所以不需要训 RM。
- **梯度在推谁**：

$$
\nabla_\theta\mathcal{L}_{\text{DPO}}=-\beta\,\mathbb{E}\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)}_{\text{模型判得越反，权重越大}}\big(\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x)\big)\Big]
$$

- **什么时候会坏**：
  - **似然位移**（likelihood displacement）：损失只管**差值**变大，$\log\pi_\theta(y_w)$ 本身完全可以一起下降——训练日志里 chosen 的 logprob 掉下去是常态，不是 bug，但掉太多就意味着概率质量流到了第三类答案上。
  - 只覆盖离线数据分布，数据没覆盖的行为无法纠正。
  - $\pi_{\text{ref}}$ 选错（不是同一份 SFT 检查点）→ 隐式奖励全盘失真。


## C.8 DPO 家族的其余成员

| 方法 | 关键式子 | 它在解决什么 | 代价 |
|---|---|---|---|
| **IPO**（2023.10） | $\big(h_\theta-\tfrac{1}{2\tau}\big)^2$，$h_\theta$ 为对数比之差 | 偏好确定（几乎全 1）时 BT 的 logit 会跑向无穷 → 改成回归到固定间隔 | 多一个 $\tau$ 要调 |
| **KTO**（2024.2） | $\lambda_D\,\sigma\big(\beta(\hat r_\theta - z_{\text{ref}})\big)$，好坏样本非对称加权 | 只有单边“好/坏”标签也能训；$z_{\text{ref}}$ 是**批内估计的 KL 参考点**（原稿漏了这一项，它是 KTO 的核心） | 需要设定好/坏样本权重 |
| **ORPO**（2024.3） | $\mathcal{L}_{\text{SFT}}+\lambda_{\text{OR}}\,\mathcal{L}_{\text{OR}}$，$\mathcal{L}_{\text{OR}}=-\log\sigma\big(\log\frac{\text{odds}(y_w)}{\text{odds}(y_l)}\big)$ | 把 SFT 和偏好对齐合成一步，不需要 $\pi_{\text{ref}}$ | 超参敏感 |
| **SimPO**（2024.5） | $\hat r=\frac{\beta}{\|y\|}\log\pi_\theta(y\mid x)$，$-\log\sigma(\hat r_w-\hat r_l-\delta)$ | 去掉 $\pi_{\text{ref}}$（省一份显存）+ 长度归一化抑制长度偏差 | 无锚点，偏离基座更自由 |
| **CPO**（2024.1） | $\mathcal{L}_{\text{SFT}}+\mathcal{L}_{\text{pref}}$（把 $\pi_{\text{ref}}$ 近似成均匀分布） | 省掉 reference 的前向 | 近似有偏 |


## C.9 辅助项

| 项 | 式子 | 作用 | 风险 |
|---|---|---|---|
| KL 惩罚 | $-\beta\,\mathrm{KL}(\pi_\theta\Vert\pi_{\text{ref}})$ | 锚住语言质量与安全行为 | 过强则学不动；RLVR 中常直接去掉 |
| 熵奖励（entropy bonus） | $+c_2H(\pi_\theta)$ | 抵抗熵坍塌 | 系数稍大就会让输出发散 |
| 格式奖励 | $+w_{\text{fmt}}\mathbb{I}[\text{格式正确}]$ | 稳定可解析性 | 权重过大 → 只学格式不学对错 |
| 长度惩罚 | $-\alpha_{\text{len}}\lvert y\rvert$ 或软区间惩罚 | 抑制长度爆炸 | 过强 → 该长的题也不敢想 |
| 预训练混合项（PPO-ptx） | $+\zeta\,\mathbb{E}_{\text{pretrain}}[\log\pi_\theta]$ | 缓解对齐税（alignment tax） | 稀释 RL 信号 |


## C.10 KL 的三种估计器（第 6 讲详讲）

设 $u=\dfrac{\pi_{\text{ref}}(y_t\mid s_t)}{\pi_\theta(y_t\mid s_t)}$，样本来自 $\pi_\theta$：

| 估计器 | 式子 | 性质 |
|---|---|---|
| $\hat k_1$ | $-\log u$ | 无偏，方差大，**可能为负** |
| $\hat k_2$ | $\tfrac12(\log u)^2$ | 有偏，非负，方差小 |
| $\hat k_3$ | $u-\log u-1$ | 无偏且非负（GRPO 采用）。注意：它对**值**无偏，但常见实现里对**梯度**并不无偏 |

---


## C.11 一分钟自检：写完一个新 loss 之后问自己

1. 这个 loss 的最优解是什么？有没有不需要变强就能拿满分的**退化解**？
2. 样本从哪个分布来？训练过程中这个分布会不会漂移？
3. 每个 token 的权重是什么符号、什么量级？会不会被长度、难度、格式系统性放大？
4. 哪一项在防止模型跑飞（KL、clip、margin、长度惩罚）？去掉它会先坏在哪里？
5. 如果奖励涨了但评测没涨，最可能是这个 loss 的哪一项被 hack 了？
