# 第 13 讲　训练系统与稳定性：算法只是一半

> **本讲在主线中的位置**：前面十二讲讲的是**该优化什么**，这一讲讲**怎么真正跑起来**。LLM 的 RL 里，算法差异带来的收益常常小于系统缺陷带来的损失——尤其是那个最隐蔽的：训练引擎与推理引擎算出来的 logprob 不一样。
>
> **学完你应该能做到**
> 1. 画出 RL 训练系统的四个组成部分与数据流，说清 colocate 与分离式的取舍。
> 2. 定义训推不一致，列出它的成因，并说明为什么"统一精度"解决不了。
> 3. 用截断重要性采样（importance sampling）修正它，并说出配套的监控指标。
> 4. 面对六类典型崩溃，给出排查顺序。


## 0. 新增符号

$\pi_{\text{infer}}$（推理引擎实际运行的策略）、$\Delta_t$（logprob 偏差）、$\rho^{\text{TIS}}$（截断重要性权重）、$T_{\text{sync}}$（参数同步周期）、$D_{\text{off}}$（陈旧度（staleness））。见附录 A。


## 1. 系统总览

一个 RL 后训练（post-training）系统有四个部分：

```
┌─────────────── Trainer ────────────────┐
│ Actor(训练) · Critic(可选) · Reference  │
└──────┬──────────────────────▲──────────┘
       │ 参数同步              │ logprob / 优势 / 梯度
┌──────▼──────────────────────┴──────────┐
│  Rollout Engine：vLLM / SGLang / TRT    │
└──────┬──────────────────────▲──────────┘
       │ 采样出的轨迹           │ prompt 批
┌──────▼──────────────────────┴──────────┐
│ Reward / Verifier / 沙箱环境（第 11 讲）│
└─────────────────────────────────────────┘
```

**两种部署形态**：

| | Colocate（同卡混布） | Disaggregated（分离式） |
|---|---|---|
| 做法 | 训练与推理共享同一批 GPU，交替占用 | 采样集群与训练集群分开 |
| 优点 | 参数同步快（同机显存拷贝）；资源利用高 | 各自独立扩缩容；便于异步 |
| 缺点 | 显存争抢；切换有开销 | 权重传输跨网络；更容易陈旧 |
| 适合 | 中等规模、同步 RL | 大规模、异步 RL、长尾轨迹（trajectory） |

**常见框架**（选型看三件事：是否支持你的并行策略、异步能力、以及算法实现的可改性）：

| 框架 | 特点 |
|---|---|
| [verl](https://github.com/volcengine/verl) | HybridFlow 编程模型，算法与并行解耦；DAPO 等配方的官方复现基于它 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 基于 Ray + vLLM + DeepSpeed，结构清晰，易改 |
| [TRL](https://huggingface.co/docs/trl) | 生态最好，适合中小规模与快速实验 |
| NeMo-RL / AReaL / slime 等 | 面向大规模与全异步场景 |


## 2. Rollout：成本大头

回到第 1 讲的物理事实：decode 阶段是显存带宽瓶颈，且严格串行。RL 的一次迭代要为每个 prompt 生成 $G$ 条长回答，全部走 decode，因此**rollout 通常吃掉整个训练的大部分时间**。

优化手段，按收益排序：

1. **连续批处理（continuous batching） + 分页 KV cache**（用专门的推理引擎，而不是训练框架的 `generate`）。
2. **前缀缓存（prefix caching）**：同一个 prompt 采样 $G$ 条，prompt 部分只需 prefill 一次。这是 GRPO 族天然的优化点。
3. **Partial rollout**：把没生成完的轨迹存下来，下一轮接着生成。Kimi k1.5 用它对付长 CoT 的长尾，agent 场景（第 11 讲）更依赖它。
4. **动态采样与补齐**：丢弃零方差组的同时补采，保持有效 batch（DAPO 的做法）。
5. **异步**：采样与训练重叠（§4）。
6. **投机解码（speculative decoding）**：对长输出有效，但与频繁更新的策略配合较麻烦（草稿模型也会过时）。

**长尾是调度问题**：同一批里最慢的那条轨迹决定了这一步的墙钟时间。长度分桶、超时截断（配合软惩罚而非判错）、partial rollout 都是在处理它。


## 3. 训推不一致：最隐蔽的崩溃源


### 3.1 定义

同一个 token，训练引擎与推理引擎算出的对数概率不同：

$$
\Delta_t=\log\pi_{\text{train}}(y_t\mid x,y_{<t})-\log\pi_{\text{infer}}(y_t\mid x,y_{<t})
$$

理论上二者是同一组权重、同一个函数，$\Delta_t$ 应该恒为 0。实际上不是。


### 3.2 成因

| 来源 | 说明 |
|---|---|
| kernel 实现不同 | 训练用的 attention/norm kernel 与推理引擎不是同一份代码 |
| **批大小导致的非确定性** | 归约顺序随 batch 组成变化，浮点加法不满足结合律，同一条输入在不同批里结果不同 |
| 精度与量化 | 推理常用更低精度；LM head 的精度影响尤其大 |
| KV cache | 推理复用缓存，训练重新计算全序列 |
| 并行策略 | TP/PP 的切分方式不同，归约路径不同 |
| MoE 路由 | 参数更新后激活专家改变（第 1、9 讲） |
| 采样参数 | 温度、top-p 的实现细节，以及 logprob 是在采样前还是采样后计算 |
| 版本漂移 | 训练与推理侧的库版本不一致 |


### 3.3 后果

重要性比 $\rho_t=\pi_\theta/\pi_{\text{old}}$ 的分母本应是**真正产生这条数据的策略**。如果你用训练引擎重算的 $\pi_{\text{train}}$ 当分母，而数据其实是 $\pi_{\text{infer}}$ 生成的，那么：

- $\rho_t$ 系统性偏离 1，PPO/GRPO 的 clip 在错误的位置生效；
- 梯度方向出现偏差，且这个偏差与长度相关（长序列上 $\Delta_t$ 累积）；
- 表面上你在做 on-policy 训练，**实际上你在做一个没有做任何修正的 off-policy 训练**；
- 症状是训练前期正常、中后期突然崩溃，且难以复现。

**所以“统一精度”这个建议是不够的**（原稿的说法）。即使 dtype 完全一致，批组成变化带来的归约顺序差异依然存在。


### 3.4 修复手段

**手段一：截断重要性采样（TIS）** —— 2025 年下半年成为标准做法。把推理与训练的差异显式地当作 off-policy 来修正：

$$
\rho^{\text{TIS}}_t=\min\left(\frac{\pi_{\text{train}}(y_t\mid s_t)}{\pi_{\text{infer}}(y_t\mid s_t)},\ C\right)
$$

然后把它乘进策略梯度（policy gradient）项。截断上界 $C$（常取 2 左右）防止个别 token 的巨大比值主导梯度。

```python
def tis_correction(logp_train, logp_infer, cap=2.0):
    """logp_infer 必须是 rollout 时推理引擎返回的值，不能是事后重算的。"""
    ratio = torch.exp(logp_train - logp_infer)
    w = torch.clamp(ratio, max=cap)
    return w.detach()          # 作为系数，不回传梯度

# 用法：pg_loss = -(w_tis * min(rho * A, clip(rho,1-e,1+e) * A))
```

关键前提：**rollout 时必须把推理引擎返回的 logprob 保存下来**。很多框架默认丢掉它，于是这条路直接走不通。

**手段二：让两边真的一致**

- **批不变 kernel**：让归约顺序不依赖批组成，从根上消除非确定性。Thinking Machines 2025 年 9 月的工作系统讲了怎么做；DeepSeek-V4 的报告里也有一套批不变的确定性 kernel 库，目标是训练与推理逐位可复现。
- **提高关键路径精度**：MiniMax-M1 的经验是把 LM head 提到 FP32 后，训练与推理的概率相关性显著回升。
- **MoE routing replay**：缓存旧策略的路由并在新策略上重放（Qwen 团队早期做法），代价是显存与通信。
- **改用序列级比值**：GSPO 对 token 级的数值差异不敏感，报告称对精度差异更宽容，也因此不再需要 routing replay（第 9 讲 §4.3）。

**手段三：监控**

$$
\mu_\Delta=\mathbb{E}[\Delta_t],\qquad \sigma_\Delta=\sqrt{\mathrm{Var}(\Delta_t)}
$$

再加上 $\rho_t$ 的分位数（P1/P50/P99）与 clip 触发率。这三组曲线一旦开始漂移，通常比 reward 曲线早几百步给出预警。


## 4. 同步与异步

| | 同步 | 异步 |
|---|---|---|
| 流程 | 采样 → 训练 → 同步参数 → 采样 | 采样进程持续生成，训练进程持续更新，定期同步 |
| on-policy 程度 | 高 | 低（数据来自旧参数） |
| GPU 利用率 | 低（切换时一边闲置） | 高 |
| 稳定性 | 好 | 需要陈旧度控制 |

**陈旧度**可以用参数版本差或策略间的 KL 来度量：

$$
D_{\text{off}}=\mathrm{KL}\big(\pi_{\theta_{\text{old}}}\,\Vert\,\pi_\theta\big)
$$

控制手段：限制版本落后步数（例如最多落后 1–2 个更新）、重要性采样修正、双缓冲、按需强制同步。大规模异步 RL 的配方（如 ScaleRL）把这些做成了默认设置，其结论是：这类工程选择主要影响**算力效率**，而不太改变最终能达到的性能上限（第 15 讲）。

**权重同步本身也是工程量**：几百 GB 的参数要从训练集群搬到推理集群，常见做法是分片广播 + RDMA/NCCL，与生成重叠进行，避免同步时全员空转。


## 5. 显存与并行

| 组件 | 量级 | 备注 |
|---|---|---|
| Actor 参数 + 梯度 + 优化器状态 | $\approx$16 bytes/参数（BF16 混合精度） | 大头 |
| Critic（若有） | 同上 | PPO 才需要 |
| Reference / Reward | $\approx$2 bytes/参数 | 只推理，可 offload 或共享基座 |
| KV cache | 见第 1 讲公式 | 与并发和长度成正比 |
| 激活值 | 与 batch × 长度成正比 | gradient checkpointing 可换 |

常用手段：ZeRO/FSDP 分片、TP/PP/EP 并行、offload、gradient checkpointing、LoRA（第 12 讲：RL 场景低秩即可，显存收益很大）、量化推理侧。


## 6. 关键超参

| 超参 | 作用 | 经验范围 |
|---|---|---|
| $\beta$ | KL 强度；RLVR 常取 0 | $0\sim10^{-2}$ |
| $\varepsilon$ | clip 范围；DAPO 拆成上下界 | $0.2$，上界可放到 $0.28$ |
| 温度 | 探索强度 | 训练 $0.8\sim1.0$；评测另设 |
| $G$ | 组大小 | $8\sim64$，常用 16 |
| batch | 稳定性 vs 迭代速度 | 与 $G$ 一起决定每步样本数 |
| 学习率 | RL 显著低于 SFT | $10^{-7}\sim10^{-6}$ |
| $T_{\text{sync}}$ | 异步陈旧度 | 以"落后几步"计，不是以时间计 |


## 7. 六类崩溃与排查顺序

| 症状 | 最可能原因 | 排查顺序 |
|---|---|---|
| 中后期突然崩溃，难复现 | 训推不一致 | 看 $\Delta_t$ 与 $\rho$ 分位数 → 检查是否保存了 infer logprob → 上 TIS |
| reward 涨、评测不动 | reward hacking | 看长度与熵 → 抽样读输出 → 对抗测试验证器（verifier） |
| 熵快速趋零 | 熵坍塌（entropy collapse） | 动态采样 → clip-higher → 温度与 $G$ → 最后才加熵奖励（entropy bonus） |
| 长度单调上涨 | 长度偏差（length bias） | 聚合方式（token 级）→ 截断样本处理 → 软长度惩罚 |
| KL 爆炸、语言退化 | $\beta$ 过小或学习率过大 | 自适应 KL → 降学习率 → 查 logprob 偏差 |
| 梯度范数尖刺 | 优势异常或数值问题 | 优势归一化 → 梯度裁剪（gradient clipping） → 查个别超长/异常样本 |

**一条通用原则**：先怀疑**数据与测量**（mask、logprob、验证器、评测集），再怀疑**超参**，最后才怀疑**算法**。绝大多数"算法不 work"最后都查成了前两类。


## 8. 监控看板：必看的九条曲线

1. reward 均值与分布
2. 评测指标（与 reward 分开的独立信号）
3. 熵 $H(\pi_\theta)$
4. 回答长度（均值与分位数）
5. $\rho$ 的分位数与 clip 触发率
6. $\mu_\Delta$、$\sigma_\Delta$（训推偏差）
7. KL 相对参考模型
8. 梯度范数
9. 零方差组比例与有效 batch


## 9. 工程检查清单

- [ ] rollout 时保存推理引擎的 logprob（TIS 的前提）
- [ ] $\pi_{\text{old}}$ 的处理方式明确：是用训练引擎重算，还是直接用 infer 值 + TIS
- [ ] 训练与推理版本锁定，升级走灰度
- [ ] 前缀缓存开启（GRPO 的组采样天然受益）
- [ ] 长尾处理：长度分桶 / partial rollout / 超时软惩罚
- [ ] 异步陈旧度以"落后步数"设上限并监控
- [ ] 参数同步与生成重叠，避免全员空转
- [ ] 九条曲线都在看板上，且有告警阈值
- [ ] 有小规模冒烟配置，能在一小时内复现完整链路


## 10. 自测题

**1. 为什么必须用独立的推理引擎做 rollout？**

训练框架的生成路径没有分页 KV cache、连续批处理、前缀缓存等优化，吞吐差一个量级以上。而 rollout 占 RL 训练时间的大头，这个差距会直接决定实验迭代速度。

**2. 训推不一致为什么"统一精度"解决不了？**

即使 dtype 完全相同，归约顺序仍然会随批组成变化，而浮点加法不满足结合律，所以同一条输入在不同批次里的数值结果不同。要根除需要批不变的 kernel；工程上更现实的做法是承认差异并用截断重要性采样修正它。

**3. TIS 的前提条件是什么？**

rollout 时必须保存推理引擎返回的 logprob，因为修正项需要 $\pi_{\text{train}}/\pi_{\text{infer}}$ 这个比值。很多框架默认丢弃这个值，这时只能先改数据管线。

**4. 异步 RL 的陈旧度为什么用"落后步数"而不是时间来限制？**

因为影响正确性的是策略之间的差距，而不是墙钟时间。同样十分钟，学习率大的训练策略可能已经跑远，学习率小的几乎没变。用版本差（或直接测 $\mathrm{KL}(\pi_{\text{old}}\Vert\pi_\theta)$）才是对的度量。

**5. 看到 reward 涨但评测不动，你的前三步是什么？**

看长度曲线和熵曲线是否异常；抽样读输出判断是否格式化/复读；用第二套验证器或隐藏测试集复评。这三步都属于"怀疑数据与测量"，应该在调超参之前做。

**6. 前缀缓存为什么对 GRPO 特别有用？**

GRPO 对同一个 prompt 采样 $G$ 条回答，prompt 部分完全相同。开启前缀缓存后 prefill 只需做一次，其余 $G-1$ 条直接复用，节省的是与 $G$ 成正比的计算量。

**7. 长尾轨迹为什么是调度问题而不是算法问题？**

同步 RL 里一步的墙钟时间由最慢的那条轨迹决定，与算法无关。处理手段是长度分桶、partial rollout、超时后按软惩罚计分而不是判错，以及异步化。把它当算法问题去调超参不会有效果。


## 11. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Systems design — design an RL post-training system for a 32B model on 512 GPUs.**
*Loops: RL infra (NVIDIA, Meta, ByteDance Seed-style teams, frontier-lab infra).*

*Answer key*: separate rollout engine (vLLM/SGLang) from the trainer; decide colocate vs disaggregated and justify with your sync cost and staleness budget; prefix caching for group sampling; partial rollouts and length bucketing for the tail; sharded weight broadcast overlapped with generation; dynamic sampling to keep the effective batch full. State the monitoring plan as part of the design, not as an afterthought.

**2. Depth — the training engine and the inference engine disagree on token log-probs. Why, and what do you do?**
*Anchored to: the 2025 truncated-importance-sampling line of work and batch-invariance research. Loops: RL infra and post-training research (frontier labs). This is the question that separates people who have run RL at scale from people who have not.*

*Answer key*: causes include different kernels, reduction order varying with batch composition (floating-point addition is not associative), precision and quantization especially in the LM head, KV cache reuse, parallelism layout, and MoE routing changes. Consequences: the importance ratio is wrong, clipping fires in the wrong place, and you are silently doing uncorrected off-policy RL. Fixes: save inference log-probs and apply truncated importance sampling; raise LM-head precision; use batch-invariant kernels; or move to sequence-level ratios, which are more tolerant.

**3. Debugging — training was fine for 400 steps, then collapsed, and you cannot reproduce it. Walk me through it.**
*Loops: post-training engineer (every lab).*

*Answer key*: non-reproducibility itself points at numerics or data ordering rather than hyperparameters. Check the log-prob gap statistics and the ratio distribution first; check whether any batch contained pathological samples (truncated, ultra-long, empty tool outputs); check whether a rollout-engine version or sampling parameter changed mid-run; only then consider learning rate and clip range.

**4. Trade-off — synchronous or asynchronous RL?**
*Anchored to: ScaleRL (2025) and async RL frameworks. Loops: RL infra / research engineer.*

*Answer key*: asynchronous raises utilization and throughput but introduces staleness that must be bounded by version lag plus importance correction. Synchronous is simpler and more stable but leaves GPUs idle during the phase switch. Note the empirical finding that these choices mostly shift compute efficiency rather than the asymptotic performance ceiling, so start synchronous, move async when rollout dominates.

**5. Estimation — how much GPU memory does PPO need relative to SFT for the same model?**
*Loops: applied ML / infra screens.*

*Answer key*: roughly four to five times. Actor and critic each cost about 16 bytes per parameter under BF16 mixed precision (weights, gradients, two Adam moments); reference and reward add about 2 bytes per parameter each as inference-only; then KV cache and activations on top. This arithmetic is the economic argument behind critic-free methods and behind LoRA-based RL.

**6. Prioritization — your rollouts take 80% of wall-clock. Name the first three things you would do.**
*Loops: applied/infra (startups and platform teams).*

*Answer key*: enable prefix caching (free for group sampling), fix the tail with length bucketing plus partial rollouts, and increase inference batch through continuous batching. Then consider asynchrony. Speculative decoding comes later because the draft model goes stale as the policy updates.

**7. Judgment — a colleague proposes switching from GRPO to a new variant to fix instability. What do you check first?**
*Loops: senior/staff research (frontier labs).*

*Answer key*: check measurement and data before algorithms — mask correctness, the log-prob gap, verifier reliability, truncated-sample handling, and whether zero-variance groups are shrinking the effective batch. Most "the algorithm does not work" reports resolve into one of these. If those are clean, then a variant that targets the specific failure (clip-higher for entropy collapse, sequence-level ratios for MoE) is reasonable.


## 12. 本讲相对原稿的修订

| 位置 | 原稿 | 本讲 |
|---|---|---|
| 训推不一致的修复 | “统一精度、统一算子、重算 logprob” | 说明为什么不够；补 TIS、批不变 kernel、FP32 LM head、routing replay、序列级比值五条 |
| 成因 | 列了精度与并行 | 补批组成导致的归约顺序非确定性（根因） |
| 框架 | 只提 vLLM/SGLang/TensorRT | 补 verl / OpenRLHF / TRL 等训练侧框架与选型维度 |
| 长尾 | 未提 | 补 partial rollout、长度分桶、超时软惩罚 |
| 异步 | 定性描述 | 补陈旧度度量、以步数设限、权重同步工程 |
| 监控 | 分散提及 | 汇总为九条必看曲线 |
| 排查 | 按现象列举 | 给出“先数据与测量、再超参、最后算法”的通用顺序 |
| 讨论题 | 10 题带简答 | 7 题直接给答案 + 7 道英文面试题（含公司标注） |


## 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。

- [Kwon et al., 2023 — vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [Zheng et al., 2023 — SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)
- [verl — HybridFlow 框架与 DAPO 复现配方](https://verl.readthedocs.io/en/latest/algo/dapo.html)
- [OpenRLHF — 代码库](https://github.com/OpenRLHF/OpenRLHF)
- [TRL — 文档](https://huggingface.co/docs/trl)
- [Thinking Machines, 2025 — Defeating Nondeterminism in LLM Inference（批不变 kernel）](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [MiniMax, 2025 — MiniMax-M1（FP32 LM head 与训推概率一致性）](https://arxiv.org/abs/2506.13585)
- [Zheng et al., 2025 — GSPO（序列级比值对数值差异更宽容）](https://arxiv.org/abs/2507.18071)
- [Kimi Team, 2025 — Kimi k1.5（partial rollout）](https://arxiv.org/abs/2501.12599)
- [Khatri et al., 2025 — The Art of Scaling RL Compute（异步配方与算力效率结论）](https://arxiv.org/abs/2510.13786)
- [Yu et al., 2025 — DAPO（动态采样与超长处理）](https://arxiv.org/abs/2503.14476)

---

**下一讲**：第 14 讲　评估、安全与对齐——怎么判断后训练是真的成功了，而不是评测被污染了。
