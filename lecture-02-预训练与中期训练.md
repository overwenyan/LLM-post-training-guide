# 第 2 讲　预训练与中期训练：后训练的上限是在这里定下的

> **本讲在主线中的位置**：轴一的最左端——信号是“下一个 token”。这一讲要回答的核心问题是：**base 模型的哪些性质，决定了后面所有后训练（post-training）手段的天花板？**
>
> **学完你应该能做到**
> 1. 复述 scaling law 从 Kaplan 到 [Chinchilla](https://arxiv.org/abs/2203.15556) 再到“过训练”的三次转向，并说明每次转向的驱动力。
> 2. 解释 loss spike 的成因与现代的稳定性工具箱。
> 3. 讲清 mid-training（退火（annealing）、长上下文扩展、推理数据前移）在做什么，以及它与后训练的边界在哪。
> 4. 论证“base 模型的 pass@k 是 RL 的起点”这个说法的合理性与局限。

**新增符号**：$N$（参数量）、$D$（训练 token 数）、$C$（计算量）、$L(\cdot)$（损失）。见附录 A。

---


## 1. 目标与 tokenizer

预训练（pre-training）目标就是第 1 讲那个式子在整个语料上的版本：$\mathcal{L}=-\sum_t\log\pi_\theta(y_t\mid y_{<t})$。

真正值得后训练从业者注意的是 **tokenizer**，因为它是少数几个“定下来就几乎改不了”的决定：

- **数字切分**：把数字按固定位数切（如每 1–3 位一个 token）会显著影响算术能力，也影响第 9 讲验证器（verifier）的答案匹配。
- **特殊 token 预留**：`<|im_start|>`、`<think>`、工具调用标记等必须在预训练阶段就存在于词表中，否则后训练要么被迫复用普通 token（容易与正文冲突），要么扩词表（新 embedding 未经预训练，初期不稳）。
- **多语言效率**：同样一句中文，不同 tokenizer 的 token 数可能差一倍，直接影响上下文预算与推理成本。


## 2. 数据流水线

| 环节 | 做法 | 对后训练的影响 |
|---|---|---|
| 来源 | 网页、代码、书籍、论文、多语言、合成数据 | 代码与数学占比直接决定 RLVR 阶段的“可激发性” |
| 去重 | MinHash / SimHash 近重复去除 | 去重不足 → 记忆与复读，RL 时更容易退化成模板 |
| 质量过滤 | 分类器打分、启发式规则、困惑度筛选 | 过滤太狠会砍掉长尾知识，OOD 泛化变差 |
| 配比 | 领域混合比例，常用小模型做代理实验 | 配比是 base 能力结构的主要旋钮 |
| 去污染 | 移除评测集及其变体 | **直接决定你的后训练实验结论是否可信**（第 14 讲） |

> 去污染这一条放在预训练讲，是因为绝大多数污染发生在这里，但后果全部由后训练的人承担：你在 GSM8K 上涨了 5 分，可能只是把记忆调动了出来。第 9 讲提到的“随机奖励也能涨分”现象，很可能与此有关。


## 3. Scaling law 的三次转向

**第一次（Kaplan et al., 2020.1）**：损失随参数量、数据量、算力呈幂律下降，且建议在算力增加时**主要增大模型**。这直接催生了 GPT-3 这一代“大模型小数据”的设计（175B 参数、约 300B token）。

**第二次（Chinchilla, 2022.3）**：在固定算力 $C\approx 6ND$ 下重新做实验，结论是参数量和数据量应当**等比例增长**，大致每个参数配 20 个 token。按这个标准，此前的大模型普遍训练不足。DeepMind 用 70B 的 Chinchilla 打败了 280B 的 Gopher，同算力下推理还便宜四倍。

**第三次（2023 年至今）：把推理成本算进来**。Chinchilla 最优是**训练最优**，但模型训完要服务很久，推理成本按调用次数累积。于是业界普遍走向“**过训练**”：用远超 Chinchilla 最优的数据量去训一个更小的模型。[Llama 3](https://arxiv.org/abs/2407.21783) 的 8B 模型训了约 15T token，是 Chinchilla 建议的近百倍。

$$
\underbrace{C\approx 6ND}_{\text{训练算力}}\quad\text{vs}\quad\underbrace{C_{\text{infer}}\approx 2N\times\text{(总生成 token 数)}}_{\text{推理算力}}
$$

**第四次转向正在发生**：推理模型把大量算力挪到了生成阶段与 RL 阶段。所以今天的预算分配问题变成了四项——预训练、后训练（含 RL）、测试时计算（test-time compute）、以及服务成本——这是第 15 讲的核心议题。


## 4. 优化与稳定性

大规模预训练最怕的不是效果差，而是**训练崩掉**。

**Loss spike 的成因**：某些 batch 触发极大梯度（常见于罕见 token、长重复串）；注意力 logits 在某些层爆炸；bf16 下的数值累积误差；MoE 路由突变。

**现代工具箱**：

| 工具 | 作用 |
|---|---|
| warmup + 余弦/WSD 学习率 | WSD（warmup-stable-decay）便于中途加数据、便于做退火，近年更常见 |
| 梯度裁剪（gradient clipping）、跳过异常 batch | 最朴素也最有效 |
| z-loss、QK-norm | 压住 softmax logits 的尺度 |
| $\mu$P 等超参迁移方法 | 小模型调好的超参直接迁到大模型 |
| **Muon 优化器** | 2024 年底提出、2025 年起被验证可扩展到大规模；Moonshot 的 [Kimi K2](https://arxiv.org/abs/2507.20534) 用带 QK 裁剪的 MuonClip 在 15.5T token 上实现零 loss spike；[DeepSeek-V4](https://arxiv.org/abs/2606.19348) 也在 1.6T 参数规模上采用 Muon，报告收敛更快、比 AdamW 类基线更稳 |

**精度的下移趋势**：BF16 → FP8（[DeepSeek-V3](https://arxiv.org/abs/2412.19437) 是第一个把 FP8 混合精度用在超大规模训练并公开细节的） → FP4 量化感知训练（DeepSeek-V4 把 FP4 用在 MoE 专家权重上）。每下一档，显存和通信减半，但数值稳定性的工程负担上升——而且这件事会一路传导到第 13 讲：**训练与推理的精度差，是 RL 阶段重要性比失真的主要来源之一**。

> **案例｜DeepSeek-V3（2024 年 12 月）**
> 671B 总参数 / 37B 激活，14.8T token，FP8 混合精度训练，报告的预训练成本约 278.8 万 H800 GPU 小时（按其假定的租用价约合 557.6 万美元）。它的意义不在于参数量，而在于**把“前沿模型训练成本”这件事公开量化了**，此后每一份技术报告都被期待给出可比数字。
>
> **案例｜AI2 [OLMo 2](https://arxiv.org/abs/2501.00656)（2024 年 11 月）**
> 另一个方向的参考：完全开源数据、代码、中间检查点和稳定性排障过程。如果你要做的是可复现的研究而不是刷榜，OLMo 系列是唯一能从预训练一路复现到后训练的公开流水线。


## 5. Mid-training：被忽视但极关键的一段

预训练末期到后训练之间，有一段现在通常被单独称作 **mid-training** 的阶段。它做四件事：

1. **退火（annealing）**：学习率衰减到很低的同时，把数据配比切换成高质量数据（教科书、精标代码、数学推导）。这一步的效果常常比多训几千亿普通 token 更显著。
2. **长上下文扩展**：调整 [RoPE](https://arxiv.org/abs/2104.09864) 基频或用插值方法，在少量长文档上继续训练，把窗口从 4K–8K 扩到 128K 乃至 1M。
3. **推理数据前移**：把长 CoT、带推导过程的数据放进这一阶段，让 base 模型在进入 RL 之前**就已经会生成推理链**。这是 2025 年以后各家的普遍做法，也是“R1-Zero 式纯 RL 能涌现长 CoT”的前提条件之一。
4. **能力预埋**：工具调用格式、特殊 token、多语言补强。

> **为什么这段对后训练的人重要**：很多被归功于 RL 的收益，其实是 mid-training 埋下的。当你对比“我的 RL 方法 vs 别人的 RL 方法”时，如果 base 模型不同，比较基本无效——这是第 9 讲面试题里“你会追问什么”的第一条。


## 6. Base 模型的“可激发性”决定 RL 的上限

一个实用的心智模型：

$$
\text{RL 能拿到的收益}\ \approx\ f\big(\underbrace{\text{基座能采样出正确解的概率}}_{\text{可激发性}},\ \underbrace{\text{奖励信号的可靠性}}_{\text{验证器}},\ \underbrace{\text{RL 算力}}_{C_{\text{RL}}}\big)
$$

如果一道题基座采样一万次都出不来正确解，结果奖励（outcome reward）恒为 0，梯度也恒为 0——**RL 无法从零学会它**。这就是为什么：

- 评估 base 模型时要看 **pass@k（大 k）**，而不只是 pass@1；
- 冷启动（cold start） SFT 或 mid-training 的作用是把正确解的采样概率从 $10^{-6}$ 提到 $10^{-2}$，让 RL 有梯度可用；
- 第 9、10 讲“RL 是激发还是扩展”的争论，本质上是在问这个 $f$ 的第一个自变量能不能被 RL 本身改变。

注意这个模型的**局限**：它假定奖励只有结果对错。一旦引入过程奖励（process reward）、教师分布（on-policy 蒸馏）或多轮环境反馈（environment feedback），即使基座从未采样出完整正确解，也可能获得有效梯度。


## 7. 与后训练的交接清单

- [ ] chat template 的特殊 token 是否已在词表中，且未被其他用途占用
- [ ] base 与 instruct 检查点是否都保留（RL 的 reference 通常取 SFT 后的模型，但做消融要用 base）
- [ ] 评测集污染检查是否在预训练阶段就做过，并留下报告
- [ ] 长上下文窗口是否覆盖你打算训练的最大 CoT 长度（否则 RL 阶段大量样本被截断，见第 9 讲 overlong 处理）
- [ ] base 的 pass@k 基线是否测过（作为 RL 收益的对照）
- [ ] 训练用精度与推理引擎精度是否一致，差异是否已测量（第 13 讲）


## 8. 自测题

**1. Chinchilla 的结论是什么？为什么今天大家又普遍不按它来？**

在固定训练算力下，参数量与数据量应等比例增长（约每参数 20 token），此前的大模型普遍数据不足。今天不按它来，是因为 Chinchilla 只优化**训练**算力；模型训完要服务大量请求，推理成本按调用累积，所以用更多数据训更小的模型（过训练）在总成本上更划算。

**2. 什么是 mid-training？它和 SFT 的边界在哪？**

预训练末期的一段：学习率退火 + 高质量数据配比切换 + 长上下文扩展 + 推理/工具数据前移。它与 SFT 的区别在于目标和形式——mid-training 仍是纯语言建模、不带 chat template、不做 loss mask；SFT 是在对话格式上只对 assistant token 计损失。实践中两者界限在变模糊。

**3. 为什么说“RL 无法教会基座完全不会的题”？这个说法的漏洞在哪？**

在结果奖励下，若所有采样都错，组内奖励全为 0，优势为 0，梯度为 0，模型得不到任何信号。漏洞：一旦有过程奖励、部分分（partial credit）、教师分布或多轮环境反馈，就能在没有完整正确解的情况下提供梯度；此外“采样不出来”是相对于有限采样预算而言的，延长训练与提高温度可以改变它。

**4. 精度选择为什么会影响 RL 训练？**

训练引擎与推理引擎若使用不同精度或不同 kernel，同一 token 的 logprob 会有差异，重要性比 $\rho$ 从一开始就偏离真实值，PPO/GRPO 的裁剪与梯度方向都会失真。MiniMax 的经验是 LM head 的精度影响最大，提到 FP32 后一致性显著改善（第 13 讲）。


## 9. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Estimation — training compute.**
*Loops: research engineer / pretraining (Meta, NVIDIA, Mistral).*

Using `C ≈ 6ND`, how many FLOPs to train a 7B model on 2T tokens? On 1000 H100s at 40% MFU, how long?

*Answer key*: `6 × 7e9 × 2e12 ≈ 8.4e22` FLOPs. At ~1e15 FLOP/s per H100 × 1000 × 0.4 $\approx$ 4e17 FLOP/s → ~2.1e5 s $\approx$ 2.5 days. Order of magnitude is what matters; the point is whether you can connect the formula to real hardware numbers.

**2. Judgment — Chinchilla vs over-training.**
*Loops: research scientist (Google DeepMind, Anthropic).*

Chinchilla says ~20 tokens per parameter. Why do most labs now train far past that? When would you actually follow Chinchilla?

*Answer key*: Chinchilla optimizes *training* compute only; inference cost accumulates per request, so a smaller over-trained model is cheaper in total cost of ownership. You would follow Chinchilla when the model is trained for research/ablation purposes, when serving volume is low, or when you are compute-bound rather than latency-bound.

**3. Debugging — loss spike at step 300k that does not recover.**
*Loops: pretraining infra (Meta, NVIDIA, Moonshot-style open labs).*

*Answer key*: roll back to the last good checkpoint and skip the offending data shard; inspect that shard for pathological repetition or ultra-long sequences; check grad-norm history and per-layer attention logits; mitigate with lower LR, z-loss, QK-norm, or gradient clipping; if MoE, check for router collapse onto a few experts.

**4. Concept — why extend context in mid-training instead of pretraining long from the start?**
*Loops: research engineer (Google, Alibaba Qwen, DeepSeek-style labs).*

*Answer key*: attention cost grows with length and most documents are short, so long-from-scratch is uneconomical. Train short, then adapt RoPE behavior (base-frequency change or interpolation) on a small long-document mixture.

**5. Research hygiene — how would you check contamination before trusting an RL result?**
*Loops: research scientist (Anthropic, AI2, Scale).*

*Answer key*: n-gram / near-duplicate search of eval items against the pretraining corpus if you have access; held-out private eval built after the training cutoff; time-split evaluation (e.g. AIME of the current year); and cross-model-family replication — if a gain only reproduces on one base family, suspect the base, not your method.

**6. Ownership question — where would you spend the next 10% of compute: pretraining, post-training, or test-time?**
*Loops: senior/staff research (OpenAI, Anthropic).*

*Answer key*: unanswerable without the serving profile. Ask for: request volume, whether tasks are verifiable, current base pass@k headroom, and latency budget. Then argue: verifiable tasks + high volume → post-training RL; low volume + hard tasks → test-time compute; weak base coverage → pretraining/mid-training.


## 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。2026 年的条目超出我的训练数据，来自本次检索。

- [Kaplan et al., 2020 — Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al., 2022 — Training Compute-Optimal LLMs（Chinchilla）](https://arxiv.org/abs/2203.15556)
- [Dubey et al., 2024 — The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [DeepSeek-AI, 2024 — DeepSeek-V3（FP8 训练、MTP、无辅助损失负载均衡）](https://arxiv.org/abs/2412.19437)
- [OLMo Team, 2024 — OLMo 2](https://arxiv.org/abs/2501.00656)
- [Peng et al., 2023 — YaRN（长上下文扩展）](https://arxiv.org/abs/2309.00071)
- [Liu et al., 2025 — Moonlight / Muon 可扩展性](https://arxiv.org/abs/2502.16982)
- [DeepSeek-AI, 2026 — DeepSeek-V4 技术报告](https://arxiv.org/abs/2606.19348)

---

**下一讲**：第 3 讲　大模型谱系——每一代引入了什么新特性，以及这些特性分别对应后面哪一讲。
