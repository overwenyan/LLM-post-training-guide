# 第 1 讲　从 n-gram 到 Transformer：后训练之前发生了什么

> **本讲在主线中的位置**：整套讲义的主线是“学习信号从哪来”。在进入后训练（post-training）之前，先把信号最原始的那一端——“预测下一个 token”——的载体讲清楚。
>
> **学完你应该能做到**
> 1. 讲清 NLP 每一次范式跃迁分别解决了上一代的什么瓶颈。
> 2. 手写 scaled dot-product attention，并解释为什么要除以 $\sqrt{d_k}$。
> 3. 说出 [RoPE](https://arxiv.org/abs/2104.09864)、[RMSNorm](https://arxiv.org/abs/1910.07467)、[SwiGLU](https://arxiv.org/abs/2002.05202)、GQA/MLA、MoE 各自替换掉了原始 Transformer 的哪一块、换来了什么。
> 4. 估算一次 rollout 的 KV cache 显存，并解释为什么采样比训练慢。

**新增符号**：$\mathcal{V}$（词表）、$x$、$y$、$y_{<t}$、$|y|$、$d_{\text{model}}$、$d_k$、$n_{\text{layer}}$、$n_{\text{head}}$、$n_{\text{kv}}$。见附录 A。

---


## 0. 为什么做后训练的人必须懂这一讲

三个非常具体的接口，后面每一讲都会用到：

| 预训练（pre-training）侧的东西 | 在后训练里变成什么问题 |
|---|---|
| tokenizer 与特殊 token | chat template、loss mask、EOS 是否计入损失（第 5 讲）；数字切分方式直接影响数学答案的匹配（第 9 讲） |
| 架构（尤其 MoE 路由） | MoE 每次更新后激活专家会变，token 级重要性比剧烈波动 → [GSPO](https://arxiv.org/abs/2507.18071) 的由来（第 9 讲） |
| KV cache 与解码特性 | rollout 吃掉 RL 训练大部分算力；PagedAttention、前缀复用决定采样吞吐（第 13 讲） |

换句话说：**你在第 9 讲遇到的算法选择，根子常常在第 1 讲的架构里。**


## 1. 五次范式跃迁

| 阶段 | 代表 | 上一代的瓶颈 | 这一代怎么解 | 留下的新瓶颈 |
|---|---|---|---|---|
| 统计语言模型（1990s–2000s） | n-gram + 平滑 | —— | 数频率 | 上下文窗口只有几个词，参数随 $n$ 指数爆炸，语义靠字符串匹配 |
| 词向量（2013） | [word2vec](https://arxiv.org/abs/1301.3781)、GloVe | 词之间没有相似度 | 把词嵌到连续空间，语义变成几何关系 | 一个词只有一个向量，无法处理多义；仍然没有句子表示 |
| 序列到序列 + 注意力（2014–2015） | seq2seq、Bahdanau attention | 定长向量装不下整句 | 解码时按需回看编码器每个位置 | RNN 沿时间串行，长依赖仍然衰减，训练无法并行 |
| Transformer（2017） | Attention Is All You Need | RNN 串行、长依赖弱 | 完全用注意力，全序列并行，任意两位置一步可达 | 复杂度 $O(L^2)$；需要大量数据才能发挥 |
| 预训练范式（2018–2020） | ELMo、GPT、BERT、T5、GPT-3 | 每个任务单独标注、单独训练 | 先自监督预训练，再迁移；到 GPT-3 甚至不用改参数（in-context learning） | 模型会续写但不听指令、不对齐人类偏好 → **后训练登场** |

最后一行就是本讲义的起点。**预训练解决“会不会”，后训练解决“听不听话、答得对不对”。**

> **案例｜Google[《Attention Is All You Need》](https://arxiv.org/abs/1706.03762)（2017 年 6 月）**
> 原始 Transformer 是为机器翻译设计的编码器-解码器结构，6 层、$d_{\text{model}}=512$、8 头、Post-LN、正弦位置编码。今天几乎每一个部件都被换掉了（见 §3），但三件事保留至今：多头注意力、残差连接、逐位置前馈网络。
>
> **案例｜OpenAI GPT-2（2019 年 2 月）**
> 技术上把 LayerNorm 移到子层输入端（Pre-LN 的前身），让深层模型训练更稳；另一件事影响更深远——OpenAI 以“可能被滥用”为由分阶段发布权重，这是模型发布伦理第一次成为公共议题，后来的开源/闭源之争由此开始（第 3 讲）。


## 2. 自回归语言模型的形式化

$$
\pi_\theta(y\mid x)=\prod_{t=1}^{|y|}\pi_\theta(y_t\mid x,y_{<t}),\qquad \mathcal{L}_{\text{LM}}=-\sum_t\log\pi_\theta(y_t\mid x,y_{<t})
$$

这个式子在本讲义里会反复变形：

- 第 5 讲：加 mask，只对 assistant token 计损失 → SFT。
- 第 6 讲：每个 token 前面乘一个权重 $w_t$ → 策略梯度（policy gradient）。
- 第 9 讲：$w_t$ 来自组内优势 → GRPO。

**所以“语言模型”和“策略”从来就是同一个对象**：$\pi_\theta(y_t\mid x,y_{<t})$ 既是下一个 token 的分布，也是 RL 里状态 $s_t=(x,y_{<t})$ 下的动作分布。


## 3. Transformer 逐部件拆解（以及现代替换）


### 3.1 自注意力

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

其中 $M$ 是因果掩码：$M_{ij}=0$ 若 $j\le i$，否则 $-\infty$。

**为什么除以 $\sqrt{d_k}$**（面试必考）：设 $q,k$ 各分量独立、零均值、方差为 1，则 $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ 的方差是 $d_k$。$d_k=128$ 时点积的标准差约 11，softmax 会被推进饱和区，梯度接近零。除以 $\sqrt{d_k}$ 把方差拉回 1。

**多头**：把 $d_{\text{model}}$ 切成 $n_{\text{head}}$ 份分别做注意力再拼接，让不同头关注不同关系（语法、指代、位置模式），且不增加总计算量。

**复杂度**：训练时 $O(L^2 d)$；推理时有了 KV cache，每生成一个 token 是 $O(L d)$，但要读取整个 cache——**解码阶段是显存带宽瓶颈，不是算力瓶颈**。这一条直接解释了第 13 讲为什么 rollout 那么贵。


### 3.2 位置信息：从正弦编码到 RoPE

原始做法是把正弦位置编码加到 embedding 上，模型只能间接感知相对位置。**RoPE（旋转位置编码）**改为在每个注意力层里对 $q,k$ 按位置做旋转：

$$
\langle R_m q,\ R_n k\rangle = f(q,k,\,m-n)
$$

内积只依赖**相对距离** $m-n$。好处有三：相对位置天然、不增加参数、可以通过调整旋转基频（或 NTK/YaRN 式插值）把训练时 4K 的窗口外推到 128K 甚至 1M（第 2 讲、第 12 讲）。


### 3.3 归一化与激活

| 原始 | 现代 | 为什么换 |
|---|---|---|
| Post-LN（残差后归一化） | **Pre-LN**（子层前归一化） | Post-LN 深层需要精细的 warmup 才不发散；Pre-LN 梯度路径更干净，几十上百层才训得动 |
| LayerNorm（减均值、除方差、缩放平移） | **RMSNorm** | 去掉减均值和偏置项，效果相当但更省算子时间 |
| ReLU / GELU 的 FFN | **SwiGLU** | 门控结构在同等参数下效果更好；代价是 FFN 变成三个矩阵，中间维度通常取 $\tfrac{8}{3}d$ 来对齐参数量 |

> **案例｜[Meta LLaMA](https://arxiv.org/abs/2302.13971)（2023 年 2 月）**
> 它本身没有发明新部件，但把 **RoPE + Pre-Norm + RMSNorm + SwiGLU** 这套组合固定下来，之后几乎所有开源模型（Mistral、Qwen、DeepSeek、Llama 系列自身）都沿用。它同时是“开源权重”这条产业路线的起点——第 3 讲会看到，整个开源生态基本长在这次发布之上。


### 3.4 注意力的显存优化：MHA → MQA → GQA → MLA

KV cache 的大小（每个请求）：

$$
\text{KV bytes}=2\times n_{\text{layer}}\times n_{\text{kv}}\times d_{\text{head}}\times L\times \text{bytes per元素}
$$

其中 $2$ 是 K 和 V。以 32 层、32 个 KV 头、$d_{\text{head}}=128$、BF16、$L=8192$ 为例：
$2\times32\times32\times128\times8192\times2\ \text{B}\approx 4.3\ \text{GB}$——**一个请求**。批量 rollout 时这就是吞吐的天花板。

| 方案 | 做法 | 代价 |
|---|---|---|
| MQA | 所有查询头共享一组 K/V | 省得最多，质量损失明显 |
| **GQA** | 查询头分组，每组共享一组 K/V | 当前默认折中方案 |
| **MLA**（[DeepSeek-V2](https://arxiv.org/abs/2405.04434)，2024.5） | 把 K/V 压缩成低秩潜在向量，缓存潜在向量而非 K/V | 实现复杂，与 RoPE 的组合需要特殊设计 |

> **案例｜[Mistral 7B](https://arxiv.org/abs/2310.06825)（2023 年 9 月）与 DeepSeek-V2（2024 年 5 月）**
> 前者用 GQA + 滑动窗口注意力把 7B 的推理成本压到可日常部署；后者用 MLA 把 KV cache 降到同级模型的一小部分，从而让长上下文与高并发在成本上成立。**推理成本是架构选择的第一驱动力**，这条规律在 2026 年的百万上下文模型上依然成立（第 3 讲）。


### 3.5 MoE：把参数量和计算量解耦

FFN 换成 $N$ 个专家 + 路由器，每个 token 只激活 top-$k$ 个：

$$
y=\sum_{i\in \text{top-}k}g_i\,\mathrm{FFN}_i(x)
$$

- **好处**：总参数量（知识容量）可以做到万亿级，而每个 token 的计算量只相当于一个几十亿参数的 dense 模型。
- **麻烦**：负载均衡（否则少数专家被挤爆、其余闲置）；训练不稳；以及本讲最重要的一条——**路由是离散决策**，参数一更新，同一个输入激活的专家就可能换掉一批。
- **和后训练的连接**：这正是 GRPO 在 MoE 上不稳的根因。Qwen 团队起初靠缓存并重放旧策略的路由（Routing Replay）硬扛，后来用 GSPO 的序列级比值绕开了这个问题（第 9 讲 §4.3）。


## 4. 从架构到采样：为什么 rollout 这么慢

一次生成分两个阶段：

| 阶段 | 特点 | 瓶颈 |
|---|---|---|
| Prefill（处理 prompt） | 所有位置并行 | 算力 |
| Decode（逐 token 生成） | 严格串行，每步只算一个 token | 显存带宽（要反复读 KV cache 和权重） |

RL 的一次迭代要为每个 prompt 生成 $G$ 条长回答，全都走 decode 路径。这就是第 13 讲里“rollout 占训练总成本大头”的物理原因，也是为什么必须用 vLLM / SGLang 这类专门的推理引擎，而不是训练框架的 `generate`。


## 5. 自测题

**1. 为什么注意力要除以 $\sqrt{d_k}$？不除会怎样？**

点积的方差随 $d_k$ 线性增长，logits 尺度过大会把 softmax 推入饱和区，梯度趋近于零，训练停滞。除以 $\sqrt{d_k}$ 让方差保持在 1 左右。

**2. RoPE 相比可学习的绝对位置嵌入好在哪？**

注意力内积只依赖相对位置，符合语言的平移不变性；不引入额外参数；可以通过调整基频或插值把上下文窗口外推到远超训练长度，这是长上下文扩展的主流手段。

**3. GQA 省的是计算还是显存？**

主要是显存（以及随之而来的带宽）。KV cache 大小正比于 KV 头数，GQA 把 KV 头数从 $n_{\text{head}}$ 降到若干组。注意力本身的 FLOPs 变化不大，但解码阶段是带宽瓶颈，所以实际吞吐提升很明显。

**4. 为什么 MoE 模型做 RL 比 dense 模型更容易不稳？**

路由是离散的。一次参数更新后，同一 token 激活的专家可能变了，意味着分子分母走的是不同的子网络，token 级重要性比 $\rho_{i,t}$ 会剧烈波动，进而放大梯度噪声。解决方案要么重放旧路由，要么把重要性比抬到序列级（GSPO）。

**5. tokenizer 会怎样影响后训练？举两个例子。**

其一，数字的切分方式决定模型做多位数运算的难度，也影响验证器（verifier）的答案匹配（`3.50` 与 `3.5`）。其二，chat template 用到的特殊 token 必须在预训练阶段就预留并被正确解析，否则 SFT 时会被当成普通文本切碎，导致角色边界错乱、模型不会停（第 5 讲）。


## 6. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Whiteboard code — scaled dot-product attention with a causal mask.**
*Loops: LLM infra / research engineer (Meta GenAI, NVIDIA, Anthropic).*

Write it in ≤10 lines of PyTorch, then extend to multi-head. Expected follow-ups: where exactly does the mask go, why `-inf` instead of `0`, and what changes if you use `F.scaled_dot_product_attention`.

*Answer key*: `scores = q @ k.transpose(-2,-1) / math.sqrt(d_k)`; add mask of `-inf` on the strictly-upper triangle **before** softmax (masking after softmax breaks normalization); `attn = softmax(scores) @ v`. Multi-head = reshape to `[B, H, L, d_head]`, run the same op, then merge heads and apply the output projection.

**2. Estimation — KV cache.**
*Loops: inference / serving (NVIDIA, Together AI, Fireworks), and any RL-infra role.*

A 70B model: 80 layers, 8 KV heads, `d_head=128`, BF16, batch 32, sequence length 16K. How much KV cache? What does that imply for rollout batch size in RL?

*Answer key*: `2 × 80 × 8 × 128 × 16384 × 2 bytes ≈ 5.4 GB per sequence`, ×32 $\approx$ 172 GB — larger than a single H100. Implication: rollout batch size is capped by KV memory, not FLOPs, which is why paged attention and prefix sharing matter so much for RL throughput.

**3. Concept — why divide by √d_k?**
*Loops: almost every ML screen.*

*Answer key*: with unit-variance entries the dot product has variance `d_k`; at `d_k=128` the std is ~11, which saturates softmax and kills gradients. Strong follow-up: at scale people still see attention-logit blowup, which is why QK-norm and z-loss appear in large-scale training recipes.

**4. Trade-off — dense 32B vs MoE 200B/A20B under a fixed serving budget.**
*Loops: applied/infra (OpenAI, xAI, Databricks).*

*Answer key*: name at least four axes — memory footprint vs active FLOPs; expert-parallel deployment complexity; RL training stability (routing volatility makes token-level importance ratios noisy, see GSPO); and fine-tuning ergonomics (LoRA on MoE is messier). Say what you would measure before deciding.

**5. Concept — why Pre-LN? What does it cost?**
*Loops: research engineer (Google DeepMind, Meta).*

*Answer key*: Post-LN needs careful warmup to avoid divergence at depth; Pre-LN gives a clean identity path for gradients. Cost: reduced effective depth and growing residual-stream magnitude in late layers, which is why a final norm (and often QK-norm) is added back.

**6. Systems reasoning — why is decoding bandwidth-bound?**
*Loops: RL infra (Anthropic, OpenAI), inference startups.*

*Answer key*: prefill processes all positions in parallel (compute-bound); decode generates one token at a time and must re-read weights plus the whole KV cache each step, so arithmetic intensity is low. Consequence for RL: rollouts live entirely in the decode regime, which is why they dominate training cost and why continuous batching / speculative decoding matter (Lecture 13).


## 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。2026 年的条目超出我的训练数据，来自本次检索。

- [Vaswani et al., 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Mikolov et al., 2013 — word2vec](https://arxiv.org/abs/1301.3781)
- [Sutskever et al., 2014 — Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215)
- [Bahdanau et al., 2014 — Neural MT by Jointly Learning to Align and Translate（注意力机制的起点）](https://arxiv.org/abs/1409.0473)
- [Devlin et al., 2018 — BERT](https://arxiv.org/abs/1810.04805)
- [Raffel et al., 2019 — T5](https://arxiv.org/abs/1910.10683)
- [Su et al., 2021 — RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [Zhang & Sennrich, 2019 — RMSNorm](https://arxiv.org/abs/1910.07467)
- [Shazeer, 2020 — GLU Variants（SwiGLU）](https://arxiv.org/abs/2002.05202)
- [Shazeer, 2019 — Multi-Query Attention](https://arxiv.org/abs/1911.02150)
- [Ainslie et al., 2023 — GQA](https://arxiv.org/abs/2305.13245)
- [Dao et al., 2022 — FlashAttention](https://arxiv.org/abs/2205.14135)
- [Kwon et al., 2023 — vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [Touvron et al., 2023 — LLaMA](https://arxiv.org/abs/2302.13971)
- [Jiang et al., 2023 — Mistral 7B](https://arxiv.org/abs/2310.06825)
- [DeepSeek-AI, 2024 — DeepSeek-V2（MLA）](https://arxiv.org/abs/2405.04434)
- [Zheng et al., 2025 — GSPO（MoE 上 RL 不稳的处理）](https://arxiv.org/abs/2507.18071)

---

**下一讲**：第 2 讲　预训练与中期训练（mid-training）——scaling law、loss spike、长上下文扩展，以及 base 模型的“可激发性”为什么决定了 RL 的上限。
