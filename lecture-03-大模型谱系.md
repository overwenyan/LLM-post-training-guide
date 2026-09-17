# 第 3 讲　大模型谱系：每一代引入了什么

> **本讲怎么用**：这一讲不是模型清单，而是一张**索引**。每一代引入的新特性，都标注了它对应本讲义的哪一讲。读到后面某个算法觉得抽象时，回来看它是在哪个模型、哪一年、为解决什么问题被提出的。
>
> **学完你应该能做到**
> 1. 按代际说清每一次跃迁引入了什么，以及为什么是那个时候出现。
> 2. 指出五条反复出现的规律（它们比具体型号更耐用）。
> 3. 在面试里回答“最近这一年模型领域发生了什么变化”这类开放题。

> **关于日期的说明**：2018–2025 年的条目来自各家官方发布与技术报告。2026 年的条目中，有一部分超出我的训练数据范围，来自本次检索到的发布时间线与聚合站点，**不同来源在具体日期上有 1–3 天的出入**，我按月份粒度写，并在不确定处标注。你正式使用前建议按官方发布页再核一遍。

---


## 1. 代际总表

| 代 | 时期 | 核心跃迁 | 一句话 |
|---|---|---|---|
| G0 | 2018–2019 | 预训练（pre-training）-微调范式 | 不再为每个任务从头训模型 |
| G1 | 2020–2022 | 规模与 in-context learning | 不改参数也能做新任务 |
| G2 | 2022–2023 | 对齐与对话 | 会听话了，产品化起点 |
| G3 | 2023.12–2024 | MoE、长上下文、原生多模态、开源追赶 | 成本与能力同时被拉开 |
| G4 | 2024.9–2025 | 推理模型与测试时计算（test-time compute） | 把算力从训练挪到生成 |
| G5 | 2025 | Agent 与工具使用 | 从答问题到做任务 |
| G6 | 2026 | 百万上下文 + 后训练（post-training）结构重组 | 算力从混合 RL 挪向专家 RL + 蒸馏 |


## 2. G0（2018–2019）：预训练-微调范式

| 时间 | 模型 | 新引入 |
|---|---|---|
| 2018.6 | OpenAI GPT-1 | 无监督预训练 + 有监督微调的两段式 |
| 2018.10 | Google BERT | 双向编码 + 掩码语言建模，刷新几乎所有理解类任务 |
| 2019.2 | OpenAI GPT-2 | 生成质量跃升；分阶段发布权重，模型发布伦理首次成为公共议题 |
| 2019.10 | Google T5 | 所有 NLP 任务统一成 text-to-text |

**这一代的遗产**：text-to-text 的统一接口一直用到今天——今天所有的后训练任务（偏好、验证、工具调用）本质上都还是把问题编码成文本序列。


## 3. G1（2020–2022）：规模与 in-context learning

| 时间 | 模型 | 新引入 | 对应讲次 |
|---|---|---|---|
| 2020.5 | OpenAI GPT-3（175B） | few-shot in-context learning：不改参数解决新任务 | 第 12 讲（这是最轻量的 adaptation） |
| 2021.8 | OpenAI Codex | 代码专用继续训练，催生 Copilot | 第 12 讲（领域自适应） |
| 2022.3 | DeepMind [Chinchilla](https://arxiv.org/abs/2203.15556) | 参数与数据应等比例增长 | 第 2 讲 |
| 2022.4 | Google PaLM（540B） | 大规模稀疏基础设施与思维链的早期证据 | 第 10 讲 |

**这一代暴露的问题**：模型知识很多，但你问它问题，它可能继续续写而不是回答。这句话就是整个后训练领域的立项理由。


## 4. G2（2022–2023）：对齐与对话

| 时间 | 模型 | 新引入 | 对应讲次 |
|---|---|---|---|
| 2022.1 上线 / 2022.3 论文 | OpenAI [InstructGPT](https://arxiv.org/abs/2203.02155) | SFT → RM → PPO 三阶段范式 | 第 4、7 讲 |
| 2022.11.30 | ChatGPT | 对话式产品形态；RLHF 的第一次大规模验证 | 第 7 讲 |
| 2022.12 | Anthropic [Constitutional AI](https://arxiv.org/abs/2212.08073) | 用 AI 反馈替代部分人类标注（RLAIF） | 第 7、14 讲 |
| 2023.2 | [Meta LLaMA](https://arxiv.org/abs/2302.13971) | 开源权重路线起点；[RoPE](https://arxiv.org/abs/2104.09864)+[RMSNorm](https://arxiv.org/abs/1910.07467)+[SwiGLU](https://arxiv.org/abs/2002.05202) 成为默认配方 | 第 1 讲 |
| 2023.3 | OpenAI GPT-4 | 多模态输入；技术报告披露 RLHF 后**校准（calibration）变差** | 第 1、14 讲 |
| 2023.7 | Meta [Llama 2-Chat](https://arxiv.org/abs/2307.09288) | 双奖励模型（有用/安全）+ 拒绝采样（rejection sampling） + PPO 的迭代式对齐 | 第 4、7 讲 |
| 2023.9 | [Mistral 7B](https://arxiv.org/abs/2310.06825) | GQA + 滑动窗口，小模型推理成本战 | 第 1 讲 |
| 2023.10 | HuggingFace [Zephyr-7B-$\beta$](https://arxiv.org/abs/2310.16944) | 第一个广泛复现的 DPO 配方 | 第 8 讲 |


## 5. G3（2023.12–2024）：MoE、长上下文、多模态、开源追赶

| 时间 | 模型 | 新引入 | 对应讲次 |
|---|---|---|---|
| 2023.12 | Mistral [Mixtral 8×7B](https://arxiv.org/abs/2401.04088) | 开源 MoE 走向主流 | 第 1 讲 |
| 2024.2 | Google Gemini 1.5 Pro | 百万级上下文首次进入产品 | 第 2、12 讲 |
| 2024.2 | DeepSeek [DeepSeekMath](https://arxiv.org/abs/2402.03300) | **GRPO** | 第 9 讲 |
| 2024.5 | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | **MLA** 压缩 KV cache，长上下文成本大降 | 第 1 讲 |
| 2024.5 | OpenAI GPT-4o | 原生多模态（文本/音频/视觉端到端） | —— |
| 2024.4 / 2024.7 | Meta [Llama 3](https://arxiv.org/abs/2407.21783) / 3.1 405B | 15T token 过训练；SFT + 拒绝采样 + **多轮 DPO** 的公开配方 | 第 2、8 讲 |
| 2024.6 | Anthropic Claude 3.5 Sonnet | 中等规模模型在编码上反超大模型，成本效率成为竞争主轴 | —— |
| 2024.11 | AI2 [Tülu 3](https://arxiv.org/abs/2411.15124) | 提出 **RLVR**，全流程开源 | 第 9 讲 |
| 2024.12 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | FP8 大规模训练、MTP、无辅助损失的负载均衡（load balancing）；公开训练成本 | 第 2 讲 |


## 6. G4（2024.9–2025）：推理模型与测试时计算

| 时间 | 模型 | 新引入 | 对应讲次 |
|---|---|---|---|
| 2024.9 | OpenAI o1-preview | 长 CoT + RL；给出“训练算力与测试时算力都能带来提升”的曲线；隐藏推理内容 | 第 10 讲 |
| 2024.12 | OpenAI o3（发布预告） | 测试时算力继续加码，ARC-AGI 上的跳变 | 第 10、15 讲 |
| 2025.1.20 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) / R1-Zero | **纯 RL 涌现长 CoT**；四阶段流水线；蒸馏到小模型；权重与方法全公开 | 第 9、10 讲 |
| 2025.1.20 | Moonshot [Kimi k1.5](https://arxiv.org/abs/2501.12599) | 长上下文 RL、长度惩罚、partial rollout | 第 10、13 讲 |
| 2025.2 | Anthropic Claude 3.7 Sonnet | **混合推理**：同一模型可切普通/扩展思考，思考预算（thinking budget）可控 | 第 10 讲 |
| 2025.3–4 | Google Gemini 2.5 系列 | thinking budget 作为 API 参数暴露给用户 | 第 10 讲 |
| 2025.3 | [DAPO](https://arxiv.org/abs/2503.14476) / [Dr. GRPO](https://arxiv.org/abs/2503.20783) | GRPO 的偏差修补与开源大规模 RL 系统 | 第 9 讲 |
| 2025.4 | Alibaba [Qwen3](https://arxiv.org/abs/2505.09388) | 开源模型的混合思考模式；强到弱蒸馏替代部分 RL | 第 5、10 讲 |
| 2025.6–7 | [MiniMax-M1](https://arxiv.org/abs/2506.13585)（[CISPO](https://arxiv.org/abs/2506.13585)）/ Qwen3 [GSPO](https://arxiv.org/abs/2507.18071) | RL 算法向稳定性工程收敛 | 第 9 讲 |
| 2025.9 | DeepSeek-R1 登上 Nature 封面 | 推理模型的方法首次经过同行评议 | 第 10 讲 |


## 7. G5（2025）：Agent 与工具使用

| 时间 | 模型 | 新引入 | 对应讲次 |
|---|---|---|---|
| 2025.2 | OpenAI Deep Research | 端到端 RL 训练的浏览与研究 agent | 第 11 讲 |
| 2025.5 | Anthropic Claude 4（Opus/Sonnet） | 长程编码任务与 agent 工作流成为主要评测维度 | 第 11、14 讲 |
| 2025.7 | Moonshot [Kimi K2](https://arxiv.org/abs/2507.20534) | 1T MoE，MuonClip 稳定训练；**大规模 agentic 数据合成 + RL** | 第 2、11 讲 |
| 2025.8 | OpenAI GPT-5 | 统一路由（按难度自动决定是否深思） | 第 10 讲 |
| 2025.11 | Google Gemini 3 | 多模态 + agent 能力的整体升级 | —— |
| 2025.12.1 | DeepSeek-V3.2（V3.2-Exp 于 9.29） | **DSA 稀疏注意力**；可扩展 RL 框架；覆盖 1800+ 环境、8.5 万条复杂指令的 agentic 任务合成；工具调用中原生“边想边用” | 第 11、13 讲 |

**这一代的判断**：评测重心从“答题准确率”转向“任务完成率”。SWE-bench、Terminal-Bench、$\tau$²-bench 取代 MMLU 成为主看指标（第 14 讲）。


## 8. G6（2026 年至今）：百万上下文与后训练结构重组

> 以下条目按月份粒度，具体日期以官方发布页为准。

| 时间 | 模型 / 工作 | 新引入 | 对应讲次 |
|---|---|---|---|
| 2026.1 | 小米 [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780) | 提出 **[MOPD](https://arxiv.org/abs/2606.30406)**（多教师在线策略蒸馏（on-policy distillation））这一命名 | 第 5、15 讲 |
| 2026.2 | Z.AI [GLM-5](https://arxiv.org/abs/2602.15763) | 把在线策略蒸馏用在**跨训练阶段**，恢复前序阶段被侵蚀的能力 | 第 5、15 讲 |
| 2026.2–3 | OpenAI GPT-5.3-Codex（2.5）、GPT-5.4（3.5） | 编码 agent 专线；点版本迭代常态化 | 第 11 讲 |
| 2026.4 | Anthropic Claude Opus 4.7、Moonshot Kimi K2.6、Z.AI GLM-5.1 | 开源与闭源在 agent 基准上的差距进一步收窄 | —— |
| 2026.4.23 | OpenAI GPT-5.5 | —— | —— |
| **2026.4.24** | **[DeepSeek-V4](https://arxiv.org/abs/2606.19348)（Pro 1.6T/49B 激活，Flash 284B/13B）** | CSA+HCA 混合注意力、mHC、Muon、FP4 QAT；**后训练改为“领域专家 SFT+GRPO → 多教师 OPD 合并”**；难验证任务用 GRM 且 actor 自任评审；三档思考模式；1M 上下文 | 第 1、2、5、11、15 讲 |
| 2026.6 | NVIDIA [Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007) | 十余个专家教师的多教师 OPD 在旗舰规模上的应用 | 第 15 讲 |
| 2026.6–7 | Anthropic Claude 5 系列（Fable 5、Mythos 5、Opus 5、Sonnet 5） | 新的能力分层与受限访问机制 | 第 14 讲 |
| 2026.7 | OpenAI GPT-5.6（Sol / Terra / Luna 三条线） | 同代内按速度与深度分线 | —— |
| 2026.7 | Moonshot Kimi K3（2.8T 总参 / 约 104B 激活，1M 上下文） | 目前规模最大的开源权重模型 | —— |
| 2026.7 | Thinking Machines Inkling | 该实验室的首个模型 | —— |
| 2026.8–9 | GLM-5.3、Grok 4.6、Gemini 3.x Flash 系列、Claude Fable 5.1 等 | 点版本节奏进入按周计 | —— |

**G6 最值得后训练从业者注意的一件事**：不是上下文长度，而是**后训练的结构变了**。DeepSeek-V4 把 V3.2 里那个混合 RL 阶段整体替换成了 on-policy 蒸馏，RL 退到“训练领域专家”的位置。同一时期 MiMo、GLM、Nemotron、Qwen3 走的是同一方向。这件事的完整讨论在第 15 讲。


## 9. 五条比型号更耐用的规律

1. **推理成本是架构选择的第一驱动力**。GQA、MLA、DSA、CSA/HCA，这条线上每一步都是为了让长上下文与高并发在经济上成立，而不是为了刷分。
2. **每一代的“新特性”会变成下一代的标配**。RoPE、MoE、长上下文、混合思考、agent 训练，无一例外。判断一个新技术会不会留下来，看它是否降低了某项主导成本。
3. **开源滞后闭源的窗口在收窄，但没有消失**。2023 年是一年多，2025–2026 年缩到几个月量级。对研究者的意义是：**你能拿到权重的最强模型，通常落后前沿半年以内**，这足以做绝大多数后训练研究。
4. **后训练的阶段数单调增加**。从 3 段（2022）到 4 段（2025）到“N 个专家 + 1 次合并”（2026）。每增加一段，都是为了解决上一段留下的能力互相侵蚀问题。
5. **评测重心每两年换一次**。知识问答 → 数学与代码 → agent 任务完成率。你的评测集如果三年没换，结论多半已经失效（第 14 讲）。


## 10. 自测题

**1. 为什么推理模型（G4）出现在 2024 年底，而不是更早？**

三个前提同时成熟：一是基座模型经过 mid-training 已经能采样出像样的推理链（可激发性，第 2 讲）；二是可验证奖励的基础设施成熟，数学/代码可以自动判分（第 9 讲）；三是长上下文和推理成本下降，让几千到几万 token 的 CoT 在训练和服务上都可承受。

**2. 开源模型和闭源模型的后训练路线有什么系统性差异？**

闭源可以把大量成本放在人工标注与内部环境构建上，且不必公开配方；开源为了让社区可复现，更依赖可验证奖励、合成数据与算法层面的效率改进（GRPO 家族几乎全部来自开源阵营）。另一个差异是开源必须交付 base 检查点，所以 mid-training 与 post-training 的边界被迫说清楚。

**3. 2026 年“把混合 RL 换成专家 RL + 在线策略蒸馏”解决的是什么问题？**

多域联合 RL 时，后一个阶段的训练会侵蚀前一个阶段的收益（能力互相打架）。分开训各域专家可以各自取得最优，再用学生自采样 + 教师逐 token 信号把能力合并，信息密度远高于 RL 的单标量奖励，收敛更快也更便宜。

**4. 如果让你判断某个刚发布的新技术会不会成为标配，你看什么？**

看它是否降低了当前的主导成本（KV cache、rollout、标注、训练不稳定性），以及它是否需要改变已有基础设施。降低主导成本且能嵌进现有栈的，通常一两代内变成默认；需要全栈改造的（比如非 Transformer 架构）阻力大得多。


## 11. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Open-ended — what changed in this field over the last 12–18 months, and why does it matter for the work you'd do here?**
*Loops: research scientist / research engineer (OpenAI, Anthropic, Google DeepMind).*

*Answer key*: do not list model names. Pick three structural changes and tie each to consequences: (i) long-context cost structure (sparse/compressed attention makes agentic workloads economical); (ii) RL environments and sandboxes became the bottleneck and the moat; (iii) post-training reorganized from one mixed RL stage into per-domain RL experts merged by on-policy distillation.

**2. Comparison — DeepSeek-R1 (2025-01) vs DeepSeek-V4 (2026-04) post-training.**
*Anchored to: DeepSeek technical reports. Loops: post-training research (most frontier labs).*

*Answer key*: R1 = single policy, rule-based rewards, four-stage bootstrap (cold-start SFT → reasoning RL → rejection sampling + SFT → all-scenario RL). V4 = per-domain experts each trained SFT → GRPO, then merged into one student by on-policy distillation; hard-to-verify domains judged by a generative reward model rather than a scalar RM. GRPO is present in both — what changed is *where* RL sits in the pipeline.

**3. Product/engineering — why did every lab ship hybrid thinking or a thinking budget in 2025?**
*Anchored to: Claude 3.7 Sonnet, Gemini 2.5, Qwen3. Loops: applied AI (Anthropic, Google, Alibaba).*

*Answer key*: reasoning tokens are cost, and required depth varies enormously by request. Exposing depth as a parameter fixes both spend and the "thinks too long about trivial questions" UX problem. Training-side implication: you need different length penalties and context budgets per mode, i.e. effectively different RL configurations.

**4. Pushback — "open-weight models have caught up with closed ones." Respond.**
*Loops: any; tests whether you argue with dimensions instead of vibes.*

*Answer key*: split by axis. Knowledge/math: close. Long-horizon agentic tasks and multimodal: still a gap. Also insist on matched conditions — closed models are often evaluated with tools and higher test-time compute. Ask "which axis, at what budget" before answering.

**5. Taste — how do you judge whether a new technique will become standard?**
*Loops: senior research (OpenAI, Anthropic, Meta).*

*Answer key*: does it reduce the currently dominant cost (KV cache, rollout, annotation, instability), and can it be adopted without rewriting the stack? Techniques that do both become defaults within a generation; those requiring full-stack change (non-Transformer backbones) face far more friction regardless of paper results.


## 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。2026 年的条目超出我的训练数据，来自本次检索。

- [Brown et al., 2020 — GPT-3](https://arxiv.org/abs/2005.14165)
- [OpenAI, 2023 — GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
- [Ouyang et al., 2022 — InstructGPT](https://arxiv.org/abs/2203.02155)
- [Bai et al., 2022 — Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Touvron et al., 2023 — Llama 2](https://arxiv.org/abs/2307.09288)
- [Jiang et al., 2024 — Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [Lambert et al., 2024 — Tülu 3（RLVR 的出处）](https://arxiv.org/abs/2411.15124)
- [DeepSeek-AI, 2025 — DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Kimi Team, 2025 — Kimi k1.5](https://arxiv.org/abs/2501.12599)
- [Qwen Team, 2025 — Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Kimi Team, 2025 — Kimi K2](https://arxiv.org/abs/2507.20534)
- [Xiaomi, 2026 — MiMo-V2-Flash（MOPD 命名出处）](https://arxiv.org/abs/2601.02780)
- [Z.AI, 2026 — GLM-5](https://arxiv.org/abs/2602.15763)
- [NVIDIA, 2026 — Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007)
- [DeepSeek-AI, 2026 — DeepSeek-V4（含 HF 模型卡）](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [Hugging Face 博客, 2026 — Distillation in 2026: which frontier models use it and how](https://huggingface.co/blog/sergiopaniego/distillation-2026)

---

**下一讲**：第 5 讲　SFT、拒绝采样与蒸馏（第 4 讲已完成，作为全景图先行）。
