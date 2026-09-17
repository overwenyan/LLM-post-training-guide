# 附录 E　论文时间线与阅读顺序

> 两部分：**按时间的全景表**（看清脉络），以及**四条阅读路线**（按你的目标选一条）。
>
> arXiv 编号与链接按记忆与检索整理，第一次使用前建议点开确认；2026 年的条目超出训练数据范围，来自检索。


## E.1 时间线


### 架构与预训练

| 时间 | 工作 | 贡献 | 讲次 |
|---|---|---|---|
| 2017.6 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer | 1 |
| 2018.10 | [BERT](https://arxiv.org/abs/1810.04805) | 双向预训练（pre-training） | 1 |
| 2020.1 | [Kaplan scaling laws](https://arxiv.org/abs/2001.08361) | 幂律与规模优先 | 2 |
| 2020.5 | [GPT-3](https://arxiv.org/abs/2005.14165) | in-context learning | 3 |
| 2021.4 | [RoPE](https://arxiv.org/abs/2104.09864) | 相对位置与外推 | 1 |
| 2022.3 | [Chinchilla](https://arxiv.org/abs/2203.15556) | 参数与数据等比 | 2 |
| 2023.2 | [LLaMA](https://arxiv.org/abs/2302.13971) | 开源配方定型 | 1、3 |
| 2023.5 | [GQA](https://arxiv.org/abs/2305.13245) | KV cache 折中 | 1 |
| 2023.9 | [YaRN](https://arxiv.org/abs/2309.00071) | 长上下文扩展 | 2、12 |
| 2024.1 | [Mixtral](https://arxiv.org/abs/2401.04088) | 开源 MoE | 1、3 |
| 2024.5 | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | MLA | 1 |
| 2024.7 | [Llama 3 Herd](https://arxiv.org/abs/2407.21783) | 过训练 + 公开后训练（post-training）配方 | 2、8 |
| 2024.12 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | FP8 训练、MTP、公开成本 | 2 |
| 2025.2 | [Moonlight / Muon](https://arxiv.org/abs/2502.16982) | 新优化器可扩展性 | 2 |
| 2026.4 | [DeepSeek-V4](https://arxiv.org/abs/2606.19348) | 混合稀疏注意力、FP4、专家 RL + OPD | 1、2、5、11 |


### 对齐与偏好优化

| 时间 | 工作 | 贡献 | 讲次 |
|---|---|---|---|
| 2017.6 | [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) | 偏好学习起点 | 4、7 |
| 2020.9 | [Learning to Summarize from HF](https://arxiv.org/abs/2009.01325) | 语言任务上的完整 RLHF | 4 |
| 2022.3 | [InstructGPT](https://arxiv.org/abs/2203.02155) | 三阶段范式 | 4、7 |
| 2022.10 | [RM 过优化 scaling law](https://arxiv.org/abs/2210.10760) | 代理与金标的分叉 | 7 |
| 2022.12 | [Constitutional AI](https://arxiv.org/abs/2212.08073) | RLAIF | 7 |
| 2023.5 | [DPO](https://arxiv.org/abs/2305.18290) | 把 RM 折叠进策略 | 8 |
| 2023.7 | [Llama 2](https://arxiv.org/abs/2307.09288) | 双 RM + 拒绝采样（rejection sampling） + PPO | 7 |
| 2023.10 | [IPO](https://arxiv.org/abs/2310.12036) / [Zephyr](https://arxiv.org/abs/2310.16944) | 回归式偏好 / 首个广泛复现的 DPO 配方 | 8 |
| 2024.2 | [KTO](https://arxiv.org/abs/2402.01306) | 单边标签 | 8 |
| 2024.3–5 | [ORPO](https://arxiv.org/abs/2403.07691) / [SimPO](https://arxiv.org/abs/2405.14734) | 去 reference | 8 |
| 2024.10 | [似然位移](https://arxiv.org/abs/2410.08847) | DPO 的失效模式 | 8 |
| 2025.11 | [reward hacking 泛化成不对齐](https://arxiv.org/abs/2511.18397) | 安全后果 | 7、14 |


### RL 算法与推理模型

| 时间 | 工作 | 贡献 | 讲次 |
|---|---|---|---|
| 2015.6 | [GAE](https://arxiv.org/abs/1506.02438) | 优势估计 | 6 |
| 2017.7 | [PPO](https://arxiv.org/abs/1707.06347) | clip 目标 | 6 |
| 2023.5 | [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | PRM 与过程监督 | 9、10 |
| 2024.2 | [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) | 组相对优势 | 9 |
| 2024.2 | [RLOO](https://arxiv.org/abs/2402.14740) | 留一法 baseline | 9 |
| 2024.11 | [Tülu 3](https://arxiv.org/abs/2411.15124) | RLVR 命名与全流程开源 | 9 |
| 2025.1 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) / [Kimi k1.5](https://arxiv.org/abs/2501.12599) | 纯 RL 涌现推理 / 长度与长上下文 RL | 10 |
| 2025.1 | [REINFORCE++](https://arxiv.org/abs/2501.03262) | 全局归一化 | 9 |
| 2025.3 | [DAPO](https://arxiv.org/abs/2503.14476) / [Dr. GRPO](https://arxiv.org/abs/2503.20783) | 四件套消融 / 两处偏差 | 9 |
| 2025.4 | [pass@k 反超](https://arxiv.org/abs/2504.13837) | 激发 vs 扩展之争开端 | 10 |
| 2025.5 | [ProRL](https://arxiv.org/abs/2505.24864) | 延长 RL 的反例 | 10 |
| 2025.6 | [MiniMax-M1 / CISPO](https://arxiv.org/abs/2506.13585) / [Spurious Rewards](https://arxiv.org/abs/2506.10947) | 裁权重不裁项 / 伪收益警告 | 9、14 |
| 2025.7 | [GSPO](https://arxiv.org/abs/2507.18071) | 序列级比值 | 9、13 |
| 2025.10 | [ScaleRL](https://arxiv.org/abs/2510.13786) | RL 算力的 sigmoid 拟合 | 15 |


### 蒸馏、适配与系统

| 时间 | 工作 | 贡献 | 讲次 |
|---|---|---|---|
| 2021.6 | [LoRA](https://arxiv.org/abs/2106.09685) | 低秩适配 | 12 |
| 2022.3 | [STaR](https://arxiv.org/abs/2203.14465) | 自举式拒绝采样 | 5 |
| 2023.4 | [LLaVA](https://arxiv.org/abs/2304.08485) | 视觉指令微调（instruction tuning） | 12 |
| 2023.5 | [LIMA](https://arxiv.org/abs/2305.11206) / [QLoRA](https://arxiv.org/abs/2305.14314) | 少量高质量数据 / 量化微调 | 5、12 |
| 2023.9 | [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) | rollout 基础设施 | 1、13 |
| 2024.5 | [LoRA 学得少也忘得少](https://arxiv.org/abs/2405.09673) | 容量与遗忘的权衡 | 12 |
| 2025.1 | [SFT 记忆、RL 泛化](https://arxiv.org/abs/2501.17161) | 两者分工之争 | 5 |
| 2025.9 | [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) / [批不变 kernel](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | RL 下低秩够用 / 训推一致性 | 12、13 |
| 2025.10 | [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | 反向 KL 在线策略蒸馏（on-policy distillation） | 5 |
| 2026.1–6 | [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780) / [GLM-5](https://arxiv.org/abs/2602.15763) / [MOPD](https://arxiv.org/abs/2606.30406) / [Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007) | 多教师在线策略蒸馏成为主流 | 5、15 |


### 评估与安全

| 时间 | 工作 | 讲次 |
|---|---|---|
| 2023.3 | [GPT-4 技术报告](https://arxiv.org/abs/2303.08774)（RLHF 后校准（calibration）退化） | 14 |
| 2023.6 | [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) | 14 |
| 2023.10 | [GPQA](https://arxiv.org/abs/2311.12022) / [谄媚研究](https://arxiv.org/abs/2310.13548) | 14 |
| 2023.10 | [SWE-bench](https://arxiv.org/abs/2310.06770) | 11、14 |
| 2024.3 | [LiveCodeBench](https://arxiv.org/abs/2403.07974) | 14 |
| 2024.12 | [Deliberative Alignment](https://arxiv.org/abs/2412.16339) | 14 |
| 2025.1 | [Humanity's Last Exam](https://arxiv.org/abs/2501.14249) | 14 |
| 2025.3 | [CoT 监控与混淆型作弊](https://arxiv.org/abs/2503.11926) | 10、14 |
| 2025.4 | [排行榜幻觉](https://arxiv.org/abs/2504.20879) | 14 |
| 2025.7 | [CoT 可监控性立场文件](https://arxiv.org/abs/2507.11473) | 10、14 |


## E.2 四条阅读路线


### 路线一：最少必读七篇（一周）

给只想搞懂主线的人。按顺序读，每篇只抓"它解决了上一篇留下的什么问题"。

1. [InstructGPT](https://arxiv.org/abs/2203.02155)　范式起点
2. [DPO](https://arxiv.org/abs/2305.18290)　把 RM 折叠进策略
3. [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)　去掉 critic
4. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)　完整推理流水线与两个负面结论
5. [DAPO](https://arxiv.org/abs/2503.14476)　GRPO 的工程修补与消融
6. [ScaleRL](https://arxiv.org/abs/2510.13786)　RL 算力怎么 scale
7. [DeepSeek-V4](https://arxiv.org/abs/2606.19348)　2026 年的后训练形态


### 路线二：动手复现顺序（四周）

1. SFT + 拒绝采样（[LIMA](https://arxiv.org/abs/2305.11206)、[STaR](https://arxiv.org/abs/2203.14465) 的思路）
2. DPO（[原论文](https://arxiv.org/abs/2305.18290) + [Zephyr](https://arxiv.org/abs/2310.16944) 配方）
3. GRPO（[DeepSeekMath](https://arxiv.org/abs/2402.03300)，用 [verl](https://verl.readthedocs.io/en/latest/algo/dapo.html) 或 [TRL](https://huggingface.co/docs/trl)）
4. 逐项加 [Dr. GRPO](https://arxiv.org/abs/2503.20783) 与 [DAPO](https://arxiv.org/abs/2503.14476) 的修正，做消融
5. 加 [TIS](https://arxiv.org/abs/2506.13585) 与 $\Delta_t$ 监控，检查训推一致性


### 路线三：按岗位

| 岗位 | 优先读 |
|---|---|
| Post-training research | 路线一全部 + IPO/KTO/SimPO + Dr. GRPO + GSPO + pass@k 之争三篇 |
| RL infra | PPO、GAE、vLLM、GSPO、批不变 kernel、MiniMax-M1、ScaleRL |
| Applied AI | InstructGPT、DPO、LoRA 三篇、RAG、LLaVA、评估四篇 |
| Evaluation | LLM-as-a-Judge、排行榜幻觉、Spurious Rewards、GPT-4 报告校准部分、LiveCodeBench |
| Alignment / safety | Constitutional AI、RM 过优化、CoT 监控两篇、emergent misalignment、Deliberative Alignment |


### 路线四：想做研究的人的"争议阅读法"

挑一个未决问题，把**支持与反对的论文成对读**，这是最快形成判断的方式：

| 争议 | 正 | 反 |
|---|---|---|
| RL 是否扩展能力 | [ProRL](https://arxiv.org/abs/2505.24864) | [pass@k 反超](https://arxiv.org/abs/2504.13837) + [Spurious Rewards](https://arxiv.org/abs/2506.10947) |
| 长 CoT 靠 SFT 还是 RL | [R1 的冷启动设计](https://arxiv.org/abs/2501.12948) | [R1-Zero 与 Dr. GRPO 的分析](https://arxiv.org/abs/2503.20783) |
| SFT 与 RL 的分工 | [SFT 记忆、RL 泛化](https://arxiv.org/abs/2501.17161) | [V4 每个专家仍从 SFT 起步](https://arxiv.org/abs/2606.19348) |
| 蒸馏能否取代 RL | [多教师 OPD 的成功](https://arxiv.org/abs/2606.30406) | 教师本身仍由 RL 产生（同篇内部逻辑） |
| critic 该不该保留 | 无 critic 的 GRPO 族 | 价值方法的改进工作（第 6 讲 §5 案例） |


## E.3 怎么读一篇后训练论文

按这个顺序，二十分钟能判断一篇值不值得细读：

1. **它改了哪一项**？用附录 C 的统一视角：样本来源变了，还是权重 $w_t$ 变了？
2. **基座是什么**？不同基座的结果不可比（第 14 讲的伪收益案例）。
3. **算力对齐了吗**？步数相同不等于算力相同。
4. **评测怎么做的**？采样次数、温度、是否时间切分（time split）、pass@k 报了几个 $k$。
5. **消融在哪**？没有逐项消融的"我们的方法包含五个技巧"基本无法采信。
6. **负面结果有没有**？写了什么没跑通的论文，通常更可信。
