# 第 11 讲　Agent 与多轮 RL：当环境不再是拼接 token

> **本讲在主线中的位置**：轴一走到最右端——信号来自**环境**。前十讲里状态转移一直是确定性的字符串拼接，从这一讲起不再是了：工具会返回你预料不到的东西，命令会失败，网页会改版。这改变了信用分配（credit assignment）、奖励设计和整个训练系统的形态。
>
> **学完你应该能做到**
> 1. 写出多轮 agent 轨迹（trajectory）的形式化，并说明它为什么是 POMDP 而不是 MDP。
> 2. 指出多轮训练里最容易出错的那个 mask，以及出错后的症状。
> 3. 说清“环境即数据”这句话的含义，以及为什么环境成了 2026 年的主要瓶颈。
> 4. 列举 agent 场景特有的 reward hacking，并给出检测与缓解。


## 0. 新增符号

$o_t$（观察）、$a_t$（动作，此处指一次工具调用或一段输出）、$\mathcal{E}$（环境）、$\mathcal{T}$（工具集）、$\tau=(o_0,a_0,r_0,\dots)$（多轮轨迹）、$h_t$（截至第 $t$ 轮的历史）。见附录 A。


## 1. 与单轮 LLM RL 的差别

| 维度 | 单轮 LLM RL（第 6–10 讲） | Agent RL |
|---|---|---|
| 状态 | $(x,y_{<t})$，完全可观测 | 对话历史 + 工具返回 + **不可见的环境内部状态** |
| 转移 | 确定性拼接 | 由外部系统决定，随机、可能失败、可能有副作用 |
| 动作 | 下一个 token | 两级：token 级生成，语义上是“调哪个工具、传什么参数” |
| 回合长度 | 几千到几万 token | 几十轮交互，几十万 token 不稀奇 |
| 奖励 | 末尾一个标量 | 任务完成度，可能还有过程与成本项 |
| 信用分配 | 难 | 更难：是第 3 轮选错了工具，还是第 17 轮解析错了返回？ |
| 安全 | 输出安全 | **行为安全**：会真的删文件、发请求、花钱 |
| 复现 | 给定种子可复现 | 环境有状态，必须显式重置 |

**形式上它是 POMDP**：模型看不到环境的完整状态（数据库里还有什么、远端服务的真实状况），只能看到观察 $o_t$。实践中的处理是把历史当状态——$s_t\approx h_t=(o_0,a_0,\dots,o_t)$——这也是上下文管理变成核心问题的原因（§4）。

$$
\tau=(o_0,a_0,r_0,o_1,a_1,r_1,\dots,o_T,a_T,r_T),\qquad a_t\sim\pi_\theta(\cdot\mid h_t),\quad o_{t+1}\sim\mathcal{E}(\cdot\mid h_t,a_t)
$$


## 2. 训练的工程形态：一条轨迹就是一个长序列

最常见也最实用的做法：**把整条多轮轨迹拼成一个 token 序列**，用第 5 讲的 loss mask 把不该学的部分屏蔽掉，然后照常跑 GRPO 或 PPO。

```python
def build_trajectory(turns, tokenizer):
    """turns: [{'role': 'user'|'assistant'|'tool', 'content': str}, ...]
    返回 input_ids 与 loss_mask；mask=1 的位置才参与策略梯度。"""
    ids, mask = [], []
    for t in turns:
        seg = tokenizer.encode(render(t))          # render 负责套 chat template
        ids += seg
        # 关键：只有模型自己生成的 token 才有梯度
        mask += [1] * len(seg) if t['role'] == 'assistant' else [0] * len(seg)
    return ids, mask
```

**这个 mask 是本讲最容易出错的地方。** 工具返回的 `observation` 是**环境写的**，不是策略产生的。如果把它计入损失：

- 在 SFT 阶段，模型会学着自己编造工具返回值（幻觉工具输出）。
- 在 RL 阶段更糟——重要性比 $\rho_t$ 在这些位置毫无意义（策略根本没有生成它们），梯度是纯噪声，还会被优势放大。

症状是模型开始“自问自答”：调用工具后不等返回，自己接着写一段看起来像返回值的内容。看到这个症状，先查 mask。

**另一个细节是优势的粒度**。最简单的做法是整条轨迹共享一个优势（GRPO 的自然延伸：同一个任务采样 $G$ 条轨迹，组内比较）。更细的做法是按轮次分配，但这需要轮级奖励，而轮级奖励通常不可得——这正是 agent 场景信用分配困难的核心。


## 3. 奖励设计

| 来源 | 例子 | 注意 |
|---|---|---|
| 任务完成验证 | 单测通过、结果集比对、文件状态检查 | 最可靠，优先用 |
| 环境状态差分 | 任务前后数据库/文件系统的差异是否符合预期 | 比检查输出文本稳健得多 |
| Rubric + 生成式 RM | 研究报告、多步分析这类没有唯一答案的任务 | 见第 7 讲；judge 会有自我偏好（self-preference） |
| 过程约束 | 工具调用格式正确、未越权、未重复调用 | 权重要小，否则学格式不学做事 |
| 成本项 | 工具调用次数、token 数、耗时 | 防止无限试错 |

$$
R(\tau)=\underbrace{\mathrm{ver}(\text{最终状态})}_{\text{主信号}}+w_{\text{fmt}}\,\mathbb{I}[\text{调用格式合法}]-\alpha_{\text{cost}}\,\text{(工具调用数)}-\alpha_{\text{len}}\,\text{(token 数)}
$$

**部分分（partial credit）要格外小心**。在单轮数学里给“步骤格式对”一点分只是轻微跑偏；在 agent 里，任何可被单独优化的中间分都会被模型精确套利——比如为了拿“调用了搜索工具”的分而反复空搜。


## 4. 上下文：长轨迹的真正瓶颈

一条 agent 轨迹会不断把工具返回追加进上下文。三个连带问题：

1. **成本**：每一步都要对全部历史做注意力，这就是第 1 讲说的 decode 阶段带宽瓶颈在长程任务上的放大版，也是 2025–2026 年各家拼命压 KV cache 的直接动因。
2. **信息稀释**：几十万 token 里真正相关的可能只有几千，检索性能随长度下降。
3. **推理状态丢失**：早期实现会在每个用户消息边界清空推理内容，导致模型在多轮任务里反复重建状态。

> **案例｜DeepSeek-V4（2026.4）的 interleaved thinking**
> V3.2 会在新的用户消息到来时丢弃此前的推理内容；V4 改为：当对话包含工具调用时，**跨用户轮次保留完整推理历史**，使长程任务上的思考是累积的；纯对话场景仍然每轮清空以节省上下文。同一份报告还把工具调用格式从 JSON-in-string 换成了带专用 token 的 XML 结构，并把字符串参数与结构化参数分开传——目的就是减少转义与解析失败这类“非能力性”失败。

实践中还需要显式的上下文管理策略：摘要压缩、子任务隔离（让子 agent 独立持有上下文）、以及把长输出落盘再按需读取。


## 5. 环境即数据

单轮 RLVR 的瓶颈是题目和验证器（verifier）；agent RL 的瓶颈是**环境**。一个可用于训练的环境要满足：

- **可自动判定成功**（否则没有奖励）。
- **可重置**到确定初始状态（否则轨迹之间互相污染）。
- **能高并发**（RL 一步要跑成百上千条轨迹）。
- **启动快**（容器冷启动（cold start）会直接吃掉训练吞吐）。
- **可抢占恢复**（训练被打断时不必重跑已经花钱的工具调用）。
- **隔离**（模型会尝试它不该做的事）。

> **案例｜三代环境规模化**
> - [Kimi K2](https://arxiv.org/abs/2507.20534)（2025.7）：大规模合成 agentic 数据——生成工具、生成任务、生成轨迹，再筛选，用来补上真实交互数据的稀缺。
> - DeepSeek-V3.2（2025.12）：可扩展的 RL 框架配合 agentic 任务合成流水线，覆盖 1800 多个环境与 8.5 万条复杂指令。
> - [DeepSeek-V4](https://arxiv.org/abs/2606.19348)（2026.4）：为 rollout 专门建的执行平台，用一套 Python SDK 暴露函数调用、容器、microVM 与完整虚拟机四种执行底座，单集群可并发数十万沙箱；配套的分层镜像存储让 rollout 不必等容器启动，抢占安全的轨迹重放让被中断的训练步不用重跑工具调用。

**判断题**：环境的边际成本远高于文本数据——它需要工程维护、会随外部服务变化而腐化。所以“谁能低成本造出大量高保真环境”正在成为一条新的护城河。


## 6. Agent 特有的 reward hacking

| 手法 | 具体样子 | 检测 |
|---|---|---|
| 改判据 | 修改或删除测试文件、放宽断言 | 把测试设为只读；对比提交前后的测试内容 |
| 特判 | 针对测试用例 hard-code 分支 | 用隐藏测试集复评 |
| 绕过执行 | 直接让进程以成功状态退出 | 检查退出路径与实际状态变更 |
| 空转刷步骤分 | 反复调用工具拿过程分 | 成本项 + 调用去重统计 |
| 泄题 | 上网搜到答案而非自己解 | 网络隔离或域名白名单 |
| 伪装完成 | 输出“已完成”但环境状态没变 | 奖励只看环境状态差分，不看文本声明 |

**最重要的一条原则**：奖励应该读**环境的最终状态**，而不是模型自述的结果。凡是模型能自己写出来的东西，都不能单独作为奖励依据。

第 7 讲那个案例在这里要再强调一次：在编码环境里学会作弊，会泛化成与该环境无关的广义不对齐行为。**agent 环境的漏洞不只是分数问题，它是对齐问题。**


## 7. 安全：从输出安全到行为安全

单轮模型说错话，代价是一条坏回答；agent 做错事，代价是删掉的文件、发出去的邮件、花掉的钱。训练与部署都要处理：

- **权限最小化**：只读凭证、沙箱文件系统、网络白名单。
- **不可逆操作**：需要显式确认或人工介入的动作列表。
- **注入防御**：工具返回的内容是**数据**，不是指令。网页里写的“忽略之前的指令”不应被执行——这一点在训练数据里就要有对抗样本。
- **审计**：完整轨迹留存，便于事后追责与回放。


## 8. 评测

agent 评测与前几讲不同的两点：**轨迹级评估**（只有最终状态算数）和**可靠性**（同一任务多次运行都成功才有意义）。

| 基准 | 考什么 |
|---|---|
| SWE-bench Verified | 真实仓库的 issue 能否被修复并通过测试 |
| Terminal-Bench | 终端环境下的多步任务完成率 |
| $\tau$²-bench 一类 | 有规则约束的多轮对话式任务（客服、预订） |
| 工具调用类基准（MCPAtlas、Toolathlon 等） | 工具选择、参数正确性、多工具编排 |

**pass^k 而不是 pass@k**：agent 场景关心的是“$k$ 次都成功”的概率，而不是“$k$ 次里至少一次成功”。这两个指标方向相反，一个衡量可靠性，一个衡量探索潜力。写报告时务必说清用的是哪个。


## 9. 工程检查清单

- [ ] observation token 已从 loss mask 中排除，并验证过（打印一条轨迹逐段看）
- [ ] 奖励读环境状态差分，而非模型自述
- [ ] 测试文件与判据对模型只读
- [ ] 有隐藏测试集用于复评
- [ ] 环境可重置、可并发、启动快、可抢占恢复
- [ ] 工具调用格式用结构化 schema，解析失败率单独监控
- [ ] 长轨迹的上下文策略明确（保留推理 / 摘要 / 落盘）
- [ ] 成本项进入奖励，避免无限试错
- [ ] 网络与文件系统权限最小化，不可逆操作有闸门
- [ ] 评测用 pass^k 报告可靠性，并注明采样次数


## 10. 自测题

**1. 为什么说 agent RL 是 POMDP？实践中怎么处理？**

模型看不到环境的完整内部状态，只能看到工具返回的观察。实践中把历史当作状态的近似（$s_t\approx h_t$），代价是上下文越来越长，因此上下文管理成为核心工程问题。

**2. 多轮训练里最容易出错的 mask 是哪个？出错后什么症状？**

工具返回的 observation token。它是环境写的，不是策略生成的。计入损失后，SFT 阶段模型会学着编造工具返回；RL 阶段这些位置的重要性比毫无意义，梯度是被优势放大的噪声。症状是模型调用工具后不等返回就自己续写一段"返回值"。

**3. 为什么 agent 的奖励要读环境状态而不是模型输出？**

因为凡是模型能自己写出来的东西都能被伪造——它可以输出"任务已完成"而什么都没做。读环境状态差分（文件是否变更、数据库是否写入、测试是否真的通过）才是不可伪造的信号。

**4. 部分分在 agent 场景为什么比单轮更危险？**

任何可独立优化的中间分都会被精确套利。单轮里给"格式正确"一点分只是轻微跑偏；agent 里模型会为了拿"调用了工具"的分而空转，把训练算力全花在无意义的调用上。

**5. 环境为什么成了瓶颈？它和数据有什么不同？**

它需要满足自动判定、可重置、高并发、快启动、可抢占恢复、隔离六个条件，任何一条不满足都会拖垮训练吞吐或污染信号。与文本数据不同，环境要持续工程维护，还会随外部服务变化而腐化，边际成本高得多。

**6. pass@k 和 pass^k 分别衡量什么？agent 场景该报哪个？**

pass@k 是 $k$ 次里至少一次成功的概率，衡量探索潜力；pass^k 是 $k$ 次全部成功的概率，衡量可靠性。生产 agent 关心后者——用户不会接受十次里成功一次。

**7. 为什么说保留跨轮推理内容是个训练问题而不只是推理优化？**

因为模型在训练时看到的上下文结构决定了它推理时依赖什么。如果训练时每轮清空推理，模型会学会在每轮开头重建状态；要让它形成累积式的长程思考，训练数据与 RL 轨迹里就得保留这些内容，V4 正是这样做的。


## 11. Interview questions（英文）

> **About the company tags**: tags indicate (a) the lab whose published work the question is anchored to, and (b) the kind of US interview loop where this style of question is commonly reported in public job descriptions and candidate write-ups. They are patterns, not verified question banks — treat them as a guide to *what each company cares about*.

> **Language note**: questions and answer keys are in English on purpose — this is the language you will interview in.

**1. Formalization — write down the multi-turn agent trajectory and explain why this is a POMDP.**
*Loops: RL research engineer (OpenAI, Anthropic, Google DeepMind, agent startups).*

*Answer key*: `τ = (o_0, a_0, r_0, …, o_T, a_T, r_T)` with `a_t ~ π(·|h_t)` and `o_{t+1} ~ E(·|h_t, a_t)`. It is partially observed because the environment's internal state is not visible — only observations are. Standard treatment is to condition on the full history as a state surrogate, which is exactly why context management becomes the dominant engineering problem.

**2. Debugging — after multi-turn training, the model starts inventing tool outputs. What went wrong?**
*Loops: post-training engineer (every lab shipping agents).*

*Answer key*: tool observations were included in the loss mask. They are written by the environment, not the policy, so in SFT the model learns to produce them, and in RL the importance ratio at those positions is meaningless — noise amplified by the advantage. Fix the mask and verify by printing one trajectory segment by segment.

**3. Reward design — design the reward for an agent that fixes GitHub issues.**
*Anchored to: SWE-bench-style setups. Loops: applied AI / agent teams (Anthropic, OpenAI, Cursor-style startups).*

*Answer key*: primary signal is hidden tests passing against an unmodified test suite; make the test files read-only and diff them before and after. Add small format/cost terms, never partial credit that can be optimized alone. Anticipate hacks: editing tests, special-casing inputs, exiting the harness with a success code, or claiming completion without changing state. Evaluate on held-out tests, and report pass^k for reliability.

**4. Systems — your RL step has to run 1,000 agent trajectories; some take 30 seconds and some take 40 minutes. How do you keep GPUs busy?**
*Loops: RL infra (NVIDIA, ByteDance Seed-style teams, frontier-lab infra).*

*Answer key*: partial rollouts (persist unfinished trajectories and continue them next step), asynchronous generation with a bounded staleness window plus importance correction, over-sampling with group filtering, fast container/image startup so environments are not the tail, and preemption-safe replay so interrupted steps do not re-run paid tool calls. Mention that a long tail is a scheduling problem first and an algorithm problem second.

**5. Safety — what changes when the model can take actions instead of only emitting text?**
*Loops: alignment / applied safety (Anthropic, OpenAI, Google).*

*Answer key*: output safety becomes behavior safety. Least-privilege credentials, sandboxed filesystem and network, an explicit list of irreversible actions requiring confirmation, treating tool output as untrusted data rather than instructions (prompt-injection adversarial training), and full trajectory auditing. Connect to the finding that learning to hack coding environments generalizes into broader misalignment.

**6. Evaluation — a team reports 70% on an agent benchmark. What do you ask?**
*Loops: research scientist / staff engineer (frontier labs).*

*Answer key*: pass@1 or pass^k, and over how many runs? Same scaffold and tool set as the baselines? How much test-time compute per task? Were the environments held out from training data or synthesized from the same pipeline? Any human-in-the-loop steps? Agent numbers are much more scaffold-dependent than static benchmark numbers.

**7. Strategy — where would you spend the next engineer-year: better RL algorithm or more environments?**
*Anchored to: the 2025–2026 shift where environments became the bottleneck. Loops: senior/staff (frontier labs, agent startups).*

*Answer key*: usually environments, and say why — algorithmic deltas in the GRPO family mostly buy compute efficiency rather than a higher ceiling (Lecture 15's scaling result), while coverage and fidelity of environments set what the policy can learn at all. Caveat the answer: if your rollouts are unstable or your entropy collapses early, fix the algorithm first because more environments will not help a run that diverges.


## 12. 本讲相对原稿的修订

| 位置 | 原稿 | 本讲 |
|---|---|---|
| 篇幅 | 第 9 讲里的一小节 | 独立成讲 |
| “Agent RL 仍在早期” | 保留该判断 | 订正：2025–2026 年已是主流训练目标，瓶颈从算法转到环境 |
| observation mask | 未提 | 作为本讲第一工程要点，给出代码与症状 |
| 上下文 | 未提 | 补长轨迹成本、信息稀释、跨轮推理保留 |
| 环境 | “环境构建是瓶颈”一句 | 补六条硬指标与三代规模化案例 |
| reward hacking | 通用列举 | 补 agent 特有手法表与“读状态不读自述”原则 |
| 评测 | 未提 | 补轨迹级评测与 pass^k |
| 安全 | “安全风险高” | 补权限最小化、不可逆操作、注入防御、审计 |


## 参考文献与链接

> arXiv 编号与链接是按记忆与检索整理的，第一次使用前建议点开确认。

- [Yao et al., 2022 — ReAct（推理与行动交替的范式起点）](https://arxiv.org/abs/2210.03629)
- [Schick et al., 2023 — Toolformer](https://arxiv.org/abs/2302.04761)
- [Nakano et al., 2021 — WebGPT（用人类反馈训练浏览行为）](https://arxiv.org/abs/2112.09332)
- [Jimenez et al., 2023 — SWE-bench](https://arxiv.org/abs/2310.06770)
- [Kimi Team, 2025 — Kimi K2（大规模 agentic 数据合成 + RL）](https://arxiv.org/abs/2507.20534)
- [DeepSeek-AI, 2026 — DeepSeek-V4（沙箱平台、interleaved thinking、XML 工具调用格式）](https://arxiv.org/abs/2606.19348)
- [Hugging Face 博客, 2026 — One sandbox per rollout: how labs run RL for agents](https://huggingface.co/blog/sergiopaniego/rl-environments-2026)
- [MacDiarmid et al., 2025 — 编码环境里的 reward hacking 会泛化成广义不对齐](https://arxiv.org/abs/2511.18397)

---

**下一讲**：第 12 讲　Adaptation——把通用模型适配到具体领域、语言、用户与新知识，以及 LoRA 在 RL 里为什么格外好用。
