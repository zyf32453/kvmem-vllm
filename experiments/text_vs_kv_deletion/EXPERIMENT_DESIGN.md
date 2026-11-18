# Text-level 与 KV-level Deletion 实验设计

同一段历史信息可以在进入模型前从文字里删掉，也可以在模型已经读过之后限制它再访问那段历史的 K/V。两种操作可能保留不同的状态。研究同时测量**后续任务质量、干预后推理成本及可复用的显存容量**。先从同一段冻结历史做离线 Probe，再在固定 DeepSWE 任务中验证。

## 假设与可观察证据

| 问题 | 观察方法 |
| --- | --- |
| 删除入口是否影响推理 | 对同一个候选、相同 Probe 做 G1 对 G2。记录答案、逐层输出、首个生成位置 logits 及相对 G0 的变化。方向由数据决定。 |
| G2 能否节省历史 Prefill | 先以同一全历史计算一个可复用快照。计数器验证各 arm 本次计算的历史 tokens，再量干预、Probe Prefill 和 decode。先导的全历史重算 Shadow 可以作为输出检查，单独标注。 |
| 剩余状态是否保留候选事实 | 比较后续 K/V 与 recurrent state 指纹，设计唯一且无显式替代证据的事实 Probe，再比较 G0、G1、G2 的正确率和置信度。层输出差异只是机制诊断。 |
| 物理粒度带来多少附带删除 | 比较候选 token、被排除的 Kernel Page tokens、命中的其他单元及将来 G3 allocator block 涉及的位置。保持两种对齐口径各有独立表格。 |

## 实验组

| 组 | 起点和唯一干预 | 说明 |
| --- | --- | --- |
| G0 Full | 同一 Frozen Snapshot 的完整输入与完整 KV | 用原生冷 Prefill 和同后端的暖零删除对照校准，按数据和测量路线匹配下游比较。 |
| G1 Text | 用同一计划编辑原始消息中的历史内容，重新渲染、分词和冷 Prefill | 隔离引擎/请求与缓存命名空间，关闭 APC 和 KVMem session 自动导入。记录历史计算 tokens 与 cache hits，后者要为 0。 |
| G2 Shadow | 从同一历史边界复用原始 Attention KV 和 recurrent 初态，在 Probe 与 decode 的 Attention 读页表里排除候选页 | 保留绝对位置和其余原始历史 K/V。逻辑排除与物理释放分开报告，G2 物理释放为 0。 |
| G3 Physical | 在 G2 上实现已验收的 native allocator ownership 转移、候选页实际释放和恢复 | 独立阶段和运行矩阵。先证明语义与 G2 匹配、没有悬空读者、释放后可被另一请求安全复用，再测成本。 |
| G4 Random | 对同一快照同一剂量抽历史位置，分别配对做 G4-Text 和 G4-KV | 固定抽样 seed、校验边界并保存候选空间。Strict 对齐 token/页数，Practical 各自匹配语义正文 token 数或页数。同角色与历史长度分层优先，条件不足列为缺配。 |

G0/G1/G2 先跑。DeepSWE 的五组齐套以 G3 和代理干预入口验收为前提。Phase 3 可先报告已跑的 G0/G1/G2/G4 配对，G3 保留独立证据状态。

## 同一个候选的两套 Deletion Plan

输入为被冻结的 system/user 消息、重放 Prompt、原始 token IDs、offsets、明确的历史和 Probe 边界，以及含角色、Trace 行号、字段和区间的候选。所有坐标使用左闭右开区间。Token 坐标针对渲染后的完整 Prompt，字符坐标针对 Python Unicode 字符。出厂计划绑定快照 SHA-256 和候选 ID，两个 runner 都读取同一份。

| 模式 | 文本组 G1 | KV 组 G2 | 纳入该对比较的条件 |
| --- | --- | --- | --- |
| Strict Alignment | 将 Shadow 内核页覆盖的原始 token 区间反向映射到可编辑历史字符，重新渲染和分词 | 排除覆盖原候选的整页。首轮依据已验证 worker 的 32-token 内核页假设建计划，原生页归属需再校验。 | 原始被删 token 集完全相同。重分词后的整个 token 序列等于原序列删去目标区间。页触及 system、Probe、超出历史，或文本边界无法严格重分词对齐时保留失败原因。 |
| Practical Deletion | 删除经过审核的完整 Thinking 语义单元正文，保留重放用外层包装 | 向外舍入到完整内核页。记录所有附带从 Attention 排除的位置。 | 候选全部覆盖，Probe token IDs 一致，干预范围完全在允许编辑的历史中。正文与页的差异是本模式的实测变量。 |

例如一个候选覆盖 `[350,365)` 共 15 个 token。在 32-token 内核页中它落在逻辑页 10 和 11，页对齐范围是 `[320,384)` 共 64 个 token。只要历史边界允许，Strict 的文本组也尝试删除这 64 个。Practical 的文本组只编辑正文，KV 组额外排除 49 个，附带比例按 `49/64` 记录。实际是否能做 Strict 还须检查重新分词。

每个 plan 保留目标字符/Token 区间、G1 字符编辑及重分词结果指纹、G2 逻辑页和 token 范围、目标/附带计数和状态。实际 GPU 页/槽随后用关联快照的运行时证据侧车文件绑定，按 layer 和 cache 实例寻址。vLLM allocator block 与 32-token 内核页分别记录。Strict 与 Practical 样本集合、排除数、结果图和统计完全独立。

## 快照和污染检查

- 由现有 Graph 数据和源包定位原始消息字段。检查源包内容哈希、候选原文、原始字段、冻结 checkpoint 与重渲染的字符位置，然后保存原始重放消息顺序、IDs 和 offsets。为实际 tokenizer 保存文件指纹，模型、vLLM 和采样配置在准备阶段以声明记录，再由运行时核验。
- 离线准备保存逻辑页映射，GPU 证据初始为空。Phase 2 捕获 `snapshot_sha256`、request incarnation、原生 slot mapping、全历史逐层 Attention K/V 哈希、recurrent 布局与 conv/SSM 状态哈希、已接受 token 边界及状态保管方式。GPU 命名空间、读者租约和快照生命周期可追溯。
- 先跑零删除。冷 Full 复现完整 IDs，暖 G0 和暖 G2 零删除共享同初态、Probe、后端和采样，检查页归属、历史 K/V、recurrent 初态与 Query logits。复用先导相对 L2 门槛作为兼容参考，以同机重复误差提前锁定容差。记录每个 arm 前后快照哈希、请求 ID 和显式缓存命中。任一失败先保留证据并暂停配对测量。
- G1 从 position 0 按缩短后的位置编码。`query_start` 可以随历史变短。逐层对比按同一 Probe token 的相对序号配对，并记录两边实际绝对位置。被删除区域之后的 K/V 和 recurrent 状态在 G1 重新计算，G2 的位置及旧状态保留。观察差异时考虑这两个因素。

### 真续跑与状态残留

Phase 2 的 Replay 入口用既有 eager worker 做等输入重算诊断，并把 G1 冷 Prefill 与其配对。Warm 入口在原生请求到达历史边界时冻结 Attention K/V、recurrent 状态和已接受计数器。两个独立暖请求通过 vLLM 的同步 connector 在首个 Probe 前恢复同一份字节，逐层核验物理槽位、状态指纹和绝对位置。正式暖续跑需通过真实 GPU 的零删除对照，并在日志中观察到零历史 Token 重算。模型输出、诊断耗时和 GPU 资格证据见 [运行说明](README.md#phase-2-配对推理)。

现有 Shadow 对应 KV-1。KV-2 需要按定义排除候选并重算它之后被保留的 token K/V。混合模型中 KV 的重算依赖状态传播，必须单独记录初态来源。KV-3 再验证相关 recurrent 状态从候选之前的真实 checkpoint 或受控完整前缀重建，比较其中 conv 与 SSM。二者目前是后续假设。可以用只在候选处出现的合成事实诊断残留，同时用真实被审核样本检验结果能否推广。

## 样本、执行与统计

- 首轮以既有 24 个 Graph 候选为起点。先核对正文边界、Trace 证据与快照完整性，目标 20–30 个经审核候选。人工标注 Explicit Duplicate、Superseded Information、Obsolete Information、Critical Information，可保留未判定和补充理由。`content_type` 与人工语义标签独立。Thinking 是主分析对象，工具返回和任务约束仅作为分层控制。若 Thinking 数不足，从同一冻结 Trace、同一入选时点补抽并记录入选规则。
- 固定权重、tokenizer、代码、硬件、GPU 时钟策略、采样与长度预算。以 4090 GPTQ 先导为资格起点，实际机器核验后选同一配置。单序列诊断与有并发负载的服务性能受不同条件约束。
- 每候选/任务/模式/arm 至少三次独立重复。每重复重置隔离状态并以成组交错顺序运行各 arm，分开留冷、暖、校准、失败和超时记录。配对单位是候选/任务及运行 seed。对同一任务下多个候选和重复做聚类，避免把重复当独立任务。模型参数固定，随机抽样与 arm 顺序固定在预注册清单。
- Probe 质量分别统计事实正确、保留、关键遗漏、矛盾和工具调用准确。盲审以原始历史和证据为参照。System One 先过同义、相同、遗漏、矛盾和跨字段对照，再与标注一致性比较。DeepSWE 使用固定任务与官方 verifier reward 计算 Task Score 和 Task Pass Rate，选用官方通过定义。同任务、种子、容器、超时与 verifier 保持一致。
- 先汇总每个单位的 arm 差值和重复波动，再报告均值、标准差、任务簇配对 bootstrap 95% CI。连续指标用配对置换检验或满足条件时用配对 t 检验。二元 pass 在独立单任务配对上用精确 McNemar，重复或同任务多轨迹用任务簇配对重采样。报告有效配对数、排除原因、结果方向和效应量，分层、次要指标及多重检验分别交代。

## 测量口径与阶段验收

| 指标 | 记录方式 |
| --- | --- |
| Prefill、Decode、Deletion、E2E | 分开测历史生成/恢复、文本编辑和分词、删除应用、Probe Prefill、队列、TTFT、Decode 及整次 run。共同历史起点已付成本单列，另报从零启动的总成本。GPU 使用同步事件分解，服务端同步与 hook 诊断开销明确区分。 |
| Throughput、GPU/CPU Memory | 同负载窗口计算 tokens/s，报告原生指标、有效输出量和失败。GPU 报 CUDA allocated/reserved、进程设备内存峰值与池空闲槽变化，CPU 报 RSS 峰值、归档与缓存增量。 |
| Text/KV Reduction | 文本报告字符减少和重分词 token 减少。KV 报原始历史内核页/逻辑排除 token 比例，物理容量用真实空闲归还量乘实测 allocator bytes，CUDA backing allocation 实际缩减单列。 |
| Collateral Deletion | 同时报 count、`(排除数−目标覆盖数)/排除数`、相对历史的比值与共同占页单元。G3 再报 allocator 粒度带来的额外范围。 |
| Recovery Cost | 分开测源读取、H2D、页分配/安装、recurrent 状态恢复、调度等待和重算。验证比特与 Probe，记录恢复前后可复用容量。 |

| 阶段 | 通过标准 |
| --- | --- |
| 0 调研 | 源码路径、现状及风险有可复查依据，组、变量和指标定稿。 |
| 1 数据 | 可复读和核验的 Frozen Snapshot/Plan，候选同源，坏源、越界、不完整或不相容映射有明确结果，输出目录含人工审核表和清单。 |
| 2 模型 | 零删除与重复基线通过，G1 冷启动和 G2 原生页映射可查，Probe IDs 一致，逐层与时延可追溯，暖续跑资格独立。 |
| 3 Agent | 固定任务保留各 arm 原始日志、轨迹、补丁与官方得分，输出配对差。G3 缺资质的结果保持待补状态。 |
| 4 物理回收 | 独占/共享、并发读、取消、页复用、代际、回滚和状态一致性测试通过，之后再量化容量和恢复。 |

G3 先从隔离引擎的独占 allocator block 开始。对每个目标先冻结映射、取消新读、等待所有在途事件和共享引用清空，再原子提交页表和 allocator 更新。共享 prefix、COW、GDN state、partial tail 和 preemption 各有测试，失败需可回滚。vLLM 通常预留整个 pool，归还一页首先体现在其他请求可用容量。能否减少设备进程的占用由实测决定。
