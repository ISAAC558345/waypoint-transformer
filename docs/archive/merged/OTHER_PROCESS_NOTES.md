# OTHER_PROCESS_NOTES

以下内容由 `docs/archive/reports/` 中的过程文档按原文件全文合并。合并后的正文是这些过程记录的归档副本。

## 源文件 `benchmark_complexity_redesign_report.md`

# Benchmark Complexity Redesign Analysis

只做研究设计。未改代码、未训练、未开 Round、未改当前环境 / PPO / Transformer / HGNN。

依据：HTW Phase-1～5.1（当前测试床上 TW / option 均未超过 State-only），以及仓库内已锁定的 Round-19/30（L1 学习床）与 Round-34/35（L2 结构床）事实。后者只作复杂度参照，**本期不实现、不训练**。

---

## 1. 当前测试床是什么

HTW 全部对照发生在 **Level-1 学习床**：

- M=2，K∈{3,5}，共享频谱、到达过程、能量状态、排他 matching 均已存在。
- 学习动作只有 **movement (M,2)**。Matching 由冻结启发式决定（Round-35 所称 GDOR **0/3**：图上的关联决策不属于控制器）。
- Payload 对小范围运动极不敏感（多组策略出现同一 3.431e7 平台）。
- State-only 对当前 `joint_aware` 意图的完成率约 **95%**（Phase-5.1）。

因此：用户草案里 Level-2 的「动态 queue / shared spectrum」**大部分已在 L1**。L1 缺的不是这两项，而是 **规模、关联所有权、以及运动是否改变通信**。

---

## 2. 为什么当前 benchmark 显不出三种优势

### 2.1 Task-Waypoint

TW 要有价值，必须同时满足：

1. 存在 **未来空间子目标**，不能由当前观测的贪心 matching 代替；
2. 到达该子目标会 **改变后续服务/干扰**；
3. 低层把子目标当可完成任务，而不是多余特征。

L1 上三条都不成立：

| 机制 | L1 事实 |
| --- | --- |
| 信息冗余 | 队列、位置、速率已在 state；TW_A 是其确定性滚动（Phase-4 审计） |
| 通信对运动不敏感 | 换 planner / 条件化 / tracking，payload 持平或下降 |
| 任务已被 matching 做完 | Oracle buoy 与环境 assignment 高度重合（Phase-5.1） |
| 几何目标与通信冲突 | 强制跟 W 提高 ρ、降低 payload（Phase-2） |

所以 **不是预测器或 HGNN 没接到 TW**，而是 L1 的「任务」不是 waypoint。

### 2.2 Hierarchical RL

层次要有价值，必须有 **低频任务开关** 且低层无法用逐步贪心复现。

L1 上：option = 当前最优匹配，与 env matching 同频、同目标。State-only 已完成该 option。再加完成奖励只会多走无效路程（76 m vs 29 m）。  
**不是「没做成 HRL」这么简单，而是 L1 没有可分层的任务间隙。**

### 2.3 HGNN

HGNN 要有价值，必须有 **图上的决策变量**（边/关联/干扰结构）且组合爆炸使扁平 DeepSets 不够。

L1：U–B 边 6～10 条，合法部分注入匹配 13 或 31 种，匹配还不归策略。消息传递没有可学习的边动作。Round-35 已写明 L1 GDOR=0/3。  
在 L1 上加 HGNN 只会多一组被忽略的通道，与 TW 条件化同类。

---

## 3. 三级复杂度设计

原则：每一级只增加 **一种** 使某方法开始「有接口」的机制；不把功率、关联、规模一次全上。当前环境保持冻结，下列为 **未来** 设计，不是本期实现。

### Level-1（当前环境）

- M=2，K=3/5，T 短，共享频谱 + 到达队列 + 能量状态。
- 动作：仅 movement。Matching 冻结。
- **预期：** State-only / 规则 matching 足够。TW、HRL、HGNN **均无必要**（已被 HTW 证伪）。
- 用途：负结果、诊断、回归，而不是主方法赛场。

### Level-2（规模 + 关联所有权）

在 L1 物理语义上增加（与 Round-35 结构方向一致，但学习主张仍未证明）：

- **UAV / buoy 数量**：建议最小学习规格 M=4，K=8（合法关联 3393；M=8/K=16 解码不可穷举，不宜作第一步）。
- 共享频谱、队列到达：保留（不是新机制）。
- **控制器拥有 association**：动作拆成 movement `(M,2)` + association preference `(M,K)`；可行性仍由环境投影。GDOR 从 0/3 升到至少 1/3。
- 几何：拉开服务距离，使「现在匹配」与「飞过去再匹配」的 payload 可分。
- 动态队列：提高到达非平稳性 / 空间不均，使 **延迟服务** 有回报（谁先去哪颗 buoy 在 τ 步内会改结局）。

此级 **不要** 上功率分配，避免与关联决策混杂。

### Level-3（资源耦合）

在 L2 已可学关联之后再增加：

- **更强 UAV–UAV 干扰**（η 与几何共同决定，冲突不可被简单排他 matching 消掉）。
- **Association coupling**：关联与运动、干扰同时决定 SINR，不能先匹配再忽略位置。
- **Power allocation**：每链路或每机功率进入动作，与 association 耦合。
- **紧约束能量**：耗尽会切服务，使长程绕行与本地贪心可分。

此级组合空间与物理耦合同时变大，才需要关系模型和长时任务。

---

## 4. 四个问题

### 4.1 Task-Waypoint 从哪一级开始有价值？

**不早于 Level-2，且 L2 仍须额外满足「空间延迟服务」。**

- L1：无价值（已实验）。
- L2：若 association 可学 **且** 飞向未匹配高队列 buoy 能提高区间 payload，则 TW 的 `W` 才可能是「到该 buoy 附近」的可执行子目标；`goal` 对应服务对象。若只加 M、K 而几何仍使 matching 在原地完成任务，TW 仍无价值。
- L3：能量/功率使路径必须提前规划，TW/option 更自然，但应在 L2 先打通空间延迟。

### 4.2 Hierarchical RL 从哪一级有必要？

**Level-2 开始「有必要讨论」，Level-3 才「几乎必要」。**

- L1：无必要；option 与 matching 重合。
- L2：关联低频、运动高频——这是 Options/HRL 的接口。但仅当「保持同一关联 τ 步」优于每步重匹配时才必要。最小检验仍应是 Oracle association hold，而不是先训 π_H。
- L3：功率+能量+干扰使一步贪心失效，层次几乎必要。

### 4.3 HGNN 从哪一级有必要？

**Level-2 开始「结构上有接口」，Level-3 才「性能上可能必要」。**

- L1：无必要（无边动作，组合很小）。
- L2：关联定义在 U–B 边上，干扰定义在并发链路上；HGNN 可输出 association logits。Round-35 已给结构接口，**未给性能优势**。穷举 3393 仍可行，HGNN 不是计算必须，只是归纳偏置候选。
- L3：M、K、功率使穷举失败，关系模型才有计算动机。HGNN 应只在 high-level / association 头，不进低层运动 PPO。

### 4.4 论文应改问题定义，还是继续扩展环境？

**两条都要，但顺序不同；不能继续在 L1 上宣称 TW/HRL/HGNN 主贡献。**

| 选择 | 含义 | 建议 |
| --- | --- | --- |
| 改问题定义 | 承认海洋采集在 M=2、冻结 matching 下是 **运动+规则匹配** 问题，State-only PPO 已饱和 | 适合写清 **负结果与诊断**（HTW Phase-1～5.1）。主方法论文若仍讲 waypoint-transformer，L1 证据是反例。 |
| 扩展环境 | 把可发表的方法主张移到 **L2（关联所有权+空间延迟）**，L3 作展望 | 适合原 TW / 层次 / 图方法故事。环境必须让「任务」从 matching 里拿出来交给策略。 |

推荐：

1. **短期（本文/本仓库叙事）：** 把 L1 结论写成明确负结果——在当前 benchmark 上 TW 与层次 option 均不优于 State-only；原因是任务分解无间隙。不要再堆 λ、planner、HGNN。
2. **若坚持原方法：** **扩展环境到 L2**，先做非学习/Oracle 关联与空间延迟的因果门（payload 必须随关联与位移改变），再谈可学习 TW/HRL/HGNN。
3. **不要：** 在 L1 改问题定义成「我们只学运动」却仍把 HGNN/TW 当核心卖点。

---

## 5. 方法 × 等级（预期，非实验结果）

| 方法 | L1 当前 | L2 规模+关联所有权 | L3 功率/能量/强干扰 |
| --- | --- | --- | --- |
| State-only 运动 PPO | 足够 | 可能不够（忽略关联） | 不够 |
| Task-Waypoint | 无价值 | 有条件有价值（需空间延迟） | 更合适 |
| Hierarchical RL | 无必要 | 开始有必要 | 几乎必要 |
| HGNN | 无必要 | 有结构接口，非计算必须 | 开始有计算动机 |

---

## 6. 建议的下一步（仍不实现本期）

若未来开新阶段（需单独指令）：

1. **不要**在 L1 上训练 HGNN 或 π_H。
2. 用已锁定 L2 环境做 **非学习因果**：改 association / 强制位移，payload 是否显著变。不通过则 L2 仍是假复杂度。
3. 通过后再做 Oracle association-hold vs State-only 运动；这才是 HRL 的最小门。
4. TW 只在「到达未服务区域改变关联与 payload」之后再进入。

本期停止：无代码、无训练、无 Round。

## 源文件 `benchmark_level2_design_report.md`

# Benchmark Level-2 Design Report

只做研究设计。未改代码、未训练、未加 HGNN、未改 PPO、未开 Round、未改当前 Level-1 环境。

---

## 1. Level-1 问题分析

HTW Phase-1～5.1 的对照床是 **Level-1**：M=2，K∈{3,5}，学习动作只有位移，matching 由冻结规则决定。

| L1 性质 | 后果 |
| --- | --- |
| Matching 高度确定（合法关联 13 或 31 种，且不归策略） | 无 task allocation uncertainty |
| State 已含位置、队列、当前匹配、速率代理 | 额外 TW / option 无新决策信息 |
| 一步 matching 即可大量卸队列 | 无多 slot 任务、无坚持服务的必要 |
| 飞行时间相对「当场可服务」过短 | 无空间延迟规划 |
| M=2 排他匹配几乎无冲突组合 | 多智能体耦合弱 |
| Payload 对运动不敏感（3.431e7 平台） | Oracle TW / Oracle option 都不超过 State-only |

已有、但 **不足以** 构成 L2 的机制：到达过程 `A_k(t)`、共享频谱、排他占用、能量状态。L2 必须新增的是：**规模、关联所有权、多 slot 服务、使旅行时间与服务时间同量级的几何**。

---

## 2. Level-2 场景定义

通信任务背景保持不变：多 UAV 采集多 buoy 队列，共享频谱，排他服务。

### 2.1 推荐分档（因 K=20 的组合爆炸）

| 档 | M | K | 合法部分单射关联数 | 穷举 matching |
| --- | ---: | ---: | ---: | --- |
| L1（当前） | 2 | 5 | 31 | 可 |
| **L2-A 实现入口** | 5 | 12 | ~0.25×10⁶ 量级 | 勉强可（需计时） |
| **L2-B 目标规格** | **5** | **20** | **≈ 2.51×10⁶** | 不可作每步精确解码 |
| **L2-C 压力规格** | **8** | **20** | **≈ 9×10⁹** | 必须近似 |

计数：\(\sum_{m=0}^{\min(M,K)}\binom{M}{m} P(K,m)\)。

- L2-B：M=5,K=20 → U–B 边 100，无向 U–U 边 10。
- L2-C：M=8,K=20 → U–B 边 160，无向 U–U 边 28。

**设计结论：** 论文级 L2 **目标写 M=5/8、K=20**；**第一次实现用 L2-A 或 L2-B 的近似关联头**，不要第一步就穷举 M=8。

### 2.2 五个机制（必须同时成立）

**Scale。** 单机无法覆盖全部高队列 buoy；存在「现在服务谁、放弃谁」。

**Dynamic queue。** 保留并加强 \(A_k(t)\)：非平稳、空间不均。关键要求：\(t\) 时刻的服务选择改变 \(Q_k(t+1)\)，从而改变 \(t+H\) 的最优集合。到达应大到「未服务会堆积、溢出」，而不是 L1 那种几乎总被当前 matching 吃掉。

**Service duration。** buoy \(k\) 的一次任务需要 \(D_k\ge 3\) 个 slot（或等价地：以最大速率也要多步才能卸完该任务块）。中途离开则进度丢失或打折（见数学模型）。这制造 **坚持 vs 切换**。

**UAV coupling。** 同一 buoy 同一 slot **至多一个 UAV**（已有排他，L2 要让冲突成为常态：热门 buoy 的竞争者 ≥2）。共享频谱干扰随并发链路增加。

**Spatial delay。** 典型旅行时间 \(T_{\mathrm{fly}}=\|p-b\|/v_{\max}\) 满足 \(T_{\mathrm{fly}}\ge 3\) slots，且与 \(D_k\) 同量级。于是「选目标」必须比较：飞 A 服务 4 步 vs 留在 B 再服务 2 步。

Episode 建议 T=40～80（L1 的 12～20 不够飞+服务）。

---

## 3. 数学模型变化

仍用离散时间 \(t=0,\ldots,T-1\)。UAV \(i\in\{1..M\}\)，buoy \(k\in\{1..K\}\)。

### 3.1 队列与到达（强化）

\[
Q_k(t+1)=\mathrm{clip}\big(Q_k(t)+A_k(t)-S_k(t)-D_k^{\mathrm{drop}}(t),\,0,\,Q_{\max}\big)
\]

\(S_k(t)\) 仅当某机占用 \(k\) 且已在服务半径内连续占用足够进度时才大。L1 的 \(S_k\) 几乎只取决于瞬时匹配；L2 的 \(S_k\) 还取决于 **占用进度** \(\rho_{i,k}(t)\)。

### 3.2 多 slot 服务

引入进度 \(\rho_{i,k}\in[0,1]\)。仅当关联 \(x_{i,k}(t)=1\) 且 \(\mathrm{dist}(p_i,b_k)\le R_{\mathrm{svc}}\)：

\[
\rho_{i,k}(t+1)=\min\big(1,\,\rho_{i,k}(t)+1/D_k\big)
\]

否则 \(\rho\) 清零或衰减（推荐 **切换则清零**，以惩罚朝三暮四）。完成一步「任务块」当 \(\rho=1\)，发放该块 payload，再开下一块。这样 **option 必须 hold \(D_k\) 步** 才有通信收益。

### 3.3 关联与耦合

\[
\sum_i x_{i,k}(t)\le 1,\quad \sum_k x_{i,k}(t)\le 1,\quad x_{i,k}\in\{0,1\}
\]

L1：\(x\) 由环境启发式从 \(p,Q\) 算出。  
L2：\(x\) 由 **控制器关联动作** 经可行性投影得到（与 Round-35 方向一致，规格换成 5/8×20）。

并发关联进入共享频谱 SINR：干扰项随占用集合增大。选同一热门区域会互相压速率。

### 3.4 空间延迟

\[
p_i(t+1)=\Pi_{\mathrm{safe}}\big(p_i(t)+v_{\max}\Delta t\cdot a_i(t)\big),\quad \|a_i\|\le 1
\]

设计区域直径，使跨区 \(T_{\mathrm{fly}}\ge 3\)。Oracle 规划必须对「现在不服务、先飞」做正的未来累计 payload，否则仍会塌回 L1。

### 3.5 目标

区间收益仍是累计 payload / 通信效用。State-only 若不能同时选对 **长期关联** 和 **飞行**，应下降；Oracle option（hold 正确 buoy \(D_k\) 步）与 Oracle TW（飞向未来目标位置）才可能超过它。

---

## 4. 新增状态变量

在 L1 已有 \(\{p_i,v_i,E_i,Q_k,A_k,\text{当前匹配}\}\) 之上，L2 **必须暴露**：

| 变量 | 含义 | 为何需要 |
| --- | --- | --- |
| \(x_{i,k}(t)\) 或等价 one-hot | 当前关联 | 控制器拥有任务 |
| \(\rho_{i,k}\) 或每 buoy 剩余服务 slot | 多 slot 进度 | 坚持/放弃 |
| \(T^{\mathrm{eta}}_{i,k}\) 或距离 | 飞行剩余时间 | 空间延迟 |
| 热门竞争：对 \(k\) 的意向数（可选） | 耦合 | 避免全员冲同一 buoy |
| 到达强度的可预测部分（可选，因果） | 如已知 \(\mathbb{E}[A_k]\) 非未来样本 | 规划需要，禁止泄漏真实未来轨迹 |

**禁止** 把完整未来 \(A_k(t'>t)\) 写进学习观测。Oracle 评估器可以在 **独立影子** 里用未来，学习策略不能。

---

## 5. 新增动作变量

| 通道 | 形状（L2-B） | 形状（L2-C） | 说明 |
| --- | --- | --- | --- |
| Movement | (5, 2) | (8, 2) | 与 L1 同类，环境投影安全 |
| Association preference | (5, 20) | (8, 20) | 高层/联合策略输出；环境做单射可行解码 |
| （L3 再加）功率 | — | — | L2 故意不加，以免与关联混杂 |

低层 HRL：只出 movement，关联由高层 option 冻结 \(\tau=D_k\) 步。  
TW：高层出 \((G,W,H)\)，\(G\) 为未来服务分布，\(W\) 为需飞行才能达到的位置，\(H\sim T_{\mathrm{fly}}\)。

L1 只有 movement，这是 HRL/TW 失败的接口原因之一。

---

## 6. 为何该环境需要 TW / HRL / HGNN

### HRL（Options）

多 slot 服务 + 旅行时间 ⇒ 低频「选 buoy 并 hold」与高频「怎么飞」分离。Oracle Task Option = 每 \(\tau\) 步给出应占用的 \(k_i^*\) 并保持。  
**验证：** Oracle Option > State-only ⇒ 层次有必要。

### TW

存在必须飞过去才能开始 \(\rho\) 的目标：\(W\) 是服务点，\(H\) 是飞行窗，\(G\) 是到达后的服务对象。Option 只回答「谁」，TW 还回答「途经何处、何时开始服务」。当几何使路径选择影响到达时刻与干扰时，TW 可优于纯 option。  
**验证：** Oracle TW > Oracle Option ⇒ HTW 有必要。

### HGNN

关联定义在 U–B 边上，干扰定义在并发占用的超边上。L2-B 已 100 条 U–B 边、约 2.5e6 合法关联；L2-C 无法穷举。HGNN 放在 **关联头 / 高层**，对边打分，而不是进低层运动 PPO。L2-A 仍可穷举时，HGNN 只是归纳偏置，不是计算必须。

---

## 7. 验证标准（实现后，非本期）

在 **同一随机种子与预算** 下，只比学习床，不改原始通信 reward 公式（关联与进度是环境机制，不是把 \(\lambda\|p-W\|\) 写进 env）：

1. **State-only PPO 是否下降：** 相对 L1 平台，L2 上纯运动+启发式关联的 payload / 完成块数应明显变差或方差变大（否则 L2 仍是假复杂度）。
2. **Oracle Task Option 是否提升：** hold 正确 buoy \(D_k\) 步的规则高层 + 低层运动。若 **Option > State-only** ⇒ HRL 门通过。
3. **Oracle TW 是否提升：** 规划窗给出 \((G,W,H)\)（影子里可用未来到达）。若 **TW > Option** ⇒ HTW 门通过；若只 Option 过、TW 不过 ⇒ 先做层次，不要上 waypoint。

三个门都不过：规模/时长/几何仍不够，回到场景参数，而不是加 HGNN。

---

## 8. 后续实现计划（需新指令才开工）

1. **不改** 现有 L1 与 Round-35 锁定产物；新床放独立模块。
2. **L2-A 物理门：** \(D_k\ge 3\)，\(T_{\mathrm{fly}}\ge 3\)，排他冲突频繁；非学习脚本证明「改关联 / 改是否飞」能改累计 payload。
3. **关联接口：** preference → 可行单射；L2-B/C 用采样或网络，不用 2.5e6/9e9 穷举。
4. **Oracle Option vs State-only**（无 HGNN、可复用 PPO 算法、不改 GAE/clip）。
5. 仅当 (4) 通过后做 **Oracle TW vs Option**。
6. 仅当 TW 或关联组合真正需要边模型时，才讨论 HGNN 关联头。

本期停止：无代码、无训练。

## 源文件 `cleanup_deletion_plan.md`

# Cleanup deletion plan

No file has been deleted. Every action below waits for human confirmation.

Protected records stay in place: deployment criterion registries, Phase-4C/4D/4E reports, data registries, scenario and holdout manifests, integrity audits, B2 fit records, tau / Gate-1 records, deployment verification outputs, and failure attribution outputs. `backup/snapshots` stays untouched because those snapshots are the round history.

| file | reason | risk | action |
| --- | --- | --- | --- |
| `src/**/__pycache__/*.pyc` (906 files) | Python bytecode cache | low | delete after confirmation |
| `round_36/**/__pycache__/*.pyc` (735 files) | Python bytecode cache | low | delete after confirmation |
| `tests/**/__pycache__/*.pyc` (122 files) | Python bytecode cache | low | delete after confirmation |
| `experiments/**/__pycache__/*.pyc` (14 files) | Python bytecode cache | low | delete after confirmation |
| `backup/snapshots/**` | immutable pre-round snapshot, including its bytecode | high | keep |
| `src/wt/lowlevel_v2/**/outputs/HTW_Phase4*.md` (19 files) | duplicate of the root audit report | medium | archive only after confirmation; do not delete |
| `src/wt/lowlevel_v2/outputs/phase0/LLV2_DATA_REGISTRY.json` | data registry | high | keep |
| `src/wt/lowlevel_v2/phase4d_deployment_criterion_registry/deployment_criterion_registry.json` | official criterion | high | keep |
| `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/deployment_holdout_manifest.json` | consumed holdout manifest | high | keep |
| `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/holdout_labels.csv` | holdout ground truth | high | keep |
| `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/holdout_integrity_audit.csv` | holdout integrity audit | high | keep |
| `src/wt/lowlevel_v2/phase4e_deployment_verification/outputs/deployment_verification_results.csv` | deployment verification | high | keep |
| `src/wt/lowlevel_v2/phase4e_deployment_verification/outputs/deployment_failure_attribution.json` | failure attribution | high | keep |
| `src/wt/lowlevel_v2/htw_phase4a_hindsight_credit/outputs/phase4a_hindsight_credit.csv` | B2 label source | high | keep |
| `src/wt/lowlevel_v2/htw_phase4b_b2_shadow/outputs/b2_shadow_decisions.csv` | frozen B2 / tau evidence | high | keep |
| `.test-tmp/**` empty directories | empty test temp dirs | low | delete after confirmation |
| `round_02/t*` empty directories | empty tempfile directories | low | delete after confirmation |
| empty directories inside `backup/snapshots` | snapshot structure | high | keep |

The 1,777 delete candidates are all `.pyc` files outside `backup/`. Their paths are the `DELETE_CANDIDATE` rows in `src/wt/lowlevel_v2/project_cleanup_audit/outputs/cleanup_file_inventory.csv`.

The 19 archive rows are duplicate Phase-4 markdown reports. The root copy of each report stays. Do not delete the module copy until the root copy has been checked.

## 源文件 `Current_Hierarchical_MultiUAV_System_Analysis.md`

# 当前多无人机分层规划系统分析

本文件只描述代码里正在运行的分层系统。没有修改代码，没有提出新算法。B2 / Gate-1 是尚未接入 `decide()` 的旁路实验，`planner_ready = False`，不把它写成当前高层控制器。

# 1. 系统总体结构

环境是 L3-B：6 架无人机、12 个浮标、`T = 48`、option hold = 15、服务半径 16 m。通信由 SINR 和速率决定，每完成一个服务块增加 `1e6` bit payload。

```text
High-level
  DeepSets order-free decoder
  joint greedy assignment
        |
        | TaskOption.association，保持 15 步
        v
Low-level
  L0 fly-toward
  + Residual MAPPO 位移残差
  + service lock / safety projector
        |
        | 单位步位移
        v
Environment
  L3BEnv.step(action, association)
  SINR -> rate -> payload
        |
        | 15 步之后的位置、队列、到达
        v
High-level 下一次 decide()
```

代码位置：

- 高层：`src/wt/htw_phase100/high_level.py` 的 `decide()`，解码在 `src/wt/htw_phase943/evaluation.py` 的 `decode_assignment()`
- 低层闭环：`src/wt/lowlevel_v2/phase1b/controller.py` 的 `ResidualWorker`
- 环境：`src/wt/l3b/env.py`

# 2. 高层方法分析

当前高层不是在线 RL，也不是 GNN 规划器。正式推理使用冻结的 DeepSets order-free assignment model，按无人机逐个做 joint greedy 解码。HGNN assignment head 在 `src/wt/htw_phase92/models.py` 里存在，但 `decide()` 加载的是 `OrderFreeAssignmentModel("deepsets")`。

- 算法：监督学习得到的集合匹配解码。Integration Phase-3 的 EXEC 检查点用冻结低层执行器的效用 `q` 做 `L_exec = -sum q log P(Gamma|s)`。推理本身仍是 joint greedy，不在解码时调用低层策略。
- 输入：`_batch_from_env()`。每架无人机有归一化位置、速度、服务进度、是否已分配、已分配浮标队列。能耗槽位被写成全 0。浮标有位置、队列、上一拍到达、占用。全局量是时间、总队列和无人机数量。另外有 UAV–buoy 边和 UAV–UAV 边。
- 输出：长度 6 的整数分配。每个分量是浮标编号或 idle（-1）。`make_option()` 把它包成 `TaskOption`，`remaining_hold = 15`。
- 网络：DeepSets。UAV、buoy、edge 各自 MLP，pair score 和 idle score 再各一个 MLP。解码时逐步选尚未决定的 UAV–token。
- 优化目标：训练目标是分配序列的加权对数似然，权来自冻结执行器效用或更早的 oracle assignment。它不直接最大化整段 `payload_bits`。
- 多机关系：解码是联合的，同一浮标的冲突由 assignment mask 处理，所以高层考虑互斥匹配。边特征里有机间关系，但能耗通道没有真实能耗。
- 通信和干扰：当前队列、位置和占用会进入特征。SINR 不作为高层网络的直接输入；干扰要等环境用该分配算出速率后才出现。高层不显式优化能耗。

优点是分配在一个 option 内排他、可重复，并且每 15 步用执行后的位置和队列重新决策。

# 3. 低层方法分析

低层是 MAPPO，不是 SAC 或 DQN。实现是 `ICARMAPPOLocked` / 后续 Active-aware 变体，更新在 `src/wt/lowlevel_v2/phase1b/mappo.py`。策略学的是相对 L0 的残差，不是从零输出轨迹。

L0 在 `src/wt/lowlevel_v2/base_executor.py`：冻结的 fly-toward，进入服务半径后可由 `InRangeZeroMotionGate` 停住。残差经过 service lock 和 safety projector，再交给环境。

Actor 在 `src/wt/lowlevel_v2/residual_policy.py`。观测由 `build_actor_observation()` 按无人机分开构造：

- 自身，5 维：位置、上一拍位移、常数 1
- 任务，8 维：当前 association 是否有效、相对目标位置和距离、目标队列、自身服务进度、剩余 hold、剩余 episode
- 通信，4 维：该机当前 SINR、速率、总干扰、干噪比。没有目标时为 0
- 邻居集合：其他 5 架无人机的相对位置、相对位移、距离、是否在服务、是否 idle、对方目标相对位置、对方进度、对方是否在服务半径内。`NeighborSetEncoder` 把集合压成 16 维
- 最近机间距 1 维

Actor 主体是两层宽度 64 的 Tanh MLP。输出是 2 维高斯均值，经 tanh 后作为平面位移残差。没有单独的航向角或速度标量。环境步长内的位移上限由 `clip_unit_step` 和 `max_speed` 约束。

Critic 是 centralized critic。`build_critic_state()` 把全部无人机的 actor 特征、全部分配、全部队列、进度、SINR、速率、机间距离和交叉干扰拼成一个向量，MLP 宽度 128，输出一个标量价值。优势用这个全局价值做 GAE。Actor 更新还有 per-UAV mask：service lock 锁住的无人机不更新残差。

Reward 在 `ResidualWorker.step()`：

```text
r = payload_inc / PAYLOAD_SCALE
    - LAMBDA_PROJECT * mean ||a_exec - a_des||^2
    - LAMBDA_RESIDUAL * mean ||delta||^2
```

`payload_inc` 是这一步新增的 payload bit。干扰没有第二项惩罚；`reward_spec.py` 写明 SINR 已经进入速率和 payload。没有单独的能耗项、距离奖励或碰撞计数奖励。碰撞由 projector 和 `D_SAFE` 处理，投影失败时这一步位移被置零。

训练方式：高层冻结。MAPPO 在给定 `TaskOption` 的闭环里采样，用 GAE 和 PPO clip 同时更新 actor 与 centralized critic。低层不能在 hold 内改分配，`refuse_rematch()` 会拒绝改写 option。

# 4. 高低层交互机制

高层每 15 步输出一次 UAV–浮标匹配，不是 waypoint 序列，也不是服务顺序。低层收到的是 `TaskOption.association`：目标编号、目标位置、剩余 hold。飞行中的目标位置来自环境里的浮标坐标。

存在的反馈是状态反馈，不是代价反馈。15 步执行之后，`decide()` 读到新的位置、队列、上一拍到达和由此算出的速度。低层不能在 hold 内改高层目标。下一次高层决策也不读取低层的 reward、残差、投影失败或未来轨迹代价。

EXEC 检查点在训练时用过冻结执行器的效用标签，所以训练分布里曾经包含“这个分配交给低层后 payload 如何”。在线 `decide()` 不再调用低层来给候选打分。

因此接口是：

```text
High-level assignment
    -> Low-level residual tracking for 15 steps
    -> environment state
    -> next High-level assignment
```

不存在在线的 `Low-level cost -> High-level score`。

# 5. 当前系统效果

代码和已完成实验能够支持这些能力：

- 多机排他任务分配。依据：`decode_assignment()` 的逐步联合解码和环境里的 `conflict_mask`。
- 固定周期的目标重选。依据：`ResidualWorker._refresh_option()`，`pending` 到 0 时调用 `decide()`。
- 给定目标后的轨迹跟踪与服务。依据：L0 fly-toward、残差 MAPPO、服务半径内的进度累积，以及 `L3BEnv.step()` 把完成的服务块计入 payload。
- 通信性能进入执行结果。依据：SINR、干扰矩阵和速率决定 payload 增量。
- 机间几何进入低层动作。依据：邻居集合编码、最近距离和 safety projector。
- 集中式价值、分散式动作。依据：centralized critic 与 per-UAV actor。

当前没有接入正式规划器的部分：

- B2 用 ExtraTrees 预测 `U_traj - U_single`，只在旁路审计里给 first-hold 候选打分。`high_level.decide()` 不调用它。
- Phase-4E 对冻结 B2 + Gate-1 的独立 holdout 结果是 `DEPLOYMENT_VERIFICATION_FAILED`。`deployment_verified = False`，`planner_ready = False`。

能耗没有被优化。高层特征中的 energy 通道是 0。低层 reward 没有能量项。

# 6. 当前存在的问题

## A. 高层规划问题

高层训练优化的是分配似然，推理优化的是解码器分数下的贪心匹配。这两个目标都不是低层将要执行的 15 步 payload，也不是 `T = 48` 的累计 payload。

高层能看到当前位置和队列，所以能反映上一周期飞到了哪里。它看不到该分配的到达难度、飞行残差代价、未来 SINR 变化或动态干扰。energy 特征恒为 0。

## B. 低层轨迹优化问题

低层 reward 是当前步 payload 增量减去投影误差和残差大小。它优化的是“在已经给定的 association 下，这一拍如何移动”。它不知道本次选择对后续 option 的影响，也不能改目标。

Actor 能看到其他无人机的位置、任务和一部分通信状态，所以不是完全局部。它看不到未来到达，也没有全局任务价值作为自己的动作输入；全局信息在 critic 里，用于训练，不直接变成每架无人机的动作。

## C. 高低层耦合问题

问题 1：目标不一致。高层优化分配对数似然。低层优化 `payload_inc / scale - project - residual`。系统被记录的结果是整段 `payload_bits`。三者不是同一个目标。

问题 2：高层没有充分计入低层执行因素。在线选择不计算机到达成本、能耗、轨迹可行性或选择之后的干扰变化。只有 option 边界上的状态回传。

问题 3：低层有部分全局观测，但没有任务重规划权。邻居和当前 SINR 可见；后续任务的价值不可见，association 在 15 步内锁定。

问题 4：信息闭环不完整。存在 `Environment state -> next decide()`。不存在 `Low-level return -> current assignment score`。

问题 5：系统执行会产生 48 步累计 payload，但两个学习器都没有把这个累计回报当作自己的更新目标。高层更新匹配似然，低层更新 option 内的逐步 reward。这不是一个统一的 `max R_system`。

# 7. 最终总结

当前系统已经实现 DeepSets 联合贪心的多机浮标分配，以及在 15 步锁定目标下用 Residual MAPPO 跟踪并完成 SINR 服务，但是由于高层优化分配似然、低层优化当前步 payload 残差、在线接口只回传环境状态而不回传低层执行代价，导致目标选择和轨迹执行没有同一个长期回报，因此后续需要研究的是这两层目标为何不一致，以及 option 边界上的状态回传为什么还不能代表低层轨迹代价。

## 源文件 `HL_RL_SUPPORT_LOSS_EQUATIONS.md`

# HL-RL-SUPPORT：正式 utility loss 方程（来自磁盘代码，非推断）

本文件只记录 Phase-9.4.5 / Integration Phase-2 实际执行的公式。未改代码。

## 1. Candidate set log-likelihood \(\log P_\theta(\Gamma_j\mid s)\)

代码：

- `src/wt/htw_phase945/utility_loss.py` `candidate_logprobs`
- `src/wt/htw_phase943/set_likelihood_dp.py` `subset_dp_logprob`

对每个完整 assignment \(\Gamma\)（6 个 UAV，目标为 buoy 或 idle），在所有决定顺序上做 order-marginalized 子集 DP：

在每个已决定 UAV 子集 \(S\) 上，decoder 对尚未决定的 \((uav, token)\) 合法对做 **legal log-softmax**（`_log_softmax_legal`）。然后

\[
F(\emptyset)=0,\qquad
F(S\cup\{u\})=\mathrm{logsumexp}_{\text{orders}} \big(F(S)+\log \pi_\theta(token_u\mid S)\big).
\]

返回 \(F(\{0,\ldots,5\})=\log P_\theta(\Gamma\mid s)\)。

`set_nll = -subset_dp_logprob`（同一文件 `set_nll`）。

逐步 decoder（`decode_assignment` / `pick_candidate`）在每一步对同一 legal 集合取 argmax。该过程走完 6 步后得到一个完整 assignment。逐步 legal softmax 覆盖同一完整 assignment 空间，因此 \(\sum_{\Gamma} P_\theta(\Gamma\mid s)=1\) 在 **完整合法 assignment 空间** 上成立（不是只在 L=24 上）。本审计据此把 \(\sum_{j=1}^{24} P_\theta(\Gamma_j\mid s)\) 称为 **PoolMass**，并把 \(1-\mathrm{PoolMass}\) 记为 **OutsideMass**（对池内重复 hash 只计一次）。

## 2. Utility loss 之前是否对 candidate logP 做 softmax over L=24？

**是。** `utility_losses`：

```python
logp = candidate_logprobs(...)          # shape [L], L=24, 每项是 absolute log P_theta
log_p = torch.log_softmax(logp, dim=0)  # 仅在 24 个 candidate 上归一化
l_util = -(q * log_p).sum()
```

## 3. 实际优化的条件分布

\[
p_j=\frac{\exp(\log P_\theta(\Gamma_j\mid s))}{\sum_{l=1}^{L}\exp(\log P_\theta(\Gamma_l\mid s))}
=\mathrm{softmax}_{l\in\mathrm{pool}}(\log P_\theta(\Gamma_l\mid s))_j
\]

\[
q_j=\mathrm{softmax}\!\left(\frac{\mathrm{median\text{-}MAD}(U)_j}{\tau}\right),\quad \tau=1
\]

`soft_targets` / `normalize_state`：`src/wt/htw_phase945/utility_targets.py`。冻结：`phase945_utility_normalization.json` `method=median_mad`, `eps=1`。

\[
L_{\mathrm{utility}}=-\sum_{j=1}^{L} q_j\log p_j
\]

这是 **candidate-conditional** 交叉熵。对所有 \(\log P_j\) 加同一常数 \(c\) 不改变 \(p_j\)，因此 **不约束** candidate pool 的绝对概率质量 \(\sum_j P_\theta(\Gamma_j\mid s)\)。

## 4. Reference anchor

\[
L_{\mathrm{ref}}=\mathrm{set\_nll}(\Gamma_{\mathrm{ref}}\mid s)=-\log P_\theta(\Gamma_{\mathrm{ref}}\mid s)
\]

\(\Gamma_{\mathrm{ref}}\) 是数据集中的 oracle/reference assignment（`train["assignment"]`），**使用绝对 set probability**，不经过 pool softmax。权重 \(\lambda_{\mathrm{ref}}=0.10\)。

OLD-HL 与 NEW-HL 的 Phase-9.4.5 / Phase-2 训练都含此项；NEW-HL 未改 \(\lambda_{\mathrm{ref}}\)。

## 5. Ranking loss

`ranking_pairs`：仅当 \(|U_i-U_j|>\delta\)，\(\delta=\max(0.05\cdot\mathrm{range}(U),1)\)。

\[
L_{\mathrm{rank}}=\sum_{(i,j)} w_{ij}\,\mathrm{softplus}\big(-y_{ij}(\log P_i-\log P_j)\big)
\]

差 \(\log P_i-\log P_j\) **平移不变**，因此 ranking 也 **不约束** 绝对 pool mass。权重 \(\lambda_{\mathrm{rank}}=0.25\)。使用的是 absolute logP 的差，等价于 pool-conditional logits 的差。

## 6. Absolute vs pool-conditional

| 项 | 绝对 \(P_\theta(\Gamma\mid s)\) | 仅 pool-conditional \(p_j\) |
|---|---|---|
| \(L_{\mathrm{utility}}\) | 否（先 `log_softmax` over L=24） | 是 |
| \(L_{\mathrm{rank}}\) | 差，平移不变 | 等价于 conditional |
| \(L_{\mathrm{ref}}\) | 是 | 否 |

总损失（`phase945_loss_spec.json`）：

\[
L=L_{\mathrm{utility}}+0.25\,L_{\mathrm{rank}}+0.10\,L_{\mathrm{ref}}
\]

自由解码 **不** 在 L=24 上 argmax，而是逐步 `pick_candidate`。因此提高 \(p_j\) 不必提高 \(P_\theta(\Gamma_j)\) 相对 pool 外 assignment 的质量。

## 源文件 `HL_SEARCH_DECODER_EQUATIONS.md`

# Free decoder vs order-marginalized set likelihood

Source of truth: `wt.htw_phase943.evaluation.decode_assignment`, `pick_candidate`, `set_likelihood_dp.subset_dp_logprob`.

This document does **not** change inference. It records the frozen equations.

## 1. Candidate at each step

Partial assignment \(A\in\{-2\}\cup\{-1\}\cup\{0,\ldots,11\}^6\). Undecided UAV is \(-2\). Idle is \(-1\). Buoy index is \(0\ldots 11\).

A candidate is a pair \((m,k)\):

- \(m\in\{0,\ldots,5\}\) an undecided UAV
- \(k\in\{0,\ldots,11,12\}\) a buoy token or idle token `IDLE_INDEX=12`

Legal mask (`legal_candidate_mask`):

- already-decided UAV: no candidates
- remaining UAV: every unoccupied available buoy, plus idle always

Choosing \((m,k)\) writes \(A_m \leftarrow -1\) if \(k=12\), else \(A_m \leftarrow k\), and that buoy becomes occupied.

## 2. Score / logit

Encoder is frozen DeepSets. Decoder is permutation-equivariant `OrderFreeSetDecoder`.

Given partial \(A\), it builds set summaries of decided edges, remaining UAVs, remaining buoys, plus candidate physics, then a shared MLP scores every legal \((m,k)\). Illegal cells are \(-10^9\).

## 3. Free decoder is greedy joint argmax

`decode_assignment` loops \(M=6\) times:

1. score all legal \((m,k)\)
2. `pick_candidate`: \(\arg\max\) over legal logits
3. ties within `TIE_ATOL=1e-6` broken by `canonical_keys` (UAV xy, buoy xy, idle flag, \(-rate\)), not by sampling unless `shuffle_seed` is set

Official free decode does **not** sample and does **not** search a beam. It is **not** UAV-then-buoy sequential softmax. It is joint greedy over the same \((m,k)\) grid that the legal log-softmax uses.

## 4. Context update

After \((m,k)\): that UAV is decided; if \(k\) is a buoy it is occupied; remaining-UAV / remaining-buoy DeepSets summaries and physics are recomputed from the unordered partial set. Order of previous choices is not stored as a sequence in the state, only as the set of decided edges.

## 5. Idle

Idle is always a legal token for every undecided UAV. It competes in the same joint softmax as buoys.

## 6. Sequence / path log-probability

A generation path is an ordered list of legal edges \((m_t,k_t)_{t=1}^{6}\).

At partial \(A^{(t)}\):

\[
\log P_{\mathrm{path}}(A^{(t+1)}\mid A^{(t)})
=
\log
\frac{\exp z_{m_t k_t}(A^{(t)})}
{\sum_{(m',k')\in\mathrm{Legal}(A^{(t)})}\exp z_{m'k'}(A^{(t)})}
\]

\[
\log P_{\mathrm{path}}(\pi,A)
=
\sum_{t=1}^{6}
\log P_{\mathrm{path}}(A^{(t)}\mid A^{(t-1)})
\]

Beam search (Phase-2A diagnostic, unused at official inference) ranks incomplete hypotheses by this **path** sum.

## 7. Unordered set log-probability (training)

Training likelihood is order-marginalized set probability (`subset_dp_logprob`):

\[
\log P_{\mathrm{set}}(\Gamma\mid s)
=
\mathrm{logsumexp}_{\pi\in S_6}
\log P_{\mathrm{path}}(\pi,\Gamma\mid s)
\]

implemented as subset DP over UAV subsets: at mask of decided UAVs, legal log-softmax, then add the unique remaining UAV that matches \(\Gamma\).

This is **not** a score inside the L=24 pool. The pool softmax was Phase-2 COND loss; Phase-2B \(L_{\mathrm{abs}}\) uses these absolute \(\log P_{\mathrm{set}}\).

## 8. Distinctions that this audit tests

| Object | Maximized by official greedy? | Used in ABS training? |
|---|---|---|
| joint one-step logit / path | yes, myopically | no, except as DP factors |
| full path probability of one order | only implicitly | no |
| unordered \(P_{\mathrm{set}}\) | **not guaranteed** | **yes** |

`training_inference_objective_mismatch` is true if greedy/beam-path often misses the assignment with highest \(P_{\mathrm{set}}\).

## 源文件 `HTW_Final_Code_and_Reproducibility_Report_CN.md`

# HTW 最终代码冻结与复现包报告（Phase-11）

本阶段是工程收尾，不是科研实验。未重新训练，未改方法，未改 checkpoint，未重跑 Phase-10.6，未访问 final untouched。

## 1. 最终方法是什么？

FINAL-HIER =

DeepSets-util
+ Permutation-Equivariant Order-Free Global Assignment Decoder
+ Utility-Aligned Supervision
+ Task Option
+ hold=15
+ fly-toward
+ InRangeZeroMotionGate

`HGNN_selected=false`，`low_level_PPO_selected=false`，`Transformer_selected=false`。这些只作为历史实验与消融证据保留。

流水线：state → DeepSets → Order-Free Decoder → assignment → Task Option → fly-toward → InRangeZeroMotionGate → movement → L3-B environment。

## 2. 最终 checkpoint 在哪里？

`final_artifacts/checkpoints/deepsets_util_seed0.pt`

源归档（同一文件）：`src/wt/htw_phase100/outputs/handoff/frozen_deepsets_util/deepsets_util_seed0.pt`

训练：Phase-9.4.5 utility-aligned DeepSets，seed=0，best epoch=60。不是根据 Phase-10.6 重选的。

## 3. checkpoint SHA？

`dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109`

与 Phase-10.5 / 10.6 使用的文件一致。

## 4. 最终配置在哪里？

`final_artifacts/configs/`：`environment` / `model` / `decoder` / `executor` / `evaluation` 的 json 与 yaml。

来自当前代码：`L3BConfig`、`HOLD`、decoder 常量、gate 规则。M=6，K=12，T=48，hold=15，service_radius=16 m，step_length=22 m。环境 SHA：`de22347ea625a330f61e8f62b2a973066f4669c7498fd58bac2fd8bf706632ba`。

## 5. 最终运行入口是什么？

```bash
python -m src.wt.final.run_final_hier
```

等价：`PYTHONPATH=src` 后 `python -m wt.final.run_final_hier`。

## 6. demo 怎么运行？

```bash
python -m src.wt.final.run_final_hier --seed 8
```

默认 seed=8（design/demo）。入口会拒绝 `106000–106063`。本轮 demo 实测：payload=1.2e7，completed_blocks=12，flight=330，legality=true，untouched_used=false。这是普通 demo 场景，**不是**论文主表。

## 7. 最终绘图怎么运行？

```bash
python scripts/plot_final_results.py
```

只读取已保存的 Phase-10.6 CSV/JSON，不 import 环境、不跑模型。图在 `final_artifacts/figures/`。

## 8. 论文结果表在哪里？

- `final_results_table.csv` 与 `final_artifacts/results/final_results_table.csv`
- `final_pairwise_statistics.csv` 与 `final_artifacts/results/final_pairwise_statistics.csv`

Phase-10.6 磁盘摘要（沿用，未重算口径）：

| method_id | method | payload | completed_blocks | residual_queue | drop | flight | legality |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | Myopic | 8984375 | 8.984 | 30.016 | 78.000 | 80.4 | 1 |
| M1 | Exclusive | 1718750 | 1.719 | 37.328 | 77.953 | 129.9 | 1 |
| M2 | DeepSets-old | 6703125 | 6.703 | 33.609 | 76.688 | 249.9 | 1 |
| M3 | FINAL-HIER | 10218750 | 10.219 | 30.969 | 75.812 | 456.5 | 1 |

配对 payload（10.6 bootstrap）：M3−M0 mean 1.234e6，CI [5.31e5, 1.969e6]，相对增益 0.137；M3−M2 mean 3.516e6，CI [2.70e6, 4.28e6]。

## 9. 消融总结在哪里？

`FINAL_ABLATION_SUMMARY_CN.md`

## 10. 历史代码是否保留？

是。Phase-1～Phase-10.6、HGNN、低层 PPO、Transformer、旧 TW 均未删除。最终 facade 在 `src/wt/final/`，与历史目录分开。

## 11. final untouched 是否再次访问？

**否。** 只读取已保存的 CSV/JSON。demo 入口拒绝 106000–106063。未重新运行 untouched 环境。

## 12. 是否重新训练？

**否。**

## 13. 所有工程测试是否通过？

是。`python -m unittest tests.test_htw_phase11`：15/15 OK。覆盖 SHA、加载、demo、shape、assignment 合法、idle token、Order-Free 接口、hold=15、fly-toward、gate、关 gate 与旧规则接口、绘图不调环境、demo 不用 untouched、配置与 10.6 一致、无 NaN/Inf。

## 14. 是否存在可能影响科研结论的 bug？

未发现 B 类（可能影响历史科研结果）问题。

本阶段修过一处纯工程问题：绘图脚本用源码字符串自检 `l3b.env`，自检语句自身触发误报；已删除该自检，测试仍用 AST 禁止环境 import。不影响任何历史数字。

## 15. 代码开发是否正式完成？

是。`FINAL_CODE_FREEZE.json`：

- architecture_finalized=true
- training_finished=true
- method_development_stopped=true
- final_untouched_consumed=true
- scientific_experiments_finished=true
- code_development_complete=true
- reproducibility_package_complete=true
- checkpoint_sha=上述 SHA
- code_freeze_timestamp=2026-09-18T21:32:14+08:00

冻结状态来源核对（只读，未改历史文件）：Phase-10.6 `phase106_final_scientific_status.json` 中 architecture_finalized / training_finished / method_development_stopped / final_untouched_consumed / final_system_confirmed 均为 true，`final_gate=A`（即 FINAL-A）。

运行时版本（现有环境，未升级）：Python 3.10.20，PyTorch 2.7.1+cu128，NumPy 1.26.4，matplotlib 3.10.9。见 `final_artifacts/requirements.txt`。仓库根目录 `requirements.txt` 仍是早期 Decision Transformer 依赖，不要盲目替换。

后续只做论文写作、图表整理、方法说明、结果讨论和答辩材料。不再启动训练、调参、PPO、HGNN、改 hold/gate/reward，或再次跑 untouched。

## 源文件 `HTW_Phase3_Failure_Diagnosis_CN.md`

# HTW Phase-3 Failure Diagnosis

本报告只做离线审计。未训练、未改现有方法、未改 checkpoint、未重跑 Phase-3、不进入下一阶段。

审计输入：冻结的 `hl_exec_align_{train,val}.npz`（SHA `ff493cf0828e3fa8232b4d5d54cfcee7d34dddc1f42976bc6e9670beaefa3f57`）、`deepsets_exec_seed{0-4}.pt`、`deepsets_prefix_seed{0,1}.pt`、Phase-3/2D/2B 的 Decode / Within-pool / closed-loop / training history CSV，以及 `shadow_h15` / `evaluate_design` / `decide` / `executor_targets` / `utility_losses_*` 源码。

已确认：`repeated_low_level_rollout_in_training = false`，`code_changed = false`。失败不归因于重复 rollout 或 cache。

---

## 1. Phase-3 失败的事实

Gate = C；`confirmed = false`。四项门控全部失败。

| 方法 | closed-loop payload | DecodeNR | Spearman | Kendall | NOM5 | payload std |
|---|---:|---:|---:|---:|---:|---:|
| OLD-HL + RL | 1.09125e7 | 0.2500 | 0.1593 | 0.1853 | 0.2913 | （单 checkpoint） |
| ABS-HL + RL | 8.44e6 | 0.3258 | 0.3944 | 0.3386 | 0.4808 | 高于 PREFIX |
| PREFIX-HL + RL | **1.12875e7** | **0.2375** | **0.4239** | **0.3640** | 0.4373 | **1.061e5** |
| EXEC-HL + RL | 8.6925e6 | 0.5000 | 0.3976 | 0.3385 | 0.4372 | 2.212e6 |

EXEC 五 seed closed-loop：5.55e6 / 1.05625e7 / 7.275e6 / 1.0575e7 / 9.50e6。seed0 崩塌到 5.55e6。

结论（已冻结）：executor-conditioned supervision **没有**同时改善 closed-loop payload、DecodeNR、seed stability 和 utility alignment。

---

## 2. Candidate utility 与 High-Level policy alignment

问题：DecodeNR=0.5 是因为 (A) 网络没学到 \(U_{LL}\)，还是 (B) \(U_{LL}\) 对最终 assignment 的排序本身不稳定？

在 val 的 32 个 state、每个 state 的 L=24 pool 上，用冻结 checkpoint 的 \(P_\theta=\log P_{\mathrm{set}}(\Gamma\mid s)\) 对 \(U_{LL}\) 计算 alignment。正式 Spearman/Kendall/NOM5 仍以 Phase-3 报告为准（`pool_metrics`）。本诊断额外计算 top-1 / top-3 / top-25% / NDCG / 归一化 regret。

定义：`regret = (U_best - U_selected) / max(1, |U_best|)`，`U_selected = U[argmax P]`。

| 模型 | top-1 agree | top-3 overlap | top-25% overlap | NDCG | regret mean / median / p90 |
|---|---:|---:|---:|---:|---:|
| EXEC seed0–4 均值 | 0.4125 | 0.489 | 0.534 | 0.899 | 0.145 / **0.0** / ~0.39 |
| PREFIX seed0–1 均值 | 0.4063 | 0.510 | 0.555 | 0.897 | 0.143 / **0.0** / ~0.48 |

分 seed（EXEC）：

| seed | top-1 | top-3 | top-25% | NDCG | regret mean | 正式 Spearman | 正式 NOM5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.469 | 0.510 | 0.536 | 0.911 | 0.117 | 0.380 | 0.460 |
| 1 | 0.375 | 0.479 | 0.542 | 0.879 | 0.190 | 0.407 | 0.432 |
| 2 | 0.406 | 0.458 | 0.505 | 0.906 | 0.118 | 0.368 | 0.377 |
| 3 | 0.438 | 0.500 | 0.536 | 0.906 | 0.147 | 0.414 | 0.462 |
| 4 | 0.375 | 0.500 | 0.552 | 0.891 | 0.154 | 0.419 | 0.455 |

**对 A/B 的判定：**

- **不是单纯的 A。** 网络在 pool 内学到了中等偏上的 \(U_{LL}\) 排序：正式 Spearman≈0.40，NOM5≈0.44，诊断 NDCG≈0.90，regret 中位数为 0。EXEC 与 PREFIX 的 **pool 内 alignment 几乎相同**（top-1 0.41 vs 0.41）。
- **B 在标签层是真的，但不是 EXEC 独有的 DecodeNR 原因。** 同一 assignment hash 在不同 `(scenario,t)` 上的 \(U_{LL}\) 方差很大（第 4 节）。PREFIX 用的是同一套 \(U_{LL}\)，DecodeNR 却是 0.2375。
- **DecodeNR=0.5 的直接原因是 official decode 几乎不在被监督的 24-pool 里选点。**

| 模型 | in_pool | OOD DecodeNR median | \(U_{\mathrm{decode}}\) mean | \(U_{\mathrm{best,pool}}\) mean |
|---|---:|---:|---:|---:|
| EXEC seed0 | 0.031 | 0.600 | 1.50e6 | 3.51e6 |
| EXEC seed1 | **0.000** | 0.480 | 1.93e6 | 3.51e6 |
| EXEC seed2 | 0.094 | **0.750** | **1.07e6** | 3.51e6 |
| EXEC seed3 | 0.031 | 0.400 | 2.13e6 | 3.51e6 |
| EXEC seed4 | 0.031 | 0.600 | 1.86e6 | 3.51e6 |
| PREFIX seed0 | 0.094 | 0.250 | **2.58e6** | 3.51e6 |
| PREFIX seed1 | 0.094 | 0.250 | 2.25e6 | 3.51e6 |
| OLD | **0.156** | 0.250 | 2.23e6 | 3.51e6 |

代码：`eval_model` 用 `greedy_trace` 得到 assignment，再用 `u_new`（pool 命中则读标签，否则 shadow）算 \(U_{\mathrm{decode}}\)；`DecodeNR=(U_{best,pool}-U_{decode})/(|U_{best,pool}|+\epsilon)`。`in_pool = assignment_hash(greedy) in pool_hashes`。

EXEC 五 seed 平均 in_pool = **0.0375**。因此 DecodeNR 比较的是 **pool 外 greedy assignment 的 \(U_{LL}\)** 与 **pool 内最优 \(U_{LL}\)**，不是 `argmax P` 对 `argmax U` 的 pool 内 regret。PREFIX 同样大量 OOD，但其 OOD \(U_{\mathrm{decode}}\) 均值约 2.41e6，EXEC 约 1.70e6。

**本节结论：** EXEC 的 DecodeNR=0.5 主要不是 “没学到 pool 内 \(U_{LL}\)”（否决纯 A），也不是 “\(U_{LL}\) 排序完全不可用”（否决纯 B 作为 EXEC 特有原因）。它是 **监督在 24-pool 的 set-logP 上，推理却是全空间 joint greedy**，且 EXEC 的 OOD greedy 比 PREFIX 差。

---

## 3. 训练监督 \(U_{LL}\) 与最终 closed-loop payload 的一致性

这是本诊断最重要的一条。只报告代码实际在算什么。

### 3.1 \(U_{LL}\) 实际是什么

`dataset.py` 标签：`U_LL_payload_increment`。来源是 Phase-2 `generate_state_utilities`：

1. `replay_to_hold(scenario, t)`：从 `env.reset(scenario)` 用 **`oracle_option_assignment` + `fly_toward`** 步进到时刻 `t`（不是 EXEC / PREFIX / RL 闭环轨迹）。
2. 对每个 candidate（完整 6-UAV assignment）调用 `shadow_h15`：
   - `restore_worker` 回到该 hold snapshot；
   - `bind_assignment` 冻结该 assignment，`pending=HOLD_STEPS=15`；
   - 走最多 15 步 Low-Level（`zero_residual=False`，即真实 RL residual）；
   - `U = env.payload_bits - start_payload`。
3. 对 5 个 Low-Level seed 取均值。

因此 \(U_{LL}\) **不是**：

- 单个 UAV–浮标边的 immediate increment；
- 完整 H=48 / 整局 closed-loop payload；
- 多个 hold 的累计 assignment 序列效用。

它是：**在 oracle 回放到的某一 hold 状态上，把某一个完整 assignment 冻结 15 步后的 payload 增量。**

### 3.2 最终 closed-loop 实际是什么

`evaluate_design`（Phase-2B eval，Phase-3 复用）：

1. `worker.reset(seed)`，从 **t=0** 开始，不是 hold snapshot；
2. 循环到 `episode_horizon`（设计集每局 `n_steps=48`，16 scenario × 48 = CSV 中的 768）；
3. `payload = payload_bits - payload0`（整局累计）；
4. High-Level 在 `pending<=0` 时调用 `decide` → `decode_assignment`（joint greedy），**不是**从 L=24 pool 取 `argmax P`；
5. `CompatWorker.frozen_assignment` 在闭环里保持 `None`，因此每个 hold **重新 decode**，不是 `shadow_h15` 那种冻结 15 步。

闭环 CSV 实测：所有方法 `high_level_decision_count = 4`；assignment 切换次数均值 EXEC 2.525、PREFIX 2.375、ABS 2.53、OLD 2.825。一局有 **4 个会互相改变后续状态的 assignment**。

### 3.3 中间断层（代码证据，不是理论猜测）

```
oracle 回放到达的 hold snapshot
        ↓
candidate-level U_LL = 冻结该 assignment 的 15 步 payload 增量
        ↓
L_exec 只提高这 24 个 set 的加权 log P（绝对 set-logP，不是 pool 上的 log_softmax）
        ↓
official / 闭环 decode = 全空间 joint greedy（in_pool ≈ 3–9%）
        ↓
闭环从 t=0 走 48 步、重规划 4 次
        ↓
最终 payload = 整局 payload_bits 增量
```

已确认的不一致：

| 断层 | 证据 |
|---|---|
| candidate-conditioned 执行 ≠ 实际选中 assignment | 监督的是 pool 内 24 个 assignment；闭环 `decide()` 走 greedy，EXEC in_pool 均值 3.75%。 |
| 单 candidate utility ≠ 组合 assignment 序列 utility | \(U_{LL}\) 只标一个 hold；闭环 4 次决策、约 2.5 次切换。没有序列标签。 |
| 状态转移依赖 | 后一个 hold 的状态由前一个 assignment + RL 执行产生；标签状态由 **oracle + fly_toward** 产生（`replay_to_hold`）。 |
| 后续 15 步 interaction | `shadow_h15` 冻结 assignment 15 步；闭环每个 hold 结束就重绑。 |
| 非加性 | 没有 “(A then B)” 的联合 \(U\)。4 × 均值 \(U_{LL}\)（1.645e6）≈ 6.58e6，低于 PREFIX/OLD 整局 ~1.1e7，也不同于 EXEC 的 8.69e6；数量级对不上简单相加。 |
| candidate 竞争 | 每个 candidate 已是完整 6-UAV assignment。闭环竞争发生在 **时间上相继的 4 个 assignment** 之间，不在 24-pool 的一次选择里。 |

补充：val snapshot 上的 \(U_{\mathrm{decode}}\) 与闭环 payload **不是单调关系**。EXEC seed2 的 \(U_{\mathrm{decode}}\) 均值最低（1.07e6）但闭环 7.275e6；seed0 的 \(U_{\mathrm{decode}}\) 1.50e6 反而闭环 5.55e6。这是 “单 hold 标签效用 ≠ 整局闭环效用” 的直接数据。

---

## 4. Context / assignment dependence

每个 candidate 就是一个完整 6-UAV assignment（hash = assignment digest）。比较同一 hash 在不同 `(scenario, t)` 上的 \(U_{LL}\)。

**Train（64 state × 24 = 1536）：**

- unique hashes = 1103；reused hashes = 196（17.8%）；reused 观测 = 629
- 全局 var = 2.50e12
- between-state（state 均值的 var）份额 ≈ 0.53
- within-state（state 内 24 个 candidate 的 var 的均值）份额 ≈ 0.50
- **同一 hash 跨 state 的 within-var 均值 = 1.40e12，中位数 = 3.67e11**
- 同一 scenario、不同 t 的 within-var 均值 = 1.96e12

**Val（32 × 24 = 768）：** reused 18.4%（105/571）；within-reused var 均值 8.76e11。

\(\sqrt{1.40\times 10^{12}}\approx 1.18\times 10^{6}\)，与 \(U_{LL}\) 总体 std 1.52e6 同量级。

**判定：** \(U_{LL}\) **不能** 写成 \(U(\mathrm{candidate})\)。数据支持

\[
U_{LL}=U(\mathrm{candidate}\mid \mathrm{state},\ \mathrm{oracle\text{-}replayed\ context},\ t)
\]

其中 context 至少包括：时刻 t、队列/进度/payload（snapshot 内）、UAV/浮标几何。同一 assignment 在 t=0 与 t=15 可以差一个数量级。

因此，把 \(q_i=U_i/\sum U\) 当作 “这个 assignment 的固有好坏” 做 candidate-level 监督，存在结构性错位：网络看到的是 **条件效用**，目标却被写成 **与后续闭环状态无关的静态 q**。

这一点 PREFIX/ABS/EXEC **共享**，它解释 “为何 candidate-level 真 \(U_{LL}\) 仍推不动闭环”，不单独解释 PREFIX > EXEC。

---

## 5. Utility 稀疏性与尺度

全量 train+val：n=2304，mean=1.645e6，std=1.522e6，min=0，max=7e6，非正个数=629。

| 统计 | 值 |
|---|---:|
| \(U_{LL}=0\) 比例 | **0.2730** |
| near-zero (\(\lvert U\rvert<1\)) | 0.2730（与零重合，没有介于 0 与 1 的碎值） |
| top 5% utility 占总正效用 | 0.1609（全局并不极端尖） |
| 每 state utility range 均值 | 3.731e6 |
| 每 state CV 均值 | 1.010 |

OLD / ABS / PREFIX / EXEC **共用同一套 \(U_{LL}\)**（Phase-2 SHA 一致）。四者的差异在 **q 的构造** 和 **likelihood**，不在原始 U 的分布。

监督目标分布对比（同一 U，不同 q）：

| q 构造 | 使用方法 | entropy 均值 | \(n_{\mathrm{eff}}=e^{H}\) | \(q_{\max}\) 中位数 |
|---|---|---:|---:|---:|
| 线性 `U/sum(U)`，非正则 shift | **EXEC** | 2.667 | **17.05** | 0.095 |
| 不 shift 的 `U/sum(U+)` | 对照 | 2.469 | — | — |
| `soft_targets`（MAD + τ=1 softmax） | OLD / ABS / PREFIX | **1.840** | **8.69** | **0.344** |

EXEC 的 q 把有效支持摊到约 17/24 个 candidate；PREFIX/ABS 的 q 更尖（有效约 9 个，`q_max` 中位数 0.34）。`q_i>0.05` 的 EXEC 支持均值只有 7.2，但长尾仍占大量质量。

---

## 6. \(q_i\) normalization 是否扭曲监督

代码（`exec_loss.executor_targets`）：

- 默认 \(q_i=U_i/\sum U\)；
- 若 `any(U<=0)`：\(U'=U-\min(U)+\epsilon\)，再归一化。

审计结果：

- **98.96% 的 state 触发了 shift**（几乎每个 state 都有至少一个 \(U=0\)）。
- 全部非正值都是 **精确 0**，没有负效用。因此 shift 就是 \(U'=U+\epsilon\)。
- **shift 前后 ranking 不变：`rank_unchanged_rate=1.0`。** `U-\min(U)+\epsilon` 是严格保序变换，不会把优劣颠倒。
- shift 后 entropy 从 2.469 升到 2.667：原来 \(q=0\) 的零效用 candidate 获得 \(\epsilon/\sum\) 的微小质量，分布更平。
- 与 PREFIX/ABS 的 `soft_targets` 相比：`L1(q_exec,q_soft)` 均值 **0.736**；约 10.4% state 的 `soft_targets` 排序与原始 U 不完全一致（近并列 + MAD/softmax 数值），但这不是 shift 造成的。

**会不会 “把接近的 utility 拉开/压扁”？**

- 拉开/压扁的是 **相对质量**，不是名次。
- 线性归一化 + 给零效用发 \(\epsilon\)，系统性把 q **压扁**（\(n_{\mathrm{eff}}\) 17 vs 8.7）。高 U candidate 的 \(q_{\max}\) 中位数从 soft 的 0.34 降到 0.095。
- 这会让 \(L_{\mathrm{exec}}=-\sum q_i\log P_{\mathrm{set}}(\Gamma_i)\) 接近 “对几乎整个 pool 做加权 NLL”，而不是 “集中推高少数高 \(U_{LL}\) assignment”。

**但是：** ABS 用尖的 `soft_targets` + **同一条** 绝对 set-logP，闭环 8.44e6；EXEC 用扁的线性 q + 同一条 set-logP，闭环 8.69e6。两者接近。因此 **q 压扁不是 PREFIX（1.13e7）对 EXEC 的主因**；它解释监督更散，不解释闭环比 PREFIX 低 2.6e6。

---

## 7. High-Level 训练过程到底在优化什么

可审计的完整 epoch 曲线只有 **EXEC seed4**（`HL_EXEC_TRAINING_HISTORY.csv`）。checkpoint 规则：`best_validation_mean_candidate_regret`，seed4 选中 epoch 70。

| epoch | \(L_{\mathrm{exec}}\) | val Spearman | val NOM5 | val top1 regret |
|---:|---:|---:|---:|---:|
| 1 | 14.454 | — | — | — |
| 10 | ~11.3 量级（loss 已降） | 0.257 | 0.236 | 756250 |
| 70（选中） | ~10.32 | 0.419 | 0.455 | 318750 |
| 80（last） | 10.851 | 0.432 | 0.425 | 325000 |

PREFIX 对照（同一套 \(U_{LL}\) + ranking/ref）：seed0 的 \(L_{\mathrm{prefix}}\) 21.18→17.15，val Spearman 0.235→0.435，选中 epoch 80，regret 325000；seed1 选中 epoch 70。

**训练过程发生了什么：**

1. \(L_{\mathrm{exec}}\) 下降，pool 内 Spearman / NOM5 上升、regret 下降。**训练目标在自己的 val 指标上有效。**
2. seed4 闭环 payload 仍是 9.50e6 < PREFIX 1.129e7，且 DecodeNR median=0.50。
3. 其他 seed 的 pool 指标与 seed4 接近（Spearman 0.37–0.42，NOM5 0.38–0.46），闭环却从 5.55e6 到 1.06e7。**pool 排序改进 ≠ 闭环改进，也不保证 seed 稳定。**

这就是

\[
\text{training objective improvement}\ \ne\ \text{closed-loop execution improvement}
\]

的具体证据：优化的是 “oracle-hold 上 24 个 set 的加权绝对 logP + pairwise rank”；闭环要的是 “从 t=0 起 4 次 greedy 重规划的整局 payload”。

---

## 8. OLD / ABS / PREFIX / EXEC 失败机制对比

四者 **标签 \(U_{LL}\) 相同**。差异只在 q 与 likelihood。

| 方法 | supervision target | likelihood | DecodeNR | Spearman | NOM5 | payload | seed std | 主要失败机制 |
|---|---|---|---:|---:|---:|---:|---:|---|
| OLD | `soft_targets(U,τ=1)` | **pool 上 log_softmax**（条件） | 0.250 | 0.159 | 0.291 | 1.091e7 | n=1 | pool 排序最弱；但 in_pool 最高（15.6%），idle 最低（0.553），闭环仍高 |
| ABS | 同上 soft q | **绝对 set-logP** \(-\sum q\log P_{\mathrm{set}}\) | 0.326 | 0.394 | **0.481** | 8.44e6 | 高（seed4 DecodeNR median 0.925） | 排序变好，greedy 不稳定，闭环掉到 8.44e6 |
| PREFIX | 同上 soft q | **路径/前缀局部 \(\log\pi\)** | **0.238** | **0.424** | 0.437 | **1.129e7** | **1.06e5** | pool 外 greedy 的 \(U_{\mathrm{decode}}\) 最高；闭环最好 |
| EXEC | **线性 \(q=U/\sum U\)**（零值 shift） | **绝对 set-logP**（与 ABS 同构） | 0.500 | 0.398 | 0.437 | 8.69e6 | 2.21e6 | 与 ABS 同族；q 更扁；OOD greedy 更差；seed0 崩塌 |

**为什么 PREFIX 的 1.12875e7 高于 EXEC 的 8.6925e6，即使 EXEC “直接使用真实 Low-Level utility”？**

数据不允许把答案写成 “EXEC 没用真 utility”。EXEC **没有引入新的 executor 标签**，只是把同一套 \(U_{LL}\) 的 q 从 `soft_targets` 改成线性归一化，likelihood 仍是 ABS 那种绝对 set-logP。

对照：

1. **EXEC ≈ ABS（8.69e6 vs 8.44e6）**，二者 likelihood 相同、q 不同 → 线性 executor q **不是**相对 PREFIX 掉 2.6e6 的主因。
2. **PREFIX ≫ ABS ≈ EXEC**，PREFIX 与 ABS 的 q 相同、likelihood 不同 → **路径一致的局部 \(\log\pi\)（对齐 official greedy）才是三条 \(U_{LL}\) 方法里闭环差距的主对照。**
3. EXEC/PREFIX 的 **pool 内** top-1 / NDCG / Spearman 接近，但 PREFIX 的 OOD \(U_{\mathrm{decode}}\) 高约 40%，DecodeNR 低一半。
4. “真实 \(U_{LL}\)” 本身只是单 hold、oracle 状态上的冻结增量，不是闭环 4-hold 轨迹效用。把它线性塞进 set-NLL，并不能让 greedy 在整局里选出高 payload 序列。

OLD 的 Spearman 最差（0.16）但闭环高于 EXEC：再次说明 **pool 内 \(P\)–\(U\) alignment 不是闭环的充分统计量**。

---

## 9. 最可能的结构性原因（必须有数据/代码证据）

不是单一口号，而是三条互相咬合、都有证据的结构：

**结构 1 — 监督对象不是闭环对象。**  
\(U_{LL}=\) oracle-hold 上冻结 15 步的 payload 增量；闭环 \(=\) t=0 起 48 步、4 次 greedy 重规划的整局 `payload_bits`。标签状态分布（`replay_to_hold` + oracle）≠ EXEC 闭环状态分布。seed0/seed2 的 \(U_{\mathrm{decode}}\) 与闭环交叉，说明单 hold 效用排不好整局。

**结构 2 — 监督的似然不是推理的 decode。**  
\(L_{\mathrm{exec}}=-\sum q_i\log P_{\mathrm{set}}(\Gamma_i\mid s)\) 不在 24-pool 上做 `log_softmax`，也不训练 greedy 的逐步 \(\log\pi\)。官方推理是 `decode_assignment` / `greedy_trace`。EXEC in_pool 均值 3.75%，PoolMass 中位数 \(3\times10^{-4}\) 量级。PREFIX 用同一 \(U_{LL}\) 却训练局部路径，闭环与 DecodeNR 都更好。ABS 与 EXEC 同构，闭环同档。

**结构 3 — candidate-level \(q_i\) 假定了 \(U(\Gamma)\)，数据是 \(U(\Gamma\mid s,t)\)。**  
同一 assignment hash 跨 state 的 \(U_{LL}\) 方差与全局方差同量级。简单的 \(q_i=U_i/\sum U\)（再 shift）不能表达 context。q 压扁（\(n_{\mathrm{eff}}\approx17\)）加重了 “对一池条件效用做平均拟合”，但对照 ABS 可知这不是 PREFIX 差距的主项。

这三条合在一起回答标题问题：

> 为什么已经用 candidate-level 真实 Low-Level closed-loop utility \(U_{LL}\) 做 High-Level supervision，EXEC-HL 仍然无法稳定产生高 closed-loop payload？

因为所谓 “真实 closed-loop utility” 在代码里 **并不是** 最终评估的那条闭环轨迹效用；监督也 **没有** 约束最终用来产生轨迹的 greedy decoder；而 \(U_{LL}\) 还是 **强依赖 state/time 的条件量**。Phase-3 只改了 q 的归一化，没有补上这三处错位。

不归因于：重复 Low-Level rollout、cache、改代码、没学到任何 pool 排序。

---

## 10. 下一阶段候选方向（只提机制，不实现、不训练）

原则：先由诊断决定机制，不为了继续实验而强行加 loss。

| 方向 | 诊断是否支持 | 依据 | 是否值得现在就做 |
|---|---|---|---|
| A. state/context-conditioned utility | **支持** | 同 hash 跨 `(scenario,t)` 的 U 方差 ≈ 全局方差 | 作为标签定义问题，是的 |
| B. assignment-**序列** / 整局 utility | **支持** | 4 hold、~2.5 次切换；单 hold \(U_{\mathrm{decode}}\) 与闭环交叉 | **最值得先验证** |
| C. pairwise preference | 弱 | `L_rank` 已存在且权重相同；EXEC 与 PREFIX 的 pool 排序接近 | 否，不是缺口 |
| D. listwise 再改 q | 弱 | Phase-3 已经 listwise 线性 q；ABS 尖 q 闭环仍差 | 不要再为实验改 q |
| E. counterfactual executor utility | 部分已做 | \(U_{LL}\) 已是 hold 上的反事实冻结增量；缺的是 **策略诱导状态 + 多 hold** | 要扩展定义，不是重做 hold-15 |
| F. utility + ranking 联合 | 已在用 | `0.25 L_rank + 0.10 L_ref` 四方法共用 | 不是新方向 |
| G. decode-path / greedy-consistent supervision | **支持（PREFIX 已是对照）** | PREFIX ≫ ABS≈EXEC，q 相同、likelihood 不同 | 不要再发明一条 set-NLL 变体；若做，应先 **测量** greedy 轨迹的整局 U，而不是先写新 loss |

不自动进入下一阶段。不实现上述任何一项。

---

## 已确认的问题

1. Phase-3 Gate C：EXEC 闭环 8.6925e6 < OLD 1.09125e7 与 PREFIX 1.12875e7；DecodeNR 0.50 > PREFIX 0.2375；seed std 2.212e6；Spearman 0.3976 < PREFIX 0.4239。
2. 失败与 `repeated_low_level_rollout_in_training` / cache / `code_changed` 无关。
3. \(U_{LL}\) 是 oracle-hold 上冻结 15 步的 payload 增量，不是整局闭环 payload。
4. 闭环是 48 步、4 次 `decide()`/`decode_assignment` 重规划。
5. EXEC 与 PREFIX/ABS 共用同一 SHA 的 \(U_{LL}\)；Phase-3 只改了 q，likelihood 与 ABS 同构。
6. EXEC 与 PREFIX 的 **pool 内** alignment 接近；DecodeNR 差在 OOD greedy 的 \(U_{\mathrm{decode}}\)。
7. EXEC in_pool 均值 3.75%；官方 decode 不是 `argmax_{pool} P`。
8. 同一 assignment 的 \(U_{LL}\) 随 state/t 大幅变化。
9. `U-\min U+\epsilon` **不改变名次**，但把 q 压扁；对照 ABS，压扁不是 PREFIX 差距的主因。
10. seed4 上 \(L_{\mathrm{exec}}\)↓ 且 val 排序↑，闭环仍低于 PREFIX；seed 间闭环方差由 seed0 崩塌主导。

## 尚未确认的问题

1. 4 个相继 assignment 的效用是否非加性、交互有多大（没有序列标签，不能量化）。
2. seed0 崩塌是 decoder 局部模式还是闭环状态分布上的倒霉轨迹（没有逐步闭环 assignment 日志）。
3. 若强制闭环从 24-pool 取 `argmax P`（而不是 greedy），EXEC payload 会不会接近 PREFIX——未做，也不在本阶段做。
4. q 压扁对 DecodeNR 的因果贡献（只有 ABS 对照，不是析因实验）。
5. 诊断 Spearman（phase1 stats）与正式 `pool_metrics` Spearman 数值不同；Kendall 与正式一致。正式比较以 Phase-3 报告为准。

## 下一阶段最值得验证的一个假设

**不要先写新 loss。**

假设（单一）：

> EXEC 无法稳定产生高闭环 payload，是因为监督的 \(U_{LL}\) 只标了 **oracle 状态上单个 hold 的冻结 assignment**，而正式闭环的 payload 来自 **策略自己走出的 4-hold greedy 轨迹**；在这套错位下，把 q 改成线性真实 utility 不会优于已经用同一 \(U_{LL}\) 的 ABS，也打不过训练了 greedy 局部路径的 PREFIX。

最小验证（仅提案，不执行）：在冻结 EXEC/PREFIX checkpoint 上，记录闭环每一次 `decide()` 的 assignment，并对其 **整条 4-hold 轨迹** 计算 payload；看该轨迹效用与 (i) 各 hold 的 \(U_{LL}\) 之和、(ii) pool 内 `argmax U`、(iii) greedy 的单 hold \(U_{\mathrm{decode}}\) 是否同序。若 (i)(ii)(iii) 都与整局 payload 不同序，则下一阶段的对象应是 **轨迹级 / 状态诱导的 executor utility**，而不是另一条 candidate-level q。

本阶段停止。

## 源文件 `HTW_Phase3G_TrajectoryAwareCredit_StructuralAudit_CN.md`

# HTW Phase-3G：Trajectory-Aware Credit Structural Approximation Audit

本阶段只做离线诊断：检验 `F(Γ)` 能否被分解为若干 **first-hold 可观测** 的 trajectory-aware credit components，而不是再学 `x → F`。未训练 Low-Level / High-Level，未改 Actor/Critic、reward、SafetyProjector、decoder、candidate pool、ranking、checkpoint、正式 evaluation、`L_exec`。未把 `F` / `U_traj` / 后续 hold / 未来 state-action-trajectory / future decoder 当作输入。未做 Transformer / MLP / XGBoost 大规模训练或超参搜索。

指定结构（预先写死，不是搜参赢家）：

`O_traj = leftover_queue · (1 + idle_frac) · leftover_access / (1 + competition)`

`C_transition = in_range_frac / (1 + competition)`

`leftover_access = 1 / (1 + d_left / service_radius)`，`d_left` 为当前未分配且 queue>0 的 buoy 到最近 UAV 的距离均值。

组内 z-score 后：`S = U_single + α · std(U) · z(O) + δ · std(U) · z(C)`。

指定对照：`STRUCT_O = (α=0.5, δ=0)`，`STRUCT_C = (α=0, δ=0.25)`，`STRUCT_OC = (0.5, 0.25)`。Phase-3E 对照：`FHAT = U_single + 0.25 F_hat`（OOF ExtraTrees-C，scenario GroupKFold）。α 网格 `{0.25, 0.5, 0.75, 1.0}` 只报敏感性，不取最优当结论。

协议与 Phase-3C/3D/3E/3F 相同：只改 first-hold 一次；hold 2/3/4 继续正式 greedy。设计集 16 scenario × 24 candidate。主判断以 **EXEC** 为准。

Sanity（必须对齐 Phase-3C）：NATURAL=5.812e6，**LOCAL=9.000e6**，**TRAJ=11.88e6**。本阶段复现通过。

**经验判断：指定 leftover / access / transition 公式不能构成稳定的 F 分解。STRUCT 相对 LOCAL 只有 2/16 state 的脆弱抬升，且不优于 Phase-3E 的 generic F_hat。证据不足以支持“应用结构化近似替代直接预测 F”。**

---

## 0. 与前序阶段的关系

| 阶段 | 已锁定事实 |
|---|---|
| 3C | first-hold `U_single` 与最终 `U_traj` mismatch：LOCAL 9.00e6 vs TRAJ 11.88e6 |
| 3D | oracle `S_β = U_single + β F` 有价值；β=0.50 时 EXEC `U_traj`=11.44e6，Spearman=0.946 |
| 3E | OOF：U_single only=0.193，state+cand=0.527，full=0.553；预测选择最佳 `U_traj`=9.375e6，`R_pred`≈0.135 |
| 3F | candidate geometry OOF=0.546≈full 0.553；within-state Spearman(`U_single`,F)≈0.259，(`F_hat`,F)≈0.577；top-1≈0.062；判断 information-limited |

Phase-3F 的建议是停止堆模型、改看 F 的结构。本阶段就是那一次结构假设检验，不是算法上线。

---

## 1. Observable proxy inventory

见 `outputs/observable_proxy_inventory.csv`。每一项都经过 `available_at_first_hold = true` 才允许进入 score。`F`、`U_traj`、future trajectory 标记为 forbidden，未使用。

| 组 | proxy | 含义 | 入模 |
|---|---|---|---|
| A | `U_single` | first-hold shadow increment | 是 |
| B | `mean/max_uav_target_dist`, `cand_mean_sinr` | candidate–state / UAV / 通信几何 | 是 |
| C | 同上 + assignment 放到 t=0 的距离 | candidate–UAV | 是 |
| D | `n_unique_buoys`, leftover buoy 集合 | 与其它任务 / buoy 的关系 | 是 |
| E | `n_in_range`, `leftover_min_dist`, `leftover_access` | 可达 / 距离 | 是 |
| F | `leftover_queue`, `leftover_hot`, `idle_frac`, `O_traj` | neighborhood / remaining-task opportunity | 是 |
| G | `competition_count` | assignment 竞争 | 是 |
| H | `assigned_queue_mean`, `queue_hot_mass` | t=0 已有 queue / progress | 是 |

`queue_hot_mass` 在 t=0 设计集上几乎跨 candidate 常数，只能作 state 背景，不能作 within-state 排序。`remaining_arrivals` 与后续 hold 一律不用。

---

## 2. Component 与 F（EXEC within-state）

只报 **state 内** 相关，不把跨 scenario 的 pooled 相关当成排序证据。

| component | Spearman vs F median | p25 | p75 |
|---|---:|---:|---:|
| U_single | 0.259 | 0.115 | 0.377 |
| -competition | 0.183 | 0.041 | 0.250 |
| leftover_access | 0.081 | -0.007 | 0.277 |
| leftover_queue | 0.018 | -0.165 | 0.177 |
| C_transition | 0.018 | -0.120 | 0.177 |
| O_traj | 0.002 | -0.109 | 0.112 |
| n_in_range | -0.010 | -0.106 | 0.068 |
| -mean_dist | -0.088 | -0.272 | 0.006 |

最相关的 **可观测量仍是 `U_single`（0.259）**。在非 `U_single` 的结构项里，只有 **-competition（0.183）** 有稳定弱正号；leftover-queue / leftover-access / `O_traj` / `C_transition` 都接近 0，区间跨过 0。

Pooled scatter（`component_vs_F.png`）与组内结论一致：`O_traj`、`C_transition`、`leftover_queue` 对 F 几乎是云团，Spearman≈0。

四方法对照（within-state median Spearman vs F）：

| family | U_single | O_traj | leftover_queue | C_transition | -competition | -distance |
|---|---:|---:|---:|---:|---:|---:|
| OLD | 0.343 | 0.080 | 0.071 | -0.095 | 0.093 | -0.293 |
| ABS | 0.097 | 0.155 | 0.140 | -0.053 | 0.010 | 0.078 |
| PREFIX | -0.064 | **-0.293** | **-0.306** | 0.313 | -0.027 | **0.319** |
| EXEC | 0.259 | 0.002 | 0.018 | 0.018 | 0.183 | -0.088 |

`O_traj` / leftover-queue 的符号 **跨方法翻转**。PREFIX 上“留下更多 queue”反而对应更低 F。因此 leftover-opportunity 不是一个可移植的 F 分量。

---

## 3. Within-state ranking（必须看，不能只看全局相关）

对每个 EXEC state 的 24 个 candidate，比较 score 与 F、`U_traj`。

### 3.1 vs F

| score | Spearman med [p25,p75] | Kendall med | top-1 mean | top-3 mean | NDCG@3 mean |
|---|---:|---:|---:|---:|---:|
| LOCAL (`U_single`) | 0.259 [0.115, 0.377] | 0.167 | 0.062 | 0.104 | 0.473 |
| O_traj | 0.002 [-0.109, 0.112] | -0.011 | **0.000** | 0.146 | 0.378 |
| C_transition | 0.018 [-0.120, 0.177] | -0.007 | 0.000 | 0.083 | 0.316 |
| STRUCT_OC | 0.187 [-0.003, 0.334] | 0.143 | 0.062 | 0.125 | 0.507 |
| FHAT | **0.353** [0.176, 0.524] | 0.236 | 0.062 | 0.125 | **0.528** |

`O_traj` 单独排序 **从未** 命中组内 F 第一名。STRUCT_OC 的组内 Spearman（0.187）**低于** LOCAL（0.259），也低于 FHAT（0.353）。NDCG@3 只有 FHAT 略高。top-1 全部停在 0.062，与 Phase-3F 一致。

### 3.2 vs `U_traj`

| score | Spearman med [p25,p75] | top-1 mean |
|---|---:|---:|
| LOCAL | 0.581 [0.420, 0.646] | 0.125 |
| O_traj | 0.111 [0.037, 0.238] | — |
| STRUCT_OC | 0.476 [0.402, 0.584] | 0.125 |
| FHAT | **0.627** [0.552, 0.724] | 0.125 |

对最终轨迹效用，FHAT 仍高于 LOCAL；STRUCT_OC 反而略低于 LOCAL。结构化加项没有把 first-hold 选择推得更接近 `argmax U_traj`。

---

## 4. First-hold 实际闭环（后续仍正式 greedy）

| method | U_traj | remaining F | F percentile | R vs LOCAL | DecodeNR | top-1 vs TRAJ | ranking-regret median |
|---|---:|---:|---:|---:|---:|---:|---:|
| LOCAL | 9.000e6 | 4.388e6 | 0.565 | 0.000 | 0.000 | 0.125 | 1.000 |
| FHAT | **9.375e6** | 4.800e6 | 0.617 | 0.135 | 0.010 | 0.125 | 1.000 |
| STRUCT_O | 9.188e6 | 4.588e6 | 0.602 | 0.058 | 0.003 | 0.125 | 1.000 |
| STRUCT_C | **8.500e6** | 3.913e6 | 0.529 | 0.077* | 0.006 | 0.062 | 1.000 |
| STRUCT_OC | 9.250e6 | 4.675e6 | 0.615 | 0.135 | 0.009 | 0.125 | 1.000 |
| TRAJ-ORACLE | 11.88e6 | 9.625e6 | 0.987 | 1.000 | 0.516 | 1.000 | 0.000 |

`payload` 与 `U_traj` 相同（轨迹终点 payload 增量）。`R = (U_method - U_local) / (U_traj_oracle - U_local)`，分母为 0 的 scenario 记 NaN，不进均值。

\*STRUCT_C 的 **mean R=0.077 与 mean `U_traj`=8.50e6 方向不一致**：这是「比率均值 ≠ 均值比率」。有的 scenario 分母很小、轻微正恢复会抬高 mean R。闭环判断以 **mean `U_traj`** 为准：STRUCT_C **低于** LOCAL，单独加 transition proxy 有害。

α 敏感性（只加 `O_traj`，不取最优）：

| alpha | U_traj | R | F percentile |
|---:|---:|---:|---:|
| 0.25 | 9.188e6 | 0.058 | 0.602 |
| 0.50 | 9.188e6 | 0.058 | 0.602 |
| 0.75 | 9.250e6 | 0.135 | 0.615 |
| 1.00 | 9.250e6 | 0.135 | 0.615 |

更大 α 只是把那 2 个改选 scenario 再确认一遍，不是新的最优算法。

---

## 5. Counterfactual consistency：STRUCT 选中的 candidate 是否更接近 TRAJ-ORACLE

EXEC 16 个 state，STRUCT_OC 与 LOCAL **14/16 选同一个 candidate**。仅两处改选：

| scenario | LOCAL `U_traj` / F | STRUCT_OC | FHAT |
|---|---|---|---|
| 221003 | 6.00e6 / 1.80e6 | **9.00e6 / 5.00e6** | 9.00e6 / 5.00e6 |
| 221006 | 8.00e6 / 3.60e6 | **9.00e6 / 5.00e6** | 8.00e6 / 3.60e6（未改） |
| 221000 | 7.00e6 / 3.60e6（未改） | 同 LOCAL | **10.00e6 / 7.00e6** |

STRUCT_OC 的整段 +0.250e6 均值，**全部来自这 2 个 state**。FHAT 的 +0.375e6 来自 221000 与 221003。

选中 candidate 的真实 F percentile：LOCAL 0.565 → STRUCT_OC 0.615 → FHAT 0.617。有抬升，但仍远低于 TRAJ 的 0.987。top-1 vs `argmax U_traj` **三者都是 0.125**，STRUCT **并没有更经常贴近轨迹最优**。

ranking-regret 中位仍是 1.0：对绝大多数 state，STRUCT 选中的 F 等于 LOCAL 选中的 F。相关略变、选点几乎不变——这正是本阶段禁止「相关升高就宣称算法有效」的原因。

---

## 6. Component ablation

| score | EXEC `U_traj` | F pct | R | top-1 vs TRAJ | within Spearman vs F |
|---|---:|---:|---:|---:|---:|
| U_single | 9.000e6 | 0.565 | 0.000 | 0.125 | 0.259 |
| U_single + O_traj | 9.188e6 | 0.602 | 0.058 | 0.125 | — |
| U_single + C_transition | 8.500e6 | 0.529 | 0.077* | 0.062 | — |
| U_single + O_traj + C_transition | 9.250e6 | 0.615 | 0.135 | 0.125 | 0.187 |
| U_single + 0.25 F_hat | 9.375e6 | 0.617 | 0.135 | 0.125 | 0.353 |
| TRAJ-ORACLE | 11.88e6 | 0.987 | 1.000 | 1.000 | — |

O_traj 单独有很小正贡献；C_transition 单独损害 mean `U_traj`；二者相加的闭环 R 追上 FHAT，但绝对 `U_traj` 仍低于 FHAT，组内排序还更差。没有一项 ablation 把 top-1 从 0.125 抬上去。

---

## 7. 四方法指定 STRUCT_OC

| method | LOCAL | FHAT | STRUCT_O | STRUCT_OC | TRAJ | R_OC | R_FHAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLD | 1.169e7 | **1.219e7** | 1.206e7 | 1.194e7 | 1.381e7 | 0.145 | 0.182 |
| ABS | 1.112e7 | **1.188e7** | 1.156e7 | 1.181e7 | 1.312e7 | 0.222 | 0.333 |
| PREFIX | 1.031e7 | 1.075e7 | 1.050e7 | **1.088e7** | 1.438e7 | 0.132 | 0.069 |
| EXEC | 9.000e6 | **9.375e6** | 9.188e6 | 9.250e6 | 1.188e7 | 0.135 | 0.135 |

只有 PREFIX 上 STRUCT_OC 的闭环略高于 FHAT；OLD / ABS / EXEC 都是 FHAT 更好或持平。STRUCT 不是跨方法更稳的选择器。

---

## 8. 为什么 candidate geometry 有 signal，但直接预测 F 的 top-1 很差

这是本阶段对 Phase-3F 现象的结构解释，不是新模型。

1. **Phase-3F 的 geometry signal 是多特征联合、且含跨 state 变异。** ExtraTrees 在 17 维 candidate geometry 上 OOF Spearman=0.546，within-state Spearman(`F_hat`,F)≈0.577。那不是 leftover-queue 这一维。
2. **把 geometry 压成一条物理公式后，组内排序消失。** `O_traj` within-state Spearman≈0.002，top-1=0。leftover-queue、access、in-range 单独都不能复现 ExtraTrees 抓住的交互。
3. **F 含 later-greedy。** t=0 leftover 只描述“现在没吃掉的 queue 质量/距离”，不描述 hold 2/3/4 的正式 greedy 会如何走。PREFIX 上 leftover 甚至与 F 负相关：留下更多任务并不等于留下更好的后续机会。
4. **组内 F 的第一名对局部扰动极敏感。** 即使 ExtraTrees 相关 0.55、FHAT 组内 0.35，top-1 仍是 0.062。STRUCT 相关更低，更不可能稳定打中 `argmax F` / `argmax U_traj`。
5. **因此：geometry 有粗粒度 signal ≠ 存在可加的、可解释的 first-hold credit 分解。** 当前失败的是「leftover × access / competition」这条假设，不是 Phase-3D「oracle F 有用」被推翻。

---

## 9. 五个问题

**Q1. first-hold 可观测信息中，哪些结构 component 与 F 最相关？**  
EXEC 上仍是 **`U_single`（median Spearman=0.259）**。非 local-value 项里只有 **-competition（0.183）** 有弱正号；`leftover_access`=0.081；`O_traj` / leftover-queue / `C_transition` ≈ 0。PREFIX 上最相关的反而是 **-distance（0.319）** 与 `C_transition`（0.313），leftover-queue 为 **-0.306**。没有一个跨方法稳定的“未来机会”分量。

**Q2. `O_traj` 是否比 `U_single` 更能解释 future trajectory value？**  
**否。** EXEC：`O_traj` vs F = 0.002，`U_single` vs F = 0.259；`O_traj` vs `U_traj` = 0.111，`U_single` vs `U_traj` = 0.581。`O_traj` 组内 top-1 vs F = 0。

**Q3. 结构化 score 是否比 Phase-3E generic predictor 更稳定？**  
**否。** 组内 Spearman：STRUCT_OC 0.187 < FHAT 0.353 <（Phase-3F 的 `F_hat` 0.577）。闭环：STRUCT_OC 9.250e6 < FHAT 9.375e6；R 同为 0.135。跨方法 4 家里 3 家 FHAT 不差于 STRUCT。STRUCT 没有表现出更好的稳定性。

**Q4. 结构化 score 能否提高实际 first-hold trajectory recovery？**  
**相对 LOCAL：能抬一点，但是脆弱的。** STRUCT_OC=9.250e6 vs LOCAL=9.000e6，R=0.135；选中 F percentile 0.565→0.615。14/16 state 选点不变，增益只来自 221003 与 221006。top-1 vs TRAJ 仍是 0.125。STRUCT_C 把 mean `U_traj` 降到 8.500e6。相对 FHAT：**没有提高**。不能把相关或某一个 α 写成有效算法。

**Q5. 当前证据是否足以支持“trajectory-aware credit 应采用可观测结构化近似，而不是直接预测完整 F”？**  
**不足，不能这么写。**  
支持结构路线的部分：leftover/competition 等字段确实可观测；竞争项有弱信号；STRUCT_OC 相对 LOCAL 有 2-state 闭环正贡献。  
不支持“用结构化近似替代预测 F”的部分：指定 `O_traj` 几乎不解释 F；STRUCT 组内排序与闭环都不优于 Phase-3E `F_hat`；公式符号跨方法翻转；top-1 / ranking-regret 几乎不动。  
更准确的阶段结论是：

> first-hold 可写成 leftover / access / transition 的简单加性 credit，**当前证据不支持**；Phase-3F 的 geometry signal 也 **不能** 被这条公式吸收。oracle F 仍然有价值（Phase-3D），但可观测结构近似尚未成立。剩余 gap 仍主要是 later-greedy / 未观测未来，而不是“缺一个更漂亮的 α”。

本阶段不把任一 α、δ 写成最终参数。相关升高 ≠ 算法有效。

---

## 10. 阶段纪律与产物

未训练、未改正式方法、未进入 Phase-4。

产物目录：`src/wt/lowlevel_v2/phase3g_trajectory_credit_structural_audit/outputs/`

- `observable_proxy_inventory.csv`
- `component_metrics.csv`
- `within_state_metrics.csv`
- `selection_metrics.csv`
- `ablation_metrics.csv`
- `ranking_metrics.csv`
- `recovery_metrics.csv`
- `figures/component_vs_F.png`
- `figures/within_state_ranking.png`
- `figures/first_hold_selection.png`
- `figures/recovery_comparison.png`
- `figures/ranking_regret.png`

本阶段停止。

## 源文件 `HTW_Phase3H_CandidateConditionedDecisionCredit_Audit_CN.md`

# HTW Phase-3H：Candidate-Conditioned Decision Credit Audit

本阶段只做离线诊断：Phase-3F 的 candidate-geometry signal（OOF≈0.546）究竟来自连续距离，还是 first-hold 已存在的 **离散决策 / feasible-set 结构**。未训练 Low-Level / High-Level，未改 Actor/Critic、reward、decoder、pool、ranking、checkpoint、正式 greedy。未把未来 hold / state / action / trajectory / `F` / `U_traj` 当作输入。不重复 Phase-3G 的 leftover×access 连续公式作为新发现。

指定离散 proxy（先看原始计数，再组内 z-score；不发明复杂公式）：

- `degree` / `N_feasible(Γ)`：选择 Γ 后，t=0 未占用 buoy 上仍 `in_range` 的 (UAV, buoy) 边数。
- `weighted_degree`：同上，边权为该 leftover buoy 的当前 queue。
- `Flex`：exclusive occupation 下，每个 UAV 的合法单边改写数（改 idle，或改到未占用且 in-range 的 buoy）。

`S_disc = z(U_single) + α z(Flex)`，指定 α=0.50；0.25 / 0.75 只报敏感性。STRUCT_OC 与 FHAT 沿用 Phase-3G / 3E。只改 first-hold 一次；hold 2/3/4 的 `U_traj` 来自已锁定的 Phase-3C 表，不重新 rollout。主判断以 **EXEC** 为准。

Sanity：LOCAL=9.000e6，TRAJ=11.88e6。复现通过。

**经验判断：当前可观测的 leftover feasible-set / Flex 不能解释 candidate-geometry 的 future-value signal。最大化 degree 或 Flex 会把闭环打到约 4.5e6。证据支持 C：连续几何公式与离散 feasible-set 都不足，first-hold observable information 本身不够。**

---

## 0. 已锁定前序事实

| 阶段 | 锁定数字 |
|---|---|
| 3C | LOCAL=9.000e6，TRAJ=11.88e6 |
| 3E | U_single OOF=0.193，state+cand=0.527，full=0.553；预测恢复≈0.135 |
| 3F | candidate geometry OOF≈0.546≈full 0.553；U_single OOF≈0.295 |
| 3G | within-state：U=0.259，O_traj=0.002，STRUCT_OC=0.187，FHAT=0.353；闭环 STRUCT_OC=9.250e6，FHAT=9.375e6 |

Phase-3G 已否定 leftover_queue×access/competition 的加性结构。本阶段换成 **离散边计数 / 合法改写数**，不再重做那条连续公式。

---

## 1. 原始分布（EXEC，未归一化）

| proxy | min | max | mean | std | n_unique |
|---|---:|---:|---:|---:|---:|
| degree | 0 | 36 | 21.66 | 8.35 | 7 |
| weighted_degree | 0 | 90 | 39.89 | 24.88 | 13 |
| Flex | 6 | 39 | 25.10 | 7.26 | 22 |
| leftover_coverable | 0 | 6 | 3.61 | 1.39 | 7 |
| conflict | 0 | 0 | 0 | 0 | 1 |
| pool_degree | 0 | 11 | 1.38 | 1.99 | 8 |

degree / Flex 不是常数：每个 state 内 degree 有 6–7 个不同值，Flex 有 11–14 个。先看分布再归一化是成立的。

`conflict` 全为 0：正式 24-pool 已经 exclusive legalize，组内没有 assignment conflict 可用来排序。不能把“更少 conflict”写成高 F 的可观测标记。

`degree` 与 `Flex` 几乎共线（pooled Spearman=0.949），二者主要是 `n_occupied` 的离散反面（与占用数 Spearman 约 −0.72 / −0.58）。这是 leftover 任务集合的边计数，不是一条新的物理量。

额外记录 `pool_degree`：当前 24-pool 里 Hamming 距离=1 的邻居数。用户指定的三个最小 proxy 仍是 degree / weighted_degree / Flex；`S_disc` 只用 Flex。

---

## 2. Within-state ranking vs F（EXEC）

| score | Spearman med [p25,p75] | Kendall med | top-1 | top-3 | NDCG@3 |
|---|---:|---:|---:|---:|---:|
| U_single | 0.259 [0.115, 0.377] | 0.167 | 0.062 | 0.104 | 0.473 |
| degree | 0.036 [−0.085, 0.132] | 0.007 | 0.000 | 0.104 | 0.358 |
| weighted_degree | 0.018 [−0.165, 0.177] | −0.002 | 0.000 | 0.083 | 0.346 |
| Flex | 0.041 [−0.079, 0.084] | 0.005 | 0.000 | 0.083 | 0.363 |
| STRUCT_OC | 0.187 [−0.003, 0.334] | 0.143 | 0.062 | 0.125 | 0.507 |
| S_disc | 0.187 [0.022, 0.326] | 0.129 | 0.062 | 0.104 | 0.482 |
| FHAT | **0.353** [0.176, 0.524] | 0.236 | 0.062 | 0.125 | 0.528 |
| pool_degree（额外） | 0.175 [−0.012, 0.350] | 0.145 | 0.188 | 0.250 | 0.590 |

degree / Flex 的组内相关接近 0，区间跨过 0，top-1=0。它们比 Phase-3G 的 `O_traj`（0.002）只高一点，远低于 Phase-3F geometry OOF 0.546，也低于 `U_single` 0.259。

`pool_degree` 是本阶段最强的纯离散项（0.175，NDCG@3=0.590），仍解释不了 0.546，也不能写成未来价值的原因。

vs `U_traj`：LOCAL 0.581，S_disc 0.497，Flex 0.098，degree 0.121。离散项单独几乎不排序轨迹效用。

四方法对照（within-state median Spearman vs F）：

| family | U_single | degree | Flex | S_disc | FHAT |
|---|---:|---:|---:|---:|---:|
| OLD | 0.343 | 0.120 | 0.147 | 0.332 | 0.483 |
| ABS | 0.097 | 0.162 | 0.114 | 0.117 | 0.197 |
| PREFIX | −0.064 | **−0.257** | **−0.307** | −0.130 | 0.096 |
| EXEC | 0.259 | 0.036 | 0.041 | 0.187 | 0.353 |

PREFIX 上 leftover degree / Flex 与 F **负相关**。离散 feasible-set 不是可移植的正 credit。

---

## 3. First-hold 闭环（后续仍正式 greedy）

| method | U_traj | F percentile | R | top-1 vs TRAJ |
|---|---:|---:|---:|---:|
| LOCAL | 9.000e6 | 0.565 | 0.000 | 0.125 |
| STRUCT_OC | 9.250e6 | 0.615 | 0.135 | 0.125 |
| DEGREE | **4.500e6** | 0.487 | −1.995 | 0.000 |
| WDEGREE | **4.500e6** | 0.487 | −1.995 | 0.000 |
| FLEX | **4.688e6** | 0.529 | −1.687 | 0.000 |
| S_disc | 9.250e6 | 0.607 | 0.077 | 0.125 |
| FHAT | **9.375e6** | 0.617 | 0.135 | 0.125 |
| TRAJ | 11.88e6 | 0.987 | 1.000 | 1.000 |

单独 `argmax degree / Flex` 把 EXEC 闭环打到约 4.5e6，低于 NATURAL 量级的另一侧崩溃。最大化“留下更多后续可行边”在 first-hold 上是有害选择，不是弱选择。

`S_disc`（α=0.25/0.50/0.75）三个指定值的 `U_traj` 都是 9.250e6，R=0.077。不是搜参赢家。

相对 LOCAL 的 +0.250e6 **只来自 1/16 个 state**（221003：6.00e6 → 10.00e6）。其余 15 个选点与 LOCAL 相同。top-1 vs TRAJ 仍是 0.125。

---

## 4. Replacement / structure consistency（不是因果）

对每个 state，把 first-hold 的 LOCAL candidate 在纸面上换成另一个合法 candidate，只比较 t=0 可观测结构，不 rollout。

EXEC：

- top-F 比 LOCAL 有更高 Flex 的比例 = **0**
- 更高 degree 的比例 = **0**
- 更低 conflict 的比例 = **0**（conflict 全 0）
- mean ΔFlex(topF−LOCAL) = **−6.19**
- mean Δdegree = **−7.13**

高 F candidate 在 first-hold 上已经表现出 **更小** 的 leftover feasible-set，不是更大。这与“留下更多后续选择 → 更高 F”的假设方向相反。

高/低 F 四分位的均值几乎重叠（degree 22.4 vs 21.2，Flex 25.8 vs 24.6）。组内排序没有被 leftover 边数拉开。

只能写：高 F 与更小 leftover degree **一致地同时出现**；不能写 Flex 是 F 的原因。

---

## 5. 四方法指定 S_disc

| method | LOCAL | STRUCT_OC | S_disc | FHAT | TRAJ | R_disc |
|---|---:|---:|---:|---:|---:|---:|
| OLD | 1.169e7 | 1.194e7 | 1.194e7 | **1.219e7** | 1.381e7 | 0.091 |
| ABS | 1.112e7 | 1.181e7 | 1.156e7 | **1.188e7** | 1.312e7 | 0.111 |
| PREFIX | 1.031e7 | **1.088e7** | 1.069e7 | 1.075e7 | 1.438e7 | 0.042 |
| EXEC | 9.000e6 | 9.250e6 | 9.250e6 | **9.375e6** | 1.188e7 | 0.077 |

四家都是 FHAT 不差于 S_disc。S_disc 没有更稳的 first-hold 选择。

---

## 6. 对 Phase-3F geometry signal 的含义

Phase-3F：17 维 candidate geometry 联合 OOF≈0.546。Phase-3G：连续 leftover×access 压不成那条信号。Phase-3H：把同一几何收成 **离散边 / 合法改写 / leftover 可覆盖任务**，组内 Spearman 仍约 0.04。

因此：

1. geometry signal 不是“选 Γ 后还剩多少 in-range 后续边”。
2. 也不是 assignment flexibility（合法单边改写数）。
3. 正式 pool 内没有 conflict 可分。
4. pool 内 Hamming-1 邻居数只有弱相关（0.175），仍远低于 0.546。
5. ExtraTrees 用的是多维联合（SINR / 干扰 / 距离 / assignment token），不是一张 leftover 匹配图。
6. F 含 later-greedy；t=0 leftover 匹配图描述不了 hold 2/3/4。

candidate geometry 有 signal，但 **不能进一步解释为当前可观测 feasible-set structure**。

---

## 7. 六个问题

**Q1. candidate geometry 的 signal 能否由当前 feasible-set structure 解释？**  
**不能。** EXEC 最强指定离散项 Flex=0.041，degree=0.036。对照 `O_traj`=0.002、geometry OOF=0.546。额外的 `pool_degree`=0.175 仍不够。PREFIX 上 degree/Flex 甚至为负。

**Q2. degree / weighted-degree / Flex 与 F 的 within-state ranking？**  
Spearman median：degree **0.036**，weighted_degree **0.018**，Flex **0.041**。Kendall 接近 0。top-1 全是 0。NDCG@3 约 0.35–0.36，低于 LOCAL 的 0.473。

**Q3. S_disc 是否比 U_single 有稳定改善？**  
**否。** 组内 Spearman 0.187 < U_single 0.259。闭环 9.250e6 > 9.000e6，但只改 1 个 state，且排序变差。不是稳定改善。

**Q4. S_disc 能否提高实际 first-hold trajectory recovery？**  
**相对 LOCAL：能抬一点，但是脆弱的。** 9.250e6 vs 9.000e6，R=0.077，F percentile 0.565→0.607。15/16 选点不变。单独用 degree/Flex 会把 `U_traj` 打到 4.5e6。不能写成有效算法。

**Q5. S_disc 是否超过 Phase-3E FHAT？**  
**否。** 9.250e6 < 9.375e6；within Spearman 0.187 < 0.353；R 0.077 < 0.135。四方法里也没有一家 S_disc 稳定压过 FHAT。

**Q6. 当前证据支持哪一种解释？**  
**C. 两者都不足，observable information 本身不足。**

- 不是 A：Phase-3G 已否定连续 leftover/access；本阶段不再把它救回来。
- 不是 B：离散 leftover feasible-set / Flex 组内相关≈0，高 F 的 leftover degree 更小，最大化它有害。
- 是 C：first-hold 能写成的连续公式和离散匹配图，都吸收不了 Phase-3F 的 0.546 OOF；预测恢复仍停在约 0.13。剩余 gap 仍是 later-greedy / 未观测未来，不是“缺一张更漂亮的可行集计数”。

Flex 即使有弱相关，也只是与 F 一致的可观测结构相关，**不是** future value 的原因。本阶段不把任一 α 写成最终参数。

---

## 8. 阶段纪律与产物

未训练。未改正式方法。未进入 End-to-End Independent Validation。未进入 Phase-4。

产物：`src/wt/lowlevel_v2/phase3h_candidate_decision_credit_audit/outputs/`

- `candidate_structure.csv`
- `within_state_metrics.csv`
- `replacement_audit.csv`
- `selection_metrics.csv`
- `recovery_metrics.csv`
- `method_comparison.csv`
- `figures/candidate_f_vs_degree.png`
- `figures/candidate_f_vs_flex.png`
- `figures/within_state_ranking.png`
- `figures/first_hold_selection.png`
- `figures/recovery_comparison.png`

本阶段停止。

## 源文件 `HTW_Phase4D6_DeploymentCriterionRegistration_CN.md`

# HTW Phase-4D-6 Deployment Criterion Registration

本阶段只把项目负责人选定的 Candidate-C 写入正式 deployment criterion registry。
未修改 B2，未修改 τ，未修改 Gate-1，未修改 planner / decide()，未重新训练。
未根据已有 64 个 independent scenario 调整 criterion 数值，未重算这 64 个 scenario，未寻找新阈值。
未宣布 planner_ready=True，未进入 Phase-4E。

**deployment_criterion = CONSERVATIVE_WITH_FALLBACK_V1**

**planner_ready = False**

**status = CRITERION_REGISTERED_PENDING_INDEPENDENT_HOLDOUT**

Registry 字段 `status = REGISTERED`。系统阶段状态仍是待独立 hold-out，不是 deployment PASS。

## 1–4. 来源链

1. Phase-4D3 检索仓库后确认：NO PRE-REGISTERED DEPLOYMENT CRITERION FOUND。
2. Phase-4D4 只整理 C1–C8 人工决策维度，正式 acceptance value 全部为 null。
3. Phase-4D5 按工程原则给出 Candidate-A / B / C，且明确不选择 official criterion。
4. 本阶段由项目负责人明确选择 **Candidate-C**（Conservative-with-Fallback）。

## 5. 数值不是由已有 64 个 scenario 反推

正式字段来自 Phase-4D5 的设计常数，并经本阶段人工确认：

- C1 max_degradation_rate = 0.02
- C2 max_worst_case_delta_R = -0.125
- C3 min_mean_delta_R = 0.02
- C4 min_mean_U_gain = 0.0
- C5 scenario_degradation_policy = allow_some
- C6 require_all_scenarios_noninferior = False
- C7 independent_holdout_required = True
- C8 required_independent_batches = 2

这些不是“因为当前 3/64 所以取 5%”，也不是“因为 worst-case −0.214 所以取 −0.25”。本阶段没有用已有 64 个 scenario 重新拟合 threshold。

## 6. 已有 64 个 scenario 的角色

220000–220063 只作为此前 Phase-4C / Phase-4D independent validation evidence。
它们不构成本 criterion 的来源，本阶段也不用它们宣布 deployment PASS。

## 7–8. 新的 independent batch 尚未完成

Candidate-C 要求 **2** 个新的、独立生成的 independent batch。
在这两批完成并按本 criterion 审计之前，不得宣称 deployment PASS。

## 9. planner_ready

本阶段 **planner_ready = False**。
approval_scope 仅为 “deployment criterion only”，不包含 planner 接入。

## 正式 registry

路径：`src/wt/lowlevel_v2/phase4d_deployment_criterion_registry/deployment_criterion_registry.json`

- criterion_id = CONSERVATIVE_WITH_FALLBACK_V1
- criterion_name = Conservative-with-Fallback
- status = REGISTERED
- source = HUMAN_SPECIFICATION
- human_approved = true
- approval_scope = deployment criterion only

## Gate-1 保持冻结

Deployment criterion 只判断整套冻结 B2 + 冻结 Gate-1 是否满足进入正式 planner 的部署条件。
它不是新的 scenario-level gate。

- margin = U_hat(argmax U_hat) − U_hat(argmax U_single)
- tau = 1.535e6
- 若 margin ≥ tau 则使用 B2，否则使用 LOCAL

锁定 B2：C_hat = 已冻结 ExtraTrees(U_single + Phase-3E candidate-conditioned geometry)，U_hat = U_single + C_hat。

## 最终状态

- deployment_criterion = CONSERVATIVE_WITH_FALLBACK_V1
- planner_ready = False
- status = CRITERION_REGISTERED_PENDING_INDEPENDENT_HOLDOUT

## 下一阶段唯一任务

下一阶段不是重新训练 B2。

下一阶段是：

Independent Holdout Batch-1 / Batch-2
→ 使用已经冻结的 B2
→ 使用已经冻结的 Gate-1
→ 不重新调参
→ 检验 Candidate-C 的 acceptance criterion。

本阶段停止。不进入 Phase-4E。

## 源文件 `HTW_Phase4D7_IndependentHoldoutAudit_CN.md`

# HTW Phase-4D-7 Independent Holdout Audit

本阶段先检查是否存在两个真正新的 independent batch。没有找到，因此停止。
未伪造数据，未把 220000–220063 拆成两个 batch，未给旧 scenario 换 seed，未 rollout，未训练 B2，未改 τ / Gate-1 / planner / decide() / criterion，未进入 Phase-4E。

**status = BLOCKED_ON_INDEPENDENT_HOLDOUT_DATA**

**planner_ready = False**

**deployment verification = NOT_RUN**

## 检索结果

- `LLV2_DATA_REGISTRY.json`：`LLV2_confirmation = null`，`LLV2_final_untouched = null`。
- `data_registry.py` 写明 confirmation 与 final untouched pool 在 LL-V2.0 中没有创建。
- Phase-4D6 registry 只锁定 Candidate-C，没有给出 Batch-1 / Batch-2 的 scenario ID。

因此 **Batch-1 不存在，Batch-2 不存在**。`independent_holdout_manifest.json` 的 `scenarios` 为空，没有可记录的新 scenario。

## 已有池为什么不能充当 holdout

| 池 | ID | 判定 |
|---|---|---|
| LLV2_train | 220000–220063 | 已用于 Phase-4C/4D。禁止拆成 Batch-1/2。 |
| LLV2_design | 221000–221015 | 已进入 B2 fit、GroupKFold、特征与超参数选择。 |
| Phase-2 val | 222000–222063 | 历史验证池，`refuse_compat` 拒绝，不是登记过的 holdout。 |
| Phase-3 val | 223000–223063 | 同上。 |
| Phase-10.6 untouched | 106000–106063 | 保留种子，Phase-4C 明确拒绝。从未登记为 Deployment Batch-1/2，本阶段不打开。 |

## Candidate-C

C1–C8 **未评估**。没有新 batch，不能报告 degradation rate、worst-case ΔR、mean ΔR 或 mean U gain，也不能给 C1–C8 打 PASS/FAIL。

已有 64 个 scenario 上的 mean R=0.892 相对 LOCAL=0.763，以及 permutation sanity，都不能提前 PASS。

## 最终状态

- status = BLOCKED_ON_INDEPENDENT_HOLDOUT_DATA
- planner_ready = False
- criterion 仍是 CONSERVATIVE_WITH_FALLBACK_V1，未改
- Gate-1 与 τ=1.535e6 未改

下一阶段仍然只能是：先登记两个真正新的、未参与 B2 fit / 特征选择 / 阈值选择 / criterion 选择 / 既往验证的 independent batch，再使用冻结 B2 与冻结 Gate-1 做审计。

本阶段停止。不进入 Phase-4E。

## 源文件 `HTW_Phase4D7A_HoldoutSourceQualificationAudit_CN.md`

# HTW Phase-4D-7A Holdout Source Qualification Audit

本阶段只审计数据源资格。未训练、未 rollout、未注册 holdout，未修改 B2、τ、Gate-1、Candidate-C、planner 或 `decide()`。没有 deployment PASS/FAIL。

**holdout_source_status = NO_ELIGIBLE_SOURCE**

**status = BLOCKED_ON_INDEPENDENT_HOLDOUT_DATA**

**planner_ready = False**

## 1. 已有 64 个 unseen scenario

220000–220063 是 Phase-4C/4D 的 independent validation evidence。
它们不是 Candidate-C 的 deployment holdout，也不能拆成 Batch-1 / Batch-2。

## 2. 106000–106063：candidate pool，不是 official holdout

单独审计：

1. 它相对历史训练场景曾被保留为 untouched，但 Phase-10.6 已经一次性消耗：`final_untouched_consumed=True`。对新的 deployment 声明，它不再是未使用池。
2. 未进入 B2 ExtraTrees 训练。Phase-10.6 本身也没有训练。
3. 未进入 B2 feature selection。
4. 未进入 B2 hyperparameter selection。
5. 未进入 τ / Gate-1 selection。
6. 未进入 Candidate-C criterion selection。
7. 没有 B2 协议的 `U_single`。
8. 没有 B2 协议的逐候选 `U_traj`。Phase-10.6 只有 FINAL-HIER 闭环 payload，不是 first-hold candidate label。
9. 冻结 B2 代码本身不必为了打分而修改，但 `refuse_compat` 禁止打开这批种子，且本阶段禁止 rollout。当前不能重新执行。
10. Phase-4C 拒绝它，是因为它属于已消耗的 Phase-10.6 final test，不是登记过的 B2 deployment holdout。
11. 当前工程协议不允许把它正式登记为 deployment holdout。Phase-10.6 规定 final test 之后不得据此改方法。本阶段不注册。

“untouched” 不等于 deployment holdout。这里只记为 **candidate untouched pool**。

## 3. 其他检索到的池

都不是 deployment batch：

- 221000–221015：已进入 B2 fit。
- 222000–222063、223000–223063：历史验证池，协议拒绝。
- 104000–104063、105000–105063：Phase-10.4/10.5 历史 independent，协议拒绝，没有 B2 标签。
- `LLV2_confirmation`、`LLV2_final_untouched`：registry 为 null，池不存在。

没有 Candidate Pool 同时满足：未进 B2 fit / 特征 / 超参数 / τ / Gate / criterion，且已有 B2 可观测量和真实 `U_traj`，且不是旧集拆分或换 seed。

## 资格结论

`eligible_for_holdout = TRUE` 的来源数量 = 0。

因此不能建立 Batch-1 / Batch-2，也不能进入注册阶段。

## 最终状态

- holdout_source_status = NO_ELIGIBLE_SOURCE
- status = BLOCKED_ON_INDEPENDENT_HOLDOUT_DATA
- planner_ready = False
- ready for registration 不成立，更不是 deployment PASS

本阶段停止。不进入 Phase-4E。

## 源文件 `HTW_Phase4D7B_IndependentHoldoutGeneration_CN.md`

# HTW Phase-4D-7B Independent Holdout Generation

本阶段先审计现有 scenario 生成机制和 batch size 规范，未生成任何新 scenario。
未训练或修改 B2，未修改 τ、Gate-1、Candidate-C、planner 或 `decide()`。
未做 B2 vs LOCAL、ΔR 或 deployment PASS/FAIL。未进入 Phase-4E。

**BATCH_SIZE_SPECIFICATION_REQUIRED**

**status = BLOCKED_ON_INDEPENDENT_HOLDOUT_DATA**

**planner_ready = False**

## 1. 使用的 scenario generator

合法入口已经存在，本阶段没有新写 generator。

- Scenario realization：`L3BEnv.reset(seed=scenario_id)`。到达过程使用 L3-B 默认配置。
- First-hold candidate pool：`src/wt/htw_phase945/candidate_pool.py` 的 `generate_pool(env, seed_key=scenario_id * 1009 + 0)`。
- Pool 大小 L=24，来自既有 `POOL_SIZE`，不是本阶段新设的。
- ID 分配器：`src/wt/lowlevel_v2/data_registry.py` 的 `allocate_new_pool`。它只为 `LLV2_train` 和 `LLV2_design` 分配 ID。
- `LLV2_confirmation` 与 `LLV2_final_untouched` 在 `build_registry()` 中固定为 `None`。注释写明这两个池在 LL-V2.0 没有创建。

## 2. Generator 配置

- 环境：L3-B，M=6，K=12，T=48，hold=15。
- scenario intrinsic state 由 `scenario_id` 作为 `reset` 种子决定。换一个未使用的整数种子就是一次新的 realization。仅改已有 scenario 的 seed 不能把旧 ID 说成新 scenario。
- `scenario_id` 不是模型输出。现有分配规则是从指定 base 起取第一个未被历史种子挡住的整数。
- `U_single`：既有协议是对 pool 中每个候选做 5-seed shadow_h15 的 payload 增量均值。需要 rollout，本阶段未执行。
- `U_traj`：既有协议是 first-hold 覆盖该候选，后续 hold 走正式 greedy。需要 rollout，本阶段未执行。
- Phase-3E candidate-conditioned geometry：由当前 first-hold 状态和候选分配计算，不由 B2 参数决定。
- 不受 B2 / feature selection / criterion 影响的字段：环境 reset 状态、候选 pool、`U_single`、`U_traj`、geometry。
- 会受 B2 影响的字段：`C_hat`、`U_hat`、margin、Gate-1 选择。这些不属于生成阶段，本阶段没有计算。

## 3–5. Batch ID、seed、独立性

Batch-1 scenario IDs：无。未生成。
Batch-2 scenario IDs：无。未生成。
seed / realization policy：未执行。

因此不能证明 Batch-1 与 Batch-2 不同，也不能把空集登记进 `LLV2_confirmation` 或 `LLV2_final_untouched`。

## 6. 与历史池的交叉

未分配新 ID，所以没有新的交叉。已占用且禁止复用的范围仍是：

- 220000–220063 `LLV2_train`
- 221000–221015 `LLV2_design`
- 222000–222063、223000–223063
- 104000–104063、105000–105063、106000–106063

Phase-3 分配器还把 `220000–221199` 放进 blocked universe。新 ID 若将来生成，必须落在这些区间之外。

## 7–9. 标签完整性

`U_single`、`U_traj`、candidate geometry 均未生成。完整性检查没有对象。
生成这些标签在机制上是可能的，因为 Phase-4C 用的就是上述冻结入口。但在 batch size 未规定之前不能开始 rollout。

## 10. Leakage

没有新 scenario 进入 B2 fit、feature selection、hyperparameter selection、τ selection、Gate-1 selection 或 criterion selection。
也没有根据 `U_traj`、margin 或 Candidate-C 挑选 scenario。

## 11–12. 是否满足下一阶段前置条件

不满足。下一阶段的冻结 B2 + Gate-1 deployment verification 需要两个已生成、带完整 ground truth 的 independent batch。当前两个 batch 都不存在。

## 停止原因

仓库里已有的长度常量互相不同：

- `LLV2_TRAIN_N = 64`，这是训练池长度。
- `LLV2_DESIGN_N = 16`，这是设计池长度。
- Phase-2 / Phase-3 `VALIDATION_N = 64`，这是历史验证池长度。
- 不存在 `LLV2_CONFIRMATION_N`，也不存在 deployment holdout batch size。

这些常量描述的是不同角色的旧池，不是 Candidate-C 两个新 batch 的正式长度。按本阶段规则，不能自行假设 N=64，也不能在 16 和 64 之间自行选择。

因此标记 **BATCH_SIZE_SPECIFICATION_REQUIRED** 并停止。没有通过修改 B2、τ、criterion 或复用旧 scenario 来绕过。

## 最终状态

- status = BLOCKED_ON_INDEPENDENT_HOLDOUT_DATA
- planner_ready = False
- data_registered = False
- deployment_verified = False

本阶段停止。不进入 Phase-4E。

## 源文件 `HTW_Phase4D7C_HoldoutBatchSizeSpecificationAudit_CN.md`

# HTW Phase-4D-7C Holdout Batch Size Specification Audit

本阶段只审计项目里是否已有正式的 deployment holdout batch size。未生成 scenario，未选择任何 N，未修改 B2、τ、Gate-1、Candidate-C、planner 或 `decide()`。未做 deployment PASS/FAIL。未进入 Phase-4E。

**holdout_batch_size_status = NOT_SPECIFIED**

**deployment_criterion = CONSERVATIVE_WITH_FALLBACK_V1**

**status = BLOCKED_ON_INDEPENDENT_HOLDOUT_DATA**

**planner_ready = False**

**data_registered = False**

**deployment_verified = False**

## 1. 有没有正式定义？

没有。没有 `LLV2_CONFIRMATION_N`，没有 `deployment_holdout_batch_size`，没有 Batch-1 size，没有 Batch-2 size。

`authority_level = A`（正式 specification）的来源数量 = 0。因此 **holdout_batch_size_status = NOT_SPECIFIED**。

## 2. 找到的相关数字及其性质

| 定义 | 值 | 文件 | 性质 | 能否当作 holdout size |
|---|---:|---|---|---|
| `LLV2_TRAIN_N` | 64 | `src/wt/lowlevel_v2/protocol.py` | C. HISTORICAL_TRAIN_SIZE | 否 |
| `LLV2_DESIGN_N` | 16 | 同上 | C. HISTORICAL_DESIGN_SIZE | 否 |
| Phase-2 `VALIDATION_N` | 64 | `src/wt/lowlevel_v2/phase2/protocol.py` | C. HISTORICAL_VALIDATION_SIZE | 否 |
| Phase-3 `VALIDATION_N` | 64 | `src/wt/lowlevel_v2/phase3/protocol.py` | C. HISTORICAL_VALIDATION_SIZE | 否 |
| Phase-10.4/10.5/10.6 区间 | 64 | `protocol.py` 的 `range(104000/105000/106000, +64)` | C. 历史 independent / final test 规模 | 否 |
| Candidate-C `required_independent_batches` | 2 | `deployment_criterion_registry.json` | A 只规定 batch 个数，不规定每个 batch 的 scenario 数 | 否，不是 batch size |
| Round-33 `SEEDS_PER_STRATUM` | 4 | `src/wt/round33/scenarios.py` | C. 另一套 maritime 长时程实验，不是 L3-B B2 holdout | 否 |

`LLV2_confirmation` 与 `LLV2_final_untouched` 在 `data_registry.py` 中仍是 `None`，没有长度字段。

## 3. 为什么 64 不能直接使用？

64 是训练池、历史验证池和 Phase-10 评估池的长度。这些常量没有写成 “deployment holdout batch size = 64”。把训练规模继承成部署验证规模，等于自行选择 N。

## 4. 为什么 16 不能直接使用？

16 只是 `LLV2_design` 的长度，而且这 16 个 scenario 已经进入 B2 fit。它既不是 holdout 规范，也不能作为新 batch 的来源。

## 5. C8=2 与 batch size 的区别

C8 `required_independent_batches = 2` 只表示需要两个彼此独立的 batch。它不表示每个 batch 有 2 个 scenario，也不表示 batch size 等于 2。每个 batch 里有多少 scenario，仍然没有正式定义。

## 6–7. 隐式数量有没有正式权限？

有实现默认值和历史规模，见上表。它们是工程常量或旧实验规模，不是 deployment holdout specification。测试夹具、mock 和 debug 计数即使存在，也没有正式权限。本阶段没有把任何隐式数字提升为规范。

## 8. 现在能否生成 Batch-1 / Batch-2？

不能。没有正式 batch size，就不能生成，也不能登记。

待人工填写的模板（未填候选值）：

`src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/holdout_batch_size_specification_pending.json`

```json
{
  "status": "HUMAN_SPECIFICATION_REQUIRED",
  "batch_1_size": null,
  "batch_2_size": null,
  "required_independent_batches": 2,
  "source": null,
  "human_approved": false
}
```

## 最终状态

- holdout_batch_size_status = NOT_SPECIFIED
- deployment_criterion = CONSERVATIVE_WITH_FALLBACK_V1
- status = BLOCKED_ON_INDEPENDENT_HOLDOUT_DATA
- planner_ready = False
- data_registered = False
- deployment_verified = False

本阶段停止。不进入 Phase-4E。

## 源文件 `HTW_Phase4D7E_HoldoutBatchSizeRegistration_CN.md`

# HTW Phase-4D-7E Holdout Batch Size Registration

本阶段只登记项目负责人指定的 deployment holdout batch size。未生成 Batch-1 或 Batch-2，未调用 `L3BEnv.reset()`，未分配 scenario ID，未 rollout，未计算 `U_single`、`U_traj`、B2、Gate-1 或 ΔR。未修改 B2、τ、Gate-1、criterion、planner 或 `decide()`。未做 deployment PASS/FAIL。未进入 Phase-4E。未修改 `data_registry.py`、`LLV2_DATA_REGISTRY.json` 或 `deployment_criterion_registry.json`。

**holdout_batch_size_status = SPECIFIED**

**deployment_criterion = CONSERVATIVE_WITH_FALLBACK_V1**

**status = BLOCKED_ON_INDEPENDENT_HOLDOUT_DATA**

**planner_ready = False**

**data_registered = False**

**deployment_verified = False**

## 1. Phase-4D7C 结论

Phase-4D7C 的结论是 `holdout_batch_size_status = NOT_SPECIFIED`。仓库当时没有正式 deployment holdout batch size。

## 2–4. 人工规格

项目负责人现在正式指定：

- Batch-1 size = 64 scenarios
- Batch-2 size = 64 scenarios
- required_independent_batches = 2
- total_new_holdout_scenarios = 128

## 5. 为什么不是从历史 64 推导

这两个 64 是本阶段的人工作出的规格。它们不是复制 `LLV2_TRAIN_N`，不是复制 `VALIDATION_N`，不是拆分已有 220000–220063，也不是从历史结果优化出来的。

## 6. C8=2 与 batch size

C8 仍然只表示需要两个 independent batch。每个 batch 的 scenario 数量是本次另行指定的 64，不是把 C8 解释成 batch size=2。

## 7. 禁止复用的历史 ID

Batch-1 和 Batch-2 必须是新的 scenario realization。不得使用：

- 220000–220063
- 221000–221015
- 222000–222063
- 223000–223063
- 104000–104063
- 105000–105063
- 106000–106063

220000–220063 即使数量也是 64，也不能直接当作 Batch-1。106000–106063 即使曾被称为 untouched candidate pool，也不能直接当作 Batch-1。

## 8. 下一阶段生成条件

下一阶段才允许生成：Batch-1 的 64 个全新 realization，以及 Batch-2 的 64 个全新 realization。新的 scenario ID 是必要条件，不能单独证明 independent。生成时还不得进入 B2 fit、feature selection、hyperparameter selection、τ selection、Gate-1 selection 或 criterion selection，也不得属于旧训练、设计、验证或拒绝池。

## 9. 当前状态

规格已登记，两个 independent batch 尚未生成。因此 data_registered 仍为 False，不得宣布 deployment PASS。

Batch size specification 已正式注册，但 independent holdout data 尚未生成，因此不得宣布 deployment PASS，也不得进入 Phase-4E。

规格文件：`src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/holdout_batch_size_specification_pending.json`

项目没有单独的 holdout specification registry。本阶段没有改全局 registry，上述文件就是本次登记结果。

## 源文件 `HTW_Phase4E_FailureAttribution_CN.md`

# HTW Phase-4E Failure Attribution Audit

本阶段只对已经完成的 Phase-4E 失败做事后归因。没有重训 B2，没有修改 τ、Gate-1、Candidate-C、planner 或 decide()，没有删除或重生成 holdout。

**status = DEPLOYMENT_VERIFICATION_FAILED**

**deployment_verified = False**

**planner_ready = False**

**criterion = CONSERVATIVE_WITH_FALLBACK_V1**

**tau = 1534933.5944128654**

## 结论

四个 degradation 都发生在 Gate-1 按冻结规则放行 B2 的 scenario 上，而且 margin 与真实 ΔR 方向相反。B2 的 top-1 在这四场都不是 oracle。Candidate-C 的失败由 B2 ranking error 和具有误导性的 margin 共同造成。

failure is not merely a threshold-boundary effect.

230091 的 margin/τ = 1.013，靠近阈值。230070、230077、230078 的 margin/τ 分别是 1.427、1.265、2.482。其中 230078 落在 margin ≥ 2τ 的桶里，真实 ΔR 仍是 -0.182。

是否应该修改 B2、τ 或 Candidate-C：**NOT ESTABLISHED**。本审计不提出新阈值。

## Batch-1 / Batch-2

Candidate-C 的直接统计原因在 Batch-2：

| 项 | Batch-1 | Batch-2 |
| --- | --- | --- |
| n | 64 | 64 |
| Gate-1 选择 B2 | 20 | 30 |
| degradation | 0/64 | 4/64 = 0.0625 |
| mean ΔR | 0.101687 | 0.113262 |
| worst-case ΔR | 0 | -0.200 |
| C1 ≤ 0.02 | PASS | FAIL |
| C2 ≥ -0.125 | PASS | FAIL |
| C3 ≥ 0.02 | PASS | PASS |
| C4 ≥ 0 | PASS | PASS |

Batch-1 PASS，Batch-2 FAIL。Combined 不能抵消 Batch-2。`independent_batches_passed = 1`。

## 四个失败 scenario

ΔU 的单位是 L3-B 的 `block_payload_bits = 1e6`。128 场已实现的 ΔU 全部是这个步长的整数倍。四场失败分别是 -2、-2、-2、-1 个 payload block。这是 evaluation reward 的离散结构，不是连续残差被四舍五入。

| scenario | margin | margin/τ | 桶 | ΔR | ΔU | B2 regret | LOCAL regret | B2–ORACLE gap R | B2 top-1 = oracle | oracle 在 B2 top-3 | LOCAL top-1 = oracle | B2 的 U_traj 排名 | LOCAL 的 U_traj 排名 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 230070 | 2.190856e6 | 1.427 | 1.25τ–1.5τ | -0.181818 | -2e6 | 5e6 | 3e6 | 0.454545 | 否 | 是 | 否 | 6 | 2 |
| 230077 | 1.941896e6 | 1.265 | 1.25τ–1.5τ | -0.200000 | -2e6 | 4e6 | 2e6 | 0.400000 | 否 | 否 | 否 | 11 | 4 |
| 230078 | 3.810212e6 | 2.482 | ≥ 2τ | -0.181818 | -2e6 | 4e6 | 2e6 | 0.363636 | 否 | 否 | 否 | 9 | 7 |
| 230091 | 1.555024e6 | 1.013 | τ–1.25τ | -0.090909 | -1e6 | 1e6 | 0 | 0.090909 | 否 | 是 | 是 | 2 | 1 |

Gate-1 四场都选择 B2。LOCAL candidate / B2 candidate / oracle 分别是：

- 230070：6 / 20 / 19
- 230077：22 / 20 / 2
- 230078：16 / 5 / 6
- 230091：7 / 5 / 7

四场的 B2 candidate 在 U_single 上的排名是 3、3、2、3。也就是说 B2 放行的是 U_single 的第 2 或第 3 名，不是一个 U_single 很差的远端 candidate。它的 U_single 低于 LOCAL，C_hat 明显高于 LOCAL，于是 U_hat margin 超过 τ。真实 U_traj 上，这四个被选中的 candidate 分别排第 6、11、9、2，LOCAL 分别排第 2、4、7、1。

LOCAL 在四场上的 regret 都小于 B2。只有 230091 的 LOCAL 正好是 oracle。其余三场 LOCAL 也没有选中 oracle，只是比 B2 更接近。

## margin-risk

按已实现的 Gate-1 结果分桶。margin < τ 时选择 LOCAL，所以该桶的 ΔR 和 U gain 为 0。

| 桶 | n | n_B2 | degradation | rate | mean ΔR | median ΔR | worst ΔR | mean U gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| margin < τ | 78 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| τ ≤ margin < 1.25τ | 13 | 13 | 1 | 0.0769 | 0.1702 | 0.2000 | -0.0909 | 2.000e6 |
| 1.25τ ≤ margin < 1.5τ | 4 | 4 | 2 | 0.5000 | 0.1045 | 0.1091 | -0.2000 | 1.000e6 |
| 1.5τ ≤ margin < 2τ | 14 | 14 | 0 | 0 | 0.3818 | 0.4167 | 0.1667 | 4.571e6 |
| margin ≥ 2τ | 19 | 19 | 1 | 0.0526 | 0.3042 | 0.3571 | -0.1818 | 4.000e6 |

失败没有集中在 τ 边界。靠近边界的只有 230091。1.25τ–1.5τ 有两场失败，margin ≥ 2τ 还有 230078。高 margin 桶的 mean ΔR 仍然为正，失败是桶内的尾部，不是整个高 margin 区域都变差。

## B2 ranking

这里的 top-1 / top-3 是 argmax U_hat 相对 argmax U_traj，不是 Gate-1 最终选择。

| 集合 | B2 top-1 recovery | B2 top-3 recovery | LOCAL top-1 recovery |
| --- | --- | --- | --- |
| Batch-1 | 0.421875 | 0.765625 | 0.203125 |
| Batch-2 | 0.296875 | 0.656250 | 0.140625 |
| Combined | 0.359375 | 0.710938 | 0.171875 |

四个失败场景上，B2 top-1 都不等于 oracle。230077 和 230078 的 oracle 也不在 B2 top-3。230070 和 230091 的 oracle 在 B2 top-3，但被选中的仍是另一个 candidate。230091 的 LOCAL top-1 就是 oracle。

因此这不是“Gate-1 单独放错、B2 排序正确”，也不是“只有阈值边界上的一次误放”。冻结规则确实放行了 B2；放行之后，B2 排序在这四场上相对真实 U_traj 明显偏差，而 margin 没有反映出随后的负 ΔR。

## LOCAL fallback 的事后对照

Batch-2 中 margin < τ 的 scenario 有 34 个。它们都被 Gate-1 选为 LOCAL：degradation = 0，mean ΔR = 0，worst ΔR = 0。

Batch-2 中被放行的 30 个 B2 scenario：degradation = 4，mean ΔR = 0.241626，worst ΔR = -0.200。

当前 τ 没有把这 4 个高风险 scenario 留在 LOCAL。它们的 margin 都大于 τ。若在这 30 场上事后改成强制 LOCAL，ΔR 会变成 0，但那也会拿掉这 30 场目前为正的平均 ΔR。这是对照，不是新的 Gate-1。本阶段不据此修改 τ。

## deployment-time 共同模式

四场失败在 deployment-time 可观测量上的共同点是：B2 candidate 的 U_single 低于 LOCAL，C_hat 明显高于 LOCAL，margin 仍大于 τ。和 Batch-2 里其余 26 个被放行且未退化的 B2 scenario 相比，失败样本的 margin 均值是 2.374e6，未失败样本是 2.804e6，失败样本的 margin 并不更大。Phase-3E geometry 差值方向也相同，例如 B2 candidate 的 `mean_uav_target_dist` 都低于 LOCAL。

事后才能分开它们的量是真实 U_traj 排名：失败样本的 B2 candidate 平均排第 7，未失败的 B2 candidate 平均排第 2.27。U_traj 排名不是 deployment-time 输入。这里只把它记为 failure-associated observable pattern，不把它变成新特征，也不据此改模型。

## distribution shift

candidate 数两边都是 24。描述统计：

| 量 | Batch-1 mean / median | Batch-2 mean / median |
| --- | --- | --- |
| margin | 1.249e6 / 1.310e6 | 1.599e6 / 1.512e6 |
| B2 使用率 | 0.3125 | 0.46875 |
| U_single(LOCAL) | 4.453e6 / 4.400e6 | 4.353e6 / 4.400e6 |
| R_LOCAL | 0.743 / 0.727 | 0.746 / 0.739 |
| U_traj(LOCAL) | 9.297e6 / 8.000e6 | 8.828e6 / 8.000e6 |
| candidate U_hat | 5.716e6 / 6.037e6 | 5.797e6 / 6.175e6 |
| candidate geometry | 均值接近；`competition_count` 两边都是 0 | 同左 |

Batch-2 的 margin 分布更高，所以 B2 使用率更高。U_single、R_LOCAL 和 Phase-3E geometry 没有同等幅度的分离。不能把 Candidate-C 失败单独写成环境 distribution shift。

## 十一个问题

1. 四个 degradation 是否都发生在 B2-selected scenario？是。
2. 它们的 margin 是否都 ≥ τ？是。
3. failure 是否集中在 τ 附近？否。只有 230091 靠近 τ。
4. 高 margin 是否仍有 failure？是。230078 的 margin ≥ 2τ。
5. B2 ranking 在 failure scenario 上是否明显错误？是。四场 top-1 都不是 oracle；两场 oracle 不在 top-3；被选 candidate 的真实排名为 6、11、9、2。
6. LOCAL 是否更接近 oracle？是，四场 LOCAL regret 都更小。只有 230091 的 LOCAL 正好是 oracle。
7. Batch-2 相对 Batch-1 是否有 distribution shift？margin 和 B2 使用率更高。U_single、R_LOCAL 和 geometry 接近。
8. Candidate-C 失败的直接统计原因是什么？Batch-2 的 degradation rate 0.0625 > 0.02，且 worst-case ΔR -0.200 < -0.125。
9. 能否据此判断应该修改 B2？**NOT ESTABLISHED**。
10. 能否据此判断应该修改 τ？**NOT ESTABLISHED**。
11. 能否据此判断应该修改 Candidate-C？**NOT ESTABLISHED**。

## 下一步

允许继续做只读审计，或由项目负责人决定是否单独立项。

禁止：重训 B2，修改 feature、hyperparameter、seed、τ、Gate-1、C1–C8，修改 planner / decide()，删除或替换 scenario，重生成 holdout，根据这次失败调参，宣布 deployment PASS，设置 planner_ready = True，进入 Phase-4F，进入 planner integration。

本阶段只完成 failure attribution。

## 源文件 `HTW_Phase4E_RemediationDevelopmentPoolSpecificationAudit_CN.md`

# HTW Phase-4E Remediation Development Pool Specification Audit

本阶段只审计是否存在尚未占用的合法 L3-B development scenario source。没有修改 B2、τ、Gate-1、Candidate-C、planner 或 `decide()`。没有生成 scenario，没有 rollout，没有把已有池改标成 development data，也没有改动 230000–230127。

**eligible_development_data = 0**

**status = HUMAN_SPECIFICATION_REQUIRED**

**phase_status = DEPLOYMENT_VERIFICATION_FAILED**

**deployment_verified = False**

**planner_ready = False**

## 审计结果

不存在可以建立 development pool 的已注册 source。

已检查 `LLV2_DATA_REGISTRY.json`、Phase-2/3 validation manifest、Phase-10 协议范围、deployment holdout manifest，以及上一阶段的 528 行 scenario 用途矩阵。`eligible_for_future_development` 仍全为 0。`LLV2_confirmation` 和 `LLV2_final_untouched` 仍是 null。另外 1353 个历史 blocked seed 不是 L3-B B2 development source。

因此本阶段不选择 pool size，不选择 seed，不生成新 scenario，也不宣布某个新区间已经是 development pool。

## Data lineage

逐 scenario 记录仍在 `src/wt/lowlevel_v2/phase4e_development_audit/outputs/phase4e_remediation_data_usage_matrix.csv`。下表是按池汇总，没有改任何标记。

| pool | ID | historical role | B2 fit | feature | hyperparameter | τ / Gate-1 | criterion | Phase-4C | Phase-4E | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLV2_design | 221000–221015 | DESIGN | 是 | 是 | 否 | 是 | 否 | 否 | 否 | 否 |
| LLV2_train | 220000–220063 | VALIDATION | 否 | 否 | 否 | 否 | 否 | 是 | 否 | 否 |
| phase2_val | 222000–222063 | HISTORICAL_TEST | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 否 |
| phase3_val | 223000–223063 | HISTORICAL_TEST | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 否 |
| phase104 | 104000–104063 | HISTORICAL_TEST | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 否 |
| phase105 | 105000–105063 | HISTORICAL_TEST | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 否 |
| phase106 | 106000–106063 | HISTORICAL_TEST | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 否 |
| deployment holdout | 230000–230127 | DEPLOYMENT_HOLDOUT_CONSUMED | 否 | 否 | 否 | 否 | 否 | 否 | 是 | 否 |
| LLV2_confirmation | 无 | ABSENT | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 否 |
| LLV2_final_untouched | 无 | ABSENT | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 否 |

Hyperparameter 没有在任何池上搜索，ET 是预先固定的。Candidate-C 的 C1–C8 来自人工规格，没有 scenario pool 参与选择。τ 来自 221000–221015 的 shadow mean `dU_hat`。

这些池都不能改标为 development data：design 已经用于 fit、特征消融和 τ；train 已经是 Phase-4C validation；历史测试池没有 B2 协议标签且被协议拒绝；holdout 已经消耗。

## 需要人工指定

待定规格写在 `src/wt/lowlevel_v2/phase4e_development_pool_specification/outputs/development_pool_specification_pending.json`。六个字段都是 null，`human_approved = false`。

1. development pool size
2. scenario ID / seed allocation policy
3. 与 221000–221015、220000–220063、222000–223063、104000–106063、230000–230127 的隔离规则
4. 是否允许重新生成新的 L3-B scenario
5. development pool 是否必须是独立 batch
6. pool 完成后允许哪些 B2 remediation 操作

在这六项被人工填写并批准之前，不能生成 scenario，也不能开始 B2 remediation。

本阶段没有进入 Phase-4F，没有进入 planner integration，没有宣布 deployment PASS。

## 源文件 `HTW_Phase5_hierarchical_design_report.md`

# HTW Phase-5 True Hierarchical Task Controller Design

只做设计。未改代码、未训练、未加 HGNN、未开 Round、未改环境 / PPO 核心 / Transformer。

依据：Phase-1 表示、Phase-1.5 诊断、Phase-2 tracking、Phase-2.5 λ、Phase-3 条件化、Phase-4 换 planner。

---

## 1. 为什么当前 HTW 失败

现有实现不是分层控制，而是 **同一个扁平 PPO** 多看了几个由规则算出的通道：

```
oracle/planner → TW=(G, W, H) 拼进 UAV 特征
        → 一个 DeepSets-PPO → movement
```

时间尺度、优化目标和动作接口都没有拆开。

| 现象 | 含义 |
| --- | --- |
| TW 能进 actor/critic（dim 16） | 接口通了，不是「没喂进去」 |
| tracking 让 ρ 上升、payload 下降 | 低层 **可以** 跟 W，但跟 W 与 `r_comm` 冲突 |
| 取消 tracking 后 payload = State-only | 条件化通道被忽略，等于没层次 |
| TW_A / B / C 都无稳定 payload 增益 | 换更好的 W 仍无效，瓶颈不在 TW 质量 |

失败机制可以收成三条：

1. **时间尺度混在一起。** High-level 的 H=5 子目标与低层逐步 `movement` 同一频率、同一策略更新。扁平 PPO 没有「先选任务、再执行若干步」的 SMDP。
2. **目标函数混在一起。** 通信收益在 matching + 队列上；XY waypoint 是运动几何。用一个 `r_comm` 训练「又要传数据又要到 W」会互相抵消；用 `r_comm−λd` 则运动压过通信。
3. **高层动作选错了空间。** `W∈R^2` 对低层不是可完成的离散任务。环境真正的任务开关是 **排他 matching（谁服务哪颗 buoy）** 和队列，不是「5 步后的平面点」。State 里已有位置、队列、当前匹配；规则 TW 近似是 state 的函数，低层没有新的决策权。

因此：不是 HGNN 缺失，也不是再调 λ / 再换 `joint_plan`。需要 **两个策略、两种动作、两种奖励、两种时间尺度**。

---

## 2. 新的 hierarchical 结构

```
s_t  (Round-30 当前可观测量，环境不改)
        │
        ▼
High-level Task Policy  π_H(z | s)     每 τ 步更新一次
        │
        ▼
z_t  任务目标（对低层冻结 τ 步）
        │
        ▼
Low-level Motion Policy π_L(a | s, z)  每环境步
        │
        ▼
a_t = movement ∈ R^{M×2}
        │
        ▼
env.step（matching / 物理 / r_comm 原样）
```

约束（与已锁定实验一致）：

- 环境、reward 公式、matching、安全投影不改。
- 低层动作空间仍是 Round-19/30 的连续位移。
- High-level **不直接输出环境动作**；它只输出 z。
- PPO 核心（GAE / clip / Adam）可复用两份独立 agent，而不是改 `round30/ppo.py` 的公式。
- HGNN 若将来使用：只在 π_H 上对 UAV–buoy 关系建模；本期设计 **不包含 HGNN**。

τ：建议 5 或 10 环境步，与已有 H 对齐，使 high-level 真正低频。

---

## 3. High-level 应该输出什么

比较四种 z。

### A. Buoy selection（推荐作为最小接口）

每机选择 `k∈{0..K−1}∪{idle}`，或输出对 buoy 的分布后取 argmax。环境仍用 **冻结 matching** 决定实际链路；z 只表示「本段任务意图」，不改 `env.step`。

- 优点：与排他服务、队列、干扰直接对应；离散、可完成、可检查「是否在服务 k」；低层目标清楚（靠近 k、提高对 k 的速率）。
- 缺点：真正匹配可能与意图不一致（高层是建议不是执行器）；K=5 时联合意图空间为 `(K+1)^M`，M=2 仍可枚举。

### B. Service region

输出服务区域（圆心+半径，或对 buoy 的软掩码）。

- 优点：比单点 W 更宽容，减轻 Phase-2 那种点跟踪。
- 缺点：区域仍是几何，不直接对应 matching；完成判定模糊；和现有失败的 waypoint 同类。

### C. Latent task embedding

Feudal / 无监督 z∈R^d。

- 优点：表达力强，不手写任务。
- 缺点：不可解释、难做 completion；当前样本和预算下极易坍缩成被忽略的噪声通道（Phase-3 已出现）。**不作为下一阶段最小实验。**

### D. Waypoint（已证伪作为高层动作）

`W∈R^2` 或 `(G,W,H)`。

- 优点：连续、已有 oracle。
- 缺点：Phase-2～4 表明它既不是通信最优接口，也不能被扁平策略当任务完成。只可作低层的 **可选辅助**，不再作为 high-level action。

**选择：High-level action = 每机 buoy 选择（A），可选附带 hold 长度 τ。**  
这是海洋采集里「先定服务对象、再运动」的自然分层。G 可作为该选择的 one-hot/分布，但不再把 XY 的 W 当高层动作。

---

## 4. Low-level / High-level 动作定义

### High-level

- 输入：当前 s（可另加 buoy 队列、粗粒度干扰统计；仍只用环境已暴露量）。
- 输出：`z = (k_1,…,k_M)`，k_m 为目标 buoy 或 idle。
- 频率：每 τ 步一次；在区间内 z 对低层恒定。
- 优化：区间结束时的 **通信结果**（累计 payload / 区间 `r_comm` 和），不是点跟踪。

### Low-level

- 输入：`s` + `z`（例如目标 buoy 的 one-hot、该 buoy 相对位置、该 buoy 队列；**不要**再拼一套与 z 无关的长窗 W 作为必须通道）。
- 输出：与现在相同的 `movement (M,2)`。
- 频率：每环境步。
- 优化：**goal completion**，即「在冻结 z 下把任务做完」，不是全局 payload 最优。

Completion 建议用可测、且不改环境的量（只在 controller 内）：

- 接近：`d(p_m, buoy_{k_m})` 下降；
- 服务：该步实际 assignment 等于 k_m、或对该 buoy 的瞬时 payload > 0；
- 冲突：两机选同一 buoy 时 completion 降权（反映排他 matching，但不改 matching 代码）。

---

## 5. Reward 设计（环境公式不动）

环境继续只产生 `r_comm`（及现有分解）。分层奖励是 **controller 内部** 的两套信号。

### High-level（长期通信）

对每个高层决策，在随后 τ 步上：

`R_H = sum_{t=t0}^{t0+τ-1} r_comm(t)`

或区间 payload 增量。这是唯一对齐「最大化采集量」的层。不要在 R_H 里减 `‖p−W‖`。

### Low-level（任务完成）

`R_L = r_complete − β d(p, buoy_z)  [+ 可选很小的 ε r_comm]`

- `r_complete`：本步服务于 z 指定 buoy 则正，否则 0。
- 距离项只对 **选定 buoy**，不是任意 planner 轨迹点。
- `ε=0` 作为最小消融：证明低层先会完成任务；若完成了但 R_H 仍差，再给低层微弱通信项。

禁止再把 `r_comm − λ‖p−W‖` 当作唯一损失（Phase-2 已否定）。

训练顺序建议（HIRO 式，但不把目标设成 XY）：

1. 先训低层：z 来自 oracle（当前 joint-aware matching 或最大 queue buoy），只优化 R_L。
2. 冻结或慢更新低层，再训高层 π_H 优化 R_H。
3. 可选：hindsight 把「实际服务过的 buoy」回标成 z，减轻稀疏 completion。

---

## 6. 与 HIRO / Options / Feudal / GC-RL

| 方法 | 核心 | 对本任务 |
| --- | --- | --- |
| Goal-conditioned RL | π(a\|s,g)，g 常为状态子空间 | Phase-3/4 已做「g=TW」；XY 目标不合适。应改成 **g=离散 buoy**。 |
| HIRO | 高低层不同频率；高层出状态目标；off-policy + 回标 | **时间尺度和回标**值得借鉴；**不要**把目标设成绝对 XY（与 HIRO 原论文的 state-goal 同类，已被我们的 W 实验打脸）。目标改为 buoy id / 相对向量到选定 buoy。 |
| Options / SMDP | 高层选 option，option 有内部策略与终止 | **最贴合。** Option =「服务 buoy k，直到 τ 或队列清空/匹配失败」。低层是 option 策略。离散 option 集合小（K+1）。 |
| Feudal RL | Manager 出隐空间目标 | 表达力过剩、不可查 completion；预算小。可列为远期，不是最小原型。 |

适合当前 M=2 共享频谱采集的组合：

**Options（高层选 buoy）+ 短时 goal-conditioned 低层（以该 buoy 为 g）+ HIRO 的低频与回标。**  
不适合：再做 Feudal 隐变量，或继续 HIRO-XY / TW waypoint。

---

## 7. 代码修改范围（设计，尚未实施）

保持不动：

- `round19` 环境、matching、`r_comm`
- `round30/ppo.py` 的 GAE/clip/loss 公式
- HGNN、Transformer、TW 合同若仍当日志可留，但 **不再作为高层动作**
- Round / benchmark / 受保护 JSONL

新增（未来实现时，独立包，例如 `htw_phase6`，不开 Round）：

- High-level 动作编解码：buoy id ↔ one-hot
- 低频包装：每 τ 步调用 π_H，中间把 z 写入 conditioner
- Low-level `ConditionedVector`：特征 = state + z（不用 λd 到任意 W）
- 两套 reward 记账：R_H 积区间 `r_comm`，R_L 用 assignment/距离
- 两份 `PPOAgent` 或一份低层 + 规则/可学习高层
- 评估：payload、return、queue、**意图达成率**（assignment==z）、**不再**以 action–W cosine 为主指标

不在最小原型里做：HGNN、可学习 Transformer 高层、改环境 matching 为「执行 z」。

---

## 8. 下一阶段最小实验（仍不训 HGNN）

目标：验证 **离散任务 z + 完成型低层** 是否比「拼 TW 的扁平 PPO」更能改行为，且不先追求超过 State-only。

协议（与 Phase-2/3 同 smoke 预算即可）：

1. **Oracle-z 低层：** z = 每 τ 步用当前 `joint_aware` / 最大队列 buoy（规则，不学习高层）。π_L(a\|s,z)，奖励 R_L，ε=0。  
   看：意图达成率、对目标 buoy 的距离是否下降、payload 是否被伤害。
2. **对照：** State-only；Phase-3 TW 条件化（无 tracking）。
3. **通过线（层次开始成立）：** 达成率明显高于 TW 条件化，且 payload **不低于** State-only 超过噪声（允许持平）。  
   若达成率高但 payload 明显掉：z 的 oracle 与通信冲突，再改 z 的生成，而不是加 HGNN。
4. **不通过：** 低层仍忽略 z → 先查特征与 R_L，而不是上高层学习。
5. 只有 (3) 通过后，才做 **可学习 π_H** 优化 R_H。那才是完整 HTW-C。

本期停止：不实现、不训练、不加 HGNN。

## 源文件 `HTW_Phase5D_TrajectoryAwareUtilityRankingDesignAudit_CN.md`

# HTW Phase-5D Trajectory-Aware Utility Ranking Design Audit

**planner_ready = False**

**deployment_verified = False**

本阶段只核对数据流并写出下一版 high-level 的标签规格。没有改代码，没有改模型，没有训练，没有生成 scenario，没有改 checkpoint，没有调用 `decide()`，没有重训 Residual MAPPO，没有修改 B2、τ、Gate-1、Candidate-C 或 planner，没有使用 230000–230127，也没有进入 deployment verification 或高低层联合训练。

## 1. 当前 high-level pipeline

在线路径是：

```text
当前环境状态
    -> feature_dict_l3b + build_graph
    -> encoder
    -> OrderFreeSetDecoder logits
    -> joint greedy
    -> 长度 6 的 assignment
```

`high_level.py` 的在线函数只是加载冻结 DeepSets 并做这次解码。本阶段没有调用它。解码器每步输出的是 shape `(6, 13)` 的 logit：12 个浮标加 idle。贪心选完后变成无人机到浮标编号或 -1。它不是轨迹，也不是 payload。

有两套训练目标，不能写成同一个。

| 环节 | 冻结 DeepSets 检查点 | Phase-5 HGNN 探针 |
| --- | --- | --- |
| 输入 | 当前 UAV、浮标、UAV–Buoy 边、UAV–UAV 边。能耗槽为 0 | 同一套当前图。完整探针另在浮标上拼接队列比例 |
| encoder | DeepSets：UAV、浮标、边各自 MLP，没有消息传递 | HGNN：UAV–Buoy 与 UAV–UAV 消息传递 |
| decoder 输出 | `(6, 13)` logit，再贪心成 assignment | 同一个 `OrderFreeSetDecoder` |
| label | 候选池上的 15 步 fly-toward payload，再做成 softmax 软标签 | 一条排他匹配：`queue * 当前速率 / (distance + 1)` |
| loss | `L_util + 0.25 L_rank + 0.1 L_ref`。τ = 1.0，本阶段不改 | 这条匹配的 order-free set NLL |
| reward | 无。解码时不算 payload | 无 |
| label 是否含未来轨迹 | 含结果，不含轨迹本身。payload 来自离线 15 步 fly-toward | 不含。只用当前队列、距离和假设速率 |

冻结 DeepSets 已经在候选池上贴近 15 步 payload，但池外目标不会进入 loss，在线解码也不再算 payload。Phase-5 HGNN 则是 assignment similarity。Phase-5C 在 t=15 的探针里，这个匹配分数和 15 步 payload 的 Spearman 是 -0.39，Top-1 相同比例是 0。这就是要换掉的目标。

## 2. 新的 utility label

不调用 `decide()`。离线 shadow 的规格如下。

对一个决策时刻，冻结当前 UAV 状态、其他无人机的 assignment 和环境。只替换一名无人机的候选浮标 `k`，用已有的 fly-toward 在副本上走 15 步，读 payload 增量。副本轨迹不写回网络输入。

```text
U_k = α * payload_gain
    + β * urgency
    - γ * travel_cost
    - δ * interference_cost
    - η * energy_cost
```

系数现在不拟合。

| 项 | 定义 | 现在的取值 |
| --- | --- | --- |
| payload_gain | 该候选锁定后，15 步 fly-toward 的 payload 增量 | 只做 label |
| urgency | `u_k = q_k / (max q + 1e-8)` | 只有队列比例 |
| travel_cost | UAV–Buoy 边上已有的 distance / ETA | 用当前几何，不新造特征 |
| interference_cost | UAV–UAV 边上已有的当前干扰 | `δ = 0` |
| energy_cost | 环境没有能耗状态 | `η = 0`，不虚构 |

`δ = 0` 是因为 15 步 payload 已经由 SINR 和速率算出来，再减一次干扰会算两次。边上的干扰仍然可以进 encoder，只是不在 `U` 里再减。`η = 0` 是因为 L3-B 没有能耗变量，UAV 特征里的能耗槽恒为 0。

以后若环境真的记下等待时间、AoI、deadline 或能耗，再把它们加进 urgency 或 `η`。在那之前这些系数保持 0。

排序标签是同一状态下各候选 `U_k` 的名次，不是“是否等于某一条匹配”。

## 3. feature 和 label 的分离

high-level 的输入只能是决策时刻已经能看到的量。完整未来轨迹不能作为输入。

| 量 | 角色 | 是否含未来轨迹 |
| --- | --- | --- |
| UAV / 浮标状态、queue、当前 channel | encoder 输入 | 否 |
| UAV–Buoy distance、ETA | encoder 输入 | 否 |
| UAV–UAV 当前干扰 | encoder 输入 | 否 |
| 队列比例 urgency | encoder 输入 | 否 |
| 能耗 | 不使用 | 没有这个状态 |
| 15 步 payload_gain | 只做 label | 是离线 rollout 的结果 |
| 合成的 `U_k` 和它的名次 | 只做 label | 只因 payload_gain 而含未来结果 |
| 未来位置序列、未来动作 | 禁止进入 encoder，也不单独存成输入特征 | 是 |

label 可以来自 offline rollout。这次 rollout 的轨迹本身留在 evaluator 里，不拼进图。

## 4. HGNN 输入

encoder 保持现有的 UAV–UAV 边、UAV–Buoy 边、queue 和 channel。不增加会泄漏标签的特征。

| feature | 进入 encoder | 只做 label | 说明 |
| --- | --- | --- | --- |
| UAV–target distance | 是 | 否 | 已在 UAV–Buoy 边第 0 维 |
| ETA | 是 | 否 | 已在 UAV–Buoy 边第 2 维 |
| estimated payload gain | 否 | 是 | 15 步 shadow 的 payload。放进输入就是把标签泄漏给 encoder |
| interference relationship | 是 | 否 | 已在 UAV–UAV 边。`U` 里不再减一次 |
| urgency | 是，仅队列比例 | 比例本身不是未来标签 | 等待时间、AoI、deadline 仍没有 |
| queue | 是 | 否 | 当前队列 |
| channel | 是 | 否 | 当前假设速率，不是 15 步收益 |

因此本阶段不需要新的边类型。要换的是监督信号。

## 5. 三个方案

| 方案 | 做什么 | 复杂度 | 能否对准 payload | 要不要重训低层 |
| --- | --- | --- | --- | --- |
| 方案1 | 保持 DeepSets，只把 label 改成 `U` 的排序 | 低。encoder 和 MAPPO 都不动 | 部分。候选级 payload 能进 loss，但 DeepSets 仍不做 UAV–UAV 消息传递，池外目标仍可能看不见 | 否 |
| 方案2 | 现有 HGNN 边 + `U` 的排序标签 | 中。只换高层监督，不换低层 | 是当前要解决的那一项：不再优化 assignment similarity | 否 |
| 方案3 | 方案2 再加低层反馈或联合更新 | 高。要把 MAPPO 的回报写进高层标签或一起反传 | 理论上能继续收口，但 Phase-5B 已说明低层不是主瓶颈 | 是，或至少要联合查询低层 |

推荐方案2。

方案1 改动小，但冻结 DeepSets 已经有一套候选池上的 payload 软标签，在线问题并没有因此消失。Phase-5 换了 HGNN 却仍用匹配标签，payload 下降，说明缺的是 utility ranking，不是再换一次无监督的 encoder。

方案3 需要低层反馈。Phase-5B 里 oracle 分配交给当前 MAPPO 已经能产生 payload，而且本阶段禁止重训 MAPPO、禁止联合训练。现在不做。

方案2 不重训低层：label 用冻结 fly-toward 的 15 步 payload，MAPPO 检查点保持不动。高低层目标不一致先由“高层去拟合 15 步 payload 排序”处理，而不是由低层改目标。

## 6. 离线验证协议

以下协议只作为下一步的实验设计。本阶段不跑。

数据只用已经存在的隔离重置，例如 310100、310101。不新建 scenario 生成器，不使用 230000–230127，不进入 deployment verification，不调用 `decide()`。

在每个决策状态：

1. 用第 2 节的 shadow 得到候选的 `U` 和 15 步 payload，作为离线名次。
2. 让待检 high-level 只读当前图，输出同一批候选的分数。不把 shadow 轨迹送进模型。
3. 低层检查点不更新。若要报告执行 payload，用冻结 fly-toward 或已经存在的 MAPPO 权重做 15 步，二选一写清楚，不能混在一个数里。

| 指标 | 定义 | 合格方向 |
| --- | --- | --- |
| utility ranking Spearman | high-level 分数与 `U` 的 Spearman | 高于 Phase-5C 的匹配分数对 payload（-0.39） |
| Top-1 utility accuracy | argmax 分数是否等于 argmax `U` | 高于 Phase-5C 的 0 |
| Top-k payload recovery | `U` 的最优候选是否落在分数的前 k 名，k = 3 | 高于随机前 3 |
| oracle overlap | 贪心 assignment 与离线 `U` 最优单目标替换的重合比例 | 作为对照，不把旧簇 oracle 当成 payload 上界 |
| 15-step payload | 按该分数锁定目标后的 15 步 payload 增量 | 高于用匹配标签选出的目标 |

对照固定为三条，都不训练：当前匹配标签的排序、冻结 DeepSets 的 logit、方案2 的 utility 分数。Phase-5D 不产生这三条的新 checkpoint。

## 7. 停止点

方案2 只是下一阶段的规格。本文件不实现 shadow 训练循环，不改 loss，不改 τ。

`planner_ready = False`

`deployment_verified = False`

## 源文件 `HTW_Phase8_relational_complexity_report.md`

# HTW Phase-8 Relational Complexity Benchmark Redesign

只做 **Level-3 关系环境设计**。未改代码、未训练、未调网络、未改 HGNN、未改低层 PPO、未改 option hold、未改 reward、未用 Task-Waypoint、未开 Round。

依据：Phase-6.x 证明 HRL 在 L2-A 上成立；Phase-7 证明 **同一 HRL 下 MLP > DeepSets > HGNN**。本阶段目标是先证明「环境需要图」，再允许改编码器。

---

## 1. 为什么 L2-A 上 HGNN 没有优势

Phase-7（hold=15，冻结低层，`r_H=Δpayload`）：

| 高层编码器 | payload | option accuracy |
| --- | ---: | ---: |
| MLP | **1.575e7** | **0.210** |
| DeepSets | 1.25e7 | 0.131 |
| HGNN | 4.50e6 | 0.086 |

L2-A 已经有：M=5、K=12、D=3、远簇到达、排他 `x_mk`、空间延迟、以及「邻机距离内进度 ×1/n」。这些 **不足以** 构成图推理任务。

| L2-A 机制 | 实际决策形态 | 对编码器的含义 |
| --- | --- | --- |
| 排他投影「更近者留下」 | 冲突在环境里被消掉，不进策略损失 | 不必对 UAV–UAV 竞争做消息传递 |
| 干扰 = 邻机计数 1/n | 与具体 buoy 无关，近乎各向同性 | 没有 SINR 交叉项，边特征可被节点距离代替 |
| 一 buoy 至多一机 | 无协同补全 | 不存在必须对齐的 UAV–UAV collaboration |
| 双簇 + 近高远迟 | 最优近似「派若干机去远簇」 | 簇标签 + 最近队列即可，接近 **独立 matching 分类** |
| 图边随几何缓变 | MLP 用固定拼接顺序记住机号 | 身份泄漏压过置换不变 / 图归纳偏置 |

因此 Phase-7 失败的主因不是「HGNN 实现太差」这一条能单独解释：在 **可分解的匹配分类** 上，更简单的编码器更容易在 80 updates 内把 entropy 压下去。要验证 HGNN，必须让 **最优 z 依赖于邻居的选择**，而不是只依赖于自己的 (p, Q, η)。

判定条件（环境侧，先于再训 HGNN）：

1. 固定其他 UAV 的 option，只改自己的 z，未来 payload 变化小；**同时改两机** 的 z，出现非加性交互（竞争或协同）。
2. 最优 assignment 不能由「每机独立 argmax 节点分数」复现；需要知道 **谁已经占了哪条边**。
3. 逐步 Oracle 若忽略 SINR / 协同次数 / 竞争边，相对完整 Oracle 有稳定 payload 缺口。

L2-A 不满足 1–2：`project_exclusive` 把竞争变成最近者独占；协同不存在；SINR 不存在。

---

## 2. Level-3 场景定义

通信背景不变：多 UAV 采集多 buoy 队列。HRL 骨架冻结：

```
π_H(z|s)  --hold=15-->  z (buoy option / idle / 协同槽)
        --> 冻结 π_L(a|s,z) --> movement
```

**只换环境物理与合法 z 语义，不换低层、不换 hold、不换高层 reward 公式、不改现有 HGNN 代码。**

### 2.1 建议规模（第一次实现不要跳到 L2-C）

| 档 | M | K | 关系 | 穷举 matching |
| --- | ---: | ---: | --- | --- |
| L2-A（现状） | 5 | 12 | 近独立匹配 | 约 1.7e5 部分单射 |
| **L3-A 实现入口** | **6** | **12** | 竞争 + SINR + 2 类协同 buoy | 不可逐步穷举最优 |
| L3-B 论文规格 | 6 | 16 | 更密干扰图、时变到达 | 必须消息传递 / 近似 |

L3-A 相对 L2-A 只加 **1 架 UAV**，避免把「变难」完全归因于规模。核心增量是 **边语义**，不是 K=20。

Episode T=48～64，与 L2-A 同量级，以便 hold=15 仍覆盖飞+服务。

---

## 3. 四种必须同时成立的关系

下列机制若只实现子集，仍可能被 MLP 用「最近高队列」解开。

### 3.1 UAV–UAV 竞争（competition edge）

设热门子集 \(\mathcal{H}_t=\{k: Q_k(t)\ge Q_{\mathrm{hot}}\}\)，设计上 \(|\{i:\eta_{ik}\le \eta_{\mathrm{comp}}\}|\ge 3\) 对至少一个 \(k\in\mathcal{H}_t\)。

合法约束改为 **显式竞争图**，不再「环境默默丢掉第二近的机」：

\[
c_{ij,k}(t)=1 \iff x_{ik}(t)=x_{jk}(t)=1,\ i\neq j
\]

单占 buoy：若 \(k\notin\mathcal{C}\)（非协同类），\(c_{ij,k}=1\) 则 **双方本步服务为 0**（冲突惩罚），而不是留下更近者。于是 π_H 必须在图上协调「谁放弃」。

竞争边特征（随时间变）：

\[
e^{\mathrm{comp}}_{ij}(t)=\big(\mathbf{1}[\exists k\in\mathcal{H}_t:\ \eta_{ik},\eta_{jk}\le\eta_{\mathrm{comp}}],\ \max_k \min(Q_k,Q_{\max}),\ \|p_i-p_j\|\big)
\]

**L2-A 差异：** 现在冲突被 `project_exclusive` 消掉，第二机变 idle，策略看不到竞争损失。

### 3.2 Interference coupling（SINR）

并发链路集合 \(\mathcal{L}(t)=\{(i,k):x_{ik}(t)=1,\ \mathrm{dist}(p_i,b_k)\le R_{\mathrm{svc}}\}\)。

\[
\mathrm{SINR}_{ik}(t)=\frac{G_{ik}P}{\sigma^2+\sum_{(j,\ell)\in\mathcal{L}(t),\ j\neq i} G_{j k}P\cdot \kappa_{\ell k}}
\]

服务进度：

\[
\rho_{i}(t+1)=\rho_i(t)+\frac{1}{D_k}\cdot \frac{\log_2(1+\mathrm{SINR}_{ik})}{\log_2(1+\mathrm{SINR}_0)}
\]

\(\kappa_{\ell k}\) 在同簇/邻频高、远簇低。这样 **i 选 k 会降低 j 在附近 buoy 的速率**，最优 z 是联合的。

**L2-A 差异：** 现为「邻机欧氏距离 ≤ R_int 则全体 ×1/n」，与配对的 buoy 无关，DeepSets 均值即可近似。

### 3.3 Cooperative service（UAV–UAV collaboration）

划分 buoy：

- **Solo** \(\mathcal{S}\)：至多 1 机；第二机冲突（3.1）。
- **Coop** \(\mathcal{C}\)：需要 \(n_k\in\{2,3\}\) 机 **同时在服务半径内且 option 指向该 k**，才推进 \(\rho^{\mathrm{coop}}_k\)。

\[
\rho^{\mathrm{coop}}_k(t+1)=\begin{cases}
\rho^{\mathrm{coop}}_k(t)+1/D_k^{\mathrm{coop}} & \text{if }\sum_i \mathbf{1}[x_{ik}=1\wedge \text{in-range}]\ge n_k\\
0 & \text{otherwise（拆散则清零）}
\end{cases}
\]

协同完成一块的 payload 显著高于单机块（例如 ×2），但拆散则前功尽弃。协作边：

\[
e^{\mathrm{collab}}_{ij,k}(t)=\big(\mathbf{1}[k\in\mathcal{C}],\ n_k-\#\{\text{已承诺 }k\},\ \eta_{ik}+\eta_{jk}\big)
\]

**L2-A 差异：** 禁止第二机，不存在必须成对 hold 的 option。

建议 L3-A：12 buoy 中 **2 个 coop**（各需 2 机）、**2 个高价值 solo 热门**（强竞争）、其余普通 solo。

### 3.4 Dynamic graph

每步重算边，禁止静态图缓存：

| 边类型 | 端点 | 时变特征 |
| --- | --- | --- |
| UAV–Buoy | \(i{-}k\) | distance, ETA, 瞬时 rate/SINR 分子, queue pressure \(Q_k/Q_{\max}\) |
| UAV–UAV competition | \(i{-}j\) | 是否争同一 \(\mathcal{H}_t\)、相对距离、热度 |
| UAV–UAV collaboration | \(i{-}j\) | 是否共瞄 coop buoy、还差几机、会合 ETA |
| Buoy–Buoy（可选、弱） | \(k{-}\ell\) | 同簇干扰 \(\kappa_{k\ell}\)、到达相关性 |

Queue pressure \(Q_k(t)\) 随服务与 \(A_k(t)\) 变；SINR 随 \(\mathcal{L}(t)\) 变。**图拓扑可以保留全连接二部 + 稀疏 UU 边（竞争/协同阈值）**，但特征必须逐步更新。

高层观测允许这些边；**不得**把未来到达轨迹 `remaining_arrivals()` 写进 π_H。到达预测仍用因果 persistence / 已发生的 \(A_k\)。

---

## 4. 高层动作与 HRL（保持不变的部分）

- **Hold=15**、低层 **冻结** Phase-6.2 控制器（若状态维变化，只做观测适配，不改 PPO 核心与已锁权重训练流程）。
- **高层 reward 仍用 Phase-6.4 的 `delta`（窗口 Δpayload）**，本阶段不改公式。
- z 仍为离散 option。L3 扩展语义，而不是改成连续 waypoint：
  - solo：buoy index 或 idle；
  - coop：同一 `k\in\mathcal{C}` 允许多机（最多 \(n_k\)）；超出 \(n_k\) 的多余机按竞争规则 **全灭或只保留约定的 n_k**（推荐：超出则该 buoy 本步协同失败，迫使图上精确计数）。

**不要**把 HGNN 接到低层 movement。低层仍是 `[state, z] → a`。

---

## 5. 验证协议（实现之后才跑；本期不训练）

保持 HRL，**只替换 high-level encoder**：

1. MLP（固定拼接）
2. DeepSets（节点集合，无 UU 边）
3. HGNN（二部 U–B + UU 竞争/协同边）

相同 seed、相同 updates、相同 hold=15、相同 `delta`。

| 指标 | 定义 | 图必要性解读 |
| --- | --- | --- |
| payload | 回合累计 bits | 最终任务收益 |
| option accuracy | 与 **L3 Oracle**（含 SINR+协同约束的影子搜索）逐步/hold 对齐率 | 是否选对关系约束下的 z |
| graph attention / entropy | HGNN 注意力或离散策略熵 | 是否在用边；是否过早塌缩 |
| decision conflict rate | 同步指向同一 solo 热门、或 coop 人数 ≠ \(n_k\) 的 UAV 对比例 | MLP 应显著更高 |

**先做非学习因果探针（对照 L2-A Phase 因果验证）：**

- 同态：两机同时改 z vs 只改一机，Δpayload 是否非加性。
- 无图 Oracle（独立贪心分数）vs 有图 Oracle（联合 SINR+协同）。
- 冲突率：独立贪心的 conflict rate 应 ≫ 联合 Oracle。

探针通过后再训三个编码器。 **禁止**先改 HGNN 宽度/层数来「救」L2-A。

---

## 6. 成功标准

若在 L3-A 上（同 HRL、同预算）：

\[
\text{HGNN payload} > \text{DeepSets} > \text{MLP}
\]

且 HGNN 的 option accuracy 更高、decision conflict rate 更低，则 **关系建模必要**，再进入「改进 HGNN / 论文模型」阶段。

若仍是 MLP ≥ DeepSets ≥ HGNN：

- 检查因果探针是否未过（环境仍可分解）；
- 再考虑训练预算/熵，而不是立刻加宽 HGNN；
- **重新判断 HGNN 必要性**（可能 HRL+hold 已够，图不是论文卖点）。

---

## 7. 实现边界（给下一阶段，本期不动工）

| 允许 | 禁止 |
| --- | --- |
| 新模块 `src/wt/l3a/` 类比 L2-A | 修改 `src/wt/l2a/` 锁定因果床 |
| 新物理：SINR、coop、冲突全灭 | 改 `round30/ppo.py`、低层权重、hold、`delta` |
| 新图特征供给高层 | 改 `src/wt/htw_phase7/` HGNN 实现来适配旧环境 |
| 因果探针脚本、设计报告 | 训练 MLP/DeepSets/HGNN（Phase-8 不做） |

Oracle 联合搜索：L3-A 上对 coop 子集枚举「哪 \(n_k\) 机去 coop」× 其余机对 solo 的贪心/匈牙利，禁止每步 6! 全枚举作为训练标签泄漏；评估可用短时域影子。

---

## 8. 结论

L2-A 让 HRL 成立，但高层仍是 **可独立打分的 matching 分类**，所以 MLP 赢、HGNN 输。Level-3 必须让 payload 依赖 **竞争边、SINR 交叉、协同计数、时变边**。先用因果探针证明环境需要图，再在 **不改 HGNN 代码的前提下** 只换环境重跑三编码器。本期不实现、不训练。

## 源文件 `HTW_Project_Cleanup_Audit_CN.md`

# HTW Project Cleanup Audit

本阶段只做文件清理审计。没有删除文件，没有修改代码逻辑、B2、τ、Gate-1、Candidate-C、planner 或 `decide()`，没有重新生成 scenario，也没有改实验状态。

**status = DEPLOYMENT_VERIFICATION_FAILED**

**deployment_verified = False**

**planner_ready = False**

未进入 Phase-4F，未进入 planner integration，未宣布 deployment PASS。

## 1. 文件总数量

扫描跳过 `.git`。清单共 **26525** 个文件，合计约 **4.78 GB**。

完整清单：`src/wt/lowlevel_v2/project_cleanup_audit/outputs/cleanup_file_inventory.csv`

每行包含路径、大小、UTC 修改时间、类型、是否被代码引用、是否被报告引用、是否属于实验 lineage，以及建议分类。

另有 **231** 个空目录。`backup/snapshots` 里的空目录保持不动。

## 2. 分类统计

| 分类 | 数量 | 说明 |
| --- | --- | --- |
| A. 必须保留 | 24729 | 代码、模型、registry、manifest、审计报告、实验结果、round 输出、snapshot |
| B. 可以归档 | 19 | 与仓库根目录报告字节对应的 Phase-4 markdown 副本 |
| C. 删除候选 | 1777 | snapshot 之外的 `.pyc`，约 30.1 MB |

删除候选按目录：

| 位置 | 文件数 |
| --- | --- |
| `src` | 906 |
| `round_36` | 735 |
| `tests` | 122 |
| `experiments` | 14 |

`backup/` 中的删除候选数量为 0。

## 3. 删除候选

1777 个文件全部是 Python 字节码缓存，不在 `backup/snapshots` 中。它们不是 B2 标签、holdout、criterion 或 verification 的唯一记录。

建议只在人工确认后删除。本阶段没有执行删除。

空目录里，`.test-tmp/` 和 `round_02/t*` 看起来是测试临时目录。它们也等待确认，没有删除。snapshot 内的空目录不在此列。

## 4. 必须保留的关键文件

- `src/wt/lowlevel_v2/outputs/phase0/LLV2_DATA_REGISTRY.json`
- `src/wt/lowlevel_v2/phase4d_deployment_criterion_registry/deployment_criterion_registry.json`
- `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/deployment_holdout_manifest.json`
- `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/holdout_labels.csv`
- `src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/holdout_integrity_audit.csv`
- `src/wt/lowlevel_v2/phase4e_deployment_verification/outputs/deployment_verification_results.csv`
- `src/wt/lowlevel_v2/phase4e_deployment_verification/outputs/deployment_failure_attribution.json`
- `src/wt/lowlevel_v2/htw_phase4a_hindsight_credit/outputs/phase4a_hindsight_credit.csv`
- `src/wt/lowlevel_v2/htw_phase4b_b2_shadow/outputs/b2_shadow_decisions.csv`
- 仓库根目录的 `HTW_Phase4C`、`HTW_Phase4D`、`HTW_Phase4E` 报告
- `backup/snapshots/` 下的全部 snapshot

Phase-4C / 4D / 4E 的 registry、manifest、integrity audit、verification 和 failure attribution 都在保留集合里。

## 5. 风险检查

| 检查 | 结果 |
| --- | --- |
| 是否已删除文件 | 否 |
| 删除候选是否包含 snapshot | 否 |
| 删除候选是否包含唯一实验结果 | 否。候选只有 `.pyc` |
| Phase-4E audit、deployment verification、B2 标签、scenario manifest 是否仍在 | 是 |
| `deployment_verified` | False |
| `planner_ready` | False |
| status | DEPLOYMENT_VERIFICATION_FAILED |

19 份模块内 Phase-4 报告与根目录同名报告重复。归档它们的风险是中等：根目录副本还在，但模块输出目录里的副本也常被阶段脚本写回。因此只列入归档建议，不删除。

字节相同的 round 输出和 snapshot 最初被标成重复文件。它们仍是对应轮次的记录，已改回保留。

## 6. 删除建议

可以在确认后删除 snapshot 之外的 1777 个 `.pyc`。

不要删除或改写 `backup/snapshots`。不要删除 holdout、criterion、B2 标签、verification 或 failure attribution。19 份重复 Phase-4 报告如需归档，先保留根目录副本，并单独确认。

删除计划：`cleanup_deletion_plan.md`。等待人工确认。本阶段没有自动删除高风险文件。

## 源文件 `level2_causal_validation_report.md`

# Level-2-A Causal Validation Report

独立 L2-A 物理验证床：**M=5，K=12**。未训练、未改 PPO 核心、未加 HGNN、未训练 Task-Waypoint、未开 Round、未改锁定 Level-1 / Round-19/30/35 环境。

本阶段目标不是提高学习性能，而是证明该 benchmark **开始需要** 长期任务规划 / HRL（以及后续 HTW/HGNN 的结构空间）。

「State-only」在此定义为非学习代理：**当前队列+距离上的最近正向队列 matching**，再沿该关联飞向 buoy。它对应 L1 中「状态已足够解 assignment」的机制，而不是 PPO 训练曲线。

## 1. Level-2 环境定义

代码：`src/wt/l2a/`。与锁定海事 L1 隔离。

- 规模：M=5，K=12，T=48。合法部分单射关联数 **169021**（L1 的 M=2,K=5 仅为 31）。
- 几何：近簇 6 buoy（x≈18–42 m），远簇 6 buoy（x≈118–142 m）；UAV 从近簇出发。
- 运动：v_max Δt = 22.0 m，簇间约 100 m ⇒ **飞行 ≥4 slot**（空间延迟）。
- 服务：D_k=3；仅关联且进入服务半径 16 m 才累积进度 ρ；**切换则 ρ←0**。
- 队列：Q_k(t+1)=clip(max(Q_k(t)-S_k(t),0)+A_k(t), 0, 4)。近簇仅有初始 Q=3；远簇在 t=4…36 的偶数步 A_k=1。溢出计入 dropped。
- 完成一块（ρ=1）发放 1e6 bits payload。附近并发服务有干扰缩放 1/n。
- 关联：控制器给出 x_mk 请求，环境投影为排他匹配（每 buoy 至多一机，每机至多一 buoy）。

## 2. 新增状态变量

写入 `info()`、**不含未来到达轨迹**：

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| `uav_positions_m` | (5,2) | UAV 位置 |
| `buoy_positions_m` | (12,2) | buoy 位置 |
| `queue_blocks` | (12,) | Q_k(t) |
| `service_progress` | (5,) | 当前占用进度 ρ |
| `assignment` | (5,) | 当前 x 的 UAV→buoy，idle=-1 |
| `eta_slots` | (5,12) | 以当前步长计的飞行时间 |
| `cumulative_payload_bits` | scalar | 累计 payload |

`remaining_arrivals()` 只给 Oracle 脚本，不进入观测。

## 3. 新增动作变量

| 动作 | 形状 | 含义 |
| --- | --- | --- |
| movement | (5,2) | 单位方向位移，范数截断到 1 |
| association | (5,) | 请求的 buoy index 或 idle；经 `project_exclusive` |

因果实验中运动固定为「飞向当前关联目标」，只对照 **关联/option**。

## 4. 长期耦合证明

- 同一初态、整段 hold 两种关联：平均 \|Δpayload\| = 1875000 bits。不同 buoy 选择 **改变未来 payload：是**。
- Myopic 与 Oracle option 的逐步 assignment 一致率 = 0.648（L1 上 PPO 曾达到约 0.95 的 option 完成率）。
- 当前贪心是否仍「接近 100% 解 assignment」：**否**。
- 机制来源：远簇到达 + 多 slot 飞行 + D≥3 占用 + 排他 x_mk。贪心先清近簇，远簇队列溢出。

## 5. Oracle Option vs State-only

Oracle：在 t=0 对簇间机数 n_far∈{0..5} 做 **影子全轨迹** 搜索，然后 **hold 该簇 option**，簇内再最近队列排他。影子可用已知 A_k 过程；观测向量仍无未来 A。

| 方法 | mean payload bits | mean dropped blocks |
| --- | ---: | ---: |
| Myopic nearest-queue（State-only 代理） | 17000000 | 77.6 |
| Oracle cluster option | 28750000 | 67.5 |
| 差（Oracle − Myopic） | 11750000 | -10.1 |

**Oracle Option > State-only：成立**

种子 0–7：近簇出发位置有小抖动。未训练 PPO。

## 6. 对 HTW / HRL / HGNN 的含义

- **HRL**：存在可 hold 的 buoy/簇 option，且优于当前状态贪心 ⇒ 需要高层任务分配。
- **HTW**：L1 失败是因为无任务分解间隙；L2-A 已出现该间隙，后续才值得把 option 做成可学习 TW。
- **HGNN**：本阶段未加入。图关联在 M=5,K=12（约 1.7e5 合法匹配）上有结构空间，但 **尚未证明必须用 HGNN**；先锁定因果床。

数值：`src/wt/l2a/outputs/causal/summary.json`

## 源文件 `planner_audit_report.md`

# Planner Audit Report

范围：Round-19 `src/wt/round19/planner.py`，以及 Phase-1/3 用来生成 TW 的 `independent_queue_rate`。未改环境、reward、PPO、HGNN。

## 1. 当前 planner 优化目标

`independent_queue_rate_action` **不优化**环境 reward，也 **不优化**折扣累计通信收益。

对每个 UAV 独立取

`k* = argmax_k (queue_k / capacity) * rate_{uav,k}`

其中 `rate` 来自 `rate_matrix`（单 UAV 静态大尺度信道，**不含**共享频谱干扰与联合 matching）。动作是指向 `k*` 的单位向量。

`joint_plan` 才对 **预测 payload 之和** 做有限 beam search，并在 shadow 里用 `joint_aware` matching。Phase-1/3 默认 TW **没有**用它。

## 2. 输入状态

仅当前可观测量：

- UAV 位置
- buoy 位置
- 当前队列 `raw_queue_bits`
- 由位置算出的当前速率/距离

不输入：其他 UAV 的未来意图、未来到达、累计 return、干扰后的 SINR（独立策略路径上）。

## 3. 输出 TW 的方式

滚动该启发式 H=5 步（shadow，不改 live env）：

- `goal` = 窗口内各 UAV 对 buoy 的累计 payload 份额
- `waypoint` = 第 H 步位置（归一化）
- `horizon` = 5

这是 **myopic 方向的积分终点**，不是最优控制子目标。

## 4. 为什么不能提供长期价值

1. 目标是瞬时 queue×正交速率，不是 `sum_t r_comm`。
2. 忽略干扰：两机可能冲向同一高队列 buoy，共享频谱下互相压速率。
3. H 步滚动仍每步 myopic，终点不必是累计 payload 最优位置。
4. 这些量（队列、位置、近似速率）**已经在低层 state 里**。TW_A 近似是 state 的确定性函数，所以 Phase-3 条件化没有新信息。

Phase-4 用同一 TW 槽位接入：

- TW_A：当前 queue-rate
- TW_B：`joint_plan(H=1)`，最大化一步干扰感知 payload
- TW_C：一次 `joint_plan` 长度 H=5 的 beam 轨迹终点（累计预测 payload），不是对 H 次独立 receding 搜索。

## 源文件 `PLANS.md`

# Project Round Plan

## Round-31 (completed and locked)

Round-31 is the expressly authorized inference-only diagnosis of the locked
Round-30 state-only DeepSets-PPO result. It re-audits the implemented PPO
semantics and analyzes frozen histories/checkpoints plus current-state
train/validation diagnostics for optimization health, advantage and credit
quality, observation sufficiency, DeepSets pooling aliasing, action geometry,
and K/eta-specific failure mechanisms. It does not train, tune, mutate the
environment/reward/matching/action contract, rerun fresh, or access untouched
evaluation. Frozen Round-30 fresh outputs are read only as existing evidence.
The locked result is R31-D: implementation, numerical instability,
advantage/credit attenuation, relational aliasing, and distribution shift do
not establish a single minimal fix. PPO is already below Independent on both
registered training and validation scenarios, while K5 task scale is only
moderate evidence. The recommendation is to stop the current direct PPO
baseline branch. It generated no fresh rollout or untouched access, passed
986/986 regression tests and compileall, and did not create Round-32.

## Round-30 (completed and locked)

Round-30 is the expressly authorized Direct Multi-UAV State-Only
DeepSets-PPO Learnability Baseline. It starts from a verified locked Round-29
and tests only whether direct, on-policy, current-observable RL can stably
outperform Random Valid, Independent Myopic, and the frozen state-only BC
policy in the existing Round-19 M=2 shared-spectrum environment. The physical
environment, reward, action interface, and all completed scientific artifacts
remain frozen. Training, validation, and one-access fresh scenario sets are
new and disjoint; historical protected and untouched sets remain inaccessible.
No Task-Waypoint, planner-trigger, planner target, graph/Transformer model,
offline BC warm start, reward shaping, or Round-31 was created. Five fixed
307,200-step PPO seeds completed, validation selected their frozen checkpoints,
and the pre-frozen fresh set was accessed once. The result is PPO-LC and PBC-B:
the paired validation and fresh PPO-minus-Independent hierarchical intervals
are below zero, so this round does not establish direct RL learnability. The
evidence is locked at `round_30/ROUND_LOCK.json`; no successor is authorized.

## Round-29 (completed and locked)

Round-29 is the expressly authorized diagnostic-only Cross-Scale Deferred-Service
Pressure Normalization and Independent Confirmation Audit. It tests whether the
Round-28 K=3/K=5 DSP reversal is a one-buoy-capacity scaling artifact or a
physical task-scale mechanism. Exactly three pre-registered current-observable
normalizations retain the frozen .5/.5 spatial combination and use one
train-P75 threshold each. DSP-V3 was selected using only the Round-28 train
replay rows, then assessed exactly once on the new, disjoint, K-balanced
40-scenario confirmation set (160 states, 320 H5-versus-H1 branches). The
locked CS-C / R29-C result does not confirm a cross-scale trigger and does not
authorize trigger-model training; its recommendation is to stop the current
planner-trigger representation route. No model, policy, trigger, checkpoint,
environment, reward, planner, protected dataset, or Round-30 was created or
modified. Evidence is locked at `round_29/ROUND_LOCK.json`.

## Round-28 (completed and locked)

Round-28 is the expressly authorized, diagnostic-only audit of whether direct
same-snapshot H5-versus-H1 planner utility has stable current-state physical
representations. It uses exactly the stored Round-20 Joint-H1 train/validation
states at timesteps 0, 2, 4, and 6; fixed current-only features, train-only
normalization, equal-weight composites, train-derived thresholds, and
scenario-cluster statistics. The locked REP-B / R28-B evidence identifies
some useful representation signal, but not causal planning-need separability
or readiness for trigger-model training. No policy or trigger model was
trained, no scientific artifact or protected data changed, no fresh rollout
ran, and Round-29 was not created. Evidence is locked at
`round_28/ROUND_LOCK.json`.

## Round-27 (completed and locked)

Round-27 is the explicitly authorized, diagnostic-only audit of direct
same-snapshot H=5 planner-action communication utility relative to frozen H=1,
and of pre-registered current-physical-state trigger rules. Its locked result
is CPA-B / TR-B / R27-B: useful H=5 first actions occur (13/64), and the
oracle has validation payload headroom, but the train-selected duplicate-intent
trigger is not validation-stable. It used the fixed 64-scenario Round-20
train/validation t=0 state pool, fixed deterministic H1 continuation, no
future-derived trigger input, and a train-only threshold. No training,
scientific-artifact change, protected-data access, fresh rollout, or Round-28
was made. Evidence is locked at `round_27/ROUND_LOCK.json`.

## Round-26 (completed and locked)

Round-26 is the explicitly authorized, audit-only study of mixed-behavior
executed joint-action ambiguity, whether true or frozen-predicted
Task-Waypoints reduce local action variance, and whether the frozen causal H5
planner supplies a coherent action-distillation target. It may use only
Round-20 train/validation data, frozen inference, and current-state causal
planner calls. No training, data/environment/planner modification, fresh
rollout, development-test/untouched access, or Round-27 creation is allowed.
The locked audit yields AS-B / TD-C / PAD-B and R26-D: executed actions are
locally heterogeneous and often multimodal, but frozen true or predicted
Task-Waypoints do not materially reduce that conditional variance, while the
causal H5 planner is deterministic and replays H10 exactly but lacks the
required payload-utility relation. This is a design-only conditional
planner-action distillation readiness result, not a training authorization.
Evidence is locked at `round_26/ROUND_LOCK.json`.

## Round-25 (completed and locked)

Round-25 is the explicitly authorized, inference-and-audit-only investigation
of whether mixed Round-20 behavior trajectories supply conflicting H=5
Task-Waypoint targets for the same observable state, and whether frozen
Round-19 causal planner targets are better aligned with high-quality behavior
than the unconditional mixed target. It reads only train/validation raw data,
runs frozen predictor/policy/planner inference, and must not train, alter data
or environment semantics, read development-test or untouched data, re-run any
fresh set, or create Round-26. The locked evidence yields SA-TW-C and PT-B:
the pre-registered evidence does not establish material overall supervision
ambiguity or a general planner-derived target candidate. The registered
recommendation is to stop the current overall Task-Waypoint policy-advantage
claim; no retraining was performed. Evidence is locked at
`round_25/ROUND_LOCK.json`.

## Round-24 (completed and locked)

Round-24 completed the explicitly authorized, pre-registered seed extension
of the locked Round-22 multi-UAV Task-Waypoint policy protocol. It trained
exactly seeds 3--7 for the predicted Task-Waypoint Transformer, matched
State-only Transformer, and predicted Task-Waypoint MLP (15/15 runs), while preserving
the H=5 DeepSets predictor, Hc=10 policy protocol, Round-20 train/validation
splits, environment, and all old seeds 0--2 byte-for-byte. The new
80-configuration `round24_seed_stability_dev_v1` set is pre-registered before
training and was evaluated exactly once only after checkpoints and analysis
were frozen. Round-22 fresh development was not rerun; Round-20
development-test reuse and untouched evaluation access remain zero. The
result is MS-C / R24-C: 4/8 fresh seeds were positive, but the hierarchical
interval crosses zero and tail sensitivity prevents an overall reproducible
Task-Waypoint advantage claim. No seed 8+, architecture/loss/optimizer
modification, HGNN, or PPO/MARL was made. At its lock time Round-25 had not
been created. Evidence is locked at `round_24/ROUND_LOCK.json`.

## Round-23 (completed and locked)

Round-23 is an inference-only audit of the already locked Round-22 frozen
policy checkpoints, Round-20 train/validation development data, and the
already saved Round-22 fresh-development raw outputs. It explains seed-level
closed-loop inconsistency, Task-Waypoint head causal utilization, and the
alignment of offline action error with payload. No model, optimizer, backward
pass, checkpoint, sampler, architecture, environment, reward, matching,
projection, fresh-development rollout, or untouched multi-UAV evaluation is
permitted. It may run new paired validation-only inference diagnostics with
the frozen policies and fixed validation snapshots. It must stop after its own
lock and must not create Round-24.

## Round-22 (completed and locked)

The explicitly authorized Round-22 study freezes the Round-21 validation-only
DeepSets H=5 Task-Waypoint predictor and trains exactly twelve offline
multi-UAV movement policies: predicted Task-Waypoint Transformer, matched
State-only Transformer, Task-Waypoint MLP, and offline-only Oracle Transformer,
over seeds 0/1/2.  It uses a shared-parameter edge-free DeepSets state encoder,
Hc=10 causal state histories without action tokens, validation-only action-RMSE
selection, and the unchanged Round-19/20 safety/matching/shared-spectrum
environment.  Round-20 train/validation remain the only model data; the former
development test is not reused and Round-20 untouched evaluation remains
absolutely prohibited.  A 60-scenario disjoint fresh policy-development set is
pre-registered before training and may be accessed once only after a V-A/V-B
validation gate.  No HGNN, graph Transformer, PPO/MAPPO/MARL, predictor
training, or resource-action policy is permitted.  The round stops after its
own lock and does not create Round-23.

Current-status correction: the completed Round-22 formal matrix contains
12/12 runs, passed 843/843 regression tests and compileall, and has final gate
MU-WP-B. The fresh development set was accessed exactly once after V-B; the
Round-20 development test was not reused, untouched multi-UAV evaluation
remains unaccessed, and `candidate_ready_for_untouched_eval=false`. Evidence
is locked at `round_22/ROUND_LOCK.json`; Round-23 has not been created.

## Round-21 (completed and locked)

The explicitly authorized supervised learning follow-on to locked Round-20
trained only a fixed two-layer heterogeneous spatial graph encoder and
Task-Waypoint predictor, comparing it with DeepSets at H=5/H=10 over three
seeds.  Validation-only selection is followed by the fixed selected-horizon
no-UU and no-interference-feature ablations.  The Round-20 datasets, labels,
environment and protected untouched evaluation set remain read-only; exactly
one guarded development-test access is reserved until after selection is
frozen.  No action policy, Transformer, PPO/MAPPO/MARL, resource policy, or
closed-loop neural evaluation occurred.  All 18 fixed runs, the one guarded
development-test access, 802 regression tests and compileall passed; Round-20
had 86/86 registered hashes before and after.  The H=5 selection yielded
TWG-A but GV-C/UU-C/IF-C, so policy-stage readiness is false. Evidence is
locked at `round_21/ROUND_LOCK.json`; no Round-22 exists.

## Round-20 (completed and locked)

The explicitly authorized follow-on to the locked Round-19 IPN-A result.  It
constructs a fixed 80-scenario, four-tier M=2 shared-spectrum development
dataset with deterministic mixed-quality behavior sources, scenario-level
60/20/20 isolation, graph-ready current-observable state, and H=5/H=10
multi-UAV Task-Waypoint supervision.  The Round-19 physical environment,
matching, safety projection, and frozen planners are reused without changes.
No neural training, HGNN, PPO/MARL, protected historical-set access, or
rollout from the newly pre-registered untouched-evaluation set is allowed.
Round-21 was subsequently opened by an explicit instruction; this locked
Round-20 description remains historical evidence.
The completed formal evidence contains 80 scenarios, 800 episodes, 9,600
transitions, 6,400 H5 and 2,400 H10 labels, 20 PNG + 20 SVG figures, and a
759/759 final regression. All three data-readiness gates are true. Evidence is
locked at `round_20/ROUND_LOCK.json`.

## Round-19 (completed and locked)

Shared-spectrum M=2 buoy-uplink interference coordination/planning
revalidation.  It begins from the verified immutable Round-18 lock and adds
only cross-buoy uplink interference under frozen eta={0.25,0.50,1.00}, with an
exact orthogonal control.  It is deterministic and non-learning: no HGNN,
PPO, MARL, power control, backhaul, protected data, or Round-20 work is
permitted until its registered gates and lock are complete. The exact
orthogonal and M=1 controls passed. ICN-B did not establish immediate
coordination, while the low shared tier established IPN-A (H10-H1 0.322 Mbit,
95% CI [0.075, 0.600]), realistic planning headroom, and
`multi_uav_shared_spectrum_offline_dataset_ready=true`. The coupling-v2 gate
remains MCI-C because immediate coordination is not stable. Evidence is locked
at `round_19/ROUND_LOCK.json`; Round-20 was later opened by explicit user
authorization.

## Round-17 (completed and locked)

Predictable time-varying channel planning revalidation completed with a
current-observable slow multiplier and static buoy drift in every primary
scenario. The 48-row static control was exact and the 576-episode formal audit
returned CPN-C: no established realistic causal long-horizon planning headroom.
The evidence is locked in `round_17/ROUND_LOCK.json`.

## Round-18 (completed and locked)

Minimal multi-UAV cooperative maritime-environment and long-horizon
coordination-necessity audit. This non-learning round adds only an M=2 primary
task coupling: a shared buoy queue, exclusive deterministic matching, and
hard UAV-UAV separation. Both buoy drift and channel variation are static in
primary experiments. The 42-scenario/168-episode audit returned CN-B, MPN-C,
and MC-C; it does not authorize training or Round-19. Evidence is locked in
`round_18/ROUND_LOCK.json`.

## Round-15 (completed and locked)

Long-Horizon Planning Necessity and Hierarchical-Control Headroom Audit.
This deterministic, non-learning round compares current-state myopic rules
against a pre-registered finite action-lattice lookahead at H={1,5,10,20} on
new Round-15 diagnostic scenarios.  It separates a causal current-arrival
persistence planner from a non-deployable oracle-trace upper bound. A
metadata-only pre-initialization holdout-report read is recorded as one access
and excluded from all computation. The paired actual-payload result is PN-C:
long-horizon planning was not required under this audit. The locked output is
`round_15/outputs/round15_planning_headroom_audit_20260825_1245`; no further
protected data may be opened and no Round-16 work is authorized.
The round stops at PN-A/PN-B/PN-C and must not create Round-16.

## Round-14 (completed and locked)

Task-Waypoint-to-Motion Mapping Redesign and Actionable Intermediate Guidance
Audit.  The scope is a deterministic, inference-only mapping from the frozen
H=10 task waypoint `[service, gross progress, H-step return]` plus current
valid geometry/rates to the existing two-dimensional movement action.  It
will measure teacher alignment on permitted train/validation data, assess
future-outcome relevance as an offline diagnostic, and run paired,
same-snapshot movement counterfactuals with a fixed `queue_rate_energy`
continuation.  No model/policy training, Spatial Waypoint work, service action
extension, environment/reward/resource modification, old-development-test,
or untouched-holdout access is allowed.  The round stopped at MAP-B: C had
the strongest validation teacher-action cosine (0.332), but its paired payload
interval against `queue_rate_energy` crossed zero.  It is therefore
diagnostic-only; no future policy round was created.  Evidence is locked at
`round_14/ROUND_LOCK.json`.

## Round-13 (completed and locked)

Evaluation-only Task-Waypoint action-controllability audit and minimal
service-intent action design. No neural model, policy, PPO, HGNN, or other
optimizer may be trained. The audit is limited to train/validation development
scenarios plus, if needed, explicitly marked diagnostic-only scenarios; old
development test and untouched holdout remain prohibited. Its goal is to
measure whether the three Task-Waypoint heads are controllable by the existing
two-dimensional movement action and whether a physically bounded explicit
service-intent action has diagnostic headroom. It must stop after its own
acceptance gate and must not create Round-14.

Current-status update: Round-13 is completed and locked at
`round_13/ROUND_LOCK.json`.  Its non-destructive completion amendment is
tracked separately by `round_13/ROUND_LOCK_AMENDMENT.json`; it does not alter
the original formal result or create Round-14.

## Round-7 (completed)

Gate-B diagnosis and `maritime_offline_dataset_v2` construction completed
without model training. The output has 60 scenario configurations, 720 legal
episodes, outcome-derived 40/40/20 high/medium/low quality tiers, seed-level
development/holdout isolation, explicit stitching/goal/horizon audits, and
H=10/H=20 retraining candidates. See
`report/round_07_report_20260809_1810.md`. No Round-8 was created.

## round_01 — completed and locked

Minimal single-UAV, static multi-buoy maritime communication environment:
configuration validation, deterministic large-scale channel, normalized state,
continuous motion, boundary handling, placeholder energy, reward decomposition,
smoke rollout, and 45 environment tests. Full pre-round-02 regression: 78/78.

## round_02 — completed and locked

Completed:

- four reproducible non-learning baselines;
- a shared evaluation and collection runner;
- semantically structured episode trajectory storage;
- episode-level train/validation/test splitting;
- K-step availability and leakage audit;
- baseline metrics, data-quality reports, and research figures;
- small, bounded smoke collection over multiple seeds and buoy counts.

Explicitly excluded: queue/environment redesign, offline RL training, Waypoint
Network, Transformer, Decision Transformer, behavior cloning, optimizers, and
all other learning models.

Acceptance evidence:

- 100/100 repository tests passed;
- canonical smoke run: 24 episodes and 192 transitions;
- train/validation/test each contain 8 episodes with no group leakage;
- exact reload, checksum, continuity, finite-value, and K-step audits passed;
- six figures were generated and visually checked;
- `training_performed=false`.

The smoke data validates the pipeline and `K=1,5` sample availability. It is
not sufficient for substantive Waypoint sample construction because eight-step
episodes yield no `K=10,20,30` samples and the current environment has no
desired goal, queue, urgency, task completion, or actual payload.

At its lock time no next round had been created. Round 03 was later opened by
an explicit user instruction.

## round_03 — completed and locked

Completed:

- opt-in task-semantics v1 with real per-buoy queues, reproducible arrivals,
  queue-bounded payload, task completion, desired/achieved goals, terminal
  reasons, and truthful reward components;
- four behaviorally distinct non-learning baselines:
  `random_valid`, `nearest_buoy`, `max_queue`, and `queue_rate_energy`;
- shared materialized arrival traces for fair cross-policy comparison;
- checksum-protected schema-v1 trajectories and deterministic
  train/validation/test grouping;
- padded and masked Waypoint samples for `K={1,5,10,20,30}`;
- 22-item data-quality/readiness audit and 12 visually checked figures.

Acceptance evidence:

- 146/146 repository tests and 46/46 focused Round-3 tests passed;
- canonical smoke: 24 episodes, 192 transitions;
- canonical pilot: 80 episodes, 6,604 transitions, 28,140 Waypoint samples;
- split episodes/transitions:
  train 48/3,910, validation 16/1,279, test 16/1,415;
- valid K samples: 6,604 / 6,284 / 5,884 / 5,084 / 4,284;
- exact reload/checksums/source-target recomputation passed;
- all 22 quality checks passed, with 55 high-quality and 25 mid-quality
  episodes, no duplicate trajectories, and no boundary or integrity errors;
- `training_performed=false` and no learning model was constructed.

Round-01 and Round-02 locks and generated artifacts remain unchanged. No next
round has been created; work stops pending explicit human authorization.

## round_04 — completed and locked

Completed:

- read-only, checksum-bound adaptation of the locked Round-3 pilot without
  regenerating or re-splitting it;
- two-dimensional physical relative-displacement targets for
  `H={1,5,10,20,30}`;
- 15 goal-conditioned and 15 matched state-only spatial MLP training runs for
  seeds `{0,1,2}`;
- zero, train-mean, and causal constant-velocity non-learning baselines;
- validation-only checkpoint and horizon selection;
- grouped raw/projected offline evaluation, strict provenance, resumable
  checkpoints, CSV metrics, and scientific figures.

Acceptance evidence:

- 56/56 focused and 202/202 repository tests passed; compileall passed;
- final canonical formal run:
  `round04_formal_canonical_20260802_173717`;
- 30/30 training runs completed and 60/60 best/last checkpoints strictly
  reloaded with current source/config/tensor provenance;
- validation selected H=30 without test data;
- selected-H test RMSE is 44.281 m for goal-conditioned, 49.006 m for
  state-only, and 52.564 m for zero displacement;
- 72 CSV files/10,415 rows contained no NaN/Inf;
- all 36 PNG/SVG figures passed visual and semantic QA;
- Round-3 lock, data, implementation, outputs, and figures remained unchanged.

Explicitly excluded and not performed: Transformer or Decision Transformer
training, action-policy training, behavior cloning, reinforcement learning,
closed-loop evaluation, HGNN, multiple UAVs, moving buoys, and
communication-resource actions.

The result demonstrates conditional offline supervised learnability, not
policy improvement. Hindsight-goal availability, policy-dependent errors,
split shift, correlated samples, and absence of low-quality trajectories
remain material limitations. No next round has been created; later work
requires explicit human authorization.

## round_05 — completed and locked

Completed:

- froze the validation-selected Round-4 H=30 spatial Waypoint Network and
  materialized provenance-bound predicted-waypoint caches for all locked splits;
- trained the exact 27-run causal two-dimensional offline policy matrix with
  no historical action tokens: WT, Direct-Goal, State-only, Oracle diagnostic,
  and two one-step MLP controls;
- selected WT `H_c=10`, seed 0, epoch 7 strictly on validation RMSE, then
  evaluated all 27 frozen selections on test exactly once;
- added policy-balanced sampling, raw/projected action metrics, grouped
  evaluation, recorded-state replay diagnostics, and a five-step disposable
  environment interface smoke;
- fixed float32 boundary-validity accounting and seeded model construction
  before parameter initialization; independent seed-reproducibility smoke runs
  produced bitwise-identical model state, history, and predictions.

Acceptance evidence:

- 247/247 repository tests and compileall passed after the final seed fix;
- formal canonical output: `round05_formal_canonical_20260803_1500`;
- 27/27 training runs and 27/27 test-once evaluations completed; 54/54
  best/last checkpoints carry optimizer, sampler, RNG, resume, and provenance;
- 27/27 best checkpoints reloaded with identical predictions; no CSV/JSON/NPZ
  NaN/Inf; target and projected action validity are 1.0;
- 16 PNG plus 16 SVG figures passed strict semantic and visual QA;
- Round-3 data/provenance and the Round-4 lock/checkpoint remained unchanged.

The selected WT test RMSE is 7.772 m. Direct-Goal, State-only, and Oracle
Hc=10 controls have closely competing test means (7.949, 7.975, and 7.905 m),
so this round does not establish a stable Waypoint advantage. It establishes
only offline hindsight-conditioned action prediction and a bounded interface.

Explicitly excluded and not performed: formal closed-loop return ranking, PPO
or online fine-tuning, Round-4 retraining, HGNN, multi-UAV, moving-buoy, and
communication-resource action extensions. No next round has been created.
## Round-6 (completed)

Frozen-model, paired reference-goal closed-loop evaluation completed on 60
new scenarios. Gate B: no reliable WT advantage over Direct-Goal; stop before
any HGNN/multi-UAV/PPO expansion. See `report/round_06_report_20260809_1600.md`.

Round-23 completed the frozen checkpoint audit with no training, no new fresh-development rollout, and no untouched multi-UAV evaluation. Evidence is locked at `round_23/ROUND_LOCK.json`; Round-24 has not been created.

## Round-32 (in progress)

Round-32 is an authorized read-only cross-round audit of episode, planner,
Task-Waypoint, and PPO effective-future semantics. It follows the byte-stable
Round-31 lock from `backup/snapshots/before_round_32_20260830_2231`. No model,
environment, planner, dataset, rollout, fresh evaluation, or protected
development-test record may be created or changed; work stops after Round-32.

Round-32 completed and locked with EH-C / R32-C. It corrects the conflicting
five-step premise to the locked twelve-step fact, preserves historical gates,
and recommends only a future benchmark-design interval of 11--20 slots. No
Round-33 was created.

## Round-33 (completed and locked)

Round-33 follows the immutable Round-32 lock after explicit authorization. It
created only a new T=20, M=2 physical benchmark and a non-learning H10-vs-H5
planner-separation validation. The sole planner exception was H=20 membership
in the existing joint finite-beam search; it remained descriptive only, while
H1/H5/H10 passed exact T=12 compatibility regression. The 64-scenario design
and 64-scenario confirmation populations produced HS-C and BENCH-C: H10-vs-H5
first-action and same-snapshot utility differences are both exactly zero, so
Task-Waypoint re-entry remains false. The governance compatibility repair
preserved the immutable Round-32 lock, passed 1040/1040 repository tests and
compileall, and Round-33 is locked. Its original lock remains immutable;
Round-34 proceeds only through a separately recorded successor amendment.

## Round-34 (completed and locked)

Round-34 is a no-training, no-environment-execution audit of benchmark
complexity, relational-decision ownership, and method suitability. A single
accidental `development_test` preflight decode is recorded in
`round_34/ROUND34_PROTECTED_ACCESS_EXCEPTION.json`; it is excluded from every
scientific output. Formal analysis restarts only from hash-locked clean
train/validation manifests and must stop on any second protected decode. The
audit returned BCX-B / R34-B and is locked after 1,087 regression tests and
compileall passed. It may recommend a future relational benchmark design but
cannot implement it, train any method, or create Round-35.

## Round-35 (in progress)

Round-35 is an explicitly authorized, append-only successor to the immutable
Round-34 lock. It implements only the Level-2 M=4/K=8/T=20 relational
shared-spectrum environment and an external association-preference action
under exact feasibility decoding. The historical Round-34 prohibition remains
its lock-time fact; the successor is authorized solely by
`round_34/ROUND_LOCK_AMENDMENT.json`. No model training or inference,
protected-data access, reward/physics redesign, planner work, Level-3, or
Round-36 is allowed.

Round-35 completed and locked with RAO-A / L2-A. It passed 38 dedicated tests,
1,125/1,125 full regression tests, and compileall. Its external association
preference causally controls exact feasible assignment, while Task-Waypoint
re-entry remains false and no HGNN performance claim is made. No Round-36 has
been created.

## Round-36 (in progress)

Round-36 is the explicitly authorized, append-only successor of the immutable
Round-35 lock, recorded in `round_35/ROUND_LOCK_AMENDMENT.json`. It is a
non-learning Level-2 M=4/K=8/T=20 exact first-action association audit with
64 frozen design and 64 frozen confirmation scenarios, anchors {0,8}, and all
3,393 legal assignments. Formal scientific execution uses exactly four workers;
candidate pruning, approximation, protected or untouched access, model
inference, training, and Round-37 are prohibited.
