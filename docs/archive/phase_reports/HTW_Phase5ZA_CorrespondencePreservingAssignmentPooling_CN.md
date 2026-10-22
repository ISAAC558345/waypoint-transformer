# HTW Phase-5ZA Correspondence-Preserving Assignment Pooling

**planner_ready = False**

**deployment_verified = False**

Structural gate 没有通过。按预先写定的规则，训练没有开始，310110–310117 没有用来评估性能，310118–310125 没有打开。

## Q1

旧 pooling 先把 `(H_UAV_i, H_Buoy_{a_i}, H_edge_{i,a_i})` 拼成 triple，再对选中 UAV 做 mean。buoy 段的和只取决于选中了哪些浮标，不取决于哪一架 UAV 连到哪一个浮标。UAV 段同样只保留选中的 UAV 集合。`H_UU` 本身不含目标。edge 的和还带对应关系，但 Phase-5Z 显示它在 mean 之后只剩很小的残差。

本阶段预注册的 phi 是 Linear(504) → SiLU → Linear(640)，隐藏维度 128。`z_assignment = mean_i phi(pair_i)`，idle 也占一个 token。全局状态和 value head 的输入宽度 887 保持 Phase-5Y。REP 和 TEMP 在冻结代码里只有 assignment 级向量，因此广播到每个 UAV token，没有新的 per-UAV 特征。

## Q2–Q4

UAV 整体重编号、buoy 整体重编号、候选枚举顺序的 value 误差分别是 0.000e+00、0.000e+00、0.000e+00，都低于 1e-5。permutation equivariance 成立。

训练集 310100–310109 上，Phase-5Z 已登记且 `|ΔJ15| ≥ 1 block` 的主动 swap 有 2834 对。旧 post-pool cosine ≥ 0.99 的比例是 1.0000。随机初始化的 pair-before-pool 降到 0.3119。cosine 中位数是 0.9662，10% 分位是 0.8303。预先写定的上限是 0.10。0.31 仍然高于 0.10，所以不能把 correspondence collapse 记成已经消除。

effective rank、确认集 collision 和 selected J15 都没有计算。那些指标要求先通过 structural gate，再训练。

## Q5–Q11

没有训练，因此没有新的 value，也没有 310114 的新 rank、相对 Phase-5V / Phase-5Y 的 selected J15、residual 修复数、seed 方向或 inference 计时。用这次没通过的随机投影去报性能，会把结构失败说成决策结果。

## Q12

没有达到 POOL-A，不能冻结后进入 310118–310125。

Gate 是 **POOL-D**。结构检查的名字是 **POOLSTRUCT-C**：pair-before-pool 保留了合法的重编号等价，但在登记的高价值 swap 上，仍有约三成 pair 的 pooled cosine ≥ 0.99。下一步只能重新设计 pair binding / aggregation，不能加 HGNN 深度，也不能把这次的 phi 拿去训练。

`planner_ready = False`

`deployment_verified = False`
