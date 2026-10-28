# Stage E8：显式冻结评估运行时上下文并复核单场景重放

- 仓库：ISAAC558345/waypoint-transformer
- 正式基线：feature/maritime-wt-stage1a
- 来源诊断分支：experiment/priority-tag-replay-repro
- 当前实验分支：experiment/priority-tag-eval-context
- 来源 E7 提交：64ae1907eae2354d64625361c0a0afd866982d42
- 前置 Gate：REPLAY_DIVERGENCE_LOCALIZED
- 本阶段 Gate：EVAL_CONTEXT_REPRO_PASS / EVAL_CONTEXT_REPRO_INCONCLUSIVE / EVAL_CONTEXT_REPRO_BLOCKED
- 本阶段只固化评估运行时上下文并验证单场景；不训练、不修改奖励、不运行 224 条归因、不合并分支。

## 1. E7 已确认的结果

E7 对 L0 seed 0、DEV 场景 321101 的 48 步重放定位到运行时上下文差异：
- 来源评估的相关代码、checkpoint 与当前实验代码 hash 一致；
- fresh 进程默认环境为 torch threads=12、deterministic_algorithms=False；
- 评估结束时继承的上下文为 torch threads=1、process seed=2、deterministic_algorithms=True；
- fresh 与 inherited 的 reset state 一致，但第一步 mu 开始出现微小数值差异，之后轨迹累计差异扩大；
- fresh 与保存结果的 payload 差为 1.553167 bit；继承上下文的重放与保存结果完全一致；
- instrumentation 路径与 fresh 路径逐步一致，说明归因包装不是本次分叉来源。

因此下一步必须将运行时上下文显式记录并设置，而不能依赖训练进程遗留状态。注意：E7 中 seed=2 是历史评估进程最后一次训练留下的上下文证据，不应不加分析地把它解释为每个模型的通用评估 seed。必须分别记录模型 seed、环境/reset seed、全局 Python/NumPy/Torch RNG seed 和运行时 flags，避免概念混淆。

## 2. 目标

仅验证：在显式配置与记录评估运行时上下文后，L0 seed 0、场景 321101 是否能在新进程中重复匹配已保存评估结果，且不依赖训练进程遗留状态。

## 3. 实现边界

- 只在当前实验分支的独立诊断 runner/config 中添加 context capture/restore helper；不改正式 worker、正式 evaluate_all、正式 Planner、正式 reward、正式 observation、SafetyProjector 或任何 checkpoint。
- 保存一个明确的 runtime context manifest，至少包含：
  - Python、NumPy、PyTorch 版本和平台；
  - torch.get_num_threads()；
  - torch.are_deterministic_algorithms_enabled()；
  - cuDNN deterministic/benchmark flags（如果适用）；
  - Python、NumPy、Torch CPU、Torch CUDA RNG 状态或可复现的 seed；
  - 环境 reset seed 与 scenario ID；
  - 模型 checkpoint SHA-256、相关代码 SHA-256、git commit、配置文件 SHA-256；
  - 模型 eval 模式、deterministic action 设置、观测归一化状态。
- 按 E7 已复现保存结果的上下文明确设置 torch threads=1 和 deterministic_algorithms=True。seed 必须从历史评估协议/日志核实：如果无法确定真实的全局 seed，分别保留“已证实的历史继承上下文”和“规范化的显式 seed 上下文”两种标签；不得把后者冒称为历史原始设置。
- 不要只在同一进程内顺序执行两次。至少运行两次独立新进程，每次显式加载相同 manifest。
- 记录每步 observation hash、mu/desired/projected/executed action、position、q_bits、served/dropped/snap、payload 累计和 path length。
- 不允许四舍五入、增加容差或修改 checkpoint 来制造匹配。

## 4. 运行范围与成功标准

只运行：
- L0 正式 checkpoint phase6g_seed0.pt；
- model seed 0；
- DEV 场景 321101；
- 48 步；
- 两次独立新进程的受控重放。

通过 registry/split guard 确认该场景为允许的 development 场景。不得访问 reserved/test，也不得运行其他场景或重训 L1/L2。

EVAL_CONTEXT_REPRO_PASS 必须满足：
1. 两次独立进程的 reset state hash、逐步 observation/action/position/q_bits/served/dropped/snap 轨迹完全一致，或仅存在既有容差内差异；
2. 两次运行均与保存的 episode 结果满足 payload/accounting 差不超过 1 bit；
3. path length 差小于或等于 1e-9 m；
4. ledger accounting 与 E1 既有规则一致；
5. manifest 完整，能解释所有显式 seed 与 flags；
6. 正式 checkpoint、正式代码、历史结果哈希不变。

若两次进程互相一致，但与保存结果不一致，Gate 为 INCONCLUSIVE，并报告历史评估缺少的环境元数据；不能把“当前可重复”误报为“历史已复现”。若两次新进程不一致或来源文件/split/hash 校验失败，Gate 为 BLOCKED。

## 5. 交付物

- docs/research/HTW_UrgencyAware_Eval_Context_Repro_Report_CN.md
- results/phase6i_priority_tag_eval_context/runtime_context.json
- results/phase6i_priority_tag_eval_context/step_diff.csv
- results/phase6i_priority_tag_eval_context/summary.json
- results/phase6i_priority_tag_eval_context/schema.json
- 独立诊断 runner/helper 与对应测试

报告需记录来源 E7 commit、运行环境、所有 hash、两个独立进程的对比、与保存结果的差异、Gate 和下一步建议。所有修改仅留在当前实验分支。

## 6. 禁止事项与停止规则

- 不重新训练 L1/L2；
- 不改 lambda_u、rho、beta_tag、tag_rate、队列顺序、动作、reward、SafetyProjector 或模型；
- 不执行完整 224-episode attribution replay；
- 不修改正式分支或合并任何实验分支；
- 不因历史结果缺少元数据而放宽 1 bit 或 1e-9 m 阈值。

完成后提交报告和结果并停止。如果 E8 通过，下一阶段才制定 E9 完整归因重放方案；如果不通过，先修复/记录运行环境可复现性，不继续做年龄与标签因果解释。
