# HTW Phase-4E Independent Holdout Deployment Verification

本阶段只用 Phase-4D7F 已生成的 holdout 标签，评估冻结的 B2 + Gate-1 + Candidate-C。
U_traj 只在 Gate-1 选定策略之后用于评价。没有重训、调参、改 criterion、改 planner 或重新生成 scenario。

**status = DEPLOYMENT_VERIFICATION_FAILED**

**deployment_verified = False**

**planner_ready = False**

## 冻结审计

冻结审计通过 = True。
B2 仍是 Phase-4C 的重建方式：ExtraTrees 只在 221000–221015 的 EXEC 标签上 fit，超参数为 n_estimators=300、max_depth=8、min_samples_leaf=5、random_state=0、n_jobs=1。
τ = 1534933.5944128654。criterion = CONSERVATIVE_WITH_FALLBACK_V1。C1–C8 与注册表一致。
评估脚本没有调用 `decide()`。

## 数据

Batch-1 IDs = 230000–230063，n = 64。
Batch-2 IDs = 230064–230127，n = 64。
两批 ID 不相交，并且都不在 B2 fit 的 16 个 design state 中。

## Batch 结果

### Batch-1

- n = 64
- n_B2 / n_LOCAL = 20 / 44
- B2 使用率 = 0.312500
- mean / median R = 0.844700 / 0.916667
- mean / median U_traj = 1.051562e+07 / 1.100000e+07
- mean / median ΔR = 0.101687 / 0.000000
- degradation count / rate = 0 / 0.000000
- worst-case ΔR = 0.000000（scenario 230002）
- mean U gain = 1.218750e+06
- top-1 / top-3 = 0.390625 / 0.671875
- B2-ORACLE overlap = 0.421875
- C1 degradation_rate <= 0.02：True
- C2 worst_case_delta_R >= -0.125：True
- C3 mean_delta_R >= 0.02：True
- C4 mean_U_gain >= 0：True
- batch PASS = True
- degraded scenarios = []

### Batch-2

- n = 64
- n_B2 / n_LOCAL = 30 / 34
- B2 使用率 = 0.468750
- mean / median R = 0.858877 / 0.909091
- mean / median U_traj = 1.026562e+07 / 1.000000e+07
- mean / median ΔR = 0.113262 / 0.000000
- degradation count / rate = 4 / 0.062500
- worst-case ΔR = -0.200000（scenario 230077）
- mean U gain = 1.437500e+06
- top-1 / top-3 = 0.296875 / 0.453125
- B2-ORACLE overlap = 0.296875
- C1 degradation_rate <= 0.02：False
- C2 worst_case_delta_R >= -0.125：False
- C3 mean_delta_R >= 0.02：True
- C4 mean_U_gain >= 0：True
- batch PASS = False
- degraded scenarios = [230070, 230077, 230078, 230091]

### Combined

- n = 128
- n_B2 / n_LOCAL = 50 / 78
- B2 使用率 = 0.390625
- mean / median R = 0.851788 / 0.912879
- mean / median U_traj = 1.039062e+07 / 1.000000e+07
- mean / median ΔR = 0.107475 / 0.000000
- degradation count / rate = 4 / 0.031250
- worst-case ΔR = -0.200000（scenario 230077）
- mean U gain = 1.328125e+06
- top-1 / top-3 = 0.343750 / 0.562500
- B2-ORACLE overlap = 0.359375
- C1 degradation_rate <= 0.02：False
- C2 worst_case_delta_R >= -0.125：False
- C3 mean_delta_R >= 0.02：True
- C4 mean_U_gain >= 0：True
- batch PASS = False
- degraded scenarios = [230070, 230077, 230078, 230091]

## 保留的 Batch-2 失败证据

这 4 个 scenario 全部保留。Gate-1 因为 margin >= τ 选择了 B2。没有删除、替换或据此改 τ。

| scenario | policy | margin | ΔR | ΔU | R_LOCAL | R_selected |
| --- | --- | --- | --- | --- | --- | --- |
| 230070 | B2 | 2.190856e6 | -0.181818 | -2.000000e6 | 0.727273 | 0.545455 |
| 230077 | B2 | 1.941896e6 | -0.200000 | -2.000000e6 | 0.800000 | 0.600000 |
| 230078 | B2 | 3.810212e6 | -0.181818 | -2.000000e6 | 0.818182 | 0.636364 |
| 230091 | B2 | 1.555024e6 | -0.090909 | -1.000000e6 | 1.000000 | 0.909091 |

Batch-2 的 C1 失败是因为 4/64 = 0.0625 > 0.02。C2 失败是因为最差 ΔR = -0.200 < -0.125。C3 和 C4 通过。

## Acceptance

Batch-1 PASS = True
Batch-2 PASS = False
Combined 数值检查 PASS = False
independent_batches_passed = 1
deployment_pass = False

Combined 的数值结果不能代替 required_independent_batches = 2。一个 batch 失败时，另一个 batch 不能抵消。
C5 = allow_some，C6 = False，所以单个 scenario 的 degradation 只计入 C1/C2，不单独改写 criterion。

## 二十个问题

1. Batch-1 完整 64：True
2. Batch-2 完整 64：True
3. 两个 batch 真正独立：ID 区间不相交，且都不在历史 fit / validation 池中。
4. B2 完全冻结：是。fit 数据只有 221000–221015。
5. Gate-1 完全冻结：是。margin >= τ 才选 B2。
6. criterion 完全冻结：是。
7. Batch-1 B2 使用率 = 0.312500（20/64）
8. Batch-2 B2 使用率 = 0.468750（30/64）
9. Batch-1 mean ΔR = 0.101687
10. Batch-2 mean ΔR = 0.113262
11. Batch-1 degradation rate = 0.000000
12. Batch-2 degradation rate = 0.062500
13. worst-case ΔR：Batch-1 0.000000，Batch-2 -0.200000
14. mean U gain：Batch-1 1.218750e+06，Batch-2 1.437500e+06
15. Batch-1 通过 Candidate-C：True
16. Batch-2 通过 Candidate-C：False
17. 两个 independent batch 同时通过：False
18. leakage：False
19. deployment_verified = False
20. planner_ready 仍然为 False。

verification 未通过。B2、τ、Gate-1、criterion、holdout 和 planner 都保持原状。

本阶段只做 evaluation、audit 和 PASS/FAIL，不进入 planner integration。
