# HTW Phase-4D-7F Independent Holdout Generation

本阶段生成两个新的 deployment holdout batch，并只做完整性审计。
未计算 R、ΔR、B2 vs LOCAL、degradation、ORACLE overlap 或 deployment PASS/FAIL。
未调用正式 planner，未调用 `decide()`。
未修改 B2、τ、Gate-1、Candidate-C、criterion、planner。未进入 Phase-4E。

**status = INDEPENDENT_HOLDOUT_DATA_CREATED_PENDING_VERIFICATION**

**planner_ready = False**

**data_registered = True**

**deployment_verified = False**

**deployment_criterion = CONSERVATIVE_WITH_FALLBACK_V1**（未改）

**holdout_batch_size_status = SPECIFIED**（64 + 64，来源仍是 HUMAN_SPECIFICATION）

## 1. Registry 检查

先读了 `src/wt/lowlevel_v2/data_registry.py`、`LLV2_DATA_REGISTRY.json`、Phase-10.4 `all_blocked_scenario_seeds()` / `scan_historical_seeds()`，以及 Phase-4D7E 的 holdout 规格。

结论：

- `allocate_new_pool(blocked, base, count)` 只从 `LLV2_TRAIN_BASE=220000` 或 `LLV2_DESIGN_BASE=221000` 起跳过已占用 ID。它产出的是 LLV2_train / LLV2_design，不是 deployment holdout。本阶段没有调用它。
- `build_registry()` 把 `LLV2_confirmation` 和 `LLV2_final_untouched` 保持为 `null`。仓库里没有现成的 deployment holdout namespace。
- 因此在 `phase4d7b_independent_holdout_generation/outputs/` 下新建独立 namespace `deployment_holdout`，用单独的 manifest 和 `holdout_scenario_registry.json` 登记。`data_registry.py` 与 `LLV2_DATA_REGISTRY.json` 未改。LLV2_train / LLV2_design 的语义未改。

可以区分的池：

| 池 | ID |
| --- | --- |
| LLV2_train | 220000–220063 |
| LLV2_design | 221000–221015 |
| Phase-2 validation | 222000–222063 |
| Phase-3 validation | 223000–223063 |
| historical independent | 104000–104063，105000–105063，106000–106063 |
| deployment_holdout_batch_1 | 230000–230063 |
| deployment_holdout_batch_2 | 230064–230127 |

## 2. ID 分配

分配前把下列来源并成 `used_scenario_ids`，共 2873 个历史 ID：

- 显式禁止区间：220000–220063、220000–221199、221000–221015、222000–222063、223000–223063、104000–106063
- `LLV2_DATA_REGISTRY.json` 的 train / design
- `scan_historical_seeds()`
- `all_blocked_scenario_seeds()`

230000–230127 与这 2873 个 ID 的交集为空，才锁定分配。分配写入 `allocated_holdout_ids.json`。后续 resume 只读这份分配，不会改已经成功的 ID。

- Batch-1：`deployment_holdout_batch_1`，230000–230063，64 个
- Batch-2：`deployment_holdout_batch_2`，230064–230127，64 个
- `scenario_id` 就是 `L3BEnv.reset(seed=scenario_id)` 的 seed
- 两个 batch 的 ID 不相交，所以是两批新 realization，不是把 Batch-1 换一个 seed 再写成 Batch-2

环境保持 L3-B 默认：M=6，K=12，T=48，hold=15。

`config_id = L3BEnv.reset(seed=scenario_id); env_sha=de22347ea625a330f61e8f62b2a973066f4669c7498fd58bac2fd8bf706632ba; M=6;K=12;T=48;hold=15`

## 3. 标签

复用 Phase-4C 同一套官方入口，输出只写在本阶段目录：

- `auditor_env_reset(scenario_id)` → `L3BEnv.reset`
- `generate_pool(env, seed_key=scenario_id * 1009 + 0)`，每场 24 个 first-hold candidate
- `U_single`：`LOW_SEEDS=(0,1,2,3,4)` 上 `shadow_h15` 的 `payload_increment` 均值，`zero_residual=False`
- `U_traj`：`exec_ckpt(0)` + `make_override_stack(..., LL_RUN=0)` + `run_episode(..., compute_ull=False)`
- Phase-3E geometry：`extract_candidate_geometry`
- `C_abs = U_traj - U_single`，只作为恒等式落盘，没有拿来选 candidate 或删 scenario

没有用 B2、Gate-1 或 criterion 生成标签、选择 candidate 或筛选 scenario。128 个 ID 全部保留。没有按 U_traj、margin 或异常结果删行、换 seed 或补采样。

`generation_failure.json` 不存在。没有生成失败项。

复核（对全部 3072 行，而不是只看每个 scenario 的第一行）：

- 行数 3072 = 128 × 24
- scenario–candidate 对唯一
- 每个 scenario 恰好 24 个 candidate
- `U_single`、`U_traj`、`C_abs` 和 17 个 Phase-3E 字段全部可解析，缺失数 0
- 标签列里没有 R、ΔR 或 PASS/FAIL
- `n_hold` 取值为 {2, 3, 4}，这是 T=48、hold=15 下 episode 的自然长度，没有据此删除任何 scenario

## 4. 时间顺序

文件 UTC 修改时间早于第一场 holdout 的 `generation_timestamp`（2026-09-28T03:29:42Z）。生成结束于 2026-09-28T04:10:26Z。这些冻结文件在生成期间没有被改写：

| 对象 | LastWriteTimeUtc |
| --- | --- |
| `high_level.py` | 2026-09-28 03:08:34 |
| `data_registry.py` | 2026-09-28 03:08:38 |
| B2 `protocol.py` / `shadow_scorer.py` | 2026-09-28 03:08:50 |
| criterion registry | 2026-09-28 03:09:39 |
| Phase-4D6 注册报告 | 2026-09-28 03:09:46 |
| holdout 64+64 规格 | 2026-09-28 03:23:21 |
| Phase-4D7E 报告 | 2026-09-28 03:23:25 |
| 第一场 holdout 开始 | 2026-09-28 03:29:42 |

顺序是：B2 已冻结 → Gate-1 / τ 已冻结 → Candidate-C 已注册 → 人工指定 64+64 → 才开始 reset 230000。生成之后没有回改这些对象。

`LLV2_DATA_REGISTRY.json` 的时间戳仍是 2026-09-19，本阶段没有写入。

## 5. 完整性审计

`holdout_integrity_audit.csv` 共 128 行。

| 检查 | 结果 |
| --- | --- |
| A. scenario ID 不重复 | 128 个唯一 ID |
| B. Batch-1 内部不重复 | 230000–230063，64 个 |
| C. Batch-2 内部不重复 | 230064–230127，64 个 |
| D. 两批不重复 | 交集为空 |
| E. 不属于历史池 | `in_historical_pool` 全为 0 |
| F. 未进入 B2 fit | fit 池仍是 221000–221015；`in_b2_fit` 全为 0 |
| G–K. 未进入 feature / hyperparameter / τ / Gate-1 / criterion selection | 分配只看历史 ID 是否占用，不读这些选择结果 |
| L–N. 未改模型、未重选 Candidate-C、未改 planner / decide() | 上述冻结文件时间戳早于生成，且生成脚本不写这些文件 |

## 6. 十四个问题

1. Batch-1 恰好 64。是。230000–230063。
2. Batch-2 恰好 64。是。230064–230127。
3. 共 128 个新 scenario。是。标签 3072 行。
4. scenario ID 全部是新的。是。位于 230000–230127，与禁止区间和 2873 个已扫描历史 ID 无交集。
5. seed 全部是新的。是。每个 seed 等于对应的新 scenario_id。
6. 与历史池完全无交集。是。
7. 没有进入 B2 fit。B2 fit 仍是 16 个 design state 221000–221015。
8. 没有进入 feature、hyperparameter、τ、Gate-1 或 criterion selection。
9. U_single 完整。128 场 × 24 candidate，缺失 0。
10. U_traj 完整。同样 3072 行，缺失 0。
11. Phase-3E candidate geometry 完整。17 个字段在 3072 行上全部可解析。
12. 没有生成失败。
13. 没有 leakage。新 ID 未进入上述选择，也未改冻结模型、阈值或 criterion。
14. 数据条件已经满足，下一阶段才可以做 deployment verification。本阶段没有执行 verification，`deployment_verified = False`，`planner_ready = False`。

## 7. 输出

目录：`src/wt/lowlevel_v2/phase4d7b_independent_holdout_generation/outputs/`

- `deployment_holdout_batch_1_manifest.json`
- `deployment_holdout_batch_2_manifest.json`
- `deployment_holdout_manifest.json`
- `holdout_scenario_registry.json`
- `holdout_integrity_audit.csv`
- `holdout_labels.csv`（3072 行 ground truth）
- `allocated_holdout_ids.json`
- `scenario_meta.json`

总 manifest 记录 `batch_1_size=64`、`batch_2_size=64`、`total_size=128`、两批 scenario 列表、`source=HUMAN_SPECIFICATION`、`b2_frozen=true`、`gate1_frozen=true`、`criterion_frozen=true`、`deployment_evaluation=false`。

本阶段只完成 independent holdout data creation，不执行 deployment verification，不修改任何冻结模型/阈值/criterion，不进入 Phase-4E。
