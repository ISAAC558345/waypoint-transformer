# Phase-10.0 协议追溯

## Phase-9.4.5 磁盘事实
- DeepSets-old payload = 3200000.0
- DeepSets-util payload = 6950000.0
- HGNN-old payload = 3393750.0
- HGNN-util payload = 6125000.0
- UOBJ = UOBJ-A
- GRAPH-U = GRAPH-U-C
- next_action = retain DeepSets
- 与用户表逐项相等 = {'deepsets_old': True, 'deepsets_util': True, 'hgnn_old': True, 'hgnn_util': True, 'uobj': True, 'graph_u': True, 'next_action': True}

## L=24 vs 口头 L=32
- `protocol.py` POOL_SIZE = 24
- `phase945_prefinal_freeze.json` pool_size = 24
- acceleration `frozen_pool_size` = 24
- candidate/utility manifest pool_size = 24/24
- acceleration note_L = Phase-9.4.5 froze L=24 before confirmation (allowed runtime clause). Acceleration did not change L.
- 源码中的 POOL_SIZE=32：无
- 说明：Disk freeze is L=24 in protocol.py, prefinal_freeze.json, candidate/utility manifests, and acceleration amendment note_L. No POOL_SIZE=32 remains in Phase-9.4.5 source or JSON. An earlier acceleration instruction mentioning 32 was never written as a frozen scientific L. Recorded as oral-vs-disk discrepancy / protocol deviation; history not rewritten; no rerun.
- checkpoint_selection.csv 偏差：{'path': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase945\\outputs\\utility_aligned\\phase945_checkpoint_selection.csv', 'all_best_epoch_minus_one': True, 'note': 'CSV lists best_epoch=-1/nan (skip-train leftover). Official .pt files are still the train() best-val-regret state_dicts; recover epoch from phase945_train_history_*.csv. Not rewritten.'}

## Checkpoint 规则
- 旧（9.4.3）= `last_epoch_fixed_phase94`
- 新（9.4.5）= `best_validation_mean_candidate_regret`
- 是否一致 = False
- 表述：收益对齐训练流程整体改进

## 系统边界
- 已验证：DeepSets 高层 → assignment/task option → fly-toward 规则执行器 → L3-B。
- 未验证：DeepSets 高层 → L3-B 目标条件低层 PPO → L3-B。
- 不能把「Round-30 PPO 核心文件未改」写成「当前实际使用低层 PPO」。
- 9.4.5 utility 标签在 fly-toward 下计算；换成学习低层后高层收益可能变化。本阶段未重算标签。
