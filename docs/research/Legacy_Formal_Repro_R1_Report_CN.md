# 正式路径轻量复现 R1

运行代码 `ed2c87c83dcdaccee1420ee3e14a8dfa19982e92`。指令提交 `d2b214f35c5c35133f8412c57b86cd3605ef8ed6`。
首次写清单时把源码目录当成了仓库根目录。已有的 192 步结果被移到 results/legacy_formal_repro_r1，没有再跑环境步。
本包通过 `run_active_episode` 的配置副本运行两个 TRAIN 场景。每个场景比较 scheme_c 与 g1_rate，两条臂的候选池都是 G1_RATE k=6。
192 步只说明正式入口可以运行并留下可追溯输出，不构成泛化评估，也不表示系统已经发布。

## 运行合同

- 场景：[322000, 322001]。进程种子为 1000000 加场景号；环境 reset 种子仍是场景号。
- 每臂 48 步，登记上限 192 步，实际环境步数 192。
- 训练更新 0。Scheme C 调用 8 次。完成的低层动作步 192。下载 0。
- 正式策略文件保持未改：True。
- 实际读取的 MAPPO 文件哈希 `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`。
- 实际读取的 Scheme C 文件哈希 `f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864`。

## 分场景结果

| 场景 | Scheme C payload (bit) | G1_RATE payload (bit) | 差值 (bit) | Scheme C 耗时 (s) | G1_RATE 耗时 (s) | 两端完成 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 322000 | 63785945.289846 | 64731141.622913 | -945196.333067 | 2.299303 | 2.068875 | True / True |
| 322001 | 68952073.582256 | 63445121.872806 | 5506951.709450 | 1.799467 | 1.957489 | True / True |

## 原始证据

归档目录与新 192 步结果分开。恢复 40 个文件，缺档 1 个。
AB 汇总是开发集统计，AC 汇总是已保存的 holdout 统计。本次没有重开这些场景。
缺档列表：['results/phase6if_ranking_loss/summary.json']。

## 历史失败索引

来源 `7e9e372e6ab603342f59c334301cdcc42ccc2e61:results/ocean_observation_m0/baseline_tests.log`，签名 206 个。
分类计数：{'historical_state': 11, 'missing_file': 172, 'assertion_failure': 23}。
其中触及当前正式入口或比特队列测试的签名为 0 个。
这些历史失败保持原分类。本包没有改写队列、锁或测试来清零。

## 必要检查

test_policy_copy_leaves_the_official_file_unchanged 通过。test_saved_run_matches_the_registered_contract 通过：四个臂的决策时刻都是 0、15、30、45，逐步 served 之和与 payload 一致。test_live_graph_uses_the_raw_edge_contract 通过。test_accounting_cases_pass 通过。这些检查内部的环境步数不计入 192 步方法证据。

## 保留差距

正式入口可以运行，输出能够追溯到配置副本、模型哈希和逐步 served。
历史套件没有重跑，既有 206 个失败签名仍然存在。ranking-loss 的 summary.json 仍为 MISSING。
本结果的 deployment_verified 不适用，也不把两个场景写成总体优劣结论。
