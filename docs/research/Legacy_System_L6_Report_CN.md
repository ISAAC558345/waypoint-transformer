# 旧系统 L6 报告

父指令是 `2b2fa2f3ca66d64688a122d939e2b9bdb5f6605a`。本包补了正式单场景入口，并只执行一次 48 步 Scheme C 回归。没有训练，没有使用 L3 权重或 `ROW_MEAN_IDLE`。

## 工程完善

`wt.legacy_system.run` 默认只预览配置，不建目录、不加载 torch。显式 `--run` 通过 R1 的 `prepare_policy_copy` 和 `run_arm` 调用正式 `run_active_episode`。策略副本放在新目录，现行字段是 G1_RATE、k=6、hold=15、scheme_c。`historical_formal_k=5` 只保留旧候选数来源。

`run_active_episode` 增加了默认空的 `fault_log` 参数，并传到已有的 `run_closed_episode`。本次没有异常，`faults.jsonl` 没有写入。官方 `active_policy.json` 的 SHA-256 在运行前后都是 `3fc2f0b1277188e6f12084370c9c1d5acb70e897f6a4fba8b47bedf031b25ebb`。

## 48 步回归

场景 322000，process_seed=1322000，reset_seed=322000，stop_time=48。回合完成，终端时刻 48，决策时刻 0/15/30/45，Scheme C 调用 4 次。真实 payload 是 63785945.28984571 bit，与 R1 原件和公布参考值的绝对差都是 0。served 绝对差是 0。48 条逐步 served 都在 1 bit 容差内，没有失配行。运行内 `payload_matches_served=1`，逐步 served 与回合 served 一致。

环境步 48，训练更新 0。异常数 0。回归状态是 matched。

本次读取的权重与登记哈希一致：Scheme C `f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864`，MAPPO-V2 `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`，decoder `de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5`。E6 只出现在正式启动哈希里，不是本次高层策略。已有运行环境是 Python 3.14.2、NumPy 2.4.1、torch 2.12.0+cpu。这不是一次新机器安装验证。

## 研究结案

L3–L5 的配置链已写入 [结案索引](Legacy_ArrivalAware_L3_L5_Closure_CN.md)。A/B 权重和固定参照补丁都未采用。该索引没有进入论文正文或 PDF。

## 遗留

- 当前 LaTeX 工程仍未编译。本包没有检查编译器。
- `results/phase6if_ranking_loss/summary.json` 仍然缺失。
- `lambda_H` 与 `delta_payload_max` 仍未设定。
- 到达感知和 payload/标签服务的研究问题仍然开放。本包没有因此再跑场景或训练。
