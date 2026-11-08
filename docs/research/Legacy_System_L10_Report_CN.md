# L10 按场景正确运行与比较

322001 的一次 48 步正式运行与对应 R1 参考 matched，进程退出码 0。入口现在按登记场景读取各自的参考、种子和默认输出。

## 解除的硬编码

场景、参考 episode、参考 config、process_seed、reset_seed 和默认输出集中在 `wt.legacy_system.references`。当前只登记已有完整 R1 参考的 322000 与 322001。

预览、执行和 manifest 都使用这一处解析。322000 的参考路径不再被写进其他场景。比较器从本次 reference 读取 payload 和 served。`reference_payload_bits` 是该参考 episode 的实际 payload，并标明与同一份 R1 文件同源。`scenario_id` 和 `policy` 必须两边都存在且一致，缺失字段不能通过。

未指定 `--output` 时，默认目录为 `results/legacy_system/scenario_{scenario_id}`。显式路径保持可用，包括已文档化的 322000 命令。新 manifest 的包名是 `legacy_system`，指令来源是 `204dd3309115dfd369236c408f7e59306a0871d4`。原 L6 与 L7 manifest 没有回写。

## 保存结果检查

`results/legacy_system_l10/stdlib_check.json` 在加载模型之前完成，环境步、低层推理和高层调用都是 0。

- 322000 与 322001 的预览分别指向自己的 R1 参考、种子和默认输出，没有创建预览目录，也没有导入 torch。
- 原 R1 322000 与已保存 L6 比较为 matched。
- 原 R1 322001 与其自身比较为 matched，参考 payload 为 68952073.58225587 bit。
- 322000 参考与 322001 记录交叉比较为 different。
- 两套参考都有 scenario_id、policy，并且是完整 48 步。去掉身份字段后的比较为 different。
- 未登记的 322002 在 `--run` 时返回 2，输出目录没有创建。

## 322001 的 48 步

命令是 `py -3 -m wt.legacy_system.run --scenario 322001 --output results/legacy_system_l10/run_322001 --run`。

| 字段 | 结果 |
| --- | --- |
| scenario_id | 322001 |
| process_seed | 1322001 |
| reset_seed | 322001 |
| stop_time / terminal_time / environment_steps | 48 |
| decision_times | 0, 15, 30, 45 |
| scheme_c_calls | 4 |
| payload_bits | 68952073.58225587 |
| served_bits | 68952073.58225589 |
| payload_matches_served | 1 |
| step_served_matches_episode | true |
| error | null |
| regression_status | matched |
| success | true |
| exit_code | 0 |

总量和全部 48 条逐步 served 与 R1 322001 的绝对差都是 0 bit，容差仍为 1 bit。这次核对的是入口一致性。

## 实际消耗

- 标准库检查环境步：0
- 322001 环境步：48
- 低层推理：48，按正式执行中的 MAPPO 动作调用单独计数
- 正式高层调用：4
- 参数更新：0
- 反向传播：0

没有重跑，也没有重跑 322000。

## 仍保持的边界

L8 与 L9 仍是 NOT_PROMOTED，没有接入正式 Planner。Scheme C、MAPPO-V2、decoder 的哈希与合同一致，正式策略哈希在运行前后都是 `3fc2f0b1277188e6f12084370c9c1d5acb70e897f6a4fba8b47bedf031b25ebb`。

当前可运行的只有登记参考的两个训练场景。没有开放整个训练池、validation 或 reserved test。正式分支与海洋观测隔离分支没有修改。
