# M1.6：可执行候选与浮标分配一致性

性能判定：被动候选在本组登记场景得到性能支持。

默认选择模式仍是 legacy，默认终端模式仍是 legacy。本轮候选同时使用 M1.5 的 return_tail_when_no_business 和 initial_guard_consistent，只在显式打开时生效。主要候选是保留普通业务遥测的 OFF，ON 是对照。校准开关的差值只说明行为，不解释成信息价值。本轮没有验证风浪感知或共享环境收益。

## 期初替代

b3_loose 在 180 秒前的请求与 M1.3 一致：True。重放状态与 M1.5 日志一致：True。
期初替代成立：True。区间中途替代：False。
lookahead_calibration_off UAV0：电量 41404.386 J，service:0 报价 31533.310 J，准入 True，期初有效动作 service:0。
lookahead_calibration_off UAV1：电量 34532.108 J，service:0 报价 41548.383 J，准入 False，期初有效动作 return:None。
lookahead_calibration_on UAV0：电量 41404.386 J，service:0 报价 31533.310 J，准入 True，期初有效动作 service:0。
lookahead_calibration_on UAV1：电量 35050.184 J，service:0 报价 41570.519 J，准入 False，期初有效动作 return:None。
lookahead_calibration_off UAV1 请求 service:0，期初守卫输出 return:None，返航守卫改变相位 None，区间末执行 return:None。
lookahead_calibration_on UAV1 请求 service:0，期初守卫输出 return:None，返航守卫改变相位 None，区间末执行 return:None。

## 候选次序与 20000 bit

lookahead_calibration_off：legacy 联合请求 service:0|backhaul:None，M1.5 尾模式联合请求 return:None|service:0，新选择联合请求 service:0|return:None。尾模式首选 service:0|service:0，legacy 首选 service:0|backhaul:None。
lookahead_calibration_on：legacy 联合请求 service:0|backhaul:None，M1.5 尾模式联合请求 return:None|service:0，新选择联合请求 service:0|return:None。尾模式首选 service:0|service:0，legacy 首选 service:0|backhaul:None。
M1.5 在 180 秒把两机首选都放到 service:0 后，冲突修复让绝对 Q 较低的一架改为 RETURN；另一架分到的 service:0 在期初就被原准入器改成 RETURN。缺掉的是浮标 0 的两个 10000 bit 包。
m15_replay lookahead_calibration_off b3_loose-lookahead_calibration_off-b0-p0 最终持有者 buoy，交付时刻 None。
m15_replay lookahead_calibration_off b3_loose-lookahead_calibration_off-b0-p1 最终持有者 buoy，交付时刻 None。
m15_replay lookahead_calibration_on b3_loose-lookahead_calibration_on-b0-p0 最终持有者 buoy，交付时刻 None。
m15_replay lookahead_calibration_on b3_loose-lookahead_calibration_on-b0-p1 最终持有者 buoy，交付时刻 None。
两条新航次在第 6 个决策区间的请求都是 UAV0 service:0、UAV1 return:None，并且实际执行与请求相同。区间 9 结束时 UAV0 位于浮标 0。两包随后都由 UAV0 回传：OFF 与 ON 的交付时刻同为 1712.897 s 和 1715.897 s，均在 1800 s 截止前，最终去向是岸站。相对 M1.5，这 20000 bit 被收回；OFF 比 M1.5 OFF 多耗电 7765.915 J，ON 比 M1.5 ON 多耗电 4398.444 J。其余 11 个场景相对 M1.5 的 bit、净效用和能耗差都是 0。

## 新航次

b3_loose 先行 OFF 是否保持 M1.3 与 M1.5 的交付量和净效用：True。其余场景是否运行：True。
两条先行航次在 180 秒前与冻结前缀一致，它们同时作为同状态反事实续跑。

| 场景 | 比较 | bit差 | J差 | 能耗差 J | 回退 | 物理变差 | 备用电量变差 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| b3_loose | new_OFF-m15_OFF | 20000 | 1.978428013 | 7765.915 | 0 |  | 0 |
| b3_loose | new_OFF-m13_OFF | 0 | 0.042206959 | -15194.505 | 0 |  | 0 |
| b3_loose | new_ON-m15_ON | 20000 | 1.987782101 | 4398.444 | 0 |  | 0 |
| b3_loose | new_ON-m13_ON | 0 | 0.055998822 | -20159.576 | 0 |  | 0 |
| b1_tight | new_OFF-m15_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b1_tight | new_OFF-m13_OFF | 0 | 0.014371654 | -5173.795 | 0 |  | 0 |
| b1_tight | new_ON-m15_ON | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b1_tight | new_ON-m13_ON | 0 | 0.161836428 | -58261.114 | 0 |  | 0 |
| b1_window | new_OFF-m15_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b1_window | new_OFF-m13_OFF | 0 | 0.028743307 | -10347.591 | 0 |  | 0 |
| b1_window | new_ON-m15_ON | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b1_window | new_ON-m13_ON | 0 | 0.131162539 | -47218.514 | 0 |  | 0 |
| b1_loose | new_OFF-m15_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b1_loose | new_OFF-m13_OFF | 0 | 0.028743307 | -10347.591 | 0 |  | 0 |
| b1_loose | new_ON-m15_ON | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b1_loose | new_ON-m13_ON | 0 | 0.102311628 | -36832.186 | 0 |  | 0 |
| b2_tight | new_OFF-m15_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b2_tight | new_OFF-m13_OFF | 0 | 0.017881234 | -6437.244 | 0 |  | 0 |
| b2_tight | new_ON-m15_ON | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b2_tight | new_ON-m13_ON | 0 | 0.132049609 | -47537.859 | 0 |  | 0 |
| b2_window | new_OFF-m15_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b2_window | new_OFF-m13_OFF | 0 | 0.017881234 | -6437.244 | 0 |  | 0 |
| b2_window | new_ON-m15_ON | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b2_window | new_ON-m13_ON | 0 | 0.166379632 | -59896.668 | 0 |  | 0 |
| b2_loose | new_OFF-m15_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b2_loose | new_OFF-m13_OFF | 0 | 0.046312579 | -16672.529 | 0 |  | 0 |
| b2_loose | new_ON-m15_ON | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b2_loose | new_ON-m13_ON | 0 | 0.087369056 | -31452.860 | 0 |  | 0 |
| b3_tight | new_OFF-m15_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b3_tight | new_OFF-m13_OFF | 0 | 0.038821201 | -13975.632 | 0 |  | 0 |
| b3_tight | new_ON-m15_ON | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b3_tight | new_ON-m13_ON | 0 | 0.086517992 | -31146.477 | 0 |  | 0 |
| b3_window | new_OFF-m15_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b3_window | new_OFF-m13_OFF | 0 | 0.041695532 | -15010.392 | 0 |  | 0 |
| b3_window | new_ON-m15_ON | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b3_window | new_ON-m13_ON | 0 | 0.127182418 | -45785.671 | 0 |  | 0 |
| b4_tight | new_OFF-m15_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b4_tight | new_OFF-m13_OFF | 0 | 0.014371654 | -5173.795 | 0 |  | 0 |
| b4_tight | new_ON-m15_ON | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b4_tight | new_ON-m13_ON | 0 | 0.146412615 | -52708.541 | 0 |  | 0 |
| b4_window | new_OFF-m15_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b4_window | new_OFF-m13_OFF | 0 | 0.032252887 | -11611.039 | 0 |  | 0 |
| b4_window | new_ON-m15_ON | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b4_window | new_ON-m13_ON | 0 | 0.119955409 | -43183.947 | 0 |  | 0 |
| b4_loose | new_OFF-m15_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b4_loose | new_OFF-m13_OFF | 0 | 0.032252887 | -11611.039 | 0 |  | 0 |
| b4_loose | new_ON-m15_ON | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b4_loose | new_ON-m13_ON | 0 | 0.103712512 | -37336.504 | 0 |  | 0 |
| b3_loose | new_ON-new_OFF | 0 | 0.001437225 | -517.401 | 0 |  | 0 |
| b1_tight | new_ON-new_OFF | 0 | 0.000997264 | -359.015 | 0 |  | 0 |
| b1_window | new_ON-new_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b1_loose | new_ON-new_OFF | 0 | 0.001220240 | -439.287 | 0 |  | 0 |
| b2_tight | new_ON-new_OFF | 0 | 0.001249358 | -449.769 | 0 |  | 0 |
| b2_window | new_ON-new_OFF | 0 | 0.001249358 | -449.769 | 0 |  | 0 |
| b2_loose | new_ON-new_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b3_tight | new_ON-new_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b3_window | new_ON-new_OFF | 0 | 0.002224295 | -800.746 | 0 |  | 0 |
| b4_tight | new_ON-new_OFF | 0 | 0.001249358 | -449.769 | 0 |  | 0 |
| b4_window | new_ON-new_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b4_loose | new_ON-new_OFF | 0 | 0.001525300 | -549.108 | 0 |  | 0 |

| 场景 | 方法 | bit | J | 能耗 J | 180秒前一致 | 首次分歧 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| b3_loose | lookahead_calibration_off | 140000 | 13.678652898 | 115684.957 | True | 6 |
| b3_loose | lookahead_calibration_on | 140000 | 13.680090123 | 115167.556 | True | 6 |
| b1_tight | lookahead_calibration_off | 40000 | 3.895623457 | 37575.556 |  |  |
| b1_tight | lookahead_calibration_on | 40000 | 3.896620721 | 37216.541 |  |  |
| b1_window | lookahead_calibration_off | 90000 | 8.829704965 | 61306.212 |  |  |
| b1_window | lookahead_calibration_on | 90000 | 8.829704965 | 61306.212 |  |  |
| b1_loose | lookahead_calibration_off | 140000 | 13.747448566 | 90918.516 |  |  |
| b1_loose | lookahead_calibration_on | 140000 | 13.748668807 | 90479.230 |  |  |
| b2_tight | lookahead_calibration_off | 40000 | 3.872631368 | 45852.708 |  |  |
| b2_tight | lookahead_calibration_on | 40000 | 3.873880725 | 45402.939 |  |  |
| b2_window | lookahead_calibration_off | 40000 | 3.872631368 | 45852.708 |  |  |
| b2_window | lookahead_calibration_on | 40000 | 3.873880725 | 45402.939 |  |  |
| b2_loose | lookahead_calibration_off | 140000 | 13.703600409 | 106703.853 |  |  |
| b2_loose | lookahead_calibration_on | 140000 | 13.703600409 | 106703.853 |  |  |
| b3_tight | lookahead_calibration_off | 90000 | 8.813600775 | 67103.721 |  |  |
| b3_tight | lookahead_calibration_on | 90000 | 8.813600775 | 67103.721 |  |  |
| b3_window | lookahead_calibration_off | 90000 | 8.808789151 | 68835.905 |  |  |
| b3_window | lookahead_calibration_on | 90000 | 8.811013447 | 68035.159 |  |  |
| b4_tight | lookahead_calibration_off | 40000 | 3.884616287 | 41538.137 |  |  |
| b4_tight | lookahead_calibration_on | 40000 | 3.885865644 | 41088.368 |  |  |
| b4_window | lookahead_calibration_off | 90000 | 8.818697795 | 65268.794 |  |  |
| b4_window | lookahead_calibration_on | 90000 | 8.818697795 | 65268.794 |  |  |
| b4_loose | lookahead_calibration_off | 140000 | 13.719336240 | 101038.953 |  |  |
| b4_loose | lookahead_calibration_on | 140000 | 13.720861541 | 100489.845 |  |  |

## 计算

诊断评分展开 540，上限 6000。主试评分展开 155790，上限 250000。同状态续跑展开 0，上限 24000。
选择别名 2988，保留候选 10454。准入报价新算 13462，命中 273086。尾报价逐秒 153450。总耗时 174.1 秒。
去重少展开的是被期初守卫改写的标签。报价缓存只复用同一物理状态，不改变动作价值。这里的耗时不与旧航次的墙钟时间比作加速。

## 仍保留的边界

这只规范期初会被原守卫改写的标签。当前可准入、但本区间后段才触发返航的动作仍按原模型预测。冲突规则、tie 容差和终端公式没有改。另一架 UAV 在单机评分里等待。负结果和未跑的场景都不另做规则搜索。
