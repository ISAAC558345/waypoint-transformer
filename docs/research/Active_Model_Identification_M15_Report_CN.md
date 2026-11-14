# M1.5：无正收益业务分支的返航尾成本修正

性能判定：本轮未满足性能保留标准。

默认终端模式仍是 legacy。本轮候选 `return_tail_when_no_business` 只在显式打开时生效，不替换旧默认，也不把校准次数或效用变化解释成主动信息收益。

同开关比较没有交付量或净效用回退：False。至少一处净效用改善：True。物理完成未变差：True。备用电量计数未变差：True。
corrected_ON 相对 legacy_OFF 是否回退：True。相对 corrected_OFF 是否回退：True。主策略保留资格：False。
唯一回退场景是 b3_loose。corrected_ON 与 corrected_OFF 相对各自的 legacy 都少 20000 bit，净效用分别下降 1.931783280 与 1.936221054。首次请求分歧在第 6 个决策（180 秒）：UAV0 从 service:0 改为 return。其余 11 个场景 bit 不减、J 不降，其中多条航次因更早返航少耗能量。b3_loose 上 corrected_ON 还比 corrected_OFF 低 0.007916864 J。因此同开关保留标准不成立，corrected_ON 也不获得主策略资格。默认模式保持 legacy。
超过旧 ON 不能单独保留为主策略。corrected_ON 若低于 legacy_OFF 或 corrected_OFF，就不描述成优于当前最好执行。校准次数或效用变化不证明主动信息价值。
below_reserve 单列，不称为物理返航失败。J 的比较容差是绝对差 1e-9；按时比特按整数比较，不允许减少。评分 tie 容差不作为性能回退容差。

## 逐场景差值

| 场景 | 比较 | bit差 | J差 | 能耗差 J | 回退 | 物理变差 | 备用电量变差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b1_tight | corrected_ON-legacy_ON | 0 | 0.161836428 | -58261.114 | 0 |  | 0 |
| b1_tight | corrected_OFF-legacy_OFF | 0 | 0.014371654 | -5173.795 | 0 |  | 0 |
| b1_tight | corrected_ON-corrected_OFF | 0 | 0.000997264 | -359.015 | 0 |  | 0 |
| b1_tight | corrected_ON-legacy_OFF | 0 | 0.015368917 | -5532.810 | 0 |  | 0 |
| b1_window | corrected_ON-legacy_ON | 0 | 0.131162539 | -47218.514 | 0 |  | 0 |
| b1_window | corrected_OFF-legacy_OFF | 0 | 0.028743307 | -10347.591 | 0 |  | 0 |
| b1_window | corrected_ON-corrected_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b1_window | corrected_ON-legacy_OFF | 0 | 0.028743307 | -10347.591 | 0 |  | 0 |
| b1_loose | corrected_ON-legacy_ON | 0 | 0.102311628 | -36832.186 | 0 |  | 0 |
| b1_loose | corrected_OFF-legacy_OFF | 0 | 0.028743307 | -10347.591 | 0 |  | 0 |
| b1_loose | corrected_ON-corrected_OFF | 0 | 0.001220240 | -439.287 | 0 |  | 0 |
| b1_loose | corrected_ON-legacy_OFF | 0 | 0.029963548 | -10786.877 | 0 |  | 0 |
| b2_tight | corrected_ON-legacy_ON | 0 | 0.132049609 | -47537.859 | 0 |  | 0 |
| b2_tight | corrected_OFF-legacy_OFF | 0 | 0.017881234 | -6437.244 | 0 |  | 0 |
| b2_tight | corrected_ON-corrected_OFF | 0 | 0.001249358 | -449.769 | 0 |  | 0 |
| b2_tight | corrected_ON-legacy_OFF | 0 | 0.019130591 | -6887.013 | 0 |  | 0 |
| b2_window | corrected_ON-legacy_ON | 0 | 0.166379632 | -59896.668 | 0 |  | 0 |
| b2_window | corrected_OFF-legacy_OFF | 0 | 0.017881234 | -6437.244 | 0 |  | 0 |
| b2_window | corrected_ON-corrected_OFF | 0 | 0.001249358 | -449.769 | 0 |  | 0 |
| b2_window | corrected_ON-legacy_OFF | 0 | 0.019130591 | -6887.013 | 0 |  | 0 |
| b2_loose | corrected_ON-legacy_ON | 0 | 0.087369056 | -31452.860 | 0 |  | 0 |
| b2_loose | corrected_OFF-legacy_OFF | 0 | 0.046312579 | -16672.529 | 0 |  | 0 |
| b2_loose | corrected_ON-corrected_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b2_loose | corrected_ON-legacy_OFF | 0 | 0.046312579 | -16672.529 | 0 |  | 0 |
| b3_tight | corrected_ON-legacy_ON | 0 | 0.086517992 | -31146.477 | 0 |  | 0 |
| b3_tight | corrected_OFF-legacy_OFF | 0 | 0.038821201 | -13975.632 | 0 |  | 0 |
| b3_tight | corrected_ON-corrected_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b3_tight | corrected_ON-legacy_OFF | 0 | 0.038821201 | -13975.632 | 0 |  | 0 |
| b3_window | corrected_ON-legacy_ON | 0 | 0.127182418 | -45785.671 | 0 |  | 0 |
| b3_window | corrected_OFF-legacy_OFF | 0 | 0.041695532 | -15010.392 | 0 |  | 0 |
| b3_window | corrected_ON-corrected_OFF | 0 | 0.002224295 | -800.746 | 0 |  | 0 |
| b3_window | corrected_ON-legacy_OFF | 0 | 0.043919828 | -15811.138 | 0 |  | 0 |
| b3_loose | corrected_ON-legacy_ON | -20000 | -1.931783280 | -24558.019 | 1 |  | 0 |
| b3_loose | corrected_OFF-legacy_OFF | -20000 | -1.936221054 | -22960.421 | 1 |  | 0 |
| b3_loose | corrected_ON-corrected_OFF | 0 | -0.007916864 | 2850.071 | 1 |  | 0 |
| b3_loose | corrected_ON-legacy_OFF | -20000 | -1.944137917 | -20110.350 | 1 |  | 0 |
| b4_tight | corrected_ON-legacy_ON | 0 | 0.146412615 | -52708.541 | 0 |  | 0 |
| b4_tight | corrected_OFF-legacy_OFF | 0 | 0.014371654 | -5173.795 | 0 |  | 0 |
| b4_tight | corrected_ON-corrected_OFF | 0 | 0.001249358 | -449.769 | 0 |  | 0 |
| b4_tight | corrected_ON-legacy_OFF | 0 | 0.015621011 | -5623.564 | 0 |  | 0 |
| b4_window | corrected_ON-legacy_ON | 0 | 0.119955409 | -43183.947 | 0 |  | 0 |
| b4_window | corrected_OFF-legacy_OFF | 0 | 0.032252887 | -11611.039 | 0 |  | 0 |
| b4_window | corrected_ON-corrected_OFF | 0 | 0.000000000 | 0.000 | 0 |  | 0 |
| b4_window | corrected_ON-legacy_OFF | 0 | 0.032252887 | -11611.039 | 0 |  | 0 |
| b4_loose | corrected_ON-legacy_ON | 0 | 0.103712512 | -37336.504 | 0 |  | 0 |
| b4_loose | corrected_OFF-legacy_OFF | 0 | 0.032252887 | -11611.039 | 0 |  | 0 |
| b4_loose | corrected_ON-corrected_OFF | 0 | 0.001525300 | -549.108 | 0 |  | 0 |
| b4_loose | corrected_ON-legacy_OFF | 0 | 0.033778188 | -12160.148 | 0 |  | 0 |

## 分档最差值

下面先给每档最差差值，再给合计。最差值为负就是回退，不拿合计掩盖。

| 比较 | 档 | 条数 | J最小差 | J合计 | bit最小差 | 回退条数 | 物理变差 | 备用电量变差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| corrected_ON-legacy_ON | tight | 4 | 0.086517992 | 0.526816645 | 0 | 0 | 0 | 0 |
| corrected_OFF-legacy_OFF | tight | 4 | 0.014371654 | 0.085445742 | 0 | 0 | 0 | 0 |
| corrected_ON-corrected_OFF | tight | 4 | 0.000000000 | 0.003495979 | 0 | 0 | 0 | 0 |
| corrected_ON-legacy_OFF | tight | 4 | 0.015368917 | 0.088941721 | 0 | 0 | 0 | 0 |
| corrected_ON-legacy_ON | window | 4 | 0.119955409 | 0.544679998 | 0 | 0 | 0 | 0 |
| corrected_OFF-legacy_OFF | window | 4 | 0.017881234 | 0.120572961 | 0 | 0 | 0 | 0 |
| corrected_ON-corrected_OFF | window | 4 | 0.000000000 | 0.003473653 | 0 | 0 | 0 | 0 |
| corrected_ON-legacy_OFF | window | 4 | 0.019130591 | 0.124046613 | 0 | 0 | 0 | 0 |
| corrected_ON-legacy_ON | loose | 4 | -1.931783280 | -1.638390084 | -20000 | 1 | 0 | 0 |
| corrected_OFF-legacy_OFF | loose | 4 | -1.936221054 | -1.828912280 | -20000 | 1 | 0 | 0 |
| corrected_ON-corrected_OFF | loose | 4 | -0.007916864 | -0.005171323 | 0 | 1 | 0 | 0 |
| corrected_ON-legacy_OFF | loose | 4 | -1.944137917 | -1.834083602 | -20000 | 1 | 0 | 0 |

## 新航次原始结果

| 场景 | 方法 | bit | J | 能耗 J | 请求校准 | 执行校准 | 修复改派校准 | 首服务0 | 时刻0 s | 首服务1 | 时刻1 s | 未返航 | 未落地 | 耗尽 | 返航失败 | 低于备用 | none触发 | 首次分歧时段 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_tight | lookahead_calibration_off | 40000 | 3.895623457 | 37575.556 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 59 | 2 |
| b1_tight | lookahead_calibration_on | 40000 | 3.896620721 | 37216.541 | 1 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 59 | 2 |
| b1_window | lookahead_calibration_off | 90000 | 8.829704965 | 61306.212 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| b1_window | lookahead_calibration_on | 90000 | 8.829704965 | 61306.212 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| b1_loose | lookahead_calibration_off | 140000 | 13.747448566 | 90918.516 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 5 |
| b1_loose | lookahead_calibration_on | 140000 | 13.748668807 | 90479.230 | 1 | 1 | 1 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 5 |
| b2_tight | lookahead_calibration_off | 40000 | 3.872631368 | 45852.708 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 59 | 2 |
| b2_tight | lookahead_calibration_on | 40000 | 3.873880725 | 45402.939 | 1 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 59 | 2 |
| b2_window | lookahead_calibration_off | 40000 | 3.872631368 | 45852.708 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 59 | 2 |
| b2_window | lookahead_calibration_on | 40000 | 3.873880725 | 45402.939 | 1 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 59 | 2 |
| b2_loose | lookahead_calibration_off | 140000 | 13.703600409 | 106703.853 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 4 |
| b2_loose | lookahead_calibration_on | 140000 | 13.703600409 | 106703.853 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 4 |
| b3_tight | lookahead_calibration_off | 90000 | 8.813600775 | 67103.721 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| b3_tight | lookahead_calibration_on | 90000 | 8.813600775 | 67103.721 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| b3_window | lookahead_calibration_off | 90000 | 8.808789151 | 68835.905 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| b3_window | lookahead_calibration_on | 90000 | 8.811013447 | 68035.159 | 1 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| b3_loose | lookahead_calibration_off | 120000 | 11.700224886 | 107919.041 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 52 | 6 |
| b3_loose | lookahead_calibration_on | 120000 | 11.692308022 | 110769.112 | 2 | 2 | 2 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 52 | 6 |
| b4_tight | lookahead_calibration_off | 40000 | 3.884616287 | 41538.137 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 59 | 2 |
| b4_tight | lookahead_calibration_on | 40000 | 3.885865644 | 41088.368 | 1 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 59 | 2 |
| b4_window | lookahead_calibration_off | 90000 | 8.818697795 | 65268.794 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| b4_window | lookahead_calibration_on | 90000 | 8.818697795 | 65268.794 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| b4_loose | lookahead_calibration_off | 140000 | 13.719336240 | 101038.953 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 5 |
| b4_loose | lookahead_calibration_on | 140000 | 13.720861541 | 100489.845 | 1 | 1 | 1 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 5 |

## 成本修正证据

六个快照的 legacy 重评最大绝对差见下表。评分展开 2400 次，缓存命中 1392 次，尾分段进入积分的片段计数 666。

| 快照 | 类别 | legacy Q绝对差 | 正业务不一致 | 排序变化 | 新联合动作 | 修复仍请求校准 |
| --- | --- | --- | --- | --- | --- | --- |
| b1_tight | conflict_repair | 0.000 | 0 | 0 | calibrate:None|service:0 | 1 |
| b3_window | conflict_repair | 0.000 | 0 | 0 | service:0|calibrate:None | 1 |
| b1_loose | conflict_repair | 0.000 | 0 | 0 | calibrate:None|service:1 | 1 |
| b1_window | single_score | 0.000 | 0 | 4 | service:1|service:3 | 0 |
| b4_window | single_score | 0.000 | 0 | 4 | service:1|service:3 | 0 |
| b2_loose | single_score | 0.000 | 0 | 7 | return:None|service:3 | 0 |

三条 single_score 快照的 C 与替代动作。Q 差减去终端差应接近 0，说明尾项进入的是正式评分而不是旁路显示。

| 快照 | 动作 | R30 | 旧终端 | 新终端 | 新Q | 后验分位 | 落地时间 s | 后验推进 J | 地面辅助 J | 通信 J | Q差减终端差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_window | calibrate:None | -0.027175913 | 0.000000000 | -0.024105794 | -0.051281707 | 1.040218 | 43.0 | 6825.147 | 1637.919 | 0.939 | 0.000000000 |
| b1_window | return:None | -0.027583952 | 0.000000000 | -0.010091819 | -0.037675771 | 1.039996 | 12.0 | 1904.116 | 1668.935 | 0.939 | 0.000000000 |
| b4_window | calibrate:None | -0.029626043 | 0.000000000 | -0.027961314 | -0.057587357 | 1.251760 | 43.0 | 8213.134 | 1637.919 | 0.939 | 0.000000000 |
| b4_window | return:None | -0.030119616 | 0.000000000 | -0.011167682 | -0.041287299 | 1.251538 | 12.0 | 2291.427 | 1668.935 | 0.939 | -0.000000000 |
| b2_loose | calibrate:None | -0.033017151 | 0.000000000 | -0.069946988 | -0.102964140 | 1.294229 | 114.0 | 23104.011 | 1506.852 | 0.905 | 0.000000000 |
| b2_loose | return:None | -0.033640864 | 0.000000000 | -0.050131539 | -0.083772404 | 1.294081 | 79.0 | 16110.449 | 1541.868 | 0.905 | 0.000000000 |

新分支动作差的局部细网格使用已登记的 31 外层、15 内层。误差界是两边绝对误差相加。未分辨的差距保留原 tie 规则，不加大扫描。

| 快照 | 状态 | 15点差距 | 31点差距 | 合并误差界 |
| --- | --- | --- | --- | --- |
| b1_window | resolved | -0.013605937 | -0.013605937 | 0.000000000 |
| b4_window | resolved | -0.016300058 | -0.016300058 | 0.000000000 |
| b2_loose | resolved | -0.019191736 | -0.019191736 | 0.000000000 |

## 计算代价

快照评分展开 2400 次，上限 6000。主试评分展开 203220 次，上限 250000。
主试缓存命中 43800，代数积分 210262，代数耗时 7.207 秒，执行步数 43200，规划步数 6096600。
名义返航报价 1894 次，报价缓存命中 201326，逐秒模拟 152730 秒，解析地面区间 55635。解析区间的一到两个采样已经计入逐秒步数，定价时不另加一遍。
总耗时 271.5 秒。

## 局部近似仍有的边界

这只替换旧 none 分支的零值。被零下限舍弃的负收益业务只记录最大值，不在本轮重排。正收益报价里已经包含的回传、返航和落地不再加一份费用。
尾过程按 RETURN 飞行并落地，不把缓存回传当成纯返航，也不给缓存虚构交付收益。另一架 UAV 在报价里等待；它的能耗不进这架 UAV 的终端。
名义轨迹在充足电量下生成，再用当前电量把总能耗截断。截断后的代价是实际耗尽的能量，没有额外惩罚，也不假设已经成功返航。备用电量仍只是准入约束。
落地之后的重复地面区间用同一采样相乘。任务时钟停在 1800 秒。这个修正不是完整的最优终端，也没有改变默认模式。
