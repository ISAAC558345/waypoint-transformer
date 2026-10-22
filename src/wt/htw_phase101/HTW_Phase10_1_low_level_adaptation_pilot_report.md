# HTW Phase-10.1 L3-B 六机目标条件低层适配与单种子试验

只训练低层。未训练高层。未加载旧五机权重。未改 L3-B 物理/SINR/关联/原始 payload。未开 confirmation。未开新 Round。

1. 新六机低层：是。禁止路径 `E:\srdp\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\waypoint_transformer_official\src\wt\htw_phase63\outputs\checkpoints\low_level_c05_d005.pt`。fresh init seed=0。
2. 适配：max_uavs=6，K=12 idle token=12，option one-hot/相对位置/距离/进度/valid，hold 与剩余时隙进入 global；维度由 schema 决定。
3. 高层冻结：high_model_seed=0，SHA 训练前后一致=True。
4. smoke 通过：True。smoke 成绩不用于判断优劣。
5. 实际训练：100 updates × 48 × 4 = 19200 env transitions；总墙钟见 runtime CSV。
6. design val 配对 payload：B0 mean=8500000.0，B1 mean=7250000.0。这不是 9.4.5 的 6.95e6 confirmation 数字。
7. 真实完成块 B0/B1 mean=8.5/7.25；从未完成 UAV 数 B0/B1=20/26。
8. 进度清零 B0/B1=9.125/11.625；飞行距离 B0/B1=2802.25/1675.628029175367。
9. 学习趋势：late 4420000 vs early 110000 (completed-episode payload during training; sparse)。init eval payload=0.0 vs final 7250000.0。
10. 与 fly-toward 差值 mean Δ=-1250000.0，ratio=0.8663194444444444。A 可运行不等于 C 超过规则。
11. 下一步：建议下一轮多 seed 复核，本轮不自动执行。

图：{'learning_curves': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase101\\outputs\\pilot\\phase101_learning_curves.png', 'paired_payload': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase101\\outputs\\pilot\\phase101_paired_payload.png'}

