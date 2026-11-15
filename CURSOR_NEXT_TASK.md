# Cursor 下一步：W2.1 已完成，本轮停止

`research/ocean-observation-isolated` 上的 W2.1 已完成。逆风 300 秒时 1 号机已经落地，旧准入按脚本结束前的 31.137 J 认为可以回传；仅待机到 1800 秒的任务末上界是 3409.496 J，低于 4000 J 备用。显式地面报价去掉了虚构的再次飞行，但没有放行回传。六臂与 W2 相同。默认守卫不变。

报告：[W2.1 结果](docs/research/Wind_Observation_W21_Report_CN.md)。结果目录：`results/wind_observation_w21/w21_20261115/`。复现：`$env:PYTHONPATH = "src"; py -3 -m wt.active_model_identification.run_w21`。

本轮停止。不要把该报价并入默认守卫，也不要据此开始下一项全程能量估计、风场或训练工作。下一条任务以新的指令推送为准。
