# W4.1 已完成：240 秒回传是准入越过 4000 J

只在 `research/ocean-observation-isolated` 记录。基线是 0dc1655 的 W4 归档。本轮没有改默认调度，也没有开始下一阶段。

240 秒两臂的位置、缓存和信念一致，电量只差已支付的 0.010386 J。没有报文时，回传的 0.95 分位末端电量是 3911.725 J，被挡在 4000 J 下面；报文把它抬到 5227.488 J，70000 bit 的分数被选中。守卫没有改写请求。

5202.704 J 按航段对上，未解释差额可以忽略。同一报文信息下改回返航反而多耗 212.468 J。不据此修改备用阈值、风权或报文时限。

报告见 [W4.1 结果](docs/research/Wind_Observation_W41_Report_CN.md)，结果在 `results/wind_observation_w41/w41_20261115/`。复现：`$env:PYTHONPATH="src"; py -3 -m wt.active_model_identification.run_w41`。

下一份推送到达前停在这里。用户亲自运行并明确同意后方可并入主分支，修改其他分支前先询问用户。
