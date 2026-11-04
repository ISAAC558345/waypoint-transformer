"""Chinese M0 report rendered from the saved study dictionary."""

from __future__ import annotations

from typing import Optional


def _j(value: Optional[float]) -> str:
    if value is None:
        return "未定义"
    if abs(value) < 10.0:
        return f"{value:.3f} J"
    return f"{value:.1f} J"


_REASON = {
    "feasible": "可行",
    "timescale": "超出 1800 s",
    "energy_reserve": "余量不足",
    "track_infeasible": "航迹不可行",
    "collection_link": "采集链路不闭合",
    "shore_session_or_command_link": "岸站会话不成立",
    "direct_link": "直传不闭合",
    "shore_window": "窗口容不下业务",
}
_DOM = {
    "flight_J": "飞行",
    "return_J": "返航",
    "wait_J": "等待",
    "handshake_J": "握手",
    "payload_radio_J": "载荷无线电",
}


def _n(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "未定义"
    return f"{value:.{digits}f}"


def _yes(flag: bool) -> str:
    return "可行" if flag else "不可行"


def _reasons(route: dict) -> str:
    labels = [_REASON.get(reason, reason) for reason in route["reasons"]]
    return "、".join(labels)


def render_report(study: dict, verification: Optional[dict]) -> str:
    conclusion = study["conclusion"]["conclusion"]
    reason = study["conclusion"]["reason"]
    facts = study["facts"]
    packets = study["packets"]
    links = study["links"]
    cases = study["cases"]
    wind = study["wind_illustration"]
    example = study["observation_example"]
    manifest = study["manifest"]
    direct_ok = sum(case["direct"]["delivery_feasible"] for case in cases)
    far_ok = sum(case["farthest"]["delivery_feasible"] for case in cases)
    tour_ok = sum(case["tour"]["delivery_feasible"] for case in cases)
    lines = [
        "# 海洋观测 M0 核算报告",
        "",
        f"结论：**{conclusion}**。{ _reason_cn(reason) }",
        "",
        "本报告只核算一种带测量方差的有效波高包、一个固定岸站，以及登记的 24 组解析案例。解析路线不是环境 episode，也不是策略得分。`performance_pass=false`，`adopted=false`，`release_eligible=false`。",
        "",
        "## 1 范围、基线和预算",
        "",
        f"- 设计基准 `{manifest['design_baseline']}`，正式来源 `{manifest['formal_source_commit']}`。",
        f"- 交接提交 `{manifest['handoff_commit']}` 与 E19 结果 `{manifest['e19_result_commit']}` 只作只读引用，没有重跑，也没有把 E19 代码并入本分支。",
        f"- 实现提交 `{manifest['code_commit']}`。正式 checkpoint 文件 SHA-256 为 `{manifest['formal_checkpoint_sha256']}`，加载次数 {manifest['policy_model_loads']}，推理次数 {manifest['policy_inferences']}。",
        f"- 训练更新 {manifest['training_updates']}，新环境 episode 步数 {manifest['new_environment_episode_steps']}，历史 holdout 重跑 {manifest['historical_holdout_reruns']}，大体量下载 {manifest['large_external_dataset_downloads']}。",
        f"- 解析案例 {manifest['analytical_cases']} 组，官方核算运行 {manifest['analytical_runs']} 次。现有回归测试另行记账，不作为本研究证据。",
        "",
        "输入哈希、参数表、链路表、案例清单和接口草案分别在：",
        "",
        "- `results/ocean_observation_m0/run_manifest.json`",
        "- `results/ocean_observation_m0/parameter_table.json`",
        "- `results/ocean_observation_m0/packet_sizes.json`",
        "- `results/ocean_observation_m0/link_budget.json`",
        "- `results/ocean_observation_m0/case_manifest.json`",
        "- `results/ocean_observation_m0/case_results.json`",
        "- `results/ocean_observation_m0/case_results.csv`",
        "- `results/ocean_observation_m0/observation_example.json`",
        "- `results/ocean_observation_m0/timelines.json`",
        "- `results/ocean_observation_m0/coverage_drift.json`",
        "- `results/ocean_observation_m0/interface_draft.json`",
        "- `results/ocean_observation_m0/wind_illustration.json`",
        "",
        "## 2 参数与单位",
        "",
        f"旋翼功率按 Zeng、Xu、Zhang 的式 (6) 用空速计算。由原始量算出的剖面功率 P0 为 {_n(facts['P0_W'], 2)} W，诱导功率 Pi 为 {_n(facts['Pi_W'], 2)} W，悬停功率为 {_n(facts['hover_W'], 2)} W。原始量来自与该式配套的公开数值例子；本次没有重读 PDF 表格，因此这组原始量是文献复现合同，不是海上实测。",
        "",
        f"零风条件下，1 至 20 m/s 整数空速里，每米推进能量的最小值在 {facts['cruise_airspeed_m_s']:.0f} m/s，对应推进功率 {_n(facts['cruise_power_W'], 2)} W。这个速度在看 24 组案例之前由功率曲线选定，不在网格端点。",
        "",
        f"可用能量合同是悬停功率乘 1200 s，即 {_j(facts['usable_energy_J'])}（{_n(facts['usable_energy_J'] / 3600.0, 2)} Wh）。登记余量 20%，单机可消耗上限 {_j(facts['energy_limit_J'])}。这不是一块测过的电池，任务中不充电。",
        "",
        "无线电使用 SX1276 数据手册的同一工作点：Band 1、SF7、125 kHz、编码率 4/6、RFO 引脚 +13 dBm、发射电流 28 mA、3.3 V。灵敏度取 RFS_L125_HF 的 SF7 典型值 -123 dBm。接收电流取 LnaBoost 关闭时 Band 1、125 kHz 的 10.3 mA，低于灵敏度行所对应的 LnaBoost 状态，因此是接收功耗的下界。另外只加一个 6 dB 合成实施裕量。链路成功是阈值判断，速率是帧比特除以空中时间，没有再乘一次丢包率。",
        "",
        "天线增益、岸高、浮标天线高度、相关长度、漂移三分量和回传值班周期都标成合成假设。旧环境的 16 m 服务半径和 1 Mbit block 没有进入链路或产品。参数的来源类别、单位和缺失状态见参数表。",
        "",
        "## 3 业务量",
        "",
        "下表的长度是编码器实际输出，不是事先写定的字节数。",
        "",
        "| 帧 | 应用载荷字节 | 魔数与长度 | CRC32 | 成帧字节 | 成帧比特 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in (("product", "有效波高包"), ("command", "任务命令"), ("ack", "确认"), ("telemetry", "遥测")):
        row = packets[name]
        lines.append(
            f"| {label} | {row['application_bytes']} | {row['magic_and_length_bytes']} | {row['crc32_bytes']} | {row['framed_bytes']} | {row['framed_bits']} |"
        )
    raw_bits = study["raw_record_bits"]
    product_bits = packets["product"]["framed_bits"]
    lines.extend(
        [
            "",
            f"对照用的原始记录是 3 通道 × 4 Hz × 16 bit × 1200 s = {raw_bits} bit（{raw_bits // 8} byte）。它只作对照，不是发出去的第一版产品。成帧后的有效波高包是 {product_bits} bit，约为这条原始记录的 1/{raw_bits / product_bits:.1f}。",
            "",
            "暖启动窗口 [-1200, 0] 在 t=0 生成每标一包，t=1200 再生成一包。评估地平线 1800 s 内每标两包。初始缓存是 t=0 的那一包。同一 `(buoy_id, seq)` 只融合一次。窗口结束早于接收时间超过 3600 s 的包只记录、不融合，采样时间不改写成送达时间。部分帧不融合。M0 不实现断点续传的第二阶段；账本把部分接收单独记为未融合。",
            "",
            "## 4 观测算子示例",
            "",
            f"合成场 `Hs = 1.2 + 0.3 sin(2πx/4000) cos(2πy/4000)` 不是海浪模型。示例浮标在 t=0 位于 (500, 500) m，窗口内按登记漂移回推，每 60 s 取一次样，共 {example['n_samples']} 个样本。测量噪声标准差 0.05 m，种子 {example['seed']}。包内均值 {_n(example['mean_hs_m'], 4)} m，均值的测量方差 {_n(example['measurement_variance_m2'], 6)} m²。",
            "",
            f"接收前岸站保持先验均值 {_n(example['belief_before']['mean_m'], 3)} m、方差 {_n(example['belief_before']['variance_m2'], 3)} m²。完整包在 t=30 s 通过校验后，后验均值 {_n(example['belief_after']['mean_m'], 4)} m、方差 {_n(example['belief_after']['variance_m2'], 6)} m²。窗口结束时刻仍是 {example['window_end_s']} s。调度器在这次 `receive` 之前读不到该包的值。这个标量更新不是时空海况图，也不能当成任务收益。",
            "",
            "## 5 链路预算",
            "",
            f"三跳和直传使用同一发射功率、带宽和扩频因子。产品空中时间 {_n(links['product_airtime_s'], 4)} s，成帧速率 {_n(links['product_rate_bps'], 1)} bit/s。丢包模型为空。",
            "",
            "| 几何 | 距离 m | 接收 dBm | 相对 -117 dBm 阈值的裕量 dB | 地平线内 | 闭合 |",
            "| --- | ---: | ---: | ---: | --- | --- |",
            _link_row("浮标正上方 100 m 对无人机", links["overhead_collection"]),
            _link_row("任务区远角浮标对岸站", links["domain_corner_direct"]),
            _link_row("任务区远角无人机对岸站", links["domain_corner_uav"]),
            _link_row("落地后的无人机对岸站", links["landed_backhaul"]),
            "",
            f"1 m 浮标天线到 10 m 岸站天线的 4/3 地球无线电地平线约 {_n(links['horizon_buoy_shore_m'] / 1000.0, 2)} km。同一对增益下，Friis 裕量降到 0 dB 的距离约 {_n(links['friis_zero_margin_buoy_shore_m'] / 1000.0, 1)} km。2 km 方区远在地平线之内，远角直传仍有约 {_n(links['domain_corner_direct']['margin_db'], 1)} dB 裕量。限制远距离直传的将是地平线，不是这次登记的发射功率。海面反射没有再加一项衰落，避免和阈值裕量重复扣减；这也是未建模的缺口，不是已经证明的海上链路。",
            "",
            "命令、确认和遥测走同一物理层。遥测按空中时间计能量，不额外悬停一个运动步。间歇回传区间里，岸站接收窗口在最坏相位下于飞行期间关闭，空中遥测记为未送达；这不把产品再丢一次。",
            "",
            "## 6 航程、风和能量",
            "",
            f"平静空气里 1000 m 直线的地速等于空速 {_n(wind['calm']['ground_speed_m_s'], 2)} m/s，推进能量 {_j(wind['calm']['propulsion_J'])}。同一空速、8 m/s 东风时，向东顺风地速 {_n(wind['tailwind']['ground_speed_m_s'], 2)} m/s、推进 {_j(wind['tailwind']['propulsion_J'])}；向西逆风地速 {_n(wind['headwind']['ground_speed_m_s'], 2)} m/s、推进 {_j(wind['headwind']['propulsion_J'])}。往返推进 {_j(wind['round_trip_propulsion_J'])}，小于两倍逆风 {_j(wind['two_headwind_propulsion_J'])}。功率用空速，时间用地速。每条腿另加一段加速和一段减速，能量按悬停功率计，不为负。",
            "",
            "每架无人机在起飞前于地面等待本制度的岸站会话并接收命令；访问浮标时只为本地时隙悬停；回传在落地后的地面等待中完成，不在空中悬停等窗口。间歇制度的采集链路不套用岸站关闭窗口。两包产品对应两次同样的架次，中间不充电。单机两次架次的能量必须留下 20% 余量。",
            "",
            f"24 组里，浮标直传可行 {direct_ok}/24，最远单标采集回传可行 {far_ok}/24，分标巡航可行 {tour_ok}/24。不可行的结果全部保留。",
            "",
            "| 案例 | 直传 | 最远单标 | 分标巡航 | 巡航主导成本 | 最贵单机两次能量 | 直传两次能量 |",
            "| --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for case in cases:
        far = case["farthest"]
        tour = case["tour"]
        direct = case["direct"]
        max_uav = _max_uav(tour)
        lines.append(
            "| {case_id} | {direct} | {far} | {tour} | {dom} | {uav} | {direct_j} |".format(
                case_id=case["case_id"],
                direct=_yes(direct["delivery_feasible"]),
                far=_reasons(far),
                tour=_reasons(tour),
                dom=_DOM.get(case["tour_dominant"], "无"),
                uav=_j(max_uav),
                direct_j=_j(direct["episode"]["total_J"] if direct.get("episode") else None),
            )
        )
    reserve = [case for case in cases if "energy_reserve" in case["tour"]["reasons"]]
    lines.extend(["", _reserve_paragraph(reserve, facts), "", _regime_paragraph(cases), ""])
    lines.extend(_coverage_section(cases))
    lines.extend(_questions(cases, links, example))
    lines.extend(_rights(study["timelines"]))
    lines.extend(_interfaces())
    lines.extend(_verification(verification))
    lines.extend(_revision(links))
    lines.extend(_decisions())
    lines.extend(
        [
            "## 14 停止",
            "",
            "M0 停在这份报告。不开始最小闭环，不训练，不搜索标签权重，不把未用完的其他实验步数转进来，也不执行第 12 节的修订。",
            "",
        ]
    )
    return "\n".join(lines)


def _reason_cn(reason: str) -> str:
    if "direct buoy-to-shore" in reason:
        return "登记的通信宽裕案例里，浮标直传全部可行，当前产品和几何不需要无人机摆渡才能把包送到岸站。"
    return reason


def _link_row(label: str, link: dict) -> str:
    inside = "是" if link["distance_m"] <= link["horizon_m"] else "否"
    closed = "是" if link["closes"] else "否"
    return f"| {label} | {_n(link['distance_m'], 1)} | {_n(link['received_dbm'], 1)} | {_n(link['margin_db'], 1)} | {inside} | {closed} |"


def _max_uav(route: dict) -> Optional[float]:
    if not route.get("defined"):
        return None
    return max(vehicle["uav_J"] for vehicle in route["vehicles"]) * 2.0


def _reserve_paragraph(reserve: list, facts: dict) -> str:
    if not reserve:
        return "没有任何登记巡航因为余量失败。"
    case = reserve[0]
    spent = _max_uav(case["tour"])
    usable = facts["usable_energy_J"]
    limit = facts["energy_limit_J"]
    return (
        f"唯一因余量失败的巡航是 `{case['case_id']}`。最贵的一架无人机两次架次消耗 {_j(spent)}，"
        f"可用能量 {_j(usable)}，20% 余量所允许的上限是 {_j(limit)}。"
        f"剩余 {_j(usable - spent)} 仍为正，但低于要求保留的 {_j(usable - limit)}。"
        "这架无人机在模型里能够沿航迹返回，失败的是登记余量，不是负能量或瞬移。"
        "该案例保留。没有在看到这个差值之后改电池、改速度或改时隙。"
    )


def _regime_paragraph(cases: list) -> str:
    bits = []
    for regime in (
        "communication_abundant",
        "joint_access_and_communication_constraint",
        "intermittent_backhaul",
    ):
        subset = [case for case in cases if case["regime"] == regime]
        sample = next(case for case in subset if case["n_uav"] == 2 and case["layout"] == "dispersed" and case["wind_name"] == "calm")
        episode = sample["farthest"]["episode"]
        bits.append(
            f"- `{regime}`：2 机 4 标、无风、分散布局的最远单标系统能量中，飞行 {_j(episode['flight_J'])}，返航 {_j(episode['return_J'])}，等待 {_j(episode['wait_J'])}，握手 {_j(episode['handshake_J'])}，载荷无线电 {_j(episode['payload_radio_J'])}，遥测 {_j(episode['telemetry_J'])}。主导项是 {sample['farthest_dominant']}。同案例直传两次共 {_j(sample['direct']['episode']['total_J'])}，主导项是 {sample['direct_dominant']}。"
        )
    return "同一条最远单标航迹在三种制度下的能量拆分如下。飞行和返航不随制度改变；制度改变的是等待。\n\n" + "\n".join(bits)


def _coverage_section(cases: list) -> list:
    lines = ["## 7 覆盖、相关和漂移", ""]
    seen = set()
    for case in cases:
        if case["wind_name"] != "calm" or case["regime"] != "communication_abundant":
            continue
        key = (case["layout"], case["n_buoy"])
        if key in seen:
            continue
        seen.add(key)
        cov = case["coverage"]
        lines.append(
            f"- {case['layout']}、{case['n_buoy']} 个浮标：100 m 网格 {cov['grid_points']} 点中，{cov['covered_points']} 点落在合成相关长度 1000 m 内，覆盖比例 {_n(cov['covered_fraction'], 4)}，只靠先验的格点 {cov['prior_only_points']}。浮标对 {cov['pairs_within_correlation_length']}/{cov['pairs']} 对的间距不超过 1000 m。"
        )
    calm = next(case for case in cases if case["wind_name"] == "calm" and case["n_buoy"] == 12 and case["layout"] == "dispersed" and case["regime"] == "communication_abundant")
    windy = next(case for case in cases if case["wind_name"] == "wind_8mps_east" and case["n_buoy"] == 12 and case["layout"] == "dispersed" and case["regime"] == "communication_abundant")
    exits = sum(case["drift"]["exited"] for case in cases)
    lines.extend(
        [
            "",
            f"漂移把海流、Stokes 漂移和风漂移分成三项，波浪没有在海流里再计一次。12 个分散浮标在 1800 s 内，无风时东向和北向位移都是 {_n(_component(calm, 0), 1)} m；8 m/s 东风时东向 {_n(_component(windy, 0), 1)} m、北向 {_n(_component(windy, 1), 1)} m。24 组登记浮标离开任务域的次数是 {exits}。位置不反弹。航线仍用 t=0 的登记坐标；漂移只作为覆盖诊断。",
            "",
            "1000 m 是合成诊断尺度，不是估计出来的海洋相关长度。在这个尺度下，分散布局已经盖住整个 2 km 方区，两个簇则在簇内高度重复、在簇外留下先验区。2 km 方区大约只有两个这样的长度，空间差异很弱。",
            "",
        ]
    )
    return lines


def _component(case: dict, axis: int) -> float:
    return max(item["end_m"][axis] - item["start_m"][axis] for item in case["drift"]["positions"])


def _questions(cases: list, links: dict, example: dict) -> list:
    intermittent = [case for case in cases if case["regime"] == "intermittent_backhaul"]
    return [
        "## 8 必须回答的问题",
        "",
        f"直接浮标回传在 24 组里全部闭合。远角裕量约 {_n(links['domain_corner_direct']['margin_db'], 1)} dB，直传能量是数焦耳到数百焦耳，而一次最远摆渡是十万焦耳量级。直传应当作为对照，并且在这个 2 km、52 字节产品的合同下已经足够完成送达。这不是把带宽调小或把包加大之后的结论。",
        "",
        "摆渡的焦耳成本主要来自飞行和返航。握手和载荷无线电都低于 1 J。等待只在共同受限制度里因为每标 30 s 悬停而升到约 10 kJ，仍然小于飞行。间歇制度把等待放在地面，焦耳增加有限，但最坏关闭时间使两次架次超出 1800 s，8 组间歇摆渡都因此不可行。直传在间歇制度下仍然可行，因为浮标等待时不需要悬停。制度改变的是时间和等待的承担者，不改变“飞行主导摆渡、无线电可忽略”这个排序。",
        "",
        "12 个分散点在合成 1000 m 相关长度下盖住全部评估格点，所以没有只靠先验的格点；12 个点的两个簇仍有只靠先验的格点，同时簇内多数点对都落在相关长度内。冗余、过时和空洞不能从这次合成尺度推广到真实海况。过时规则是 3600 s，示例包的采样结束时刻是 "
        f"{example['window_end_s']} s，接收时刻是 30 s，二者没有被写成同一个时间。",
        "",
        "岸站在完整、校验通过、未见过且未过期的包到达时才更新海况。命令到达无人机后无人机激活；确认未回到岸站时，岸站保持待确认，不能把任务当成一定没有执行。命令没有到达时，无人机保持上一份已激活且未过期的任务，否则执行固定返航。间歇窗口期间，岸站没有新的空中遥测。失联后的产品仍留在无人机缓存里，直到下一次打开的回传窗口；窗口关闭本身不制造第二次丢包。",
        "",
        "候选集合按接口草案包含每一个浮标、回传、返航和等待，不用速率前几名做入口。本次登记几何里所有浮标的直传和采集链路都闭合，没有出现“速率低但有价值”的浮标。这个事实也说明当前几何不能区分信息价值。",
        "",
        "物理不可行的登记结果有两类，都保留：8 组间歇回传的摆渡超出 1800 s；1 组 8 m/s 东风、共同受限、6 机 12 标分散巡航超出 20% 能量余量。直传没有不可行组。侧风大于空速的航迹在模型里没有地速，24 组所选 18 m/s 空速对 8 m/s 风仍能保持航迹。",
        "",
    ]


def _rights(timelines: dict) -> list:
    lines = [
        "## 9 信息权限和四个时序",
        "",
        "| 主体 | 能用的数据 | 得不到的数据 | 失败时 |",
        "| --- | --- | --- | --- |",
        "| 岸站 | 已完整校验且未重复、未过期的包；已确认的命令版本；窗口打开时收到的遥测 | 未发出的测量、未来场、未确认命令的执行事实 | 待确认保持为待确认；地图不更新 |",
        "| 无人机 | 本机位置、本机电量、已激活且未过期的任务、缓存的包 | 岸站尚未送达的新任务；其他无人机的缓存 | 保持最后任务，到期或触及返航条件则返航 |",
        "| 浮标 | 本机窗口样本和已编码的包 | 其他浮标；岸站地图 | 没有浮标间转发 |",
        "| 离线评价器 | 真值场和账本 | 不能把真值写进岸站收件箱 | 评价未运行 |",
        "",
        "四个事件都由同一套接收和命令函数生成：",
        "",
    ]
    for name, events in timelines.items():
        lines.append(f"- `{name}`：")
        for event in events:
            lines.append(f"  - t={event.get('t_s', 0)} s，{event.get('actor', '')}，{event.get('fact', '')}")
    lines.append("")
    return lines


def _interfaces() -> list:
    return [
        "## 10 后继接口",
        "",
        "草案在 `results/ocean_observation_m0/interface_draft.json`。节点数量来自案例配置。可以沿用的是按配置决定节点数、以及把字节、时间、能量和重复接收分行记账的方式。必须换掉的是 16 m 服务半径、1 Mbit block、正式策略权重，以及把旧海上环境的转移当成新的海况真值。",
        "",
        "建议的最小闭环是一个固定风、固定波高场、2 机 4 标、通信宽裕的确定性场景，对照为直传和一条固定巡航，指标为全区域全过程岸站 RMSE。5% 相对 RMSE 仍只是建议阈值。该闭环没有启动。",
        "",
    ]


def _verification(verification: Optional[dict]) -> list:
    lines = ["## 11 验证", ""]
    if not verification:
        lines.append("回归对照尚未写入。")
        lines.append("")
        return lines
    lines.extend(
        [
            f"基线是设计提交的干净工作树，`py -3 -m unittest discover -s tests`：运行 {verification['baseline']['tests_run']}，失败 {verification['baseline']['failures']}，错误 {verification['baseline']['errors']}，跳过 {verification['baseline']['skipped']}。",
            f"叠加上本包和 `tests/test_ocean_observation_m0.py` 之后：运行 {verification['overlay']['tests_run']}，失败 {verification['overlay']['failures']}，错误 {verification['overlay']['errors']}，跳过 {verification['overlay']['skipped']}。",
            f"新增失败签名 {verification['new_failure_count']} 条，消失的基线签名 {verification['lost_baseline_signature_count']} 条。历史套件不是全绿。新增测试文件的结果是 {verification['related_tests']}。",
            "签名对照见 `results/ocean_observation_m0/verification.json`。基线日志和叠加日志保留在同一目录。回归进程数 "
            f"{verification['unittest_processes']}，不计入解析案例，也不计入策略推理。",
            "",
        ]
    )
    return lines


def _revision(links: dict) -> list:
    horizon = links["horizon_buoy_shore_m"] / 1000.0
    uav_buoy = links["horizon_uav_buoy_m"] / 1000.0
    uav_shore = links["horizon_uav_shore_m"] / 1000.0
    return [
        "## 12 唯一修订建议（未执行）",
        "",
        f"保持现在的 52 字节有效波高包、SX1276 的 +13 dBm / SF7 / 125 kHz，以及式 (6) 的功率模型。不要把包改成原始记录，也不要减小带宽。把任务域扩到浮标—岸站无线电地平线之外。按本次天线高度，这条地平线约 {horizon:.2f} km；无人机对浮标约 {uav_buoy:.1f} km，无人机对岸站约 {uav_shore:.1f} km。远侧浮标的直传会先被地平线挡住，无人机飞回岸站附近后的短距回传仍然可以闭合。相关长度在宣称空间信息增益之前换成有出处的沿岸尺度。新的 24 组坐标要在重新计算之前登记。本组 M0 不执行这个修订，也不追加第 25 个案例。",
        "",
    ]


def _decisions() -> list:
    return [
        "## 13 需要用户决定的事项",
        "",
        "1. 是否把第 12 节的地平线修订登记为下一项。M0 不代为执行。",
        "2. 若登记下一项，合成合同是否维持：20 分钟悬停当量能量、20% 余量、2/2/6 dBi 天线、10 m 岸高、1000 m 相关长度、海流 0.10 m/s、Stokes 0.02 m/s、风漂移系数 0.01、600 s 周期中打开 60 s、6 dB 实施裕量、地面怠速 1 W。不要为了让那一组超出余量的巡航变可行而事后改电池。",
        "3. 最小闭环、训练和主动辨识都不在本次授权内。只有另行登记之后才能开始。",
        "",
    ]
