"""Chinese report for the registered M0A comparison."""
from __future__ import annotations


def _n(value, digits: int = 1) -> str:
    if value is None:
        return "无"
    return f"{float(value):.{digits}f}"


def render_report(geometry: dict, table: dict) -> str:
    lines = [
        "# 海洋观测 M0A：地平线口径与同一任务比较",
        "",
        "主合同使用 k = 4/3，地球半径 6371000 m。公式是 d = sqrt(2*k*R*h1) + sqrt(2*k*R*h2)，长度单位都是米。",
        "1 m 与 10 m 的手算结果：k = 1 时约 14857.64 m（14.858 km），k = 4/3 时约 17156.13 m（17.156 km）。",
        "M0 的 `radio_horizon_m` 仍使用 3570 倍的平方根和，1 m 与 10 m 得到约 14859.33 m。该系数对应 k = 1 的几何地平线，历史结果文件没有改写。",
        "6 dB 实现损耗仍只作为接收门限的余量，不另加海面传播模型。",
        "",
        "## 冻结几何",
        "",
        "三个几何各配静风和 8 m/s 东风，共 18 个方法结果。漂移不进入任何一条臂，两个生成时刻都使用下面的坐标。",
        "",
    ]
    for row in geometry["geometries"]:
        lines.append(f"- {row['geometry']}：{row['reason']}")
        lines.append(f"  浮标 {row['buoys_xy_m']}。固定悬停 {row['fixed']['xy_m']}，覆盖全部浮标：{row['fixed']['covers_all_buoys']}。")
        for visit in row["moving"]["visits"]:
            lines.append(f"  移动接收 {visit['group']} 由 UAV {visit['uav']} 在 {visit['receive_xy_m']} 执行，覆盖：{visit['covers']}。")
    lines.extend(
        [
            "",
            "固定位置的规则是：从机库沿指向质心的线段每 100 m 取一点，取第一个能同时闭合全部浮标链路和岸站链路的点；没有这样的点时保留质心，并记为未覆盖。",
            "移动航线对每个簇使用同一条规则。质心 x 不小于 0 的簇由 UAV 0 执行，其余由 UAV 1 执行。接收点不取浮标坐标本身，除非该点正好是线段上的第一个可通点。",
            "",
            "## 同一任务",
            "",
            "4 个浮标、每标在 0 s 和 1200 s 各一个 52 字节产品，截止 1800 s。两条臂和直传面对同一组 8 个产品。",
            f"单机可用能量 {_n(table['usable_energy_J'], 1)} J，20% 余量后的上限 {_n(table['energy_limit_J'], 1)} J。悬停功率 {_n(table['hover_W'], 2)} W。",
            "",
            "| 几何 | 风 | 方法 | 未送达 | 全送达比例 | 最早送达 (s) | 最晚送达 (s) | 平均时延 (s) | 系统能量 (J) | 可接受 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in table["rows"]:
        lines.append(
            "| {geometry} | {wind} | {method} | {missing} | {fraction} | {first} | {last} | {delay} | {energy} | {ok} |".format(
                geometry=row["geometry"],
                wind=row["wind"],
                method=row["method"],
                missing=row["undelivered_count"],
                fraction=_n(row["full_delivery_fraction"], 3),
                first=_n(row["first_delivery_s"], 1),
                last=_n(row["latest_delivery_s"], 1),
                delay=_n(row["mean_delay_s"], 1),
                energy=_n(row["system_energy_J"], 1),
                ok=row["acceptable"],
            )
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"结论：{table['conclusion']}。",
            "比较只针对这组链路任务的送达和代价。这里没有岸图均方误差，也没有闭合观测质量回路。",
            "能量超过单机余量上限的结果保留为方法不可行，不另开安全专项，也不改电池或坐标。",
            "",
        ]
    )
    return "\n".join(lines)
