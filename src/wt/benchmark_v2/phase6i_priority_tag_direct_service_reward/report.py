"""Chinese report rendered from the result summary. Numbers are not typed by hand."""
from __future__ import annotations


def _num(value: float) -> str:
    return f"{float(value):.6f}"


def _interval(item: dict) -> str:
    return f"{_num(item['mean'])} [{_num(item['ci_low'])}, {_num(item['ci_high'])}]"


def _seed_lines(score: dict) -> list[str]:
    lines = []
    for item in score["seeds"]:
        seed = int(item["model_seed"])
        lines.append(f"### L3 seed {seed}")
        lines.append("")
        lines.append(f"- 是否通过该种子的全部预注册检查：{bool(item['passes_preregistered_criteria'])}")
        lines.append(f"- 相对 L0 的实际标签服务配对均值与 95% 区间：{_interval(item['tagged_versus_l0'])}")
        lines.append(f"- 相对 L0 的年龄加权服务配对均值与 95% 区间：{_interval(item['age_versus_l0'])}")
        lines.append(f"- 相对 L0 的普通服务配对均值与 95% 区间：{_interval(item['ordinary_versus_l0'])}")
        lines.append(f"- 相对 L0 的总载荷配对均值与 95% 区间：{_interval(item['payload_versus_l0'])}")
        lines.append(f"- 载荷区间下界门槛（L0 均值的 -0.5%）：{_num(item['payload_ci_floor'])}")
        lines.append(f"- 载荷为负的场景：{item['payload_negative_scenarios']}")
        lines.append(f"- 安全违规配对和：{_num(item['safety_diff_sum'])}；约束失败配对和：{_num(item['constraint_diff_sum'])}")
        lines.append(f"- SINR 配对均值：{_num(item['sinr_versus_l0_mean'])}；干扰配对均值：{_num(item['interference_versus_l0_mean'])}")
        lines.append(f"- 留一场景后标签增益均值全部大于 0：{bool(item['tagged_not_single_scenario'])}")
        lines.append(f"- 去掉两个最大标签增益后均值仍大于 0：{bool(item['tagged_not_two_scenarios'])}")
        lines.append(f"- 相对 L2 的实际标签服务：{_interval(item['tagged_versus_l2'])}")
        lines.append(f"- 相对 L2 的总载荷：{_interval(item['payload_versus_l2'])}")
        lines.append(f"- 相对 L0 的丢弃比特（报告项，不作为合并门槛）：{_interval(item['dropped_versus_l0'])}")
        lines.append(f"- 相对 L0 的最终队列均值：{_num(item['queue_versus_l0_mean'])}")
        lines.append("")
    return lines


def render_report(summary: dict) -> str:
    score = summary.get("score")
    lines = [
        "# 分离年龄奖励与直接标签服务奖励实验报告",
        "",
        f"Gate: {summary['gate']}",
        "",
        "本阶段奖励是 E9 归因之后预先声明的新候选：年龄项和直接标签服务项分开封顶。",
        "它不是对已经完成的仅标签奖励实验的系数回调。标签是合成标签，只检验注册协议下的算法行为，不表示真实海事优先级。",
        "",
        "## 结论",
        "",
        summary["reason"],
        "",
        f"- 科学门控：{summary.get('scientific_gate')}",
        f"- 发布门控：{summary['gate']}",
        f"- 通过全部预注册检查的种子数：{None if score is None else score['passing_seeds']}",
        f"- 是否合并：{bool(summary['merged'])}",
        f"- 分支差异是否仅含本实验文件：{bool(summary.get('isolated_diff'))}",
        "",
        "## 预先锁定的奖励与判定",
        "",
        "- r_base = served_bits/1e6 - 0.05*c_project - 0.01*c_residual",
        "- bonus_age = min(0.0005 * age_service_bits/1e6, 0.00025 * served_bits/1e6)",
        "- bonus_tag = min(0.0010 * served_tagged_bits/1e6, 0.00075 * served_bits/1e6)",
        "- served_bits = 0 时两项奖金都为 0。年龄项只用服务前 FIFO 的 clip(age/48, 0, 1)，标签项只用实际服务到的合成标签比特。",
        "- 任何一个种子的安全违规或约束失败配对和大于 0，阶段记为 FAIL。其余合并条件需要至少 2 个种子同时满足：标签服务相对 L0 的配对 95% 区间下界大于 0；载荷均值不小于 0 且区间下界不低于该种子 L0 载荷均值的 -0.5%；SINR 配对均值不小于 0；干扰配对均值不大于 0；每个留一场景子集的标签增益均值大于 0；去掉两个最大标签增益后均值仍大于 0。",
        "- 相对 feature/maritime-wt-stage1a 的差异若不限于本实验文件，则不发布 PASS，也不合并。",
        "- 丢弃比特和队列只报告，不额外加进本阶段的合并门槛。相对 L2 的区间也只报告。",
        "",
        "## 种子角色",
        "",
        "- 训练模型种子是 0、1、2。评估时模型种子仍是这三枚，不是赋值种子。",
        "- 环境 reset 种子是场景编号。赋值 RNG 是 7000000 加场景编号。",
        "- 每个评估回合开始前重新施加历史进程上下文：线程数 1，确定性算法开启，全局种子 2。种子 2 是历史继承的进程状态，不是预先声明的评估协议种子；规范的显式评估种子保持为空。",
        "",
    ]
    if summary.get("training"):
        lines.extend(["## 训练", ""])
        for item in summary["training"]:
            lines.append(
                f"- seed {item['model_seed']}：SHA-256 {item['sha256']}，用时 {_num(item['elapsed_seconds'])} 秒，"
                f"最终策略损失 {_num(item['final_policy_loss'])}，价值损失 {_num(item['final_value_loss'])}，"
                f"梯度范数 {_num(item['final_gradient_norm'])}，源提交 {item['source_commit']}"
            )
        lines.append("")
    if summary.get("preflight"):
        item = summary["preflight"]
        lines.extend(
            [
                "## L0 预检",
                "",
                f"- 场景 {item['scenario_id']} 是否匹配已保存行：{bool(item['matches_saved'])}",
                f"- 最大比特差：{_num(item['max_bit_gap'])}；路径差（米）：{item['path_gap_m']}",
                "",
            ]
        )
    if summary.get("replay"):
        item = summary["replay"]
        lines.extend(
            [
                "## 冻结策略回放",
                "",
                f"- 匹配 {item['matched']} / {item['compared']}。不匹配 {item['mismatches']}。",
                f"- 最大比特差：{item['max_bit_gap']}；最大路径差（米）：{item['max_path_gap_m']}",
                "",
            ]
        )
    if score is not None:
        lines.extend(["## 配对评估", ""] + _seed_lines(score))
        lines.append("以上区间由场景 CSV 重新汇总，bootstrap 为 10000 次、随机数种子 0、分位 2.5 和 97.5。")
        lines.append("")
    lines.extend(
        [
            "## 未改动的正式结果",
            "",
            f"- 正式 checkpoint 与既有 L1/L2 checkpoint 哈希保持不变：{bool(summary.get('prior_checkpoints_ok'))}",
            f"- 观察名单文件哈希在运行前后一致：{bool(summary.get('watched_unchanged'))}",
            f"- 相对正式分支的非本实验路径数量：{summary.get('unexpected_path_count')}",
            "",
            "## 范围",
            "",
            "没有重跑仓库里的全部历史单元测试。没有读取保留集或测试集。没有重训 L0、L1、L2，也没有把本阶段 checkpoint 接进正式规划器。",
            "",
            "## 下一步",
            "",
            summary["next_step"],
            "",
        ]
    )
    return "\n".join(lines)
