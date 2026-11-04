"""Chinese summary of the registered 192-step reproduction."""
from __future__ import annotations


def _number(value) -> str:
    if value is None:
        return "缺失"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_report(manifest: dict) -> str:
    pairs = manifest["pairs"]
    lines = [
        "# 正式路径轻量复现 R1",
        "",
        f"实现提交 `{manifest['implementation_commit']}`。指令提交 `{manifest['instruction_commit']}`。",
        "本包通过 `run_active_episode` 的配置副本运行两个 TRAIN 场景和两条 G1_RATE k=6 臂。",
        "192 步只说明正式入口可以运行并留下可追溯输出，不构成泛化评估，也不表示系统已经发布。",
        "",
        "## 运行合同",
        "",
        f"- 场景：{manifest['scenarios']}。进程种子为 1000000 加场景号；环境 reset 种子仍是场景号。",
        f"- 每臂 {manifest['horizon']} 步，登记上限 {manifest['step_budget']} 步，实际环境步数 {manifest['environment_steps']}。",
        f"- 训练更新 {manifest['training_updates']}。正式策略文件保持未改：{manifest['official_policy_unchanged']}。",
        f"- 实际读取的 MAPPO 文件哈希 `{manifest['models']['mappo_sha256']}`。",
        f"- 实际读取的 Scheme C 文件哈希 `{manifest['models']['scheme_c_sha256']}`。",
        "",
        "## 分场景结果",
        "",
        "| 场景 | Scheme C payload (bit) | G1_RATE payload (bit) | 差值 (bit) | Scheme C 耗时 (s) | G1_RATE 耗时 (s) | 两端完成 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in pairs:
        lines.append(
            "| {scenario_id} | {scheme} | {baseline} | {delta} | {scheme_seconds} | {baseline_seconds} | {done} |".format(
                scenario_id=row["scenario_id"],
                scheme=_number(row["scheme_c_payload_bits"]),
                baseline=_number(row["g1_rate_payload_bits"]),
                delta=_number(row["payload_difference_bits"]),
                scheme_seconds=_number(row["scheme_c_seconds"]),
                baseline_seconds=_number(row["g1_rate_seconds"]),
                done=f"{row['scheme_c_complete']} / {row['g1_rate_complete']}",
            )
        )
    archive = manifest["archive_summary"]
    failures = manifest["failure_summary"]
    lines.extend(
        [
            "",
            "## 原始证据",
            "",
            f"归档目录与新 192 步结果分开。恢复 {archive['recovered_count']} 个文件，缺档 {archive['missing_count']} 个。",
            "AB 汇总是开发集统计，AC 汇总是已保存的 holdout 统计。本次没有重开这些场景。",
            f"缺档列表：{archive['missing'] if archive['missing'] else '无'}。",
            "",
            "## 历史失败索引",
            "",
            f"来源 `{failures['source_commit']}:{failures['source_path']}`，签名 {failures['signature_count']} 个。",
            f"分类计数：{failures['counts']}。",
            f"其中触及当前正式入口或比特队列测试的签名为 {failures['current_formal_path_count']} 个。",
            "这些历史失败保持原分类。本包没有改写队列、锁或测试来清零。",
            "",
            "## 必要检查",
            "",
            manifest["checks_note"],
            "",
            "## 保留差距",
            "",
            "正式入口可以运行，输出能够追溯到配置副本、模型哈希和逐步 served。",
            "历史套件没有重跑，既有失败签名仍然存在。ranking-loss 的 summary.json 若仍为 MISSING，就继续缺档。",
            "本结果的 deployment_verified 不适用，也不把两个场景写成总体优劣结论。",
            "",
        ]
    )
    return "\n".join(lines)


def render_runtime(runtime: dict) -> str:
    pins = runtime["historical_pins"]
    return "\n".join(
        [
            "# R1 正式入口运行依赖",
            "",
            "本文件记录这次正式入口实际导入的版本。历史 `requirements.txt` 保持原样，不按本文件批量升级旧模块。",
            "",
            f"- Python：{runtime['python']}",
            f"- NumPy：{runtime['numpy']}",
            f"- torch：{runtime['torch']}",
            f"- 历史 requirements.txt 中的 NumPy 行：{pins.get('numpy', '未找到')}",
            f"- 历史 requirements.txt 中的 torch 行：{pins.get('torch', '未找到')}",
            "",
            "当前研究环境的 Python、NumPy 和 torch 与历史 requirements.txt 的钉扎不一致。正式入口使用当前已安装的版本运行，没有另装一套依赖。",
            "",
        ]
    )
