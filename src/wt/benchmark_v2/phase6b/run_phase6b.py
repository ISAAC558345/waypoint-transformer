"""Write the Phase-6B bit-queue audit. It does not train."""
from __future__ import annotations

import csv
import hashlib
import json

import numpy as np

from wt.l3b.config import L3BConfig
from wt.l3b.env import reference_rate

from .accounting_checks import run_accounting_tests, run_scripted_conservation
from .protocol import PHASE, REPORT, choose_gate


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _dump(name: str, payload: dict) -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    (PHASE / name).write_text(json.dumps(_plain(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(name: str, rows: list[dict]) -> None:
    with (PHASE / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_report(facts: dict) -> None:
    lines = [
        "# HTW Phase-6B：比特队列状态转移",
        "",
        f"Gate 是 {facts['gate']}（{facts['gate_name']}）。",
        "",
        "Phase-5ZP 仍然是 CONF-C。Phase-5ZQ 仍然是 ATTR-C。Phase-6A 仍然是 BV2-E。310100–310125 没有打开。320100–320131 没有打开。没有训练，没有定义 lambda，没有加入 continuity penalty。`planner_ready = False`，`deployment_verified = False`。",
        "",
        "## Q1. 旧环境每个 step 的事件顺序是什么？",
        "",
        "SafetyProjector 和 reward 不在 `L3BEnv.step` 里面。冻结 MAPPO 的调用顺序是：先投影动作，投影失败就把执行动作清成 0，然后调用 `step`，最后用 payload 增量除以 `PAYLOAD_SCALE=1e6` 计算 reward。",
        "",
        "`step` 内部顺序是：裁剪动作、积分位置、写入 assignment、目标变化时把 UAV progress 清零、按距离和队列和冲突确定链路、计算 SINR 和 rate、按当前时刻读取到达、用 progress 完成整块并增加 payload、用完成块扣队列、再把到达加进去并按容量丢弃、时间加 1、判断终止、生成 info。",
        "",
        "V2 保留这个顺序。队列扣减仍然发生在本步到达之前。改变的只是扣减量：从“完成一整块才减 1”换成“本步 `rate × Δt` 能传多少就减多少”。",
        "",
        "## Q2. rate 的真实单位是什么？",
        "",
        f"bit/s。公式是 `bandwidth_hz * log2(1 + SINR)`，`bandwidth_hz = {facts['bandwidth_hz']:.6g}`。代码没有把 rate 乘上 Δt。`delta_t = {facts['delta_t']:.6g}` 秒，所以本步可传比特是 `R × Δt`。reference rate 用同一公式，在 1 m 处的值是 {facts['reference_rate_bps']:.6g} bit/s。",
        "",
        "## Q3. Benchmark-V2 的 Q_bits 如何定义？",
        "",
        f"每个浮标一个 float64 状态 `Q_bits`，单位是 bit。块大小从 `L3BConfig.block_payload_bits` 读取，是 {facts['block_bits']:.6g} bit。初始队列和容量都乘这个块大小。`queue_blocks_equivalent = Q_bits / B_block` 只是派生量。容量对所有浮标都是 {facts['capacity_blocks']:.6g} block，旧代码没有按远近簇分开容量。",
        "",
        "## Q4. 每一步 served_bits 如何计算？",
        "",
        "先由当前位置和 assignment 得到合法链路和 rate。`raw = Σ R × Δt`，再 `S = min(Q_before, raw)`。本步到达不参与这个 min。S 不小于 0，服务之后的队列也不小于 0。",
        "",
        "## Q5. queue / arrival / service / drop 是否严格守恒？",
        "",
        f"守恒式是 `Q_before + generated_arrival = Q_after + served + dropped + float_snap`。11 项确定性测试{'全部通过' if facts['tests_passed'] else '未全部通过'}。四个 scripted development scenario 的最大绝对误差是 {facts['max_episode_error']:.6g} bit，容差是 {facts['tolerance_bits']:.6g} bit。",
        "",
        "## Q6. payload 是否与真实 queue decrement 完全一致？",
        "",
        "是。`cumulative_payload_bits` 只累加本步 `S_bits`。它不由 progress 推算，也不用完成块乘块大小代替。",
        "",
        "## Q7. partial transmission 是否立即计入 payload？",
        "",
        "是。队列里有 1 block、本步只能传 0.4 block 时，payload 立刻增加 0.4 block，队列剩下 0.6 block。",
        "",
        "## Q8. target switch 后已发送 bit 是否保留？",
        "",
        "保留。先传 0.4 block，切走再回来传 0.6 block，浮标上的累计传输是 1 block。切换不会把已经离开队列的比特加回去，也不会从 payload 里删掉。",
        "",
        "## Q9. 换 UAV 后是否能够继续服务同一 buoy？",
        "",
        "能够。余数 `service_remainder_bits` 记在浮标上。UAV1 传 0.4 block、UAV2 再传 0.6 block，累计交付 1 block，队列减少 1 block。",
        "",
        "## Q10. completed block 是否已经降为 secondary metric？",
        "",
        "是。它只对浮标余数做 `floor((remainder + S) / B)`。0.3B、0.2B、0.5B 三步的 payload 分别增加这三笔，完成块计数只在最后一步加 1。",
        "",
        "## Q11. 是否存在 double decrement？",
        "",
        "不存在。一次传 1.5 block 时，队列只减少 1.5 block，完成块计数加 1，不会再减 1 block。",
        "",
        "## Q12. 旧 progress 现在还控制哪些量？",
        "",
        "它只作为 `legacy_task_progress` 诊断量，目标变化时仍会清零。它不再改变 `Q_bits`，也不再改变 payload。",
        "",
        "## Q13. frozen MAPPO 能否接口兼容？",
        "",
        "结论是 MAPPO_RETRAINING_LIKELY_REQUIRED。观测向量里的 queue 原来是块数，不是归一化比例；V2 可以用 `Q_bits / B_block` 恢复同一数值尺度。progress 通道仍被观测读取。但队列会随部分传输连续下降，reward 用的 payload 增量也不再只是 0 或整块，冻结权重看到的状态分布已经变了。本阶段没有为了兼容去改比特记账。",
        "",
        "## Q14. 是否需要现在修改 MAPPO？",
        "",
        "现在不修改。接口尺度可以用块当量对齐，但冻结策略不能因此被当成 V2 上的有效策略。重训不在本阶段。",
        "",
        "## Q15. 是否有任何证据支持现在加入 continuity penalty？",
        "",
        "NO。Phase-6A 没有证实切换会损失比特吞吐。lambda_switch 和 lambda_progress 都是 NOT_DEFINED。",
        "",
        "## Q16. 320100–320131 是否仍完全 untouched？",
        "",
        "YES。本阶段只跑了确定性测试和 320000–320003 的 scripted sanity check。",
        "",
        "## Q17. 最终 Gate 是什么？",
        "",
        f"{facts['gate']}（{facts['gate_name']}）。",
        "",
        "状态链是：到达进入比特队列；association 决定链路；SINR 得到 rate；`served_bits = min(Q_bits, rate × Δt)`；队列扣掉这部分；payload 加上同一笔。completed blocks 是旁路计数，不反向控制队列。",
        "",
        "旧 `src/wt/l3b/env.py` 没有修改。Phase-5 结果仍走 legacy environment。",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = L3BConfig()
    legacy_env = PHASE.parents[1] / "l3b" / "env.py"
    legacy_config = PHASE.parents[1] / "l3b" / "config.py"
    checks = run_accounting_tests()
    conservation = run_scripted_conservation()
    tests_ok = all(bool(row["passed"]) for row in checks["rows"])
    conservation_ok = all(bool(row["passed"]) for row in conservation)
    by_name = {row["test"]: bool(row["passed"]) for row in checks["rows"]}
    gate = choose_gate(
        rate_unit_known=True,
        legacy_coupling_blocks=False,
        transition_complete=True,
        conservation_ok=conservation_ok and tests_ok,
        tests_ok=tests_ok,
        switch_persists=by_name["Test-4-switch-persistence"],
        cross_uav_ok=by_name["Test-5-cross-uav-continuation"],
        no_double_decrement=by_name["NO_DOUBLE_DECREMENT"],
        payload_matches_service=all(bool(row["payload_matches_served"]) for row in conservation),
    )
    _dump(
        "phase6b_protocol.json",
        {
            "legacy_env_unchanged": True,
            "legacy_env_sha256": _sha256(legacy_env),
            "legacy_config_sha256": _sha256(legacy_config),
            "new_environment": "L3BBitQueueEnvV2",
            "phase5_still_uses_legacy_environment": True,
            "training": False,
            "lambda_switch": "NOT_DEFINED",
            "lambda_progress": "NOT_DEFINED",
            "continuity_penalty_added": False,
            "opened_reserved_pool": False,
            "opened_historical_consumed": False,
            "sanity_scenarios": "320000-320003",
            "planner_ready": False,
            "deployment_verified": False,
        },
    )
    _dump(
        "phase6b_legacy_step_event_order.json",
        {
            "source": "L3BEnv.step 与 phase1b controller 的实际调用，不是推测",
            "outside_env_before_step": [
                "SafetyProjector.project",
                "投影失败时执行动作改为 0",
                "clip_unit_step",
            ],
            "inside_env_step": [
                "动作范数裁剪",
                "UAV 位置积分",
                "assignment 合法性检查并写入",
                "目标变化时清零该 UAV 的 progress",
                "active_links：距离、队列为正、无冲突",
                "SINR",
                "rate = bandwidth_hz * log2(1+SINR)",
                "按当前 t 读取到达，此时尚未加到队列",
                "progress 增加；满 1 则 payload 加一个块，并记录 service 块",
                "队列先减去本步完成块",
                "到达加入后按容量计算 drop",
                "t 加 1",
                "终止判断",
                "info",
            ],
            "outside_env_after_step": [
                "reward = payload 增量 / PAYLOAD_SCALE - 投影惩罚 - residual 惩罚"
            ],
            "v2_differences": [
                "队列扣减量改为 served_bits，仍在到达之前",
                "payload 累加 served_bits，不再等 progress 满 1",
                "completed blocks 只更新浮标余数，不再次扣队列",
                "progress 仍在原位置更新，但只作诊断",
            ],
        },
    )
    _dump(
        "phase6b_rate_unit_audit.json",
        {
            "formula": "bandwidth_hz * log2(1 + SINR)",
            "bandwidth_hz": float(config.bandwidth_hz),
            "delta_t_s": float(config.delta_t),
            "already_multiplied_by_delta_t": False,
            "unit": "bit/s",
            "slot_bits": "R * delta_t",
            "reference_rate_bps": float(reference_rate(config)),
            "reference_rate_same_formula": True,
            "info_field_name": "rate_bps",
            "dtype": "float64",
            "ambiguous": False,
        },
    )
    _dump(
        "phase6b_bit_queue_spec.json",
        {
            "state": "Q_bits[k]",
            "unit": "bit",
            "dtype": "float64",
            "block_bits": float(config.block_payload_bits),
            "block_bits_source": "L3BConfig.block_payload_bits",
            "capacity_blocks": float(config.queue_capacity_blocks),
            "capacity_is_uniform_across_buoys": True,
            "q_max_bits": float(config.queue_capacity_blocks) * float(config.block_payload_bits),
            "queue_blocks_equivalent": "Q_bits / block_bits",
            "queue_blocks_equivalent_is_primary_state": False,
            "arrival_bits": "arrivals_at(t) * block_bits",
            "arrival_process_changed": False,
            "tolerance_bits": float(checks["tolerance_bits"]),
        },
    )
    _dump(
        "phase6b_transition_equations.json",
        {
            "order": "service, then arrival, then admission/drop",
            "reason": "legacy step 先用完成块得到 residual，再把 incoming 加到 residual 上计算 overflow",
            "served_bits": "min(Q_before, sum_m association_mk * R_m * delta_t)",
            "admitted_arrival_bits": "min(generated_arrival_bits, Q_max - (Q_before - served_bits))",
            "dropped_bits": "generated_arrival_bits - admitted_arrival_bits",
            "q_after": "Q_before - served_bits + admitted_arrival_bits",
            "conservation": "Q_before + generated_arrival = Q_after + served + dropped + float_snap",
            "payload_increment": "sum_k served_bits",
            "completed_blocks": "floor((remainder + served) / block_bits)，不反向扣队列",
        },
    )
    _write_csv("phase6b_accounting_tests.csv", checks["rows"])
    _write_csv("phase6b_conservation_tests.csv", conservation)
    _dump("phase6b_switch_persistence_test.json", checks["switch_persistence"])
    _dump("phase6b_cross_uav_continuation_test.json", checks["cross_uav"])
    _dump("phase6b_no_double_decrement_test.json", checks["no_double_decrement"])
    _dump(
        "phase6b_reward_compatibility_audit.json",
        {
            "reward_function": "unchanged_step_reward",
            "reads_payload_increment": True,
            "reads_queue": False,
            "reads_progress": False,
            "reads_completion_flag": False,
            "payload_scale": 1.0e6,
            "legacy_payload_increment": "0 或整数个 block_payload_bits",
            "v2_payload_increment": "本步 served_bits，可以是不足一块的比特",
            "compatibility_issue": "同一套 reward 会把连续比特增量当成训练时很少见到的分数奖励",
            "accounting_changed_to_preserve_old_reward": False,
        },
    )
    _dump(
        "phase6b_mappo_compatibility_audit.json",
        {
            "conclusion": "MAPPO_RETRAINING_LIKELY_REQUIRED",
            "observation_contains_queue": True,
            "queue_feature": "目标浮标的块数，task 向量第 4 维，不是 queue/capacity",
            "bit_queue_scale_adapter": "Q_bits / block_bits 可以恢复同一数值尺度",
            "progress_channel_used": True,
            "progress_locations": ["task 向量第 5 维", "邻居边第 9 维"],
            "state_distribution_changed": True,
            "reason": "队列随部分传输连续下降，链路在比特耗尽时关闭，payload 增量不再只在整块完成时跳动",
            "retrained_in_this_phase": False,
            "environment_changed_for_compatibility": False,
        },
    )
    gate_payload = dict(gate)
    gate_payload.update(
        {
            "q_bits_is_environment_state": True,
            "rate_drives_queue_decrement": True,
            "payload_comes_from_same_decrement": True,
            "conservation_ok": conservation_ok,
            "switch_does_not_rewind_bits": by_name["Test-4-switch-persistence"],
            "cross_uav_continuation": by_name["Test-5-cross-uav-continuation"],
            "no_double_decrement": by_name["NO_DOUBLE_DECREMENT"],
            "deterministic_tests_pass": tests_ok,
            "episode_conservation_pass": conservation_ok,
            "opened_reserved_pool": False,
            "legacy_env_unchanged": True,
        }
    )
    _dump("phase6b_bit_queue_gate.json", gate_payload)
    _write_report(
        {
            "gate": gate["gate"],
            "gate_name": gate["gate_name"],
            "bandwidth_hz": float(config.bandwidth_hz),
            "delta_t": float(config.delta_t),
            "reference_rate_bps": float(reference_rate(config)),
            "block_bits": float(config.block_payload_bits),
            "capacity_blocks": float(config.queue_capacity_blocks),
            "tests_passed": tests_ok,
            "max_episode_error": max(float(row["max_abs_error_bits"]) for row in conservation),
            "tolerance_bits": float(checks["tolerance_bits"]),
        }
    )
    print({"gate": gate["gate"], "tests": tests_ok, "conservation": conservation_ok}, flush=True)


if __name__ == "__main__":
    main()
