"""Single entry for one Benchmark-V2 hierarchical reference episode."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6f_system_bringup.episode import EpisodeResult, run_episode
from wt.benchmark_v2.phase6f_system_bringup.figures import write_figures
from wt.benchmark_v2.phase6f_system_bringup.protocol import (
    CANDIDATE_K,
    GENERATOR,
    HOLD,
    INTEGRATION_POOL,
    PHASE,
    REPORT,
    SMOKE_POOL,
    choose_gate,
)


def _plain(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(result: EpisodeResult) -> dict:
    return {
        "scenario_id": result.scenario_id,
        "episode_complete": int(result.complete),
        "illegal_assignment": result.illegal,
        "shadow_state_contamination": result.contamination,
        "accounting_failures": result.accounting_failures,
        "nan_inf": result.nan_inf,
        "hold_ok": int(result.hold_ok),
        "chain_ok": int(result.chain_ok),
        "decision_count": result.decision_count,
        "mean_candidate_count": result.mean_candidate_count,
        "mean_j15_seconds": result.mean_j15_seconds,
        "episode_seconds": result.elapsed_seconds,
        "cumulative_payload_bits": result.payload_bits,
    }


def _passed(result: EpisodeResult) -> bool:
    return bool(
        result.complete
        and result.illegal == 0
        and result.contamination == 0
        and result.accounting_failures == 0
        and result.nan_inf == 0
        and result.hold_ok
        and result.chain_ok
        and len(result.steps) == 48
    )


def _store(result: EpisodeResult) -> None:
    logs = PHASE / "logs"
    episodes = PHASE / "episodes"
    _write_csv(logs / f"{result.scenario_id}_{result.mode}_steps.csv", result.steps)
    _write_csv(logs / f"{result.scenario_id}_{result.mode}_uav.csv", result.uav_rows)
    _write_csv(logs / f"{result.scenario_id}_{result.mode}_buoy.csv", result.buoy_rows)
    _write_json(
        episodes / f"{result.scenario_id}_{result.mode}_summary.json",
        {
            "scenario_id": result.scenario_id,
            "mode": result.mode,
            "complete": result.complete,
            "payload_bits": result.payload_bits,
            "runtime": result.runtime,
            "decisions": result.decisions,
        },
    )
    if result.mode == "j15" and result.steps:
        write_figures(result.scenario_id, result.steps, result.uav_rows, result.decisions, PHASE / "figures")


def _flatten(results: list[EpisodeResult], key: str) -> list[dict]:
    rows = []
    for result in results:
        rows.extend(result.audits[key])
    return rows


def _report(gate: dict, smoke: list[EpisodeResult], integration: list[EpisodeResult], maxrate: list[EpisodeResult]) -> None:
    j15 = smoke + integration
    mean_episode = float(np.mean([row.elapsed_seconds for row in j15])) if j15 else 0.0
    mean_j15 = float(np.mean([row.mean_j15_seconds for row in j15])) if j15 else 0.0
    mean_pool = float(np.mean([row.mean_candidate_count for row in j15])) if j15 else 0.0
    payloads = ", ".join(f"{row.scenario_id}={row.payload_bits:.6f}" for row in maxrate) if maxrate else "not run"
    text = f"""# HTW Phase-6F：Benchmark-V2 端到端系统接通

## 结论

Gate 为 **{gate['gate']} {gate['gate_name']}**。本阶段只把已冻结的 `L3BBitQueueEnvV2`、`G1_RATE k=5`、`J15_CURRENT` 和 deterministic fly-toward 串成一条 episode。`planner_ready = False`，`deployment_verified = False`。

## Q1. 完整系统能否从 reset 跑到 episode 结束？

Smoke 完成 {sum(int(row.complete) for row in smoke)}/{len(SMOKE_POOL)}，integration 完成 {sum(int(row.complete) for row in integration)}/{len(INTEGRATION_POOL)}。完成的 episode 都从 reset 走到 T=48。

## Q2. candidate → J15_CURRENT → assignment 是否完整工作？

每个高层决策调用 Phase-6E 的 `build_pools` 和 `batch_j15`，再用已有的字典序、1 bit 容差选出 `A_star`。平均候选池大小为 {mean_pool:.3f}。链路完整：{"是" if j15 and all(row.chain_ok for row in j15) else "否"}。

## Q3. TaskOption hold=15 是否正确？

高层决策时刻是 t=0、15、30、45。`remaining_hold` 在决策时设为 15，之后每步减 1，持有期内不会被重置。t=45 只剩下 3 个环境步，计数仍从 15 递减，episode 在 T=48 停止。Hold 语义：{"正确" if j15 and all(row.hold_ok for row in j15) else "不正确"}。

## Q4. fly-toward 是否正确执行 assignment？

低层调用 `wt.l3b.policies.fly_toward`。目标不是 idle 时，动作是朝当前浮标的单位步；idle 时位移为 0。Phase-6C/6D 的这个执行器没有单独的服务半径内停留规则，本阶段没有新增停留或最小运动。方向检查：{"通过" if j15 and all(row.chain_ok for row in j15) else "未通过"}。

## Q5. shadow evaluator 是否完全不污染主环境？

每次 `batch_j15` 前后都比对主环境的队列、位置、关联、SINR、速率、服务与丢弃、累计 payload、步数、到达、RNG 和已有 TaskOption。该环境没有速度状态，检查记录为速度字段不存在，没有另造速度。污染计数为 {sum(row.contamination for row in j15)}。

## Q6. bit queue accounting 是否继续严格守恒？

逐步和 episode 级都使用 Phase-6B 的 `conservation_error_bits`，容差是环境的 `tol_bits`。失败次数为 {sum(row.accounting_failures for row in j15)}。

## Q7. 所有 assignment 是否合法？

非法计数为 {sum(row.illegal for row in j15)}。

## Q8. 4 个 smoke scenario 是否全部通过？

{"是" if len(smoke) == 4 and all(_passed(row) for row in smoke) else "否"}。

## Q9. 8 个 integration regression 是否全部通过？

{"是" if len(integration) == 8 and all(_passed(row) for row in integration) else "否"}。payload 大小不作为失败条件。

## Q10. 完整一个 episode 平均耗时是多少？

J15 系统的平均 episode 时间为 {mean_episode:.3f} 秒。

## Q11. J15 exhaustive 当前耗时多少？

平均每个高层决策 {mean_j15:.3f} 秒。这个耗时不作为失败，本阶段没有做 batching 或 pruning。

## Q12. MAX_RATE baseline 是否同样能完整跑通？

{"是" if len(maxrate) == 4 and all(row.complete for row in maxrate) else "否"}。累计 payload：{payloads}。

## Q13. 是否在本阶段为了性能修改了任何参数？

NO。

## Q14. 最终 Gate 是否为 RUN-A？

{"是" if gate["gate"] == "RUN-A" else "否"}。当前 Gate 是 {gate["gate"]}。

## Q15. 如果 RUN-A，下一阶段是否只训练 low-level MAPPO？

{"YES" if gate["gate"] == "RUN-A" else "本阶段不是 RUN-A，不进入下一阶段。"}

## Q16. 现在是否训练 HGNN？

NO。

## Q17. 现在是否优化 k=5 / J15 runtime？

NO。

## Q18. 320100–320131 是否仍完全 untouched？

YES。
"""
    REPORT.write_text(text, encoding="utf-8")


def _finalize(smoke, integration, maxrate, *, crashed: bool) -> None:
    j15 = smoke + integration
    logs_ok = bool(j15) and all(len(row.steps) == 48 and len(row.uav_rows) == 48 * 6 and len(row.buoy_rows) == 48 * 12 for row in j15)
    gate = choose_gate(
        contamination=sum(row.contamination for row in j15),
        accounting_failures=sum(row.accounting_failures for row in j15 + maxrate),
        crashed=crashed,
        smoke_ran=len(smoke),
        smoke_complete=sum(int(row.complete) for row in smoke),
        integration_complete=sum(int(row.complete) for row in integration),
        illegal=sum(row.illegal for row in j15 + maxrate),
        nan_inf=sum(row.nan_inf for row in j15 + maxrate),
        hold_ok=bool(j15) and all(row.hold_ok for row in j15),
        chain_ok=bool(j15) and all(row.chain_ok for row in j15),
        logs_ok=logs_ok,
    )
    _write_csv(PHASE / "phase6f_smoke_results.csv", [_summary_row(row) for row in smoke])
    _write_csv(PHASE / "phase6f_integration_results.csv", [_summary_row(row) for row in integration])
    _write_csv(PHASE / "phase6f_maxrate_sanity.csv", [_summary_row(row) for row in maxrate])
    _write_csv(PHASE / "phase6f_taskoption_audit.csv", _flatten(j15, "taskoption"))
    _write_csv(PHASE / "phase6f_candidate_audit.csv", _flatten(j15, "candidate"))
    _write_csv(PHASE / "phase6f_shadow_isolation.csv", _flatten(j15, "shadow"))
    _write_csv(PHASE / "phase6f_accounting_audit.csv", _flatten(j15 + maxrate, "accounting"))
    _write_csv(
        PHASE / "phase6f_runtime.csv",
        [{"scenario_id": row.scenario_id, "mode": row.mode, **row.runtime} for row in j15 + maxrate],
    )
    _write_json(PHASE / "phase6f_system_bringup_gate.json", gate)
    protocol_path = PHASE / "phase6f_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["j15_evaluations_started"] = True
    protocol["rules_frozen_before_first_evaluation"] = True
    _write_json(protocol_path, protocol)
    _report(gate, smoke, integration, maxrate)
    print(gate["gate"], flush=True)


def main() -> None:
    _write_json(
        PHASE / "phase6f_protocol.json",
        {
            "smoke": list(SMOKE_POOL),
            "integration": list(INTEGRATION_POOL),
            "reserved_untouched": list(range(320100, 320132)),
            "generator": GENERATOR,
            "k": CANDIDATE_K,
            "hold": HOLD,
            "j15": "J15_CURRENT",
            "future_arrivals_in_j15": 0,
            "tie_break": "phase6e argmax_value lexicographic assignment within 1 bit",
            "executor": "wt.l3b.policies.fly_toward",
            "stay_rule_added": False,
            "j15_horizon": "min(hold, episode_horizon - t)",
            "taskoption_initial_hold": HOLD,
            "parameters_modified_for_speed": False,
            "j15_evaluations_started": False,
        },
    )
    _write_json(
        PHASE / "phase6f_system_contract.json",
        {
            "chain": [
                "L3BBitQueueEnvV2",
                "G1_RATE top-5",
                "legal joint assignments",
                "J15_CURRENT exhaustive",
                "TaskOption",
                "deterministic fly-toward",
                "live environment step",
            ],
            "reused": [
                "wt.benchmark_v2.bit_queue_v2.L3BBitQueueEnvV2",
                "wt.benchmark_v2.phase6e.generators.build_pools",
                "wt.benchmark_v2.phase6e.value.batch_j15",
                "wt.l3b.policies.fly_toward",
            ],
            "velocity_state": "not_exposed",
            "planner_ready": False,
            "deployment_verified": False,
        },
    )
    smoke: list[EpisodeResult] = []
    integration: list[EpisodeResult] = []
    maxrate: list[EpisodeResult] = []
    try:
        for scenario in SMOKE_POOL:
            result = run_episode(int(scenario), "j15")
            smoke.append(result)
            _store(result)
            print(f"smoke {scenario} complete={result.complete} seconds={result.elapsed_seconds:.1f}", flush=True)
            if not _passed(result):
                _finalize(smoke, integration, maxrate, crashed=not result.complete and result.contamination == 0 and result.accounting_failures == 0)
                return
        for scenario in SMOKE_POOL:
            result = run_episode(int(scenario), "maxrate")
            maxrate.append(result)
            _store(result)
            print(f"maxrate {scenario} complete={result.complete}", flush=True)
            if not result.complete or result.accounting_failures or result.nan_inf:
                _finalize(smoke, integration, maxrate, crashed=not result.complete)
                return
        for scenario in INTEGRATION_POOL:
            result = run_episode(int(scenario), "j15")
            integration.append(result)
            _store(result)
            print(f"integration {scenario} complete={result.complete} seconds={result.elapsed_seconds:.1f}", flush=True)
            if not _passed(result):
                _finalize(smoke, integration, maxrate, crashed=not result.complete and result.contamination == 0 and result.accounting_failures == 0)
                return
    except Exception:
        _finalize(smoke, integration, maxrate, crashed=True)
        raise
    _finalize(smoke, integration, maxrate, crashed=False)


if __name__ == "__main__":
    main()
