"""Generate two new holdout batches. No B2 scoring and no planner call."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wt.htw_phase104.protocol import all_blocked_scenario_seeds
from wt.htw_phase945.candidate_pool import assignment_hash, generate_pool
from wt.lowlevel_v2.data_registry import scan_historical_seeds
from wt.lowlevel_v2.htw_phase4b_b2_shadow.protocol import P3E_CAND_KEYS
from wt.lowlevel_v2.htw_phase4b_b2_shadow.shadow_scorer import extract_candidate_geometry
from wt.lowlevel_v2.integration_phase1.shadow import shadow_h15
from wt.lowlevel_v2.integration_phase2.utility_gen import load_ll_worker
from wt.lowlevel_v2.integration_phase3.protocol import exec_ckpt
from wt.lowlevel_v2.phase2.protocol import env_sha
from wt.lowlevel_v2.phase4b.snapshot import snapshot_env
from wt.lowlevel_v2.phase4d7b_independent_holdout_generation.protocol_ids import (
    OUTPUT,
    ROOT,
    allocate_holdout_ids,
    historical_ids,
    load_saved_allocation,
    save_allocation,
)
from wt.lowlevel_v2.traj_audit.protocol import LL_RUN, LOW_SEEDS
from wt.lowlevel_v2.traj_audit.run_audit import auditor_env_reset, load_high, make_override_stack, run_episode

LABELS = OUTPUT / "holdout_labels.csv"
CACHE = OUTPUT / "ull_cache.json"
META = OUTPUT / "scenario_meta.json"
FAILURE = OUTPUT / "generation_failure.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def _csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(v) for k, v in row.items()})


def _load_rows() -> list[dict]:
    if not LABELS.is_file():
        return []
    with LABELS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["scenario_id"] = int(float(row["scenario_id"]))
        row["candidate_id"] = int(float(row["candidate_id"]))
    return rows


def _load_meta() -> dict:
    if not META.is_file():
        return {}
    return json.loads(META.read_text(encoding="utf-8"))


def _state_key(state: dict, assignment: np.ndarray) -> str:
    raw = np.asarray(state["env"]["uav_positions"], dtype=np.float64).tobytes()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f"{int(state['scenario'])}|{int(state['t'])}|{float(state['env']['payload_bits'])}|{digest}|{assignment_hash(assignment)}|n{len(LOW_SEEDS)}"


def _reset_state(scenario: int) -> dict:
    env = auditor_env_reset(int(scenario))
    return {
        "scenario": int(scenario),
        "t": int(env.t),
        "env": snapshot_env(env),
        "last_arrival": np.zeros(env.config.num_buoys, dtype=np.float32),
        "last_positions": np.asarray(env.uav_positions, dtype=np.float32).copy(),
        "last_movement": np.zeros((env.config.num_uavs, 2), dtype=np.float64),
    }


class OfficialLabeler:
    """Same U_single definition as Phase-4C. Cache stays in the holdout directory."""

    def __init__(self) -> None:
        self.shadows = {int(s): load_ll_worker(int(s)) for s in LOW_SEEDS}
        self.cache: dict[str, float] = {}
        if CACHE.is_file():
            self.cache = json.loads(CACHE.read_text(encoding="utf-8"))

    def save(self) -> None:
        _dump(CACHE, self.cache)

    def u_single(self, state: dict, assignment: np.ndarray) -> float:
        key = _state_key(state, assignment)
        if key in self.cache:
            return float(self.cache[key])
        vals = [
            float(shadow_h15(self.shadows[int(s)], state, assignment, zero_residual=False)["payload_increment"])
            for s in LOW_SEEDS
        ]
        value = float(np.mean(vals))
        self.cache[key] = value
        if len(self.cache) % 8 == 0:
            self.save()
        return value


def _write_id_manifests(batch1: list[int], batch2: list[int], used_n: int, start: int) -> None:
    config_id = f"L3BEnv.reset(seed=scenario_id); env_sha={env_sha()}; M=6;K=12;T=48;hold=15"
    policy = (
        "New contiguous IDs outside the scanned historical universe. "
        "scenario_id is the L3BEnv reset seed. allocate_new_pool was not called."
    )
    common = {
        "config_id": config_id,
        "generation_policy": policy,
        "source": "HUMAN_SPECIFICATION",
        "b2_frozen": True,
        "gate1_frozen": True,
        "criterion_frozen": True,
        "b2_used_for_labels": False,
        "gate_used_for_selection": False,
        "allocate_new_pool_called": False,
        "historical_id_count_scanned": used_n,
        "namespace_start": start,
    }
    _dump(
        OUTPUT / "deployment_holdout_batch_1_manifest.json",
        {**common, "batch_id": "deployment_holdout_batch_1", "size": len(batch1), "scenario_ids": batch1},
    )
    _dump(
        OUTPUT / "deployment_holdout_batch_2_manifest.json",
        {**common, "batch_id": "deployment_holdout_batch_2", "size": len(batch2), "scenario_ids": batch2},
    )


def generate_labels(batch1: list[int], batch2: list[int]) -> list[dict]:
    rows = _load_rows()
    done = {(int(r["scenario_id"]), int(r["candidate_id"])) for r in rows}
    meta = _load_meta()
    labeler = OfficialLabeler()
    ckpt = exec_ckpt(0)
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing official EXEC checkpoint: {ckpt}")
    high = load_high(ckpt)
    worker = make_override_stack(high, LL_RUN)
    config_id = f"L3BEnv.reset(seed=scenario_id); env_sha={env_sha()}; M=6;K=12;T=48;hold=15"
    plan = [("deployment_holdout_batch_1", sc) for sc in batch1] + [("deployment_holdout_batch_2", sc) for sc in batch2]
    try:
        for batch_id, scenario in plan:
            if all((scenario, i) in done for i in range(24)):
                continue
            if str(scenario) not in meta:
                meta[str(scenario)] = {
                    "scenario_id": int(scenario),
                    "seed": int(scenario),
                    "batch_id": batch_id,
                    "generation_timestamp": _now(),
                    "config_id": config_id,
                }
                _dump(META, meta)
            env = auditor_env_reset(int(scenario))
            if int(env.config.num_uavs) != 6 or int(env.config.num_buoys) != 12:
                raise RuntimeError(f"environment shape changed for {scenario}")
            pool = generate_pool(env, seed_key=int(scenario) * 1009 + 0)
            if len(pool) != 24:
                raise RuntimeError(f"pool size {len(pool)} != 24 for {scenario}")
            state0 = _reset_state(int(scenario))
            for index, gamma in enumerate(pool):
                if (scenario, index) in done:
                    continue
                u_single = labeler.u_single(state0, gamma)
                worker.set_first_assignment(gamma)
                rec = run_episode(worker, int(scenario), labeler, compute_ull=False)
                geom = extract_candidate_geometry(env, gamma)
                c_abs = float(rec["U_traj"]) - float(u_single)
                row = {
                    "family": "EXEC",
                    "batch_id": batch_id,
                    "scenario_id": int(scenario),
                    "seed": int(scenario),
                    "state_id": int(scenario),
                    "candidate_id": int(index),
                    "U_single": float(u_single),
                    "U_traj": float(rec["U_traj"]),
                    "C_abs": float(c_abs),
                    "assignment_hash": assignment_hash(gamma),
                    "gamma": json.dumps([int(x) for x in np.asarray(gamma, dtype=np.int64)[:6].tolist()]),
                    "n_hold": int(rec["n_hold"]),
                    "ll_run": int(LL_RUN),
                    "high_family": "EXEC",
                    "high_seed": 0,
                    "config_id": config_id,
                    "generation_timestamp": meta[str(scenario)]["generation_timestamp"],
                    **geom,
                }
                rows.append(row)
                done.add((scenario, index))
                if len(rows) % 4 == 0:
                    _csv_write(LABELS, rows)
                    labeler.save()
                print(f"holdout {batch_id} sc={scenario} cand={index + 1}/24", flush=True)
            _csv_write(LABELS, rows)
            labeler.save()
    except Exception as exc:
        _csv_write(LABELS, rows)
        labeler.save()
        _dump(META, meta)
        _dump(FAILURE, {"error": repr(exc), "completed_rows": len(rows), "timestamp": _now()})
        raise
    del worker
    _csv_write(LABELS, rows)
    labeler.save()
    _dump(META, meta)
    return rows


def audit(rows: list[dict], batch1: list[int], batch2: list[int], history: set[int]) -> list[dict]:
    by = {}
    for row in rows:
        by.setdefault(int(row["scenario_id"]), []).append(row)
    b2_fit = set(range(221000, 221016))
    out = []
    for batch_id, ids in (("deployment_holdout_batch_1", batch1), ("deployment_holdout_batch_2", batch2)):
        for scenario in ids:
            got = by.get(int(scenario), [])
            geom_ok = bool(got) and all(key in got[0] and got[0][key] not in ("", None) for key in P3E_CAND_KEYS)
            u_single_ok = len(got) == 24 and all(row.get("U_single") not in ("", None) for row in got)
            u_traj_ok = len(got) == 24 and all(row.get("U_traj") not in ("", None) for row in got)
            out.append(
                {
                    "scenario_id": int(scenario),
                    "seed": int(scenario),
                    "batch_id": batch_id,
                    "n_candidates": len(got),
                    "u_single_complete": int(u_single_ok),
                    "u_traj_complete": int(u_traj_ok),
                    "geometry_complete": int(geom_ok and len(got) == 24),
                    "in_historical_pool": int(int(scenario) in history),
                    "in_b2_fit": int(int(scenario) in b2_fit),
                    "in_feature_or_threshold_selection": 0,
                    "in_criterion_selection": 0,
                }
            )
    return out


def finalize(rows: list[dict], batch1: list[int], batch2: list[int], history: set[int], start: int) -> dict:
    checked = audit(rows, batch1, batch2, history)
    _csv_write(OUTPUT / "holdout_integrity_audit.csv", checked)
    ids = batch1 + batch2
    ok = (
        len(batch1) == 64
        and len(batch2) == 64
        and len(set(ids)) == 128
        and set(batch1).isdisjoint(batch2)
        and set(ids).isdisjoint(history)
        and all(row["n_candidates"] == 24 and row["u_single_complete"] and row["u_traj_complete"] and row["geometry_complete"] for row in checked)
        and not any(row["in_historical_pool"] or row["in_b2_fit"] for row in checked)
    )
    status = "INDEPENDENT_HOLDOUT_DATA_CREATED_PENDING_VERIFICATION" if ok else "BLOCKED_ON_GENERATION_FAILURE"
    meta = _load_meta()
    registry = {
        "namespace": "deployment_holdout",
        "allocate_new_pool_called": False,
        "distinct_from": ["LLV2_train", "LLV2_design", "phase2_val", "phase3_val", "phase104", "phase105", "phase106"],
        "scenarios": [meta[str(sc)] for sc in ids if str(sc) in meta],
    }
    _dump(OUTPUT / "holdout_scenario_registry.json", registry)
    manifest = {
        "batch_1_size": 64,
        "batch_2_size": 64,
        "total_size": 128,
        "batch_1_scenarios": batch1,
        "batch_2_scenarios": batch2,
        "namespace_start": start,
        "generation_policy": "L3BEnv.reset(seed=scenario_id); generate_pool(seed_key=scenario_id*1009+0); official U_single and U_traj workers",
        "source": "HUMAN_SPECIFICATION",
        "b2_frozen": True,
        "gate1_frozen": True,
        "criterion_frozen": True,
        "deployment_evaluation": False,
        "status": status,
        "data_registered": bool(ok),
        "deployment_verified": False,
        "planner_ready": False,
        "label_rows": len(rows),
    }
    _dump(OUTPUT / "deployment_holdout_manifest.json", manifest)
    body = _report(checked, manifest, ok)
    (OUTPUT / "HTW_Phase4D7F_IndependentHoldoutGeneration_CN.md").write_text(body, encoding="utf-8")
    (ROOT / "docs/archive/phase_reports/HTW_Phase4D7F_IndependentHoldoutGeneration_CN.md").write_text(body, encoding="utf-8")
    print(json.dumps({"status": status, "rows": len(rows), "ok": ok}), flush=True)
    return manifest


def _report(checked: list[dict], manifest: dict, ok: bool) -> str:
    n1 = sum(1 for row in checked if row["batch_id"] == "deployment_holdout_batch_1")
    n2 = sum(1 for row in checked if row["batch_id"] == "deployment_holdout_batch_2")
    complete = all(row["u_single_complete"] and row["u_traj_complete"] and row["geometry_complete"] for row in checked) if checked else False
    overlap = any(row["in_historical_pool"] for row in checked)
    b2 = any(row["in_b2_fit"] for row in checked)
    status = manifest["status"]
    return "\n".join(
        [
            "# HTW Phase-4D-7F Independent Holdout Generation",
            "",
            "本阶段生成两个新的 deployment holdout batch，并只做完整性审计。",
            "未计算 R、ΔR、B2 vs LOCAL 或 deployment PASS/FAIL。未调用正式 decide()。",
            "未修改 B2、τ、Gate-1、criterion、planner。未进入 Phase-4E。",
            "",
            f"**status = {status}**",
            "",
            "**planner_ready = False**",
            "",
            f"**data_registered = {manifest['data_registered']}**",
            "",
            "**deployment_verified = False**",
            "",
            "## 分配",
            "",
            "`allocate_new_pool()` 没有被调用。它只服务 LLV2_train / LLV2_design。",
            "本阶段在 phase4d7b 下建立独立 namespace `deployment_holdout`。",
            f"扫描到的历史 ID 之后，新区间起点为 {manifest['namespace_start']}。",
            f"Batch-1 IDs：{manifest['batch_1_scenarios'][0]}–{manifest['batch_1_scenarios'][-1]}，数量 {n1}。",
            f"Batch-2 IDs：{manifest['batch_2_scenarios'][0]}–{manifest['batch_2_scenarios'][-1]}，数量 {n2}。",
            "scenario_id 同时是 `L3BEnv.reset` 的 seed。两个 batch 的 ID 不重叠，因此 realization 不同，不是把 Batch-1 改 seed 后冒充 Batch-2。",
            "",
            "## 标签",
            "",
            "复用官方入口：`generate_pool(seed_key=scenario_id*1009+0)`、5-seed `shadow_h15` 的 `U_single`、`run_episode` 的 `U_traj`、`extract_candidate_geometry`。",
            "这些标签不使用 B2，也不使用 Gate-1 或 criterion 筛选。128 个 ID 全部保留，没有按结果删除或补采样。",
            f"标签行数 = {manifest['label_rows']}。U_single / U_traj / geometry 完整 = {complete}。",
            "",
            "## 完整性",
            "",
            f"1. Batch-1 恰好 64：{n1 == 64}",
            f"2. Batch-2 恰好 64：{n2 == 64}",
            f"3. 共 128 个新 scenario：{n1 + n2 == 128 and len(set(manifest['batch_1_scenarios'] + manifest['batch_2_scenarios'])) == 128}",
            "4. scenario ID 全部位于新区间，不在禁止清单内。",
            "5. seed 等于新的 scenario_id，因此也是新的。",
            f"6. 与历史池交集：{overlap}",
            f"7. 进入 B2 fit：{b2}",
            "8. 未进入 feature、hyperparameter、τ、Gate-1 或 criterion selection。生成脚本不读取这些选择结果来决定 ID。",
            f"9–11. U_single、U_traj、Phase-3E geometry 完整：{complete}",
            f"12. 生成失败：{not ok}",
            "13. leakage：新 ID 未进入上述选择过程，也未改模型、τ、Gate-1、criterion 或 planner。",
            "14. 数据完整时，下一阶段才允许做 deployment verification。本阶段不允许。",
            "",
            "## 时间顺序",
            "",
            "B2、Gate-1 和 Candidate-C 在 Phase-4D6 已冻结。Holdout 长度在 Phase-4D7E 已由人工指定。本脚本在此之后才 reset 新 ID。",
            "生成过程没有根据标签回改这些冻结对象。",
            "",
            "本阶段只完成 independent holdout data creation，不执行 deployment verification，不修改任何冻结模型/阈值/criterion，不进入 Phase-4E。",
            "",
        ]
    )


def resolve_ids() -> tuple[list[int], list[int], int, set[int]]:
    saved = load_saved_allocation()
    history = historical_ids()
    history.update(scan_historical_seeds())
    history.update(all_blocked_scenario_seeds())
    if saved is None:
        batch1, batch2, start = allocate_holdout_ids(history)
        if set(batch1 + batch2) & history:
            raise RuntimeError("allocated holdout IDs overlap history")
        save_allocation(batch1, batch2, start)
    else:
        batch1, batch2, start = saved
        overlap = set(batch1 + batch2) & history
        if overlap:
            raise RuntimeError(f"saved holdout IDs overlap history: {sorted(overlap)[:8]}")
    history.difference_update(batch1)
    history.difference_update(batch2)
    return batch1, batch2, start, history


def main() -> None:
    import sys

    OUTPUT.mkdir(parents=True, exist_ok=True)
    batch1, batch2, start, history = resolve_ids()
    _write_id_manifests(batch1, batch2, len(history), start)
    print(
        json.dumps(
            {
                "namespace_start": start,
                "batch_1": [batch1[0], batch1[-1], len(batch1)],
                "batch_2": [batch2[0], batch2[-1], len(batch2)],
                "history_without_new_ids": len(history),
                "overlap": len(set(batch1 + batch2) & history),
            }
        ),
        flush=True,
    )
    if "--allocate-only" in sys.argv:
        return
    rows = generate_labels(batch1, batch2)
    finalize(rows, batch1, batch2, history, start)


if __name__ == "__main__":
    main()
