"""Write the E2 replication artifacts. The actor is loaded in eval mode and not updated."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .protocol import (
    BUDGET_SECONDS,
    CHECKPOINT,
    E1_SUMMARY,
    EPISODE_COUNT,
    EPISODE_STEPS,
    GATE_BLOCKED,
    GATE_INCONCLUSIVE,
    INSTRUCTION_COMMIT,
    LAMBDA_CANDIDATES,
    LAMBDA_REFERENCE,
    LAMBDA_SELECTED,
    OUTPUT,
    REPORT,
    ROOT,
    WATCHED,
    choose_gate,
)
from .report import render_report
from .rollout import checkpoint_sha256, load_frozen_actor, roll_episode, train_scenarios
from .scale import summarize


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _actor_digest(actor) -> str:
    digest = hashlib.sha256()
    for name, value in actor.state_dict().items():
        digest.update(str(name).encode("utf-8"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _gap(left: dict[str, float], right: dict[str, float]) -> float:
    return float(max(abs(float(left[key]) - float(right[key])) for key in left))


def _episode_public(episode: dict) -> dict:
    rows = episode["rows"]
    served = float(sum(float(row["served_bits"]) for row in rows))
    ordinary = float(sum(float(row["served_ordinary_bits"]) for row in rows))
    tagged = float(sum(float(row["served_tagged_bits"]) for row in rows))
    generated = float(sum(float(row["generated_bits"]) for row in rows))
    admitted = float(sum(float(row["admitted_bits"]) for row in rows))
    dropped = float(sum(float(row["dropped_bits"]) for row in rows))
    draws = int(sum(int(row["tag_draws"]) for row in rows))
    tagged_draws = int(sum(int(row["tagged_draws"]) for row in rows))
    return {
        "scenario_id": int(episode["scenario_id"]),
        "elapsed_s": float(episode["elapsed_s"]),
        "steps": int(len(rows)),
        "served_bits": served,
        "served_ordinary_bits": ordinary,
        "served_tagged_bits": tagged,
        "urgency_served_bits": float(sum(float(row["urgency_served_bits"]) for row in rows)),
        "generated_bits": generated,
        "admitted_bits": admitted,
        "admitted_ordinary_bits": float(sum(float(row["admitted_ordinary_bits"]) for row in rows)),
        "admitted_tagged_bits": float(sum(float(row["admitted_tagged_bits"]) for row in rows)),
        "dropped_bits": dropped,
        "dropped_ordinary_bits": float(sum(float(row["dropped_ordinary_bits"]) for row in rows)),
        "dropped_tagged_bits": float(sum(float(row["dropped_tagged_bits"]) for row in rows)),
        "snap_bits": float(sum(float(row["snap_bits"]) for row in rows)),
        "payload_bits": float(rows[-1]["payload_bits"]) if rows else 0.0,
        "final_q_bits": float(rows[-1]["q_bits"]) if rows else 0.0,
        "remaining_ordinary_bits": episode["remaining_ordinary_bits"],
        "remaining_tagged_bits": episode["remaining_tagged_bits"],
        "max_ledger_gap_bits": float(max(float(row["ledger_gap_bits"]) for row in rows)) if rows else None,
        "max_conservation_error_bits": float(episode["max_conservation_error_bits"]),
        "max_official_reward_gap": float(episode["max_official_reward_gap"]),
        "safety_violations": int(sum(int(row["safety_violation"]) for row in rows)),
        "projection_steps": int(sum(int(row["projected"]) for row in rows)),
        "constraint_failures": int(sum(int(row["constraint_failure"]) for row in rows)),
        "tag_draws": draws,
        "tagged_draws": tagged_draws,
        "tag_rate": None if draws == 0 else float(tagged_draws) / float(draws),
        "sum_penalty": float(sum(float(row["penalty"]) for row in rows)),
        "sum_projection_penalty": float(sum(float(row["projection_penalty"]) for row in rows)),
        "sum_residual_penalty": float(sum(float(row["residual_penalty"]) for row in rows)),
        "served_split_gap_bits": abs(ordinary + tagged - served),
        "arrival_split_gap_bits": abs(generated - admitted - dropped),
        "queue_split_gap_bits": None
        if episode["remaining_ordinary_bits"] is None
        else abs(float(episode["remaining_ordinary_bits"]) + float(episode["remaining_tagged_bits"]) - (float(rows[-1]["q_bits"]) if rows else 0.0)),
    }


def _accounting_ok(episode: dict) -> bool:
    public = _episode_public(episode)
    if int(public["steps"]) != int(EPISODE_STEPS):
        return False
    if float(public["max_ledger_gap_bits"]) > 1.0 or float(public["max_conservation_error_bits"]) > 1.0:
        return False
    if float(public["max_official_reward_gap"]) > 1.0e-6:
        return False
    if float(public["served_split_gap_bits"]) > 1.0 or float(public["arrival_split_gap_bits"]) > 1.0:
        return False
    if float(public["queue_split_gap_bits"]) > 1.0:
        return False
    for row in episode["rows"]:
        if abs(float(row["generated_bits"]) - float(row["tag_draws"]) * 1.0e6) > 1.0:
            return False
        if abs(float(row["served_ordinary_bits"]) + float(row["served_tagged_bits"]) - float(row["served_bits"])) > 1.0:
            return False
        if abs(float(row["generated_bits"]) - float(row["admitted_bits"]) - float(row["dropped_bits"])) > 1.0:
            return False
    forbidden = {"remaining_arrivals", "arrivals_future", "future_trajectory"}
    if forbidden.intersection(episode["observation_keys"]) or any("tag" in key for key in episode["observation_keys"]):
        return False
    return True


def _same_trace(left: dict, right: dict) -> bool:
    if left["tag_rows"] != right["tag_rows"]:
        return False
    if len(left["action_checksums"]) != len(right["action_checksums"]):
        return False
    if not np.allclose(left["action_checksums"], right["action_checksums"], atol=1.0e-8, rtol=0.0):
        return False
    if _gap(left["fingerprint"], right["fingerprint"]) != 0.0:
        return False
    served_left = [float(row["served_bits"]) for row in left["rows"]]
    served_right = [float(row["served_bits"]) for row in right["rows"]]
    payload_left = [float(row["payload_bits"]) for row in left["rows"]]
    payload_right = [float(row["payload_bits"]) for row in right["rows"]]
    return served_left == served_right and payload_left == payload_right


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _schema() -> dict:
    return {
        "schema_version": "frozen_mappo_priority_tag_reward_scale_v1",
        "actor": "phase6g_seed0.pt mean action, eval mode, weights frozen",
        "assignment": "phase6g sample_assignment G1_RATE hold 15, per-scenario numpy seed 6000000+scenario_id",
        "synthetic_tag": "E1 protocol tag_rate 0.25 beta_tag 0.25 SeedSequence[606, scenario_id, admission_step, buoy]",
        "lambda_candidates": list(LAMBDA_CANDIDATES),
        "lambda_reference": LAMBDA_REFERENCE,
        "lambda_selected": None,
        "penalty_split": "c_project and c_residual recomputed with the formal low_level_reward formula",
        "budget_seconds_for_eight_episodes": BUDGET_SECONDS,
        "preregistered_bounds": {
            "urgency_fraction_span_max": 2.0,
            "e1_relative_tolerance": 0.5,
            "p95_bonus_over_penalty_max": 1.0,
            "mean_bonus_over_payload_max": 0.01,
        },
        "held_out_splits_read": False,
        "trained": False,
    }


def _emit(summary: dict, step_rows: list[dict], episode_rows: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "step_rows.csv", step_rows)
    _write_csv(OUTPUT / "episode_rows.csv", episode_rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "schema.json").write_text(json.dumps(_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(summary), encoding="utf-8")
    print(summary["gate"], summary.get("reason"), summary.get("estimated_eight_s"))


def _base_summary(scenarios: tuple[int, ...], sha: str, e1_sha: str, watched: dict[str, str], watched_ok: bool) -> dict:
    return {
        "gate": GATE_BLOCKED,
        "reason": "",
        "instruction_commit": INSTRUCTION_COMMIT,
        "scenarios": [int(item) for item in scenarios],
        "episodes_completed": 0,
        "elapsed_first_s": None,
        "estimated_eight_s": None,
        "budget_seconds": BUDGET_SECONDS,
        "budget_exceeded": False,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha,
        "checkpoint_matches_e1": bool(sha == e1_sha),
        "e1_checkpoint_sha256": e1_sha,
        "watched_sha256": watched,
        "watched_matches_e1": bool(watched_ok),
        "actor_digest_unchanged": None,
        "integrity": {},
        "trained": False,
        "formal_system_modified": False,
        "held_out_splits_read": False,
        "lambda_selected": LAMBDA_SELECTED,
        "episodes": [],
        "scale": None,
    }


def main() -> None:
    scenarios = train_scenarios()
    e1 = json.loads(E1_SUMMARY.read_text(encoding="utf-8"))
    e1_sha = str(e1["watched_sha256"][str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/")])
    sha = checkpoint_sha256()
    watched = {relative: _sha256(ROOT / relative) if (ROOT / relative).is_file() else "missing" for relative in WATCHED}
    watched_ok = all(watched[relative] == str(e1["watched_sha256"][relative]) for relative in WATCHED)
    summary = _base_summary(scenarios, sha, e1_sha, watched, watched_ok)
    if tuple(scenarios) != tuple(range(321000, 321008)) or sha != e1_sha or not watched_ok:
        summary["reason"] = "registry, checkpoint, or watched formal hash mismatch"
        _emit(summary, [], [])
        return
    actor = load_frozen_actor()
    before = _actor_digest(actor)
    first = roll_episode(actor, int(scenarios[0]), use_ledger=True)
    summary["elapsed_first_s"] = float(first["elapsed_s"])
    summary["estimated_eight_s"] = float(first["elapsed_s"]) * float(EPISODE_COUNT)
    summary["episodes_completed"] = 1
    summary["episodes"] = [_episode_public(first)]
    if float(summary["estimated_eight_s"]) > float(BUDGET_SECONDS):
        summary["budget_exceeded"] = True
        summary["reason"] = "estimated eight episodes exceed 600 seconds"
        summary["integrity"] = {"first_episode_accounting_ok": _accounting_ok(first)}
        _emit(summary, list(first["rows"]), summary["episodes"])
        return
    repeated = roll_episode(actor, int(scenarios[0]), use_ledger=True)
    plain = roll_episode(actor, int(scenarios[0]), use_ledger=False)
    episodes = [first]
    for scenario in scenarios[1:]:
        episodes.append(roll_episode(actor, int(scenario), use_ledger=True))
    after = _actor_digest(actor)
    accounting = all(_accounting_ok(episode) for episode in episodes)
    repeated_ok = _same_trace(first, repeated)
    plain_gap = _gap(first["fingerprint"], plain["fingerprint"])
    plain_actions = bool(np.allclose(first["action_checksums"], plain["action_checksums"], atol=1.0e-8, rtol=0.0))
    integrity_ok = bool(accounting and repeated_ok and plain_gap == 0.0 and plain_actions and before == after)
    summary["episodes_completed"] = int(len(episodes))
    summary["episodes"] = [_episode_public(episode) for episode in episodes]
    summary["actor_digest_unchanged"] = bool(before == after)
    summary["integrity"] = {
        "accounting_ok": accounting,
        "repeat_trace_match": repeated_ok,
        "ledger_off_fingerprint_gap": plain_gap,
        "ledger_off_action_match": plain_actions,
        "actor_digest_unchanged": bool(before == after),
        "observation_keys": list(first["observation_keys"]),
    }
    summary["scale"] = summarize(episodes) if accounting else None
    scale = summary["scale"] or {}
    summary["gate"] = choose_gate(
        integrity_ok=integrity_ok,
        episodes=int(len(episodes)),
        budget_exceeded=False,
        positive_candidate_supported=bool(scale.get("positive_candidate_supported", False)),
        ratio_span_ok=bool(scale.get("ratio_span_ok", False)),
        e1_scale_close=bool(scale.get("e1_scale_close", False)),
    )
    if not integrity_ok:
        summary["reason"] = "ledger, accounting, repeatability, or frozen-weight check failed"
    elif summary["gate"] == GATE_INCONCLUSIVE:
        summary["reason"] = "正候选超出预注册惩罚界限，且 E1 脚本不能代表冻结策略"
    elif summary["gate"] != GATE_BLOCKED:
        summary["reason"] = str(summary["gate"])
    _emit(summary, [row for episode in episodes for row in episode["rows"]], summary["episodes"])


if __name__ == "__main__":
    main()
