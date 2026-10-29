"""Run the authorized six-by-300 clip comparison and the 224-episode evaluation."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.run import parse_unittest
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import WATCHED_PATHS
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256
from wt.benchmark_v2.phase6i_priority_tag_reward.train import source_commit

from .criteria import choose_gate, score_full
from .evaluate import evaluate_pool, load_full, run_preflight
from .protocol import (
    DIRECTION_CHOSEN,
    E14_RESULT_COMMIT,
    EVAL_EPISODES,
    FORMAL_ACTOR_SHA256,
    GATE_BLOCKED,
    INSTRUCTION_COMMIT,
    OUTPUT,
    ROOT,
    SCHEMA_VERSION,
    SHARED_SOURCES,
    TRAIN_SEEDS,
    UPDATES,
    VARIANTS,
    assert_registered_budget,
    expected_adam_steps,
    schema,
)
from .reconcile import reconcile_e14
from .report import write_report
from .train import RunStopped, train_seed

HISTORICAL_ROOTS = (
    "results/phase6i_priority_tag_optimizer_intervention",
    "results/phase6i_priority_tag_queue_lookahead",
    "results/phase6i_priority_tag_credit_audit",
    "results/phase6i_priority_tag_direct_service_reward",
    "results/phase5_hgnn_urgency/checkpoints",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints",
)
UPDATE_FIELDS = (
    "variant", "model_seed", "ppo_update", "mean_served_bits", "mean_served_tagged_bits", "mean_served_ordinary_bits",
    "mean_bonus_age", "mean_bonus_tag", "mean_motion_penalty", "projection_rate", "positive_tag_bonus_steps",
    "safety_violation_count", "constraint_failure_count", "explained_variance", "approx_kl", "clip_fraction",
    "actor_delta_norm", "critic_delta_norm", "joint_pre_norm", "joint_post_norm", "optimizer_steps",
)
STEP_FIELDS = (
    "variant", "model_seed", "ppo_update", "epoch", "minibatch", "optimizer_step",
    "actor_pre_norm", "critic_pre_norm", "joint_pre_norm", "clip_rule",
    "actor_limit", "critic_limit", "actor_scale", "critic_scale",
    "actor_post_norm", "critic_post_norm", "joint_post_norm", "joint_bound",
    "learning_rate", "beta1", "beta2", "adam_eps", "adam_step_count",
    "actor_moment_norm", "actor_second_moment_norm", "critic_moment_norm", "critic_second_moment_norm",
    "actor_delta_norm", "critic_delta_norm", "new_column_delta_norm",
    "approx_kl", "clip_fraction", "value_loss", "entropy", "policy_loss", "action_std",
)
SCENARIO_FIELDS = (
    "variant", "model_seed", "scenario_id", "steps", "payload_bits", "served_tagged_bits", "served_ordinary_bits",
    "age_service_bits", "bonus_age_sum", "bonus_tag_sum", "queue_age_mean", "queue_age_max",
    "dropped_tagged_bits", "dropped_ordinary_bits", "final_queue_bits", "final_tagged_queue_bits",
    "final_ordinary_queue_bits", "sinr_mean", "interference_mean", "path_length_m", "projection_rate",
    "safety_violations", "constraint_failures", "env_reset_seed", "assignment_rng_seed",
    "process_seed_role", "model_seed_role", "use_lookahead",
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    return value


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (list, dict)):
        return json.dumps(_plain(value), ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def _runtime() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "torch_threads": 1,
        "deterministic_algorithms": True,
        "device": "cpu",
    }


def _assert_ready() -> str:
    decision = json.loads((OUTPUT / "direction_decision.json").read_text(encoding="utf-8"))
    if decision.get("direction_status") != DIRECTION_CHOSEN:
        raise RuntimeError("direction A is not the recorded choice")
    head = source_commit()
    if head == INSTRUCTION_COMMIT:
        raise RuntimeError("the implementation commit is missing")
    if _git(["merge-base", "--is-ancestor", INSTRUCTION_COMMIT, "HEAD"]).returncode != 0:
        raise RuntimeError("HEAD does not descend from the clip-benchmark instruction")
    dirty = _git(["status", "--porcelain", "--", "src", "tests"])
    if dirty.stdout.strip():
        raise RuntimeError("src or tests changed after the implementation commit")
    if file_sha256(ROOT / "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt") != FORMAL_ACTOR_SHA256:
        raise RuntimeError("formal low-level checkpoint hash changed")
    observed = _runtime()
    expected = {"python": "3.14.2", "numpy": "2.4.1", "torch": "2.12.0+cpu", "cuda": False}
    for key, value in expected.items():
        if observed[key] != value:
            raise RuntimeError(f"runtime {key} is {observed[key]!r}, not the verified {value!r}")
    assert_registered_budget()
    return head


def _tree_hashes() -> dict[str, str]:
    found = {}
    for relative in HISTORICAL_ROOTS:
        root = ROOT / relative
        if not root.exists():
            found[relative] = "MISSING"
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                found[str(path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(path)
    for relative in tuple(dict.fromkeys(SHARED_SOURCES + WATCHED_PATHS)):
        found[relative] = file_sha256(ROOT / relative)
    return found


def _frozen_gates() -> dict:
    e12 = json.loads((ROOT / "results/phase6i_priority_tag_queue_lookahead/summary.json").read_text(encoding="utf-8"))
    e13 = json.loads((ROOT / "results/phase6i_priority_tag_credit_audit/summary.json").read_text(encoding="utf-8"))
    e14 = json.loads((ROOT / "results/phase6i_priority_tag_optimizer_intervention/summary.json").read_text(encoding="utf-8"))
    if e12.get("scientific_gate") != "QUEUE_LOOKAHEAD_BENCHMARK_FAIL" or e13.get("scientific_gate") != "TAG_CREDIT_AUDIT_COMPLETE":
        raise RuntimeError("frozen E12 or E13 gate changed")
    if e14.get("scientific_gate") != "OPTIMIZER_INTERVENTION_COMPLETE" or e14.get("effect_assessment") != "MEASURABLE_OPTIMIZER_EFFECT":
        raise RuntimeError("frozen E14 conclusion changed")
    return {
        "e12_gate": e12["scientific_gate"],
        "e13_gate": e13["scientific_gate"],
        "e14_gate": e14["scientific_gate"],
        "e14_effect": e14["effect_assessment"],
        "e14_result_commit": E14_RESULT_COMMIT,
    }


def _public_run(run: dict) -> dict:
    keys = (
        "variant", "model_seed", "complete", "ppo_updates", "adam_steps", "safety_violation_count",
        "constraint_failure_count", "mean_served_bits", "mean_served_tagged_bits", "mean_approx_kl",
        "max_approx_kl", "final_actor_sha256", "checkpoint", "snapshots", "first_rollout_max_abs",
        "initialization", "error",
    )
    return {key: run.get(key) for key in keys}


def _tables(runs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    updates, steps, events = [], [], []
    for run in runs:
        updates.extend(run.get("update_rows", []))
        steps.extend(run.get("step_rows", []))
        events.extend(run.get("events", []))
    return updates, steps, events


def _assert_first(seed_runs: list[dict]) -> None:
    if len(seed_runs) != 2 or not all(run.get("complete") for run in seed_runs):
        raise RuntimeError("a seed did not finish both clip groups")
    gap = float(np.max(np.abs(np.asarray(seed_runs[0]["first_rewards"]) - np.asarray(seed_runs[1]["first_rewards"]))))
    for run in seed_runs:
        run["first_rollout_max_abs"] = gap
    if gap != 0.0:
        raise RuntimeError(f"first rollout diverged before the clip rule, max abs {gap}")


def _summary(runs, *, gate: str, reason: str, implementation: str, before: dict, after: dict, score, episodes: int, preflight) -> dict:
    updates, steps, events = _tables(runs)
    return {
        "direction_status": DIRECTION_CHOSEN,
        "scientific_gate": gate,
        "release_eligible": False,
        "merged": False,
        "performance_conclusion_is_not_a_payload_significance_claim": True,
        "ppo_updates": int(sum(int(run.get("ppo_updates") or 0) for run in runs)),
        "adam_steps": int(sum(int(run.get("adam_steps") or 0) for run in runs)),
        "evaluation_episodes": int(episodes),
        "registered_ppo_updates": int(len(TRAIN_SEEDS) * len(VARIANTS) * UPDATES),
        "registered_adam_steps": int(expected_adam_steps()),
        "implementation_commit": implementation,
        "instruction_commit": INSTRUCTION_COMMIT,
        "schema_version": SCHEMA_VERSION,
        "reason": reason,
        "score": score,
        "runs": [_public_run(run) for run in runs],
        "training_safety_violation_count": int(sum(int(run.get("safety_violation_count") or 0) for run in runs)),
        "training_constraint_failure_count": int(sum(int(run.get("constraint_failure_count") or 0) for run in runs)),
        "training_safety_events": len(events),
        "preflight": preflight,
        "frozen_gates": _frozen_gates(),
        "historical_hash_unchanged": before == after,
        "historical_file_count": len(before),
        "runtime": _runtime(),
        "rng_roles": {
            "model_seed": "seed_everything(model_seed) immediately before each group's workers",
            "scenario_assignment_train": "321000000 + model_seed * 10000 + env_index",
            "scenario_assignment_eval": "7000000 + scenario_id",
            "env_reset_seed": "scenario_id",
            "process_seed": 2,
            "process_seed_role": "historical_inherited_process_context",
        },
        "test_comparison": None,
        "e14_reconciliation_path": "results/phase6i_priority_tag_clip_benchmark/e14_reconciliation.json",
    }


def _persist(runs, summary, scenario_rows, eval_events) -> None:
    updates, steps, events = _tables(runs)
    events.extend(eval_events)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "training_update_rows.csv", UPDATE_FIELDS, updates)
    _write_csv(OUTPUT / "optimizer_step_rows.csv", STEP_FIELDS, steps)
    _write_csv(OUTPUT / "scenario_rows.csv", SCENARIO_FIELDS, scenario_rows)
    _write_json(OUTPUT / "safety_events.json", events)
    _write_json(OUTPUT / "summary.json", summary)
    _write_json(OUTPUT / "schema.json", schema())
    _write_json(OUTPUT / "runtime_manifest.json", summary["runtime"] | {"rng_roles": summary["rng_roles"]})
    _write_json(
        OUTPUT / "checkpoint_manifest.json",
        [
            {
                "variant": run.get("variant"),
                "model_seed": run.get("model_seed"),
                "complete": run.get("complete"),
                "path": run.get("checkpoint"),
                "sha256": file_sha256(ROOT / str(run["checkpoint"])) if run.get("checkpoint") else None,
                "actor_sha256": run.get("final_actor_sha256"),
                "seamless_resume": False,
                "snapshots": run.get("snapshots"),
            }
            for run in runs
        ],
    )
    _write_json(OUTPUT / "snapshot_manifest.json", [item for run in runs for item in run.get("snapshots", [])])
    _write_json(
        OUTPUT / "evidence_gaps.json",
        [
            "E14 的 4 条安全事件没有位置和距离，记为未知。",
            "定点快照不含环境、账本和 option，不能把 32 次试点与本次训练拼成一次无缝 300 次更新。",
            "开发池 321096–321127 已被多轮使用，本次不是未见场景泛化。",
            "payload 区间下界的 -0.5% 是不确定性容忍；均值仍必须非负。通过标准不表示总 payload 显著高于 L0。",
        ],
    )
    blockers = [
        "完整历史测试套件仍有既有失败。",
        "继承分支含未合并实验，不是正式采用的最小差异。",
        "release_eligible 保持 false。",
        "没有合并。",
    ]
    if summary["scientific_gate"] != "CLIP_GROUP_BENCHMARK_PASS":
        blockers.insert(0, f"科学 Gate 为 {summary['scientific_gate']}。")
    if summary["scientific_gate"] in {"CLIP_GROUP_BENCHMARK_FAIL", "CLIP_GROUP_BENCHMARK_INCONCLUSIVE"}:
        blockers.append("本轮采用尝试停止，不改阈值、不换种子、不追加预算。")
    _write_json(OUTPUT / "release_blockers.json", blockers)
    write_report(summary)


def execute_trial() -> dict:
    implementation = _assert_ready()
    reconciliation = reconcile_e14()
    _write_json(OUTPUT / "e14_reconciliation.json", reconciliation)
    before = _tree_hashes()
    _write_json(OUTPUT / "source_manifest.json", {"phase": "before", "implementation_commit": implementation, "files": before})
    runs: list[dict] = []
    preflight = None
    score = None
    scenario_rows: list[dict] = []
    eval_events: list[dict] = []
    gate = "CLIP_GROUP_BENCHMARK_BLOCKED"
    reason = "registered full-budget comparison finished"
    try:
        preflight = run_preflight()
        _write_json(OUTPUT / "preflight_summary.json", preflight)
        if not preflight["matches_saved"] or float(preflight["max_bit_gap"]) != 0.0 or float(preflight["path_gap_m"]) != 0.0:
            raise RuntimeError("L0 scenario 321101 did not match the frozen row")
        for seed in TRAIN_SEEDS:
            seed_runs = train_seed(int(seed))
            runs.extend(seed_runs)
            _assert_first(seed_runs)
        after_train = _tree_hashes()
        if before != after_train:
            raise RuntimeError("historical files changed during training")
        if len(runs) != 6 or any(not run.get("complete") for run in runs):
            raise RuntimeError("not all six full-budget runs completed")
        consumed_ppo = sum(int(run["ppo_updates"]) for run in runs)
        consumed_adam = sum(int(run["adam_steps"]) for run in runs)
        if consumed_ppo != 1800 or consumed_adam != int(expected_adam_steps()):
            raise RuntimeError(f"consumed budget {consumed_ppo} PPO / {consumed_adam} Adam does not match the registration")
        actors = {(variant, int(seed)): load_full(variant, int(seed)) for variant in VARIANTS for seed in TRAIN_SEEDS}
        scenario_rows, eval_events = evaluate_pool(actors)
        score = score_full(scenario_rows)
        gate = choose_gate(tests_ok=True, train_ok=True, hashes_ok=True, preflight_ok=True, score=score)
    except RunStopped as exc:
        runs.extend(exc.runs)
        gate = GATE_BLOCKED
        reason = str(exc)
    except Exception as exc:
        gate = GATE_BLOCKED
        reason = f"{type(exc).__name__}: {exc}"
    after = _tree_hashes()
    if before != after and gate != GATE_BLOCKED:
        gate = GATE_BLOCKED
        reason = "historical source or result hash changed"
    elif before != after and "historical" not in reason:
        reason = reason + " historical source or result hash changed"
    summary = _summary(
        runs,
        gate=gate,
        reason=reason,
        implementation=implementation,
        before=before,
        after=after,
        score=score,
        episodes=len(scenario_rows),
        preflight=preflight,
    )
    summary["gate_before_repository_tests"] = gate
    _write_json(OUTPUT / "source_manifest.json", {"phase": "after", "implementation_commit": source_commit(), "before_after_match": before == after, "files": after})
    _persist(runs, summary, scenario_rows, eval_events)
    print(
        f"scientific_gate={summary['scientific_gate']} ppo={summary['ppo_updates']} adam={summary['adam_steps']} episodes={summary['evaluation_episodes']}",
        flush=True,
    )
    return summary


def _normalize(detail: str) -> str:
    text = str(detail)
    for root in (
        str(ROOT),
        str(ROOT).replace("\\", "/"),
        r"E:\srdp\wt_e15_baseline",
        "E:/srdp/wt_e15_baseline",
        r"E:\srdp\wt_e14_baseline",
        "E:/srdp/wt_e14_baseline",
        r"E:\srdp\wt_e13_baseline",
        "E:/srdp/wt_e13_baseline",
        r"E:\srdp\wt_e12_baseline",
        "E:/srdp/wt_e12_baseline",
    ):
        text = text.replace(root, "<ROOT>")
    return text


def compare_logs(baseline_text: str, current_text: str) -> dict:
    baseline = parse_unittest(baseline_text)
    current = parse_unittest(current_text)
    baseline_ids = {(item["test_id"], item["kind"]): item for item in baseline["failures"]}
    historical = []
    new_items = []
    for item in current["failures"]:
        previous = baseline_ids.get((item["test_id"], item["kind"]))
        same = previous is not None and previous["exception"] == item["exception"] and _normalize(previous["detail"]) == _normalize(item["detail"])
        if same:
            historical.append(item)
        else:
            new_items.append(item)
    return {
        "baseline_ran": baseline["ran"],
        "repository_ran": current["ran"],
        "baseline_failure_count": len(baseline["failures"]),
        "repository_failure_count": len(current["failures"]),
        "historical_count": len(historical),
        "new_failure_count": len(new_items),
        "new_failures": new_items,
        "comparison_root": "E:/srdp/wt_e15_baseline",
    }


def attach_tests() -> dict:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    baseline = (OUTPUT / "baseline_tests.log").read_text(encoding="utf-8", errors="replace")
    current = (OUTPUT / "repository_tests.log").read_text(encoding="utf-8", errors="replace")
    related = (OUTPUT / "related_tests.log").read_text(encoding="utf-8", errors="replace")
    comparison = compare_logs(baseline, current)
    comparison["related_ok"] = related.rstrip().endswith("OK")
    comparison["related_failures"] = parse_unittest(related)["failures"]
    summary["test_comparison"] = comparison
    if comparison["new_failure_count"] != 0 or not comparison["related_ok"]:
        summary["scientific_gate"] = GATE_BLOCKED
        summary["reason"] = str(summary.get("reason", "")) + " new tests or the clean-suite comparison failed"
    _write_json(OUTPUT / "test_comparison.json", comparison)
    _write_json(OUTPUT / "summary.json", summary)
    write_report(summary)
    print(f"new_failure_count={comparison['new_failure_count']} related_ok={comparison['related_ok']} gate={summary['scientific_gate']}", flush=True)
    return comparison


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "trial":
        execute_trial()
        return
    if command == "tests":
        attach_tests()
        return
    raise SystemExit("usage: trial|tests")


if __name__ == "__main__":
    main()
