"""Trial gate, manifests, and the registered six-run budget."""
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

from .assess import assess_effect, recommendation
from .protocol import (
    E13_RESULT_COMMIT,
    EFFECT_MEASURABLE,
    FORMAL_ACTOR_SHA256,
    GATE_BLOCKED,
    GATE_COMPLETE,
    INSTRUCTION_COMMIT,
    OUTPUT,
    ROOT,
    SCHEMA_VERSION,
    SHARED_SOURCES,
    TRAIN_SEEDS,
    UPDATES,
    assert_registered_ppo,
    schema,
)
from .report import write_report
from .train import PilotStopped, train_seed

HISTORICAL_ROOTS = (
    "results/phase6i_priority_tag_queue_lookahead",
    "results/phase6i_priority_tag_credit_audit",
    "results/phase6i_priority_tag_direct_service_reward",
    "results/phase6i_priority_tag_signal_audit",
    "results/phase6i_priority_tag_reward",
    "results/phase5_hgnn_urgency/checkpoints",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints",
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
UPDATE_FIELDS = (
    "variant", "model_seed", "ppo_update", "mean_served_bits", "mean_served_tagged_bits",
    "mean_served_ordinary_bits", "mean_bonus_age", "mean_bonus_tag", "mean_motion_penalty",
    "projection_rate", "safety_violation_count", "constraint_failure_count",
    "positive_tag_bonus_steps", "lookahead_nonzero_input_uav_steps", "explained_variance",
    "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
    "actor_delta_norm", "critic_delta_norm", "joint_pre_norm", "joint_post_norm", "optimizer_steps",
)
SHADOW_FIELDS = (
    "source_variant", "model_seed", "ppo_update", "epoch", "minibatch", "repeat_ok", "repeat_max_abs",
    "actor_delta_j", "actor_delta_s", "actor_delta_abs_diff", "cosine", "r",
    "j_norm_zero", "s_norm_zero", "new_column_delta_j", "new_column_delta_s", "new_column_abs_diff",
    "policy_mu_l2", "policy_logprob_mean_abs", "layer_l2_json", "source_state_sha256", "shadow_adam_steps",
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


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _assert_runtime() -> None:
    observed = _runtime()
    expected = {"python": "3.14.2", "numpy": "2.4.1", "torch": "2.12.0+cpu", "cuda": False}
    for key, value in expected.items():
        if observed[key] != value:
            raise RuntimeError(f"runtime {key} is {observed[key]!r}, not the verified {value!r}")


def _git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def _assert_source_ready() -> str:
    head = source_commit()
    if head == INSTRUCTION_COMMIT:
        raise RuntimeError("the implementation commit is missing")
    ancestor = _git(["merge-base", "--is-ancestor", INSTRUCTION_COMMIT, "HEAD"])
    if ancestor.returncode != 0:
        raise RuntimeError("HEAD does not descend from the optimizer-intervention instruction")
    dirty = _git(["status", "--porcelain", "--", "src", "tests"])
    if dirty.stdout.strip():
        raise RuntimeError("src or tests changed after the implementation commit")
    formal = file_sha256(ROOT / "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt")
    if formal != FORMAL_ACTOR_SHA256:
        raise RuntimeError("formal low-level checkpoint hash changed")
    assert_registered_ppo()
    _assert_runtime()
    return head


def _tree_hashes() -> dict[str, str]:
    found = {}
    for relative in HISTORICAL_ROOTS:
        root = ROOT / relative
        if not root.exists():
            found[relative] = "MISSING"
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            key = str(path.relative_to(ROOT)).replace("\\", "/")
            found[key] = file_sha256(path)
    for relative in SHARED_SOURCES + WATCHED_PATHS:
        found[relative] = file_sha256(ROOT / relative)
    return found


def _package_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    found = {}
    for path in sorted(root.glob("*.py")):
        found[str(path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(path)
    test = ROOT / "tests" / "test_priority_tag_optimizer_intervention.py"
    if test.exists():
        found[str(test.relative_to(ROOT)).replace("\\", "/")] = file_sha256(test)
    return found


def _passing_seeds(payload) -> list[int]:
    found = []

    def walk(node) -> None:
        if isinstance(node, dict):
            if "passing_seeds" in node:
                found.append(int(node["passing_seeds"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return found


def _frozen_gates() -> dict:
    e12 = json.loads((ROOT / "results/phase6i_priority_tag_queue_lookahead/summary.json").read_text(encoding="utf-8"))
    e13 = json.loads((ROOT / "results/phase6i_priority_tag_credit_audit/summary.json").read_text(encoding="utf-8"))
    if e12.get("scientific_gate") != "QUEUE_LOOKAHEAD_BENCHMARK_FAIL":
        raise RuntimeError("frozen E12 gate is no longer the registered FAIL")
    if e13.get("scientific_gate") != "TAG_CREDIT_AUDIT_COMPLETE":
        raise RuntimeError("frozen E13 gate changed")
    passing = _passing_seeds(e12)
    if not passing or any(value != 0 for value in passing):
        raise RuntimeError("frozen E12 passing seed count changed")
    return {
        "e12_gate": e12["scientific_gate"],
        "e12_passing_seeds": 0,
        "e13_gate": e13["scientific_gate"],
        "e13_recommendation": e13.get("recommendation"),
        "e13_result_commit": E13_RESULT_COMMIT,
    }


def _shadow_csv_row(row: dict) -> dict:
    copied = dict(row)
    copied["layer_l2_json"] = row.get("layer_gaps", [])
    return copied


def _public_run(run: dict) -> dict:
    keys = (
        "variant", "model_seed", "complete", "ppo_updates", "adam_steps", "shadow_adam_steps",
        "safety_violation_count", "constraint_failure_count", "mean_served_bits", "mean_served_tagged_bits",
        "mean_approx_kl", "max_approx_kl", "final_actor_sha256", "final_critic_sha256", "checkpoint",
        "first_rollout_max_abs", "initialization", "error",
    )
    return {key: run.get(key) for key in keys}


def _collect_tables(runs: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    updates, steps, shadows, events = [], [], [], []
    for run in runs:
        updates.extend(run.get("update_rows", []))
        steps.extend(run.get("step_rows", []))
        shadows.extend(_shadow_csv_row(row) for row in run.get("shadow_rows", []))
        events.extend(run.get("events", []))
    return updates, steps, shadows, events


def _assert_first_rollout(seed_runs: list[dict]) -> None:
    if len(seed_runs) != 2 or not all(run.get("complete") for run in seed_runs):
        raise RuntimeError("a seed did not finish both clip groups")
    left = seed_runs[0]["first_rewards"]
    right = seed_runs[1]["first_rewards"]
    if left is None or right is None or tuple(left.shape) != tuple(right.shape):
        raise RuntimeError("first rollout was not recorded")
    gap = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    for run in seed_runs:
        run["first_rollout_max_abs"] = gap
    if gap != 0.0:
        raise RuntimeError(f"first rollout diverged before the clip intervention, max abs {gap}")


def _base_summary(runs: list[dict], *, trial_gate: str, reason: str, implementation: str, before: dict, after: dict) -> dict:
    updates, steps, shadows, events = _collect_tables(runs)
    effect = assess_effect(shadows) if shadows else {
        "effect_assessment": "MIXED_OR_INSUFFICIENT",
        "measurable_seed_count": 0,
        "valid_effect_comparisons": 0,
        "above_threshold_comparisons": 0,
        "complete_coverage": False,
        "per_seed": [],
        "r_threshold": 0.01,
        "threshold_role": "descriptive_direction_screen",
    }
    safety = int(sum(int(run.get("safety_violation_count") or 0) for run in runs))
    constraints = int(sum(int(run.get("constraint_failure_count") or 0) for run in runs))
    complete = trial_gate == GATE_COMPLETE
    choice = recommendation(
        effect["effect_assessment"],
        trial_complete=complete,
        constraint_failure_count=constraints,
    )
    if not complete:
        choice = "do_not_extend_the_pilot"
    return {
        "scientific_gate": trial_gate,
        "effect_assessment": effect["effect_assessment"],
        "performance_conclusion": None,
        "ppo_updates": int(sum(int(run.get("ppo_updates") or 0) for run in runs)),
        "adam_steps": int(sum(int(run.get("adam_steps") or 0) for run in runs)),
        "shadow_adam_steps": int(sum(int(run.get("shadow_adam_steps") or 0) for run in runs)),
        "shadow_source_states": int(len(shadows)),
        "registered_ppo_updates": int(len(TRAIN_SEEDS) * 2 * UPDATES),
        "release_eligible": False,
        "merged": False,
        "implementation_commit": implementation,
        "instruction_commit": INSTRUCTION_COMMIT,
        "schema_version": SCHEMA_VERSION,
        "reason": reason,
        "effect": effect,
        "recommendation": choice,
        "runs": [_public_run(run) for run in runs],
        "safety_violation_count": safety,
        "constraint_failure_count": constraints,
        "safety_events": len(events),
        "frozen_gates": _frozen_gates(),
        "historical_hash_unchanged": before == after,
        "historical_file_count": len(before),
        "source_hashes": {key: before[key] for key in SHARED_SOURCES if key in before},
        "package_hashes": _package_hashes(),
        "runtime": _runtime(),
        "rng_roles": {
            "model_seed": "seed_everything(model_seed) immediately before workers",
            "scenario_assignment": "321000000 + model_seed * 10000 + env_index",
            "pool": "TRAIN_POOL",
            "eval_process_seed_used": False,
            "shadow": "python, numpy, and torch RNG restored before the real clip",
        },
        "test_comparison": None,
    }


def _persist(runs: list[dict], summary: dict) -> None:
    updates, steps, shadows, events = _collect_tables(runs)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "training_update_rows.csv", UPDATE_FIELDS, updates)
    _write_csv(OUTPUT / "optimizer_step_rows.csv", STEP_FIELDS, steps)
    _write_csv(OUTPUT / "shadow_update_rows.csv", SHADOW_FIELDS, shadows)
    _write_json(OUTPUT / "summary.json", summary)
    _write_json(OUTPUT / "schema.json", schema())
    _write_json(OUTPUT / "runtime_manifest.json", summary["runtime"] | {"rng_roles": summary["rng_roles"]})
    _write_json(
        OUTPUT / "repeat_checks.json",
        [
            {
                "source_variant": row["source_variant"],
                "model_seed": row["model_seed"],
                "ppo_update": row["ppo_update"],
                "repeat_ok": row["repeat_ok"],
                "repeat_max_abs": row["repeat_max_abs"],
                "r": row["r"],
                "cosine": row["cosine"],
                "j_norm_zero": row["j_norm_zero"],
            }
            for row in shadows
        ],
    )
    _write_json(
        OUTPUT / "shadow_state_manifest.json",
        [
            {
                "source_variant": row["source_variant"],
                "model_seed": row["model_seed"],
                "ppo_update": row["ppo_update"],
                "source_state_sha256": row["source_state_sha256"],
                "path": f"results/phase6i_priority_tag_optimizer_intervention/shadow_states/{str(row['source_variant']).lower()}_seed{int(row['model_seed'])}_update{int(row['ppo_update'])}.pt",
            }
            for row in shadows
        ],
    )
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
                "critic_sha256": run.get("final_critic_sha256"),
            }
            for run in runs
        ],
    )
    _write_json(OUTPUT / "safety_events.json", events)
    _write_json(
        OUTPUT / "init_hashes.json",
        [run.get("initialization") for run in runs],
    )
    _write_json(
        OUTPUT / "evidence_gaps.json",
        [
            "这 32 次 PPO 更新只识别优化器机制，不能判断 300 次更新后的策略质量。",
            "训练曲线来自训练池滚动，不是独立场景样本，也不是 DEV benchmark。",
            "影子步骤是同一源状态上的一次离线 Adam 更新，不计入 PPO 训练预算，也不会写回正式训练。",
            "第 1 次更新的 Adam 矩为空，只记录，不进入效应规则。",
            "第 16 和 32 次更新的矩来自本次试点，不是 E12 的历史矩。",
            "没有读取保留场景或验证场景。",
        ],
    )
    blockers = [
        "E12 配对 benchmark 仍是 QUEUE_LOOKAHEAD_BENCHMARK_FAIL。",
        "完整历史测试套件仍有既有失败，不能写成全绿。",
        "本试点没有性能通过标准，release_eligible 保持 false。",
        "没有合并。",
    ]
    if summary["scientific_gate"] != GATE_COMPLETE:
        blockers.append(f"机制试验未完成：{summary['reason']}")
    if summary["effect_assessment"] != EFFECT_MEASURABLE:
        blockers.append("裁剪分组没有达到预注册的可测更新效应，不能据此扩大训练。")
    _write_json(OUTPUT / "release_blockers.json", blockers)
    write_report(summary)


def execute_trial() -> dict:
    implementation = _assert_source_ready()
    before = _tree_hashes()
    _write_json(OUTPUT / "source_manifest.json", {"phase": "before", "implementation_commit": implementation, "files": before})
    runs: list[dict] = []
    reason = "registered clip-group pilot finished"
    gate = GATE_COMPLETE
    try:
        for seed in TRAIN_SEEDS:
            seed_runs = train_seed(int(seed))
            runs.extend(seed_runs)
            _assert_first_rollout(seed_runs)
    except PilotStopped as exc:
        runs.extend(exc.runs)
        gate = GATE_BLOCKED
        reason = str(exc)
    except Exception as exc:
        gate = GATE_BLOCKED
        reason = f"{type(exc).__name__}: {exc}"
    after = _tree_hashes()
    if before != after:
        gate = GATE_BLOCKED
        reason = "historical source or result hash changed during the pilot"
    if gate == GATE_COMPLETE:
        expected_adam = int(len(TRAIN_SEEDS) * 2 * UPDATES * 4 * 4)
        expected_shadow = int(len(TRAIN_SEEDS) * 2 * 3 * 3)
        summary_probe = _base_summary(runs, trial_gate=gate, reason=reason, implementation=implementation, before=before, after=after)
        if summary_probe["ppo_updates"] != int(len(TRAIN_SEEDS) * 2 * UPDATES) or summary_probe["adam_steps"] != expected_adam or summary_probe["shadow_adam_steps"] != expected_shadow:
            gate = GATE_BLOCKED
            reason = "consumed optimizer budget does not match the registration"
        if any(not run.get("complete") for run in runs) or len(runs) != 6:
            gate = GATE_BLOCKED
            reason = "not all six pilot runs completed"
    summary = _base_summary(runs, trial_gate=gate, reason=reason, implementation=implementation, before=before, after=after)
    _write_json(OUTPUT / "source_manifest.json", {"phase": "after", "implementation_commit": source_commit(), "before_after_match": before == after, "files": after})
    _persist(runs, summary)
    print(
        f"trial_gate={summary['scientific_gate']} effect={summary['effect_assessment']} "
        f"ppo={summary['ppo_updates']} adam={summary['adam_steps']} shadow={summary['shadow_adam_steps']}",
        flush=True,
    )
    return summary


def _normalize(detail: str) -> str:
    text = str(detail)
    for root in (
        str(ROOT),
        str(ROOT).replace("\\", "/"),
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
        "comparison_root": "E:/srdp/wt_e14_baseline",
    }


def attach_tests() -> dict:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    baseline = (OUTPUT / "baseline_tests.log").read_text(encoding="utf-8", errors="replace")
    current = (OUTPUT / "repository_tests.log").read_text(encoding="utf-8", errors="replace")
    related = (OUTPUT / "related_tests.log").read_text(encoding="utf-8", errors="replace")
    comparison = compare_logs(baseline, current)
    comparison["related_ok"] = "\nOK\n" in f"\n{related}\n" or related.rstrip().endswith("OK")
    comparison["related_failures"] = parse_unittest(related)["failures"]
    summary["test_comparison"] = comparison
    if comparison["new_failure_count"] != 0 or not comparison["related_ok"]:
        summary["recommendation"] = "do_not_extend_the_pilot"
        summary["reason"] = summary.get("reason", "") + " new tests or the clean-suite comparison failed"
    _write_json(OUTPUT / "test_comparison.json", comparison)
    _write_json(OUTPUT / "summary.json", summary)
    write_report(summary)
    print(f"new_failure_count={comparison['new_failure_count']} related_ok={comparison['related_ok']}", flush=True)
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
