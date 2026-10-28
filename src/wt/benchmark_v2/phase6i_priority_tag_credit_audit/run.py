"""Write the audit after the implementation commit. Probes stay behind that commit."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys

import numpy as np
import torch

from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.run import parse_unittest
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256
from wt.benchmark_v2.phase6i_priority_tag_reward.train import source_commit

from .history import audit_history, shared_source_hashes
from .probe import execute_probes
from .protocol import (
    E12_GATE,
    E12_RESULT_COMMIT,
    GATE_BLOCKED,
    GATE_COMPLETE,
    INSTRUCTION_COMMIT,
    OUTPUT,
    REPORT,
    ROOT,
    schema,
)
from .report import render_report


PARTIAL = OUTPUT / "probe_partial.json"
BASELINE_LOG = OUTPUT / "baseline_tests.log"
REPOSITORY_LOG = OUTPUT / "repository_tests.log"
RELATED_LOG = OUTPUT / "related_tests.log"


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        raise TypeError("public audit JSON cannot contain an array")
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def _write_csv(path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _normalize(detail: str) -> str:
    roots = (
        str(ROOT),
        str(ROOT).replace("\\", "/"),
        r"E:\srdp\wt_e13_baseline",
        "E:/srdp/wt_e13_baseline",
        r"E:\srdp\wt_e12_baseline",
        "E:/srdp/wt_e12_baseline",
    )
    text = str(detail)
    for root in roots:
        text = text.replace(root, "<ROOT>")
    return text


def compare_logs(baseline_text: str, current_text: str) -> dict:
    baseline = parse_unittest(baseline_text)
    current = parse_unittest(current_text)
    baseline_ids = {item["test_id"]: item for item in baseline["failures"]}
    historical = []
    new_items = []
    for item in current["failures"]:
        previous = baseline_ids.get(item["test_id"])
        same = previous is not None and previous["exception"] == item["exception"] and _normalize(previous["detail"]) == _normalize(item["detail"])
        if same:
            historical.append(item)
        else:
            new_items.append(item)
    return {
        "related_ok": None,
        "baseline_ran": baseline["ran"],
        "repository_ran": current["ran"],
        "baseline_failure_count": len(baseline["failures"]),
        "repository_failure_count": len(current["failures"]),
        "historical": historical,
        "new_failures": new_items,
        "new_failure_count": len(new_items),
        "baseline_commit": E12_RESULT_COMMIT,
        "baseline_tree": r"E:\srdp\wt_e13_baseline",
    }


def assert_probe_ready() -> str:
    head = source_commit()
    if head == INSTRUCTION_COMMIT:
        raise RuntimeError("probes cannot run on the instruction commit")
    dirty = subprocess.run(
        ["git", "-c", "core.longpaths=true", "status", "--porcelain", "--", "src", "tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise RuntimeError("uncommitted source or test changes are present; probes stay stopped")
    ancestor = subprocess.run(
        ["git", "-c", "core.longpaths=true", "merge-base", "--is-ancestor", INSTRUCTION_COMMIT, head],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("probe commit does not descend from the audit instruction")
    return head


def _runtime() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "torch_threads": int(torch.get_num_threads()),
        "platform": platform.platform(),
    }


def _history_public(history: dict) -> dict:
    copied = dict(history)
    copied.pop("source_hashes", None)
    copied.pop("log_rows", None)
    copied.pop("sample_rows", None)
    return copied


def run_probes() -> dict:
    head = assert_probe_ready()
    history = audit_history()
    probes = execute_probes(history)
    payload = {"implementation_commit": head, "history": history, "probes": probes, "runtime": _runtime()}
    _write_json(PARTIAL, payload)
    return payload


def _advantage_row(batch: dict) -> dict:
    credit = batch.get("credit") or {}
    positive = credit.get("tag_advantage_positive_bonus") or {}
    zero = credit.get("tag_advantage_zero_bonus") or {}
    separate = credit.get("separate_renorm_difference_all") or {}
    return {
        "role": batch["role"],
        "variant": batch["variant"],
        "model_seed": batch["model_seed"],
        "source": batch["source"],
        "published": bool(credit.get("published", False)),
        "float32_linear_gap": credit.get("float32_linear_gap"),
        "shared_scale_additive_gap": credit.get("shared_scale_additive_gap"),
        "float64_diagnostic_linear_gap": credit.get("float64_diagnostic_linear_gap"),
        "positive_count": positive.get("count"),
        "positive_mean_abs": positive.get("mean_abs"),
        "positive_q50_abs": positive.get("q50_abs"),
        "positive_q95_abs": positive.get("q95_abs"),
        "zero_bonus_count": zero.get("count"),
        "zero_bonus_mean_abs": zero.get("mean_abs"),
        "zero_bonus_q50_abs": zero.get("q50_abs"),
        "separate_renorm_mean_abs": separate.get("mean_abs"),
        "float32_distinguishable_steps": credit.get("float32_distinguishable_steps"),
        "future_tag_advantage_on_zero_bonus": credit.get("future_tag_advantage_on_zero_bonus"),
    }


def _gradient_row(batch: dict) -> dict:
    credit = batch.get("credit") or {}
    return {
        "role": batch["role"],
        "variant": batch["variant"],
        "model_seed": batch["model_seed"],
        "source": batch["source"],
        "published": bool(credit.get("published", False)),
        "logprob_max_abs": credit.get("logprob_max_abs"),
        "ratio_max_abs_from_one": credit.get("ratio_max_abs_from_one"),
        "policy_norm_full": credit.get("policy_norm_full"),
        "policy_norm_no_tag_separate": credit.get("policy_norm_no_tag_separate"),
        "policy_norm_base_separate": credit.get("policy_norm_base_separate"),
        "tag_difference_separate_norm": credit.get("tag_difference_separate_norm"),
        "tag_difference_shared_norm": credit.get("tag_difference_shared_norm"),
        "tag_difference_separate_cosine_vs_full": credit.get("tag_difference_separate_cosine_vs_full"),
        "tag_difference_separate_cosine_vs_base": credit.get("tag_difference_separate_cosine_vs_base"),
        "tag_difference_shared_cosine_vs_full": credit.get("tag_difference_shared_cosine_vs_full"),
        "new_input_column_grad_norm": credit.get("new_input_column_grad_norm"),
        "per_uav_gradient_sum_max_abs": credit.get("per_uav_gradient_sum_max_abs"),
        "policy_zero_parameter_count": credit.get("policy_zero_parameter_count"),
        "tag_difference_zero_parameter_count": credit.get("tag_difference_zero_parameter_count"),
    }


def _team_row(batch: dict) -> dict:
    credit = batch.get("credit") or {}
    return {
        "role": batch["role"],
        "variant": batch["variant"],
        "model_seed": batch["model_seed"],
        "source": batch["source"],
        "mask_min": credit.get("mask_min"),
        "mask_max": credit.get("mask_max"),
        "within_step_policy_term_max_abs": credit.get("within_step_policy_term_max_abs"),
        "exposed_uav_steps": credit.get("exposed_uav_steps"),
        "served_uav_steps": credit.get("served_uav_steps"),
        "mean_abs_advantage_exposed": credit.get("mean_abs_advantage_exposed"),
        "mean_abs_advantage_not_exposed": credit.get("mean_abs_advantage_not_exposed"),
        "mean_abs_advantage_served": credit.get("mean_abs_advantage_served"),
        "mean_abs_advantage_not_served": credit.get("mean_abs_advantage_not_served"),
        "per_uav_gradient_sum_max_abs": credit.get("per_uav_gradient_sum_max_abs"),
        "future_tag_advantage_on_zero_bonus": credit.get("future_tag_advantage_on_zero_bonus"),
        "advantage_broadcast": credit.get("advantage_broadcast"),
    }


def _clip_row(batch: dict) -> dict:
    credit = batch.get("credit") or {}
    return {
        "role": batch["role"],
        "variant": batch["variant"],
        "model_seed": batch["model_seed"],
        "source": batch["source"],
        "actor_policy_entropy_norm": credit.get("actor_policy_entropy_norm"),
        "critic_weighted_norm": credit.get("critic_weighted_norm"),
        "joint_norm": credit.get("joint_norm"),
        "critic_squared_share": credit.get("critic_squared_share"),
        "joint_clip_scale": credit.get("joint_clip_scale"),
        "separate_actor_clip_scale": credit.get("separate_actor_clip_scale"),
        "separate_critic_clip_scale": credit.get("separate_critic_clip_scale"),
        "clip_applied": credit.get("clip_applied"),
        "adam_update_inferred": credit.get("adam_update_inferred"),
        "clipping_bottleneck": credit.get("clipping_bottleneck"),
    }


def _match_row(batch: dict) -> dict:
    match = batch.get("sample_match") or {}
    worst = match.get("max_abs") or {}
    first = match.get("first") or {}
    sample_match = match.get("sample_match") or {}
    sample_first = sample_match.get("first") or {}
    return {
        "variant": batch["variant"],
        "model_seed": batch["model_seed"],
        "initialization_matches_checkpoint": batch.get("initialization_matches_checkpoint"),
        "sample_ok": match.get("ok"),
        "localized": match.get("localized"),
        "first_field": first.get("field") or match.get("field") or sample_first.get("field"),
        "first_step": first.get("step", sample_first.get("step")),
        "first_abs": first.get("abs", sample_first.get("abs")),
        "reward_max_abs": worst.get("reward", (sample_match.get("max_abs") or {}).get("reward")),
        "served_bits_max_abs": worst.get("served_bits", (sample_match.get("max_abs") or {}).get("served_bits")),
        "lookahead_input_max_abs": worst.get("lookahead_input", (sample_match.get("max_abs") or {}).get("lookahead_input")),
        "reward_float32_max_abs": worst.get("reward_float32", (sample_match.get("max_abs") or {}).get("reward_float32")),
        "historical_gradient_published": batch.get("historical_gradient_published"),
    }


def _choose(history: dict, probes: dict, tests: dict) -> tuple[str, list[str]]:
    reasons = []
    if not history["ok"] or history["gate"]["gate"] != E12_GATE or int(history["gate"]["passing_seeds"]) != 0:
        reasons.append("冻结日志重算没有保持 QUEUE_LOOKAHEAD_BENCHMARK_FAIL。")
    if probes["registered_batches"] != 12 or probes["gaps"] or not probes["frozen_hashes_unchanged"] or not probes["outer_rng_restored"]:
        reasons.append("有界探针、参数冻结或文件哈希没有通过。")
    if not tests.get("related_ok") or int(tests.get("new_failure_count", 1)) != 0:
        reasons.append("新测试没有通过，或完整测试相对 fabe040 干净基线出现新的失败签名。")
    initial = [batch for batch in probes["batches"] if batch["role"] == "initial_reconstruction"]
    for batch in initial:
        match = batch.get("sample_match") or {}
        if match.get("ok"):
            continue
        if not match.get("localized") and not match.get("first") and not match.get("field"):
            reasons.append("初始历史切片失配且没有定位到字段。")
            break
    terminal = [batch for batch in probes["batches"] if batch["role"] == "terminal_frozen_batch"]
    if len(terminal) != 6 or any(not (batch.get("credit") or {}).get("published") for batch in terminal):
        reasons.append("终态诊断梯度没有完整发布。")
    if reasons:
        return GATE_BLOCKED, reasons
    return GATE_COMPLETE, []


def finalize() -> dict:
    head = assert_probe_ready()
    partial = json.loads(PARTIAL.read_text(encoding="utf-8"))
    if partial["implementation_commit"] != head:
        raise RuntimeError("probe partial was produced by a different commit")
    history = partial["history"]
    probes = partial["probes"]
    related_text = RELATED_LOG.read_text(encoding="utf-8")
    related = parse_unittest(related_text)
    comparison = compare_logs(BASELINE_LOG.read_text(encoding="utf-8"), REPOSITORY_LOG.read_text(encoding="utf-8"))
    comparison["related_ok"] = bool(related["ran"] and not related["failures"] and "OK" in related_text)
    gate, reasons = _choose(history, probes, comparison)
    gaps = [
        "历史样本没有完整观测、价值、终止标志、旧对数概率、minibatch 顺序和 Adam 矩，不能复原更新 150 或 300 的真实梯度。",
        "环境 0 的保存切片不能代表其余环境。",
        "终态批次是 diagnostic_proxy，不是最后一次真实训练 rollout。",
        "float64 GAE 只是数值诊断。",
        "分开的 actor/critic 裁剪比例没有应用。没有历史 Adam 矩，联合裁剪是不是主要瓶颈并未证实。",
        "团队优势广播是原 MAPPO 设计。无关 UAV 稀释信用只是假设。",
        "同配置第二次 rollout 没有列入 12 个注册批次。",
        "phase3d、phase3h 和时序信用 oracle 的增益不作为本 synthetic-tag MAPPO 的证据。",
        "开发池已被前期实验使用。标签是 synthetic tag。",
    ]
    gaps.extend(history["gaps"])
    gaps.extend(probes["gaps"])
    if any(not batch.get("historical_gradient_published") for batch in probes["batches"] if batch["role"] == "initial_reconstruction"):
        gaps.append("至少一组初始批次没有发布为历史梯度。")
    blockers = [
        "E12 科学 Gate 保持 QUEUE_LOOKAHEAD_BENCHMARK_FAIL。",
        "本审计不是性能验收，release_eligible 为 false，未合并。",
        "完整测试中的既有失败仍是发布阻断项。",
    ]
    blockers.extend(reasons)
    summary = {
        "scientific_gate": gate,
        "e12_gate": E12_GATE,
        "training_updates": 0,
        "merged": False,
        "release_eligible": False,
        "implementation_commit": head,
        "instruction_commit": INSTRUCTION_COMMIT,
        "e12_result_commit": E12_RESULT_COMMIT,
        "history": _history_public(history),
        "probes": {
            "registered_batches": probes["registered_batches"],
            "verification_passes": probes["verification_passes"],
            "rng_check": probes["rng_check"],
            "frozen_hashes_unchanged": probes["frozen_hashes_unchanged"],
            "frozen_hash_count": probes["frozen_hash_count"],
            "gaps": probes["gaps"],
            "batches": probes["batches"],
        },
        "tests": comparison,
        "exposure_totals": history["exposure_totals"],
        "recommendation": "no_single_component",
        "evidence_gaps": gaps,
        "release_blockers": blockers,
        "runtime": partial["runtime"],
    }
    _write_tables(history, probes)
    _write_json(OUTPUT / "summary.json", summary)
    _write_json(OUTPUT / "schema.json", schema())
    _write_json(OUTPUT / "runtime_manifest.json", partial["runtime"])
    _write_json(OUTPUT / "source_manifest.json", {"implementation_commit": head, "shared_sources": shared_source_hashes(), "package": _package_hashes()})
    _write_json(OUTPUT / "evidence_gaps.json", {"gaps": gaps})
    _write_json(OUTPUT / "release_blockers.json", {"release_eligible": False, "merged": False, "blockers": blockers})
    _write_json(OUTPUT / "test_comparison.json", comparison)
    _write_json(OUTPUT / "initial_batch_match.json", {"rows": [_match_row(batch) for batch in probes["batches"] if batch["role"] == "initial_reconstruction"]})
    _write_json(OUTPUT / "probe_manifest.json", {"registered_batches": probes["registered_batches"], "verification_passes": 0, "rng_check": probes["rng_check"], "batches": [_manifest_batch(batch) for batch in probes["batches"]]})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_report(summary), encoding="utf-8")
    return summary


def _manifest_batch(batch: dict) -> dict:
    return {
        "role": batch["role"],
        "variant": batch["variant"],
        "model_seed": batch["model_seed"],
        "source": batch["source"],
        "batch_path": batch["batch_path"],
        "batch_sha256": batch["batch_sha256"],
        "parameters_unchanged": batch.get("parameters_unchanged"),
        "historical_gradient_published": batch.get("historical_gradient_published", False),
        "diagnostic_gradient_published": batch.get("diagnostic_gradient_published", False),
    }


def _package_hashes() -> dict[str, str]:
    package = ROOT / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag_credit_audit"
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in sorted(package.glob("*.py"))}
    hashes["tests/test_priority_tag_credit_audit.py"] = file_sha256(ROOT / "tests" / "test_priority_tag_credit_audit.py")
    return hashes


def _write_tables(history: dict, probes: dict) -> None:
    _write_csv(OUTPUT / "historical_log_summary.csv", history["log_rows"], list(history["log_rows"][0].keys()) if history["log_rows"] else ["variant"])
    _write_csv(OUTPUT / "historical_sample_checks.csv", history["sample_rows"], list(history["sample_rows"][0].keys()) if history["sample_rows"] else ["variant"])
    batches = probes["batches"]
    _write_csv(OUTPUT / "advantage_rows.csv", [_advantage_row(batch) for batch in batches], list(_advantage_row(batches[0]).keys()) if batches else ["role"])
    _write_csv(OUTPUT / "gradient_rows.csv", [_gradient_row(batch) for batch in batches], list(_gradient_row(batches[0]).keys()) if batches else ["role"])
    _write_csv(OUTPUT / "team_credit_rows.csv", [_team_row(batch) for batch in batches], list(_team_row(batches[0]).keys()) if batches else ["role"])
    _write_csv(OUTPUT / "clipping_rows.csv", [_clip_row(batch) for batch in batches], list(_clip_row(batches[0]).keys()) if batches else ["role"])


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "probes":
        run_probes()
        return
    if command == "finalize":
        summary = finalize()
        print(summary["scientific_gate"], flush=True)
        return
    raise RuntimeError("use probes or finalize")


if __name__ == "__main__":
    main()
