"""Run the isolated L3 experiment and stop at its preregistered gate."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import LEGACY_ACTOR, OBSERVATION_SCHEMA, TAG_RATE, TAG_SEED
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256, watched_hashes
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.reconcile import load_saved_rows, saved_index

from .criteria import choose_gate, publish_gate, score_l3
from .evaluate import assert_l3_checkpoint, evaluate_l3, evaluate_saved, load_l3, load_reference, preflight_l0
from .protocol import (
    ATTRIBUTION_COMMIT,
    FORMAL_ACTOR_SHA256,
    GATE_BLOCKED,
    INSTRUCTION_COMMIT,
    LAMBDA_AGE,
    LAMBDA_TAG,
    OUTPUT,
    PRIOR_CHECKPOINT_SHA256,
    REPORT,
    REWARD_SCHEMA,
    RHO_AGE,
    RHO_TAG,
    ROOT,
    SCHEMA_VERSION,
    TRAIN_SEEDS,
)
from .report import render_report
from .reward import validate_config
from .smoke import run_smoke
from .train import train_l3


ALLOWED_PREFIXES = (
    "src/wt/benchmark_v2/phase6i_priority_tag_direct_service_reward/",
    "tests/test_priority_tag_direct_service_reward.py",
    "docs/research/HTW_UrgencyAware_Direct_Tag_Reward_Experiment_Instructions_CN.md",
    "docs/research/HTW_UrgencyAware_Direct_Tag_Reward_Experiment_Report_CN.md",
    "results/phase6i_priority_tag_direct_service_reward/",
)
LEGACY_ACTOR_SHA256 = "44d44cf96f666aa7810bcd9b6546e2956d35fbf280e5f312c983444b25dbce14"
FROZEN_JOBS = [("L0", 0)] + [(variant, int(seed)) for variant in ("L1", "L2") for seed in TRAIN_SEEDS]


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _checkpoint_path(variant: str, seed: int):
    if variant == "L0":
        return ROOT / "src" / "wt" / "benchmark_v2" / "phase6g_lowlevel_mappo" / "checkpoints" / "phase6g_seed0.pt"
    return ROOT / "results" / "phase6i_priority_tag_reward" / "checkpoints" / f"{variant.lower()}_seed{int(seed)}.pt"


def prior_checkpoint_problems() -> list[str]:
    problems = []
    for key, expected in PRIOR_CHECKPOINT_SHA256.items():
        path = _checkpoint_path(key[0], int(key[1]))
        if file_sha256(path) != expected:
            problems.append(f"{key[0]} seed {key[1]} hash changed")
    if file_sha256(LEGACY_ACTOR) != LEGACY_ACTOR_SHA256:
        problems.append("legacy actor hash changed")
    return problems


def _git_paths() -> list[str]:
    commands = (
        ["git", "-c", "core.longpaths=true", "diff", "--name-only", "origin/feature/maritime-wt-stage1a...HEAD"],
        ["git", "-c", "core.longpaths=true", "diff", "--name-only", "origin/feature/maritime-wt-stage1a"],
        ["git", "-c", "core.longpaths=true", "ls-files", "--others", "--exclude-standard"],
    )
    found = []
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        found.extend(line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip())
    return sorted(set(found))


def isolation() -> dict:
    unexpected = [path for path in _git_paths() if not any(path == prefix or path.startswith(prefix) for prefix in ALLOWED_PREFIXES)]
    return {"isolated_diff": bool(not unexpected), "unexpected_path_count": len(unexpected), "unexpected_paths": unexpected[:12]}


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


def _public(row: dict) -> dict:
    return {key: value for key, value in row.items() if not str(key).startswith("_") and not isinstance(value, np.ndarray)}


def write_rows(rows: list[dict]) -> None:
    public = [_public(row) for row in rows]
    columns = [
        "variant",
        "model_seed",
        "scenario_id",
        "steps",
        "payload_bits",
        "urgency_served_bits",
        "age_service_bits",
        "tag_increment_service_bits",
        "served_tagged_bits",
        "served_ordinary_bits",
        "dropped_tagged_bits",
        "dropped_ordinary_bits",
        "dropped_bits",
        "final_queue_bits",
        "final_tagged_queue_bits",
        "queue_age_mean",
        "queue_age_max",
        "sinr_mean",
        "interference_mean",
        "path_length_m",
        "projection_rate",
        "safety_violations",
        "constraint_failures",
        "return_base",
        "return_augmented",
        "bonus_sum",
        "bonus_age_sum",
        "bonus_tag_sum",
        "max_ledger_gap_bits",
        "matches_saved",
        "max_bit_gap",
        "path_gap_m",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "scenario_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in public:
            writer.writerow({key: _cell(row.get(key)) for key in columns})


def read_rows() -> list[dict]:
    numeric = {
        "payload_bits",
        "urgency_served_bits",
        "age_service_bits",
        "served_tagged_bits",
        "served_ordinary_bits",
        "dropped_bits",
        "final_queue_bits",
        "safety_violations",
        "constraint_failures",
        "sinr_mean",
        "interference_mean",
    }
    with (OUTPUT / "scenario_rows.csv").open(encoding="utf-8", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            item = {"variant": raw["variant"], "model_seed": int(float(raw["model_seed"])), "scenario_id": int(float(raw["scenario_id"]))}
            for key in numeric:
                item[key] = float(raw[key])
            rows.append(item)
    return rows


def _schema() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "reward_schema": REWARD_SCHEMA,
        "observation_schema": OBSERVATION_SCHEMA,
        "synthetic_tag": True,
        "tag_seed": int(TAG_SEED),
        "tag_rate": float(TAG_RATE),
        "lambda_age": float(LAMBDA_AGE),
        "rho_age": float(RHO_AGE),
        "lambda_tag": float(LAMBDA_TAG),
        "rho_tag": float(RHO_TAG),
        "instruction_commit": INSTRUCTION_COMMIT,
        "attribution_commit": ATTRIBUTION_COMMIT,
        "sinr_interference_rule": "paired SINR mean >= 0 and paired interference mean <= 0 versus L0",
        "safety_rule": "any seed with a positive paired safety or constraint sum fails the stage",
        "scenario_concentration_rule": "leave-one-out tagged mean > 0 on every subset and mean > 0 after removing the two largest tagged gains",
        "payload_floor": "paired payload CI low >= -0.5 percent of that seed's L0 mean payload",
        "bootstrap_draws": 10000,
        "bootstrap_seed": 0,
        "historical_process_seed": 2,
        "historical_process_seed_role": "historical_inherited_process_seed",
        "canonical_explicit_eval_seed": None,
    }


def _runtime() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "torch_num_threads": int(torch.get_num_threads()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "process_rng_seed": 2,
        "process_rng_role": "historical_inherited_process_seed",
        "canonical_explicit_eval_seed": None,
        "model_seeds": list(TRAIN_SEEDS),
        "env_reset_seed": "scenario_id",
        "assignment_rng_seed": "7000000 + scenario_id",
    }


def _next_step(gate: str) -> str:
    if gate == GATE_BLOCKED:
        return "先修复本阶段的回放或训练阻断。不要调系数，不要把未完成的 L3 解释成奖励结论，也不要合并。"
    return "不要根据这次结果回头修改 0.0005/0.00025 与 0.0010/0.00075。不要重训，不要合并。若要换奖励形式，需要另一份预先声明的实验说明。"


def _reason(summary: dict) -> str:
    if summary["scientific_gate"] == GATE_BLOCKED:
        return summary.get("block_reason", "实验在训练或冻结回放之前被阻断。")
    score = summary["score"]
    passing = int(score["passing_seeds"])
    negative_tagged = sum(float(item["tagged_versus_l0"]["mean"]) < 0.0 for item in score["seeds"])
    negative_payload = sum(float(item["payload_versus_l0"]["mean"]) < 0.0 for item in score["seeds"])
    safety = [
        int(item["model_seed"])
        for item in score["seeds"]
        if float(item["safety_diff_sum"]) > 0.0 or float(item["constraint_diff_sum"]) > 0.0
    ]
    interference = [int(item["model_seed"]) for item in score["seeds"] if float(item["interference_versus_l0_mean"]) > 0.0]
    text = f"3 个 L3 种子中有 {passing} 个同时通过全部预注册检查。"
    if negative_tagged:
        text += f" 实际标签服务配对均值为负的种子有 {negative_tagged} 个。"
    if negative_payload:
        text += f" 总载荷配对均值为负的种子有 {negative_payload} 个。"
    if safety:
        text += f" 安全违规或约束失败配对和增加的种子是 {safety}。"
    if interference:
        text += f" 干扰配对均值为正的种子是 {interference}。"
    text += " 这些数字来自场景 CSV 的重新汇总，训练奖励本身不作为证据。"
    if not summary["isolated_diff"]:
        text += " 当前分支相对 feature/maritime-wt-stage1a 还包含此前的优先级标签实验历史，因此不合并。"
    if summary["gate"] != summary["scientific_gate"]:
        text += f" 科学门控是 {summary['scientific_gate']}，发布门控是 {summary['gate']}。"
    return text


def publish(summary: dict) -> None:
    summary["reason"] = _reason(summary)
    summary["next_step"] = _next_step(summary["gate"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "schema.json").write_text(json.dumps(_plain(_schema()), indent=2), encoding="utf-8")
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(summary), indent=2), encoding="utf-8")
    (OUTPUT / "runtime_manifest.json").write_text(json.dumps(_plain(summary["runtime"]), indent=2), encoding="utf-8")
    (OUTPUT / "checkpoint_manifest.json").write_text(json.dumps(_plain(summary["checkpoints"]), indent=2), encoding="utf-8")
    REPORT.write_text(render_report(summary), encoding="utf-8")
    print(summary["gate"], flush=True)


def _base_summary(smoke: dict, preflight: dict | None, training: list[dict], watched_before: dict) -> dict:
    problems = prior_checkpoint_problems()
    return {
        "smoke": smoke,
        "preflight": None
        if preflight is None
        else {
            "scenario_id": int(preflight["scenario_id"]),
            "matches_saved": bool(preflight["matches_saved"]),
            "max_bit_gap": float(preflight["max_bit_gap"]),
            "path_gap_m": float(preflight["path_gap_m"]),
        },
        "training": training,
        "score": None,
        "replay": None,
        "scientific_gate": GATE_BLOCKED,
        "gate": GATE_BLOCKED,
        "merged": False,
        "isolated_diff": False,
        "unexpected_path_count": None,
        "unexpected_paths": [],
        "prior_checkpoints_ok": bool(not problems),
        "prior_checkpoint_problems": problems,
        "watched_before": watched_before,
        "watched_unchanged": None,
        "runtime": _runtime(),
        "checkpoints": {"l3": training, "formal_sha256": FORMAL_ACTOR_SHA256, "legacy_sha256": LEGACY_ACTOR_SHA256},
        "block_reason": "",
    }


def _finish(summary: dict, *, scientific: str, score: dict | None) -> None:
    isolated = isolation()
    summary.update(isolated)
    summary["score"] = score
    summary["scientific_gate"] = scientific
    summary["gate"] = publish_gate(scientific, bool(isolated["isolated_diff"]))
    summary["merged"] = False
    after = watched_hashes()
    summary["watched_unchanged"] = bool(after == summary["watched_before"])
    summary["prior_checkpoints_ok"] = bool(not prior_checkpoint_problems())
    if not summary["watched_unchanged"] or not summary["prior_checkpoints_ok"]:
        summary["scientific_gate"] = GATE_BLOCKED
        summary["gate"] = GATE_BLOCKED
        summary["block_reason"] = "正式文件或既有 checkpoint 哈希发生变化。"
    publish(summary)


def _tests_pass() -> bool:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_priority_tag_direct_service_reward"],
        cwd=ROOT,
        env=env,
        check=False,
    )
    return int(result.returncode) == 0


def _replay_frozen(index: dict) -> tuple[list[dict], dict]:
    rows = []
    mismatches = []
    max_bit = 0.0
    max_path = 0.0
    for variant, seed in FROZEN_JOBS:
        print(f"replay {variant} seed {seed}", flush=True)
        actor = load_reference(variant, int(seed))
        for scenario in DEV_POOL:
            saved = index[(variant, int(seed), int(scenario))]
            row, status = evaluate_saved(variant, int(seed), int(scenario), saved, actor)
            rows.append(row)
            max_bit = max(max_bit, float(status["max_bit_gap"]))
            max_path = max(max_path, float(status["path_gap_m"]))
            if not bool(status["matches_saved"]):
                mismatches.append(
                    {
                        "variant": variant,
                        "model_seed": int(seed),
                        "scenario_id": int(scenario),
                        "max_bit_gap": float(status["max_bit_gap"]),
                        "path_gap_m": float(status["path_gap_m"]),
                    }
                )
    report = {
        "compared": len(rows),
        "matched": len(rows) - len(mismatches),
        "mismatches": len(mismatches),
        "mismatch_examples": mismatches[:8],
        "max_bit_gap": max_bit,
        "max_path_gap_m": max_path,
    }
    return rows, report


def main() -> None:
    started = time.perf_counter()
    validate_config()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "schema.json").write_text(json.dumps(_schema(), indent=2), encoding="utf-8")
    watched_before = watched_hashes()
    if not _tests_pass():
        summary = _base_summary({"passed": False, "problems": ["unit tests failed"]}, None, [], watched_before)
        summary["block_reason"] = "单元测试未通过，因此没有训练。"
        _finish(summary, scientific=GATE_BLOCKED, score=None)
        return
    smoke = run_smoke()
    print({"smoke_passed": smoke["passed"], "problems": smoke["problems"]}, flush=True)
    if not smoke["passed"]:
        summary = _base_summary(smoke, None, [], watched_before)
        summary["block_reason"] = "冒烟测试未通过，因此没有训练。"
        _finish(summary, scientific=GATE_BLOCKED, score=None)
        return
    if prior_checkpoint_problems():
        summary = _base_summary(smoke, None, [], watched_before)
        summary["block_reason"] = "训练前的正式或既有实验 checkpoint 哈希已经变化。"
        _finish(summary, scientific=GATE_BLOCKED, score=None)
        return
    index = saved_index(load_saved_rows())
    preflight = preflight_l0(index[("L0", 0, 321101)])
    print(
        {"preflight": preflight["matches_saved"], "bit_gap": preflight["max_bit_gap"], "path_gap": preflight["path_gap_m"]},
        flush=True,
    )
    if not bool(preflight["matches_saved"]):
        summary = _base_summary(smoke, preflight, [], watched_before)
        summary["block_reason"] = "L0 场景 321101 的预检没有在 1 比特和 1e-9 米内容差内匹配已保存结果，因此没有训练。"
        _finish(summary, scientific=GATE_BLOCKED, score=None)
        return
    training = []
    try:
        for seed in TRAIN_SEEDS:
            training.append(train_l3(int(seed)))
            assert_l3_checkpoint(int(seed))
    except Exception as exc:
        summary = _base_summary(smoke, preflight, training, watched_before)
        summary["block_reason"] = f"训练中断：{exc}"
        summary["traceback"] = traceback.format_exc()
        _finish(summary, scientific=GATE_BLOCKED, score=None)
        return
    rows, replay = _replay_frozen(index)
    if int(replay["mismatches"]) != 0:
        write_rows(rows)
        summary = _base_summary(smoke, preflight, training, watched_before)
        summary["replay"] = replay
        summary["block_reason"] = "冻结的 L0/L1/L2 回放与已保存结果不一致，因此不解释 L3。"
        _finish(summary, scientific=GATE_BLOCKED, score=None)
        return
    for seed in TRAIN_SEEDS:
        print(f"evaluate L3 seed {seed}", flush=True)
        actor = load_l3(int(seed))
        for scenario in DEV_POOL:
            rows.append(evaluate_l3(actor, int(seed), int(scenario)))
    write_rows(rows)
    score = score_l3(read_rows())
    summary = _base_summary(smoke, preflight, training, watched_before)
    summary["replay"] = replay
    summary["elapsed_seconds"] = float(time.perf_counter() - started)
    scientific = choose_gate(
        smoke_ok=True,
        train_ok=True,
        replay_ok=True,
        hashes_ok=bool(not prior_checkpoint_problems()),
        score=score,
    )
    _finish(summary, scientific=scientific, score=score)


if __name__ == "__main__":
    main()
