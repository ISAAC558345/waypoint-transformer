"""Run smoke, L3 training, and the paired evaluation. This module does not merge."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR, OUTPUT as PRIOR_OUTPUT
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256, watched_hashes
from wt.benchmark_v2.phase6i_priority_tag_reward.train import source_commit
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.reconcile import load_saved_rows, saved_index
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import public_row

from .criteria import choose_gate, score_l3
from .evaluate import assert_l3_checkpoint, evaluate_l3, evaluate_saved, load_l3, load_reference, preflight_l0
from .protocol import (
    GATE_BLOCKED,
    INSTRUCTION_COMMIT,
    OUTPUT,
    PRIOR_CHECKPOINT_SHA256,
    REPORT,
    ROOT,
    SCHEMA_VERSION,
    TRAIN_SEEDS,
)
from .report import render_report
from .smoke import run_smoke
from .train import train_l3


ALLOWED_DIFF_PREFIXES = (
    "src/wt/benchmark_v2/phase6i_direct_tag_service_reward/",
    "tests/test_direct_tag_service_reward.py",
    "docs/research/HTW_UrgencyAware_Direct_Tag_Reward_Report_CN.md",
    "docs/research/HTW_UrgencyAware_Direct_Tag_Reward_Instructions_CN.md",
    "results/phase6i_direct_tag_service_reward/",
)


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return value


def _write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    public = [public_row(row) for row in rows]
    fields: list[str] = []
    for row in public:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, np.ndarray)):
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in public:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _prior_path(variant: str, seed: int):
    if variant == "L0":
        return FORMAL_ACTOR
    return PRIOR_OUTPUT / "checkpoints" / f"{variant.lower()}_seed{int(seed)}.pt"


def _prior_hashes() -> dict[str, str]:
    found = {}
    for (variant, seed), _expected in PRIOR_CHECKPOINT_SHA256.items():
        path = _prior_path(variant, seed)
        found[f"{variant}_seed{int(seed)}"] = file_sha256(path)
    return found


def _prior_ok(found: dict[str, str]) -> bool:
    for (variant, seed), expected in PRIOR_CHECKPOINT_SHA256.items():
        if found.get(f"{variant}_seed{int(seed)}") != expected:
            return False
    return True


def _formal_diff() -> list[str]:
    result = subprocess.run(
        ["git", "-c", "core.longpaths=true", "diff", "--name-only", "origin/feature/maritime-wt-stage1a...HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    names = [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]
    return [name for name in names if not name.startswith(ALLOWED_DIFF_PREFIXES) and name not in ALLOWED_DIFF_PREFIXES]


def _publish(summary: dict, rows: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "scenario_rows.csv", rows)
    seeds = []
    for item in (summary.get("metrics") or {}).get("seeds") or []:
        seeds.append(
            {
                "model_seed": item["model_seed"],
                "passes": item["passes_preregistered_criteria"],
                "tagged_l0_mean": item["tagged_versus_l0"]["mean"],
                "tagged_l0_ci_low": item["tagged_versus_l0"]["ci_low"],
                "tagged_l0_ci_high": item["tagged_versus_l0"]["ci_high"],
                "payload_l0_mean": item["payload_versus_l0"]["mean"],
                "payload_l0_ci_low": item["payload_versus_l0"]["ci_low"],
                "payload_l0_ci_high": item["payload_versus_l0"]["ci_high"],
                "tagged_l1_mean": item["tagged_versus_l1"]["mean"],
                "tagged_l1_ci_low": item["tagged_versus_l1"]["ci_low"],
                "tagged_l2_mean": item["tagged_versus_l2"]["mean"],
                "payload_negative_scenarios": item["payload_negative_scenarios"],
            }
        )
    _write_csv(OUTPUT / "seed_summary.csv", seeds)
    manifest = {
        "training": summary.get("training") or [],
        "prior_checkpoint_sha256": summary.get("prior_checkpoint_sha256"),
        "formal_sha256_before": (summary.get("watched_before") or {}).get(
            "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt"
        ),
        "formal_sha256_after": (summary.get("watched_after") or {}).get(
            "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt"
        ),
        "runtime": summary.get("runtime"),
    }
    (OUTPUT / "checkpoint_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "runtime_manifest.json").write_text(json.dumps(summary.get("runtime"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stored = {key: value for key, value in summary.items() if key not in {"watched_before", "watched_after"}}
    (OUTPUT / "summary.json").write_text(json.dumps(stored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "schema.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "synthetic_tag": True,
                "gate": summary.get("gate"),
                "merged": False,
                "retrained_l0_l1_l2": False,
                "lambda_tag": 0.002,
                "rho_tag": 0.001,
                "urgency_in_l3_bonus": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(render_report(summary), encoding="utf-8")


def _runtime(head: str) -> dict:
    return {
        "context_label": "historical_inherited_process_context",
        "process_rng_seed": 2,
        "process_rng_role": "historical_inherited_process_seed",
        "canonical_explicit_seed": None,
        "torch_num_threads": 1,
        "deterministic_algorithms": True,
        "training_seed_role": "mappo_model_seed",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "git_commit": head,
        "instruction_commit": INSTRUCTION_COMMIT,
    }


def main() -> None:
    head = source_commit()
    before = watched_hashes()
    prior = _prior_hashes()
    summary = {
        "instruction_commit": INSTRUCTION_COMMIT,
        "gate": GATE_BLOCKED,
        "merged": False,
        "smoke": {},
        "preflight": {},
        "training": [],
        "train_ok": False,
        "replay": {},
        "metrics": {},
        "hashes_ok": False,
        "failure": None,
        "prior_checkpoint_sha256": prior,
        "watched_before": before,
        "runtime": _runtime(head),
        "unexpected_formal_diff_count": 0,
        "unexpected_formal_diff": [],
    }
    diff_names = _formal_diff()
    summary["unexpected_formal_diff_count"] = len(diff_names)
    summary["unexpected_formal_diff"] = diff_names[:12]
    if not _prior_ok(prior) or before.get("src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt") != PRIOR_CHECKPOINT_SHA256[("L0", 0)]:
        summary["failure"] = "formal or prior checkpoint hash drifted before the run"
        summary["watched_after"] = watched_hashes()
        _publish(summary, [])
        print(summary["gate"])
        return
    smoke = run_smoke()
    summary["smoke"] = {"passed": smoke["passed"], "problems": smoke["problems"], "scenarios": smoke["scenarios"]}
    if not smoke["passed"]:
        summary["failure"] = "smoke failed"
        summary["watched_after"] = watched_hashes()
        summary["hashes_ok"] = summary["watched_after"] == before
        _publish(summary, [])
        print(summary["gate"])
        return
    saved = saved_index(load_saved_rows())
    preflight = preflight_l0(saved[("L0", 0, 321101)])
    summary["preflight"] = {key: value for key, value in preflight.items() if key != "row"}
    if not preflight["matches_saved"]:
        summary["failure"] = "frozen L0 preflight missed the saved row"
        summary["watched_after"] = watched_hashes()
        summary["hashes_ok"] = summary["watched_after"] == before
        _publish(summary, [preflight["row"]])
        print(summary["gate"])
        return
    try:
        for seed in TRAIN_SEEDS:
            summary["training"].append(train_l3(int(seed)))
            assert_l3_checkpoint(int(seed))
    except RuntimeError as exc:
        summary["failure"] = f"training stopped: {exc}"
        summary["watched_after"] = watched_hashes()
        summary["hashes_ok"] = summary["watched_after"] == before and _prior_hashes() == prior
        _publish(summary, [])
        print(summary["gate"])
        return
    summary["train_ok"] = True
    rows = []
    mismatches = 0
    max_bit = 0.0
    max_path = 0.0
    actors = {}
    for variant, seed in [("L0", 0)] + [(name, int(seed)) for name in ("L1", "L2") for seed in TRAIN_SEEDS]:
        actors[(variant, int(seed))] = load_reference(variant, int(seed))
        for scenario in DEV_POOL:
            saved_row = saved[(variant, int(seed), int(scenario))]
            row, status = evaluate_saved(variant, int(seed), int(scenario), saved_row, actors[(variant, int(seed))])
            rows.append(row)
            max_bit = max(max_bit, float(status["max_bit_gap"]))
            max_path = max(max_path, float(status["path_gap_m"]))
            print(f"{variant} seed {int(seed)} scenario {int(scenario)} bit={status['max_bit_gap']:.6g} path={status['path_gap_m']:.6g}", flush=True)
            if not status["matches_saved"]:
                mismatches += 1
                break
        if mismatches:
            break
    replay_ok = mismatches == 0 and len(rows) == 32 * 7
    summary["replay"] = {"checked": len(rows), "mismatches": mismatches, "max_bit_gap": max_bit, "max_path_gap_m": max_path}
    if replay_ok:
        for seed in TRAIN_SEEDS:
            actor = load_l3(int(seed))
            for scenario in DEV_POOL:
                row = evaluate_l3(actor, int(seed), int(scenario))
                rows.append(row)
                print(
                    f"L3 seed {int(seed)} scenario {int(scenario)} payload={row['payload_bits']:.6g} tagged={row['served_tagged_bits']:.6g}",
                    flush=True,
                )
        summary["metrics"] = score_l3(rows)
    after = watched_hashes()
    summary["watched_after"] = after
    summary["hashes_ok"] = bool(after == before and _prior_hashes() == prior)
    passing = int((summary.get("metrics") or {}).get("passing_seeds") or 0)
    summary["gate"] = choose_gate(
        smoke_ok=True,
        train_ok=bool(summary["train_ok"]),
        replay_ok=bool(replay_ok),
        hashes_ok=bool(summary["hashes_ok"]),
        passing_seeds=passing,
    )
    if summary["gate"] == "DIRECT_TAG_REWARD_PASS" and int(summary["unexpected_formal_diff_count"]) > 0:
        summary["gate"] = "DIRECT_TAG_REWARD_INCONCLUSIVE"
        summary["failure"] = "metrics passed but the branch diff against the formal branch contains the unmerged urgency experiment"
    summary["merged"] = False
    _publish(summary, rows)
    print(summary["gate"])


if __name__ == "__main__":
    main()
