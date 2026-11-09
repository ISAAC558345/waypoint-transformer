"""Check completion rules on saved episodes. This command imports no model stack."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wt.evaluation.registered_episode.completion import compare_episodes, conclude, prepare_output_directory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    out = root / "results" / "legacy_delivery_l7"
    out.mkdir(parents=True, exist_ok=True)
    probe = out / "empty_dir_probe"
    if probe.exists():
        if any(probe.iterdir()):
            raise RuntimeError("empty-directory probe is not empty")
    else:
        probe.mkdir()
    empty_status = prepare_output_directory(probe)
    l6_episode = root / "results/reproducibility/scenario_322000/episode.json"
    l6_before = _sha256(l6_episode)
    nonempty_refused = False
    try:
        prepare_output_directory(root / "results/reproducibility/scenario_322000")
    except Exception as exc:
        nonempty_refused = type(exc).__name__ == "OutputRefusal"
        nonempty_message = str(exc)
    else:
        nonempty_message = "nonempty directory was accepted"
    l6_after = _sha256(l6_episode)
    reference = _load(root / "results/reproducibility/formal_scheme_c/episodes/322000_scheme_c.json")
    current = _load(l6_episode)
    manifest = _load(root / "results/reproducibility/scenario_322000/run_manifest.json")
    matched = compare_episodes(reference, current)
    matched_decision = conclude(
        {
            "error": manifest["error"],
            "complete": manifest["complete"],
            "regression_status": matched["status"],
            "model_hash_matches_contract": manifest["model_hash_matches_contract"],
            "official_policy_unchanged": manifest["official_policy_unchanged"],
        }
    )
    other = _load(root / "results/reproducibility/formal_scheme_c/episodes/322001_scheme_c.json")
    mismatched = compare_episodes(reference, other)
    mismatched_decision = conclude(
        {
            "error": None,
            "complete": bool(other["complete"]),
            "regression_status": mismatched["status"],
            "model_hash_matches_contract": manifest["model_hash_matches_contract"],
            "official_policy_unchanged": True,
        }
    )
    flagged = dict(current)
    flagged["payload_matches_served"] = 0
    flagged_comparison = compare_episodes(reference, flagged)
    record = {
        "package": "legacy_delivery_l7_completion",
        "environment_steps": 0,
        "low_level_inferences": 0,
        "scoring_forwards": 0,
        "training_updates": 0,
        "backward_passes": 0,
        "empty_directory": {"path": "results/reproducibility/delivery_check/empty_dir_probe", "status": empty_status, "remains_empty": not any(probe.iterdir())},
        "nonempty_directory": {
            "path": "results/reproducibility/scenario_322000",
            "refused": nonempty_refused,
            "message": nonempty_message,
            "episode_sha256_before": l6_before,
            "episode_sha256_after": l6_after,
            "episode_unchanged": l6_before == l6_after,
        },
        "matched_saved_l6": {
            "reference": "results/reproducibility/formal_scheme_c/episodes/322000_scheme_c.json",
            "current": "results/reproducibility/scenario_322000/episode.json",
            "status": matched["status"],
            "internal_consistency_ok": matched["internal_consistency_ok"],
            "payload_abs_diff_bits": matched["payload_abs_diff_bits"],
            "served_abs_diff_bits": matched["served_abs_diff_bits"],
            "step_mismatch_count": len(matched["step_served_mismatches"]),
            "exit_code": matched_decision["exit_code"],
            "success": matched_decision["success"],
        },
        "mismatched_r1_episode": {
            "reference": "results/reproducibility/formal_scheme_c/episodes/322000_scheme_c.json",
            "current": "results/reproducibility/formal_scheme_c/episodes/322001_scheme_c.json",
            "status": mismatched["status"],
            "exit_code": mismatched_decision["exit_code"],
            "success": mismatched_decision["success"],
            "reasons": mismatched_decision["reasons"],
        },
        "internal_consistency_gate": {
            "change": "saved L6 payload_matches_served set to 0 in memory only",
            "status": flagged_comparison["status"],
            "internal_consistency_ok": flagged_comparison["internal_consistency_ok"],
        },
        "l6_files_not_rewritten": True,
    }
    (out / "completion_check.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    ok = (
        empty_status == "empty_reused"
        and record["empty_directory"]["remains_empty"]
        and nonempty_refused
        and l6_before == l6_after
        and matched["status"] == "matched"
        and matched_decision["exit_code"] == 0
        and mismatched["status"] == "different"
        and mismatched_decision["exit_code"] == 1
        and flagged_comparison["status"] == "different"
    )
    print(json.dumps({"completion_check": "passed" if ok else "failed", "environment_steps": 0}, sort_keys=True), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())