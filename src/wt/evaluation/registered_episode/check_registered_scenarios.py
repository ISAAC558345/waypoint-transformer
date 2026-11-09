"""Saved-result check for the registered formal scenarios. This command imports no model stack."""
from __future__ import annotations

import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from wt.evaluation.registered_episode.completion import compare_episodes
from wt.evaluation.registered_episode.references import REGISTERED, resolve
from wt.evaluation.registered_episode.run import main, preview
from wt.common.paths import repo_root


ROOT = repo_root()
OUTPUT = ROOT / "results" / "legacy_system_l10"
PROTECTED = (
    "results/reproducibility/formal_scheme_c/episodes/322000_scheme_c.json",
    "results/reproducibility/formal_scheme_c/episodes/322001_scheme_c.json",
    "results/reproducibility/formal_scheme_c/configs/322000_scheme_c.json",
    "results/reproducibility/formal_scheme_c/configs/322001_scheme_c.json",
    "results/reproducibility/scenario_322000/episode.json",
    "results/reproducibility/scenario_322000/run_manifest.json",
)
PREVIEW_DIRS = (
    "results/legacy_system/scenario_322000",
    "results/legacy_system/scenario_322001",
    "results/legacy_system/scenario_322002",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _capture(argv: list[str]) -> tuple[int, dict]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(argv)
    return code, json.loads(buffer.getvalue())


def _episode_record(episode: dict) -> dict:
    times = [int(step["time"]) for step in episode["steps"]]
    return {
        "scenario_id": episode.get("scenario_id"),
        "policy": episode.get("policy"),
        "identity_present": "scenario_id" in episode and "policy" in episode and episode.get("scenario_id") is not None and episode.get("policy") is not None,
        "steps": len(episode["steps"]),
        "times_are_0_through_47": times == list(range(48)),
        "environment_steps": episode.get("environment_steps"),
        "terminal_time": episode.get("terminal_time"),
        "stop_time": episode.get("stop_time"),
        "complete": episode.get("complete"),
        "decision_times": episode.get("decision_times"),
        "scheme_c_calls": episode.get("scheme_c_calls"),
        "process_seed": episode.get("process_seed"),
        "reset_seed": episode.get("reset_seed"),
        "payload_bits": episode.get("payload_bits"),
        "served_bits": episode.get("served_bits"),
    }


def _comparison_record(result: dict) -> dict:
    return {
        "status": result["status"],
        "reference_payload_bits": result["reference_payload_bits"],
        "reference_payload_source": result["reference_payload_source"],
        "payload_abs_diff_bits": result["payload_abs_diff_bits"],
        "served_abs_diff_bits": result["served_abs_diff_bits"],
        "step_mismatch_count": len(result["step_served_mismatches"]),
        "identity": result["identity"],
        "internal_consistency_ok": result["internal_consistency_ok"],
    }


def main_check() -> int:
    if "torch" in sys.modules or "wt.evaluation.registered_episode.execute" in sys.modules:
        raise RuntimeError("model stack was imported before the saved-result check")
    before = {item: _sha256(ROOT / item) for item in PROTECTED}
    preview_existed = {item: (ROOT / item).exists() for item in PREVIEW_DIRS}
    default_322000_code, default_322000 = _capture([])
    default_322001_code, default_322001 = _capture(["--scenario", "322001"])
    explicit_code, explicit = _capture(["--scenario", "322000", "--output", "results/reproducibility/scenario_322000"])
    refused_preview_code, refused_preview = _capture(["--scenario", "322002"])
    refused_run_code, refused_run = _capture(["--scenario", "322002", "--output", "results/reproducibility/scenario_322001/refused_322002", "--run"])
    direct_322000 = preview(322000, Path("results/legacy_system/scenario_322000"))
    direct_322001 = preview(322001, Path("results/legacy_system/scenario_322001"))
    reference_322000 = _load(ROOT / "results/reproducibility/formal_scheme_c/episodes/322000_scheme_c.json")
    reference_322001 = _load(ROOT / "results/reproducibility/formal_scheme_c/episodes/322001_scheme_c.json")
    l6_episode = _load(ROOT / "results/reproducibility/scenario_322000/episode.json")
    same_322000 = compare_episodes(reference_322000, l6_episode)
    same_322001 = compare_episodes(reference_322001, json.loads(json.dumps(reference_322001)))
    crossed = compare_episodes(reference_322000, reference_322001)
    missing = json.loads(json.dumps(reference_322001))
    del missing["scenario_id"]
    missing_identity = compare_episodes(reference_322001, missing)
    both_missing = json.loads(json.dumps(reference_322001))
    reference_missing = json.loads(json.dumps(reference_322001))
    del both_missing["policy"]
    del reference_missing["policy"]
    absent_identity = compare_episodes(reference_missing, both_missing)
    after = {item: _sha256(ROOT / item) for item in PROTECTED}
    preview_created = [item for item in PREVIEW_DIRS if not preview_existed[item] and (ROOT / item).exists()]
    refused_dir = ROOT / "results/reproducibility/scenario_322001/refused_322002"
    record = {
        "package": "legacy_system_l10_stdlib_check",
        "instruction_commit": "204dd3309115dfd369236c408f7e59306a0871d4",
        "environment_steps": 0,
        "low_level_inferences": 0,
        "high_level_calls": 0,
        "scoring_forwards": 0,
        "training_updates": 0,
        "backward_passes": 0,
        "torch_imported": "torch" in sys.modules,
        "execute_imported": "wt.evaluation.registered_episode.execute" in sys.modules,
        "registered_scenarios": list(REGISTERED),
        "previews": {
            "default_322000": default_322000,
            "default_322001": default_322001,
            "explicit_322000_l6_output": explicit,
            "unregistered_322002": refused_preview,
            "direct_322000": direct_322000,
            "direct_322001": direct_322001,
        },
        "preview_exit_codes": {
            "default_322000": default_322000_code,
            "default_322001": default_322001_code,
            "explicit_322000": explicit_code,
            "unregistered_322002": refused_preview_code,
        },
        "unregistered_run": {"exit_code": refused_run_code, "payload": refused_run, "output_created": refused_dir.exists()},
        "preview_directories_created": preview_created,
        "sources": {
            item: {"path": item, "sha256": before[item], "unchanged": before[item] == after[item]}
            for item in PROTECTED
        },
        "reference_records": {
            "322000": _episode_record(reference_322000),
            "322001": _episode_record(reference_322001),
        },
        "comparisons": {
            "r1_322000_versus_saved_l6": _comparison_record(same_322000),
            "r1_322001_versus_itself": _comparison_record(same_322001),
            "r1_322000_versus_r1_322001": _comparison_record(crossed),
            "missing_scenario_id": _comparison_record(missing_identity),
            "missing_policy_on_both_sides": _comparison_record(absent_identity),
        },
        "bindings": {"322000": resolve(322000), "322001": resolve(322001), "322002": resolve(322002)},
    }
    expected_binding = {
        322000: ("results/reproducibility/formal_scheme_c/episodes/322000_scheme_c.json", "results/reproducibility/formal_scheme_c/configs/322000_scheme_c.json", 1322000, 322000, "results/legacy_system/scenario_322000"),
        322001: ("results/reproducibility/formal_scheme_c/episodes/322001_scheme_c.json", "results/reproducibility/formal_scheme_c/configs/322001_scheme_c.json", 1322001, 322001, "results/legacy_system/scenario_322001"),
    }
    preview_ok = True
    for scenario, (episode, config, process_seed, reset_seed, output) in expected_binding.items():
        shown = record["previews"][f"default_{scenario}"]
        preview_ok = preview_ok and shown["run_accepted"] is True and shown["reference_episode"] == episode and shown["reference_config"] == config
        preview_ok = preview_ok and shown["process_seed"] == process_seed and shown["reset_seed"] == reset_seed and shown["output"] == output
        preview_ok = preview_ok and shown["loads_model"] is False and shown["writes_files"] is False
    checks = {
        "previews": preview_ok and explicit["output"] == "results/reproducibility/scenario_322000" and explicit["reference_episode"].endswith("322000_scheme_c.json"),
        "preview_exit_codes": all(code == 0 for code in record["preview_exit_codes"].values()),
        "no_preview_directory": not preview_created,
        "no_model_import": not record["torch_imported"] and not record["execute_imported"],
        "unregistered_refused": refused_preview["run_accepted"] is False and refused_preview["reference_episode"] is None and refused_run_code == 2 and not refused_dir.exists(),
        "saved_322000_matched": same_322000["status"] == "matched" and same_322000["reference_payload_bits"] == reference_322000["payload_bits"],
        "saved_322001_matched": same_322001["status"] == "matched" and same_322001["reference_payload_bits"] == reference_322001["payload_bits"],
        "cross_different": crossed["status"] == "different",
        "missing_identity_different": missing_identity["status"] == "different" and absent_identity["status"] == "different",
        "sources_complete": all(item["times_are_0_through_47"] and item["steps"] == 48 and item["identity_present"] and item["complete"] is True for item in record["reference_records"].values()),
        "sources_unchanged": all(item["unchanged"] for item in record["sources"].values()),
        "direct_preview_matches_registry": direct_322000["reference_episode"] == expected_binding[322000][0] and direct_322001["reference_episode"] == expected_binding[322001][0],
    }
    record["checks"] = checks
    record["passed"] = all(checks.values())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "stdlib_check.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"stdlib_check": "passed" if record["passed"] else "failed", "environment_steps": 0, "failed": [name for name, ok in checks.items() if not ok]}, sort_keys=True), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main_check())