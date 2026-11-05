"""One formal Scheme C episode. The caller imports this module only for --run."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import EPOCH6_CKPT, POLICY_FILE, SCHEME_C_CKPT
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import DECODER_CKPT
from wt.legacy_formal_repro_r1.runner import prepare_policy_copy, public_row, run_arm


INSTRUCTION = "2b2fa2f3ca66d64688a122d939e2b9bdb5f6605a"
REFERENCE_PAYLOAD_BITS = 63785945.28984571
TOLERANCE_BITS = 1.0
EXPECTED = {
    "scheme_c_sha256": "f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864",
    "mappo_sha256": "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e",
    "decoder_sha256": "de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _runtime() -> dict:
    import numpy
    import torch

    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "install_claim": "versions already present in this environment; this run is not a new-machine install check",
    }


def _steps_csv(row: dict) -> str:
    lines = ["scenario_id,policy,time,served_bits,cumulative_payload_bits"]
    for step in row["steps"]:
        lines.append(
            f"{row['scenario_id']},{row['policy']},{step['time']},{step['served_bits']:.10f},{step['cumulative_payload_bits']:.10f}"
        )
    return "\n".join(lines) + "\n"


def _field(reference: dict, current: dict, key: str) -> dict:
    return {"reference": reference.get(key), "current": current.get(key), "match": reference.get(key) == current.get(key)}


def _compare(reference: dict, current: dict) -> dict:
    payload_gap = abs(float(current["payload_bits"]) - float(reference["payload_bits"]))
    served_gap = abs(float(current["served_bits"]) - float(reference["served_bits"]))
    published_gap = abs(float(current["payload_bits"]) - REFERENCE_PAYLOAD_BITS)
    step_gaps = []
    ref_steps = reference["steps"]
    cur_steps = current["steps"]
    count = min(len(ref_steps), len(cur_steps))
    for index in range(count):
        gap = abs(float(cur_steps[index]["served_bits"]) - float(ref_steps[index]["served_bits"]))
        if int(cur_steps[index]["time"]) != int(ref_steps[index]["time"]) or gap > TOLERANCE_BITS:
            step_gaps.append(
                {
                    "index": index,
                    "reference_time": ref_steps[index]["time"],
                    "current_time": cur_steps[index]["time"],
                    "abs_served_diff_bits": gap,
                }
            )
    if len(ref_steps) != len(cur_steps):
        step_gaps.append({"reference_count": len(ref_steps), "current_count": len(cur_steps)})
    numeric_ok = payload_gap <= TOLERANCE_BITS and served_gap <= TOLERANCE_BITS and published_gap <= TOLERANCE_BITS and not step_gaps
    fields = {
        name: _field(reference, current, name)
        for name in ("complete", "terminal_time", "process_seed", "reset_seed", "decision_times", "scheme_c_calls")
    }
    fields_ok = all(item["match"] for item in fields.values())
    return {
        "reference_episode": "results/legacy_formal_repro_r1/episodes/322000_scheme_c.json",
        "reference_payload_bits": REFERENCE_PAYLOAD_BITS,
        "tolerance_bits": TOLERANCE_BITS,
        "payload_abs_diff_bits": payload_gap,
        "served_abs_diff_bits": served_gap,
        "published_payload_abs_diff_bits": published_gap,
        "step_count": {"reference": len(ref_steps), "current": len(cur_steps)},
        "step_served_mismatches": step_gaps,
        "fields": fields,
        "current_payload_matches_served": current.get("payload_matches_served"),
        "current_step_served_matches_episode": current.get("step_served_matches_episode"),
        "status": "matched" if numeric_ok and fields_ok else "different",
    }


def run_once(root: Path, output: Path, scenario: int) -> int:
    if output.exists() and any(output.iterdir()):
        print(json.dumps({"error": "refusing to overwrite an existing output", "environment_steps": 0}), flush=True)
        return 2
    output.mkdir(parents=True)
    consumed = 0
    try:
        official_before = _sha256(POLICY_FILE)
        policy_path = prepare_policy_copy("scheme_c", output / "configs" / f"{scenario}_scheme_c.json")
        row = run_arm(scenario, "scheme_c", policy_path, fault_log=output / "faults.jsonl")
        consumed = int(row["environment_steps"])
        official_after = _sha256(POLICY_FILE)
        saved = public_row(row)
        saved["policy_path"] = _relative(output, policy_path)
        _write_json(output / "episode.json", saved)
        (output / "steps.csv").write_text(_steps_csv(saved), encoding="utf-8")
        reference = json.loads((root / "results/legacy_formal_repro_r1/episodes/322000_scheme_c.json").read_text(encoding="utf-8"))
        comparison = _compare(reference, saved) if row["error"] is None else {"status": "not_compared", "error": row["error"]}
        _write_json(output / "regression_comparison.json", comparison)
        models = {
            "scheme_c_checkpoint": _relative(root, SCHEME_C_CKPT),
            "scheme_c_sha256": _sha256(SCHEME_C_CKPT),
            "mappo_checkpoint": _relative(root, primary_checkpoint()),
            "mappo_sha256": _sha256(primary_checkpoint()),
            "decoder_checkpoint": _relative(root, DECODER_CKPT),
            "decoder_sha256": _sha256(DECODER_CKPT),
            "epoch6_checkpoint": _relative(root, EPOCH6_CKPT),
            "epoch6_sha256": _sha256(EPOCH6_CKPT),
            "epoch6_role": "formal startup hash only; E6 is not the active high-level policy",
        }
        manifest = {
            "package": "legacy_system_l6",
            "instruction_commit": INSTRUCTION,
            "this_run": {
                "policy": saved["policy_path"],
                "episode": "episode.json",
                "steps": "steps.csv",
                "comparison": "regression_comparison.json",
                "fault_log": "faults.jsonl",
                "fault_log_written": (output / "faults.jsonl").is_file(),
            },
            "historical_reference": {
                "episode": "results/legacy_formal_repro_r1/episodes/322000_scheme_c.json",
                "config": "results/legacy_formal_repro_r1/configs/322000_scheme_c.json",
                "role": "saved R1 episode used for field comparison",
            },
            "run_context": {
                "output_absolute": str(output),
                "policy_absolute": str(policy_path),
                "lookup": "manifest paths are relative to the output or repository; absolute paths are machine context",
            },
            "runtime": _runtime(),
            "models": models,
            "model_hash_matches_contract": {name: models[name] == digest for name, digest in EXPECTED.items()},
            "code_sha256": {
                "run": _sha256(root / "src/wt/legacy_system/run.py"),
                "execute": _sha256(Path(__file__).resolve()),
            },
            "official_policy_sha256_before": official_before,
            "official_policy_sha256_after": official_after,
            "official_policy_unchanged": official_before == official_after,
            "scenario_id": int(scenario),
            "environment_steps": consumed,
            "training_updates": 0,
            "scheme_c_calls": int(row["scheme_c_calls"]),
            "complete": bool(row["complete"]),
            "error": row["error"],
            "regression_status": comparison["status"],
        }
        _write_json(output / "run_manifest.json", manifest)
        print(json.dumps({"regression_status": comparison["status"], "environment_steps": consumed, "training_updates": 0}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        _write_json(
            output / "failure.json",
            {"error": f"{type(exc).__name__}: {exc}", "environment_steps": consumed, "training_updates": 0},
        )
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}", "environment_steps": consumed}), flush=True)
        return 1
