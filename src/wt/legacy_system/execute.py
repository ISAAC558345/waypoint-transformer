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
from wt.legacy_system.completion import OutputRefusal, compare_episodes, conclude, prepare_output_directory


INSTRUCTION = "2b2fa2f3ca66d64688a122d939e2b9bdb5f6605a"
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


def run_once(root: Path, output: Path, scenario: int) -> int:
    consumed = 0
    try:
        prepare_output_directory(output)
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
        comparison = compare_episodes(reference, saved) if row["error"] is None else {
            "status": "not_compared",
            "error": row["error"],
            "internal_consistency_ok": False,
        }
        comparison["reference_episode"] = "results/legacy_formal_repro_r1/episodes/322000_scheme_c.json"
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
        decision = conclude(manifest)
        manifest["success"] = decision["success"]
        manifest["exit_code"] = decision["exit_code"]
        manifest["success_reasons"] = decision["reasons"]
        _write_json(output / "run_manifest.json", manifest)
        print(
            json.dumps(
                {
                    "regression_status": comparison["status"],
                    "success": decision["success"],
                    "exit_code": decision["exit_code"],
                    "environment_steps": consumed,
                    "training_updates": 0,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return int(decision["exit_code"])
    except OutputRefusal as exc:
        print(json.dumps({"error": str(exc), "exit_code": 2, "environment_steps": 0, "success": False}), flush=True)
        return 2
    except Exception as exc:
        failure = {
            "error": f"{type(exc).__name__}: {exc}",
            "environment_steps": consumed,
            "training_updates": 0,
            "success": False,
            "exit_code": 1,
        }
        try:
            output.mkdir(parents=True, exist_ok=True)
            if not any(output.iterdir()) or not (output / "run_manifest.json").exists():
                _write_json(output / "failure.json", failure)
        except OSError:
            pass
        print(json.dumps(failure), flush=True)
        return 1
