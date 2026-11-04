"""Resolve saved R1 configs from an explicit artifact root.

The stored absolute path is kept as source metadata. Lookup uses the
scenario id and policy name, then checks the saved hash.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

POLICIES = ("scheme_c", "g1_rate")
SOURCE_RESULT_COMMIT = "1abcf1b264b6df1763cbcf2cfe410923cdc57dae"
READER_FILES = (
    "src/wt/legacy_formal_repro_r1a/__init__.py",
    "src/wt/legacy_formal_repro_r1a/read.py",
    "src/wt/legacy_formal_repro_r1a/inspect.py",
    "src/wt/legacy_formal_repro_r1a/export.py",
)


def config_relative(scenario_id: int, policy: str) -> str:
    if policy not in POLICIES:
        raise KeyError(f"no config mapping for policy {policy}")
    return f"configs/{int(scenario_id)}_{policy}.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(root: Path) -> dict:
    path = root / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing run_manifest.json under {root}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_config(root: Path, scenario_id: int, policy: str) -> tuple[Path, str]:
    relative = config_relative(scenario_id, policy)
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise FileNotFoundError(f"config path escapes the artifact root: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"missing config for scenario {scenario_id} policy {policy}: {relative}")
    return path, relative


def build_view(artifact_root: Path) -> dict:
    """Read one saved manifest. This does not run an episode or load a checkpoint."""
    root = Path(artifact_root)
    manifest = _load_manifest(root)
    arms = []
    for arm in manifest["arms"]:
        scenario_id = int(arm["scenario_id"])
        policy = str(arm["policy"])
        path, relative = _resolve_config(root, scenario_id, policy)
        digest = _sha256(path)
        expected = str(arm["policy_sha256_after"])
        if digest != expected:
            raise RuntimeError(f"hash mismatch for {relative}: saved {expected}, read {digest}")
        arms.append(
            {
                "scenario_id": scenario_id,
                "policy": policy,
                "config_relative": relative,
                "original_path": arm.get("policy_path"),
                "config_sha256": digest,
                "payload_bits": arm["payload_bits"],
                "environment_steps": int(arm["environment_steps"]),
                "decision_times": [int(item) for item in arm["decision_times"]],
            }
        )
    pairs = []
    indexed = {(row["scenario_id"], row["policy"]): row for row in arms}
    for pair in manifest["pairs"]:
        scenario_id = int(pair["scenario_id"])
        scheme = indexed[(scenario_id, "scheme_c")]["payload_bits"]
        baseline = indexed[(scenario_id, "g1_rate")]["payload_bits"]
        pairs.append(
            {
                "scenario_id": scenario_id,
                "scheme_c_payload_bits": pair["scheme_c_payload_bits"],
                "g1_rate_payload_bits": pair["g1_rate_payload_bits"],
                "payload_difference_bits": pair["payload_difference_bits"],
                "recomputed_payload_difference_bits": float(scheme) - float(baseline),
            }
        )
    missing = list(manifest.get("archive_summary", {}).get("missing", []))
    return {
        "artifact_root_role": "directory that contains run_manifest.json",
        "source_result_commit": SOURCE_RESULT_COMMIT,
        "source_environment_steps": int(manifest["environment_steps"]),
        "source_steps_are_a_new_run": False,
        "reader_environment_steps": 0,
        "reader_training_updates": 0,
        "reader_policy_inferences": 0,
        "reader_model_loads": 0,
        "reader_downloads": 0,
        "arms": arms,
        "pairs": pairs,
        "missing_original_summaries": missing,
        "provenance": {
            "original_run_code": manifest.get("runner_commit"),
            "result_organization_code": SOURCE_RESULT_COMMIT,
            "current_check_code": list(READER_FILES),
            "original_runtime": manifest.get("runtime"),
            "original_output_note": manifest.get("output_note"),
            "check_environment": {
                "python": sys.version.replace("\n", " "),
                "platform": platform.platform(),
                "torch_imported": False,
                "new_machine_install_verified": False,
            },
        },
    }


def assert_export_target(artifact_root: Path, output: Path) -> Path:
    root = Path(artifact_root).resolve()
    dest = Path(output).resolve()
    if dest == root or root in dest.parents or dest in root.parents:
        raise RuntimeError(f"export refused to write over the source artifact root: {dest}")
    manifest = dest / "derived_manifest.json"
    summary = dest / "summary.json"
    existing = [path for path in (manifest, summary) if path.exists()]
    if existing:
        raise FileExistsError(f"export refused to overwrite {existing[0]}")
    return dest
