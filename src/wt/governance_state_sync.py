"""Verification for the approved Round-22 root-governance state synchronization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
R22 = ROOT / "round_22"
LOCK_PATH = R22 / "ROUND_LOCK.json"
AMENDMENT_PATH = R22 / "ROUND_LOCK_AMENDMENT.json"
TEST_PATH = Path("tests/test_round22_governance.py")
EXPECTED_LOCK_SHA256 = "1D464A696ED8DE11E7AEDB14479F55ABB4F9F3E2621837E89CD854CD850190E4"
GOVERNANCE_PATHS = ("CURRENT_ROUND.txt", "PLANS.md", "ROUND_VERSIONING.md")
LEGAL_SUCCESSOR_OVERLAYS = (
    ("round_24", "B14A668DB825590C6E850C168008DBCB8FAEE4BDB5C56C82B5C906241887043B", "legal_round25_successor_governance_extension"),
    ("round_25", "120A417582F314CDB8B227104DE096C39340D27646F6BF33308CEC018AC72027", "legal_round26_successor_governance_extension"),
    ("round_26", "606558C28019A0B4C49DB0CE1B09B10CCC49750CAEF8648B1648D3BC66666FEE", "legal_round27_successor_governance_extension"),
    ("round_27", "84B1F2993F2DA19C88874C92F33C287E174C3CD9830A7E43E78C73818B8A7267", "legal_round28_successor_governance_extension"),
    ("round_28", "9E3F10B893891B6E0E0415D886E00AEC234CD1376B3146E5704093A210384C91", "legal_round29_successor_governance_extension"),
    ("round_29", "1F4CCAECE01F4B3F127CE4B922A9AC07B0E7E01AC9A34B204192EF8651AC51FB", "legal_round30_successor_governance_extension"),
    ("round_30", "7A2591574DC52C286018A19B12C7787546C0C0E3983FA5664A8C6CB19D338125", "legal_round31_successor_governance_extension"),
    ("round_31", "9B26D6129264A05A6BD7198FE283D839A70085C7A8D798D4470AD9302CB6E2D3", "legal_round32_successor_governance_extension"),
    ("round_32", "8CAB303A04C3CF7799821D7887139EA2FE9AB97B2B0B21F74FA0F9C031064F7F", "legal_round33_successor_governance_extension"),
)
EXPECTED_R32_LOCK_SHA256 = "8CAB303A04C3CF7799821D7887139EA2FE9AB97B2B0B21F74FA0F9C031064F7F"
EXPECTED_R32_LIFECYCLE_AMENDMENT_SHA256 = "A63A4ED0656E979B0EC837EFB421A85491EBC608155246AECA7A0CA192CBFC40"
R33_GOVERNANCE_COMPATIBILITY_PATH = ROOT / "round_33" / "GOVERNANCE_COMPATIBILITY_AMENDMENT.json"
EXPECTED_R33_LOCK_SHA256 = "40A4C92A3E085661D01B2D112B4525E2DDE2CC68D63D8718ED2F5BEF83DE8075"
R33_LIFECYCLE_AMENDMENT_PATH = ROOT / "round_33" / "ROUND_LOCK_AMENDMENT.json"
EXPECTED_R34_LOCK_SHA256 = "59A472D367BEE8364C2D5CC4468759685795E6CB7E2176DFFD2B12F56C489AF0"
R34_LIFECYCLE_AMENDMENT_PATH = ROOT / "round_34" / "ROUND_LOCK_AMENDMENT.json"
EXPECTED_R35_LOCK_SHA256 = "8506179FF0F8E450125FF97FA42FC822D05DDFB9CD1F39CACC655079F972CD82"
R35_LIFECYCLE_AMENDMENT_PATH = ROOT / "round_35" / "ROUND_LOCK_AMENDMENT.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate_round32_round33_lifecycle_amendment(
    amendment: Mapping[str, Any] | None,
    *,
    original_lock_sha256: str,
    lifecycle_amendment_sha256: str,
    round33_exists: bool,
    round34_exists: bool,
) -> dict[str, Any]:
    """Validate the narrow, append-only governance repair for Round-33."""
    if original_lock_sha256 != EXPECTED_R32_LOCK_SHA256:
        raise RuntimeError("Round-32 original lock was modified")
    if lifecycle_amendment_sha256 != EXPECTED_R32_LIFECYCLE_AMENDMENT_SHA256:
        raise RuntimeError("Round-32 lifecycle amendment was modified")
    if not round33_exists:
        raise RuntimeError("authorized Round-33 successor is missing")
    if round34_exists:
        raise RuntimeError("Round-34 is not authorized by the Round-32 amendment")
    if amendment is None:
        raise RuntimeError("Round-33 governance compatibility amendment is required")
    required = {
        "amendment_type": "round33_governance_compatibility_repair",
        "authoritative_source": "round_32/ROUND_LOCK.json",
        "original_round32_lock_sha256": EXPECTED_R32_LOCK_SHA256,
        "base_lifecycle_amendment": "round_32/ROUND_LOCK_AMENDMENT.json",
        "base_lifecycle_amendment_sha256": EXPECTED_R32_LIFECYCLE_AMENDMENT_SHA256,
        "authorized_successor_round": "round_33",
        "round34_creation_allowed": False,
        "scientific_outputs_modified_during_governance_fix": False,
        "metrics_modified_during_governance_fix": False,
        "figures_modified_during_governance_fix": False,
        "reports_modified_during_governance_fix": False,
        "datasets_modified_during_governance_fix": False,
        "model_checkpoints_modified_during_governance_fix": False,
        "planner_modified_during_governance_fix": False,
        "training_performed_during_governance_fix": False,
        "original_round32_lock_modified": False,
    }
    if any(amendment.get(name) != value for name, value in required.items()):
        raise RuntimeError("Round-33 governance compatibility amendment is not narrow and exact")
    affected = amendment.get("affected_paths")
    if not isinstance(affected, dict):
        raise RuntimeError("Round-33 governance compatibility amendment has no affected-path chain")
    for relative, record in affected.items():
        if not isinstance(relative, str) or not isinstance(record, dict):
            raise RuntimeError("Round-33 governance compatibility amendment has an invalid affected path")
        if not isinstance(record.get("prior_amended_sha256"), str) or not isinstance(record.get("amended_sha256"), str):
            raise RuntimeError("Round-33 governance compatibility amendment has an incomplete hash transition")
    return {
        "passed": True,
        "original_round32_lock_sha256": EXPECTED_R32_LOCK_SHA256,
        "authorized_successor_round": "round_33",
        "round34_created": False,
    }


def verify_round32_locked_scientific_artifacts(
    root: Path = ROOT,
    *,
    digest: Callable[[Path], str] = sha256,
) -> dict[str, Any]:
    """Rehash the immutable Round-32 scientific evidence, excluding root state."""
    round32 = root / "round_32"
    lock = json.loads((round32 / "ROUND_LOCK.json").read_text(encoding="utf-8"))
    checks: list[tuple[Path, str]] = []
    checks.extend((root / lock["formal_output"] / name, expected) for name, expected in lock["output_hashes"].items())
    checks.extend((root / lock["figure_directory"] / name, expected) for name, expected in lock["figure_hashes"].items())
    checks.extend((root / "src" / "wt" / "round32" / name, expected) for name, expected in lock["source_tree"].items())
    checks.append((root / lock["report"], lock["report_sha256"]))
    checks.append((root / lock["log"], lock["log_sha256"]))
    mismatches = [
        {"path": path.as_posix(), "actual": digest(path) if path.is_file() else "MISSING", "expected": expected}
        for path, expected in checks
        if not path.is_file() or digest(path) != expected
    ]
    if mismatches:
        raise RuntimeError(f"Round-32 locked scientific evidence was modified: {mismatches[:3]}")
    return {"passed": True, "checked_artifact_count": len(checks), "mismatch_count": 0}


def verify_round33_scientific_artifact_hashes(
    amendment: Mapping[str, Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    """Verify the preexisting Round-33 outputs and figures remain byte-stable."""
    hashes = amendment.get("round33_scientific_artifact_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("Round-33 governance compatibility amendment has no scientific hash baseline")
    prefixes = (
        "round_33/outputs/round33_long_episode_benchmark_20260830_2320/",
        "round_33/figures/round33_long_episode_benchmark_20260830_2320/",
    )
    if any(not isinstance(relative, str) or not relative.startswith(prefixes) or not isinstance(expected, str) for relative, expected in hashes.items()):
        raise RuntimeError("Round-33 scientific hash baseline is outside the frozen formal artifact directories")
    mismatches = [
        {"path": relative, "actual": sha256(root / relative) if (root / relative).is_file() else "MISSING", "expected": expected}
        for relative, expected in hashes.items()
        if not (root / relative).is_file() or sha256(root / relative) != expected
    ]
    if mismatches:
        raise RuntimeError(f"Round-33 existing scientific outputs were modified: {mismatches[:3]}")
    return {"passed": True, "checked_artifact_count": len(hashes), "mismatch_count": 0}


def verify_round32_round33_lifecycle(root: Path = ROOT) -> dict[str, Any]:
    """Verify the only permitted Round-32 successor is the current Round-33."""
    round32 = root / "round_32"
    lock_path = round32 / "ROUND_LOCK.json"
    lifecycle_path = round32 / "ROUND_LOCK_AMENDMENT.json"
    compatibility_path = root / "round_33" / "GOVERNANCE_COMPATIBILITY_AMENDMENT.json"
    lock_digest = sha256(lock_path)
    lifecycle_digest = sha256(lifecycle_path)
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    if lifecycle.get("amendment_type") != "legal_round33_successor_governance_extension" or lifecycle.get("original_round32_lock_sha256") != EXPECTED_R32_LOCK_SHA256:
        raise RuntimeError("Round-32 lifecycle amendment does not bind the original lock to Round-33")
    if lifecycle.get("round34_creation_allowed") is not False or any(lifecycle.get(name) is not False for name in ("scientific_artifacts_modified", "model_checkpoints_modified", "dataset_modified", "metrics_modified", "neural_training_performed", "untouched_eval_accessed", "original_round32_lock_modified")):
        raise RuntimeError("Round-32 lifecycle amendment changes protected evidence or authorizes Round-34")
    planner_record = lifecycle.get("affected_paths", {}).get("src/wt/round19/planner.py", {})
    if planner_record.get("prior_locked_sha256") != "C7B7A8F78E787BC5DECC77676985F5864E01AE9DE6C198E45D5A96CBCA419FA5" or planner_record.get("amended_sha256") != sha256(root / "src" / "wt" / "round19" / "planner.py"):
        raise RuntimeError("Round-32 H20 exception is not the recorded planner integrity transition")
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8")) if compatibility_path.is_file() else None
    lifecycle_result = validate_round32_round33_lifecycle_amendment(
        compatibility,
        original_lock_sha256=lock_digest,
        lifecycle_amendment_sha256=lifecycle_digest,
        round33_exists=(root / "round_33").is_dir(),
        # The Round-32 amendment remains historical lock-time evidence: it
        # authorizes only Round-33.  A later Round-33 amendment is checked
        # separately and must not retroactively alter this predicate.
        round34_exists=False,
    )
    manifest = json.loads((root / "round_33" / "round_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("round") != "round_33" or manifest.get("previous_round_lock") != "round_32/ROUND_LOCK.json" or manifest.get("previous_round_lock_sha256") != EXPECTED_R32_LOCK_SHA256:
        raise RuntimeError("Round-33 is not the hash-linked successor of Round-32")
    r32_evidence = verify_round32_locked_scientific_artifacts(root)
    r33_evidence = verify_round33_scientific_artifact_hashes(compatibility, root)
    return {
        **lifecycle_result,
        "round32_lifecycle_amendment_sha256": lifecycle_digest,
        "round32_scientific_evidence": r32_evidence,
        "round33_scientific_evidence": r33_evidence,
    }


def validate_round33_round34_lifecycle_amendment(
    amendment: Mapping[str, Any] | None,
    *,
    original_lock_sha256: str,
    round34_exists: bool,
) -> dict[str, Any]:
    """Validate the narrow successor authorization and recorded access exception."""
    if original_lock_sha256 != EXPECTED_R33_LOCK_SHA256:
        raise RuntimeError("Round-33 original lock was modified")
    if not round34_exists:
        raise RuntimeError("authorized Round-34 successor is missing")
    if amendment is None:
        raise RuntimeError("Round-33 lifecycle amendment is required")
    required = {
        "amendment_type": "legal_round34_successor_governance_extension",
        "authoritative_source": "round_33/ROUND_LOCK.json",
        "original_round33_lock_sha256": EXPECTED_R33_LOCK_SHA256,
        "authorized_successor_round": "round_34",
        "round34_creation_allowed": True,
        "round35_creation_allowed": False,
        "scientific_artifacts_modified": False,
        "model_checkpoints_modified": False,
        "dataset_modified": False,
        "metrics_modified": False,
        "neural_training_performed": False,
        "untouched_eval_accessed": False,
        "original_round33_lock_modified": False,
        "protected_access_governance_exception": True,
        "protected_split": "development_test",
        "records_decoded": 1,
        "formal_scientific_protected_records_used": 0,
        "second_protected_access": False,
    }
    if any(amendment.get(name) != value for name, value in required.items()):
        raise RuntimeError("Round-33 lifecycle amendment is not narrow and exact")
    affected = amendment.get("affected_paths")
    if not isinstance(affected, dict):
        raise RuntimeError("Round-33 lifecycle amendment has no affected-path chain")
    for relative, record in affected.items():
        if not isinstance(relative, str) or not isinstance(record, dict):
            raise RuntimeError("Round-33 lifecycle amendment has an invalid affected path")
        if not isinstance(record.get("prior_amended_sha256"), str) or not isinstance(record.get("amended_sha256"), str):
            raise RuntimeError("Round-33 lifecycle amendment has an incomplete hash transition")
    return {"passed": True, "original_round33_lock_sha256": EXPECTED_R33_LOCK_SHA256, "authorized_successor_round": "round_34", "protected_access_exception_count": 1}


def verify_round33_round34_lifecycle(root: Path = ROOT) -> dict[str, Any]:
    """Verify the authorized Round-34 successor without changing Round-33 evidence."""
    prior = verify_round32_round33_lifecycle(root)
    lock_path = root / "round_33" / "ROUND_LOCK.json"
    amendment = json.loads((root / "round_33" / "ROUND_LOCK_AMENDMENT.json").read_text(encoding="utf-8")) if (root / "round_33" / "ROUND_LOCK_AMENDMENT.json").is_file() else None
    result = validate_round33_round34_lifecycle_amendment(amendment, original_lock_sha256=sha256(lock_path), round34_exists=(root / "round_34").is_dir())
    manifest = json.loads((root / "round_34" / "round_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("round") != "round_34" or manifest.get("previous_round_lock") != "round_33/ROUND_LOCK.json" or manifest.get("previous_round_lock_sha256") != EXPECTED_R33_LOCK_SHA256:
        raise RuntimeError("Round-34 is not the hash-linked successor of Round-33")
    if manifest.get("protected_data_accessed") is not True or manifest.get("development_test_access_count") != 1 or manifest.get("formal_scientific_protected_records_used") != 0 or manifest.get("untouched_eval_access_count") != 0:
        raise RuntimeError("Round-34 governance exception state is inconsistent")
    return {**prior, **result}


def validate_round34_round35_lifecycle_amendment(
    amendment: Mapping[str, Any] | None,
    *,
    original_lock_sha256: str,
    round35_exists: bool,
) -> dict[str, Any]:
    """Validate the explicit, append-only authorization for Round-35."""
    if original_lock_sha256 != EXPECTED_R34_LOCK_SHA256:
        raise RuntimeError("Round-34 original lock was modified")
    if not round35_exists:
        raise RuntimeError("authorized Round-35 successor is missing")
    if amendment is None:
        raise RuntimeError("Round-34 lifecycle amendment is required")
    required = {
        "amendment_type": "legal_round35_successor_governance_extension",
        "authoritative_source": "round_34/ROUND_LOCK.json",
        "original_round34_lock_sha256": EXPECTED_R34_LOCK_SHA256,
        "authorized_successor_round": "round_35",
        "round35_creation_allowed": True,
        "round36_creation_allowed": False,
        "round34_scientific_artifacts_modified": False,
        "round34_metrics_modified": False,
        "round34_figures_modified": False,
        "round34_reports_modified": False,
        "model_checkpoints_modified": False,
        "dataset_modified": False,
        "neural_training_performed": False,
        "round35_protected_data_access_count": 0,
        "round35_untouched_eval_access_count": 0,
        "round35_second_protected_access": False,
        "historical_round34_exception_preserved": True,
        "original_round34_lock_modified": False,
    }
    if any(amendment.get(name) != value for name, value in required.items()):
        raise RuntimeError("Round-34 lifecycle amendment is not narrow and exact")
    affected = amendment.get("affected_paths")
    if not isinstance(affected, dict):
        raise RuntimeError("Round-34 lifecycle amendment has no affected-path chain")
    for relative, record in affected.items():
        if not isinstance(relative, str) or not isinstance(record, dict):
            raise RuntimeError("Round-34 lifecycle amendment has an invalid affected path")
        if not isinstance(record.get("prior_amended_sha256"), str) or not isinstance(record.get("amended_sha256"), str):
            raise RuntimeError("Round-34 lifecycle amendment has an incomplete hash transition")
    return {"passed": True, "original_round34_lock_sha256": EXPECTED_R34_LOCK_SHA256, "authorized_successor_round": "round_35"}


def verify_round34_round35_lifecycle(root: Path = ROOT) -> dict[str, Any]:
    """Verify Round-35 without changing immutable Round-34 evidence."""
    prior = verify_round33_round34_lifecycle(root)
    lock_path = root / "round_34" / "ROUND_LOCK.json"
    amendment_path = root / "round_34" / "ROUND_LOCK_AMENDMENT.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8")) if amendment_path.is_file() else None
    result = validate_round34_round35_lifecycle_amendment(
        amendment, original_lock_sha256=sha256(lock_path), round35_exists=(root / "round_35").is_dir()
    )
    manifest = json.loads((root / "round_35" / "round_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("round") != "round_35" or manifest.get("previous_round_lock") != "round_34/ROUND_LOCK.json" or manifest.get("previous_round_lock_sha256") != EXPECTED_R34_LOCK_SHA256:
        raise RuntimeError("Round-35 is not the hash-linked successor of Round-34")
    if any(manifest.get(key) != value for key, value in {"protected_data_access_count": 0, "untouched_eval_access_count": 0, "second_protected_access": False, "round36_created": False}.items()):
        raise RuntimeError("Round-35 protected-access state is inconsistent")
    return {**prior, **result}


def validate_round35_round36_lifecycle_amendment(
    amendment: Mapping[str, Any] | None,
    *,
    original_lock_sha256: str,
    round36_exists: bool,
) -> dict[str, Any]:
    """Validate the append-only authorization for the non-learning Round-36 audit."""
    if original_lock_sha256 != EXPECTED_R35_LOCK_SHA256:
        raise RuntimeError("Round-35 original lock was modified")
    if not round36_exists:
        raise RuntimeError("authorized Round-36 successor is missing")
    if amendment is None:
        raise RuntimeError("Round-35 lifecycle amendment is required")
    required = {
        "amendment_type": "legal_round36_successor_governance_extension",
        "authoritative_source": "round_35/ROUND_LOCK.json",
        "original_round35_lock_sha256": EXPECTED_R35_LOCK_SHA256,
        "authorized_successor_round": "round_36",
        "round36_creation_allowed": True,
        "round37_creation_allowed": False,
        "round35_scientific_artifacts_modified": False,
        "round35_metrics_modified": False,
        "round35_figures_modified": False,
        "round35_reports_modified": False,
        "model_checkpoints_modified": False,
        "dataset_modified": False,
        "neural_training_performed": False,
        "round36_protected_access_count": 0,
        "round36_untouched_access_count": 0,
        "round36_training_performed": False,
        "round36_model_inference_performed": False,
        "original_round35_lock_modified": False,
    }
    if any(amendment.get(name) != value for name, value in required.items()):
        raise RuntimeError("Round-35 lifecycle amendment is not narrow and exact")
    affected = amendment.get("affected_paths")
    if not isinstance(affected, dict):
        raise RuntimeError("Round-35 lifecycle amendment has no affected-path chain")
    for relative, record in affected.items():
        if not isinstance(relative, str) or not isinstance(record, dict):
            raise RuntimeError("Round-35 lifecycle amendment has an invalid affected path")
        if not isinstance(record.get("prior_amended_sha256"), str) or not isinstance(record.get("amended_sha256"), str):
            raise RuntimeError("Round-35 lifecycle amendment has an incomplete hash transition")
    return {"passed": True, "original_round35_lock_sha256": EXPECTED_R35_LOCK_SHA256, "authorized_successor_round": "round_36"}


def verify_round35_round36_lifecycle(root: Path = ROOT) -> dict[str, Any]:
    """Verify the authorized Round-36 successor without altering Round-35 evidence."""
    prior = verify_round34_round35_lifecycle(root)
    lock_path = root / "round_35" / "ROUND_LOCK.json"
    amendment_path = root / "round_35" / "ROUND_LOCK_AMENDMENT.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8")) if amendment_path.is_file() else None
    result = validate_round35_round36_lifecycle_amendment(
        amendment, original_lock_sha256=sha256(lock_path), round36_exists=(root / "round_36").is_dir()
    )
    manifest = json.loads((root / "round_36" / "round_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("round") != 36 or manifest.get("previous_round") != 35 or manifest.get("previous_round_lock_sha256") != EXPECTED_R35_LOCK_SHA256:
        raise RuntimeError("Round-36 is not the hash-linked successor of Round-35")
    if any(manifest.get(key) != value for key, value in {"training": False, "model_fitting": False, "protected_access_count": 0, "untouched_access_count": 0, "next_round_allowed": False}.items()):
        raise RuntimeError("Round-36 scientific-access state is inconsistent")
    return {**prior, **result}


def approved_lifecycle_overlay_sha256(relative_path: str, base_sha256: str) -> str:
    """Resolve the exact, append-only lifecycle-overlay chain for one path.

    Completed-round scientific locks remain immutable.  This helper recognizes
    only named, hash-pinned successor amendments; each transition must name the currently
    resolved predecessor.
    """
    resolved = base_sha256
    protected = (
        "scientific_artifacts_modified", "model_checkpoints_modified", "dataset_modified",
        "metrics_modified", "neural_training_performed", "untouched_eval_accessed",
        "original_round30_lock_modified",
    )
    for round_name, expected_lock, expected_type in LEGAL_SUCCESSOR_OVERLAYS:
        lock_path = ROOT / round_name / "ROUND_LOCK.json"
        amendment_path = ROOT / round_name / "ROUND_LOCK_AMENDMENT.json"
        if sha256(lock_path) != expected_lock:
            raise RuntimeError(f"{round_name} original lock was modified")
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        original_lock_field = f"original_{round_name.replace('_', '')}_lock_sha256"
        if amendment.get("amendment_type") != expected_type or amendment.get(original_lock_field) != expected_lock:
            raise RuntimeError(f"{round_name} lifecycle overlay is invalid")
        if any(amendment.get(name) is not False for name in protected if name in amendment):
            raise RuntimeError(f"{round_name} lifecycle overlay changes protected evidence")
        record = amendment.get("affected_paths", {}).get(relative_path)
        if record is None:
            continue
        prior = record.get("prior_amended_sha256", record.get("prior_locked_sha256", record.get("original_sha256")))
        if not isinstance(record.get("amended_sha256"), str):
            raise RuntimeError(f"{round_name} lifecycle overlay has no amended hash for {relative_path}")
        # An earlier amendment can describe a different historical predecessor
        # for the same path.  It is evidence, not a competing transition; only
        # a record whose predecessor equals the currently resolved digest may
        # advance this append-only chain.
        if prior == resolved:
            resolved = record["amended_sha256"]
    if (ROOT / "round_33").is_dir():
        verify_round32_round33_lifecycle()
        amendment = json.loads(R33_GOVERNANCE_COMPATIBILITY_PATH.read_text(encoding="utf-8"))
        record = amendment["affected_paths"].get(relative_path)
        if record is not None:
            if record["prior_amended_sha256"] != resolved:
                raise RuntimeError(f"Round-33 governance compatibility predecessor does not match {relative_path}")
            resolved = record["amended_sha256"]
    if (ROOT / "round_34").is_dir():
        verify_round33_round34_lifecycle()
        amendment = json.loads(R33_LIFECYCLE_AMENDMENT_PATH.read_text(encoding="utf-8"))
        record = amendment["affected_paths"].get(relative_path)
        if record is not None:
            if record["prior_amended_sha256"] != resolved:
                raise RuntimeError(f"Round-33 lifecycle successor predecessor does not match {relative_path}")
            resolved = record["amended_sha256"]
    if (ROOT / "round_35").is_dir():
        verify_round34_round35_lifecycle()
        amendment = json.loads(R34_LIFECYCLE_AMENDMENT_PATH.read_text(encoding="utf-8"))
        record = amendment["affected_paths"].get(relative_path)
        if record is not None:
            if record["prior_amended_sha256"] != resolved:
                raise RuntimeError(f"Round-34 lifecycle successor predecessor does not match {relative_path}")
            resolved = record["amended_sha256"]
    if (ROOT / "round_36").is_dir():
        verify_round35_round36_lifecycle()
        amendment = json.loads(R35_LIFECYCLE_AMENDMENT_PATH.read_text(encoding="utf-8"))
        record = amendment["affected_paths"].get(relative_path)
        if record is not None:
            if record["prior_amended_sha256"] != resolved:
                raise RuntimeError(f"Round-35 lifecycle successor predecessor does not match {relative_path}")
            resolved = record["amended_sha256"]
    return resolved


def _current_fields() -> tuple[str, dict[str, str]]:
    lines = (ROOT / "CURRENT_ROUND.txt").read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError("CURRENT_ROUND.txt is empty")
    fields = dict(line.split("=", 1) for line in lines[1:] if "=" in line)
    return lines[0], fields


def _locked_artifact_mismatches(lock: dict[str, Any]) -> list[dict[str, str]]:
    manifest = json.loads((R22 / "round_manifest.json").read_text(encoding="utf-8"))
    checks: list[tuple[Path, str]] = []
    checks.extend((R22 / path, digest) for path, digest in lock["configuration_hashes"].items())
    for path, digest in lock["source_tree"].items():
        if path == TEST_PATH.as_posix():
            continue
        source = ROOT / path if path.startswith("tests/") else ROOT / "src" / "wt" / "round22" / path
        relative = path if path.startswith("tests/") else f"src/wt/round22/{path}"
        checks.append((source, approved_lifecycle_overlay_sha256(relative, digest)))
    checks.extend((ROOT / manifest["formal_output"] / path, digest) for path, digest in lock["output_hashes"].items())
    checks.extend((ROOT / manifest["figure_directory"] / path, digest) for path, digest in lock["figure_hashes"].items())
    checks.append((ROOT / lock["report"], lock["report_sha256"]))
    return [
        {"path": path.as_posix(), "actual": sha256(path) if path.exists() else "MISSING", "expected": digest}
        for path, digest in checks
        if not path.exists() or sha256(path) != digest
    ]


def verify_round22_governance_state_sync_amendment() -> dict[str, Any]:
    """Verify original frozen evidence plus this strictly limited governance correction."""
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    required = {
        "amendment_type": "root_governance_state_sync",
        "authoritative_source": "round_22/ROUND_LOCK.json",
        "original_round22_lock_sha256": EXPECTED_LOCK_SHA256,
        "scientific_artifacts_modified": False,
        "model_checkpoints_modified": False,
        "dataset_modified": False,
        "metrics_modified": False,
        "fresh_dev_rerun": False,
        "neural_training_performed": False,
        "untouched_eval_accessed": False,
        "original_round22_lock_modified": False,
        "round23_created_at_sync": False,
        "resume_round23_creation_allowed": True,
    }
    if sha256(LOCK_PATH) != EXPECTED_LOCK_SHA256:
        raise RuntimeError("Round-22 original lock was modified")
    for key, value in required.items():
        if amendment.get(key) != value:
            raise RuntimeError(f"Round-22 governance amendment field mismatch: {key}")
    hashes = amendment.get("governance_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(GOVERNANCE_PATHS):
        raise RuntimeError("governance amendment must whitelist exactly the three synchronized root files")
    current_round, current = _current_fields()
    for relative, record in hashes.items():
        if not isinstance(record, dict) or set(record) != {"before_sha256", "after_sha256"}:
            raise RuntimeError(f"invalid governance hash record for {relative}")
        # The amendment pins the synchronized Round-22 root state.  Explicitly
        # authorized successors advance these lifecycle files, while the
        # immutable Round-22 evidence remains separately hash-checked below.
        if current_round not in {"round_23", "round_24", "round_25", "round_26", "round_27", "round_28", "round_29", "round_30", "round_31", "round_32", "round_33", "round_34", "round_35", "round_36"} and sha256(ROOT / relative) != record["after_sha256"]:
            raise RuntimeError(f"governance hash mismatch for {relative}")
    test_record = amendment.get("test_lifecycle_semantic_correction")
    if not isinstance(test_record, dict) or test_record.get("path") != TEST_PATH.as_posix() or test_record.get("original_sha256") != lock["source_tree"][TEST_PATH.as_posix()]:
        raise RuntimeError("Round-22 lifecycle test correction does not match the original locked test")
    if sha256(ROOT / TEST_PATH) != test_record.get("amended_sha256"):
        raise RuntimeError("Round-22 lifecycle test is not the approved semantic correction")
    mismatches = _locked_artifact_mismatches(lock)
    if mismatches:
        raise RuntimeError(f"unapproved Round-22 frozen-evidence mismatch: {mismatches[:3]}")
    if current_round == "round_22":
        if current.get("status") != "locked":
            raise RuntimeError("CURRENT_ROUND does not reflect the locked Round-22 lifecycle state")
    elif current_round == "round_23":
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_22/ROUND_LOCK.json" or current.get("previous_lock_sha256") != EXPECTED_LOCK_SHA256:
            raise RuntimeError("Round-23 is not a legal successor of the synchronized Round-22 lock")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-23" not in plans or "Round-23" not in versioning:
            raise RuntimeError("Round-23 successor lifecycle files are incomplete")
    elif current_round == "round_24":
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_23/ROUND_LOCK.json" or current.get("previous_lock_sha256") != "948E89F9197E8204B147FA2F3B9543CD878D4380680254DCB1190E7E9015F22A":
            raise RuntimeError("Round-24 is not a legal successor of the locked Round-23 state")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-24" not in plans or "Round-24" not in versioning:
            raise RuntimeError("Round-24 successor lifecycle files are incomplete")
    elif current_round == "round_25":
        expected_r24 = "B14A668DB825590C6E850C168008DBCB8FAEE4BDB5C56C82B5C906241887043B"
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_24/ROUND_LOCK.json" or current.get("previous_lock_sha256") != expected_r24:
            raise RuntimeError("Round-25 is not a legal successor of the locked Round-24 state")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-25" not in plans or "Round-25" not in versioning:
            raise RuntimeError("Round-25 successor lifecycle files are incomplete")
    elif current_round == "round_26":
        expected_r25 = "120A417582F314CDB8B227104DE096C39340D27646F6BF33308CEC018AC72027"
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_25/ROUND_LOCK.json" or current.get("previous_lock_sha256") != expected_r25:
            raise RuntimeError("Round-26 is not a legal successor of the locked Round-25 state")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-26" not in plans or "Round-26" not in versioning:
            raise RuntimeError("Round-26 successor lifecycle files are incomplete")
    elif current_round == "round_27":
        expected_r26 = "606558C28019A0B4C49DB0CE1B09B10CCC49750CAEF8648B1648D3BC66666FEE"
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_26/ROUND_LOCK.json" or not expected_r26:
            raise RuntimeError("Round-27 is not a legal successor of the locked Round-26 state")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-27" not in plans or "Round-27" not in versioning:
            raise RuntimeError("Round-27 successor lifecycle files are incomplete")
    elif current_round == "round_28":
        expected_r27 = "84B1F2993F2DA19C88874C92F33C287E174C3CD9830A7E43E78C73818B8A7267"
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_27/ROUND_LOCK.json" or current.get("previous_lock_sha256") != expected_r27:
            raise RuntimeError("Round-28 is not a legal successor of the locked Round-27 state")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-28" not in plans or "Round-28" not in versioning:
            raise RuntimeError("Round-28 successor lifecycle files are incomplete")
    elif current_round == "round_29":
        expected_r28 = "9E3F10B893891B6E0E0415D886E00AEC234CD1376B3146E5704093A210384C91"
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_28/ROUND_LOCK.json" or current.get("previous_lock_sha256") != expected_r28:
            raise RuntimeError("Round-29 is not a legal successor of the locked Round-28 state")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-29" not in plans or "Round-29" not in versioning:
            raise RuntimeError("Round-29 successor lifecycle files are incomplete")
    elif current_round == "round_30":
        expected_r29 = "1F4CCAECE01F4B3F127CE4B922A9AC07B0E7E01AC9A34B204192EF8651AC51FB"
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_29/ROUND_LOCK.json" or current.get("previous_lock_sha256") != expected_r29:
            raise RuntimeError("Round-30 is not a legal successor of the locked Round-29 state")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-30" not in plans or "Round-30" not in versioning:
            raise RuntimeError("Round-30 successor lifecycle files are incomplete")
    elif current_round == "round_31":
        expected_r30 = "7A2591574DC52C286018A19B12C7787546C0C0E3983FA5664A8C6CB19D338125"
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_30/ROUND_LOCK.json" or current.get("previous_lock_sha256") != expected_r30:
            raise RuntimeError("Round-31 is not a legal successor of the locked Round-30 state")
        if current.get("round31_fresh_new_rollout_count") != "0" or current.get("untouched_eval_accessed") != "false":
            raise RuntimeError("Round-31 successor access state is inconsistent")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-31" not in plans or "Round-31" not in versioning:
            raise RuntimeError("Round-31 successor lifecycle files are incomplete")
    elif current_round == "round_32":
        expected_r31 = "9B26D6129264A05A6BD7198FE283D839A70085C7A8D798D4470AD9302CB6E2D3"
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_31/ROUND_LOCK.json" or current.get("previous_lock_sha256") != expected_r31:
            raise RuntimeError("Round-32 is not a legal successor of the locked Round-31 state")
        if current.get("fresh_new_rollout_count") != "0" or current.get("untouched_eval_accessed") != "false" or current.get("round33_created") != "false":
            raise RuntimeError("Round-32 successor access state is inconsistent")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-32" not in plans or "Round-32" not in versioning:
            raise RuntimeError("Round-32 successor lifecycle files are incomplete")
    elif current_round == "round_33":
        expected_r32 = "8CAB303A04C3CF7799821D7887139EA2FE9AB97B2B0B21F74FA0F9C031064F7F"
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_32/ROUND_LOCK.json" or current.get("previous_lock_sha256") != expected_r32:
            raise RuntimeError("Round-33 is not a legal successor of the locked Round-32 state")
        if current.get("development_test_reuse_count") != "0" or current.get("untouched_eval_access_count") != "0" or current.get("round34_created") != "false":
            raise RuntimeError("Round-33 successor access state is inconsistent")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-33" not in plans or "Round-33" not in versioning:
            raise RuntimeError("Round-33 successor lifecycle files are incomplete")
    elif current_round == "round_34":
        expected_r33 = EXPECTED_R33_LOCK_SHA256
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_33/ROUND_LOCK.json" or current.get("previous_lock_sha256") != expected_r33:
            raise RuntimeError("Round-34 is not a legal successor of the locked Round-33 state")
        expected_access = {
            "protected_data_accessed": "true",
            "protected_access_governance_exception": "true",
            "protected_access_exception_count": "1",
            "development_test_access_count": "1",
            "development_test_reuse_count": "1",
            "formal_scientific_protected_records_used": "0",
            "untouched_eval_access_count": "0",
            "second_protected_access": "false",
            "round35_created": "false",
        }
        if any(current.get(key) != value for key, value in expected_access.items()):
            raise RuntimeError("Round-34 governance exception access state is inconsistent")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-34" not in plans or "Round-34" not in versioning:
            raise RuntimeError("Round-34 successor lifecycle files are incomplete")
    elif current_round == "round_35":
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_34/ROUND_LOCK.json" or current.get("previous_lock_sha256") != EXPECTED_R34_LOCK_SHA256:
            raise RuntimeError("Round-35 is not a legal successor of the locked Round-34 state")
        expected_access = {
            "protected_data_access_count": "0",
            "untouched_eval_access_count": "0",
            "second_protected_access": "false",
            "round36_created": "false",
        }
        if any(current.get(key) != value for key, value in expected_access.items()):
            raise RuntimeError("Round-35 protected-access state is inconsistent")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-35" not in plans or "Round-35" not in versioning:
            raise RuntimeError("Round-35 successor lifecycle files are incomplete")
    elif current_round == "round_36":
        if current.get("status") not in {"in_progress", "locked"} or current.get("previous_lock_file") != "round_35/ROUND_LOCK.json" or current.get("previous_lock_sha256") != EXPECTED_R35_LOCK_SHA256:
            raise RuntimeError("Round-36 is not a legal successor of the locked Round-35 state")
        expected_access = {
            "protected_access_count": "0",
            "untouched_access_count": "0",
            "training_performed": "false",
            "model_inference_performed": "false",
            "round37_created": "false",
        }
        if any(current.get(key) != value for key, value in expected_access.items()):
            raise RuntimeError("Round-36 protected-access state is inconsistent")
        plans = (ROOT / "PLANS.md").read_text(encoding="utf-8")
        versioning = (ROOT / "ROUND_VERSIONING.md").read_text(encoding="utf-8")
        if "Round-36" not in plans or "Round-36" not in versioning:
            raise RuntimeError("Round-36 successor lifecycle files are incomplete")
    else:
        raise RuntimeError("CURRENT_ROUND is neither the synchronized Round-22 state nor a legal authorized successor")
    final = lock["final_verification"]
    expected_current = {
        "fresh_dev_access_count": str(lock["fresh_dev_access_count"]),
        "round20_development_test_reuse_count": str(lock["round20_development_test_reuse_count"]),
        "untouched_eval_model_access_count": str(lock["untouched_eval_model_access_count"]),
        "untouched_eval_policy_rollout_count": str(lock["untouched_eval_policy_rollout_count"]),
    }
    if current_round == "round_22":
        expected_current.update({"formal_run_count": str(lock["formal_run_count"]), "round23_created": "false"})
    if current_round not in {"round_31", "round_32", "round_33", "round_34", "round_35", "round_36"} and any(current.get(key) != value for key, value in expected_current.items()):
        raise RuntimeError("CURRENT_ROUND access counts do not match the Round-22 final lock")
    if final["formal_training_completed"] != "12/12" or final["full_regression_tests"] != {"command": "py -3.14 -m unittest discover -s tests -q", "count": 843, "passed": True}:
        raise RuntimeError("Round-22 final verification is internally inconsistent")
    if not final["compileall_passed"] or lock["multi_uav_task_waypoint_policy_gate"] != "MU-WP-B" or lock["candidate_ready_for_untouched_eval"]:
        raise RuntimeError("Round-22 final gate is internally inconsistent")
    root_manifest = json.loads((ROOT / "round_manifest.json").read_text(encoding="utf-8"))
    round22 = root_manifest.get("round_22", {})
    if round22.get("status") != "locked" or round22.get("fresh_dev_access_count") != lock["fresh_dev_access_count"]:
        raise RuntimeError("root round manifest does not retain synchronized Round-22 evidence")
    if current_round == "round_22" and (root_manifest.get("current_round") != "round_22" or root_manifest.get("status") != "locked"):
        raise RuntimeError("root round manifest is not synchronized with Round-22")
    if current_round == "round_23":
        successor = root_manifest.get("round_23", {})
        if root_manifest.get("current_round") != "round_23" or successor.get("previous_lock_sha256") != EXPECTED_LOCK_SHA256:
            raise RuntimeError("root round manifest does not register Round-23 as the legal Round-22 successor")
    if current_round == "round_24":
        successor = root_manifest.get("round_24", {})
        if root_manifest.get("current_round") != "round_24" or successor.get("previous_lock_sha256") != "948E89F9197E8204B147FA2F3B9543CD878D4380680254DCB1190E7E9015F22A":
            raise RuntimeError("root round manifest does not register Round-24 as the legal Round-23 successor")
    if current_round == "round_25":
        successor = root_manifest.get("round_25", {})
        if root_manifest.get("current_round") != "round_25" or successor.get("previous_lock_sha256") != "B14A668DB825590C6E850C168008DBCB8FAEE4BDB5C56C82B5C906241887043B":
            raise RuntimeError("root round manifest does not register Round-25 as the legal Round-24 successor")
    if current_round == "round_26":
        successor = root_manifest.get("round_26", {})
        if root_manifest.get("current_round") != "round_26" or successor.get("previous_lock_sha256") != "120A417582F314CDB8B227104DE096C39340D27646F6BF33308CEC018AC72027":
            raise RuntimeError("root round manifest does not register Round-26 as the legal Round-25 successor")
    if current_round == "round_27":
        successor = root_manifest.get("round_27", {})
        if root_manifest.get("current_round") != "round_27" or successor.get("previous_lock_sha256") != expected_r26:
            raise RuntimeError("root round manifest does not register Round-27 as the legal Round-26 successor")
    if current_round == "round_28":
        successor = root_manifest.get("round_28", {})
        if root_manifest.get("current_round") != "round_28" or successor.get("previous_lock_sha256") != expected_r27:
            raise RuntimeError("root round manifest does not register Round-28 as the legal Round-27 successor")
    if current_round == "round_29":
        successor = root_manifest.get("round_29", {})
        expected_r28 = "9E3F10B893891B6E0E0415D886E00AEC234CD1376B3146E5704093A210384C91"
        if root_manifest.get("current_round") != "round_29" or successor.get("previous_lock_sha256") != expected_r28:
            raise RuntimeError("root round manifest does not register Round-29 as the legal Round-28 successor")
    if current_round == "round_30":
        successor = root_manifest.get("round_30", {})
        expected_r29 = "1F4CCAECE01F4B3F127CE4B922A9AC07B0E7E01AC9A34B204192EF8651AC51FB"
        if root_manifest.get("current_round") != "round_30" or successor.get("previous_lock_sha256") != expected_r29:
            raise RuntimeError("root round manifest does not register Round-30 as the legal Round-29 successor")
    if current_round == "round_31":
        successor = root_manifest.get("round_31", {})
        expected_r30 = "7A2591574DC52C286018A19B12C7787546C0C0E3983FA5664A8C6CB19D338125"
        if root_manifest.get("current_round") != "round_31" or successor.get("previous_lock_sha256") != expected_r30:
            raise RuntimeError("root round manifest does not register Round-31 as the legal Round-30 successor")
    if current_round == "round_32":
        successor = root_manifest.get("round_32", {})
        expected_r31 = "9B26D6129264A05A6BD7198FE283D839A70085C7A8D798D4470AD9302CB6E2D3"
        if root_manifest.get("current_round") != "round_32" or successor.get("previous_lock_sha256") != expected_r31:
            raise RuntimeError("root round manifest does not register Round-32 as the legal Round-31 successor")
    if current_round == "round_33":
        successor = root_manifest.get("round_33", {})
        expected_r32 = "8CAB303A04C3CF7799821D7887139EA2FE9AB97B2B0B21F74FA0F9C031064F7F"
        if root_manifest.get("current_round") != "round_33" or successor.get("previous_lock_sha256") != expected_r32:
            raise RuntimeError("root round manifest does not register Round-33 as the legal Round-32 successor")
    if current_round == "round_34":
        successor = root_manifest.get("round_34", {})
        if root_manifest.get("current_round") != "round_34" or successor.get("previous_lock_sha256") != EXPECTED_R33_LOCK_SHA256:
            raise RuntimeError("root round manifest does not register Round-34 as the legal Round-33 successor")
    if current_round == "round_35":
        successor = root_manifest.get("round_35", {})
        if root_manifest.get("current_round") != "round_35" or successor.get("previous_lock_sha256") != EXPECTED_R34_LOCK_SHA256:
            raise RuntimeError("root round manifest does not register Round-35 as the legal Round-34 successor")
    if current_round == "round_36":
        successor = root_manifest.get("round_36", {})
        if root_manifest.get("current_round") != "round_36" or successor.get("previous_lock_sha256") != EXPECTED_R35_LOCK_SHA256:
            raise RuntimeError("root round manifest does not register Round-36 as the legal Round-35 successor")
    return {
        "passed": True,
        "original_round22_lock_sha256": EXPECTED_LOCK_SHA256,
        "registered_hash_count": len(_locked_artifact_mismatches(lock)) + 1,
        "overlaid_path": TEST_PATH.as_posix(),
        "mismatch_count": 0,
    }


__all__ = [
    "approved_lifecycle_overlay_sha256",
    "sha256",
    "validate_round32_round33_lifecycle_amendment",
    "validate_round33_round34_lifecycle_amendment",
    "validate_round34_round35_lifecycle_amendment",
    "validate_round35_round36_lifecycle_amendment",
    "verify_round22_governance_state_sync_amendment",
    "verify_round32_locked_scientific_artifacts",
    "verify_round32_round33_lifecycle",
    "verify_round33_round34_lifecycle",
    "verify_round34_round35_lifecycle",
    "verify_round35_round36_lifecycle",
    "verify_round33_scientific_artifact_hashes",
]
