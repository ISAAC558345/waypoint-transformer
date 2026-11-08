"""Stdlib completion rules for one formal episode. This module does not import torch."""
from __future__ import annotations

from pathlib import Path


TOLERANCE_BITS = 1.0
IDENTITY_FIELDS = ("scenario_id", "policy")


class OutputRefusal(RuntimeError):
    """The requested directory already holds files and must not be overwritten."""


def prepare_output_directory(output: Path) -> str:
    """Accept a missing or empty directory. A nonempty directory is refused before any write."""
    path = Path(output)
    if path.exists():
        if any(path.iterdir()):
            raise OutputRefusal("refusing to overwrite an existing output")
        path.mkdir(parents=True, exist_ok=True)
        return "empty_reused"
    path.mkdir(parents=True, exist_ok=True)
    return "created"


def _field(reference: dict, current: dict, key: str) -> dict:
    return {
        "reference": reference.get(key),
        "current": current.get(key),
        "match": reference.get(key) == current.get(key),
    }


def _internal_ok(current: dict) -> bool:
    payload_flag = current.get("payload_matches_served")
    step_flag = current.get("step_served_matches_episode")
    return (payload_flag is True or payload_flag == 1) and step_flag is True


def _identity(reference: dict, current: dict) -> tuple[dict, bool]:
    """Both sides must carry scenario_id and policy. Missing values are not equal."""
    rows = {}
    ok = True
    for key in IDENTITY_FIELDS:
        present = key in reference and key in current and reference[key] is not None and current[key] is not None
        match = bool(present and reference[key] == current[key])
        rows[key] = {
            "reference": reference.get(key),
            "current": current.get(key),
            "present": present,
            "match": match,
        }
        ok = ok and match
    return rows, ok


def compare_episodes(reference: dict, current: dict) -> dict:
    """Match only when identity, saved fields, 1-bit tolerances, and both consistency flags agree."""
    reference_payload = float(reference["payload_bits"])
    payload_gap = abs(float(current["payload_bits"]) - reference_payload)
    served_gap = abs(float(current["served_bits"]) - float(reference["served_bits"]))
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
    numeric_ok = payload_gap <= TOLERANCE_BITS and served_gap <= TOLERANCE_BITS and not step_gaps
    identity, identity_ok = _identity(reference, current)
    fields = {
        name: _field(reference, current, name)
        for name in ("complete", "terminal_time", "process_seed", "reset_seed", "decision_times", "scheme_c_calls")
    }
    fields_ok = all(item["match"] for item in fields.values())
    internal_ok = _internal_ok(current)
    return {
        "reference_payload_bits": reference_payload,
        "reference_payload_source": "same R1 reference episode payload_bits",
        "tolerance_bits": TOLERANCE_BITS,
        "payload_abs_diff_bits": payload_gap,
        "served_abs_diff_bits": served_gap,
        "published_payload_abs_diff_bits": payload_gap,
        "published_payload_role": "same R1 reference payload, not an independent published constant",
        "identity": identity,
        "step_count": {"reference": len(ref_steps), "current": len(cur_steps)},
        "step_served_mismatches": step_gaps,
        "fields": fields,
        "current_payload_matches_served": current.get("payload_matches_served"),
        "current_step_served_matches_episode": current.get("step_served_matches_episode"),
        "internal_consistency_ok": internal_ok,
        "status": "matched" if numeric_ok and identity_ok and fields_ok and internal_ok else "different",
    }


def conclude(record: dict) -> dict:
    """Success requires a finished matched run, the registered weights, and an unchanged official policy."""
    reasons = []
    if record.get("error") is not None:
        reasons.append("run_error")
    if record.get("complete") is not True:
        reasons.append("incomplete")
    if record.get("regression_status") != "matched":
        reasons.append("regression_not_matched")
    hashes = record.get("model_hash_matches_contract") or {}
    if not hashes or not all(bool(value) for value in hashes.values()):
        reasons.append("model_identity")
    if record.get("official_policy_unchanged") is not True:
        reasons.append("official_policy_changed")
    success = not reasons
    return {"success": success, "exit_code": 0 if success else 1, "reasons": reasons}
