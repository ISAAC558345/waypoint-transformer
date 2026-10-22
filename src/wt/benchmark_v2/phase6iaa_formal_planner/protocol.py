"""Active high-level policy. Phase-6F's historical k=5 constant stays untouched."""
from __future__ import annotations

import json
from pathlib import Path

from wt.benchmark_v2.phase6f_system_bringup.protocol import CANDIDATE_K
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ROOT
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import GENERATOR, HOLD, SELECTED_K
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, TRAIN_POOL, VALIDATION_POOL
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT

OUTPUT = ROOT / "results/phase6iaa_formal_planner"
POLICY_FILE = OUTPUT / "active_policy.json"
HISTORICAL_FORMAL_K = int(CANDIDATE_K)
ACTIVE_CANDIDATE_K = int(SELECTED_K)
POLICIES = ("scheme_c", "g1_rate")
# Name used by an explicit write. read_policy does not apply it when the file is missing.
DEFAULT_POLICY = "scheme_c"
SMOKE_SCENARIOS = (322000, 322001)
EPISODE_SCENARIOS = (322002, 322048)
OPEN_POOL = TRAIN_POOL + VALIDATION_POOL
ADAPTER_CONFIG_SHA256 = "3535008ed197f28ed3620a25493a9cc17fbb1fca9742d49a3d58957883319a63"
SCHEME_C_SHA256 = "f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864"
EPOCH6_SHA256 = "7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde"
MAPPO_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"


def assert_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    forbidden = (EPOCH6_CKPT.resolve(), SCHEME_C_CKPT.resolve(), primary_checkpoint().resolve())
    if resolved in forbidden:
        raise RuntimeError(f"phase-6iaa refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6iaa refused a reserved-test path: {resolved}")
    return resolved


def assert_scenario(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_TEST:
        raise RuntimeError(f"phase-6iaa refused reserved test scenario {value}")
    if value not in OPEN_POOL:
        raise RuntimeError(f"phase-6iaa refuses scenario {value}")


def assert_episode_scenario(scenario_id: int, *, allow_holdout: bool = False) -> None:
    """The holdout stays outside train and validation unless this episode explicitly allows it."""
    value = int(scenario_id)
    if bool(allow_holdout) and value in RESERVED_TEST:
        return
    assert_scenario(value)


def read_policy(path: Path | None = None) -> str:
    """Read an explicit policy file. A missing or illegal file raises PolicyError."""
    from wt.benchmark_v2.phase6iaa_formal_planner.faults import PolicyError

    file = POLICY_FILE if path is None else Path(path)
    if not file.is_file():
        raise PolicyError(f"policy file is missing: {file}")
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyError(f"policy file is not valid JSON: {file}") from exc
    if not isinstance(payload, dict):
        raise PolicyError(f"policy file must be a JSON object: {file}")
    try:
        policy = payload["high_level_policy"]
        generator = payload["generator"]
        k = int(payload["k"])
        hold = int(payload["hold"])
        historical = int(payload["historical_formal_k"]) if "historical_formal_k" in payload else int(HISTORICAL_FORMAL_K)
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyError(f"policy file fields are illegal: {file}") from exc
    if policy not in POLICIES:
        raise PolicyError(f"illegal high_level_policy: {policy}")
    if generator != GENERATOR or k != int(ACTIVE_CANDIDATE_K) or hold != int(HOLD) or historical != int(HISTORICAL_FORMAL_K):
        raise PolicyError(f"policy file is not G1_RATE k=6 hold 15: {file}")
    if GENERATOR != "G1_RATE" or ACTIVE_CANDIDATE_K != 6 or HISTORICAL_FORMAL_K != 5:
        raise PolicyError("candidate contract drifted")
    return str(policy)


def write_policy(policy: str, path: Path | None = None) -> Path:
    if policy not in POLICIES:
        raise RuntimeError(f"unknown high_level_policy {policy}")
    file = assert_output_path(POLICY_FILE if path is None else Path(path))
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        json.dumps(
            {
                "high_level_policy": policy,
                "generator": GENERATOR,
                "k": ACTIVE_CANDIDATE_K,
                "historical_formal_k": HISTORICAL_FORMAL_K,
                "hold": int(HOLD),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return file


def assign_gate(*, interface_ok: bool, smoke_ok: bool, episode_ok: bool, scheme_c_called: bool, failures: int) -> str:
    """Ready only when scheme C is actually called and every check passes."""
    if int(failures) > 0 or not bool(interface_ok):
        return "PLANNER_INTEGRATION_BLOCKED"
    if bool(scheme_c_called) and bool(smoke_ok) and bool(episode_ok):
        return "PLANNER_INTEGRATION_READY"
    return "PLANNER_INTEGRATION_INCONCLUSIVE"


def status_flags(gate: str) -> dict[str, bool]:
    return {
        "planner_ready": gate == "PLANNER_INTEGRATION_READY",
        "deployment_verified": False,
    }


__all__ = [
    "ACTIVE_CANDIDATE_K",
    "ADAPTER_CONFIG_SHA256",
    "DEFAULT_POLICY",
    "EPOCH6_CKPT",
    "EPOCH6_SHA256",
    "EPISODE_SCENARIOS",
    "HISTORICAL_FORMAL_K",
    "HOLD",
    "MAPPO_SHA256",
    "OPEN_POOL",
    "OUTPUT",
    "POLICY_FILE",
    "POLICIES",
    "SCHEME_C_CKPT",
    "SCHEME_C_SHA256",
    "SMOKE_SCENARIOS",
    "assert_episode_scenario",
    "assert_output_path",
    "assert_scenario",
    "assign_gate",
    "read_policy",
    "status_flags",
    "write_policy",
]
