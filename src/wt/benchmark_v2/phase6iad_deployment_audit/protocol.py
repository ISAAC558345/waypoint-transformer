"""Phase-6I-AD deployment audit. Thresholds are not invented here."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ROOT
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import POLICY_FILE
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import TRAIN_POOL
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT

OUTPUT = ROOT / "results/phase6iad_deployment_audit"
REGRESSION_SCENARIO = 322002
REPEAT_SCENARIO = 322000
REPEAT_COUNT = 3
REPEAT_STOP = 15
FULL_DECISION_TIMES = (0, 15, 30, 45)
CANDIDATE_COUNT = 13327
ENVIRONMENT_SHA256 = "dbbf106ba7aadf1ee06e9721dd6b1018a10579fb23e34cf5bc791566b1e907fe"
SCHEME_C_SHA256 = "f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864"
EPOCH6_SHA256 = "7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde"
MAPPO_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"
ADAPTER_CONFIG_SHA256 = "3535008ed197f28ed3620a25493a9cc17fbb1fca9742d49a3d58957883319a63"
FROZEN_CODE_SHA256 = {
    "src/wt/benchmark_v2/phase6f_system_bringup/episode.py": "33a3fc4b9eb47373ca639c6709186789fe78d2ef9393c0b7bf97248495b99e87",
    "src/wt/benchmark_v2/phase6iaa_formal_planner/planner.py": "2c79b1eedeb95b756023c9a007315fb5063be2c86811c6beabd906d7e2425c6d",
    "src/wt/benchmark_v2/phase6iaa_formal_planner/protocol.py": "a97cf65c13b89254cf444d90356ca1c7c28c8ee3f2a2c78311ca4dd7b7d88541",
    "src/wt/benchmark_v2/phase6iy_candidate_adapter/adapter.py": "f9ab5e5545fee59d6d278149ce4237958eef22090aae980069e1d58ec6e64c4c",
    "src/wt/benchmark_v2/phase6iy_candidate_adapter/protocol.py": "87cc412cffbaeb71acafdbab6bc4006cd163e545838cbf758621f67866151a69",
}
INTERFACE_TESTS = (
    "tests.test_phase6f_system_bringup",
    "tests.test_phase6iw_inference_adapter",
    "tests.test_phase6iy_candidate_adapter",
    "tests.test_phase6iaa_formal_planner",
    "tests.test_phase6iab_planner_stability",
    "tests.test_phase6iac_independent_holdout",
    "tests.test_phase6iad_deployment_audit",
)


def assert_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    forbidden = (
        EPOCH6_CKPT.resolve(),
        SCHEME_C_CKPT.resolve(),
        primary_checkpoint().resolve(),
        POLICY_FILE.resolve(),
    )
    if resolved in forbidden:
        raise RuntimeError(f"phase-6iad refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6iad refused a reserved-test path: {resolved}")
    return resolved


def assert_scenario(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_TEST:
        raise RuntimeError(f"phase-6iad refused reserved test scenario {value}")
    if value not in (REGRESSION_SCENARIO, REPEAT_SCENARIO):
        raise RuntimeError(f"phase-6iad refuses scenario {value}")
    if value not in TRAIN_POOL:
        raise RuntimeError(f"phase-6iad regression scenario {value} is outside the train pool")


def assign_gate(*, observed_faults: int, requirements_specified: bool) -> str:
    """Ready only when every named requirement exists and the live checks pass."""
    if int(observed_faults) > 0:
        return "DEPLOYMENT_AUDIT_BLOCKED"
    if not bool(requirements_specified):
        return "DEPLOYMENT_REQUIREMENTS_INCOMPLETE"
    return "DEPLOYMENT_READY"


def status_flags(*, gate: str, planner_ready: bool) -> dict[str, bool]:
    return {
        "planner_ready": bool(planner_ready) and gate != "DEPLOYMENT_AUDIT_BLOCKED",
        "deployment_verified": gate == "DEPLOYMENT_READY",
    }
