"""Phase-6I-AC rules. Thresholds are fixed before any holdout episode."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ROOT
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import POLICY_FILE
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT

OUTPUT = ROOT / "results/phase6iac_independent_holdout"
PROTOCOL_FILE = OUTPUT / "protocol_frozen.json"
SCENARIOS = tuple(range(322064, 322080))
SMOKE = (322064, 322065)
BIT_TIE = 1.0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
BOOTSTRAP_LOW = 2.5
BOOTSTRAP_HIGH = 97.5
TORCH_SEED_OFFSET = 1_000_000
DECISION_TIMES = (0, 15, 30, 45)
HORIZON = 48
HOLD = 15
SCHEME_C_SHA256 = "f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864"
EPOCH6_SHA256 = "7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde"
MAPPO_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"
ADAPTER_CONFIG_SHA256 = "3535008ed197f28ed3620a25493a9cc17fbb1fca9742d49a3d58957883319a63"
ENVIRONMENT_SHA256 = "dbbf106ba7aadf1ee06e9721dd6b1018a10579fb23e34cf5bc791566b1e907fe"
CODE_FILES = (
    "src/wt/benchmark_v2/phase6f_system_bringup/episode.py",
    "src/wt/benchmark_v2/phase6iaa_formal_planner/planner.py",
    "src/wt/benchmark_v2/phase6iaa_formal_planner/protocol.py",
    "src/wt/benchmark_v2/phase6iy_candidate_adapter/adapter.py",
    "src/wt/benchmark_v2/phase6iy_candidate_adapter/protocol.py",
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
        raise RuntimeError(f"phase-6iac refused to write {resolved}")
    if resolved == PROTOCOL_FILE.resolve() and resolved.name != "protocol_frozen.json":
        raise RuntimeError(f"phase-6iac refused to write {resolved}")
    return resolved


def assert_scenario(scenario_id: int) -> None:
    value = int(scenario_id)
    if value not in SCENARIOS or value not in RESERVED_TEST:
        raise RuntimeError(f"phase-6iac refuses scenario {value}")


def assign_gate(*, failures: int, complete_pairs: int, ci_low: float, ci_high: float) -> str:
    """The interval rule is fixed. A crossing interval is not a gain."""
    if int(failures) > 0 or int(complete_pairs) != len(SCENARIOS):
        return "INDEPENDENT_TEST_AUDIT_BLOCKED"
    if float(ci_low) > 0.0:
        return "INDEPENDENT_GAIN_SUPPORTED"
    if float(ci_high) < 0.0:
        return "INDEPENDENT_GAIN_NEGATIVE"
    return "INDEPENDENT_GAIN_INCONCLUSIVE"


__all__ = [
    "ADAPTER_CONFIG_SHA256",
    "BIT_TIE",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_HIGH",
    "BOOTSTRAP_LOW",
    "BOOTSTRAP_SEED",
    "CODE_FILES",
    "DECISION_TIMES",
    "ENVIRONMENT_SHA256",
    "EPOCH6_CKPT",
    "EPOCH6_SHA256",
    "HOLD",
    "HORIZON",
    "MAPPO_SHA256",
    "OUTPUT",
    "PROTOCOL_FILE",
    "SCENARIOS",
    "SCHEME_C_CKPT",
    "SCHEME_C_SHA256",
    "SMOKE",
    "TORCH_SEED_OFFSET",
    "assert_output_path",
    "assert_scenario",
    "assign_gate",
]
