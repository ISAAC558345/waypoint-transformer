"""Stage D1 rules for the t=30 urgency-label pilot. Thresholds are fixed before rollout."""
from __future__ import annotations

import json

from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import DEV_TEST_RESERVED, TRAIN_POOL, VALIDATION_POOL, assert_allowed
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.protocol import REGISTRY, ROOT, pilot_scenarios

OUTPUT = ROOT / "results" / "phase6i_urgency_highlevel_t30_pilot"
D0_OUTPUT = ROOT / "results" / "phase6i_urgency_highlevel_pilot"
INSTRUCTION_COMMIT = "fa32cbbb928add8ae3b4e1263325b22c780a0aae"
TIME_STEP = 30
PILOT_SEED_T30 = 602
SCENARIO_COUNT = 12
CANDIDATES_PER_STATE = 128
MAX_ROLLOUTS = SCENARIO_COUNT * CANDIDATES_PER_STATE
MAX_STATE_SECONDS = 900.0
MAX_ESTIMATED_SECONDS = 30.0 * 60.0
SPEARMAN_BELOW = 0.95
REVERSAL_ABOVE = 0.05
MIN_STATES_ABOVE = 8
# Signed regret is urgent-top payload minus payload-top payload. This cutoff is not retuned after labels exist.
MATERIAL_NEGATIVE_REGRET_BITS = -100_000.0
EXPECTED_SCENARIOS = tuple(range(322004, 322016))
GATE_SUPPORTED = "T30_URGENCY_SIGNAL_SUPPORTED"
GATE_INCONCLUSIVE = "T30_URGENCY_SIGNAL_INCONCLUSIVE"
GATE_BLOCKED = "T30_URGENCY_SIGNAL_BLOCKED"
LAMBDA_H = None


def t30_scenarios() -> tuple[int, ...]:
    """Train ids after the D0 four, in scenario-id order. The registry guard stays in front."""
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    train = tuple(int(item) for item in payload["highlevel_v2_train"])
    validation = tuple(int(item) for item in payload["highlevel_v2_validation"])
    reserved = tuple(int(item) for item in payload["highlevel_v2_dev_test_reserved"])
    if train != tuple(sorted(train)):
        raise RuntimeError("registry train ids are not sorted by scenario id")
    if train != tuple(int(item) for item in TRAIN_POOL):
        raise RuntimeError("registry train pool does not match TRAIN_POOL")
    if validation != tuple(int(item) for item in VALIDATION_POOL):
        raise RuntimeError("registry validation pool does not match VALIDATION_POOL")
    if reserved != tuple(int(item) for item in DEV_TEST_RESERVED):
        raise RuntimeError("registry reserved pool does not match the reserved split")
    excluded = set(pilot_scenarios())
    if excluded & set(validation) or excluded & set(reserved):
        raise RuntimeError("D0 scenarios overlap a held-out split")
    chosen = tuple(item for item in train if item not in excluded)[:SCENARIO_COUNT]
    if len(chosen) != SCENARIO_COUNT:
        raise RuntimeError("registry does not contain 12 train scenarios after the D0 pilot")
    if any(item in validation or item in reserved or item in excluded for item in chosen):
        raise RuntimeError("t=30 scenarios overlap D0, validation, or reserved test")
    for item in chosen:
        assert_allowed(int(item), split="train")
    if chosen != EXPECTED_SCENARIOS:
        raise RuntimeError(f"registry continuation is {chosen}, not the expected train continuation")
    return chosen


def assert_t30_scenario(scenario_id: int) -> int:
    value = int(scenario_id)
    if value not in t30_scenarios():
        raise RuntimeError(f"stage D1 refuses scenario {value}")
    return value


def choose_gate(
    *,
    mechanical_ok: bool,
    budget_ok: bool,
    coverage_complete: bool,
    mean_spearman: float | None,
    mean_reversal: float | None,
    states_above: int,
) -> str:
    """Exploratory t=30 gate. A partial average signal without state coverage stays inconclusive."""
    if not mechanical_ok or not budget_ok or not coverage_complete:
        return GATE_BLOCKED
    average_signal = (mean_spearman is not None and float(mean_spearman) < float(SPEARMAN_BELOW)) or (
        mean_reversal is not None and float(mean_reversal) > float(REVERSAL_ABOVE)
    )
    if average_signal and int(states_above) >= int(MIN_STATES_ABOVE):
        return GATE_SUPPORTED
    return GATE_INCONCLUSIVE


def recommendation(gate: str, mean_regret_bits: float | None) -> str:
    if gate == GATE_BLOCKED:
        return "校验或预算未通过，停止 t=30 标签扩展，保留当前正式模型。"
    if gate == GATE_INCONCLUSIVE:
        return "t=30 排序信号仍不明确，停止高层紧急度标签扩展，不构建完整数据集，保留当前正式模型。"
    if mean_regret_bits is not None and float(mean_regret_bits) < float(MATERIAL_NEGATIVE_REGRET_BITS):
        return "排序信号达到本次探索阈值，同时 top-1 payload regret 为负。下一步只设计 utility 权衡和 lambda_H 规则，不训练 HGNN。"
    return "排序信号达到本次探索阈值，top-1 payload 损失未超过预注册幅度。utility 权衡规则尚未写定，本阶段不构建完整数据集，也不训练 HGNN。"
