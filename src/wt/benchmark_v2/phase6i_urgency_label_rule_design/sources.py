"""Read frozen D0/D1/D5/D6 tables. This module does not score candidates."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import assert_allowed

from .protocol import CANDIDATES_PER_STATE, PRIMARY_SOURCE, PRIMARY_TIMES, ROOT


@dataclass(frozen=True)
class SourceSpec:
    name: str
    result_dir: str
    report: str
    label_family: str
    arrival_semantics: str
    state_path: str
    horizon_fallback: int | None
    horizon_source: str


SPECS = (
    SourceSpec(
        name="D0",
        result_dir="results/phase6i_urgency_highlevel_pilot",
        report="docs/research/HTW_UrgencyAware_HighLevel_Label_Pilot_Report_CN.md",
        label_family="zero_arrival",
        arrival_semantics="label rollout disables arrivals on the scratch copy",
        state_path="MAX_RATE_INDEPENDENT dataset replay, frozen MAPPO-V2 seed 0",
        horizon_fallback=15,
        horizon_source="schema_j15_hold",
    ),
    SourceSpec(
        name="D1",
        result_dir="results/phase6i_urgency_highlevel_t30_pilot",
        report="docs/research/HTW_UrgencyAware_HighLevel_T30_Pilot_Report_CN.md",
        label_family="zero_arrival",
        arrival_semantics="label rollout disables arrivals on the scratch copy",
        state_path="MAX_RATE_INDEPENDENT dataset replay, frozen MAPPO-V2 seed 0",
        horizon_fallback=15,
        horizon_source="schema_j15_hold",
    ),
    SourceSpec(
        name="D5",
        result_dir="results/phase6i_urgency_arrival_aware_labels",
        report="docs/research/HTW_UrgencyAware_ArrivalAware_Label_Report_CN.md",
        label_family="real_arrival",
        arrival_semantics="real arrivals_at kept; zero-arrival J15 is a legacy column and is not the rule target",
        state_path="MAX_RATE_INDEPENDENT dataset replay, frozen MAPPO-V2 seed 0",
        horizon_fallback=15,
        horizon_source="schema_horizon",
    ),
    SourceSpec(
        name="D6",
        result_dir="results/phase6i_urgency_arrival_aware_expansion",
        report="docs/research/HTW_UrgencyAware_ArrivalAware_Expansion_Report_CN.md",
        label_family="real_arrival",
        arrival_semantics="real arrivals_at kept; zero-arrival arm is a legacy comparison and is not the rule target",
        state_path="MAX_RATE_INDEPENDENT dataset replay, frozen MAPPO-V2 seed 0",
        horizon_fallback=None,
        horizon_source="labels_csv",
    ),
)


@dataclass
class StateBatch:
    source: str
    scenario_id: int
    time_step: int
    horizon: int
    horizon_source: str
    label_family: str
    arrival_semantics: str
    state_path: str
    rows: list[dict]
    field_gaps: list[str] = field(default_factory=list)
    pool_count: int | None = None
    pool_count_source: str = ""
    state_hash: str = ""
    pool_hash: str = ""
    subset_hash: str = ""

    @property
    def primary_rule_state(self) -> bool:
        return self.source == PRIMARY_SOURCE and int(self.time_step) in PRIMARY_TIMES


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(row: dict, key: str, gaps: list[str]) -> str:
    if key not in row or row[key] is None or str(row[key]).strip() == "":
        gaps.append(key)
        return ""
    return str(row[key])


def _float_field(row: dict, key: str, gaps: list[str]) -> float | None:
    text = _require(row, key, gaps)
    if text == "":
        return None
    return float(text)


def _index_summary(summary: dict, *, time_in_state: bool) -> dict:
    indexed = {}
    for state in summary.get("states") or []:
        scenario_id = int(state["scenario_id"])
        if time_in_state:
            indexed[(scenario_id, int(state["time_step"]))] = state
        else:
            indexed[scenario_id] = state
    return indexed


def _summary_lookup(indexed: dict, scenario_id: int, time_step: int, *, time_in_state: bool) -> dict | None:
    if time_in_state:
        return indexed.get((int(scenario_id), int(time_step)))
    return indexed.get(int(scenario_id))


def _candidate(row: dict, assignment: str, payload: float, urgent: float) -> dict:
    return {
        "candidate_id": int(row["candidate_id"]),
        "assignment": assignment,
        "payload_bits": float(payload),
        "urgent_bits": float(urgent),
    }


def load_states() -> tuple[list[StateBatch], list[dict]]:
    """Return normalized states and the source manifest. Missing rule fields stay empty."""
    batches: list[StateBatch] = []
    manifest: list[dict] = []
    for spec in SPECS:
        result = ROOT / spec.result_dir
        labels_path = result / "labels.csv"
        selected_path = result / "selected_candidates.csv"
        summary_path = result / "summary.json"
        schema_path = result / "schema.json"
        report_path = ROOT / spec.report
        for path in (labels_path, selected_path, summary_path, schema_path, report_path):
            if not path.is_file():
                raise RuntimeError(f"missing frozen source {path}")
        labels = _read_csv(labels_path)
        selected = _read_csv(selected_path)
        summary = _read_json(summary_path)
        schema = _read_json(schema_path)
        manifest.append(
            {
                "source": spec.name,
                "labels": str(labels_path.relative_to(ROOT)).replace("\\", "/"),
                "selected_candidates": str(selected_path.relative_to(ROOT)).replace("\\", "/"),
                "summary": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
                "schema": str(schema_path.relative_to(ROOT)).replace("\\", "/"),
                "report": spec.report,
                "summary_gate": summary.get("gate"),
                "label_family": spec.label_family,
                "arrival_semantics": spec.arrival_semantics,
                "state_path": spec.state_path,
            }
        )
        selected_assignment = {}
        for row in selected:
            key = (int(row["scenario_id"]), int(row["time_step"]), int(row["candidate_id"]))
            selected_assignment[key] = str(row.get("assignment", ""))
        time_in_state = spec.name == "D6"
        indexed = _index_summary(summary, time_in_state=time_in_state)
        grouped: dict[tuple[int, int], list[dict]] = {}
        for row in labels:
            grouped.setdefault((int(row["scenario_id"]), int(row["time_step"])), []).append(row)
        for (scenario_id, time_step), rows in sorted(grouped.items()):
            assert_allowed(scenario_id, split="train")
            gaps: list[str] = []
            if spec.label_family == "real_arrival":
                payload_key = "real_payload_bits"
                urgent_key = "real_urgent_service_bits"
            else:
                payload_key = "payload_bits"
                urgent_key = "urgent_service_bits"
                if "real_payload_bits" not in rows[0]:
                    gaps.append("real_payload_bits")
                if "real_urgent_service_bits" not in rows[0]:
                    gaps.append("real_urgent_service_bits")
            horizon_values = []
            if "horizon" in rows[0]:
                horizon_values = sorted({int(float(row["horizon"])) for row in rows})
                horizon = int(horizon_values[0])
                horizon_source = "labels_csv"
            else:
                gaps.append("horizon")
                if spec.horizon_fallback is None:
                    raise RuntimeError(f"{spec.name} has no horizon")
                horizon = int(spec.horizon_fallback)
                horizon_source = spec.horizon_source
            if len(horizon_values) > 1:
                raise RuntimeError(f"{spec.name} scenario {scenario_id} t={time_step} mixes horizons")
            normalized = []
            hashes = set()
            subsets = set()
            pools = set()
            for row in rows:
                payload = _float_field(row, payload_key, gaps)
                urgent = _float_field(row, urgent_key, gaps)
                key = (scenario_id, time_step, int(row["candidate_id"]))
                assignment = str(row.get("assignment") or selected_assignment.get(key) or "")
                if assignment == "":
                    gaps.append("assignment")
                elif key in selected_assignment and selected_assignment[key] not in ("", assignment):
                    raise RuntimeError(f"assignment mismatch for {spec.name} {key}")
                if payload is None or urgent is None:
                    continue
                normalized.append(_candidate(row, assignment, payload, urgent))
                if str(row.get("state_hash") or "").strip():
                    hashes.add(str(row["state_hash"]))
                if str(row.get("subset_hash") or "").strip():
                    subsets.add(str(row["subset_hash"]))
                if str(row.get("pool_hash") or "").strip():
                    pools.add(str(row["pool_hash"]))
            if len({int(item["candidate_id"]) for item in normalized}) != len(normalized):
                raise RuntimeError(f"duplicate candidate id in {spec.name} {scenario_id} t={time_step}")
            if len(normalized) != int(CANDIDATES_PER_STATE):
                gaps.append(f"candidate_count:{len(normalized)}")
            info = _summary_lookup(indexed, scenario_id, time_step, time_in_state=time_in_state)
            pool_count = None
            pool_count_source = ""
            if info is not None and "pool_count" in info:
                pool_count = int(info["pool_count"])
                pool_count_source = "summary.states.pool_count"
            else:
                gaps.append("summary.states.pool_count")
            if info is not None:
                if not hashes and info.get("state_hash"):
                    hashes.add(str(info["state_hash"]))
                if not subsets and info.get("subset_hash"):
                    subsets.add(str(info["subset_hash"]))
                if not pools and info.get("pool_hash"):
                    pools.add(str(info["pool_hash"]))
            if len(hashes) > 1 or len(subsets) > 1 or len(pools) > 1:
                raise RuntimeError(f"mixed hashes in {spec.name} {scenario_id} t={time_step}")
            if not hashes:
                gaps.append("state_hash")
            schema_fields = set((schema.get("fields") or {}).get("labels.csv") or [])
            if spec.label_family == "zero_arrival" and "real_payload_bits" not in schema_fields:
                pass
            batches.append(
                StateBatch(
                    source=spec.name,
                    scenario_id=int(scenario_id),
                    time_step=int(time_step),
                    horizon=int(horizon),
                    horizon_source=horizon_source,
                    label_family=spec.label_family,
                    arrival_semantics=spec.arrival_semantics,
                    state_path=spec.state_path,
                    rows=normalized,
                    field_gaps=sorted(set(gaps)),
                    pool_count=pool_count,
                    pool_count_source=pool_count_source,
                    state_hash=next(iter(hashes), ""),
                    pool_hash=next(iter(pools), ""),
                    subset_hash=next(iter(subsets), ""),
                )
            )
    return batches, manifest
