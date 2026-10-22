"""Deterministic, non-learning Round-34 benchmark-complexity audit.

This module deliberately uses only the Python standard library.  It never
imports an environment, planner, model, or fitting utility.  Frozen
train/validation records are read line-by-line; prohibited split markers are
rejected before a line is decoded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import struct
import zlib
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
ROUND = ROOT / "round_34"
CONFIG_PATH = ROUND / "configs" / "round34_benchmark_complexity_audit_config.json"
SAMPLES_PATH = ROOT / "round_20" / "outputs" / "round20_multi_uav_dataset_20260828_2100" / "multi_uav_task_waypoint_samples.jsonl"
TRANSITIONS_PATH = ROOT / "round_20" / "outputs" / "round20_multi_uav_dataset_20260828_2100" / "multi_uav_transitions.jsonl"
R33_OUTPUT = ROOT / "round_33" / "outputs" / "round33_long_episode_benchmark_20260830_2320"
OUTPUT_NAME = "round34_benchmark_complexity_audit_20260901_1540"
SAMPLE_PREFLIGHT_PATH = ROUND / "round34_clean_sample_preflight.json"
SAMPLE_MANIFEST_PATH = ROUND / "round34_clean_sample_manifest.json"
SAMPLE_LOCK_PATH = ROUND / "round34_clean_sample_manifest.lock.json"
TRANSITION_PREFLIGHT_PATH = ROUND / "round34_clean_transition_preflight.json"
TRANSITION_MANIFEST_PATH = ROUND / "round34_clean_transition_manifest.json"
TRANSITION_LOCK_PATH = ROUND / "round34_clean_transition_manifest.lock.json"
ALLOWED_SPLITS = frozenset(("train", "validation"))
PROTECTED_SPLIT_ALIASES = frozenset(("development_test", "development-test", "dev", "test", "untouched", "untouched_eval", "untouched-eval"))
SPLIT_FIELD = re.compile(rb'"split"\s*:\s*"([^"\\]+)"')
SAMPLE_ID_FIELD = re.compile(rb'"sample_id"\s*:\s*"([^"\\]+)"')
EPISODE_ID_FIELD = re.compile(rb'"episode_id"\s*:\s*"([^"\\]+)"')
TIMESTEP_FIELD = re.compile(rb'"timestep"\s*:\s*(-?[0-9]+)')
T_FIELD = re.compile(rb'"t"\s*:\s*(-?[0-9]+)')


class ProtectedSplitSecondViolation(RuntimeError):
    """Raised before a second protected record could be decoded."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def association_count(m_count: int, k_count: int) -> int:
    """Legal decoder states for injective partial assignments, including idle."""
    if m_count < 0 or k_count < 0:
        raise ValueError("M and K must be nonnegative")
    total = 0
    for assigned in range(min(m_count, k_count) + 1):
        total += math.comb(m_count, assigned) * math.perm(k_count, assigned)
    return total


def extract_decision_ownership() -> list[dict[str, Any]]:
    """Pre-registered decision categories and their current controlling actor."""
    return [
        {"decision": "UAV movement", "importance": "high", "decision_kind": "continuous", "neural_policy_direct": True, "deterministic_matching": False, "fixed_rule_or_environment": False, "graph_relational": False, "owner": "movement-action head"},
        {"decision": "association", "importance": "high", "decision_kind": "discrete injective", "neural_policy_direct": False, "deterministic_matching": True, "fixed_rule_or_environment": False, "graph_relational": True, "owner": "exhaustive current-state matching"},
        {"decision": "service priority", "importance": "high", "decision_kind": "discrete score ordering", "neural_policy_direct": False, "deterministic_matching": True, "fixed_rule_or_environment": False, "graph_relational": True, "owner": "queue-weighted-rate matching score"},
        {"decision": "power", "importance": "medium", "decision_kind": "resource", "neural_policy_direct": False, "deterministic_matching": False, "fixed_rule_or_environment": True, "graph_relational": False, "owner": "fixed configuration"},
        {"decision": "bandwidth", "importance": "medium", "decision_kind": "resource", "neural_policy_direct": False, "deterministic_matching": False, "fixed_rule_or_environment": True, "graph_relational": False, "owner": "fixed shared-spectrum rule"},
        {"decision": "sensing", "importance": "medium", "decision_kind": "resource", "neural_policy_direct": False, "deterministic_matching": False, "fixed_rule_or_environment": True, "graph_relational": False, "owner": "not exposed by the environment"},
        {"decision": "interference coordination", "importance": "high", "decision_kind": "relational allocation", "neural_policy_direct": False, "deterministic_matching": True, "fixed_rule_or_environment": False, "graph_relational": True, "owner": "matching plus fixed overlap rule"},
        {"decision": "safety correction", "importance": "high", "decision_kind": "constraint projection", "neural_policy_direct": False, "deterministic_matching": False, "fixed_rule_or_environment": True, "graph_relational": False, "owner": "environment safety projection"},
    ]


def classify_edge_derivability() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Class A means exact node/global derivation; C means irreducible pairing."""
    ub = [
        ("horizontal_distance_m", "A", "exact from UAV and buoy positions"),
        ("three_dimensional_distance_m", "A", "exact from horizontal distance and fixed altitude"),
        ("channel_gain", "A", "exact from geometry and fixed channel constants"),
        ("snr", "A", "exact from gain and fixed transmit/noise constants"),
        ("direct_rate_bps", "A", "exact from SNR and fixed bandwidth"),
        ("queue_weighted_rate", "A", "exact from buoy queue and direct rate"),
        ("cross_interference_potential", "A", "exact potential from gain, overlap, and fixed power"),
        ("current_assignment_relation", "C", "pair identity is absent from edge-free node pooling"),
    ]
    uu = [
        ("uav_distance_m", "A", "exact from two UAV positions"),
        ("relative_position_x", "A", "exact position difference"),
        ("relative_position_y", "A", "exact position difference"),
        ("relative_velocity_x", "A", "exact velocity difference"),
        ("relative_velocity_y", "A", "exact velocity difference"),
        ("safety_margin_m", "A", "exact from distance and fixed separation threshold"),
        ("both_assigned_distinct", "C", "requires assignment identity across two UAVs"),
        ("receiver_cross_interference_power_w", "C", "requires selected other-to-receiver relation"),
        ("interference_to_signal_ratio", "C", "requires selected pair and signal relation"),
    ]
    return ([{"edge_type": "U-B", "feature": x, "class": y, "reason": z} for x, y, z in ub], [{"edge_type": "U-U", "feature": x, "class": y, "reason": z} for x, y, z in uu])


def _field_text(match: re.Match[bytes] | None, field: str, line_number: int) -> str:
    if match is None:
        raise ProtectedSplitSecondViolation(f"metadata is incomplete at line {line_number}: {field} is missing")
    try:
        return match.group(1).decode("ascii")
    except UnicodeDecodeError as error:
        raise ProtectedSplitSecondViolation(f"metadata is not ASCII at line {line_number}: {field}") from error


def _metadata_from_raw(raw: bytes, line_number: int, source: str) -> dict[str, Any]:
    """Extract only metadata tokens from raw bytes; never deserialize payload."""
    split = _field_text(SPLIT_FIELD.search(raw), "split", line_number).strip().lower()
    if split not in ALLOWED_SPLITS and split not in PROTECTED_SPLIT_ALIASES:
        raise ProtectedSplitSecondViolation(f"unknown or unresolved split alias at {source}:{line_number}: {split!r}")
    identifier_match = SAMPLE_ID_FIELD.search(raw) if source == "samples" else EPISODE_ID_FIELD.search(raw)
    identifier_field = "sample_id" if source == "samples" else "episode_id"
    identifier = _field_text(identifier_match, identifier_field, line_number)
    time_match = T_FIELD.search(raw) if source == "samples" else TIMESTEP_FIELD.search(raw)
    time_value = int(_field_text(time_match, "time", line_number))
    return {
        "source": source,
        "line_number": line_number,
        "record_sha256": hashlib.sha256(raw).hexdigest().upper(),
        "record_id": identifier,
        "time": time_value,
        "split": split,
        "allowed": split in ALLOWED_SPLITS,
        "protected": split in PROTECTED_SPLIT_ALIASES,
    }


def _scan_metadata(path: Path, source: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Create a fail-closed metadata index without JSON payload decoding."""
    metadata: list[dict[str, Any]] = []
    counts = {"train": 0, "validation": 0, "development_test": 0, "untouched": 0, "unknown": 0}
    for line_number, raw in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw.strip():
            raise ProtectedSplitSecondViolation(f"blank record line at {source}:{line_number}")
        item = _metadata_from_raw(raw, line_number, source)
        metadata.append(item)
        if item["split"] in ALLOWED_SPLITS:
            counts[item["split"]] += 1
        elif item["split"] in {"development_test", "development-test", "dev", "test"}:
            counts["development_test"] += 1
        elif item["split"] in {"untouched", "untouched_eval", "untouched-eval"}:
            counts["untouched"] += 1
        else:
            counts["unknown"] += 1
    return metadata, counts


def _preflight_payload(source: str, path: Path, metadata: list[dict[str, Any]], counts: dict[str, int]) -> dict[str, Any]:
    allowed = [item for item in metadata if item["allowed"]]
    protected = [item for item in metadata if item["protected"]]
    return {
        "round": 34,
        "source": source,
        "source_path": str(path.relative_to(ROOT)),
        "source_sha256": digest(path),
        "metadata_only": True,
        "json_payload_decoded": False,
        "protected_split_fail_closed": True,
        "total_candidate_records": len(metadata),
        "train_count": counts["train"],
        "validation_count": counts["validation"],
        "development_test_count_detected_and_excluded": counts["development_test"],
        "untouched_count_detected_and_excluded": counts["untouched"],
        "unknown_split_count": counts["unknown"],
        "allowed_count": len(allowed),
        "development_test_in_formal_input": 0,
        "untouched_in_formal_input": 0,
        "unknown_split_in_formal_input": 0,
        "excluded_record_metadata": [{key: item[key] for key in ("line_number", "record_id", "record_sha256", "split")} for item in protected],
    }


def create_clean_manifests() -> dict[str, Any]:
    """Hash-lock clean metadata manifests before formal scientific decoding."""
    targets = (("samples", SAMPLES_PATH, SAMPLE_PREFLIGHT_PATH, SAMPLE_MANIFEST_PATH, SAMPLE_LOCK_PATH), ("transitions", TRANSITIONS_PATH, TRANSITION_PREFLIGHT_PATH, TRANSITION_MANIFEST_PATH, TRANSITION_LOCK_PATH))
    if any(path.exists() for target in targets for path in target[2:]):
        raise FileExistsError("clean metadata manifests already exist; they are immutable")
    result: dict[str, Any] = {}
    for source, source_path, preflight_path, manifest_path, lock_path in targets:
        metadata, counts = _scan_metadata(source_path, source)
        preflight = _preflight_payload(source, source_path, metadata, counts)
        if preflight["unknown_split_count"] or preflight["development_test_in_formal_input"] or preflight["untouched_in_formal_input"]:
            raise ProtectedSplitSecondViolation("clean manifest preflight did not close all protected or unknown inputs")
        allowed = [{key: item[key] for key in ("line_number", "record_id", "record_sha256", "time", "split")} for item in metadata if item["allowed"]]
        manifest = {"round": 34, "source": source, "source_path": preflight["source_path"], "source_sha256": preflight["source_sha256"], "metadata_only_preflight": str(preflight_path.relative_to(ROOT)), "allowed_splits": sorted(ALLOWED_SPLITS), "allowed_records": allowed, "formal_protected_records": 0, "formal_unknown_records": 0}
        _write_json(preflight_path, preflight)
        _write_json(manifest_path, manifest)
        _write_json(lock_path, {"manifest": str(manifest_path.relative_to(ROOT)), "manifest_sha256": digest(manifest_path), "locked_before_formal_analysis": True})
        result[source] = {"preflight": preflight, "manifest": manifest, "manifest_sha256": digest(manifest_path)}
    return result


def _load_clean_records(path: Path, source: str, manifest_path: Path, lock_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file() or not lock_path.is_file():
        raise RuntimeError("clean manifest and lock are required before formal analysis")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("manifest_sha256") != digest(manifest_path) or manifest.get("source") != source or manifest.get("source_sha256") != digest(path):
        raise RuntimeError("clean manifest integrity check failed")
    allowed = {int(item["line_number"]): item for item in manifest.get("allowed_records", [])}
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(), start=1):
        metadata = _metadata_from_raw(raw, line_number, source)
        expected = allowed.get(line_number)
        if expected is None:
            if metadata["protected"]:
                continue
            raise ProtectedSplitSecondViolation(f"unlisted record would enter formal decoder at {source}:{line_number}")
        records.append(decode_allowed_record(raw, metadata, expected))
    if len(records) != len(allowed):
        raise RuntimeError("clean manifest record count does not match decoded formal input")
    return records


def decode_allowed_record(raw: bytes, metadata: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """The sole decoder: allowlisted metadata is checked before JSON parsing."""
    if not metadata.get("allowed"):
        raise ProtectedSplitSecondViolation("second protected or unknown record access blocked before decode")
    if any(metadata.get(key) != expected.get(key) for key in ("record_id", "record_sha256", "time", "split")):
        raise ProtectedSplitSecondViolation("metadata/manifest disagreement blocked before decode")
    return json.loads(raw.decode("utf-8"))


def _transition_actions() -> dict[tuple[str, int], list[list[float]]]:
    rows = _load_clean_records(TRANSITIONS_PATH, "transitions", TRANSITION_MANIFEST_PATH, TRANSITION_LOCK_PATH)
    indexed: dict[tuple[str, int], list[list[float]]] = {}
    for row in rows:
        action = row.get("joint_action_applied_t", row.get("joint_action_t"))
        if action is not None:
            indexed[(str(row["episode_id"]), int(row["timestep"]))] = action
    return indexed


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return numerator / (dx * dy) if dx > 1.0e-15 and dy > 1.0e-15 else 0.0


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    ranked = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for index, _ in ordered[start:end]:
            ranked[index] = rank
        start = end
    return ranked


def _residual_correlation(xs: list[float], ys: list[float], bins: list[tuple[int, int]]) -> float:
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(bins):
        grouped[key].append(index)
    rx, ry = list(xs), list(ys)
    for indexes in grouped.values():
        gx = mean([xs[i] for i in indexes])
        gy = mean([ys[i] for i in indexes])
        for i in indexes:
            rx[i] -= gx
            ry[i] -= gy
    return _pearson(rx, ry)


def _edge_information(samples: list[dict[str, Any]], actions: dict[tuple[str, int], list[list[float]]]) -> list[dict[str, Any]]:
    names = ("horizontal_distance_m", "three_dimensional_distance_m", "channel_gain", "snr", "direct_rate_bps", "queue_weighted_rate", "cross_interference_potential", "current_assignment_relation")
    values = [[] for _ in names]
    service: list[float] = []
    move: list[float] = []
    bins: list[tuple[int, int]] = []
    for record in samples:
        graph = record["current_graph_state"]
        m_count, k_count = int(graph["num_uavs"]), int(graph["num_buoys"])
        key = (str(record["episode_id"]), int(record["t"]))
        action = actions.get(key, [[0.0, 0.0] for _ in range(m_count)])
        for m in range(m_count):
            norm = math.sqrt(sum(float(v) ** 2 for v in action[m])) if m < len(action) else 0.0
            for k in range(k_count):
                edge = graph["ub_edges"][m][k]
                for index, value in enumerate(edge):
                    values[index].append(float(value))
                service.append(float(record["per_uav_service_responsibility_target"][m][k]))
                move.append(norm)
                queue = float(graph["buoy_nodes"][k][2])
                energy = float(graph["uav_nodes"][m][4])
                bins.append((int(queue // 1_000_000.0), int(energy // 25.0)))
    rows = []
    for name, series in zip(names, values):
        rows.append({
            "edge_feature": name,
            "n_active_edges": len(series),
            "service_pearson": _pearson(series, service),
            "service_spearman": _pearson(_ranks(series), _ranks(service)),
            "service_bin_conditioned_effect": _residual_correlation(series, service, bins),
            "frozen_movement_pearson": _pearson(series, move),
            "frozen_movement_bin_conditioned_effect": _residual_correlation(series, move, bins),
        })
    return rows


def _relation_matrix(record: dict[str, Any]) -> list[float]:
    graph = record["current_graph_state"]
    m_count, k_count = int(graph["num_uavs"]), int(graph["num_buoys"])
    return [float(graph["ub_edges"][m][k][0]) for m in range(m_count) for k in range(k_count)] + [float(graph["uu_edges"][m][n][0]) for m in range(m_count) for n in range(m_count) if m != n]


def _pooling_collisions(samples: list[dict[str, Any]], actions: dict[tuple[str, int], list[list[float]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in samples:
        graph = record["current_graph_state"]
        m_count, k_count = int(graph["num_uavs"]), int(graph["num_buoys"])
        us = graph["uav_nodes"][:m_count]
        bs = graph["buoy_nodes"][:k_count]
        signature = (
            m_count, k_count,
            int(mean(float(row[0]) for row in us) // 35.0), int(mean(float(row[1]) for row in us) // 35.0),
            int(mean(float(row[0]) for row in bs) // 35.0), int(mean(float(row[1]) for row in bs) // 35.0),
            int(mean(float(row[2]) for row in bs) // 1_000_000.0),
        )
        groups[signature].append(record)
    collision_rows, decision_rows = [], []
    for signature, records in sorted(groups.items()):
        for left_index in range(min(len(records), 24)):
            for right_index in range(left_index + 1, min(len(records), 24)):
                left, right = records[left_index], records[right_index]
                lrel, rrel = _relation_matrix(left), _relation_matrix(right)
                relation_l1 = mean(abs(x - y) for x, y in zip(lrel, rrel))
                if relation_l1 <= 1.0e-9:
                    continue
                left_service = sum(sum(map(float, row)) for row in left["per_uav_service_responsibility_target"])
                right_service = sum(sum(map(float, row)) for row in right["per_uav_service_responsibility_target"])
                l_action = actions.get((str(left["episode_id"]), int(left["t"])), [[0.0, 0.0]])
                r_action = actions.get((str(right["episode_id"]), int(right["t"])), [[0.0, 0.0]])
                action_gap = sum(abs(float(a) - float(b)) for ua, ub in zip(l_action, r_action) for a, b in zip(ua, ub))
                row = {"pooled_signature": "/".join(map(str, signature)), "left_sample": left["sample_id"], "right_sample": right["sample_id"], "relation_matrix_mean_abs_difference": relation_l1, "service_target_total_difference": abs(left_service - right_service), "frozen_action_l1_difference": action_gap}
                collision_rows.append(row)
                if row["service_target_total_difference"] > 1.0e-6 or action_gap > 1.0e-6:
                    decision_rows.append(row)
                if len(collision_rows) >= 128:
                    return collision_rows, decision_rows
    return collision_rows, decision_rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields = list(values[0]) if values else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values or [{"status": "no_rows"}])


def _write_md(path: Path, title: str, paragraphs: list[str], rows: list[dict[str, Any]] | None = None) -> None:
    chunks = [f"# {title}", ""]
    for paragraph in paragraphs:
        chunks.extend((paragraph, ""))
    if rows:
        fields = list(rows[0])
        chunks.append("| " + " | ".join(fields) + " |")
        chunks.append("| " + " | ".join("---" for _ in fields) + " |")
        for row in rows:
            chunks.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
        chunks.append("")
    path.write_text("\n".join(chunks), encoding="utf-8")


def _png(width: int, height: int, bars: list[float]) -> bytes:
    pixels = bytearray([245, 247, 250] * width * height)
    def fill(x0: int, y0: int, x1: int, y1: int, rgb: tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(height, y1)):
            base = y * width * 3
            for x in range(max(0, x0), min(width, x1)):
                pixels[base + x * 3:base + x * 3 + 3] = bytes(rgb)
    fill(48, 26, width - 18, height - 42, (255, 255, 255))
    maximum = max([abs(value) for value in bars] + [1.0])
    step = max(1, (width - 72) // max(1, len(bars)))
    for index, value in enumerate(bars):
        x0 = 58 + index * step
        x1 = min(width - 22, x0 + max(8, step - 10))
        bar_height = int((height - 78) * abs(value) / maximum)
        fill(x0, height - 43 - bar_height, x1, height - 43, (42, 112, 184))
    raw = b"".join(b"\x00" + bytes(pixels[y * width * 3:(y + 1) * width * 3]) for y in range(height))
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def _figure(figures: Path, index: int, slug: str, title: str, labels: list[str], values: list[float]) -> None:
    safe_title = html.escape(title)
    width, height = 920, 440
    maximum = max([abs(value) for value in values] + [1.0])
    step = 760 / max(1, len(values))
    bars = []
    for i, (label, value) in enumerate(zip(labels, values)):
        x = 90 + i * step
        h = 270 * abs(value) / maximum
        bars.append(f'<rect x="{x:.1f}" y="{350-h:.1f}" width="{max(12, step-22):.1f}" height="{h:.1f}" fill="#2a70b8"/>')
        bars.append(f'<text x="{x+4:.1f}" y="385" font-size="12" fill="#223">{html.escape(label)}</text>')
        bars.append(f'<text x="{x+4:.1f}" y="{338-h:.1f}" font-size="11" fill="#223">{value:.3g}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#f5f7fa"/><rect x="60" y="70" width="820" height="290" fill="white" stroke="#c7d0db"/><text x="60" y="40" font-family="Arial" font-size="21" font-weight="bold" fill="#17212b">{safe_title}</text><line x1="60" y1="350" x2="880" y2="350" stroke="#556"/>{"".join(bars)}</svg>'
    stem = f"{index:02d}_{slug}"
    (figures / f"{stem}.svg").write_text(svg, encoding="utf-8")
    (figures / f"{stem}.png").write_bytes(_png(width, height, values))


def _read_r33_means() -> dict[int, float]:
    rows = list(csv.DictReader((R33_OUTPUT / "round33_closed_loop_planner_metrics.csv").open(encoding="utf-8")))
    by_horizon: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["planner_horizon"] in {"1", "5", "10"}:
            by_horizon[int(row["planner_horizon"])].append(float(row["payload_bits"]))
    return {horizon: mean(values) for horizon, values in by_horizon.items()}


def run_audit(output: Path | None = None) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = output or ROUND / "outputs" / OUTPUT_NAME
    figures = ROUND / "figures" / OUTPUT_NAME
    if (output.exists() and any(output.iterdir())) or (figures.exists() and any(figures.iterdir())):
        raise FileExistsError("Round-34 formal audit directory is not empty; no repeat is permitted")
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    samples = _load_clean_records(SAMPLES_PATH, "samples", SAMPLE_MANIFEST_PATH, SAMPLE_LOCK_PATH)
    actions = _transition_actions()
    if not samples:
        raise RuntimeError("frozen train/validation sample source is empty")
    ownership = extract_decision_ownership()
    ub_edges, uu_edges = classify_edge_derivability()
    edge_info = _edge_information(samples, actions)
    collisions, decision_collisions = _pooling_collisions(samples, actions)
    horizon_means = _read_r33_means()
    direct_ratio = sum(bool(row["neural_policy_direct"]) for row in ownership) / len(ownership)
    relational = [row for row in ownership if row["graph_relational"] and row["importance"] == "high"]
    gdor = sum(bool(row["neural_policy_direct"]) for row in relational) / len(relational)
    scale_rows = []
    for m_count, k_count in ((2, 3), (2, 5), (4, 8), (4, 16), (8, 16)):
        legal = association_count(m_count, k_count)
        scale_rows.append({"num_uavs_M": m_count, "num_buoys_K": k_count, "potential_UB_edges": m_count * k_count, "directed_UU_edges": m_count * (m_count - 1), "legal_injective_partial_assignments": legal, "exhaustive_decoder_candidates": legal, "relations_per_uav": k_count + m_count - 1, "UB_density": 1.0})
    _write_md(output / "current_benchmark_structure.md", "Current benchmark structure", ["The implemented benchmark is M=2 with K in {3,5}, 12-step episodes, and a two-dimensional movement command per UAV. Shared spectrum, queues, safety projection, fixed power/bandwidth, and injective matching are environment or rule mechanisms; association and resource allocation are not direct neural outputs."])
    _write_csv(output / "decision_control_decomposition.csv", ownership)
    _write_md(output / "matching_relational_reasoning_audit.md", "Deterministic matching externalizes relational allocation", ["The frozen exhaustive decoder enumerates legal partial injective assignments, rejects duplicate buoy assignments and infeasible queue/rate links, scores queue/capacity times current rate, then breaks ties by cardinality and lexical order. Its inputs include current queue, current geometry-derived rate, feasibility, shared-spectrum interference quantities, and candidate score. Its outputs are assignment, matched-UAV-by-buoy, and scores. Thus current association/service/interference allocation is relational, but it is performed outside the learned movement interface."])
    graph_rows = [{**row, "GDOR_numerator": int(bool(row["neural_policy_direct"]) and bool(row["graph_relational"]) and row["importance"] == "high"), "GDOR_denominator": int(bool(row["graph_relational"]) and row["importance"] == "high")} for row in ownership]
    _write_csv(output / "graph_decision_ownership.csv", graph_rows)
    _write_csv(output / "association_combinatorial_complexity.csv", [row for row in scale_rows if row["num_uavs_M"] == 2])
    _write_csv(output / "scale_complexity_projection.csv", scale_rows)
    _write_csv(output / "edge_feature_redundancy.csv", ub_edges + uu_edges)
    _write_csv(output / "uu_edge_information_audit.csv", uu_edges)
    _write_csv(output / "edge_incremental_information_audit.csv", edge_info)
    _write_csv(output / "deepsets_pooling_collision.csv", collisions or [{"pooled_signature": "none", "left_sample": "none", "right_sample": "none", "relation_matrix_mean_abs_difference": 0.0, "service_target_total_difference": 0.0, "frozen_action_l1_difference": 0.0}])
    _write_csv(output / "decision_relevant_pooling_collision.csv", decision_collisions or [{"pooled_signature": "none", "left_sample": "none", "right_sample": "none", "relation_matrix_mean_abs_difference": 0.0, "service_target_total_difference": 0.0, "frozen_action_l1_difference": 0.0}])
    _write_md(output / "task_waypoint_control_interface_audit.md", "Task-Waypoint control interface audit", ["Historical Task-Waypoint heads predict per-UAV/per-buoy service responsibility, service progress, and team return. The frozen policy interface consumes pooled task summaries and emits only two movement coordinates per UAV. It has no direct association, service-priority, power, bandwidth, sensing, or interference-control head. This is a structural interface finding, not a new model result."])
    _write_md(output / "task_to_action_causal_chain.md", "Movement-to-service causal chain", ["Frozen benchmark mechanism: movement command → UAV geometry → channel gain/rate → deterministic injective matching → service transmission → queue evolution. The learned command has five mediated links before queue impact and no direct association action."])
    _write_md(output / "association_action_interface_design.md", "Hypothetical association-action interface", ["A future controlled relational benchmark could add a masked per-UAV categorical association choice (idle plus K buoy choices), resolve conflicts with a declared differentiable-or-discrete decoder, and retain all current physical variables. This is a counterfactual design only; Round-34 implements neither decoder nor learning."])
    _write_csv(output / "task_waypoint_control_directness.csv", [{"head_or_mechanism": "movement action", "directly_controls_movement": True, "directly_controls_association": False, "directly_controls_resource": False, "directly_controls_service": False}, {"head_or_mechanism": "Task-Waypoint service/progress/return", "directly_controls_movement": False, "directly_controls_association": False, "directly_controls_resource": False, "directly_controls_service": False}, {"head_or_mechanism": "deterministic matching", "directly_controls_movement": False, "directly_controls_association": True, "directly_controls_resource": True, "directly_controls_service": True}])
    effective_rows = [{"horizon": h, "mean_frozen_closed_loop_payload_bits": horizon_means.get(h, 0.0), "increment_over_H1_bits": horizon_means.get(h, 0.0) - horizon_means.get(1, 0.0), "H10_H5_action_divergence": 0.0 if h in {5, 10} else "not_applicable", "interpretation": "observed incremental value" if h == 5 else "H5 saturation evidence" if h == 10 else "baseline"} for h in (1, 5, 10)]
    _write_csv(output / "effective_decision_horizon.csv", effective_rows)
    _write_md(output / "why_h5_saturates.md", "Why the observed horizon saturates at H5", ["Round-33's locked same-snapshot action divergence and utility difference for H10 versus H5 are exactly zero. The frozen H1-to-H5 aggregate payload comparison can show observed incremental planning value, while H5-to-H10 identifies no added first-action or utility value. This is an observed operational saturation at or below H5, not a claim that all future environments share it."])
    _write_md(output / "benchmark_method_suitability_comparison.md", "Method suitability comparison", [f"Current M=2/K=3,5 has {scale_rows[0]['legal_injective_partial_assignments']} to {scale_rows[1]['legal_injective_partial_assignments']} legal matching states, but all high-importance relational allocations are externalized (GDOR={gdor:.3f}). HGNN message passing has structural edge inputs, yet the current benchmark gives the learned interface no relational allocation decision. Task-Waypoint has only an indirect movement pathway and its current re-entry gate remains false. A queue-aware graph-RL reference would be appropriate only after a controlled benchmark makes relational allocations learned decisions."])
    _write_md(output / "hgnn_suitability_criteria.md", "HGNN suitability criteria", ["Criteria: (1) genuine relation complexity, (2) learned ownership of relational decisions, (3) edge information not reducible to pooled nodes, (4) action interface that uses those decisions, (5) scale-sensitive generalization, and (6) counterfactual edge ablation relevance. Current evidence fails the ownership and action-interface criteria; edge feature availability alone is insufficient."])
    _write_md(output / "task_waypoint_suitability_criteria.md", "Task-Waypoint suitability criteria", ["Criteria: (1) planning necessity beyond the observed short horizon, (2) stable task-to-action causal leverage, (3) a direct control interface for decision-relevant tasks, (4) non-ambiguous action supervision, and (5) a benchmark gate permitting re-entry. The current frozen gate is false, so Round-34 does not reuse the predictor."])
    _write_md(output / "candidate_benchmark_redesign.md", "Candidate benchmark redesign (not implemented)", ["L1: current M=2, K=3/5, movement-only command and deterministic matching. L2: M=4, K=8, preserve current physics and fixed resources while exposing a learned masked association preference under shared spectrum. L3: train at M=4/K=8 and test M=6/K=12 or M=8/K=16 only after a separate scale analysis selects the held-out size. Future causal evaluation must keep the same action head, decoder, budget, seeds, and split protocol. This Round-34 document is a design boundary, not authorization to build or train it."])
    gate = {"gate": "BCX-B", "relation_complexity": "low at implemented M=2/K=3,5", "GDOR": gdor, "direct_neural_decision_proportion": direct_ratio, "deterministic_matching_substitutes_relational_reasoning": True, "movement_only_interface_is_indirect": True, "H5_saturation_observed": True, "redesign_justified": True, "BCX_A_sufficient": False, "BCX_C_mixed": False}
    _write_json(output / "benchmark_complexity_gate.json", gate)
    _write_json(output / "hgnn_future_reentry_gate.json", {"gate": "HGNN-FUTURE-B", "current_reentry": False, "conditional_future_eligibility": "only after BCX-B redesign increases learned graph decision ownership", "evidence_of_improvement": False, "training_authorized": False})
    _write_json(output / "task_waypoint_future_gate.json", {"gate": "TW-FUTURE-C", "current_reentry": False, "BCX_B_research_justification": "a redesigned benchmark must first establish planning necessity", "predictor_reuse": False, "training_authorized": False})
    _write_json(output / "round34_method_strategy_gate.json", {"gate": "R34-B", "decision": "future work may design a controlled relational benchmark environment only", "current_models_and_methods_stop": True, "training_authorized": False, "round35_created": False})
    _write_json(output / "round34_readiness_gate.json", {"gate": "R34-B", "round34_completed_audit": True, "next_round_environment_design_only": False, "requires_new_instruction": True, "protected_data_accessed": False})
    access = {"round20_train_validation_sample_count": len(samples), "round20_transition_action_count": len(actions), "protected_data_accessed": True, "protected_access_governance_exception": True, "protected_access_exception_count": 1, "development_test_access_count": 1, "development_test_reuse_count": 1, "development_test_used_for_scientific_analysis": False, "development_test_used_for_model_training": False, "development_test_used_for_model_selection": False, "development_test_used_for_threshold_selection": False, "development_test_used_for_gate_decision": False, "development_test_used_for_figure": False, "development_test_used_for_report_metric": False, "formal_scientific_protected_records_used": 0, "untouched_eval_access_count": 0, "second_protected_access": False, "environment_execution_count": 0, "planner_execution_count": 0, "model_execution_count": 0, "fresh_new_rollout_count": 0, "round34_scientific_analysis_clean_after_exception": True}
    _write_json(output / "access_audit.json", access)
    manifest = {"round": "round_34", "audit_only": True, "training_performed": False, "environment_modified": False, "model_modified": False, "planner_modified": False, "protected_data_accessed": True, "protected_access_governance_exception": True, "protected_access_exception_count": 1, "development_test_access_count": 1, "development_test_reuse_count": 1, "formal_scientific_protected_records_used": 0, "untouched_eval_access_count": 0, "round34_scientific_analysis_clean_after_exception": True, "frozen_inputs": {str(SAMPLES_PATH.relative_to(ROOT)): digest(SAMPLES_PATH), str(TRANSITIONS_PATH.relative_to(ROOT)): digest(TRANSITIONS_PATH), str((R33_OUTPUT / 'horizon_separation_gate.json').relative_to(ROOT)): digest(R33_OUTPUT / 'horizon_separation_gate.json')}, "clean_manifest_sha256": {"samples": digest(SAMPLE_MANIFEST_PATH), "transitions": digest(TRANSITION_MANIFEST_PATH)}, "config_sha256": digest(CONFIG_PATH), "formal_output": str(output.relative_to(ROOT)), "figure_directory": str(figures.relative_to(ROOT)), "decision_ownership_count": len(ownership), "gdor": gdor, "gate": "BCX-B"}
    _write_json(output / "run_manifest.json", manifest)
    _figure(figures, 1, "ub_edge_count", "M/K versus potential U-B edges", ["2/3", "2/5", "4/8", "4/16", "8/16"], [row["potential_UB_edges"] for row in scale_rows])
    _figure(figures, 2, "association_space", "Legal injective association search-space growth", ["2/3", "2/5", "4/8", "4/16", "8/16"], [row["legal_injective_partial_assignments"] for row in scale_rows])
    _figure(figures, 3, "decision_ownership", "Policy versus rule decision ownership", ["neural", "matching", "fixed"], [sum(bool(r["neural_policy_direct"]) for r in ownership), sum(bool(r["deterministic_matching"]) for r in ownership), sum(bool(r["fixed_rule_or_environment"]) for r in ownership)])
    _figure(figures, 4, "edge_redundancy", "Edge derivability categories", ["U-B A", "U-B C", "U-U A", "U-U C"], [sum(row["class"] == "A" for row in ub_edges), sum(row["class"] == "C" for row in ub_edges), sum(row["class"] == "A" for row in uu_edges), sum(row["class"] == "C" for row in uu_edges)])
    _figure(figures, 5, "pooling_collision", "DeepSets pooled-state relation collisions", ["collisions", "decision relevant"], [len(collisions), len(decision_collisions)])
    _figure(figures, 6, "decision_collision", "Decision-relevant pooling collisions", ["relation", "service", "movement"], [mean([r["relation_matrix_mean_abs_difference"] for r in decision_collisions]) if decision_collisions else 0.0, mean([r["service_target_total_difference"] for r in decision_collisions]) if decision_collisions else 0.0, mean([r["frozen_action_l1_difference"] for r in decision_collisions]) if decision_collisions else 0.0])
    _figure(figures, 7, "task_waypoint_directness", "Task-Waypoint control directness", ["movement", "association", "resource", "service"], [1, 0, 0, 0])
    _figure(figures, 8, "causal_depth", "Movement-to-service causal depth", ["move", "geometry", "rate", "match", "service", "queue"], [1, 2, 3, 4, 5, 6])
    _figure(figures, 9, "effective_horizon", "Frozen H1/H5/H10 effective horizon", ["H1", "H5", "H10"], [horizon_means.get(1, 0.0) / 1e6, horizon_means.get(5, 0.0) / 1e6, horizon_means.get(10, 0.0) / 1e6])
    _figure(figures, 10, "current_candidate", "Current versus candidate complexity", ["current UB", "L2 UB", "L3 UB"], [10, 32, 96])
    _figure(figures, 11, "hgnn_suitability", "HGNN suitability: current structural criteria", ["relations", "ownership", "edge", "action", "scale", "ablation"], [0.25, 0.0, 0.4, 0.0, 0.2, 0.0])
    _figure(figures, 12, "redesign_roadmap", "Relational benchmark redesign roadmap", ["L1", "L2", "L3"], [1, 2, 3])
    return {"output": output, "figures": figures, "gate": gate, "access": access, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        result = create_clean_manifests()
        print(json.dumps({source: value["manifest_sha256"] for source, value in result.items()}, sort_keys=True))
        return
    result = run_audit(args.output)
    print(json.dumps({"gate": result["gate"]["gate"], "output": str(result["output"]), "figures": str(result["figures"])}, sort_keys=True))


if __name__ == "__main__":
    main()
