"""Restore revised states and pair existing arrival labels.

This entry reads archived blobs and the existing D6 tables. It does not
train, step the maritime environment, load a checkpoint, or run a network.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


TRAIN_SCENARIOS = tuple(range(322016, 322022))
EXPLORE_SCENARIOS = (322022, 322023)
TARGET_SCENARIOS = TRAIN_SCENARIOS + EXPLORE_SCENARIOS
TARGET_TIMES = (15, 30)
GRAPH_KEYS = ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu")
JOIN_FIELDS = ("scenario_id", "time_step", "candidate_id", "assignment")
ROW_FIELDS = (
    "scenario_id",
    "time_step",
    "candidate_id",
    "assignment",
    "row_index",
    "subset_hash",
    "state_hash",
    "zero_payload_bits",
    "real_payload_bits",
    "group",
    "graph_file",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, args: list[str], data: bytes | None = None) -> bytes:
    proc = subprocess.run(
        ["git", "-c", "core.longpaths=true", *args],
        cwd=repo,
        input=data,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"git {' '.join(args)} failed")
    return proc.stdout


def _group(scenario: int) -> str:
    if scenario in TRAIN_SCENARIOS:
        return "train"
    if scenario in EXPLORE_SCENARIOS:
        return "explore_eval"
    raise RuntimeError(f"scenario {scenario} is outside the fixed L2 split")


def _restore(repo: Path, inventory: dict, source_root: Path) -> list[dict]:
    commit = str(inventory["source_commit"])
    rows = []
    for item in inventory["files"]:
        blob = str(item["git_blob_sha"])
        data = _git(repo, ["cat-file", "blob", blob])
        if len(data) != int(item["bytes"]):
            raise RuntimeError(f"{item['source_path']} blob length {len(data)} != {item['bytes']}")
        hashed = _git(repo, ["hash-object", "--stdin"], data).decode("ascii").strip()
        if hashed != blob:
            raise RuntimeError(f"{item['source_path']} blob {hashed} != {blob}")
        output = repo / str(item["output_path"])
        if output.parent.resolve() != (source_root / output.parent.name).resolve() and source_root not in output.parents:
            raise RuntimeError(f"inventory output escapes the source root: {output}")
        if source_root not in output.resolve().parents:
            raise RuntimeError(f"inventory output is outside {source_root}: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        rows.append(
            {
                "source_path": item["source_path"],
                "source_commit": commit,
                "git_blob_sha": blob,
                "bytes": len(data),
                "sha256": _sha256(data),
                "role": item["role"],
                "output_path": item["output_path"].replace("\\", "/"),
            }
        )
    if len(rows) != int(inventory["source_file_count"]):
        raise RuntimeError("restored file count does not match the inventory")
    if sum(row["bytes"] for row in rows) != int(inventory["total_source_bytes"]):
        raise RuntimeError("restored byte count does not match the inventory")
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _state_key(row: dict[str, str]) -> tuple[int, int]:
    return int(row["scenario_id"]), int(row["time_step"])


def _join_key(row: dict[str, str]) -> tuple[int, int, int, str]:
    return (
        int(row["scenario_id"]),
        int(row["time_step"]),
        int(row["candidate_id"]),
        str(row["assignment"]),
    )


def _pair_rows(labels: list[dict[str, str]], candidates: list[dict[str, str]], manifests: dict[tuple[int, int], dict]) -> list[dict[str, str]]:
    selected = [_join_key(row) for row in labels if _state_key(row) in manifests]
    if len(selected) != 2048:
        raise RuntimeError(f"expected 2048 label rows for the 16 states, found {len(selected)}")
    if len(selected) != len(set(selected)):
        raise RuntimeError("label join keys repeat inside the 16 states")
    label_by_key = {_join_key(row): row for row in labels if _state_key(row) in manifests}
    candidate_by_key = {}
    candidate_hits = [row for row in candidates if _state_key(row) in manifests]
    if len(candidate_hits) != 2048:
        raise RuntimeError(f"expected 2048 selected rows for the 16 states, found {len(candidate_hits)}")
    for row in candidate_hits:
        key = _join_key(row)
        if key in candidate_by_key:
            raise RuntimeError(f"selected candidate key repeats: {key}")
        candidate_by_key[key] = row
    if set(candidate_by_key) != set(label_by_key):
        raise RuntimeError("label rows and selected rows do not join on scenario, time, candidate, and assignment")
    paired = []
    for key in manifests:
        manifest = manifests[key]
        state_rows = [row for row in label_by_key.values() if _state_key(row) == key]
        if len(state_rows) != 128:
            raise RuntimeError(f"{key} has {len(state_rows)} candidates")
        hashes = {row["state_hash"] for row in state_rows}
        subsets = {row["subset_hash"] for row in state_rows}
        if hashes != {str(manifest["source_state_hash"])}:
            raise RuntimeError(f"{key} state_hash does not match source_state_hash")
        if len(subsets) != 1:
            raise RuntimeError(f"{key} has more than one subset_hash")
        for row in state_rows:
            other = candidate_by_key[_join_key(row)]
            shared = ("j15_bits", "subset_hash", "state_hash", "pool_hash", "row_index", "utility_rank")
            for field in shared:
                if row[field] != other[field]:
                    raise RuntimeError(f"{key} field {field} differs between the label and selected tables")
            paired.append(row)
    paired.sort(key=lambda row: (_state_key(row), int(row["row_index"]), int(row["candidate_id"])))
    return paired


def _manifests(restored: list[dict], repo: Path) -> dict[tuple[int, int], dict]:
    found = {}
    for row in restored:
        if row["role"] != "state_manifest":
            continue
        payload = json.loads((repo / row["output_path"]).read_text(encoding="utf-8"))
        key = (int(payload["scenario_id"]), int(payload["time_step"]))
        found[key] = payload
    expected = {(scenario, time_step) for scenario in TARGET_SCENARIOS for time_step in TARGET_TIMES}
    if set(found) != expected:
        raise RuntimeError(f"restored manifests cover {sorted(found)} rather than the 16 target states")
    return found


def _check_states(restored: list[dict], manifests: dict[tuple[int, int], dict], repo: Path) -> None:
    by_output = {row["output_path"]: row for row in restored}
    for key, manifest in manifests.items():
        scenario, time_step = key
        npz_rel = f"results/legacy_arrival_aware_ab_l2/source_states/scenario_{scenario}/t{time_step:03d}_state.npz"
        record = by_output[npz_rel]
        if record["sha256"] != str(manifest["checksum_state"]):
            raise RuntimeError(f"{npz_rel} file sha256 does not match checksum_state")
        if record["sha256"] == str(manifest["source_state_hash"]):
            raise RuntimeError(f"{npz_rel} file hash was substituted for source_state_hash")
        with np.load(repo / npz_rel) as payload:
            keys = set(payload.files)
            for field in ("service_progress", "uav_step_displacement_m"):
                if field not in keys:
                    raise RuntimeError(f"{npz_rel} is missing {field}")
                values = np.asarray(payload[field], dtype=np.float64)
                if values.size == 0 or not np.all(np.isfinite(values)):
                    raise RuntimeError(f"{npz_rel} field {field} is empty or non-finite")


def _graphs(source_root: Path, manifests: dict[tuple[int, int], dict], output: Path) -> dict:
    try:
        from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
        from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import hgnn_tensors
        from wt.lowlevel_v2.phase5l_joint_encoder.model import EDGE_UB_DIM, append_queue_edge
    except Exception as exc:
        return {
            "status": "not_passed",
            "reason": f"{type(exc).__name__}: {exc}",
            "files": [],
        }
    config = L3BBitQueueEnvV2().config
    graph_dir = output / "paired" / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for scenario, time_step in sorted(manifests):
        npz_path = source_root / f"scenario_{scenario}" / f"t{time_step:03d}_state.npz"
        with np.load(npz_path) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        built = hgnn_tensors(arrays, config)
        edge = np.asarray(built["edge"], dtype=np.float32)
        if int(edge.shape[-1]) != int(EDGE_UB_DIM) + 1:
            raise RuntimeError(f"{scenario} t={time_step} edge width {edge.shape[-1]} is not the existing revision width")
        graph = {
            "uav": np.asarray(built["uav"], dtype=np.float32)[None, ...],
            "buoy": np.asarray(built["buoy"], dtype=np.float32)[None, ...],
            "uav_mask": np.ones((1, built["uav"].shape[0]), dtype=bool),
            "buoy_mask": np.ones((1, built["buoy"].shape[0]), dtype=bool),
            "global": np.asarray(built["global"], dtype=np.float32)[None, ...],
            "edge": edge[None, ..., : int(EDGE_UB_DIM)],
            "edge_uu": np.asarray(built["edge_uu"], dtype=np.float32)[None, ...],
        }
        if tuple(graph) != GRAPH_KEYS:
            raise RuntimeError("graph key set changed")
        conditioned = append_queue_edge(graph)
        if int(conditioned["edge"].shape[-1]) != int(EDGE_UB_DIM) + 1:
            raise RuntimeError("queue-edge append changed the existing width")
        if not np.allclose(conditioned["edge"], edge[None, ...], atol=1.0e-6, rtol=0.0):
            raise RuntimeError(f"{scenario} t={time_step} queue-edge append does not match the revision edge")
        forbidden = [key for key in graph if key not in GRAPH_KEYS]
        if forbidden:
            raise RuntimeError(f"graph gained forbidden fields {forbidden}")
        relative = f"paired/graphs/scenario_{scenario}_t{time_step:03d}.npz"
        target = output / relative
        with target.open("wb") as handle:
            np.savez(handle, **graph)
        files.append(
            {
                "scenario_id": scenario,
                "time_step": time_step,
                "path": relative,
                "sha256": _sha256(target.read_bytes()),
                "shapes": {key: list(graph[key].shape) for key in GRAPH_KEYS},
                "queue_edge_width": int(conditioned["edge"].shape[-1]),
            }
        )
    return {"status": "passed", "reason": "", "files": files}


def _write_rows(path: Path, rows: list[dict[str, str]], graphs: dict[tuple[int, int], str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            key = _state_key(row)
            writer.writerow(
                {
                    "scenario_id": row["scenario_id"],
                    "time_step": row["time_step"],
                    "candidate_id": row["candidate_id"],
                    "assignment": row["assignment"],
                    "row_index": row["row_index"],
                    "subset_hash": row["subset_hash"],
                    "state_hash": row["state_hash"],
                    "zero_payload_bits": row["zero_payload_bits"],
                    "real_payload_bits": row["real_payload_bits"],
                    "group": _group(int(row["scenario_id"])),
                    "graph_file": graphs.get(key, ""),
                }
            )


def prepare(repo: Path, inventory_path: Path, labels_path: Path, candidates_path: Path, source_root: Path, output: Path) -> dict:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if source_root.resolve() != (repo / str(inventory["output_source_root"])).resolve():
        raise RuntimeError("source root does not match the inventory output root")
    if output.resolve() == (repo / "results/phase6i_urgency_arrival_aware_expansion").resolve():
        raise RuntimeError("refusing to write over the D6 result directory")
    restored = _restore(repo, inventory, source_root)
    manifests = _manifests(restored, repo)
    _check_states(restored, manifests, repo)
    labels = _read_csv(labels_path)
    candidates = _read_csv(candidates_path)
    paired = _pair_rows(labels, candidates, manifests)
    graph_report = _graphs(repo / str(inventory["output_source_root"]), manifests, output)
    graph_paths = {
        (int(item["scenario_id"]), int(item["time_step"])): item["path"]
        for item in graph_report["files"]
    }
    rows_path = output / "paired" / "rows.csv"
    _write_rows(rows_path, paired, graph_paths)
    groups = {
        "train_scenarios": list(TRAIN_SCENARIOS),
        "explore_eval_scenarios": list(EXPLORE_SCENARIOS),
        "times": list(TARGET_TIMES),
        "same_scenario_times_stay_together": True,
        "train_states": 12,
        "train_candidates": sum(1 for row in paired if _group(int(row["scenario_id"])) == "train"),
        "explore_eval_states": 4,
        "explore_eval_candidates": sum(1 for row in paired if _group(int(row["scenario_id"])) == "explore_eval"),
        "explore_eval_meaning": "scenarios 322022 and 322023 remain inside the existing training pool and were already opened by the arrival-label diagnosis; this is not a new independent validation or formal holdout",
        "graph_status": graph_report["status"],
    }
    (output / "paired" / "groups.json").write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    (output / "source_files.json").write_text(json.dumps(restored, indent=2) + "\n", encoding="utf-8")
    label_values = 2 * len(paired)
    manifest = {
        "package": "legacy_arrival_aware_ab_l2",
        "source_commit": inventory["source_commit"],
        "restored_files": len(restored),
        "restored_bytes": sum(row["bytes"] for row in restored),
        "states": len(manifests),
        "candidate_rows": len(paired),
        "label_values": label_values,
        "label_columns": ["zero_payload_bits", "real_payload_bits"],
        "join": list(JOIN_FIELDS),
        "column_mapping": {
            "labels.csv": "zero_payload_bits and real_payload_bits on one row",
            "selected_candidates.csv": "same scenario_id, time_step, candidate_id, and assignment",
            "state_manifest.source_state_hash": "labels.csv state_hash",
            "state_manifest.checksum_state": "sha256 of the restored npz bytes",
        },
        "arms_share_graph_and_order": True,
        "graph_conversion": graph_report["status"],
        "graph_reason": graph_report["reason"],
        "graph_files": graph_report["files"],
        "rows_sha256": _sha256(rows_path.read_bytes()),
        "groups": groups,
        "environment_steps": 0,
        "training_updates": 0,
        "model_forward_passes": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare_from_existing(repo: Path, source_root: Path, labels_path: Path, candidates_path: Path, output: Path) -> dict:
    """Build one package from the restored L2 states without rewriting them."""
    l2_package = (repo / "results/legacy_arrival_aware_ab_l2").resolve()
    if output.resolve() == l2_package or l2_package in output.resolve().parents:
        raise RuntimeError("refusing to write inside the L2 package")
    if source_root.resolve() == l2_package or l2_package not in source_root.resolve().parents:
        raise RuntimeError("source root must be the restored L2 state directory")
    manifests = {}
    for scenario in TARGET_SCENARIOS:
        for time_step in TARGET_TIMES:
            manifest_path = source_root / f"scenario_{scenario}" / f"t{time_step:03d}_manifest.json"
            npz_path = source_root / f"scenario_{scenario}" / f"t{time_step:03d}_state.npz"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            digest = _sha256(npz_path.read_bytes())
            if digest != str(payload["checksum_state"]):
                raise RuntimeError(f"{npz_path.name} does not match checksum_state")
            manifests[(int(scenario), int(time_step))] = payload
    paired = _pair_rows(_read_csv(labels_path), _read_csv(candidates_path), manifests)
    graph_report = _graphs(source_root, manifests, output)
    if graph_report["status"] != "passed":
        raise RuntimeError(graph_report["reason"] or "graph conversion did not pass")
    for item in graph_report["files"]:
        recorded = str(item["path"]).replace("\\", "/")
        if recorded.startswith("results/") or "legacy_arrival_aware_ab_l2" in recorded:
            raise RuntimeError(f"graph path is not package-relative: {recorded}")
        if not (output / recorded).is_file():
            raise RuntimeError(f"recorded graph is missing: {recorded}")
    graph_paths = {(int(item["scenario_id"]), int(item["time_step"])): item["path"] for item in graph_report["files"]}
    rows_path = output / "paired" / "rows.csv"
    _write_rows(rows_path, paired, graph_paths)
    return {"states": len(manifests), "candidate_rows": len(paired), "graph_files": graph_report["files"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = _repo_root()
    manifest = prepare(
        repo,
        args.inventory.resolve(),
        args.labels.resolve(),
        args.candidates.resolve(),
        args.source_root.resolve(),
        args.output.resolve(),
    )
    print(json.dumps({key: manifest[key] for key in ("restored_files", "states", "candidate_rows", "label_values", "graph_conversion", "environment_steps", "training_updates", "model_forward_passes")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
