"""Train one 28-state arm for 120 updates, then score a fixed 16-state comparison."""
from __future__ import annotations

import gc
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.envs.bit_queue import L3BBitQueueEnvV2
from wt.control.executor.protocol import primary_checkpoint
from wt.research.urgency_highlevel_pilot.protocol import FROZEN_SHARD_COMMIT, frozen_relpath
from wt.planning.formal_planner.protocol import MAPPO_SHA256, SCHEME_C_SHA256
from wt.legacy.joint_training.protocol import GRAD_CLIP
from wt.models.assignment_head.protocol import SCHEME_C_CKPT
from wt.research.urgency_arrival_aware_expansion.protocol import LABEL_SEED, MAX_CANDIDATES
from wt.research.arrival_coverage.data import COMPARISON_SCENARIOS, access_schedule, load_training_states
from wt.research.payload_objective.execute import (
    GRAPH_KEYS,
    HASHES,
    _blob_bytes,
    _combine_snap,
    _configure,
    _decoder,
    _fresh_r,
    _git,
    _git_text,
    _graph_from_arrays,
    _load_arrays,
    _load_named,
    _replay_scenario,
    _restore_counters,
    _revision_relpath,
    _same_arrays,
    _save_arrays,
    _sha256,
    _snap_payload,
    _weights,
    _write_json,
)
from wt.research.payload_objective.execute import _install_counters as _install_l8_counters
from wt.research.payload_objective.loss import expected_regret_loss
from wt.models.joint_encoder.model import append_queue_edge, batch_tensors
from wt.models.joint_encoder.protocol import LEARNING_RATE, WEIGHT_DECAY
from wt.common.paths import repo_root


UPDATES = 120
FORWARD_CAP = 204
PLANNED_FORWARDS = 200
REPLAY_STEPS = 240
ENVIRONMENT_CAP = 30960
COMPARISON_TIMES = (15, 30)
NEAR_ZERO_BITS = 1.0
INSTRUCTION_COMMIT = "42fd0cca37dd62d950a47aa3d5d1a77e007bfcf7"
BRANCH = "feature/legacy-system-maintenance"
OCEAN_BRANCH = "research/ocean-observation-isolated"
OCEAN_EXPECTED = "4d8622ef183e734f2d9a890d60d735ed716f5d74"
FORMAL_EXPECTED = "e58e5d190109e5a0e860cb5c6a412321c0d74e31"
R12_SHA256 = "4f4ffa043f13b31b83f31508ef8ade138d96cee36a9addf977d33aeb555bc27f"
REVISION_COMMIT = "8a34efc718665a7cfe4375b662e8f5249263f03d"
MODEL_ORDER = ("R28_U120", "R12_U120", "C_LISTNET", "E6", "L3A10")


def _repo() -> Path:
    return repo_root()
def _output(repo: Path) -> Path:
    return repo / "results" / "legacy_arrival_coverage_l9"


def _consumption() -> dict:
    return {
        "environment_steps": 0,
        "low_level_inferences": 0,
        "scoring_forwards": 0,
        "training_updates": 0,
        "backward_passes": 0,
        "candidate_rollouts": 0,
        "retrained_reference_models": 0,
        "old_state_environment_steps": 0,
        "old_state_scoring_forwards": 0,
        "old_state_relabels": 0,
    }


def _load_consumption(path: Path) -> dict:
    values = _consumption()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        values.update({key: int(payload[key]) for key in values if key in payload})
    return values


def _save_consumption(path: Path, consumption: dict) -> None:
    _write_json(path, consumption)


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _extra_paths(repo: Path) -> dict[str, Path]:
    return {
        "r12": repo / "results/research/payload_objective/r_epoch_010.pt",
        "l8_candidates": repo / "results/research/payload_objective/candidates.json",
        "l8_labels": repo / "results/research/payload_objective/real_labels.jsonl",
        "l8_loss": repo / "src/wt/research/payload_objective/loss.py",
        "official_policy": repo / "configs/formal_scheme_c_policy.json",
        "l2_manifest": repo / "results/research/arrival_aware/state_preparation/manifest.json",
        "historical_listnet": repo / "src/wt/algorithms/ranking_loss/losses.py",
        "mappo": primary_checkpoint(),
        "scheme_c": SCHEME_C_CKPT,
        "epoch6": repo / "checkpoints/initialization/joint_utility_epoch6.pt",
        "decoder": repo / "checkpoints/formal/matching_decoder.pt",
        "c_listnet": repo / "results/research/arrival_aware/listnet_comparison/arm_b_epoch_010.pt",
        "l3_a10": repo / "results/research/arrival_aware/listnet_comparison/arm_a_epoch_010.pt",
    }


def _hash_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {name: _sha256(path) for name, path in paths.items()}


def _expected_hashes() -> dict[str, str]:
    return {
        "epoch6": HASHES["epoch6"],
        "decoder": HASHES["decoder"],
        "c_listnet": HASHES["c_listnet"],
        "l3_a10": HASHES["l3_a10"],
        "mappo": MAPPO_SHA256,
        "scheme_c": SCHEME_C_SHA256,
        "r12": R12_SHA256,
    }


def _preflight_sources(repo: Path) -> list[dict]:
    rows = []
    for scenario in COMPARISON_SCENARIOS:
        for time_step in (0, 15, 30):
            for kind in ("manifest", "state", "candidates"):
                relpath = frozen_relpath(scenario, time_step, kind)
                disk = repo / relpath
                blob, data = _blob_bytes(repo, FROZEN_SHARD_COMMIT, relpath)
                digest = hashlib.sha256(data).hexdigest()
                if disk.is_file():
                    if digest != _sha256(disk):
                        raise RuntimeError(f"on-disk shard does not match the frozen blob: {relpath}")
                elif kind == "manifest":
                    raise RuntimeError(f"required on-disk manifest is missing: {relpath}")
                rows.append({"commit": FROZEN_SHARD_COMMIT, "path": relpath, "blob": blob, "sha256": digest, "on_disk": disk.is_file()})
        for time_step in COMPARISON_TIMES:
            relpath = _revision_relpath(scenario, time_step)
            blob, _data = _blob_bytes(repo, REVISION_COMMIT, relpath)
            rows.append({"commit": REVISION_COMMIT, "path": relpath, "blob": blob, "role": "revision_check_source"})
    return rows


def _score_tensor(model, decoder, graph: dict, assignments: np.ndarray, device: torch.device, consumption: dict) -> torch.Tensor:
    if int(consumption["scoring_forwards"]) >= FORWARD_CAP:
        raise RuntimeError("high-level scoring forward cap reached")
    if tuple(graph) != GRAPH_KEYS or any(key in graph for key in ("real_payload", "zero_payload", "scenario_id")):
        raise RuntimeError("graph contains a field outside the revised input")
    conditioned = append_queue_edge(graph)
    tensors = {key: value.to(device) for key, value in batch_tensors(conditioned).items()}
    chosen = torch.as_tensor(assignments, dtype=torch.long, device=device)
    embeddings = model.condition(tensors)
    scores = decoder.score_assignments(embeddings[0], embeddings[1], chosen).reshape(-1)
    consumption["scoring_forwards"] += 1
    if not _finite(scores) or int(scores.shape[0]) != 128:
        raise RuntimeError("scores are non-finite or not aligned to 128 candidates")
    return scores


def _choose(scores: np.ndarray, candidate_ids: np.ndarray) -> int:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    identities = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    tied = np.flatnonzero(values == np.max(values))
    return int(tied[int(np.argmin(identities[tied]))])


def _active_count(assignment: str) -> int:
    return int(sum(int(part) >= 0 for part in str(assignment).split("|")))


def _selection(scores: np.ndarray, candidate_ids: np.ndarray, assignments: list[str], payload: np.ndarray | None) -> dict:
    local = _choose(scores, candidate_ids)
    chosen = None if payload is None else float(payload[local])
    optimum = None if payload is None else float(np.max(payload))
    return {
        "local_position": local,
        "candidate_id": int(candidate_ids[local]),
        "assignment": assignments[local],
        "score": float(np.asarray(scores, dtype=np.float32)[local]),
        "active_count": _active_count(assignments[local]),
        "tie_count": int(np.sum(np.asarray(scores, dtype=np.float32) == np.max(np.asarray(scores, dtype=np.float32)))),
        "real_payload_bits": chosen,
        "subset_optimum_bits": optimum,
        "subset_regret_bits": None if payload is None else float(optimum - chosen),
    }


def _train(repo: Path, output: Path, states: list[dict], device: torch.device, consumption: dict):
    log_path = output / "train_log.jsonl"
    done_path = output / "r28_u120.pt"
    existing = 0
    if log_path.is_file():
        existing = sum(1 for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if existing not in (0, UPDATES):
        raise RuntimeError(f"partial training log has {existing} updates; refusing to train again")
    epoch_path = repo / "checkpoints/initialization/joint_utility_epoch6.pt"
    decoder_path = repo / "checkpoints/formal/matching_decoder.pt"
    if existing == UPDATES:
        if not done_path.is_file():
            raise RuntimeError("training log is complete but r28_u120.pt is missing")
        consumption["training_updates"] = UPDATES
        consumption["backward_passes"] = UPDATES
        consumption["scoring_forwards"] = max(int(consumption["scoring_forwards"]), UPDATES)
        return _load_r28(done_path, device), _decoder(decoder_path, device)
    if float(LEARNING_RATE) != 3.0e-4 or float(WEIGHT_DECAY) != 1.0e-4 or float(GRAD_CLIP) != 1.0:
        raise RuntimeError("optimizer constants differ from the L8 values")
    if int(LABEL_SEED) != 605 or int(MAX_CANDIDATES) != 128:
        raise RuntimeError("candidate sampling constants differ from seed 605 and 128 candidates")
    model, decoder, optimizer = _fresh_r(epoch_path, decoder_path, device)
    decoder_before = _weights(decoder)
    schedule = json.loads((output / "access_schedule.json").read_text(encoding="utf-8"))
    log_path.write_text("", encoding="utf-8", newline="\n")
    for update in range(UPDATES):
        state = states[update % 28]
        planned = schedule["visits"][update]
        if int(planned["scenario_id"]) != int(state["scenario_id"]) or int(planned["time_step"]) != int(state["time_step"]):
            raise RuntimeError("training departed from the fixed access schedule")
        optimizer.zero_grad(set_to_none=True)
        model.train()
        decoder.eval()
        scores = _score_tensor(model, decoder, state["graph"], state["assignments"], device, consumption)
        if not scores.requires_grad:
            raise RuntimeError("training score does not require grad")
        labels = torch.as_tensor(state["real_payload"], dtype=scores.dtype)
        loss = expected_regret_loss(scores, labels)
        if not _finite(loss):
            raise RuntimeError("non-finite expected-regret loss")
        loss.backward()
        consumption["backward_passes"] += 1
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or any(not _finite(gradient) for gradient in gradients):
            raise RuntimeError("non-finite encoder gradient")
        grad_norm = float(clip_grad_norm_(model.parameters(), float(GRAD_CLIP)))
        if not np.isfinite(grad_norm):
            raise RuntimeError("non-finite gradient norm")
        optimizer.step()
        consumption["training_updates"] += 1
        record = {
            "arm": "R28_U120",
            "update": update,
            "scenario_id": int(state["scenario_id"]),
            "time_step": int(state["time_step"]),
            "visit": int(planned["visit"]),
            "loss": float(loss.detach().cpu()),
            "grad_norm": grad_norm,
        }
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record) + "\n")
        if update % 28 == 0 or update == UPDATES - 1:
            print(f"R28 update {update + 1}/120 loss={record['loss']:.6f} grad={grad_norm:.4f}", flush=True)
    if _weights(decoder) != decoder_before or int(consumption["training_updates"]) != UPDATES:
        raise RuntimeError("training did not finish with a frozen decoder and 120 updates")
    torch.save(
        {
            "updates": UPDATES,
            "full_passes": 4,
            "tail_updates": 8,
            "arm": "R28_U120",
            "label": "real_payload_bits",
            "loss": "expected_regret_temperature_1",
            "model": model.state_dict(),
            "initialization_sha256": HASHES["epoch6"],
        },
        done_path,
    )
    _check_saved_r28(done_path, model)
    model.eval()
    return model, decoder


def _check_saved_r28(path: Path, model) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "epoch" in payload:
        raise RuntimeError("R28 checkpoint contains an epoch field")
    if int(payload["updates"]) != 120 or int(payload["full_passes"]) != 4 or int(payload["tail_updates"]) != 8:
        raise RuntimeError("R28 checkpoint does not record 4 full passes and 8 tail updates")
    saved = [(name, tensor.detach().cpu().contiguous().numpy().tobytes()) for name, tensor in payload["model"].items()]
    if saved != _weights(model):
        raise RuntimeError("saved R28 weights do not match the trained model")


def _load_r28(path: Path, device: torch.device):
    from wt.research.payload_objective.execute import _encoder

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "epoch" in payload or int(payload["updates"]) != 120:
        raise RuntimeError("R28 checkpoint is not the 120-update coverage model")
    model = _encoder(payload, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _state_dir(output: Path, scenario: int, time_step: int) -> Path:
    return output / "comparison" / f"scenario_{scenario}" / f"t{time_step:03d}"


def _replay_all(repo: Path, output: Path, consumption: dict) -> None:
    for scenario in COMPARISON_SCENARIOS:
        folder = output / "comparison" / f"scenario_{scenario}"
        stamp = folder / "started.json"
        complete = folder / "complete.json"
        if complete.is_file():
            continue
        if stamp.is_file():
            raise RuntimeError(f"scenario {scenario} replay already started; refusing to spend the steps again")
        _write_json(stamp, {"scenario_id": int(scenario), "status": "started"})
        before = int(consumption["environment_steps"])
        replay = _replay_scenario(int(scenario))
        delta = int(consumption["environment_steps"]) - before
        if delta != 30:
            raise RuntimeError(f"scenario {scenario} used {delta} environment steps")
        if [int(row["time_step"]) for row in replay["states"]] != [0, 15, 30]:
            raise RuntimeError("replay did not capture t0, t15, and t30")
        _save_consumption(output / "consumption.json", consumption)
        saved = []
        for row in replay["states"]:
            time_step = int(row["time_step"])
            if time_step not in COMPARISON_TIMES:
                continue
            target = _state_dir(output, scenario, time_step)
            arrays, scalars = _snap_payload(row["env_snap"])
            _save_arrays(target / "env_snap.npz", arrays)
            _write_json(target / "env_scalars.json", scalars)
            restored = _combine_snap(_load_arrays(target / "env_snap.npz"), json.loads((target / "env_scalars.json").read_text(encoding="utf-8")))
            for key, value in row["env_snap"].items():
                if isinstance(value, np.ndarray) and not np.array_equal(np.asarray(restored[key]), np.asarray(value)):
                    raise RuntimeError("saved environment snapshot does not round-trip")
            _write_json(target / "ledger.json", row["ledger_snap"])
            _save_arrays(target / "revised_state.npz", row["revised_arrays"])
            _blob, historical_bytes = _blob_bytes(repo, REVISION_COMMIT, _revision_relpath(scenario, time_step))
            with np.load(io.BytesIO(historical_bytes)) as historical:
                historical_arrays = {key: np.asarray(historical[key]) for key in historical.files}
            equal, mismatch = _same_arrays(row["revised_arrays"], historical_arrays)
            if not equal:
                raise RuntimeError(f"revised state mismatch at {scenario} t={time_step}: {mismatch}")
            before_graph = int(consumption["environment_steps"])
            graph = _graph_from_arrays(row["revised_arrays"])
            if int(consumption["environment_steps"]) != before_graph:
                raise RuntimeError("graph construction stepped the environment")
            if int(np.asarray(row["revised_arrays"]["time_index"]).reshape(-1)[0]) != time_step:
                raise RuntimeError("graph source time does not match the decision")
            with (target / "graph.npz").open("wb") as handle:
                np.savez(handle, **graph)
            _write_json(
                target / "state_meta.json",
                {
                    "scenario_id": int(scenario),
                    "time_step": time_step,
                    "state_hash": row["state_hash"],
                    "manifest_checksum_candidates": row["manifest_checksum_candidates"],
                    "manifest_candidate_count": int(row["manifest_candidate_count"]),
                    "manifest_illegal": int(row["manifest_illegal"]),
                    "tol_bits": float(row["tol_bits"]),
                    "ledger_gap_bits": float(row["ledger_gap_bits"]),
                    "velocity_predecessor": row["velocity_predecessor"],
                    "predecessor_time_step": row["predecessor_time_step"],
                    "graph_keys": list(GRAPH_KEYS),
                    "future_label_fields_in_graph": False,
                    "revision_matches_historical_source": True,
                },
            )
            saved.append(time_step)
        if saved != [15, 30]:
            raise RuntimeError("t15 and t30 snapshots were not both saved")
        _write_json(complete, {"scenario_id": int(scenario), "environment_steps": delta, "states": saved})
        print(f"replay {scenario} steps={delta}", flush=True)


def _load_state(output: Path, scenario: int, time_step: int) -> dict:
    folder = _state_dir(output, scenario, time_step)
    meta = json.loads((folder / "state_meta.json").read_text(encoding="utf-8"))
    scalars = json.loads((folder / "env_scalars.json").read_text(encoding="utf-8"))
    return {
        "scenario_id": int(scenario),
        "time_step": int(time_step),
        "env_snap": _combine_snap(_load_arrays(folder / "env_snap.npz"), scalars),
        "ledger_snap": json.loads((folder / "ledger.json").read_text(encoding="utf-8")),
        "graph": {key: _load_arrays(folder / "graph.npz")[key] for key in GRAPH_KEYS},
        "state_hash": meta["state_hash"],
        "manifest_checksum_candidates": meta["manifest_checksum_candidates"],
        "manifest_candidate_count": meta["manifest_candidate_count"],
        "manifest_illegal": meta["manifest_illegal"],
        "tol_bits": float(meta["tol_bits"]),
    }


def _select_all(output: Path, consumption: dict) -> list[dict]:
    from wt.research.urgency_arrival_aware_expansion.sampling import select_state

    target = output / "candidates.json"
    if target.is_file():
        return json.loads(target.read_text(encoding="utf-8"))["states"]
    states = []
    before = int(consumption["environment_steps"])
    for scenario in COMPARISON_SCENARIOS:
        for time_step in COMPARISON_TIMES:
            state = _load_state(output, scenario, time_step)
            rows, meta = select_state(int(scenario), state)
            if meta["sampling_source"] != "stored_j15" or len(rows) != 128:
                raise RuntimeError("candidate subset did not stay on the stored J15 128-candidate rule")
            identities = [int(row["candidate_id"]) for row in rows]
            if len(set(identities)) != 128 or identities == list(range(128)):
                raise RuntimeError("candidate identity is not distinct from the local position")
            packed = []
            for local, row in enumerate(rows):
                if int(row["time_step"]) != int(time_step) or str(row["state_hash"]) != str(state["state_hash"]):
                    raise RuntimeError("candidate row does not match the frozen decision")
                if row["assignment"].count("|") != 5:
                    raise RuntimeError("assignment does not describe six UAVs")
                packed.append(
                    {
                        "local_position": local,
                        "scenario_id": int(scenario),
                        "time_step": int(time_step),
                        "candidate_id": int(row["candidate_id"]),
                        "row_index": int(row["row_index"]),
                        "assignment": row["assignment"],
                        "pool_hash": row["pool_hash"],
                        "subset_hash": row["subset_hash"],
                        "state_hash": row["state_hash"],
                        "selection": row["selection"],
                        "sampling_source": row["sampling_source"],
                        "seed": 605,
                    }
                )
            states.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(time_step),
                    "pool_hash": meta["pool_hash"],
                    "subset_hash": meta["subset_hash"],
                    "state_hash": state["state_hash"],
                    "pool_count": int(meta["pool_count"]),
                    "rows": packed,
                }
            )
            print(f"candidates {scenario} t={time_step}", flush=True)
    if int(consumption["environment_steps"]) != before:
        raise RuntimeError("candidate selection stepped the environment")
    _write_json(target, {"states": states, "seed": 605, "rule": "stored J15 top 32 plus rank-stratified 96"})
    return states


def _score_comparison(output: Path, models: dict, decoder, device: torch.device, consumption: dict, states: list[dict]) -> None:
    if (output / "freeze.json").is_file():
        consumption["scoring_forwards"] = max(int(consumption["scoring_forwards"]), PLANNED_FORWARDS)
        return
    if (output / "raw_scores.npz").is_file():
        raise RuntimeError("scoring output is partial; refusing to score again")
    before_steps = int(consumption["environment_steps"])
    before_low = int(consumption["low_level_inferences"])
    selections = []
    score_rows = {}
    for state in states:
        graph = _load_state(output, int(state["scenario_id"]), int(state["time_step"]))["graph"]
        assignments = np.stack([[int(part) for part in row["assignment"].split("|")] for row in state["rows"]])
        identities = np.asarray([int(row["candidate_id"]) for row in state["rows"]], dtype=np.int64)
        texts = [row["assignment"] for row in state["rows"]]
        per_model = {}
        for name in MODEL_ORDER:
            scores = _score_tensor(models[name], decoder, graph, assignments, device, consumption).detach().cpu().numpy()
            score_rows[f"{name}_{int(state['scenario_id'])}_t{int(state['time_step']):03d}"] = np.asarray(scores, dtype=np.float32)
            per_model[name] = _selection(scores, identities, texts, None)
        selections.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "state_hash": state["state_hash"],
                "pool_hash": state["pool_hash"],
                "subset_hash": state["subset_hash"],
                "models": per_model,
            }
        )
        print(f"scored {state['scenario_id']} t={state['time_step']}", flush=True)
    if int(consumption["environment_steps"]) != before_steps or int(consumption["low_level_inferences"]) != before_low:
        raise RuntimeError("high-level scoring changed the environment counters")
    score_path = output / "raw_scores.npz"
    with score_path.open("wb") as handle:
        np.savez(handle, **score_rows)
    _write_json(output / "selections.json", {"states": selections, "labels_available": False})
    _write_json(
        output / "freeze.json",
        {
            "r28_sha256": _sha256(output / "r28_u120.pt"),
            "raw_scores_sha256": _sha256(score_path),
            "selections_sha256": _sha256(output / "selections.json"),
            "candidates_sha256": _sha256(output / "candidates.json"),
            "labels_written": False,
            "new_real_arrival_labels_revealed": False,
        },
    )
    _write_json(output / "freeze_sha256.json", {"freeze_sha256": _sha256(output / "freeze.json")})


def _assignment_vector(text: str) -> np.ndarray:
    return np.asarray([int(part) for part in text.split("|")], dtype=np.int64)


def _label_all(output: Path, consumption: dict, states: list[dict]) -> None:
    from wt.research.urgency_arrival_aware_expansion.labels import score_label
    from wt.data.highlevel_graphs.jobs import _frozen_actor

    freeze = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    if freeze["labels_written"] or freeze["new_real_arrival_labels_revealed"]:
        raise RuntimeError("freeze record was written after labels")
    labels_path = output / "real_labels.jsonl"
    done = set()
    if labels_path.is_file():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((int(row["scenario_id"]), int(row["time_step"]), int(row["candidate_id"])))
    actor = _frozen_actor()
    for state in states:
        folder = _state_dir(output, int(state["scenario_id"]), int(state["time_step"]))
        stamp = folder / "labels_started.json"
        complete = folder / "labels_complete.json"
        if complete.is_file():
            continue
        finished = [row for row in state["rows"] if (int(state["scenario_id"]), int(state["time_step"]), int(row["candidate_id"])) in done]
        if stamp.is_file() and not finished:
            raise RuntimeError(f"label rollout already started for {state['scenario_id']} t={state['time_step']}")
        _write_json(stamp, {"scenario_id": int(state["scenario_id"]), "time_step": int(state["time_step"])})
        live = _load_state(output, int(state["scenario_id"]), int(state["time_step"]))
        for row in state["rows"]:
            key = (int(state["scenario_id"]), int(state["time_step"]), int(row["candidate_id"]))
            if key in done:
                continue
            if int(consumption["environment_steps"]) + 15 > ENVIRONMENT_CAP:
                raise RuntimeError("environment-step cap reached")
            result = score_label(
                live["env_snap"],
                live["ledger_snap"],
                actor,
                _assignment_vector(row["assignment"]),
                scenario_id=int(state["scenario_id"]),
                zero_arrivals=False,
                horizon=15,
            )
            if int(result["steps"]) != 15:
                raise RuntimeError("restored label did not run 15 steps")
            served_total = float(np.sum(np.asarray(result["served_bits"], dtype=np.float64)))
            gap = abs(float(result["payload_bits"]) - served_total)
            if gap > float(live["tol_bits"]) or not np.isfinite(float(result["payload_bits"])):
                raise RuntimeError("served total does not match payload within the environment tolerance")
            record = {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "candidate_id": int(row["candidate_id"]),
                "row_index": int(row["row_index"]),
                "assignment": row["assignment"],
                "real_payload_bits": float(result["payload_bits"]),
                "served_total_bits": served_total,
                "payload_served_gap_bits": float(result["payload_served_gap_bits"]),
                "steps": int(result["steps"]),
                "zero_arrivals": False,
                "horizon": 15,
                "accounting_error_bits": float(result["accounting_error_bits"]),
            }
            with labels_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record) + "\n")
            consumption["candidate_rollouts"] += 1
            done.add(key)
        _save_consumption(output / "consumption.json", consumption)
        _write_json(complete, {"scenario_id": int(state["scenario_id"]), "time_step": int(state["time_step"]), "rollouts": 128})
        print(f"labels {state['scenario_id']} t={state['time_step']} env_steps={consumption['environment_steps']} low_level={consumption['low_level_inferences']}", flush=True)
        gc.collect()


def _labels_by_key(output: Path) -> dict[tuple[int, int, int, str], dict]:
    found = {}
    for line in (output / "real_labels.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[(int(row["scenario_id"]), int(row["time_step"]), int(row["candidate_id"]), str(row["assignment"]))] = row
    return found


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize(output: Path) -> dict:
    candidates = json.loads((output / "candidates.json").read_text(encoding="utf-8"))["states"]
    selections = {(int(row["scenario_id"]), int(row["time_step"])): row for row in json.loads((output / "selections.json").read_text(encoding="utf-8"))["states"]}
    labels = _labels_by_key(output)
    with np.load(output / "raw_scores.npz") as stored:
        score_map = {key: np.asarray(stored[key]) for key in stored.files}
    per_state = []
    for state in candidates:
        key = (int(state["scenario_id"]), int(state["time_step"]))
        payload = []
        texts = []
        for row in state["rows"]:
            label = labels[(key[0], key[1], int(row["candidate_id"]), str(row["assignment"]))]
            payload.append(float(label["real_payload_bits"]))
            texts.append(row["assignment"])
        payload_array = np.asarray(payload, dtype=np.float64)
        identities = np.asarray([int(row["candidate_id"]) for row in state["rows"]], dtype=np.int64)
        models = {}
        for name in MODEL_ORDER:
            vector = score_map[f"{name}_{key[0]}_t{key[1]:03d}"]
            chosen = _selection(vector, identities, texts, payload_array)
            saved = selections[key]["models"][name]
            if int(saved["candidate_id"]) != int(chosen["candidate_id"]) or saved["assignment"] != chosen["assignment"]:
                raise RuntimeError("selection changed after the real labels were revealed")
            labels_tensor = torch.as_tensor(payload_array, dtype=torch.float64)
            chosen["soft_expected_regret"] = float(expected_regret_loss(torch.as_tensor(vector, dtype=torch.float64), labels_tensor))
            models[name] = chosen
        per_state.append(
            {
                "scenario_id": key[0],
                "time_step": key[1],
                "state_hash": state["state_hash"],
                "pool_hash": state["pool_hash"],
                "subset_hash": state["subset_hash"],
                "subset_optimum_bits": float(np.max(payload_array)),
                "models": models,
                "R28_minus_R12_payload_bits": float(models["R28_U120"]["real_payload_bits"] - models["R12_U120"]["real_payload_bits"]),
                "R28_minus_C_payload_bits": float(models["R28_U120"]["real_payload_bits"] - models["C_LISTNET"]["real_payload_bits"]),
                "R28_minus_E6_payload_bits": float(models["R28_U120"]["real_payload_bits"] - models["E6"]["real_payload_bits"]),
                "R28_minus_L3A10_payload_bits": float(models["R28_U120"]["real_payload_bits"] - models["L3A10"]["real_payload_bits"]),
            }
        )
    per_state.sort(key=lambda row: (row["scenario_id"], row["time_step"]))

    def pack(rows: list[dict]) -> dict:
        return {
            "states": len(rows),
            "R28_minus_R12_payload_bits_mean": _mean([row["R28_minus_R12_payload_bits"] for row in rows]),
            "R28_minus_C_payload_bits_mean": _mean([row["R28_minus_C_payload_bits"] for row in rows]),
            "R28_minus_E6_payload_bits_mean": _mean([row["R28_minus_E6_payload_bits"] for row in rows]),
            "R28_minus_L3A10_payload_bits_mean": _mean([row["R28_minus_L3A10_payload_bits"] for row in rows]),
            "payload_bits_mean": {name: _mean([row["models"][name]["real_payload_bits"] for row in rows]) for name in MODEL_ORDER},
            "subset_regret_bits_mean": {name: _mean([row["models"][name]["subset_regret_bits"] for row in rows]) for name in MODEL_ORDER},
            "soft_expected_regret_mean": {name: _mean([row["models"][name]["soft_expected_regret"] for row in rows]) for name in MODEL_ORDER},
        }

    scenarios = []
    for scenario in COMPARISON_SCENARIOS:
        rows = [row for row in per_state if int(row["scenario_id"]) == int(scenario)]
        if [int(row["time_step"]) for row in rows] != [15, 30]:
            raise RuntimeError(f"scenario {scenario} does not have both decision times")
        difference = _mean([row["R28_minus_R12_payload_bits"] for row in rows])
        scenarios.append(
            {
                "scenario_id": int(scenario),
                "t15_R28_minus_R12_payload_bits": float(rows[0]["R28_minus_R12_payload_bits"]),
                "t30_R28_minus_R12_payload_bits": float(rows[1]["R28_minus_R12_payload_bits"]),
                "mean_R28_minus_R12_payload_bits": difference,
            }
        )
    overall = pack(per_state)
    t15 = pack([row for row in per_state if int(row["time_step"]) == 15])
    t30 = pack([row for row in per_state if int(row["time_step"]) == 30])

    def margin(block: dict, key: str) -> bool:
        return float(block["overall"][key]) > 1.0 and float(block["t15"][key]) >= -1.0 and float(block["t30"][key]) >= -1.0

    blocks = {"overall": overall, "t15": t15, "t30": t30}
    against_r12 = margin(blocks, "R28_minus_R12_payload_bits_mean")
    against_c = margin(blocks, "R28_minus_C_payload_bits_mean")
    references = (
        overall["payload_bits_mean"]["R28_U120"] >= overall["payload_bits_mean"]["E6"] - 1.0
        and overall["payload_bits_mean"]["R28_U120"] >= overall["payload_bits_mean"]["L3A10"] - 1.0
    )
    gate = "CONTINUE_TO_CLOSED_LOOP_COMPARISON" if against_r12 and against_c and references else "NOT_PROMOTED"
    counts = {
        "positive": int(sum(row["mean_R28_minus_R12_payload_bits"] > NEAR_ZERO_BITS for row in scenarios)),
        "negative": int(sum(row["mean_R28_minus_R12_payload_bits"] < -NEAR_ZERO_BITS for row in scenarios)),
        "near_zero": int(sum(abs(row["mean_R28_minus_R12_payload_bits"]) <= NEAR_ZERO_BITS for row in scenarios)),
        "near_zero_abs_bits": NEAR_ZERO_BITS,
    }
    losses = [float(json.loads(line)["loss"]) for line in (output / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "per_state": per_state,
        "overall": overall,
        "t15": t15,
        "t30": t30,
        "scenarios": scenarios,
        "scenario_counts": counts,
        "train_loss_mean": _mean(losses),
        "gate": {
            "label": gate,
            "r28_minus_r12_margin": bool(against_r12),
            "r28_minus_c_margin": bool(against_c),
            "not_below_references_by_more_than_1_bit": bool(references),
            "starts_next_stage": False,
            "adopts_model": False,
            "rewrites_l8": False,
        },
    }


def _fmt(value: float) -> str:
    return format(float(value), ".12g")


def _mbit(value: float) -> str:
    return format(float(value) / 1.0e6, ".12g")


def _write_report(repo: Path, summary: dict, consumption: dict, status: str, note: str) -> None:
    lines = ["# L9 固定预算的数据覆盖对照", ""]
    if status == "complete":
        gate = summary["gate"]["label"]
        overall = summary["overall"]
        lines.extend(
            [
                f"结论是 `{gate}`。继续门槛{('成立' if gate == 'CONTINUE_TO_CLOSED_LOOP_COMPARISON' else '没有成立')}。",
                "",
                f"主对照 R28−R12 在 16 个新状态上的 argmax 真实有效载荷平均差为 {_fmt(overall['R28_minus_R12_payload_bits_mean'])} bit（{_mbit(overall['R28_minus_R12_payload_bits_mean'])} Mbit）。t15 为 {_fmt(summary['t15']['R28_minus_R12_payload_bits_mean'])} bit，t30 为 {_fmt(summary['t30']['R28_minus_R12_payload_bits_mean'])} bit。",
                "",
                f"R28−ListNet 总体平均差为 {_fmt(overall['R28_minus_C_payload_bits_mean'])} bit，t15 为 {_fmt(summary['t15']['R28_minus_C_payload_bits_mean'])} bit，t30 为 {_fmt(summary['t30']['R28_minus_C_payload_bits_mean'])} bit。相对 R12 的门槛为 {summary['gate']['r28_minus_r12_margin']}，相对 ListNet 的门槛为 {summary['gate']['r28_minus_c_margin']}，不低于 E6 和 L3A10 的容差为 {summary['gate']['not_below_references_by_more_than_1_bit']}。门槛成立也只决定以后是否另写闭环对照，不执行闭环，不采纳新模型，也不改写 L8 的 NOT_PROMOTED。",
                "",
                "## 主要指标",
                "",
                "| 范围 | R28−R12 (bit) | R28−C (bit) | R28 (Mbit) | R12 (Mbit) | C (Mbit) | E6 (Mbit) | L3A10 (Mbit) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, block in (("16 个状态", summary["overall"]), ("t15", summary["t15"]), ("t30", summary["t30"])):
            lines.append(
                "| {name} | {r12} | {listnet} | {r28} | {base} | {c} | {e6} | {a10} |".format(
                    name=name,
                    r12=_fmt(block["R28_minus_R12_payload_bits_mean"]),
                    listnet=_fmt(block["R28_minus_C_payload_bits_mean"]),
                    r28=_mbit(block["payload_bits_mean"]["R28_U120"]),
                    base=_mbit(block["payload_bits_mean"]["R12_U120"]),
                    c=_mbit(block["payload_bits_mean"]["C_LISTNET"]),
                    e6=_mbit(block["payload_bits_mean"]["E6"]),
                    a10=_mbit(block["payload_bits_mean"]["L3A10"]),
                )
            )
        lines.extend(
            [
                "",
                f"R28−E6 总体平均差 {_fmt(overall['R28_minus_E6_payload_bits_mean'])} bit。R28−L3A10 总体平均差 {_fmt(overall['R28_minus_L3A10_payload_bits_mean'])} bit。128 候选内部的平均子集 regret：R28 {_fmt(overall['subset_regret_bits_mean']['R28_U120'])} bit，R12 {_fmt(overall['subset_regret_bits_mean']['R12_U120'])} bit，C {_fmt(overall['subset_regret_bits_mean']['C_LISTNET'])} bit。",
                "",
                "八个场景先对 t15 和 t30 取平均，再统计 R28−R12。绝对差不超过 1 bit 记为接近零。",
                "",
                "| 场景 | t15 R28−R12 (bit) | t30 R28−R12 (bit) | 两次平均 (bit) |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary["scenarios"]:
            lines.append(
                f"| {row['scenario_id']} | {_fmt(row['t15_R28_minus_R12_payload_bits'])} | {_fmt(row['t30_R28_minus_R12_payload_bits'])} | {_fmt(row['mean_R28_minus_R12_payload_bits'])} |"
            )
        counts = summary["scenario_counts"]
        lines.extend(
            [
                "",
                f"场景计数：正差 {counts['positive']}，负差 {counts['negative']}，接近零 {counts['near_zero']}。",
                "",
                f"训练损失均值 {_fmt(summary['train_loss_mean'])}，16 状态上 R28 的平均软期望 regret {_fmt(overall['soft_expected_regret_mean']['R28_U120'])}。二者只作辅助，没有用来选择检查点。",
                "",
            ]
        )
    else:
        lines.extend([f"结论是 `INCOMPLETE`。{note}", ""])
    lines.extend(
        [
            "## 实际消耗",
            "",
            f"- 参数更新：{consumption['training_updates']}",
            f"- 反向传播：{consumption['backward_passes']}",
            f"- 环境步：{consumption['environment_steps']}",
            f"- 低层推理：{consumption['low_level_inferences']}",
            f"- 高层评分前向：{consumption['scoring_forwards']}",
            f"- 新增候选标签回合：{consumption['candidate_rollouts']}",
            f"- 对照和参考模型重新训练：{consumption['retrained_reference_models']}",
            f"- 旧 28 状态重放、重新标注和模型前向：{consumption['old_state_environment_steps']}、{consumption['old_state_relabels']}、{consumption['old_state_scoring_forwards']}",
            "",
            "低层推理由 MAPPO 调用单独计数，不从环境步复制。预算是 120 次更新和反向传播、240 步新轨迹、2048 个 15 步标签、合计不超过 30,960 个环境步，以及计划 200、上限 204 次高层评分前向。",
            "",
            "## 证据与限制",
            "",
            "R12 直接复用 L8 的 R10，每状态训练 10 次。R28 从同一个 E6 和空 Adam 重新训练 120 次，覆盖排序后的 28 个状态：前 8 个状态各访问 5 次，其余 20 个各访问 4 次。更新预算相同，每个样本的访问次数不同。",
            "",
            "原 12 个状态来自 L2，新增 16 个状态就是 L8 的比较状态。它们现在是训练数据，图、候选和真实标签都读取原件，不能再当作 L9 的验证证据。新比较是 322032–322039 的 t15 和 t30。这些场景仍在历史训练池内，E6 可能见过零到达数据，因此这不是独立验证、未见泛化或正式留出集。",
            "",
            note,
            "",
            "lambda_H 与 delta_payload_max 仍未设置。本结果不写入论文，也不进入正式命令入口。正式分支与海洋观测隔离分支没有修改。",
            "",
        ]
    )
    (repo / "docs/research/Legacy_System_L9_Report_CN.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _fail(output: Path, repo: Path, consumption: dict, exc: Exception) -> None:
    _save_consumption(output / "consumption.json", consumption)
    _write_json(output / "failure.json", {"error": f"{type(exc).__name__}: {exc}", **consumption})
    _write_report(repo, {}, consumption, "incomplete", f"执行在生成完整对照前停止：{type(exc).__name__}: {exc}")


def run() -> None:
    repo = _repo()
    output = _output(repo)
    if (output / "manifest.json").is_file():
        raise RuntimeError("L9 output already exists")
    output.mkdir(parents=True, exist_ok=True)
    consumption = _load_consumption(output / "consumption.json")
    originals = None
    try:
        branch = _git_text(repo, ["rev-parse", "--abbrev-ref", "HEAD"])
        if branch != BRANCH or _git(repo, ["merge-base", "--is-ancestor", INSTRUCTION_COMMIT, "HEAD"]).returncode != 0:
            raise RuntimeError("L9 is not running on the instruction commit")
        hashes = _hash_paths(_extra_paths(repo))
        for name, expected in _expected_hashes().items():
            if hashes[name] != expected:
                raise RuntimeError(f"{name} hash is {hashes[name]}")
        sources = _preflight_sources(repo)
        states, index = load_training_states(repo)
        schedule = access_schedule(states)
        if not (output / "source_index.json").is_file():
            _write_json(output / "source_index.json", index)
        if not (output / "access_schedule.json").is_file():
            _write_json(output / "access_schedule.json", schedule)
        elif json.loads((output / "access_schedule.json").read_text(encoding="utf-8"))["visits"] != schedule["visits"]:
            raise RuntimeError("saved access schedule does not match the fixed rule")
        _write_json(
            output / "source_manifest.json",
            {
                "instruction_commit": INSTRUCTION_COMMIT,
                "command": "py -3 -m wt.research.arrival_coverage --run",
                "historical_objects": sources,
                "protected_sha256": hashes,
            },
        )
        trained = (output / "r28_u120.pt").is_file() and (output / "train_log.jsonl").is_file()
        if not trained and (int(consumption["environment_steps"]) != 0 or int(consumption["scoring_forwards"]) != 0):
            raise RuntimeError("environment steps or scoring forwards were consumed before training finished")
        _configure()
        device = torch.device("cpu")
        originals = _install_l8_counters(consumption)
        model_r28, decoder = _train(repo, output, states, device, consumption)
        if int(consumption["environment_steps"]) != 0 or int(consumption["low_level_inferences"]) != 0:
            raise RuntimeError("training or the reused 28 states consumed environment steps")
        _save_consumption(output / "consumption.json", consumption)
        _write_json(output / "r28_freeze.json", {"sha256": _sha256(output / "r28_u120.pt"), "new_comparison_labels_revealed": False, "updates": 120, "full_passes": 4, "tail_updates": 8})
        _replay_all(repo, output, consumption)
        if int(consumption["environment_steps"]) != REPLAY_STEPS:
            raise RuntimeError(f"replay consumed {consumption['environment_steps']} steps")
        comparison_states = _select_all(output, consumption)
        models = {
            "R28_U120": model_r28,
            "R12_U120": _load_named(repo / "results/research/payload_objective/r_epoch_010.pt", device, 10),
            "C_LISTNET": _load_named(repo / "results/research/arrival_aware/listnet_comparison/arm_b_epoch_010.pt", device, 10),
            "E6": _load_named(repo / "checkpoints/initialization/joint_utility_epoch6.pt", device, 6),
            "L3A10": _load_named(repo / "results/research/arrival_aware/listnet_comparison/arm_a_epoch_010.pt", device, 10),
        }
        for model in models.values():
            model.eval()
        decoder.eval()
        with torch.no_grad():
            _score_comparison(output, models, decoder, device, consumption, comparison_states)
        _save_consumption(output / "consumption.json", consumption)
        freeze_hash = _sha256(output / "freeze.json")
        _label_all(output, consumption, comparison_states)
        if _sha256(output / "freeze.json") != freeze_hash:
            raise RuntimeError("freeze record changed after labels")
        if int(consumption["environment_steps"]) != ENVIRONMENT_CAP or int(consumption["candidate_rollouts"]) != 2048:
            raise RuntimeError("environment consumption does not match the fixed budget")
        if int(consumption["training_updates"]) != UPDATES or int(consumption["backward_passes"]) != UPDATES:
            raise RuntimeError("training consumption does not match 120 updates")
        if int(consumption["scoring_forwards"]) != PLANNED_FORWARDS:
            raise RuntimeError(f"scoring forwards are {consumption['scoring_forwards']}")
        summary = _summarize(output)
        after = _hash_paths(_extra_paths(repo))
        if after != hashes:
            changed = sorted(key for key in hashes if hashes[key] != after[key])
            raise RuntimeError(f"protected files changed: {changed}")
        index_after = load_training_states(repo)[1]
        if index_after["l8_labels_sha256"] != index["l8_labels_sha256"] or index_after["l2_manifest_sha256"] != index["l2_manifest_sha256"]:
            raise RuntimeError("reused training inputs changed")
        ocean = _git(repo, ["rev-parse", OCEAN_BRANCH])
        formal = _git(repo, ["cat-file", "-t", FORMAL_EXPECTED])
        if formal.returncode == 0 and formal.stdout.decode().strip() != "commit":
            raise RuntimeError("formal baseline object is not a commit")
        if ocean.returncode == 0 and ocean.stdout.decode().strip() != OCEAN_EXPECTED:
            raise RuntimeError("ocean branch changed")
        _write_json(output / "paired_detail.json", summary)
        _write_json(output / "paired_summary.json", {key: value for key, value in summary.items() if key != "per_state"})
        failure_path = output / "failure.json"
        if failure_path.is_file():
            failure_path.unlink()
        _save_consumption(output / "consumption.json", consumption)
        _write_json(
            output / "manifest.json",
            {
                "package": "legacy_arrival_coverage_l9",
                "instruction_commit": INSTRUCTION_COMMIT,
                "command": "py -3 -m wt.research.arrival_coverage --run",
                "gate": summary["gate"],
                "primary_R28_minus_R12_payload_bits_mean": summary["overall"]["R28_minus_R12_payload_bits_mean"],
                "r28_sha256": _sha256(output / "r28_u120.pt"),
                "freeze_sha256": freeze_hash,
                "protected_unchanged": True,
                "low_level_counted_separately": True,
                **consumption,
            },
        )
        _write_report(
            repo,
            summary,
            consumption,
            "complete",
            "必要检查包括 28 状态和 3584 条标签的身份、本地顺序、图字段、有限值与范围不相交，以及初始化、空 Adam、冻结 decoder 和 120 次更新。新流程核对了 30 步重放、候选身份、当前观测和 served 容差。完成后复核了正式权重、decoder、原始输入、旧模型和正式策略的哈希。",
        )
        print(json.dumps({"gate": summary["gate"]["label"], "low_level_inferences": consumption["low_level_inferences"], "environment_steps": consumption["environment_steps"], "scoring_forwards": consumption["scoring_forwards"]}, ensure_ascii=False), flush=True)
    except Exception as exc:
        _fail(output, repo, consumption, exc)
        raise
    finally:
        if originals is not None:
            _restore_counters(originals)