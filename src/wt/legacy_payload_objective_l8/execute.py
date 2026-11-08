"""Train only the expected-regret arm, then score one fixed 16-state comparison."""
from __future__ import annotations

import gc
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.protocol import FROZEN_SHARD_COMMIT, frozen_relpath
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import MAPPO_SHA256, SCHEME_C_SHA256
from wt.benchmark_v2.phase6id_joint_training.protocol import GRAD_CLIP
from wt.benchmark_v2.phase6is_assignment_head.protocol import SCHEME_C_CKPT
from wt.benchmark_v2.phase6i_urgency_arrival_aware_expansion.protocol import LABEL_SEED, MAX_CANDIDATES
from wt.htw_phase9.const import BUOY_DIM, EDGE_UB_DIM
from wt.legacy_arrival_aware_ab.loader import TRAIN_SCENARIOS, load_package
from wt.legacy_payload_objective_l8.loss import check_loss, expected_regret_loss
from wt.lowlevel_v2.phase5k_joint_decoder.decoders import JointAssignmentDecoder
from wt.lowlevel_v2.phase5l_joint_encoder.model import JointUtilityHGNN, append_queue_edge, batch_tensors
from wt.lowlevel_v2.phase5l_joint_encoder.protocol import LEARNING_RATE, WEIGHT_DECAY


EPOCHS = 10
SEED = 0
UPDATES = 120
TRAIN_FORWARD_EXTRA = 12
COMPARISON_SCENARIOS = tuple(range(322024, 322032))
COMPARISON_TIMES = (15, 30)
REPLAY_STEPS = 8 * 30
LABEL_STEPS = 16 * 128 * 15
ENVIRONMENT_CAP = REPLAY_STEPS + LABEL_STEPS
FORWARD_CAP = 200
NEAR_ZERO_BITS = 1.0
REVISION_COMMIT = "8a34efc718665a7cfe4375b662e8f5249263f03d"
INSTRUCTION_COMMIT = "7c31d69894d2617d259a5b2fa5015ab5fa8f57c1"
OCEAN_BRANCH = "research/ocean-observation-isolated"
OCEAN_EXPECTED = "4d8622ef183e734f2d9a890d60d735ed716f5d74"
BRANCH = "feature/legacy-system-maintenance"
HASHES = {
    "epoch6": "7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde",
    "decoder": "de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5",
    "c_listnet": "6024f4c42cdf2de2e8452b49da6ed1824ff8d6e56e84204681a6a37161b81141",
    "l3_a10": "87f7c7673555518d80303a8226ae699057bcdaeacb4307512331b7dbad4e60d8",
    "mappo": MAPPO_SHA256,
    "scheme_c": SCHEME_C_SHA256,
}
GRAPH_KEYS = ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu")
MODEL_ORDER = ("R_EXPECTED_REGRET", "C_LISTNET", "E6", "L3A10")


def _repo() -> Path:
    return Path(__file__).resolve().parents[3]


def _output(repo: Path) -> Path:
    return repo / "results" / "legacy_payload_objective_l8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _git(repo: Path, args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=repo, capture_output=True, check=False)


def _git_text(repo: Path, args: list[str]) -> str:
    completed = _git(repo, args)
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"git {' '.join(args)} failed")
    return completed.stdout.decode("utf-8").strip()


def _consumption() -> dict:
    return {
        "environment_steps": 0,
        "low_level_inferences": 0,
        "scoring_forwards": 0,
        "training_updates": 0,
        "backward_passes": 0,
        "candidate_rollouts": 0,
        "retrained_reference_models": 0,
    }


def _load_consumption(path: Path) -> dict:
    if not path.is_file():
        return _consumption()
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = _consumption()
    values.update({key: int(payload[key]) for key in values if key in payload})
    return values


def _save_consumption(path: Path, consumption: dict) -> None:
    _write_json(path, consumption)


def _configure() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _weights(module: torch.nn.Module) -> list[tuple[str, bytes]]:
    return [(name, tensor.detach().cpu().contiguous().numpy().tobytes()) for name, tensor in module.state_dict().items()]


def _protected_paths(repo: Path) -> dict[str, Path]:
    return {
        "epoch6": repo / "results/phase6id_joint_utility_training/epoch_006.pt",
        "decoder": repo / "results/phase5k_joint_decoder/matching_decoder.pt",
        "c_listnet": repo / "results/legacy_arrival_aware_ab_l3/arm_b_epoch_010.pt",
        "l3_a10": repo / "results/legacy_arrival_aware_ab_l3/arm_a_epoch_010.pt",
        "mappo": primary_checkpoint(),
        "scheme_c": SCHEME_C_CKPT,
        "official_policy": repo / "results/phase6iaa_formal_planner/active_policy.json",
        "l2_manifest": repo / "results/legacy_arrival_aware_ab_l2/manifest.json",
        "historical_listnet": repo / "src/wt/benchmark_v2/phase6if_ranking_loss/losses.py",
    }


def _hash_protected(repo: Path) -> dict[str, str]:
    return {name: _sha256(path) for name, path in _protected_paths(repo).items()}


def _encoder(payload: dict, device: torch.device) -> JointUtilityHGNN:
    model = JointUtilityHGNN(buoy_dim=int(BUOY_DIM) + 1)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    return model


def _decoder(path: Path, device: torch.device) -> JointAssignmentDecoder:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    decoder = JointAssignmentDecoder("matching")
    decoder.load_state_dict(payload["state_dict"], strict=True)
    decoder.to(device)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder


def _fresh_r(epoch_path: Path, decoder_path: Path, device: torch.device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    payload = torch.load(epoch_path, map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != 6:
        raise RuntimeError(f"E6 epoch field is {payload.get('epoch')}")
    model = _encoder(payload, device)
    decoder = _decoder(decoder_path, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(LEARNING_RATE), weight_decay=float(WEIGHT_DECAY))
    if optimizer.state:
        raise RuntimeError("Adam state was not empty")
    if any(parameter.requires_grad for parameter in decoder.parameters()):
        raise RuntimeError("decoder parameter is trainable")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("encoder parameter is frozen")
    return model, decoder, optimizer


def _load_named(path: Path, device: torch.device, expected_epoch: int) -> JointUtilityHGNN:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != int(expected_epoch):
        raise RuntimeError(f"{path.name} epoch is {payload.get('epoch')}")
    model = _encoder(payload, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _score(model, decoder, graph: dict, assignments: np.ndarray, device: torch.device, consumption: dict) -> np.ndarray:
    if consumption["scoring_forwards"] >= FORWARD_CAP:
        raise RuntimeError("high-level scoring forward cap reached")
    forbidden = [key for key in graph if key not in GRAPH_KEYS]
    if forbidden or any(key in graph for key in ("real_payload", "zero_payload", "scenario_id")):
        raise RuntimeError(f"graph contains fields outside the revised input: {forbidden}")
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
    if values.shape != identities.shape or not np.all(np.isfinite(values)):
        raise RuntimeError("scores are not a finite 128-vector")
    tied = np.flatnonzero(values == np.max(values))
    return int(tied[int(np.argmin(identities[tied]))])


def _active_count(assignment: str) -> int:
    return int(sum(int(part) >= 0 for part in str(assignment).split("|")))


def _graph_from_arrays(arrays: dict) -> dict:
    config = L3BBitQueueEnvV2().config
    from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import hgnn_tensors

    built = hgnn_tensors(arrays, config)
    edge = np.asarray(built["edge"], dtype=np.float32)
    if int(edge.shape[-1]) != int(EDGE_UB_DIM) + 1:
        raise RuntimeError(f"revised edge width is {edge.shape[-1]}")
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
        raise RuntimeError("graph keys changed")
    conditioned = append_queue_edge(graph)
    if not np.allclose(conditioned["edge"], edge[None, ...], atol=1.0e-6, rtol=0.0):
        raise RuntimeError("queue-edge append does not reproduce the revised edge")
    if int(np.asarray(arrays["time_index"]).reshape(-1)[0]) < 0:
        raise RuntimeError("decision time is missing")
    return graph


def _revision_relpath(scenario: int, time_step: int) -> str:
    return (
        "src/wt/benchmark_v2/phase6ib_highlevel_dataset/revision_hgnn_input/dataset/train/"
        f"scenario_{int(scenario)}/t{int(time_step):03d}_state.npz"
    )


def _blob_bytes(repo: Path, commit: str, relpath: str) -> tuple[str, bytes]:
    spec = f"{commit}:{relpath}"
    blob = _git_text(repo, ["rev-parse", spec])
    completed = _git(repo, ["cat-file", "blob", spec])
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"missing historical blob {spec}: {message}")
    return blob, bytes(completed.stdout)


def _preflight(repo: Path) -> dict:
    branch = _git_text(repo, ["rev-parse", "--abbrev-ref", "HEAD"])
    head = _git_text(repo, ["rev-parse", "HEAD"])
    if branch != BRANCH:
        raise RuntimeError(f"active branch is {branch}")
    if _git(repo, ["merge-base", "--is-ancestor", INSTRUCTION_COMMIT, "HEAD"]).returncode != 0:
        raise RuntimeError("HEAD does not contain the L8 instruction commit")
    if float(LEARNING_RATE) != 3.0e-4 or float(WEIGHT_DECAY) != 1.0e-4 or float(GRAD_CLIP) != 1.0:
        raise RuntimeError("optimizer constants differ from the fixed L3/L8 values")
    if int(LABEL_SEED) != 605 or int(MAX_CANDIDATES) != 128:
        raise RuntimeError("candidate sampling constants differ from seed 605 and 128 candidates")
    hashes = _hash_protected(repo)
    for name in ("epoch6", "decoder", "c_listnet", "l3_a10", "mappo", "scheme_c"):
        if hashes[name] != HASHES[name]:
            raise RuntimeError(f"{name} hash is {hashes[name]}")
    l3 = json.loads((repo / "results/legacy_arrival_aware_ab_l3/manifest.json").read_text(encoding="utf-8"))
    if int(l3["training_updates"]) != 240 or l3["arms"]["B"] != "real_payload_bits":
        raise RuntimeError("saved L3 manifest does not describe the reused ListNet arm")
    if l3["epoch6_sha256"] != HASHES["epoch6"] or l3["decoder_sha256"] != HASHES["decoder"]:
        raise RuntimeError("saved L3 manifest does not use the shared initialization")
    if l3["arm_b_epoch_010_sha256"] != HASHES["c_listnet"] or l3["arm_a_epoch_010_sha256"] != HASHES["l3_a10"]:
        raise RuntimeError("saved L3 arm hashes differ from the fixed references")
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
                rows.append(
                    {
                        "commit": FROZEN_SHARD_COMMIT,
                        "path": relpath,
                        "blob": blob,
                        "sha256": digest,
                        "on_disk": disk.is_file(),
                    }
                )
        for time_step in COMPARISON_TIMES:
            relpath = _revision_relpath(scenario, time_step)
            blob, _data = _blob_bytes(repo, REVISION_COMMIT, relpath)
            rows.append({"commit": REVISION_COMMIT, "path": relpath, "blob": blob, "role": "revision_check_source"})
    ocean = _git(repo, ["rev-parse", OCEAN_BRANCH])
    ocean_sha = ocean.stdout.decode().strip() if ocean.returncode == 0 else ""
    if ocean_sha and ocean_sha != OCEAN_EXPECTED:
        raise RuntimeError(f"ocean branch is {ocean_sha}")
    return {
        "head": head,
        "branch": branch,
        "protected_sha256": hashes,
        "l3_reference": {
            "updates_total_recorded": int(l3["training_updates"]),
            "b_updates_recorded": 120,
            "same_initialization": True,
            "same_decoder": True,
            "same_data_package": l3["data_package"],
        },
        "historical_objects": rows,
        "ocean_branch_sha256": ocean_sha,
    }


def _train(repo: Path, output: Path, device: torch.device, consumption: dict) -> tuple[torch.nn.Module, torch.nn.Module]:
    log_path = output / "train_log.jsonl"
    done_path = output / "r_epoch_010.pt"
    existing = 0
    if log_path.is_file():
        existing = sum(1 for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if existing not in (0, UPDATES):
        raise RuntimeError(f"partial training log has {existing} updates; refusing to train again")
    decoder_path = repo / "results/phase5k_joint_decoder/matching_decoder.pt"
    epoch_path = repo / "results/phase6id_joint_utility_training/epoch_006.pt"
    if existing == UPDATES:
        if not done_path.is_file():
            raise RuntimeError("training log is complete but the epoch-10 weights are missing")
        consumption["training_updates"] = UPDATES
        consumption["backward_passes"] = UPDATES
        consumption["scoring_forwards"] = max(int(consumption["scoring_forwards"]), UPDATES)
        model = _load_named(done_path, device, 10)
        decoder = _decoder(decoder_path, device)
        return model, decoder
    loaded = load_package(repo / "results/legacy_arrival_aware_ab_l2", repo)
    states = loaded["train"]
    expected = [(scenario, time_step) for scenario in TRAIN_SCENARIOS for time_step in (15, 30)]
    if [(int(state["scenario_id"]), int(state["time_step"])) for state in states] != expected:
        raise RuntimeError("training states are not the fixed 12 L2 states")
    model, decoder, optimizer = _fresh_r(epoch_path, decoder_path, device)
    decoder_before = _weights(decoder)
    log_path.write_text("", encoding="utf-8", newline="\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        decoder.eval()
        for index, state in enumerate(states, start=1):
            optimizer.zero_grad(set_to_none=True)
            scores = _score(model, decoder, state["graph"], state["assignments"], device, consumption)
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
                "arm": "R_EXPECTED_REGRET",
                "epoch": epoch,
                "state_index": index,
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "loss": float(loss.detach().cpu()),
                "grad_norm": grad_norm,
                "update": int(consumption["training_updates"]),
            }
            with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"R epoch {epoch} {index}/12 loss={record['loss']:.6f} grad={grad_norm:.4f}", flush=True)
    if int(consumption["training_updates"]) != UPDATES or _weights(decoder) != decoder_before:
        raise RuntimeError("training did not finish with a frozen decoder and 120 updates")
    torch.save(
        {
            "epoch": EPOCHS,
            "arm": "R_EXPECTED_REGRET",
            "label": "real_payload_bits",
            "loss": "expected_regret_temperature_1",
            "model": model.state_dict(),
            "updates": UPDATES,
            "initialization_sha256": HASHES["epoch6"],
        },
        done_path,
    )
    model.eval()
    return model, decoder


def _selection(scores: np.ndarray, candidate_ids: np.ndarray, assignments: list[str], payload: np.ndarray | None) -> dict:
    local = _choose(scores, candidate_ids)
    chosen_payload = None if payload is None else float(payload[local])
    optimum = None if payload is None else float(np.max(payload))
    return {
        "local_position": local,
        "candidate_id": int(candidate_ids[local]),
        "assignment": assignments[local],
        "score": float(np.asarray(scores, dtype=np.float32)[local]),
        "active_count": _active_count(assignments[local]),
        "tie_count": int(np.sum(np.asarray(scores, dtype=np.float32) == np.max(np.asarray(scores, dtype=np.float32)))),
        "real_payload_bits": chosen_payload,
        "subset_optimum_bits": optimum,
        "subset_regret_bits": None if payload is None else float(optimum - chosen_payload),
    }


def _score_train(model, decoder, repo: Path, device: torch.device, consumption: dict, output: Path) -> None:
    target = output / "train_selection_r.json"
    if target.is_file():
        return
    loaded = load_package(repo / "results/legacy_arrival_aware_ab_l2", repo)
    rows = []
    model.eval()
    decoder.eval()
    with torch.no_grad():
        for state in loaded["train"]:
            scores = _score(model, decoder, state["graph"], state["assignments"], device, consumption).detach().cpu().numpy()
            chosen = _selection(scores, state["candidate_id"], state["assignment_text"], state["real_payload"])
            labels = torch.as_tensor(state["real_payload"], dtype=torch.float64)
            soft = expected_regret_loss(torch.as_tensor(scores, dtype=torch.float64), labels)
            chosen.update(
                {
                    "scenario_id": int(state["scenario_id"]),
                    "time_step": int(state["time_step"]),
                    "split": "train",
                    "soft_expected_regret": float(soft),
                }
            )
            rows.append(chosen)
    _write_json(target, {"rows": rows, "states": len(rows), "uses_existing_l2_labels": True})


def _copy_l4_train(repo: Path, output: Path) -> None:
    target = output / "train_selection_c_l4.json"
    if target.is_file():
        return
    source = repo / "results/legacy_arrival_aware_ab_l4/state_records.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for row in payload["B10"]:
        if int(row["scenario_id"]) not in TRAIN_SCENARIOS or row["group"] != "train":
            continue
        rows.append(
            {
                "scenario_id": int(row["scenario_id"]),
                "time_step": int(row["time_step"]),
                "candidate_id": int(row["candidate_id"]),
                "assignment": row["assignment"],
                "score": float(row["score"]),
                "active_count": int(row["active_uavs"]),
                "real_payload_bits": float(row["real_payload_bits"]),
                "subset_regret_bits": float(row["real_regret_bits"]),
                "source": "results/legacy_arrival_aware_ab_l4/state_records.json",
            }
        )
    rows.sort(key=lambda row: (row["scenario_id"], row["time_step"]))
    if len(rows) != 12:
        raise RuntimeError(f"L4 cache exposes {len(rows)} ListNet training selections")
    _write_json(target, {"rows": rows, "source_sha256": _sha256(source), "recomputed": False})


def _save_arrays(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value) for key, value in arrays.items()})


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _snap_payload(snap: dict) -> tuple[dict, dict]:
    arrays = {key: np.asarray(value) for key, value in snap.items() if isinstance(value, np.ndarray)}
    scalars = {key: value for key, value in snap.items() if not isinstance(value, np.ndarray)}
    return arrays, scalars


def _combine_snap(arrays: dict, scalars: dict) -> dict:
    snap = {key: np.asarray(value) for key, value in arrays.items()}
    snap.update(scalars)
    return snap


def _same_arrays(left: dict, right: dict) -> tuple[bool, str]:
    shared = set(left) & set(right)
    required = {"service_progress", "uav_step_displacement_m", "uav_position_m", "q_bits", "association", "time_index"}
    if not required <= shared:
        return False, "required keys " + ",".join(sorted(required - shared))
    for key in sorted(shared):
        current = np.asarray(left[key])
        historical = np.asarray(right[key])
        if np.array_equal(current, historical):
            continue
        if current.shape == historical.shape and np.issubdtype(current.dtype, np.number) and np.issubdtype(historical.dtype, np.number):
            gap = float(np.max(np.abs(current.astype(np.float64) - historical.astype(np.float64))))
            if gap == 0.0:
                continue
            return False, f"{key} max_abs {gap}"
        return False, f"{key} shape {current.shape} vs {historical.shape} dtype {current.dtype} vs {historical.dtype}"
    return True, ""


def _replay_scenario(scenario: int) -> dict:
    import wt.benchmark_v2.phase6i_urgency_highlevel_pilot.replay as pilot
    from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import step_observation
    from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_arrays

    last = {"position": None, "time": None}
    original_capture = pilot._capture_decision
    original_step = pilot._live_step

    def capture(env, ledger, scenario_id):
        row = original_capture(env, ledger, scenario_id)
        if last["position"] is None:
            previous = np.asarray(env.uav_positions, dtype=np.float64).copy()
            predecessor = None
        else:
            previous = last["position"]
            predecessor = last["time"]
        observed = step_observation(env, previous, predecessor)
        progress = np.asarray(observed["service_progress"], dtype=np.float64)
        if not np.array_equal(progress, np.asarray(row["env_snap"]["progress"])):
            raise RuntimeError("service_progress does not match the live environment")
        arrays = state_arrays(env)
        arrays["service_progress"] = progress
        arrays["uav_step_displacement_m"] = np.asarray(observed["uav_step_displacement_m"], dtype=np.float64)
        if not np.all(np.isfinite(arrays["service_progress"])) or not np.all(np.isfinite(arrays["uav_step_displacement_m"])):
            raise RuntimeError("recorded physical fields are non-finite")
        row["revised_arrays"] = arrays
        row["velocity_predecessor"] = observed["velocity_predecessor"]
        row["predecessor_time_step"] = observed["predecessor_time_step"]
        return row

    def step(env, actor, option, use_fly=False):
        last["position"] = np.asarray(env.uav_positions, dtype=np.float64).copy()
        last["time"] = int(env.t)
        return original_step(env, actor, option, use_fly=use_fly)

    pilot._capture_decision = capture
    pilot._live_step = step
    try:
        return pilot.replay_scenario(int(scenario))
    finally:
        pilot._capture_decision = original_capture
        pilot._live_step = original_step


def _state_dir(output: Path, scenario: int, time_step: int) -> Path:
    return output / "comparison" / f"scenario_{scenario}" / f"t{time_step:03d}"


def _replay_all(repo: Path, output: Path, consumption: dict) -> None:
    for scenario in COMPARISON_SCENARIOS:
        stamp = output / "comparison" / f"scenario_{scenario}" / "started.json"
        complete = output / "comparison" / f"scenario_{scenario}" / "complete.json"
        if complete.is_file():
            continue
        if stamp.is_file():
            raise RuntimeError(f"scenario {scenario} replay already started; refusing to spend the steps again")
        _write_json(stamp, {"scenario_id": int(scenario), "status": "started"})
        before = int(consumption["environment_steps"])
        replay = _replay_scenario(scenario)
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
            folder = _state_dir(output, scenario, time_step)
            arrays, scalars = _snap_payload(row["env_snap"])
            _save_arrays(folder / "env_snap.npz", arrays)
            _write_json(folder / "env_scalars.json", scalars)
            restored = _combine_snap(_load_arrays(folder / "env_snap.npz"), json.loads((folder / "env_scalars.json").read_text(encoding="utf-8")))
            if not all(np.array_equal(np.asarray(restored[key]), np.asarray(row["env_snap"][key])) for key in row["env_snap"] if isinstance(row["env_snap"][key], np.ndarray)):
                raise RuntimeError("saved environment snapshot does not round-trip")
            _write_json(folder / "ledger.json", row["ledger_snap"])
            _save_arrays(folder / "revised_state.npz", row["revised_arrays"])
            _blob, historical_bytes = _blob_bytes(repo, REVISION_COMMIT, _revision_relpath(scenario, time_step))
            with np.load(io.BytesIO(historical_bytes)) as historical:
                historical_arrays = {key: np.asarray(historical[key]) for key in historical.files}
            equal, mismatch = _same_arrays(row["revised_arrays"], historical_arrays)
            if not equal:
                raise RuntimeError(f"revised state mismatch at {scenario} t={time_step}: {mismatch}")
            graph = _graph_from_arrays(row["revised_arrays"])
            with (folder / "graph.npz").open("wb") as handle:
                np.savez(handle, **graph)
            meta = {
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
            }
            _write_json(folder / "state_meta.json", meta)
            saved.append(time_step)
        if saved != [15, 30]:
            raise RuntimeError("t15 and t30 snapshots were not both saved")
        _write_json(complete, {"scenario_id": int(scenario), "environment_steps": delta, "states": saved})
        print(f"replay {scenario} steps={delta}", flush=True)


def _load_state(output: Path, scenario: int, time_step: int) -> dict:
    folder = _state_dir(output, scenario, time_step)
    meta = json.loads((folder / "state_meta.json").read_text(encoding="utf-8"))
    scalars = json.loads((folder / "env_scalars.json").read_text(encoding="utf-8"))
    snap = _combine_snap(_load_arrays(folder / "env_snap.npz"), scalars)
    return {
        "scenario_id": int(scenario),
        "time_step": int(time_step),
        "env_snap": snap,
        "ledger_snap": json.loads((folder / "ledger.json").read_text(encoding="utf-8")),
        "revised_arrays": _load_arrays(folder / "revised_state.npz"),
        "graph": _load_arrays(folder / "graph.npz"),
        "state_hash": meta["state_hash"],
        "manifest_checksum_candidates": meta["manifest_checksum_candidates"],
        "manifest_candidate_count": meta["manifest_candidate_count"],
        "manifest_illegal": meta["manifest_illegal"],
        "tol_bits": float(meta["tol_bits"]),
    }


def _select_all(output: Path, consumption: dict) -> list[dict]:
    from wt.benchmark_v2.phase6i_urgency_arrival_aware_expansion.sampling import select_state

    target = output / "candidates.json"
    if target.is_file():
        return json.loads(target.read_text(encoding="utf-8"))["states"]
    states = []
    before = int(consumption["environment_steps"])
    for scenario in COMPARISON_SCENARIOS:
        for time_step in COMPARISON_TIMES:
            state = _load_state(output, scenario, time_step)
            if int(np.asarray(state["revised_arrays"]["time_index"]).reshape(-1)[0]) != int(time_step):
                raise RuntimeError("graph source time does not match the decision")
            rows, meta = select_state(int(scenario), state)
            if meta["sampling_source"] != "stored_j15" or len(rows) != 128:
                raise RuntimeError("candidate subset did not stay on the stored J15 128-candidate rule")
            identities = [int(row["candidate_id"]) for row in rows]
            if len(set(identities)) != 128:
                raise RuntimeError("candidate ids are not unique")
            local_positions = list(range(128))
            if identities == local_positions:
                raise RuntimeError("candidate ids collapsed onto local positions")
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
                        "stratum": int(row["stratum"]),
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
        return
    before_steps = int(consumption["environment_steps"])
    selections = []
    score_rows = {}
    for state in states:
        graph = _load_state(output, int(state["scenario_id"]), int(state["time_step"]))["graph"]
        assignments = np.stack([[int(part) for part in row["assignment"].split("|")] for row in state["rows"]])
        identities = np.asarray([int(row["candidate_id"]) for row in state["rows"]], dtype=np.int64)
        per_model = {}
        for name in MODEL_ORDER:
            scores = _score(models[name], decoder, graph, assignments, device, consumption).detach().cpu().numpy()
            key = f"{name}_{int(state['scenario_id'])}_t{int(state['time_step']):03d}"
            score_rows[key] = np.asarray(scores, dtype=np.float32)
            per_model[name] = _selection(scores, identities, [row["assignment"] for row in state["rows"]], None)
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
    if int(consumption["environment_steps"]) != before_steps:
        raise RuntimeError("high-level scoring stepped the environment")
    score_path = output / "raw_scores.npz"
    with score_path.open("wb") as handle:
        np.savez(handle, **score_rows)
    _write_json(output / "selections.json", {"states": selections, "labels_available": False})
    freeze = {
        "r_epoch_010_sha256": _sha256(output / "r_epoch_010.pt"),
        "raw_scores_sha256": _sha256(score_path),
        "selections_sha256": _sha256(output / "selections.json"),
        "candidates_sha256": _sha256(output / "candidates.json"),
        "labels_written": False,
        "new_real_arrival_labels_revealed": False,
    }
    _write_json(output / "freeze.json", freeze)
    _write_json(output / "freeze_sha256.json", {"freeze_sha256": _sha256(output / "freeze.json")})


def _assignment_vector(text: str) -> np.ndarray:
    return np.asarray([int(part) for part in text.split("|")], dtype=np.int64)


def _label_all(output: Path, consumption: dict, states: list[dict]) -> None:
    from wt.benchmark_v2.phase6i_urgency_arrival_aware_expansion.labels import score_label
    from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _frozen_actor

    if not (output / "freeze.json").is_file():
        raise RuntimeError("labels started before the prediction freeze")
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
        finished_rows = [
            row
            for row in state["rows"]
            if (int(state["scenario_id"]), int(state["time_step"]), int(row["candidate_id"])) in done
        ]
        if stamp.is_file() and not finished_rows:
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
        print(
            f"labels {state['scenario_id']} t={state['time_step']} env_steps={consumption['environment_steps']}",
            flush=True,
        )
        gc.collect()


def _labels_by_key(output: Path) -> dict[tuple[int, int, int], dict]:
    found = {}
    for line in (output / "real_labels.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[(int(row["scenario_id"]), int(row["time_step"]), int(row["candidate_id"]))] = row
    return found


def _soft_regret(scores: np.ndarray, payload: np.ndarray) -> float:
    loss = expected_regret_loss(torch.as_tensor(scores, dtype=torch.float64), torch.as_tensor(payload, dtype=torch.float64))
    return float(loss)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize(output: Path) -> dict:
    candidates = json.loads((output / "candidates.json").read_text(encoding="utf-8"))["states"]
    selections = json.loads((output / "selections.json").read_text(encoding="utf-8"))["states"]
    labels = _labels_by_key(output)
    with np.load(output / "raw_scores.npz") as scores:
        score_map = {key: np.asarray(scores[key]) for key in scores.files}
    selection_by_state = {(int(row["scenario_id"]), int(row["time_step"])): row for row in selections}
    per_state = []
    for state in candidates:
        key = (int(state["scenario_id"]), int(state["time_step"]))
        payload = []
        for row in state["rows"]:
            label = labels[(key[0], key[1], int(row["candidate_id"]))]
            if label["assignment"] != row["assignment"]:
                raise RuntimeError("label assignment does not match the frozen candidate")
            payload.append(float(label["real_payload_bits"]))
        payload_array = np.asarray(payload, dtype=np.float64)
        optimum = float(np.max(payload_array))
        identities = np.asarray([int(row["candidate_id"]) for row in state["rows"]], dtype=np.int64)
        texts = [row["assignment"] for row in state["rows"]]
        models = {}
        for name in MODEL_ORDER:
            vector = score_map[f"{name}_{key[0]}_t{key[1]:03d}"]
            chosen = _selection(vector, identities, texts, payload_array)
            saved = selection_by_state[key]["models"][name]
            if int(saved["candidate_id"]) != int(chosen["candidate_id"]) or saved["assignment"] != chosen["assignment"]:
                raise RuntimeError("selection changed after the real labels were revealed")
            chosen["soft_expected_regret"] = _soft_regret(vector, payload_array)
            models[name] = chosen
        per_state.append(
            {
                "scenario_id": key[0],
                "time_step": key[1],
                "state_hash": state["state_hash"],
                "pool_hash": state["pool_hash"],
                "subset_hash": state["subset_hash"],
                "subset_optimum_bits": optimum,
                "models": models,
                "R_minus_C_payload_bits": float(models["R_EXPECTED_REGRET"]["real_payload_bits"] - models["C_LISTNET"]["real_payload_bits"]),
                "R_minus_E6_payload_bits": float(models["R_EXPECTED_REGRET"]["real_payload_bits"] - models["E6"]["real_payload_bits"]),
                "R_minus_L3A10_payload_bits": float(models["R_EXPECTED_REGRET"]["real_payload_bits"] - models["L3A10"]["real_payload_bits"]),
            }
        )
    per_state.sort(key=lambda row: (row["scenario_id"], row["time_step"]))

    def pack(rows: list[dict]) -> dict:
        return {
            "states": len(rows),
            "R_minus_C_payload_bits_mean": _mean([row["R_minus_C_payload_bits"] for row in rows]),
            "R_minus_E6_payload_bits_mean": _mean([row["R_minus_E6_payload_bits"] for row in rows]),
            "R_minus_L3A10_payload_bits_mean": _mean([row["R_minus_L3A10_payload_bits"] for row in rows]),
            "payload_bits_mean": {name: _mean([row["models"][name]["real_payload_bits"] for row in rows]) for name in MODEL_ORDER},
            "subset_regret_bits_mean": {name: _mean([row["models"][name]["subset_regret_bits"] for row in rows]) for name in MODEL_ORDER},
            "soft_expected_regret_mean": {name: _mean([row["models"][name]["soft_expected_regret"] for row in rows]) for name in MODEL_ORDER},
        }

    scenarios = []
    for scenario in COMPARISON_SCENARIOS:
        rows = [row for row in per_state if int(row["scenario_id"]) == int(scenario)]
        if [int(row["time_step"]) for row in rows] != [15, 30]:
            raise RuntimeError(f"scenario {scenario} does not have both decision times")
        difference = _mean([row["R_minus_C_payload_bits"] for row in rows])
        scenarios.append(
            {
                "scenario_id": int(scenario),
                "t15_R_minus_C_payload_bits": float(rows[0]["R_minus_C_payload_bits"]),
                "t30_R_minus_C_payload_bits": float(rows[1]["R_minus_C_payload_bits"]),
                "mean_R_minus_C_payload_bits": difference,
            }
        )
    positive = sum(row["mean_R_minus_C_payload_bits"] > NEAR_ZERO_BITS for row in scenarios)
    negative = sum(row["mean_R_minus_C_payload_bits"] < -NEAR_ZERO_BITS for row in scenarios)
    near_zero = sum(abs(row["mean_R_minus_C_payload_bits"]) <= NEAR_ZERO_BITS for row in scenarios)
    overall = pack(per_state)
    t15 = pack([row for row in per_state if int(row["time_step"]) == 15])
    t30 = pack([row for row in per_state if int(row["time_step"]) == 30])
    gain = overall["R_minus_C_payload_bits_mean"] > 1.0
    times_ok = t15["R_minus_C_payload_bits_mean"] >= -1.0 and t30["R_minus_C_payload_bits_mean"] >= -1.0
    references_ok = (
        overall["payload_bits_mean"]["R_EXPECTED_REGRET"] >= overall["payload_bits_mean"]["E6"] - 1.0
        and overall["payload_bits_mean"]["R_EXPECTED_REGRET"] >= overall["payload_bits_mean"]["L3A10"] - 1.0
    )
    gate = "CONTINUE_TO_CLOSED_LOOP_COMPARISON" if gain and times_ok and references_ok else "NOT_PROMOTED"
    return {
        "per_state": per_state,
        "overall": overall,
        "t15": t15,
        "t30": t30,
        "scenarios": scenarios,
        "scenario_counts": {"positive": int(positive), "negative": int(negative), "near_zero": int(near_zero), "near_zero_abs_bits": NEAR_ZERO_BITS},
        "gate": {
            "label": gate,
            "overall_gain_above_1_bit": bool(gain),
            "both_times_at_least_minus_1_bit": bool(times_ok),
            "not_below_references_by_more_than_1_bit": bool(references_ok),
            "starts_next_stage": False,
            "adopts_model": False,
        },
    }


def _fmt(value: float) -> str:
    return format(float(value), ".12g")


def _mbit(value: float) -> str:
    return format(float(value) / 1.0e6, ".12g")


def _write_report(repo: Path, summary: dict, consumption: dict, status: str, note: str) -> None:
    lines = ["# L8 单变量训练目标对照", ""]
    if status == "complete":
        gate = summary["gate"]["label"]
        overall = summary["overall"]
        lines.extend(
            [
                f"结论是 `{gate}`。R 相对复用的 ListNet 对照，在 16 个新增状态上的 argmax 真实有效载荷平均差为 {_fmt(overall['R_minus_C_payload_bits_mean'])} bit（{_mbit(overall['R_minus_C_payload_bits_mean'])} Mbit）。",
                "",
                "这个门槛只决定以后是否值得另写闭环对照，不启动下一阶段，也不采纳新模型。",
                "",
                "## 主要指标",
                "",
                "| 范围 | R−C (bit) | R (Mbit) | C (Mbit) | E6 (Mbit) | L3A10 (Mbit) |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, block in (("16 个状态", summary["overall"]), ("t15", summary["t15"]), ("t30", summary["t30"])):
            lines.append(
                f"| {name} | {_fmt(block['R_minus_C_payload_bits_mean'])} | {_mbit(block['payload_bits_mean']['R_EXPECTED_REGRET'])} | {_mbit(block['payload_bits_mean']['C_LISTNET'])} | {_mbit(block['payload_bits_mean']['E6'])} | {_mbit(block['payload_bits_mean']['L3A10'])} |"
            )
        lines.extend(
            [
                "",
                f"继续条件：总体 R−C 大于 1 bit 为 {summary['gate']['overall_gain_above_1_bit']}；t15 与 t30 各自不低于 −1 bit 为 {summary['gate']['both_times_at_least_minus_1_bit']}；R 相对 E6 和 L3A10 的总体有效载荷容差成立为 {summary['gate']['not_below_references_by_more_than_1_bit']}。",
                "",
                f"R−E6 总体平均差 {_fmt(overall['R_minus_E6_payload_bits_mean'])} bit。R−L3A10 总体平均差 {_fmt(overall['R_minus_L3A10_payload_bits_mean'])} bit。128 候选内部的平均子集 regret：R {_fmt(overall['subset_regret_bits_mean']['R_EXPECTED_REGRET'])} bit，C {_fmt(overall['subset_regret_bits_mean']['C_LISTNET'])} bit，E6 {_fmt(overall['subset_regret_bits_mean']['E6'])} bit，L3A10 {_fmt(overall['subset_regret_bits_mean']['L3A10'])} bit。",
                "",
                "八个场景先对 t15 和 t30 取平均，再统计 R−C。绝对差不超过 1 bit 记为接近零。",
                "",
                "| 场景 | t15 R−C (bit) | t30 R−C (bit) | 两次平均 (bit) |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary["scenarios"]:
            lines.append(
                f"| {row['scenario_id']} | {_fmt(row['t15_R_minus_C_payload_bits'])} | {_fmt(row['t30_R_minus_C_payload_bits'])} | {_fmt(row['mean_R_minus_C_payload_bits'])} |"
            )
        counts = summary["scenario_counts"]
        lines.extend(
            [
                "",
                f"场景计数：正差 {counts['positive']}，负差 {counts['negative']}，接近零 {counts['near_zero']}。",
                "",
                "软期望 regret 和训练损失只作辅助，没有用来选择检查点。16 状态上的平均软期望 regret：R "
                f"{_fmt(overall['soft_expected_regret_mean']['R_EXPECTED_REGRET'])}，C {_fmt(overall['soft_expected_regret_mean']['C_LISTNET'])}。",
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
            f"- 对照模型重新训练：{consumption['retrained_reference_models']}",
            "",
            "预算是 120 次更新、120 次反向传播、8 次 30 步重放、2048 个 15 步真实到达标签、合计 30,960 个环境步，以及不超过 200 次高层评分前向。",
            "",
            "## 证据与限制",
            "",
            "ListNet 对照直接复用 L3 的 B10，权重哈希为 `6024f4c42cdf2de2e8452b49da6ed1824ff8d6e56e84204681a6a37161b81141`。E6 和 L3 A10 只作为固定参照。R 从同一个 E6 完整权重和空 Adam 状态出发，损失是温度 1 的期望归一化 regret。训练标签是 L2 已有的 12 个状态上的真实到达有效载荷。训练选择和新增场景对照分开记录。",
            "",
            "新增对照是场景 322024–322031 的 t15 和 t30。它们没有打开既有验证集或保留测试，也没有进入本轮选择。这些场景仍在历史训练池 322000–322047 内，E6 可能已经在零到达学习中见过，因此这不是独立验证、未见泛化或正式留出集。重放使用冻结 MAPPO-V2 seed 0 的 MAX_RATE_INDEPENDENT 路径，只到 t30，不是正式 Scheme C 轨迹。",
            "",
            "每个状态的 128 个候选沿用旧零到达 J15 的前 32 名加名次分层 96 个，seed 605。旧零到达标签因此影响了候选支持。regret 最优只在这 128 个候选内部计算。未来到达只进入离线标签，不进入图或选择。R10 的权重、候选、原始分数和选择先冻结，随后才生成真实到达标签。",
            "",
            note,
            "",
            "lambda_H 与 delta_payload_max 仍未设置。本结果不写入论文，也不进入正式命令入口。海洋观测隔离分支没有修改。",
            "",
        ]
    )
    report = repo / "docs/research/Legacy_System_L8_Report_CN.md"
    report.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _install_counters(consumption: dict) -> dict:
    import wt.benchmark_v2.phase6h_executor_conditioned_macro.jobs as jobs
    import wt.benchmark_v2.phase6i_urgency_arrival_aware_expansion.labels as labels

    originals = {
        "step": L3BBitQueueEnvV2.step,
        "jobs_move": jobs.deterministic_move,
        "labels_move": labels.deterministic_move,
    }

    def counted_step(self, *args, **kwargs):
        outcome = originals["step"](self, *args, **kwargs)
        consumption["environment_steps"] += 1
        return outcome

    def counted_jobs_move(*args, **kwargs):
        consumption["low_level_inferences"] += 1
        return originals["jobs_move"](*args, **kwargs)

    def counted_labels_move(*args, **kwargs):
        consumption["low_level_inferences"] += 1
        return originals["labels_move"](*args, **kwargs)

    L3BBitQueueEnvV2.step = counted_step
    jobs.deterministic_move = counted_jobs_move
    labels.deterministic_move = counted_labels_move
    return originals


def _restore_counters(originals: dict) -> None:
    import wt.benchmark_v2.phase6h_executor_conditioned_macro.jobs as jobs
    import wt.benchmark_v2.phase6i_urgency_arrival_aware_expansion.labels as labels

    L3BBitQueueEnvV2.step = originals["step"]
    jobs.deterministic_move = originals["jobs_move"]
    labels.deterministic_move = originals["labels_move"]


def _fail(output: Path, repo: Path, consumption: dict, exc: Exception) -> None:
    _save_consumption(output / "consumption.json", consumption)
    _write_json(output / "failure.json", {"error": f"{type(exc).__name__}: {exc}", **consumption})
    _write_report(repo, {}, consumption, "incomplete", f"执行在生成完整对照前停止：{type(exc).__name__}: {exc}")


def run() -> None:
    repo = _repo()
    output = _output(repo)
    if (output / "manifest.json").is_file():
        raise RuntimeError("L8 output already exists")
    output.mkdir(parents=True, exist_ok=True)
    consumption = _load_consumption(output / "consumption.json")
    originals = None
    try:
        loss_check = check_loss()
        _write_json(output / "loss_check.json", loss_check)
        preflight = _preflight(repo)
        _write_json(output / "source_manifest.json", {"instruction_commit": INSTRUCTION_COMMIT, "command": "py -3 -m wt.legacy_payload_objective_l8 --run", **preflight})
        before = _hash_protected(repo)
        _configure()
        device = torch.device("cpu")
        originals = _install_counters(consumption)
        model_r, decoder = _train(repo, output, device, consumption)
        _save_consumption(output / "consumption.json", consumption)
        _score_train(model_r, decoder, repo, device, consumption, output)
        _copy_l4_train(repo, output)
        _save_consumption(output / "consumption.json", consumption)
        _write_json(output / "r10_freeze.json", {"sha256": _sha256(output / "r_epoch_010.pt"), "new_comparison_labels_revealed": False})
        _replay_all(repo, output, consumption)
        if int(consumption["environment_steps"]) != REPLAY_STEPS:
            raise RuntimeError(f"replay consumed {consumption['environment_steps']} steps")
        states = _select_all(output, consumption)
        models = {
            "R_EXPECTED_REGRET": model_r,
            "C_LISTNET": _load_named(repo / "results/legacy_arrival_aware_ab_l3/arm_b_epoch_010.pt", device, 10),
            "E6": _load_named(repo / "results/phase6id_joint_utility_training/epoch_006.pt", device, 6),
            "L3A10": _load_named(repo / "results/legacy_arrival_aware_ab_l3/arm_a_epoch_010.pt", device, 10),
        }
        for model in models.values():
            model.eval()
        decoder.eval()
        with torch.no_grad():
            _score_comparison(output, models, decoder, device, consumption, states)
        _save_consumption(output / "consumption.json", consumption)
        freeze_hash = _sha256(output / "freeze.json")
        _label_all(output, consumption, states)
        if _sha256(output / "freeze.json") != freeze_hash:
            raise RuntimeError("freeze record changed after labels")
        if int(consumption["environment_steps"]) != ENVIRONMENT_CAP or int(consumption["candidate_rollouts"]) != 2048:
            raise RuntimeError("label consumption does not match the fixed budget")
        if int(consumption["training_updates"]) != UPDATES or int(consumption["backward_passes"]) != UPDATES:
            raise RuntimeError("training consumption does not match 120 updates")
        if int(consumption["scoring_forwards"]) != UPDATES + TRAIN_FORWARD_EXTRA + len(MODEL_ORDER) * 16:
            raise RuntimeError(f"scoring forwards are {consumption['scoring_forwards']}")
        summary = _summarize(output)
        after = _hash_protected(repo)
        if after != before:
            changed = sorted(key for key in before if before[key] != after[key])
            raise RuntimeError(f"protected files changed: {changed}")
        ocean = _git(repo, ["rev-parse", OCEAN_BRANCH])
        ocean_sha = ocean.stdout.decode().strip() if ocean.returncode == 0 else ""
        if ocean_sha != preflight["ocean_branch_sha256"]:
            raise RuntimeError("ocean branch changed")
        _write_json(output / "paired_detail.json", summary)
        _write_json(output / "paired_summary.json", {key: value for key, value in summary.items() if key != "per_state"})
        manifest = {
            "package": "legacy_payload_objective_l8",
            "instruction_commit": INSTRUCTION_COMMIT,
            "command": "py -3 -m wt.legacy_payload_objective_l8 --run",
            "gate": summary["gate"],
            "primary_R_minus_C_payload_bits_mean": summary["overall"]["R_minus_C_payload_bits_mean"],
            "r_epoch_010_sha256": _sha256(output / "r_epoch_010.pt"),
            "freeze_sha256": freeze_hash,
            "protected_unchanged": True,
            "ocean_branch_unchanged": True,
            **consumption,
        }
        _save_consumption(output / "consumption.json", consumption)
        failure_path = output / "failure.json"
        if failure_path.is_file():
            failure_path.unlink()
        _write_json(output / "manifest.json", manifest)
        _write_report(
            repo,
            summary,
            consumption,
            "complete",
            "必要检查包括小张量损失梯度、L3 已保存的共同初始化记录、候选身份、图字段、15 步恢复、served 总量容差和有限数值。训练后复核了解码器、E6、ListNet、L3A10、正式策略和 MAPPO 文件哈希。",
        )
        print(json.dumps({"gate": summary["gate"]["label"], **consumption}, ensure_ascii=False), flush=True)
    except Exception as exc:
        _fail(output, repo, consumption, exc)
        raise
    finally:
        if originals is not None:
            _restore_counters(originals)
