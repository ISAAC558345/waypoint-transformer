"""One paired arrival-label comparison. The historical trainer is not called."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.legacy.joint_training.protocol import GRAD_CLIP
from wt.algorithms.ranking_loss.losses import LOSSES
from wt.models.edge_bound.const import BUOY_DIM
from wt.research.arrival_aware.loader import load_package
from wt.research.arrival_aware.prepare import prepare_from_existing
from wt.models.joint_decoder.decoders import JointAssignmentDecoder
from wt.models.joint_encoder.model import append_queue_edge, batch_tensors
from wt.models.joint_encoder.model import JointUtilityHGNN
from wt.models.joint_encoder.protocol import LEARNING_RATE, WEIGHT_DECAY
from wt.common.paths import repo_root


EPOCHS = 10
SEED = 0
EXPECTED_EPOCH6 = "7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde"
EXPECTED_DECODER = "de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5"
LABELS = {"A": "zero_payload", "B": "real_payload"}


def _repo_root() -> Path:
    return repo_root()
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _configure_runtime() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def _seed() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _weights(module: torch.nn.Module) -> list[tuple[str, bytes]]:
    return [(name, tensor.detach().cpu().contiguous().numpy().tobytes()) for name, tensor in module.state_dict().items()]


def _build_pair(epoch_path: Path, decoder_path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer, torch.nn.Module]:
    _seed()
    epoch_payload = torch.load(epoch_path, map_location="cpu", weights_only=False)
    if int(epoch_payload["epoch"]) != 6:
        raise RuntimeError(f"initialization epoch is {epoch_payload.get('epoch')}")
    decoder_payload = torch.load(decoder_path, map_location="cpu", weights_only=False)
    model = JointUtilityHGNN(buoy_dim=int(BUOY_DIM) + 1)
    model.load_state_dict(epoch_payload["model"], strict=True)
    model.to(device)
    decoder = JointAssignmentDecoder("matching")
    decoder.load_state_dict(decoder_payload["state_dict"], strict=True)
    decoder.to(device)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(LEARNING_RATE), weight_decay=float(WEIGHT_DECAY))
    if optimizer.state:
        raise RuntimeError("Adam was not created empty")
    return model, decoder, optimizer, epoch_payload


def _score(model, decoder, state: dict, device: torch.device, consumption: dict) -> torch.Tensor:
    graph = state["graph"]
    if "zero_payload" in graph or "real_payload" in graph or "scenario_id" in graph:
        raise RuntimeError("graph contains a label or identity field")
    conditioned = append_queue_edge(graph)
    tensors = {key: value.to(device) for key, value in batch_tensors(conditioned).items()}
    assignments = torch.as_tensor(state["assignments"], dtype=torch.long, device=device)
    embeddings = model.condition(tensors)
    scores = decoder.score_assignments(embeddings[0], embeddings[1], assignments)
    consumption["scoring_forwards"] += 1
    if not _finite(scores):
        raise RuntimeError("non-finite score")
    return scores


def _choose_local(scores: np.ndarray, candidate_ids: np.ndarray) -> int:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    identities = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    if values.shape != identities.shape or int(values.size) != 128:
        raise RuntimeError("scores are not aligned to the 128 local positions")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("non-finite score")
    tied = np.flatnonzero(values == np.max(values))
    return int(tied[int(np.argmin(identities[tied]))])


def _reading(state: dict, scores: np.ndarray) -> dict:
    local = _choose_local(scores, state["candidate_id"])
    real = state["real_payload"]
    zero = state["zero_payload"]
    return {
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "local_position": local,
        "candidate_id": int(state["candidate_id"][local]),
        "row_index": int(state["row_index"][local]),
        "assignment": state["assignment_text"][local],
        "real_payload_bits": float(real[local]),
        "real_regret_bits": float(np.max(real) - real[local]),
        "zero_payload_bits": float(zero[local]),
        "zero_regret_bits": float(np.max(zero) - zero[local]),
        "score": float(scores[local]),
        "tie_count": int(np.sum(np.asarray(scores) == np.max(scores))),
    }


def _evaluate(model, decoder, states: list[dict], device: torch.device, consumption: dict) -> list[dict]:
    model.eval()
    rows = []
    with torch.no_grad():
        for state in states:
            scores = _score(model, decoder, state, device, consumption).detach().cpu().numpy()
            rows.append(_reading(state, scores))
    return rows


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _subset(rows: list[dict], scenario: int | None = None, time_step: int | None = None) -> list[dict]:
    return [
        row for row in rows
        if (scenario is None or int(row["scenario_id"]) == int(scenario))
        and (time_step is None or int(row["time_step"]) == int(time_step))
    ]


def _summarize(init_rows: list[dict], arm_a: list[dict], arm_b: list[dict]) -> dict:
    by_key = {
        "init": {(int(row["scenario_id"]), int(row["time_step"])): row for row in init_rows},
        "A": {(int(row["scenario_id"]), int(row["time_step"])): row for row in arm_a},
        "B": {(int(row["scenario_id"]), int(row["time_step"])): row for row in arm_b},
    }
    per_state = []
    for key in sorted(by_key["A"]):
        left, right, start = by_key["A"][key], by_key["B"][key], by_key["init"][key]
        per_state.append(
            {
                "scenario_id": key[0],
                "time_step": key[1],
                "init": start,
                "A": left,
                "B": right,
                "B_minus_A_real_payload_bits": float(right["real_payload_bits"] - left["real_payload_bits"]),
                "B_minus_A_real_regret_bits": float(right["real_regret_bits"] - left["real_regret_bits"]),
                "A_minus_init_real_payload_bits": float(left["real_payload_bits"] - start["real_payload_bits"]),
                "B_minus_init_real_payload_bits": float(right["real_payload_bits"] - start["real_payload_bits"]),
                "selection_changed": int(left["candidate_id"]) != int(right["candidate_id"]),
            }
        )

    def pack(rows: list[dict]) -> dict:
        return {
            "states": len(rows),
            "A_real_payload_bits": _mean([row["A"] for row in rows], "real_payload_bits"),
            "B_real_payload_bits": _mean([row["B"] for row in rows], "real_payload_bits"),
            "init_real_payload_bits": _mean([row["init"] for row in rows], "real_payload_bits"),
            "A_real_regret_bits": _mean([row["A"] for row in rows], "real_regret_bits"),
            "B_real_regret_bits": _mean([row["B"] for row in rows], "real_regret_bits"),
            "init_real_regret_bits": _mean([row["init"] for row in rows], "real_regret_bits"),
            "B_minus_A_real_payload_bits": _mean(rows, "B_minus_A_real_payload_bits"),
            "A_zero_payload_bits": _mean([row["A"] for row in rows], "zero_payload_bits"),
            "B_zero_payload_bits": _mean([row["B"] for row in rows], "zero_payload_bits"),
            "selection_changed": int(sum(bool(row["selection_changed"]) for row in rows)),
            "B_worse_than_A": int(sum(float(row["B_minus_A_real_payload_bits"]) < 0.0 for row in rows)),
        }

    return {
        "per_state": per_state,
        "t15": pack(_subset(per_state, time_step=15)),
        "t30": pack(_subset(per_state, time_step=30)),
        "scenario_322022": pack(_subset(per_state, scenario=322022)),
        "scenario_322023": pack(_subset(per_state, scenario=322023)),
        "all_four_states": pack(per_state),
        "generalization_interval": None,
    }


def _path_check(repo: Path, data: Path) -> dict:
    before = _sha256(data / "manifest.json")
    source_bytes = {path: _sha256(path) for path in (data / "source_states").glob("scenario_*/*.npz")}
    with tempfile.TemporaryDirectory(prefix="legacy_l3_path_") as temporary:
        output = Path(temporary) / "package"
        prepare_from_existing(
            repo,
            data / "source_states",
            repo / "results/research/urgency_arrival_aware_expansion/labels.csv",
            repo / "results/research/urgency_arrival_aware_expansion/selected_candidates.csv",
            output,
        )
        loaded = load_package(output, output)
        graph_paths = []
        for split in ("train", "explore_eval"):
            for state in loaded[split]:
                resolved = resolve_recorded(output, state["graph_file"])
                graph_paths.append(state["graph_file"])
                if output.resolve() not in resolved.parents:
                    raise RuntimeError(f"{state['graph_file']} does not point inside the new package")
        if len(graph_paths) != 16 or len(loaded["train"]) != 12 or len(loaded["explore_eval"]) != 4:
            raise RuntimeError("path check did not read 12 training and 4 explore states")
    after = _sha256(data / "manifest.json")
    if before != after:
        raise RuntimeError("path check changed the L2 manifest")
    for path, digest in source_bytes.items():
        if _sha256(path) != digest:
            raise RuntimeError(f"path check changed {path}")
    return {"status": "passed", "graphs": 16, "train_states": 12, "explore_states": 4, "l2_manifest_unchanged": True}


def resolve_recorded(package_root: Path, graph_file: str) -> Path:
    from wt.research.arrival_aware.loader import resolve_graph

    return resolve_graph(package_root, graph_file, package_root)


def _check_models(data: Path, repo: Path, device: torch.device, consumption: dict) -> dict:
    loaded = load_package(data, repo)
    if len(loaded["train"]) != 12 or len(loaded["explore_eval"]) != 4:
        raise RuntimeError("runner did not keep the fixed 12/4 split")
    epoch_path = repo / "checkpoints/initialization/joint_utility_epoch6.pt"
    decoder_path = repo / "checkpoints/formal/matching_decoder.pt"
    if _sha256(epoch_path) != EXPECTED_EPOCH6 or _sha256(decoder_path) != EXPECTED_DECODER:
        raise RuntimeError("initialization file hash changed")
    _seed()
    model_a, decoder_a, optimizer_a, _payload = _build_pair(epoch_path, decoder_path, device)
    model_b, decoder_b, optimizer_b, _payload_b = _build_pair(epoch_path, decoder_path, device)
    if _weights(model_a) != _weights(model_b) or _weights(decoder_a) != _weights(decoder_b):
        raise RuntimeError("the two arms do not start from the same tensors")
    if any(parameter.requires_grad for parameter in decoder_a.parameters()):
        raise RuntimeError("decoder parameter is trainable")
    if not all(parameter.requires_grad for parameter in model_a.parameters()):
        raise RuntimeError("encoder parameter is frozen")
    state = loaded["train"][0]
    if list(map(int, state["local_position"])) != list(range(128)):
        raise RuntimeError("local positions are not 0..127")
    if int(np.max(state["candidate_id"])) < 128:
        raise RuntimeError("the mapping check needs a candidate id outside the local range")
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        scores_a = _score(model_a, decoder_a, state, device, consumption).detach().cpu().numpy()
        scores_b = _score(model_b, decoder_b, state, device, consumption).detach().cpu().numpy()
    if not np.array_equal(scores_a, scores_b):
        raise RuntimeError("initial scores differ between arms")
    reading = _reading(state, scores_a)
    local = int(reading["local_position"])
    if reading["candidate_id"] != int(state["candidate_id"][local]):
        raise RuntimeError("local position does not map back to the candidate")
    if reading["assignment"] != state["assignment_text"][local] or reading["real_payload_bits"] != float(state["real_payload"][local]):
        raise RuntimeError("local position does not map the assignment and real label")
    if int(state["candidate_id"][local]) == local and int(np.max(state["candidate_id"])) < 128:
        raise RuntimeError("candidate id was used as the local index")
    del model_a, model_b, decoder_a, decoder_b, optimizer_a, optimizer_b
    gc.collect()
    return {
        "status": "passed",
        "strict_epoch6": True,
        "decoder_frozen": True,
        "initial_scores_identical": True,
        "checked_state": [int(state["scenario_id"]), int(state["time_step"])],
        "checked_candidate_id": reading["candidate_id"],
        "checked_local_position": reading["local_position"],
        "updates": 0,
    }


def _train_arm(name: str, model, decoder, optimizer, states: list[dict], device: torch.device, log_path: Path, consumption: dict) -> None:
    objective = LOSSES["c"]
    label_key = LABELS[name]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        decoder.eval()
        for index, state in enumerate(states, start=1):
            optimizer.zero_grad(set_to_none=True)
            scores = _score(model, decoder, state, device, consumption)
            labels = torch.as_tensor(state[label_key], dtype=scores.dtype)
            loss = objective(scores, labels)
            if not _finite(loss):
                raise RuntimeError(f"non-finite loss on arm {name}")
            loss.backward()
            consumption["backward_passes"] += 1
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or any(not _finite(gradient) for gradient in gradients):
                raise RuntimeError(f"non-finite gradient on arm {name}")
            grad_norm = float(clip_grad_norm_(model.parameters(), float(GRAD_CLIP)))
            if not np.isfinite(grad_norm):
                raise RuntimeError("non-finite gradient norm")
            optimizer.step()
            consumption["training_updates"] += 1
            record = {
                "arm": name,
                "epoch": epoch,
                "state_index": index,
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "loss": float(loss.detach().cpu()),
                "grad_norm": grad_norm,
                "update": int(consumption["training_updates"]),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"arm {name} epoch {epoch} {index}/12 loss={record['loss']:.6f} grad={grad_norm:.4f}", flush=True)


def _save_arm(path: Path, name: str, model, updates: int) -> str:
    torch.save({"epoch": EPOCHS, "arm": name, "label": LABELS[name], "model": model.state_dict(), "updates": updates}, path)
    return _sha256(path)


def _run_pair(repo: Path, data: Path, output: Path) -> dict:
    if (output / "manifest.json").exists() or (output / "failure.json").exists():
        raise RuntimeError("refusing to repeat an existing L3 output")
    output.mkdir(parents=True, exist_ok=True)
    consumption = {"environment_steps": 0, "low_level_inferences": 0, "scoring_forwards": 0, "training_updates": 0, "backward_passes": 0}
    try:
        path_check = _path_check(repo, data)
        (output / "path_check.json").write_text(json.dumps(path_check, indent=2) + "\n", encoding="utf-8")
        device = torch.device("cpu")
        _configure_runtime()
        _seed()
        check = _check_models(data, repo, device, consumption)
        (output / "model_check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
        _seed()
        loaded = load_package(data, repo)
        epoch_path = repo / "checkpoints/initialization/joint_utility_epoch6.pt"
        decoder_path = repo / "checkpoints/formal/matching_decoder.pt"
        before_epoch = _sha256(epoch_path)
        before_decoder = _sha256(decoder_path)
        model_a, decoder_a, optimizer_a, _payload = _build_pair(epoch_path, decoder_path, device)
        model_b, decoder_b, optimizer_b, _ignored = _build_pair(epoch_path, decoder_path, device)
        if _weights(model_a) != _weights(model_b):
            raise RuntimeError("rebuilt arms diverged before training")
        decoder_before = _weights(decoder_a)
        init_rows = _evaluate(model_a, decoder_a, loaded["explore_eval"], device, consumption)
        (output / "init_reference.json").write_text(json.dumps(init_rows, indent=2) + "\n", encoding="utf-8")
        log_path = output / "train_log.jsonl"
        _train_arm("A", model_a, decoder_a, optimizer_a, loaded["train"], device, log_path, consumption)
        _train_arm("B", model_b, decoder_b, optimizer_b, loaded["train"], device, log_path, consumption)
        if int(consumption["training_updates"]) != 240:
            raise RuntimeError(f"completed updates are {consumption['training_updates']}")
        arm_a = _evaluate(model_a, decoder_a, loaded["explore_eval"], device, consumption)
        arm_b = _evaluate(model_b, decoder_b, loaded["explore_eval"], device, consumption)
        if _weights(decoder_a) != decoder_before or _weights(decoder_b) != decoder_before:
            raise RuntimeError("decoder weights changed")
        summary = _summarize(init_rows, arm_a, arm_b)
        (output / "explore_arm_a.json").write_text(json.dumps(arm_a, indent=2) + "\n", encoding="utf-8")
        (output / "explore_arm_b.json").write_text(json.dumps(arm_b, indent=2) + "\n", encoding="utf-8")
        (output / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        arm_a_hash = _save_arm(output / "arm_a_epoch_010.pt", "A", model_a, 120)
        arm_b_hash = _save_arm(output / "arm_b_epoch_010.pt", "B", model_b, 120)
        if _sha256(epoch_path) != before_epoch or _sha256(decoder_path) != before_decoder:
            raise RuntimeError("source checkpoint bytes changed")
        manifest = {
            "package": "legacy_arrival_aware_ab_l3",
            "parent_instruction": "55483c752aa738b4fb0a5c1149f9439a9cd5d2a2",
            "data_package": "results/research/arrival_aware/state_preparation",
            "command": "py -3 -m wt.research.arrival_aware.runner --run-pair --data results/research/arrival_aware/state_preparation --output results/research/arrival_aware/listnet_comparison",
            "seed": SEED,
            "epochs": EPOCHS,
            "learning_rate": float(LEARNING_RATE),
            "weight_decay": float(WEIGHT_DECAY),
            "grad_clip": float(GRAD_CLIP),
            "loss": "top_regret_loss",
            "arms": {"A": "zero_payload_bits", "B": "real_payload_bits"},
            "epoch6_sha256": before_epoch,
            "decoder_sha256": before_decoder,
            "arm_a_epoch_010_sha256": arm_a_hash,
            "arm_b_epoch_010_sha256": arm_b_hash,
            "path_check": path_check,
            "model_check": check,
            "summary": summary,
            **consumption,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"updates": consumption["training_updates"], "scoring_forwards": consumption["scoring_forwards"], "B_minus_A": summary["all_four_states"]["B_minus_A_real_payload_bits"]}, sort_keys=True), flush=True)
        return manifest
    except Exception as exc:
        failure = {"error": f"{type(exc).__name__}: {exc}", **consumption}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise


def _config() -> dict:
    return {
        "mode": "config_only",
        "seed": SEED,
        "epochs": EPOCHS,
        "updates_per_arm": 120,
        "updates_total": 240,
        "train_scenarios": list(range(322016, 322022)),
        "explore_scenarios": [322022, 322023],
        "loss": "top_regret_loss",
        "optimizer": {"name": "Adam", "lr": float(LEARNING_RATE), "weight_decay": float(WEIGHT_DECAY), "grad_clip": float(GRAD_CLIP)},
        "commands": {
            "check": "py -3 -m wt.research.arrival_aware.runner --check-only --data results/research/arrival_aware/state_preparation --output results/research/arrival_aware/listnet_comparison",
            "run": "py -3 -m wt.research.arrival_aware.runner --run-pair --data results/research/arrival_aware/state_preparation --output results/research/arrival_aware/listnet_comparison",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/research/arrival_aware/state_preparation"))
    parser.add_argument("--output", type=Path, default=Path("results/research/arrival_aware/listnet_comparison"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-pair", action="store_true")
    args = parser.parse_args()
    if args.check_only and args.run_pair:
        raise RuntimeError("choose one of --check-only or --run-pair")
    if not args.check_only and not args.run_pair:
        print(json.dumps(_config(), indent=2), flush=True)
        return
    repo = _repo_root()
    data = args.data if args.data.is_absolute() else repo / args.data
    output = args.output if args.output.is_absolute() else repo / args.output
    if args.check_only:
        _configure_runtime()
        consumption = {"environment_steps": 0, "low_level_inferences": 0, "scoring_forwards": 0, "training_updates": 0, "backward_passes": 0}
        path_check = _path_check(repo, data)
        check = _check_models(data, repo, torch.device("cpu"), consumption)
        print(json.dumps({"path_check": path_check, "model_check": check, **consumption}, indent=2), flush=True)
        return
    _run_pair(repo, data, output)


if __name__ == "__main__":
    main()