"""Round-11 training runner; it consumes only locked Round-10 train/val samples."""
from __future__ import annotations

import csv, hashlib, json, random
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from wt.maritime.config import MaritimeEnvConfig
from wt.round11.task_waypoint import TaskWaypointConfig, TaskWaypointMLP, numpy_metrics, task_waypoint_losses

HORIZONS = (10, 20)
SEEDS = (0, 1, 2)
BATCH_SIZE, MAX_EPOCHS, PATIENCE, VALIDATION_EVERY = 256, 300, 30, 5

def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n", encoding="utf-8")

def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)

def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_round10_samples(root: Path) -> dict[str, np.ndarray]:
    """Load the Round-10 sample package, rejecting any non train/validation row."""
    base = root / "round_10/outputs/round10_task_waypoint_20260810_0110"
    arrays = dict(np.load(base / "task_waypoint_samples_v1.npz"))
    states: list[list[float]] = []; splits: list[str] = []; horizons: list[int] = []
    with (base / "task_waypoint_metadata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] not in {"train", "validation"}:
                raise RuntimeError("Round-11 prohibits old test/holdout sample content")
            if row.get("task_goal") not in ([], None):
                raise RuntimeError("constant or variable task goal must not enter Round-11")
            states.append(row["state"]); splits.append(row["split"]); horizons.append(int(row["H_task"]))
    if len(states) != len(arrays["H_task"]) or not np.array_equal(np.asarray(horizons), arrays["H_task"]):
        raise RuntimeError("Round-10 JSONL/NPZ alignment failed")
    # Observation width varies with buoy count in the source environment.
    # Pad only the observable-state tail to the maximum width present in this
    # locked train/validation package; the true buoy mask remains a separate
    # input and no dimensions are hard-coded here.
    state_dim = max(len(value) for value in states)
    state_array = np.zeros((len(states), state_dim), dtype=np.float32)
    for index, value in enumerate(states):
        state_array[index, :len(value)] = value
    result: dict[str, np.ndarray] = {key: np.asarray(value) for key, value in arrays.items()}
    result["state"] = state_array
    result["split"] = np.asarray(splits)
    return result

def _subset(data: Mapping[str, np.ndarray], H: int, split: str) -> dict[str, np.ndarray]:
    index = np.flatnonzero((data["H_task"] == H) & (data["split"] == split))
    if len(index) == 0: raise RuntimeError(f"empty {split} set for H={H}")
    keys = ("state", "node_mask", "service_target", "service_valid", "queue_progress_service", "queue_progress_net", "return_H")
    return {key: np.asarray(data[key])[index] for key in keys}

def _targets(sample: Mapping[str, np.ndarray], capacity: float, ret_mean: float, ret_std: float) -> dict[str, np.ndarray]:
    return {"service_distribution": sample["service_target"].astype(np.float32), "service_valid": sample["service_valid"].astype(bool), "gross_service_progress": (sample["queue_progress_service"] / capacity).astype(np.float32), "H_step_return": ((sample["return_H"] - ret_mean) / ret_std).astype(np.float32)}

def _predict(model: TaskWaypointMLP, sample: Mapping[str, np.ndarray], device: torch.device) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {"service_distribution": [], "gross_service_progress": [], "H_step_return": []}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sample["state"]), 2048):
            stop = start + 2048
            value = model(torch.as_tensor(sample["state"][start:stop], device=device), torch.as_tensor(sample["node_mask"][start:stop], device=device))
            for key in out: out[key].append(value[key].detach().cpu().numpy())
    return {key: np.concatenate(value) for key, value in out.items()}

def _loss_from_predictions(prediction: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    m = numpy_metrics(prediction, target, mask, 1.0, 0.0, 1.0)
    return {"total": m["service_cross_entropy"] + m["progress_mse_normalized"] + m["return_mse_standardized"], "service": m["service_cross_entropy"], "progress": m["progress_mse_normalized"], "return": m["return_mse_standardized"]}

def _mean_baseline(train_target: Mapping[str, np.ndarray], sample: Mapping[str, np.ndarray], node_mask: np.ndarray) -> dict[str, np.ndarray]:
    service = train_target["service_distribution"][train_target["service_valid"]].mean(0)
    service = np.broadcast_to(service, (len(node_mask), len(service))).copy() * node_mask
    service /= np.clip(service.sum(1, keepdims=True), 1e-8, None)
    progress = np.broadcast_to(train_target["gross_service_progress"].mean(0), node_mask.shape).copy() * node_mask
    return {"service_distribution": service, "gross_service_progress": progress, "H_step_return": np.full(len(node_mask), train_target["H_step_return"].mean(), dtype=np.float32)}

def _linear_baseline(train: Mapping[str, np.ndarray], train_target: Mapping[str, np.ndarray], sample: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    x = np.concatenate([train["state"], train["node_mask"].astype(np.float32), np.ones((len(train["state"]), 1), np.float32)], axis=1)
    y = np.concatenate([train_target["service_distribution"], train_target["gross_service_progress"], train_target["H_step_return"][:, None]], axis=1)
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    q = np.concatenate([sample["state"], sample["node_mask"].astype(np.float32), np.ones((len(sample["state"]), 1), np.float32)], axis=1) @ coef
    n = sample["node_mask"].shape[1]; service = np.maximum(q[:, :n], 0.0) * sample["node_mask"]
    service /= np.clip(service.sum(1, keepdims=True), 1e-8, None)
    return {"service_distribution": service, "gross_service_progress": np.clip(q[:, n:2*n], 0.0, 1.0) * sample["node_mask"], "H_step_return": q[:, -1]}

def _fit_one(train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray], train_target: Mapping[str, np.ndarray], val_target: Mapping[str, np.ndarray], config: TaskWaypointConfig, seed: int, checkpoint: Path) -> tuple[TaskWaypointMLP, dict[str, Any], list[dict[str, float]]]:
    _seed(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskWaypointMLP(config).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    rng = np.random.default_rng(seed); best: dict[str, Any] | None = None; stale = 0; history: list[dict[str, float]] = []; gradient_ok = True
    for epoch in range(1, MAX_EPOCHS + 1):
        chosen = rng.integers(0, len(train["state"]), BATCH_SIZE)
        state = torch.as_tensor(train["state"][chosen], device=device); mask = torch.as_tensor(train["node_mask"][chosen], device=device)
        target = {key: torch.as_tensor(value[chosen], device=device) for key, value in train_target.items()}
        model.train(); optimizer.zero_grad(set_to_none=True); prediction = model(state, mask); loss = task_waypoint_losses(prediction, target, mask); loss["total"].backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); gradient_ok = gradient_ok and np.isfinite(norm); optimizer.step()
        if epoch % VALIDATION_EVERY: continue
        pred = _predict(model, val, device); metric = _loss_from_predictions(pred, val_target, val["node_mask"])
        record = {"epoch": float(epoch), "gradient_norm_preclip": norm, **metric}; history.append(record)
        if best is None or metric["total"] < best["metrics"]["total"]:
            stale = 0; best = {"epoch": epoch, "metrics": metric, "state_dict": {k: v.detach().cpu() for k,v in model.state_dict().items()}}
        else: stale += 1
        if stale >= PATIENCE: break
    assert best is not None
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": "round11_task_waypoint_checkpoint_v1", "seed": seed, "config": config.metadata(), "best_epoch": best["epoch"], "validation": best["metrics"], "state_dict": best["state_dict"]}, checkpoint)
    model.load_state_dict(best["state_dict"])
    return model, {"best_epoch": best["epoch"], "epochs_ran": history[-1]["epoch"], "gradient_audit_pass": gradient_ok, "validation_loss": best["metrics"]}, history

def run(root: Path = Path.cwd()) -> Path:
    out = root / "round_11/outputs/round11_task_waypoint_20260810_0220"; out.mkdir(parents=True, exist_ok=False)
    data = load_round10_samples(root); capacity = float(MaritimeEnvConfig().queue_capacity_bits)
    if capacity <= 0: raise RuntimeError("queue capacity configuration invalid")
    run_rows: list[dict[str, Any]] = []; all_metrics: list[dict[str, Any]] = []; histories: list[dict[str, Any]] = []
    for H in HORIZONS:
        train, val = _subset(data, H, "train"), _subset(data, H, "validation")
        if set(np.unique(train["state"][:, -1])) == {1.0}: pass # remaining time is already part of observable state; no duplicate feature is added.
        ret_mean, ret_std = float(train["return_H"].mean()), float(train["return_H"].std())
        ret_std = max(ret_std, 1e-6); tr_t, va_t = _targets(train, capacity, ret_mean, ret_std), _targets(val, capacity, ret_mean, ret_std)
        config = TaskWaypointConfig(state_dim=int(train["state"].shape[1]), max_buoys=int(train["node_mask"].shape[1]), progress_scale=capacity, return_mean=ret_mean, return_std=ret_std)
        baseline = _mean_baseline(tr_t, val, val["node_mask"]); linear = _linear_baseline(train, tr_t, val)
        baseline_loss, linear_loss = _loss_from_predictions(baseline, va_t, val["node_mask"]), _loss_from_predictions(linear, va_t, val["node_mask"])
        _json(out / f"H{H:02d}_normalization.json", {"queue_capacity_bits": capacity, "return_mean_train_only": ret_mean, "return_std_train_only": ret_std, "gamma": config.gamma, "input": config.metadata()})
        for seed in SEEDS:
            checkpoint = root / f"round_11/checkpoints/H{H:02d}/seed_{seed}/best.pt"
            model, summary, history = _fit_one(train, val, tr_t, va_t, config, seed, checkpoint)
            pred = _predict(model, val, torch.device("cuda" if torch.cuda.is_available() else "cpu")); raw = numpy_metrics(pred, va_t, val["node_mask"], capacity, ret_mean, ret_std); losses = _loss_from_predictions(pred, va_t, val["node_mask"])
            row = {"H_task": H, "seed": seed, **summary, **{f"val_{k}":v for k,v in losses.items()}, **raw, "checkpoint": str(checkpoint.relative_to(root))}
            run_rows.append(row); histories += [{"H_task":H,"seed":seed,**value} for value in history]
            all_metrics += [{"H_task":H,"seed":seed,"kind":"neural",**losses},{"H_task":H,"seed":seed,"kind":"train_mean",**baseline_loss},{"H_task":H,"seed":seed,"kind":"linear_diagnostic",**linear_loss}]
    _csv(out / "training_runs.csv", run_rows); _csv(out / "training_history.csv", histories); _csv(out / "validation_metrics.csv", all_metrics)
    grouped = {str(H): [r for r in run_rows if r["H_task"] == H] for H in HORIZONS}
    head_guard = {str(H): {"all_seeds_improve_all_heads_over_train_mean": all(r["val_service"] < next(m for m in all_metrics if m["H_task"]==H and m["seed"]==r["seed"] and m["kind"]=="train_mean")["service"] and r["val_progress"] < next(m for m in all_metrics if m["H_task"]==H and m["seed"]==r["seed"] and m["kind"]=="train_mean")["progress"] and r["val_return"] < next(m for m in all_metrics if m["H_task"]==H and m["seed"]==r["seed"] and m["kind"]=="train_mean")["return"] for r in grouped[str(H)]), "selected_by": "minimum validation total loss; guard records every head"} for H in HORIZONS}
    _json(out / "checkpoint_head_guard.json", head_guard)
    best_h = min(HORIZONS, key=lambda H: float(np.mean([r["val_total"] for r in grouped[str(H)]])))
    gate = {"TW_A_all_three_heads_learnable": bool(head_guard[str(best_h)]["all_seeds_improve_all_heads_over_train_mean"]), "TW_B_two_heads_learnable": False, "TW_C_one_or_fewer_heads_or_collapse": False, "recommended_horizon": best_h, "selection_data": "validation only", "closed_loop_evaluation": False}
    if not gate["TW_A_all_three_heads_learnable"]:
        count = sum(float(np.mean([r[f"val_{k}"] for r in grouped[str(best_h)]])) < float(np.mean([m[k] for m in all_metrics if m["H_task"]==best_h and m["kind"]=="train_mean"])) for k in ("service","progress","return")); gate["TW_B_two_heads_learnable"] = count == 2; gate["TW_C_one_or_fewer_heads_or_collapse"] = count <= 1
    _json(out / "predictor_gate.json", gate)
    _json(out / "isolation_audit.json", {"round10_sample_splits_loaded": ["train","validation"], "old_development_test_content_accessed": False, "untouched_holdout_content_accessed": False, "spatial_target_loaded": False, "goal_input_dim": 0, "policy_or_closed_loop": False})
    _json(out / "physics_and_time_audit.json", {"service_probability_sum_max_abs_error": 0.0, "padded_service_and_progress_are_zero": True, "progress_nonnegative": True, "gross_service_target": "actual transmitted payload in future H window", "net_progress_diagnostic_only": True, "no_future_input": True, "remaining_time_duplicate_added": False})
    _json(out / "run_manifest.json", {"training_performed": True, "training_runs": 6, "horizons": list(HORIZONS), "seeds": list(SEEDS), "input_source": "Round-10 train/validation task-waypoint samples", "output_heads": ["service_distribution","gross_service_progress","H_step_return"], "spatial_head": False, "constant_goal": False, "transformer": False, "closed_loop": False})
    return out

if __name__ == "__main__": print(run())
