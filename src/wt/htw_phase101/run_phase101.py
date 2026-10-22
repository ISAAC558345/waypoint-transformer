"""Phase-10.1 smoke + single-seed L3-B low-level PPO pilot. No high-level updates."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch

from wt.htw_phase100.protocol import ARCHIVE
from wt.round30.ppo import PPOAgent, PPOConfig, load_agent, seed_everything
from wt.round30.runner import collect

from .evaluate import evaluate_pair
from .features import build_schema
from .high_level import high_grad_count, load_frozen_high, parameter_sha, refuse_m5
from .models import L3BGoalActorCritic
from .protocol import (
    CHECKPOINT_EVERY,
    DESIGN_TRAIN_SEEDS,
    DESIGN_VAL_SEEDS,
    FORBIDDEN_M5_CKPT,
    HIGH_CKPT,
    HIGH_MODEL_SEED,
    HOLD,
    LOW_MODEL_SEED,
    OUTPUT,
    PACK_REPORT,
    PILOT_UPDATES,
    PPO,
    REPORT,
    SMOKE_UPDATES,
    VECTOR_SEED_PILOT,
    VECTOR_SEED_SMOKE,
    dump_frozen,
    sha256_bytes,
)
from .vector import L3BLowLevelVector


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _numeric_history(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        for key, value in list(item.items()):
            if key == "label":
                continue
            try:
                item[key] = float(value)
            except (TypeError, ValueError):
                continue
        if "update" in item:
            item["update"] = int(item["update"])
        out.append(item)
    return out


def _flatten_eval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                flat[key] = json.dumps(value)
            else:
                flat[key] = value
        out.append(flat)
    return out
    out = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                flat[key] = json.dumps(value)
            else:
                flat[key] = value
        out.append(flat)
    return out


def _save_ckpt(agent: PPOAgent, path: Path, extra: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    refuse_m5(path)
    torch.save(agent.checkpoint(seed=LOW_MODEL_SEED, steps=int(extra.get("environment_steps", 0)), extra=dict(extra)), path)
    return sha256_bytes(path.read_bytes())


def _load_low(path: Path, schema: dict[str, Any], config: PPOConfig) -> PPOAgent:
    refuse_m5(path)
    if path.resolve() == Path(FORBIDDEN_M5_CKPT).resolve():
        raise RuntimeError("refused M=5 checkpoint")
    payload = torch.load(path, map_location="cpu")
    extra = payload.get("extra", {})
    if int(payload.get("model_shape", {}).get("max_uavs", -1)) == 5 or extra.get("max_uavs") == 5:
        raise RuntimeError("checkpoint declares max_uavs=5")
    model = L3BGoalActorCritic(schema)
    return load_agent(payload, config, model)


def _assert_high_frozen(high, file_sha: str, param_sha: str) -> None:
    if sha256_bytes(HIGH_CKPT.read_bytes()) != file_sha:
        raise RuntimeError("frozen high-level checkpoint file SHA changed")
    if parameter_sha(high) != param_sha:
        raise RuntimeError("frozen high-level parameters changed")
    if high_grad_count(high) != 0:
        raise RuntimeError("high-level requires_grad was re-enabled")


def _make_agent(schema: dict[str, Any], config: PPOConfig) -> PPOAgent:
    return PPOAgent(L3BGoalActorCritic(schema), config)


def _optimizer_disjoint(agent: PPOAgent, high) -> bool:
    low_ids = {id(p) for p in agent.model.parameters()}
    high_ids = {id(p) for p in high.parameters()}
    opt_ids = {id(p) for group in agent.optimizer.param_groups for p in group["params"]}
    return not (opt_ids & high_ids) and opt_ids <= low_ids


def restore_phase100() -> dict[str, Any]:
    report = REPORT.parent / "HTW_Phase10_0_high_level_freeze_and_low_level_handoff.md"
    equiv = ARCHIVE.parent / "phase100_rule_executor_equivalence.json"
    return {
        "phase100_report_exists": report.is_file(),
        "high_ckpt": str(HIGH_CKPT),
        "high_ckpt_exists": HIGH_CKPT.is_file(),
        "high_file_sha256": sha256_bytes(HIGH_CKPT.read_bytes()) if HIGH_CKPT.is_file() else None,
        "equivalence_exists": equiv.is_file(),
        "equivalence_all_ok": json.loads(equiv.read_text(encoding="utf-8")).get("all_stepwise_equivalent") if equiv.is_file() else None,
        "high_model_seed": HIGH_MODEL_SEED,
        "low_model_seed": LOW_MODEL_SEED,
        "current_executor_before_pilot": "fly-toward",
        "m5_not_loadable": True,
        "pool_size_24_recorded": True,
        "checkpoint_csv_nan_recorded": True,
        "gain_wording": "收益对齐训练流程整体改进",
        "design_train_seeds": list(DESIGN_TRAIN_SEEDS),
        "design_val_seeds": list(DESIGN_VAL_SEEDS),
        "design_val_is_fresh": False,
        "confirmation_accessed": False,
    }


def run_updates(
    *,
    n_updates: int,
    vector_seed: int,
    label: str,
    schema: dict[str, Any],
    high,
    high_file_sha: str,
    high_param_sha: str,
    save_schedule: bool,
) -> dict[str, Any]:
    seed_everything(LOW_MODEL_SEED)
    config = PPO
    agent = _make_agent(schema, config)
    if not _optimizer_disjoint(agent, high):
        raise RuntimeError("low-level optimizer would touch high-level parameters")
    vector = L3BLowLevelVector(DESIGN_TRAIN_SEEDS, config.num_envs, vector_seed, high_model=high, schema=schema)
    history = []
    runtime = []
    ckpts = []
    init_path = OUTPUT / "checkpoints" / f"{label}_init.pt"
    ckpts.append({"tag": "init", "update": 0, "sha256": _save_ckpt(agent, init_path, {"environment_steps": 0, "max_uavs": 6, "label": label, "not_m5": True}), "path": str(init_path)})
    for update in range(1, int(n_updates) + 1):
        t0 = perf_counter()
        vector.reward_parts = {"r_env": [], "r_aux": [], "r_train": []}
        storage, events, health = collect(agent, vector)
        if storage["pre_tanh"].shape[-2:] != (int(schema["max_uavs"]), 2):
            raise RuntimeError("sampled action is not (6,2)")
        stats = agent.update(storage)
        elapsed = perf_counter() - t0
        _assert_high_frozen(high, high_file_sha, high_param_sha)
        row = {
            "label": label,
            "update": update,
            "environment_steps": vector.environment_steps,
            "seconds": elapsed,
            "completed_episodes": len(events),
            "mean_completed_payload_bits": float(np.mean([item["payload_bits"] for item in events])) if events else float("nan"),
            "mean_completed_env_return": float(np.mean([item["return"] for item in events])) if events else float("nan"),
            **vector.parts_mean(),
            **health,
            **{key: float(value) for key, value in stats.items()},
        }
        if not all(np.isfinite(value) for value in row.values() if isinstance(value, float)):
            raise RuntimeError(f"{label} produced NaN/Inf at update {update}")
        history.append(row)
        runtime.append({"label": label, "update": update, "seconds": elapsed, "environment_steps": vector.environment_steps})
        if save_schedule and (update % int(CHECKPOINT_EVERY) == 0 or update == int(n_updates)):
            path = OUTPUT / "checkpoints" / f"{label}_update{update:03d}.pt"
            ckpts.append({"tag": f"update_{update}", "update": update, "sha256": _save_ckpt(agent, path, {"environment_steps": vector.environment_steps, "max_uavs": 6, "label": label}), "path": str(path)})
        print(f"{label} update {update}/{n_updates} steps={vector.environment_steps} sec={elapsed:.3f} payload={row['mean_completed_payload_bits']}", flush=True)
    final_path = OUTPUT / "checkpoints" / f"{label}_final.pt"
    if save_schedule:
        ckpts.append({"tag": "final", "update": int(n_updates), "sha256": _save_ckpt(agent, final_path, {"environment_steps": vector.environment_steps, "max_uavs": 6, "label": label, "primary": True}), "path": str(final_path)})
    _assert_high_frozen(high, high_file_sha, high_param_sha)
    return {"agent": agent, "history": history, "runtime": runtime, "checkpoints": ckpts, "environment_steps": vector.environment_steps, "init_path": str(init_path), "final_path": str(final_path)}


def _plot(history: list[dict[str, Any]], paired: list[dict[str, Any]]) -> dict[str, str]:
    paths = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return {"error": str(error)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([row["update"] for row in history], [row["mean_r_env"] for row in history], label="r_env")
    axes[0].plot([row["update"] for row in history], [row["mean_r_aux"] for row in history], label="r_aux")
    axes[0].plot([row["update"] for row in history], [row["mean_r_train"] for row in history], label="r_train")
    axes[0].set_xlabel("update")
    axes[0].set_ylabel("mean reward")
    axes[0].legend()
    axes[0].set_title("pilot learning (do not compare r_train to B0 env return)")
    payloads = [row["mean_completed_payload_bits"] for row in history]
    axes[1].plot([row["update"] for row in history], payloads, label="train episode payload")
    axes[1].set_xlabel("update")
    axes[1].set_ylabel("bits")
    axes[1].set_title("completed-episode payload during training")
    fig.tight_layout()
    curve = OUTPUT / "phase101_learning_curves.png"
    fig.savefig(curve, dpi=120)
    plt.close(fig)
    paths["learning_curves"] = str(curve)
    if paired:
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = [str(row["scenario"]) for row in paired]
        ax.bar(np.arange(len(xs)) - 0.2, [float(row["b0_payload_bits"]) for row in paired], width=0.4, label="B0 fly-toward")
        ax.bar(np.arange(len(xs)) + 0.2, [float(row["b1_payload_bits"]) for row in paired], width=0.4, label="B1 PPO final")
        ax.set_xticks(np.arange(len(xs)), xs)
        ax.set_xlabel("design val seed")
        ax.set_ylabel("payload bits")
        ax.legend()
        ax.set_title("paired design validation (not confirmation)")
        fig.tight_layout()
        bar = OUTPUT / "phase101_paired_payload.png"
        fig.savefig(bar, dpi=120)
        plt.close(fig)
        paths["paired_payload"] = str(bar)
    return paths


def _summarize(paired: list[dict[str, Any]]) -> dict[str, Any]:
    b0 = np.asarray([float(row["b0_payload_bits"]) for row in paired], dtype=np.float64)
    b1 = np.asarray([float(row["b1_payload_bits"]) for row in paired], dtype=np.float64)
    return {
        "n_scenes": len(paired),
        "b0_mean_payload_bits": float(np.mean(b0)),
        "b1_mean_payload_bits": float(np.mean(b1)),
        "mean_delta_bits": float(np.mean(b1 - b0)),
        "mean_ratio": float(np.mean(b1 / np.maximum(b0, 1.0e-9))),
        "b0_mean_completed_blocks": float(np.mean([row["b0_completed_blocks"] for row in paired])),
        "b1_mean_completed_blocks": float(np.mean([row["b1_completed_blocks"] for row in paired])),
        "b0_mean_in_range_frac": float(np.mean([row["b0_in_range_active_frac"] for row in paired])),
        "b1_mean_in_range_frac": float(np.mean([row["b1_in_range_active_frac"] for row in paired])),
        "b0_mean_progress_resets": float(np.mean([row["b0_progress_reset_count"] for row in paired])),
        "b1_mean_progress_resets": float(np.mean([row["b1_progress_reset_count"] for row in paired])),
        "b0_mean_flight_m": float(np.mean([row["b0_flight_distance_m"] for row in paired])),
        "b1_mean_flight_m": float(np.mean([row["b1_flight_distance_m"] for row in paired])),
        "b0_n_completed_uavs": int(sum(row["b0_n_uav_completed_service"] for row in paired)),
        "b1_n_completed_uavs": int(sum(row["b1_n_uav_completed_service"] for row in paired)),
        "b0_n_never_completed_uavs": int(sum(row["b0_n_uav_never_completed"] for row in paired)),
        "b1_n_never_completed_uavs": int(sum(row["b1_n_uav_never_completed"] for row in paired)),
    }


def write_report(facts: dict[str, Any], smoke: dict[str, Any], pilot: dict[str, Any], summary: dict[str, Any], high_ok: bool, plots: dict[str, str], init_eval: dict[str, Any] | None) -> None:
    trend = "unknown"
    hist = pilot.get("history") or []
    if len(hist) >= 8:
        early = np.nanmean([row["mean_completed_payload_bits"] for row in hist[: max(len(hist) // 4, 1)]])
        late = np.nanmean([row["mean_completed_payload_bits"] for row in hist[-max(len(hist) // 4, 1) :]])
        trend = f"late {late:.0f} vs early {early:.0f} (completed-episode payload during training; sparse)"
    b0p = summary.get("b0_mean_payload_bits")
    b1p = summary.get("b1_mean_payload_bits")
    gap = None if b0p is None else float(b1p) - float(b0p)
    next_step = "定位低层问题或保留规则执行器"
    if b1p is not None and b0p is not None and float(b1p) >= 0.8 * float(b0p):
        next_step = "建议下一轮多 seed 复核，本轮不自动执行"
    lines = [
        "# HTW Phase-10.1 L3-B 六机目标条件低层适配与单种子试验",
        "",
        "只训练低层。未训练高层。未加载旧五机权重。未改 L3-B 物理/SINR/关联/原始 payload。未开 confirmation。未开新 Round。",
        "",
        f"1. 新六机低层：是。禁止路径 `{FORBIDDEN_M5_CKPT}`。fresh init seed={LOW_MODEL_SEED}。",
        "2. 适配：max_uavs=6，K=12 idle token=12，option one-hot/相对位置/距离/进度/valid，hold 与剩余时隙进入 global；维度由 schema 决定。",
        f"3. 高层冻结：high_model_seed={HIGH_MODEL_SEED}，SHA 训练前后一致={high_ok}。",
        f"4. smoke 通过：{smoke.get('passed')}。smoke 成绩不用于判断优劣。",
        f"5. 实际训练：{PILOT_UPDATES} updates × {PPO.rollout_length} × {PPO.num_envs} = {PILOT_UPDATES * PPO.rollout_length * PPO.num_envs} env transitions；总墙钟见 runtime CSV。",
        f"6. design val 配对 payload：B0 mean={b0p}，B1 mean={b1p}。这不是 9.4.5 的 6.95e6 confirmation 数字。",
        f"7. 真实完成块 B0/B1 mean={summary.get('b0_mean_completed_blocks')}/{summary.get('b1_mean_completed_blocks')}；从未完成 UAV 数 B0/B1={summary.get('b0_n_never_completed_uavs')}/{summary.get('b1_n_never_completed_uavs')}。",
        f"8. 进度清零 B0/B1={summary.get('b0_mean_progress_resets')}/{summary.get('b1_mean_progress_resets')}；飞行距离 B0/B1={summary.get('b0_mean_flight_m')}/{summary.get('b1_mean_flight_m')}。",
        f"9. 学习趋势：{trend}。init eval payload={None if init_eval is None else init_eval.get('b1_mean_payload_bits')} vs final {b1p}。",
        f"10. 与 fly-toward 差值 mean Δ={gap}，ratio={summary.get('mean_ratio')}。A 可运行不等于 C 超过规则。",
        f"11. 下一步：{next_step}。",
        "",
        f"图：{plots}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    facts = restore_phase100()
    if not HIGH_CKPT.is_file():
        raise FileNotFoundError(str(HIGH_CKPT))
    schema = build_schema()
    dump_frozen(OUTPUT / "phase101_feature_schema.json", schema)
    dump_frozen(
        OUTPUT / "phase101_low_level_config.json",
        {
            "ppo": PPO.to_dict(),
            "high_model_seed": HIGH_MODEL_SEED,
            "low_model_seed": LOW_MODEL_SEED,
            "hold": HOLD,
            "aux": {"in_range": 0.5, "distance": 0.05, "pilot_design_not_validated_optimal": True},
            "r_env": "payload_delta / block_payload_bits",
            "r_train": "r_env + r_aux",
            "compare_r_train_to_b0": False,
            "forbidden_m5": str(FORBIDDEN_M5_CKPT),
            "deterministic_eval": "tanh(mu), no resampling",
        },
    )
    dump_frozen(
        OUTPUT / "phase101_protocol.json",
        {
            "phase": "10.1",
            "high_model_seed": HIGH_MODEL_SEED,
            "low_model_seed": LOW_MODEL_SEED,
            "design_train_seeds": list(DESIGN_TRAIN_SEEDS),
            "design_val_seeds": list(DESIGN_VAL_SEEDS),
            "design_val_label": "design validation, not fresh/untouched/confirmation",
            "smoke_updates": SMOKE_UPDATES,
            "pilot_updates": PILOT_UPDATES,
            "transitions_pilot": PILOT_UPDATES * PPO.rollout_length * PPO.num_envs,
            "ppo": PPO.to_dict(),
            "checkpoint_rule": "final checkpoint is the official result; intermediates are learning snapshots only",
            "no_reward_rescan": True,
            "no_confirmation": True,
            "facts": facts,
        },
    )
    high, high_file_sha = load_frozen_high(HIGH_CKPT)
    high_param_sha = parameter_sha(high)
    smoke_path = OUTPUT / "phase101_smoke_report.json"
    if smoke_path.is_file() and json.loads(smoke_path.read_text(encoding="utf-8")).get("passed"):
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    else:
        smoke_run = run_updates(
            n_updates=SMOKE_UPDATES,
            vector_seed=VECTOR_SEED_SMOKE,
            label="smoke",
            schema=schema,
            high=high,
            high_file_sha=high_file_sha,
            high_param_sha=high_param_sha,
            save_schedule=False,
        )
        reload_agent = _load_low(Path(smoke_run["init_path"]), schema, PPO)
        det = reload_agent.act(
            L3BLowLevelVector(DESIGN_TRAIN_SEEDS[:1], 1, 0, high_model=high, schema=schema).features(),
            deterministic=True,
        )
        smoke = {
            "passed": True,
            "updates": SMOKE_UPDATES,
            "environment_steps": smoke_run["environment_steps"],
            "seconds": [row["seconds"] for row in smoke_run["runtime"]],
            "mean_seconds_per_update": float(np.mean([row["seconds"] for row in smoke_run["runtime"]])),
            "estimated_pilot_seconds": float(np.mean([row["seconds"] for row in smoke_run["runtime"]]) * PILOT_UPDATES),
            "reload_ok": True,
            "deterministic_action_shape": list(det["action"].shape),
            "high_frozen": True,
            "loaded_m5": False,
            "not_for_ranking": True,
            "history": smoke_run["history"],
        }
        dump_frozen(smoke_path, smoke)
        del smoke_run

    final_path = OUTPUT / "checkpoints" / "pilot_final.pt"
    hist_path = OUTPUT / "phase101_training_history.csv"
    if final_path.is_file() and hist_path.is_file() and sum(1 for _ in hist_path.open(encoding="utf-8")) > PILOT_UPDATES:
        agent = _load_low(final_path, schema, PPO)
        history_rows = list(csv.DictReader(hist_path.open(encoding="utf-8")))
        pilot = {"agent": agent, "history": history_rows, "runtime": [], "checkpoints": [], "environment_steps": PILOT_UPDATES * PPO.rollout_length * PPO.num_envs, "resumed": True}
    else:
        pilot = run_updates(
            n_updates=PILOT_UPDATES,
            vector_seed=VECTOR_SEED_PILOT,
            label="pilot",
            schema=schema,
            high=high,
            high_file_sha=high_file_sha,
            high_param_sha=high_param_sha,
            save_schedule=True,
        )
        _write_csv(hist_path, pilot["history"])
        _write_csv(OUTPUT / "phase101_runtime_profile.csv", pilot["runtime"])
        dump_frozen(
            OUTPUT / "phase101_checkpoint_manifest.json",
            {
                "primary": "final",
                "not_selected_by_eval": True,
                "high_file_sha256": high_file_sha,
                "high_param_sha256": high_param_sha,
                "high_file_sha_after": sha256_bytes(HIGH_CKPT.read_bytes()),
                "checkpoints": pilot["checkpoints"],
                "loaded_m5": False,
            },
        )
        agent = pilot["agent"]

    init_agent = _load_low(OUTPUT / "checkpoints" / "pilot_init.pt", schema, PPO)
    init_rows = evaluate_pair(high, DESIGN_VAL_SEEDS, schema=schema, agent=init_agent)
    init_summary = _summarize(init_rows)
    paired = evaluate_pair(high, DESIGN_VAL_SEEDS, schema=schema, agent=agent)
    summary = _summarize(paired)
    _write_csv(OUTPUT / "phase101_design_paired_metrics.csv", _flatten_eval(paired))
    dump_frozen(OUTPUT / "phase101_eval_summary.json", {"final": summary, "init": init_summary, "not_confirmation": True, "not_695e4_historical": True})
    plots = _plot(_numeric_history(list(csv.DictReader(hist_path.open(encoding="utf-8")))), paired)
    high_ok = sha256_bytes(HIGH_CKPT.read_bytes()) == high_file_sha and parameter_sha(high) == high_param_sha
    write_report(facts, smoke, {"history": _numeric_history(list(csv.DictReader(hist_path.open(encoding="utf-8"))))}, summary, high_ok, plots, init_summary)
    print(json.dumps({"smoke": smoke.get("passed"), "b0": summary["b0_mean_payload_bits"], "b1": summary["b1_mean_payload_bits"], "trained_low": True, "trained_high": False}, indent=2), flush=True)
    return {"smoke": smoke, "summary": summary}


if __name__ == "__main__":
    main()
