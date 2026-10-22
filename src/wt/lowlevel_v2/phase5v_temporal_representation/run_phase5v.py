"""Train TEMP-AUG on 310100-310109 and compare it with Phase-5U once."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5k_joint_decoder.loss import joint_rank_loss
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5n_joint_planner.run_phase5n import _features
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at, quantize
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.run_phase5q import _edge_features
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5r_nonadditive_generalization.world import candidate_pool
from wt.lowlevel_v2.phase5t_scorer_failure_audit.run_phase5t import _representation
from wt.lowlevel_v2.phase5u_representation_repair.features import relational_matrix
from wt.lowlevel_v2.phase5u_representation_repair.run_phase5u import _labels, _replay, _snapshot
from wt.lowlevel_v2.phase5u_representation_repair.scorer import RelationalSetScorer
from wt.round30.ppo import seed_everything

from .metrics import choose_gate
from .protocol import (
    ABLATION_DISAGREEMENT,
    ABLATION_REGRET_BITS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CASE_SCENARIO,
    CASE_TIME,
    CHECKPOINTS,
    CONFIRM_SCENARIOS,
    DEPLOYMENT_VERIFIED,
    EPOCHS,
    FINAL_REL_DIM,
    GRAD_CLIP,
    GROUPS,
    LEARNING_RATE,
    MODEL_POOL_WIDTH,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PAYLOAD_REGRESSION,
    PHASE5U_CASE_RANK,
    PHASE5U_CHECKPOINTS,
    PLANNER_READY,
    RANK_CLEAR,
    RANK_PROMISING,
    REL_DIM,
    REPORT,
    SEEDS,
    T15_CLEAR_DROP,
    T15_REGRESSION,
    TEMP_INPUT_DIM,
    TIMES,
    TRAIN_SCENARIOS,
    WEIGHT_DECAY,
    assert_consumed,
)
from .temporal import edge_means, observability_rows, temporal_matrix


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _median(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _fmt(value, digits: str = ".4f") -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), digits)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _vector(assignment) -> str:
    return ";".join(str(int(item)) for item in np.asarray(assignment, dtype=np.int64))


def _checkpoint(seed: int):
    return CHECKPOINTS / f"phase5v_temp_aug_seed{int(seed)}.pt"


def _freeze_protocol() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "phase5v_protocol.json"
    if not path.is_file():
        path.write_text(
            json.dumps(
                {
                    "baseline": "phase5u REP-AUG",
                    "proposed": "REP-AUG static features plus candidate-conditioned temporal features",
                    "encoder": "frozen Phase-5L HGNN",
                    "optimizer": "Adam",
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "grad_clip": GRAD_CLIP,
                    "epochs": EPOCHS,
                    "checkpoint": "epoch_10",
                    "seeds": list(SEEDS),
                    "train_scenarios": list(TRAIN_SCENARIOS),
                    "evaluation_scenarios": list(CONFIRM_SCENARIOS),
                    "loss": "pairwise ranking when the block gap is at least 1e6 bit",
                    "t15_clear_drop": T15_CLEAR_DROP,
                    "rank_clear": RANK_CLEAR,
                    "rank_promising": RANK_PROMISING,
                    "hold_remaining_excluded": True,
                    "future_quantities_excluded": ["future SINR", "future trajectory", "future rate", "future payload", "J_MAPPO"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    (OUTPUT / "phase5v_temporal_feature_definition.json").write_text(
        json.dumps(
            {
                "candidate_features": [
                    "same_target",
                    "switch",
                    "prev_sinr_if_same",
                    "continue_progress",
                    "buoy_progress",
                    "closing_speed",
                    "heading_cos",
                    "velocity_valid",
                    "distance_delta",
                ],
                "global_context": ["episode_fraction", "remaining_fraction", "decision_index"],
                "aggregation": "mean, max, min, std over selected edges",
                "standardization": "training-state mean and std only",
                "excluded": ["hold_remaining", "option_age", "consecutive_service_steps", "future SINR", "J_MAPPO"],
                "groups": {name: {"start": int(start), "end": int(end)} for name, (start, end) in GROUPS.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(OUTPUT / "phase5v_temporal_observability_audit.csv", observability_rows())


def _history_snapshot(env, batch) -> dict:
    count = int(env.config.num_uavs)
    info = env.info()
    return {
        "uav_pos": np.asarray(info["uav_positions_m"], dtype=np.float64),
        "buoy_pos": np.asarray(info["buoy_positions_m"], dtype=np.float64),
        "progress": np.asarray(info["service_progress"], dtype=np.float64),
        "current_assignment": np.asarray(info["assignment"], dtype=np.int64),
        "velocity": np.asarray(batch["uav"][0, :count, 2:4], dtype=np.float64),
        "last_sinr": np.asarray(env.last_sinr, dtype=np.float64),
        "time_step": int(env.t),
    }


def _build_states(frozen, matching, config: L3BConfig) -> list[dict]:
    recorded = _labels()
    states = []
    for scenario in list(TRAIN_SCENARIOS) + list(CONFIRM_SCENARIOS):
        for moment in TIMES:
            assert_consumed(int(scenario))
            env, _movement, batch, diagnosis = _replay(int(scenario), int(moment))
            embedded = _edge_features(frozen, batch)
            additive = _features(frozen, matching, batch)
            logits = additive["logits"].detach().cpu().numpy()
            pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
            vectors = [_vector(row) for row in pool["assignments"]]
            recorded_vectors = {key[2] for key in recorded if key[0] == int(scenario) and key[1] == int(moment)}
            if set(vectors) != recorded_vectors:
                raise RuntimeError(f"pool does not match stored labels at {scenario} t={moment}")
            history = _history_snapshot(env, batch)
            relational = relational_matrix(_snapshot(env, diagnosis), pool["assignments"], config)
            temporal = temporal_matrix(history, pool["assignments"], config)
            if not np.all(np.isfinite(relational)) or not np.all(np.isfinite(temporal)):
                raise RuntimeError(f"non-finite representation at {scenario} t={moment}")
            states.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(moment),
                    "split": "train" if int(scenario) in TRAIN_SCENARIOS else "confirmation",
                    "assignments": pool["assignments"],
                    "vectors": vectors,
                    "j_mappo": np.asarray([recorded[(int(scenario), int(moment), vector)] for vector in vectors], dtype=np.float64),
                    "embedded": embedded,
                    "z_rel": relational,
                    "z_temp_raw": temporal,
                    "history": history,
                }
            )
            print(f"state {scenario} t={moment}", flush=True)
    return states


def _fit_scaler(states: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    rows = np.concatenate([state["z_temp_raw"] for state in states if state["split"] == "train"], axis=0)
    return rows.mean(axis=0).astype(np.float32), np.maximum(rows.std(axis=0), 1.0e-6).astype(np.float32)


def _scale(raw: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((raw - center) / scale).astype(np.float32)


def _combine(relational: np.ndarray, temporal: np.ndarray) -> np.ndarray:
    combined = np.concatenate((relational, temporal), axis=1).astype(np.float32)
    if combined.shape[1] != FINAL_REL_DIM:
        raise RuntimeError("temporal block changed the frozen static width")
    return combined


def _pack(embedded: dict, relational: np.ndarray) -> dict:
    packed = dict(embedded)
    packed["z_rel"] = torch.as_tensor(relational, dtype=torch.float32)
    return packed


def _levels(labels: np.ndarray) -> torch.Tensor:
    values = torch.as_tensor(labels, dtype=torch.float32)
    return torch.round(values / PAYLOAD_BLOCK_BITS) * PAYLOAD_BLOCK_BITS


def _rank(scores: np.ndarray, labels: np.ndarray) -> float:
    levels = quantize(labels)
    optimal = levels == float(np.max(levels))
    return float(1 + np.sum(scores > float(np.max(scores[optimal]))))


def _train_one(scorer, states: list[dict], seed: int, center: np.ndarray, scale: np.ndarray) -> list[dict]:
    train_states = [state for state in states if state["split"] == "train"]
    optimizer = torch.optim.Adam(scorer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, int(EPOCHS) + 1):
        scorer.train()
        losses = []
        for state in train_states:
            if int(state["scenario_id"]) not in TRAIN_SCENARIOS:
                raise RuntimeError("confirmation state entered the TEMP gradient")
            optimizer.zero_grad(set_to_none=True)
            features = _combine(state["z_rel"], _scale(state["z_temp_raw"], center, scale))
            scores = scorer(_pack(state["embedded"], features), torch.as_tensor(state["assignments"], dtype=torch.long))
            loss = joint_rank_loss(scores, _levels(state["j_mappo"]))
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError(f"TEMP-AUG loss became non-finite at seed {seed} epoch {epoch}")
            loss.backward()
            clip_grad_norm_(scorer.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scorer.eval()
        measured_rows = []
        with torch.no_grad():
            for index, state in enumerate(train_states):
                features = _combine(state["z_rel"], _scale(state["z_temp_raw"], center, scale))
                values = scorer(_pack(state["embedded"], features), torch.as_tensor(state["assignments"], dtype=torch.long)).detach().cpu().numpy()
                chosen = int(np.argsort(-values, kind="stable")[0])
                measured = metrics_at(values, state["j_mappo"], chosen)
                measured_rows.append(measured)
                curves.append(
                    {
                        "row_type": "state",
                        "seed": int(seed),
                        "epoch": int(epoch),
                        "scenario_id": int(state["scenario_id"]),
                        "time_step": int(state["time_step"]),
                        "loss": losses[index],
                        "top1": measured["top1"],
                        "normalized_regret": measured["normalized_regret"],
                        "score_std": measured["score_std"],
                        "collapsed": measured["collapsed"],
                    }
                )
        curves.append(
            {
                "row_type": "seed_epoch",
                "seed": int(seed),
                "epoch": int(epoch),
                "scenario_id": -1,
                "time_step": -1,
                "loss": _mean(losses),
                "top1": _mean(row["top1"] for row in measured_rows),
                "normalized_regret": _mean(row["normalized_regret"] for row in measured_rows),
                "score_std": _mean(row["score_std"] for row in measured_rows),
                "collapsed": _mean(row["collapsed"] for row in measured_rows),
            }
        )
        print(f"temp-aug seed {seed} epoch {epoch} loss={_mean(losses):.4f}", flush=True)
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return curves


def _train_all(states: list[dict], center: np.ndarray, scale: np.ndarray) -> None:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        seed_everything(int(seed))
        scorer = RelationalSetScorer(rel_dim=FINAL_REL_DIM)
        curves = _train_one(scorer, states, int(seed), center, scale)
        torch.save(
            {
                "seed": int(seed),
                "epoch": int(EPOCHS),
                "state_dict": scorer.state_dict(),
                "curves": curves,
                "center": center,
                "scale": scale,
                "rel_dim": FINAL_REL_DIM,
            },
            _checkpoint(int(seed)),
        )
        print(f"saved {_checkpoint(int(seed)).name}", flush=True)


def _load_v1() -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    scorers = {}
    curves = []
    center = scale = None
    for seed in SEEDS:
        payload = torch.load(_checkpoint(int(seed)), map_location="cpu", weights_only=False)
        scorer = RelationalSetScorer(rel_dim=FINAL_REL_DIM)
        scorer.load_state_dict(payload["state_dict"])
        scorer.eval()
        scorers[int(seed)] = scorer
        curves.extend(payload["curves"])
        center = np.asarray(payload["center"], dtype=np.float32)
        scale = np.asarray(payload["scale"], dtype=np.float32)
    return scorers, curves, center, scale


def _load_v0() -> dict:
    scorers = {}
    for seed in SEEDS:
        scorer = RelationalSetScorer(rel_dim=REL_DIM)
        scorer.load_state_dict(torch.load(PHASE5U_CHECKPOINTS / f"phase5u_rep_aug_seed{int(seed)}.pt", map_location="cpu", weights_only=False)["state_dict"])
        scorer.eval()
        scorers[int(seed)] = scorer
    return scorers


def _scores(scorer, state: dict, relational: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return scorer(_pack(state["embedded"], relational), torch.as_tensor(state["assignments"], dtype=torch.long)).detach().cpu().numpy()


def _latent(scorer, state: dict, relational: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return scorer.latent(_pack(state["embedded"], relational), torch.as_tensor(state["assignments"], dtype=torch.long)).detach().cpu().numpy()


def _metric_row(model: str, seed: int, state: dict, scores: np.ndarray) -> dict:
    chosen = int(np.argsort(-scores, kind="stable")[0])
    measured = metrics_at(scores, state["j_mappo"], chosen)
    return {
        "model": model,
        "seed": int(seed),
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "j_mappo": measured["j_mappo"],
        "regret": measured["regret"],
        "normalized_regret": measured["normalized_regret"],
        "top1": measured["top1"],
        "top3": measured["top3"],
        "spearman": measured["spearman"],
        "kendall": measured["kendall"],
        "optimum_rank": _rank(scores, state["j_mappo"]),
        "score_std": measured["score_std"],
        "collapsed": measured["collapsed"],
        "assignment_vector": state["vectors"][chosen],
        "pool_oracle": float(np.max(state["j_mappo"])),
    }


def _write_report(summary: dict) -> None:
    lines = [
        "# HTW Phase-5V Candidate-Conditioned Temporal State",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段只加入决策时刻已经存在、并且会随候选链路变化的执行历史。Phase-5U 的静态关系特征、冻结 encoder、ranking loss 和候选池都没有改。hold remaining 没有输入。未来 SINR 和 J_MAPPO 只作为标签。",
        "",
        "## 1. 哪些时间量真实存在",
        "",
        "当前任务、服务进度、上一拍位置、当前速度和上一拍 SINR 可以直接恢复。连续服务步数、option age、AoI 和 hold remaining 都没有可用记录。episode time 对同一个 state 里的全部 assignment 相同，只作为 context。",
        "",
        "## 2. Collision",
        "",
        f"V0 z_static collision {_fmt(summary['collision_v0'])}，其中 t=15 是 {_fmt(summary['collision_v0_t15'])}。",
        f"V1 z_final collision {_fmt(summary['collision_v1'])}，其中 t=15 是 {_fmt(summary['collision_v1_t15'])}。",
        "",
        "## 3. 310114 t=15",
        "",
        "| seed | predicted | rank | J_pred | J_opt | regret | same | progress | closing | heading | distance delta |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *summary["case_lines"],
        "",
        f"optimum rank 中位数 {_fmt(summary['case_rank'], '.1f')}。Phase-5U 是 47。",
        "",
        "## 4. 按决策时刻",
        "",
        "| t | V0 regret | V1 regret | V0 J | V1 J | ΔJ |",
        "| --- | --- | --- | --- | --- | --- |",
        *summary["time_lines"],
        "",
        f"整体 ΔJ mean {_fmt(summary['delta_j']['mean'], '.4e')}，95% CI [{_fmt(summary['delta_j']['low'], '.4e')}, {_fmt(summary['delta_j']['high'], '.4e')}]。",
        f"scenario 更高 {summary['n_up']}，更低 {summary['n_down']}。seed 正向 {summary['n_seeds_up']}/5。",
        "",
        "## 5. TEMP ablation",
        "",
        "| group | disagreement | Δregret | ΔJ | used |",
        "| --- | --- | --- | --- | --- |",
        *summary["ablation_lines"],
        "",
        "## 6. 结论",
        "",
        f"Gate 是 **{summary['gate']}**。",
        "",
        summary["advice"],
        "",
        "Q1. 哪些时间量真实存在？ " + summary["answers"]["q1"],
        "",
        "Q2. 哪些会随 candidate 改变？ " + summary["answers"]["q2"],
        "",
        "Q3. t=15 regret 降到多少？ " + summary["answers"]["q3"],
        "",
        "Q4. 310114 的最优排名是否继续前移？ " + summary["answers"]["q4"],
        "",
        "Q5. J_MAPPO 是否稳定提高？ " + summary["answers"]["q5"],
        "",
        "Q6. 哪一组有效？ " + summary["answers"]["q6"],
        "",
        "Q7. 能否解释近而失败的 assignment？ " + summary["answers"]["q7"],
        "",
        "Q8. 下一步做什么？ " + summary["answers"]["q8"],
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    _freeze_protocol()
    config = L3BConfig()
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    states = _build_states(frozen, matching, config)
    center, scale = _fit_scaler(states)
    if any(not _checkpoint(int(seed)).is_file() for seed in SEEDS):
        _train_all(states, center, scale)
    v1, curves, center, scale = _load_v1()
    v0 = _load_v0()
    _write_csv(OUTPUT / "phase5v_training_curves.csv", curves)
    confirmation = [state for state in states if state["split"] == "confirmation"]
    rep_rows = []
    state_rows = []
    paired = []
    case_rows = []
    for state in confirmation:
        for seed in SEEDS:
            static = state["z_rel"]
            final = _combine(static, _scale(state["z_temp_raw"], center, scale))
            base_scores = _scores(v0[int(seed)], state, static)
            temp_scores = _scores(v1[int(seed)], state, final)
            base_row = _metric_row("v0", int(seed), state, base_scores)
            temp_row = _metric_row("v1", int(seed), state, temp_scores)
            state_rows.extend((base_row, temp_row))
            paired.append(
                {
                    "scenario_id": int(state["scenario_id"]),
                    "time_step": int(state["time_step"]),
                    "seed": int(seed),
                    "delta_j_mappo": float(temp_row["j_mappo"] - base_row["j_mappo"]),
                    "delta_regret": float(temp_row["regret"] - base_row["regret"]),
                    "delta_normalized_regret": float(temp_row["normalized_regret"] - base_row["normalized_regret"]),
                    "delta_top1": float(temp_row["top1"] - base_row["top1"]),
                    "delta_optimum_rank": float(temp_row["optimum_rank"] - base_row["optimum_rank"]),
                }
            )
            for model, latent in (
                ("v0_static", _latent(v0[int(seed)], state, static)),
                ("v1_final", _latent(v1[int(seed)], state, final)),
            ):
                measured = _representation(latent, state["j_mappo"])
                rep_rows.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "scenario_id": int(state["scenario_id"]),
                        "time_step": int(state["time_step"]),
                        "collision_rate": measured["collision_rate"],
                        "cosine_separation": measured["cosine_separation"],
                        "fisher_ratio": measured["fisher_ratio"],
                        "effective_rank": measured["effective_rank"],
                        "nn_payload_contradiction": measured["collision_rate"],
                    }
                )
            if int(state["scenario_id"]) == CASE_SCENARIO and int(state["time_step"]) == CASE_TIME:
                levels = quantize(state["j_mappo"])
                optimum = state["vectors"][int(np.flatnonzero(levels == np.max(levels))[0])]
                chosen = np.asarray([int(item) for item in temp_row["assignment_vector"].split(";")], dtype=np.int64)
                optimal = np.asarray([int(item) for item in optimum.split(";")], dtype=np.int64)
                chosen_features = edge_means(state["history"], chosen, config)
                optimal_features = edge_means(state["history"], optimal, config)
                case_rows.append(
                    {
                        "seed": int(seed),
                        "predicted_assignment": temp_row["assignment_vector"],
                        "pool_optimal_assignment": optimum,
                        "optimum_rank": temp_row["optimum_rank"],
                        "j_mappo_pred": temp_row["j_mappo"],
                        "j_mappo_opt": temp_row["pool_oracle"],
                        "regret": temp_row["regret"],
                        "pred_same_target": chosen_features["same_target"],
                        "opt_same_target": optimal_features["same_target"],
                        "pred_continue_progress": chosen_features["continue_progress"],
                        "opt_continue_progress": optimal_features["continue_progress"],
                        "pred_closing_speed": chosen_features["closing_speed"],
                        "opt_closing_speed": optimal_features["closing_speed"],
                        "pred_heading_cos": chosen_features["heading_cos"],
                        "opt_heading_cos": optimal_features["heading_cos"],
                        "pred_distance_delta": chosen_features["distance_delta"],
                        "opt_distance_delta": optimal_features["distance_delta"],
                    }
                )
        print(f"scored {state['scenario_id']} t={state['time_step']}", flush=True)
    ablation_rows = []
    for name, (start, end) in GROUPS.items():
        disagreements = []
        regret_deltas = []
        payload_deltas = []
        for state in confirmation:
            for seed in SEEDS:
                full_raw = _scale(state["z_temp_raw"], center, scale)
                full = _scores(v1[int(seed)], state, _combine(state["z_rel"], full_raw))
                masked = np.array(state["z_temp_raw"], copy=True)
                masked[:, start:end] = 0.0
                ablated = _scores(v1[int(seed)], state, _combine(state["z_rel"], _scale(masked, center, scale)))
                full_choice = int(np.argsort(-full, kind="stable")[0])
                ablated_choice = int(np.argsort(-ablated, kind="stable")[0])
                disagreements.append(float(full_choice != ablated_choice))
                full_metrics = metrics_at(full, state["j_mappo"], full_choice)
                ablated_metrics = metrics_at(ablated, state["j_mappo"], ablated_choice)
                regret_deltas.append(float(ablated_metrics["regret"] - full_metrics["regret"]))
                payload_deltas.append(float(ablated_metrics["j_mappo"] - full_metrics["j_mappo"]))
        disagreement = _mean(disagreements)
        regret_delta = _mean(regret_deltas)
        ablation_rows.append(
            {
                "group": name,
                "disagreement": disagreement,
                "delta_regret": regret_delta,
                "delta_j_mappo": _mean(payload_deltas),
                "used": float(disagreement >= ABLATION_DISAGREEMENT or regret_delta >= ABLATION_REGRET_BITS),
            }
        )
    delta_j = cluster_bootstrap(paired, "delta_j_mappo", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    by_scenario = defaultdict(list)
    by_seed = defaultdict(list)
    for row in paired:
        by_scenario[int(row["scenario_id"])].append(float(row["delta_j_mappo"]))
        by_seed[int(row["seed"])].append(float(row["delta_j_mappo"]))
    scenario_means = {key: float(np.mean(value)) for key, value in by_scenario.items()}
    n_up = sum(value > 1.0 for value in scenario_means.values())
    n_down = sum(value < -1.0 for value in scenario_means.values())
    n_seeds_up = sum(float(np.mean(value)) > 0.0 for value in by_seed.values())
    n_seeds_down = sum(float(np.mean(value)) < 0.0 for value in by_seed.values())
    t15_v0 = _mean(row["regret"] for row in state_rows if row["model"] == "v0" and int(row["time_step"]) == 15)
    t15_v1 = _mean(row["regret"] for row in state_rows if row["model"] == "v1" and int(row["time_step"]) == 15)
    case_rank = _median(row["optimum_rank"] for row in case_rows)
    candidate_used = any(row["group"] != "global_context" and row["used"] >= 0.5 for row in ablation_rows)
    t15_down = bool(t15_v0 - t15_v1 >= T15_CLEAR_DROP)
    promising = bool(t15_down or case_rank <= RANK_PROMISING)
    gate = choose_gate(
        regression=bool(delta_j["mean"] <= -PAYLOAD_REGRESSION or t15_v1 - t15_v0 >= T15_REGRESSION),
        j_up=bool(delta_j["mean"] > 0.0),
        j_ci=bool(delta_j["low"] > 0.0),
        t15_down=t15_down,
        majority_scenario=bool(n_up * 2 > len(CONFIRM_SCENARIOS)),
        majority_seed=bool(n_seeds_up * 2 > len(SEEDS)),
        ablation_used=bool(candidate_used),
        case_clear=bool(case_rank <= RANK_CLEAR),
        promising=promising,
    )
    advice = {
        "V-A": "冻结 REP-AUG + TEMP-AUG。下一阶段用 310118–310125 做新的 development confirmation。",
        "V-B": "TEMP 对 t=15 或 310114 有作用，但总体还不稳。下一阶段只做 TEMP mechanism refinement，不增加网络复杂度。",
        "V-C": "现有历史观测还不能预测 15 步执行收益。下一阶段改为 short-horizon executor surrogate 或 reachability 设计审计，不再堆 temporal feature。",
        "V-D": "TEMP-AUG 比 Phase-5U 差。退回 Phase-5U REP-AUG，不保留这组权重。",
    }[gate]
    collision_v0 = _mean(row["collision_rate"] for row in rep_rows if row["model"] == "v0_static")
    collision_v1 = _mean(row["collision_rate"] for row in rep_rows if row["model"] == "v1_final")
    collision_v0_t15 = _mean(row["collision_rate"] for row in rep_rows if row["model"] == "v0_static" and int(row["time_step"]) == 15)
    collision_v1_t15 = _mean(row["collision_rate"] for row in rep_rows if row["model"] == "v1_final" and int(row["time_step"]) == 15)
    time_lines = []
    time_rows = []
    for moment in TIMES:
        for model in ("v0", "v1"):
            rows = [row for row in state_rows if row["model"] == model and int(row["time_step"]) == int(moment)]
            time_rows.append(
                {
                    "model": model,
                    "time_step": int(moment),
                    "j_mappo": _mean(row["j_mappo"] for row in rows),
                    "regret": _mean(row["regret"] for row in rows),
                    "normalized_regret": _mean(row["normalized_regret"] for row in rows),
                    "top1": _mean(row["top1"] for row in rows),
                    "top3": _mean(row["top3"] for row in rows),
                    "spearman": _mean(row["spearman"] for row in rows),
                    "optimum_rank": _median(row["optimum_rank"] for row in rows),
                }
            )
        base_rows = [row for row in state_rows if row["model"] == "v0" and int(row["time_step"]) == int(moment)]
        new_rows = [row for row in state_rows if row["model"] == "v1" and int(row["time_step"]) == int(moment)]
        time_lines.append(
            "| " + " | ".join((
                str(moment),
                _fmt(_mean(row["regret"] for row in base_rows), ".4e"),
                _fmt(_mean(row["regret"] for row in new_rows), ".4e"),
                _fmt(_mean(row["j_mappo"] for row in base_rows), ".4e"),
                _fmt(_mean(row["j_mappo"] for row in new_rows), ".4e"),
                _fmt(_mean(row["delta_j_mappo"] for row in paired if int(row["time_step"]) == int(moment)), ".4e"),
            )) + " |"
        )
    case_lines = [
        "| " + " | ".join((
            str(row["seed"]),
            row["predicted_assignment"],
            _fmt(row["optimum_rank"], ".0f"),
            _fmt(row["j_mappo_pred"], ".4e"),
            _fmt(row["j_mappo_opt"], ".4e"),
            _fmt(row["regret"], ".4e"),
            _fmt(row["pred_same_target"]),
            _fmt(row["pred_continue_progress"]),
            _fmt(row["pred_closing_speed"]),
            _fmt(row["pred_heading_cos"]),
            _fmt(row["pred_distance_delta"]),
        )) + " |"
        for row in case_rows
    ]
    ablation_lines = [
        f"| {row['group']} | {_fmt(row['disagreement'])} | {_fmt(row['delta_regret'], '.4e')} | {_fmt(row['delta_j_mappo'], '.4e')} | {bool(row['used'])} |"
        for row in ablation_rows
    ]
    used = [row["group"] for row in ablation_rows if row["used"] >= 0.5]
    same_failure = all(float(row["j_mappo_pred"]) <= 0.0 and float(row["regret"]) >= PAYLOAD_BLOCK_BITS for row in case_rows)
    answers = {
        "q1": "当前任务、UAV 服务进度、上一拍位置、当前速度和上一拍 SINR 存在。连续服务步数、option age、hold remaining、AoI 不存在。",
        "q2": "same target、switch、延续进度、目标浮标进度、接近速度、航向、距离变化，以及仅在延续当前目标时保留的上一拍 SINR。episode fraction 对所有候选相同。",
        "q3": f"t=15 regret 从 {_fmt(t15_v0, '.4e')} 变为 {_fmt(t15_v1, '.4e')}。",
        "q4": f"中位数 {_fmt(case_rank, '.1f')}，Phase-5U 是 {PHASE5U_CASE_RANK:.0f}。",
        "q5": f"mean ΔJ {_fmt(delta_j['mean'], '.4e')}，95% CI [{_fmt(delta_j['low'], '.4e')}, {_fmt(delta_j['high'], '.4e')}]。scenario 正向 {n_up}/8，seed 正向 {n_seeds_up}/5。",
        "q6": "、".join(used) if used else "四组都没有越过预先写好的作用阈值。",
        "q7": "还没有。310114 t=15 五个 seed 的预测 payload 仍是 0。" if same_failure else "310114 t=15 的预测 payload 已经离开 0。",
        "q8": advice,
    }
    seed_rows = []
    for seed in SEEDS:
        for model in ("v0", "v1"):
            rows = [row for row in state_rows if row["model"] == model and int(row["seed"]) == int(seed)]
            seed_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "j_mappo": _mean(row["j_mappo"] for row in rows),
                    "regret": _mean(row["regret"] for row in rows),
                    "top1": _mean(row["top1"] for row in rows),
                    "optimum_rank": _median(row["optimum_rank"] for row in rows),
                    "delta_j_mappo": _mean(by_seed[int(seed)]) if model == "v1" else "",
                }
            )
    summary = {
        "gate": gate,
        "advice": advice,
        "answers": answers,
        "collision_v0": collision_v0,
        "collision_v1": collision_v1,
        "collision_v0_t15": collision_v0_t15,
        "collision_v1_t15": collision_v1_t15,
        "case_rank": case_rank,
        "case_lines": case_lines,
        "time_lines": time_lines,
        "ablation_lines": ablation_lines,
        "delta_j": delta_j,
        "n_up": n_up,
        "n_down": n_down,
        "n_seeds_up": n_seeds_up,
    }
    _write_csv(OUTPUT / "phase5v_representation_audit.csv", rep_rows)
    _write_csv(OUTPUT / "phase5v_state_metrics.csv", state_rows)
    _write_csv(OUTPUT / "phase5v_timestep_metrics.csv", time_rows)
    _write_csv(OUTPUT / "phase5v_seed_metrics.csv", seed_rows)
    _write_csv(OUTPUT / "phase5v_temporal_ablation.csv", ablation_rows)
    _write_csv(OUTPUT / "phase5v_310114_t15_case.csv", case_rows)
    _write_csv(OUTPUT / "phase5v_paired_comparison.csv", paired)
    payload = {
        "gate": gate,
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "t15_regret_v0": t15_v0,
        "t15_regret_v1": t15_v1,
        "case_rank_median": case_rank,
        "delta_j": delta_j,
        "n_scenarios_up": n_up,
        "n_scenarios_down": n_down,
        "n_seeds_up": n_seeds_up,
        "n_seeds_down": n_seeds_down,
        "candidate_ablation_used": candidate_used,
        "keep_temp_aug": gate != "V-D",
    }
    (OUTPUT / "phase5v_temporal_gate.json").write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")
    _write_report(summary)
    print(json.dumps(_plain({"gate": gate, "t15_v0": t15_v0, "t15_v1": t15_v1, "case_rank": case_rank, "delta_j": delta_j["mean"]})), flush=True)
    return summary


if __name__ == "__main__":
    run()
