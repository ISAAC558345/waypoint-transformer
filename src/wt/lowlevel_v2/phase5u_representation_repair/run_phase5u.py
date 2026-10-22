"""Train REP-AUG on 310100-310109 and score 310110-310117 once."""
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
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.scorer import AssignmentConditionedSetScorer
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5r_nonadditive_generalization.world import candidate_pool
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import PHASE5O_LABELS, PHASE5R_LABELS
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.world import replay_confirmation, replay_train
from wt.lowlevel_v2.phase5t_scorer_failure_audit.protocol import PHASE5S_LABELS
from wt.lowlevel_v2.phase5t_scorer_failure_audit.represent import assignment_latents
from wt.lowlevel_v2.phase5t_scorer_failure_audit.run_phase5t import _representation
from wt.round30.ppo import seed_everything

from .features import relational_matrix
from .metrics import choose_gate
from .protocol import (
    ABLATION_DISAGREEMENT,
    ABLATION_REGRET_BITS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CASE_SCENARIO,
    CASE_TIME,
    CHECKPOINTS,
    COLLISION_ABS_DROP,
    COLLISION_REL_DROP,
    COLLISION_WORSE,
    CONFIRM_SCENARIOS,
    DEPLOYMENT_VERIFIED,
    EPOCHS,
    GRAD_CLIP,
    GROUPS,
    LEARNING_RATE,
    MODEL_POOL_WIDTH,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PAYLOAD_REGRESSION,
    PHASE5S_CHECKPOINTS,
    PLANNER_READY,
    RANK_CLEAR,
    RANK_IMPROVED,
    REL_DIM,
    REPORT,
    SEEDS,
    T15_STILL_HIGH,
    TIMES,
    TRAIN_SCENARIOS,
    WEIGHT_DECAY,
    assert_consumed,
)
from .scorer import RelationalSetScorer


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _median(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _std(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.std(array, ddof=1)) if array.size > 1 else float("nan")


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
    return CHECKPOINTS / f"phase5u_rep_aug_seed{int(seed)}.pt"


def _labels() -> dict[tuple[int, int, str], float]:
    table = {}
    with PHASE5O_LABELS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            if scenario not in (310100, 310101):
                continue
            table[(scenario, int(row["time_step"]), row["assignment_vector"])] = float(row["j_mappo"])
    for path, allowed in ((PHASE5R_LABELS, TRAIN_SCENARIOS), (PHASE5S_LABELS, CONFIRM_SCENARIOS)):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if float(row.get("in_model_pool", 1.0)) != 1.0:
                    continue
                scenario = int(row["scenario_id"])
                if scenario not in allowed:
                    continue
                table[(scenario, int(row["time_step"]), row["assignment_vector"])] = float(row["j_mappo"])
    return table


def _replay(scenario: int, moment: int):
    assert_consumed(int(scenario))
    if int(scenario) in TRAIN_SCENARIOS:
        return replay_train(int(scenario), int(moment))
    return replay_confirmation(int(scenario), int(moment))


def _snapshot(env, diagnosis) -> dict:
    info = env.info()
    return {
        "uav_pos": np.asarray(info["uav_positions_m"], dtype=np.float64).copy(),
        "buoy_pos": np.asarray(info["buoy_positions_m"], dtype=np.float64).copy(),
        "queue": np.asarray(diagnosis["queue"], dtype=np.float64).copy(),
        "urgency": np.asarray(diagnosis["urgency"], dtype=np.float64).copy(),
        "progress": np.asarray(info["service_progress"], dtype=np.float64).copy(),
        "current_assignment": np.asarray(info["assignment"], dtype=np.int64).copy(),
        "distance": np.asarray(diagnosis["distance"], dtype=np.float64).copy(),
        "eta": np.asarray(diagnosis["eta"], dtype=np.float64).copy(),
        "rate": np.asarray(diagnosis["rate"], dtype=np.float64).copy(),
        "reference_rate": float(env.reference_rate_bps),
    }


def _freeze_protocol() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = OUTPUT / "phase5u_protocol.json"
    if protocol.is_file():
        return
    protocol.write_text(
        json.dumps(
            {
                "encoder": "frozen Phase-5L HGNN",
                "baseline": "phase5s AssignmentConditionedSetScorer",
                "augmentation": "z_joint = concat(z_A, z_rel)",
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "epochs": EPOCHS,
                "checkpoint": "epoch_10",
                "seeds": list(SEEDS),
                "train_scenarios": list(TRAIN_SCENARIOS),
                "evaluation_scenarios": list(CONFIRM_SCENARIOS),
                "evaluation_not_used_for": ["gradient", "checkpoint_selection", "early_stopping", "hyperparameter_search"],
                "loss": "pairwise ranking when the block gap is at least 1e6 bit",
                "collision_abs_drop": COLLISION_ABS_DROP,
                "collision_rel_drop": COLLISION_REL_DROP,
                "rank_clear": RANK_CLEAR,
                "rank_improved": RANK_IMPROVED,
                "t15_still_high": T15_STILL_HIGH,
                "payload_regression": PAYLOAD_REGRESSION,
                "rel_dim": REL_DIM,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT / "phase5u_feature_definition.json").write_text(
        json.dumps(
            {
                "z_rel_dim": REL_DIM,
                "aggregation": "mean, max, min, std over selected edges and selected pairs",
                "permutation": "UAV index is not flattened",
                "future_quantities_excluded": ["future SINR", "future payload", "J_MAPPO", "future fading", "future arrivals"],
                "temporal_excluded": ["episode time", "option age", "service duration", "target lock duration", "hold remaining", "previous option id"],
                "groups": {name: {"start": int(start), "end": int(end)} for name, (start, end) in GROUPS.items()},
                "hypothetical_sinr": "sinr_rates_for_links on the candidate links at the current positions; time is not advanced",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_states(frozen, matching, config: L3BConfig) -> list[dict]:
    recorded = _labels()
    states = []
    for scenario in list(TRAIN_SCENARIOS) + list(CONFIRM_SCENARIOS):
        for moment in TIMES:
            env, _movement, batch, diagnosis = _replay(int(scenario), int(moment))
            embedded = _edge_features(frozen, batch)
            additive = _features(frozen, matching, batch)
            logits = additive["logits"].detach().cpu().numpy()
            pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
            vectors = [_vector(row) for row in pool["assignments"]]
            recorded_vectors = {key[2] for key in recorded if key[0] == int(scenario) and key[1] == int(moment)}
            if set(vectors) != recorded_vectors:
                raise RuntimeError(f"pool does not match stored labels at {scenario} t={moment}")
            relational = relational_matrix(_snapshot(env, diagnosis), pool["assignments"], config)
            if not np.all(np.isfinite(relational)):
                raise RuntimeError(f"non-finite z_rel at {scenario} t={moment}")
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
                }
            )
            print(f"state {scenario} t={moment}", flush=True)
    return states


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
    optimum_score = float(np.max(scores[optimal]))
    return float(1 + np.sum(scores > optimum_score))


def _score_base(scorer, state: dict) -> np.ndarray:
    with torch.no_grad():
        return scorer(state["embedded"], torch.as_tensor(state["assignments"], dtype=torch.long)).detach().cpu().numpy()


def _score_aug(scorer, state: dict, relational: np.ndarray | None = None) -> np.ndarray:
    values = relational if relational is not None else state["z_rel"]
    with torch.no_grad():
        return scorer(_pack(state["embedded"], values), torch.as_tensor(state["assignments"], dtype=torch.long)).detach().cpu().numpy()


def _latent_base(scorer, state: dict) -> np.ndarray:
    with torch.no_grad():
        return assignment_latents(scorer, state["embedded"], torch.as_tensor(state["assignments"], dtype=torch.long)).detach().cpu().numpy()


def _latent_aug(scorer, state: dict) -> np.ndarray:
    with torch.no_grad():
        return scorer.latent(_pack(state["embedded"], state["z_rel"]), torch.as_tensor(state["assignments"], dtype=torch.long)).detach().cpu().numpy()


def _rep_row(model: str, stage: str, seed: int, state: dict, latent: np.ndarray) -> dict:
    measured = _representation(latent, state["j_mappo"])
    return {
        "model": model,
        "stage": stage,
        "seed": int(seed),
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "collision_rate": measured["collision_rate"],
        "cosine_separation": measured["cosine_separation"],
        "fisher_ratio": measured["fisher_ratio"],
        "effective_rank": measured["effective_rank"],
        "nn_payload_contradiction": measured["collision_rate"],
    }


def _train_one(scorer, states: list[dict], seed: int) -> list[dict]:
    train_states = [state for state in states if state["split"] == "train"]
    optimizer = torch.optim.Adam(scorer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, int(EPOCHS) + 1):
        scorer.train()
        losses = []
        for state in train_states:
            if int(state["scenario_id"]) not in TRAIN_SCENARIOS:
                raise RuntimeError("confirmation state entered the REP-AUG gradient")
            optimizer.zero_grad(set_to_none=True)
            scores = scorer(_pack(state["embedded"], state["z_rel"]), torch.as_tensor(state["assignments"], dtype=torch.long))
            loss = joint_rank_loss(scores, _levels(state["j_mappo"]))
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError(f"REP-AUG loss became non-finite at seed {seed} epoch {epoch}")
            loss.backward()
            clip_grad_norm_(scorer.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scorer.eval()
        measured_rows = []
        with torch.no_grad():
            for index, state in enumerate(train_states):
                values = _score_aug(scorer, state)
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
                        "score_std": measured["score_std"],
                        "score_span": measured["score_span"],
                        "spearman": measured["spearman"],
                        "kendall": measured["kendall"],
                        "top1": measured["top1"],
                        "normalized_regret": measured["normalized_regret"],
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
                "score_std": _mean(row["score_std"] for row in measured_rows),
                "score_span": _mean(row["score_span"] for row in measured_rows),
                "spearman": _mean(row["spearman"] for row in measured_rows),
                "kendall": _mean(row["kendall"] for row in measured_rows),
                "top1": _mean(row["top1"] for row in measured_rows),
                "normalized_regret": _mean(row["normalized_regret"] for row in measured_rows),
                "collapsed": _mean(row["collapsed"] for row in measured_rows),
            }
        )
        print(f"rep-aug seed {seed} epoch {epoch} loss={_mean(losses):.4f}", flush=True)
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return curves


def _train_all(states: list[dict]) -> dict[int, float]:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    confirmation = [state for state in states if state["split"] == "confirmation"]
    init_collision = {}
    for seed in SEEDS:
        seed_everything(int(seed))
        scorer = RelationalSetScorer()
        scorer.eval()
        rows = [_rep_row("rep_aug", "init", int(seed), state, _latent_aug(scorer, state)) for state in confirmation]
        init_collision[int(seed)] = _mean(row["collision_rate"] for row in rows)
        curves = _train_one(scorer, states, int(seed))
        torch.save(
            {
                "seed": int(seed),
                "epoch": int(EPOCHS),
                "state_dict": scorer.state_dict(),
                "curves": curves,
                "init_rows": rows,
                "init_collision": init_collision[int(seed)],
                "rel_dim": REL_DIM,
            },
            _checkpoint(int(seed)),
        )
        print(f"saved {_checkpoint(int(seed)).name}", flush=True)
    return init_collision


def _load_aug() -> tuple[dict, list[dict], dict[int, float], list[dict]]:
    scorers = {}
    curves = []
    init_collision = {}
    init_rows = []
    for seed in SEEDS:
        payload = torch.load(_checkpoint(int(seed)), map_location="cpu", weights_only=False)
        scorer = RelationalSetScorer()
        scorer.load_state_dict(payload["state_dict"])
        scorer.eval()
        for parameter in scorer.parameters():
            parameter.requires_grad_(False)
        scorers[int(seed)] = scorer
        curves.extend(payload["curves"])
        init_collision[int(seed)] = float(payload["init_collision"])
        init_rows.extend(payload["init_rows"])
    return scorers, curves, init_collision, init_rows


def _load_base() -> dict:
    scorers = {}
    for seed in SEEDS:
        scorer = AssignmentConditionedSetScorer()
        scorer.load_state_dict(torch.load(PHASE5S_CHECKPOINTS / f"phase5s_model_c_seed{int(seed)}.pt", map_location="cpu", weights_only=False)["state_dict"])
        scorer.eval()
        scorers[int(seed)] = scorer
    return scorers


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
        "# HTW Phase-5U Assignment Representation Repair",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段只把当前状态能算出来的 assignment 关系特征拼到原来的 z_A 后面。Phase-5L encoder、ranking loss、训练场景和候选池都没有改。310110–310117 没有进入梯度。",
        "",
        "## 1. Collision",
        "",
        f"REP-BASE confirmation collision {_fmt(summary['collision_base'])}。",
        f"z_rel 本身 {_fmt(summary['collision_rel'])}。",
        f"REP-AUG 初始化 {_fmt(summary['collision_init'])}，epoch 10 {_fmt(summary['collision_aug'])}。",
        "",
        "## 2. 310114 t=15",
        "",
        "| seed | predicted | J_pred | optimum rank | J_opt | regret |",
        "| --- | --- | --- | --- | --- | --- |",
        *summary["case_lines"],
        "",
        f"五个 seed 的 optimum rank 中位数是 {_fmt(summary['case_rank'], '.1f')}。Phase-5T 大约在 78–104。",
        "",
        "## 3. 按决策时刻",
        "",
        "| t | BASE regret | AUG regret | BASE J | AUG J | ΔJ |",
        "| --- | --- | --- | --- | --- | --- |",
        *summary["time_lines"],
        "",
        f"整体 ΔJ mean {_fmt(summary['delta_j']['mean'], '.4e')}，95% CI [{_fmt(summary['delta_j']['low'], '.4e')}, {_fmt(summary['delta_j']['high'], '.4e')}]。",
        f"Δregret mean {_fmt(summary['delta_regret']['mean'], '.4e')}。scenario：更高 {summary['n_up']}，更低 {summary['n_down']}。seed 正向 {summary['n_seeds_up']}/5。",
        "",
        "## 4. Inference ablation",
        "",
        "| group | disagreement | Δregret | used |",
        "| --- | --- | --- | --- |",
        *summary["ablation_lines"],
        "",
        "## 5. 结论",
        "",
        f"Gate 是 **{summary['gate']}**。",
        "",
        summary["advice"],
        "",
        "Q1. collision 是否下降？ " + summary["answers"]["q1"],
        "",
        "Q2. 310114 t=15 的最优排名前移到多少？ " + summary["answers"]["q2"],
        "",
        "Q3. t=15 regret 是否明显下降？ " + summary["answers"]["q3"],
        "",
        "Q4. J_MAPPO 是否稳定提高？ " + summary["answers"]["q4"],
        "",
        "Q5. 哪一组特征在起作用？ " + summary["answers"]["q5"],
        "",
        "Q6. 是否还需要 TEMP？ " + summary["answers"]["q6"],
        "",
        "Q7. 下一阶段做什么？ " + summary["answers"]["q7"],
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
    confirmation = [state for state in states if state["split"] == "confirmation"]
    if any(not _checkpoint(int(seed)).is_file() for seed in SEEDS):
        _train_all(states)
    aug, curves, _init_collision, init_rows = _load_aug()
    base = _load_base()
    _write_csv(OUTPUT / "phase5u_training_curves.csv", curves)
    rep_rows = list(init_rows)
    for state in confirmation:
        rep_rows.append(_rep_row("rep_base", "epoch10", -1, state, state["z_rel"]))
        for seed in SEEDS:
            rep_rows.append(_rep_row("rep_base", "epoch10", int(seed), state, _latent_base(base[int(seed)], state)))
            rep_rows.append(_rep_row("rep_aug", "epoch10", int(seed), state, _latent_aug(aug[int(seed)], state)))
    state_rows = []
    case_rows = []
    paired = []
    for state in confirmation:
        per_seed = {}
        for seed in SEEDS:
            base_scores = _score_base(base[int(seed)], state)
            aug_scores = _score_aug(aug[int(seed)], state)
            base_row = _metric_row("rep_base", int(seed), state, base_scores)
            aug_row = _metric_row("rep_aug", int(seed), state, aug_scores)
            state_rows.extend((base_row, aug_row))
            paired.append(
                {
                    "scenario_id": int(state["scenario_id"]),
                    "time_step": int(state["time_step"]),
                    "seed": int(seed),
                    "delta_j_mappo": float(aug_row["j_mappo"] - base_row["j_mappo"]),
                    "delta_regret": float(aug_row["regret"] - base_row["regret"]),
                    "delta_normalized_regret": float(aug_row["normalized_regret"] - base_row["normalized_regret"]),
                    "delta_top1": float(aug_row["top1"] - base_row["top1"]),
                    "delta_optimum_rank": float(aug_row["optimum_rank"] - base_row["optimum_rank"]),
                }
            )
            per_seed[int(seed)] = aug_row
            if int(state["scenario_id"]) == CASE_SCENARIO and int(state["time_step"]) == CASE_TIME:
                levels = quantize(state["j_mappo"])
                optimum = state["vectors"][int(np.flatnonzero(levels == np.max(levels))[0])]
                case_rows.append(
                    {
                        "seed": int(seed),
                        "predicted_assignment": aug_row["assignment_vector"],
                        "predicted_score_rank": 1,
                        "pool_optimal_assignment": optimum,
                        "optimum_rank": aug_row["optimum_rank"],
                        "base_optimum_rank": base_row["optimum_rank"],
                        "j_mappo_pred": aug_row["j_mappo"],
                        "j_mappo_opt": aug_row["pool_oracle"],
                        "regret": aug_row["regret"],
                    }
                )
        print(f"scored {state['scenario_id']} t={state['time_step']}", flush=True)
    ablation_rows = []
    for name, (start, end) in GROUPS.items():
        disagreements = []
        regret_deltas = []
        for state in confirmation:
            for seed in SEEDS:
                full = _score_aug(aug[int(seed)], state)
                masked = np.array(state["z_rel"], copy=True)
                masked[:, start:end] = 0.0
                ablated = _score_aug(aug[int(seed)], state, masked)
                full_choice = int(np.argsort(-full, kind="stable")[0])
                ablated_choice = int(np.argsort(-ablated, kind="stable")[0])
                disagreements.append(float(full_choice != ablated_choice))
                full_regret = metrics_at(full, state["j_mappo"], full_choice)["regret"]
                ablated_regret = metrics_at(ablated, state["j_mappo"], ablated_choice)["regret"]
                regret_deltas.append(float(ablated_regret - full_regret))
        disagreement = _mean(disagreements)
        regret_delta = _mean(regret_deltas)
        ablation_rows.append(
            {
                "group": name,
                "disagreement": disagreement,
                "regret_increase": regret_delta,
                "used": float(disagreement >= ABLATION_DISAGREEMENT or regret_delta >= ABLATION_REGRET_BITS),
            }
        )
        print(f"ablated {name}", flush=True)
    collision_base = _mean(row["collision_rate"] for row in rep_rows if row["model"] == "rep_base" and int(row["seed"]) >= 0)
    collision_rel = _mean(row["collision_rate"] for row in rep_rows if row["model"] == "rep_base" and int(row["seed"]) < 0)
    collision_init = _mean(row["collision_rate"] for row in rep_rows if row["stage"] == "init")
    collision_aug = _mean(row["collision_rate"] for row in rep_rows if row["model"] == "rep_aug" and row["stage"] == "epoch10")
    delta_j = cluster_bootstrap(paired, "delta_j_mappo", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    delta_regret = cluster_bootstrap(paired, "delta_regret", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED + 1)
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
    drop = float(collision_base - collision_aug)
    collision_down = bool(drop >= COLLISION_ABS_DROP and drop / max(collision_base, 1.0e-6) >= COLLISION_REL_DROP)
    collision_worse = bool(collision_aug > collision_base + COLLISION_WORSE)
    case_rank = _median(row["optimum_rank"] for row in case_rows)
    t15_regret = _mean(row["regret"] for row in state_rows if row["model"] == "rep_aug" and int(row["time_step"]) == 15)
    regret_down = bool(delta_regret["mean"] < 0.0)
    j_up = bool(delta_j["mean"] > 0.0)
    no_reversal = bool(n_down * 2 <= len(CONFIRM_SCENARIOS) and n_seeds_down * 2 <= len(SEEDS))
    j_stable = bool(j_up and delta_j["low"] > 0.0 and n_up * 2 > len(CONFIRM_SCENARIOS) and n_seeds_up * 2 > len(SEEDS))
    gate = choose_gate(
        collision_down=collision_down,
        collision_worse=collision_worse,
        regret_down=regret_down,
        j_up=j_up,
        j_stable=j_stable,
        no_reversal=no_reversal,
        rank_clear=bool(case_rank <= RANK_CLEAR),
        rank_improved=bool(case_rank <= RANK_IMPROVED),
        t15_still_high=bool(t15_regret >= T15_STILL_HIGH),
        payload_regression=bool(delta_j["mean"] <= -PAYLOAD_REGRESSION),
    )
    advice = {
        "U-A": "固定 REP-AUG 表示。下一阶段用 310118–310125 做新的 development confirmation，暂时不加 TEMP。",
        "U-B": "表示有改善，但 t=15 的 regret 还高，或者 J_MAPPO 不稳定。下一阶段只做 TEMP augmentation。",
        "U-C": "这组静态关系特征没有修好 collision 和 payload。下一阶段不要继续堆同类特征，要单开 executor-conditioned short-horizon 研究。",
        "U-D": "REP-AUG 变差。退回 Phase-5S Model-C，不保留这组权重。",
    }[gate]
    time_lines = []
    for moment in TIMES:
        base_rows = [row for row in state_rows if row["model"] == "rep_base" and int(row["time_step"]) == int(moment)]
        aug_rows = [row for row in state_rows if row["model"] == "rep_aug" and int(row["time_step"]) == int(moment)]
        time_lines.append(
            "| " + " | ".join((
                str(moment),
                _fmt(_mean(row["regret"] for row in base_rows), ".4e"),
                _fmt(_mean(row["regret"] for row in aug_rows), ".4e"),
                _fmt(_mean(row["j_mappo"] for row in base_rows), ".4e"),
                _fmt(_mean(row["j_mappo"] for row in aug_rows), ".4e"),
                _fmt(_mean(row["delta_j_mappo"] for row in paired if int(row["time_step"]) == int(moment)), ".4e"),
            )) + " |"
        )
    case_lines = [
        f"| {row['seed']} | {row['predicted_assignment']} | {_fmt(row['j_mappo_pred'], '.4e')} | {_fmt(row['optimum_rank'], '.0f')} | {_fmt(row['j_mappo_opt'], '.4e')} | {_fmt(row['regret'], '.4e')} |"
        for row in case_rows
    ]
    ablation_lines = [
        f"| {row['group']} | {_fmt(row['disagreement'])} | {_fmt(row['regret_increase'], '.4e')} | {bool(row['used'])} |"
        for row in ablation_rows
    ]
    used = [row["group"] for row in ablation_rows if row["used"]]
    answers = {
        "q1": f"epoch-10 collision {_fmt(collision_aug)}，BASE {_fmt(collision_base)}，下降 {_fmt(drop)}。",
        "q2": f"中位数 {_fmt(case_rank, '.1f')}。",
        "q3": f"t=15 AUG regret {_fmt(t15_regret, '.4e')}。" + ("仍然不低于 1 个 payload block。" if t15_regret >= T15_STILL_HIGH else "已经低于 1 个 payload block。"),
        "q4": f"mean ΔJ {_fmt(delta_j['mean'], '.4e')}，95% CI [{_fmt(delta_j['low'], '.4e')}, {_fmt(delta_j['high'], '.4e')}]。scenario 正向 {n_up}/8，seed 正向 {n_seeds_up}/5。",
        "q5": "、".join(used) if used else "四组特征在推理消融里都没有越过预先写好的作用阈值。",
        "q6": {
            "U-A": "这一步先不加入。先用新的 confirmation 验证 REP-AUG。",
            "U-B": "需要。t=15 的执行差还在，下一阶段再加入 TEMP。",
            "U-C": "还不能把剩余问题归到 TEMP。静态关系特征没有把 collision 和 payload 一起修好。",
            "U-D": "先退回 Phase-5S，不在变差的表示上加 TEMP。",
        }[gate],
        "q7": advice,
    }
    summary = {
        "gate": gate,
        "advice": advice,
        "answers": answers,
        "collision_base": collision_base,
        "collision_rel": collision_rel,
        "collision_init": collision_init,
        "collision_aug": collision_aug,
        "case_rank": case_rank,
        "case_lines": case_lines,
        "time_lines": time_lines,
        "ablation_lines": ablation_lines,
        "delta_j": delta_j,
        "delta_regret": delta_regret,
        "n_up": n_up,
        "n_down": n_down,
        "n_seeds_up": n_seeds_up,
    }
    seed_rows = []
    for seed in SEEDS:
        for model in ("rep_base", "rep_aug"):
            rows = [row for row in state_rows if row["model"] == model and int(row["seed"]) == int(seed)]
            seed_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "j_mappo": _mean(row["j_mappo"] for row in rows),
                    "regret": _mean(row["regret"] for row in rows),
                    "top1": _mean(row["top1"] for row in rows),
                    "optimum_rank": _median(row["optimum_rank"] for row in rows),
                    "collapsed": _mean(row["collapsed"] for row in rows),
                    "delta_j_mappo": _mean(by_seed[int(seed)]) if model == "rep_aug" else "",
                }
            )
    time_rows = []
    for model in ("rep_base", "rep_aug"):
        for moment in TIMES:
            rows = [row for row in state_rows if row["model"] == model and int(row["time_step"]) == int(moment)]
            time_rows.append(
                {
                    "model": model,
                    "time_step": int(moment),
                    "j_mappo": _mean(row["j_mappo"] for row in rows),
                    "regret": _mean(row["regret"] for row in rows),
                    "normalized_regret": _mean(row["normalized_regret"] for row in rows),
                    "top1": _mean(row["top1"] for row in rows),
                    "spearman": _mean(row["spearman"] for row in rows),
                    "optimum_rank": _median(row["optimum_rank"] for row in rows),
                    "collapse_ratio": _mean(row["collapsed"] for row in rows),
                }
            )
    collision_summary = [
        {"representation": "rep_base_z_a", "collision_rate": collision_base},
        {"representation": "z_rel_only", "collision_rate": collision_rel},
        {"representation": "rep_aug_init", "collision_rate": collision_init},
        {"representation": "rep_aug_epoch10", "collision_rate": collision_aug},
    ]
    _write_csv(OUTPUT / "phase5u_representation_stats.csv", rep_rows)
    _write_csv(OUTPUT / "phase5u_collision_audit.csv", collision_summary)
    _write_csv(OUTPUT / "phase5u_state_metrics.csv", state_rows)
    _write_csv(OUTPUT / "phase5u_seed_metrics.csv", seed_rows)
    _write_csv(OUTPUT / "phase5u_timestep_metrics.csv", time_rows)
    _write_csv(OUTPUT / "phase5u_310114_t15_case.csv", case_rows)
    _write_csv(OUTPUT / "phase5u_paired_comparison.csv", paired)
    payload = {
        "gate": gate,
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "collision_base": collision_base,
        "collision_aug": collision_aug,
        "collision_down": collision_down,
        "case_rank_median": case_rank,
        "t15_regret": t15_regret,
        "delta_j": delta_j,
        "delta_regret": delta_regret,
        "n_scenarios_up": n_up,
        "n_scenarios_down": n_down,
        "n_seeds_up": n_seeds_up,
        "j_stable": j_stable,
        "ablation": ablation_rows,
        "keep_rep_aug": gate != "U-D",
    }
    (OUTPUT / "phase5u_representation_gate.json").write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")
    _write_report(summary)
    print(json.dumps(_plain({"gate": gate, "collision_base": collision_base, "collision_aug": collision_aug, "case_rank": case_rank, "delta_j": delta_j["mean"]})), flush=True)
    return summary


if __name__ == "__main__":
    run()
