"""Train on 310100-310109, then score 310110-310117 once."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.lowlevel_v2.phase5k_joint_decoder.loss import joint_rank_loss
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5n_joint_planner.head import JointAssignmentHead
from wt.lowlevel_v2.phase5n_joint_planner.run_phase5n import _features
from wt.lowlevel_v2.phase5p_executor_joint_planner.metrics import signal_counts
from wt.lowlevel_v2.phase5p_executor_joint_planner.run_phase5p import _eval_head, load_states
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at, quantize
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.run_phase5q import _edge_features, _eval_scorer
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.scorer import AssignmentConditionedSetScorer
from wt.lowlevel_v2.phase5r_nonadditive_generalization.world import candidate_pool
from wt.round30.ppo import seed_everything

from .metrics import choose_gate, cluster_bootstrap
from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHECKPOINTS,
    CONFIRM_SCENARIOS,
    COVERAGE_EPS,
    DEPLOYMENT_VERIFIED,
    EPOCHS,
    EXPANDED_WIDTH,
    GRAD_CLIP,
    LEARNING_RATE,
    MODEL_POOL_WIDTH,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PHASE5R_CHECKPOINTS,
    PHASE5R_LABELS,
    PLANNER_READY,
    REPORT,
    SEEDS,
    TIMES,
    TRAIN_SCENARIOS,
    WEIGHT_DECAY,
    WORKERS,
    assert_train_seed,
)
from .rollout import _init_worker, roll_confirmation
from .world import replay_confirmation, replay_train


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
    if array.size < 2:
        return float("nan")
    return float(np.std(array, ddof=1))


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
    return CHECKPOINTS / f"phase5s_model_c_seed{int(seed)}.pt"


def _opened_marker():
    return OUTPUT / "phase5s_confirmation_opened.json"


def _freeze_protocol() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "phase5s_protocol_frozen.json"
    if path.is_file():
        return
    payload = {
        "architecture": "frozen Phase-5L encoder + AssignmentConditionedSetScorer",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "epochs": EPOCHS,
        "checkpoint": "epoch_10",
        "early_stopping": False,
        "seeds": list(SEEDS),
        "loss": "assignment-level pairwise ranking; pair kept only when the block gap is at least 1e6 bit",
        "payload_block_bits": PAYLOAD_BLOCK_BITS,
        "candidate_k": MODEL_POOL_WIDTH,
        "expanded_k_audit_only": EXPANDED_WIDTH,
        "train_scenarios": list(TRAIN_SCENARIOS),
        "confirmation_scenarios": list(CONFIRM_SCENARIOS),
        "confirmation_opened": False,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "tie_abs_bits": PAYLOAD_BLOCK_BITS,
        "close_oracle_bits": PAYLOAD_BLOCK_BITS,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_registry() -> None:
    payload = {
        "train_role": "PHASE5_SCORER_DEVELOPMENT_TRAINING_CONSUMED",
        "train_scenarios": list(TRAIN_SCENARIOS),
        "train_note": "310101-310109 were previously development evaluations. After this phase they cannot be reused as validation, confirmation, or test.",
        "confirmation_role": "PHASE5_SCORER_FRESH_CONFIRMATION",
        "confirmation_scenarios": list(CONFIRM_SCENARIOS),
        "confirmation_note": "Not a deployment holdout and not a final untouched test. Not used for training, early stopping, checkpoint selection, or architecture selection.",
    }
    (OUTPUT / "phase5s_data_registry.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _phase5r_pool_labels() -> dict[tuple[int, int, str], float]:
    table = {}
    with PHASE5R_LABELS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["in_model_pool"]) != 1.0:
                continue
            table[(int(row["scenario_id"]), int(row["time_step"]), row["assignment_vector"])] = float(row["j_mappo"])
    return table


def _pack(result: dict) -> dict:
    scores = np.asarray(result["scores"], dtype=np.float64)
    finite = bool(np.all(np.isfinite(scores)))
    std = float(np.std(scores)) if finite else float("nan")
    span = float(np.max(scores) - np.min(scores)) if finite else float("nan")
    packed = dict(result)
    packed["score_std"] = std
    packed["score_span"] = span
    packed["collapsed"] = float(not finite or std < 1.0e-6 or span < 1.0e-6)
    packed["oracle_j_mappo"] = float(result["j_mappo"] + result["regret"]) if np.isfinite(result["regret"]) else float("nan")
    return packed


def _levels(state: dict) -> torch.Tensor:
    values = torch.as_tensor(state["j_mappo"], dtype=torch.float32)
    return torch.round(values / PAYLOAD_BLOCK_BITS) * PAYLOAD_BLOCK_BITS


def _build_train_states(frozen, matching) -> tuple[list[dict], list[dict]]:
    recorded = _phase5r_pool_labels()
    states = []
    for old in load_states():
        assert_train_seed(int(old["scenario_id"]))
        state = dict(old)
        state["label_source"] = "phase5o"
        state["additive_features"] = _features(frozen, matching, state["urgency_batch"])
        state["embedded"] = _edge_features(frozen, state["urgency_batch"])
        states.append(state)
    for scenario in range(310102, 310110):
        for moment in TIMES:
            _env, _movement, batch, _diagnosis = replay_train(int(scenario), int(moment))
            additive = _features(frozen, matching, batch)
            embedded = _edge_features(frozen, batch)
            logits = additive["logits"].detach().cpu().numpy()
            pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
            vectors = [_vector(row) for row in pool["assignments"]]
            recorded_vectors = {key[2] for key in recorded if key[0] == int(scenario) and key[1] == int(moment)}
            if set(vectors) != recorded_vectors:
                raise RuntimeError(f"top-3 pool changed for {scenario} t={moment}; refusing to roll out a new pool")
            labels = np.asarray([recorded[(int(scenario), int(moment), vector)] for vector in vectors], dtype=np.float64)
            states.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(moment),
                    "urgency_batch": batch,
                    "assignments": pool["assignments"],
                    "candidate_mask": pool["candidate_mask"],
                    "vectors": vectors,
                    "j_mappo": labels,
                    "label_source": "phase5r",
                    "additive_features": additive,
                    "embedded": embedded,
                }
            )
            print(f"train labels {scenario} t={moment} n={len(vectors)} source=phase5r", flush=True)
    states.sort(key=lambda state: (int(state["scenario_id"]), int(state["time_step"])))
    if len(states) != len(TRAIN_SCENARIOS) * len(TIMES):
        raise RuntimeError("training state count is not 30")
    signal_rows = []
    for state in states:
        counts = signal_counts(state["j_mappo"])
        signal_rows.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "label_source": state["label_source"],
                **counts,
            }
        )
    return states, signal_rows


def _train_scorer(scorer, states: list[dict], seed: int) -> list[dict]:
    optimizer = torch.optim.Adam(scorer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, int(EPOCHS) + 1):
        scorer.train()
        losses = []
        for state in states:
            if int(state["scenario_id"]) not in TRAIN_SCENARIOS:
                raise RuntimeError("a non-training scenario entered the gradient")
            optimizer.zero_grad(set_to_none=True)
            scores = scorer(state["embedded"], torch.as_tensor(state["assignments"], dtype=torch.long))
            loss = joint_rank_loss(scores, _levels(state))
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError(f"phase-5s loss became non-finite at seed {seed} epoch {epoch}")
            loss.backward()
            clip_grad_norm_(scorer.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scorer.eval()
        measured_rows = []
        with torch.no_grad():
            for index, state in enumerate(states):
                values = scorer(state["embedded"], torch.as_tensor(state["assignments"], dtype=torch.long)).detach().cpu().numpy()
                chosen = int(np.argsort(-values, kind="stable")[0])
                measured = metrics_at(values, state["j_mappo"], chosen)
                measured["scores"] = values
                measured = _pack(measured)
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
        print(f"phase5s seed {seed} epoch {epoch} loss={_mean(losses):.4f}", flush=True)
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return curves


def _train_all(states: list[dict]) -> None:
    if _opened_marker().is_file():
        raise RuntimeError("refusing to train after confirmation scenarios were opened")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        seed_everything(int(seed))
        scorer = AssignmentConditionedSetScorer()
        curves = _train_scorer(scorer, states, int(seed))
        torch.save(
            {
                "seed": int(seed),
                "epoch": int(EPOCHS),
                "state_dict": scorer.state_dict(),
                "train_scenarios": list(TRAIN_SCENARIOS),
                "curves": curves,
            },
            _checkpoint(int(seed)),
        )
        print(f"saved {_checkpoint(int(seed)).name}", flush=True)
    for seed in SEEDS:
        if not _checkpoint(int(seed)).is_file():
            raise RuntimeError("checkpoint save failed")
    print("CHECKPOINTS_SAVED", flush=True)


def _load_phase5s() -> tuple[dict, list[dict]]:
    scorers = {}
    curves = []
    for seed in SEEDS:
        payload = torch.load(_checkpoint(int(seed)), map_location="cpu", weights_only=False)
        scorer = AssignmentConditionedSetScorer()
        scorer.load_state_dict(payload["state_dict"])
        scorer.eval()
        for parameter in scorer.parameters():
            parameter.requires_grad_(False)
        scorers[int(seed)] = scorer
        curves.extend(payload["curves"])
    return scorers, curves


def _load_phase5q() -> tuple[dict, dict]:
    heads = {}
    scorers = {}
    for seed in SEEDS:
        head = JointAssignmentHead()
        head.load_state_dict(torch.load(PHASE5R_CHECKPOINTS / f"model_a_seed{seed}.pt", map_location="cpu", weights_only=False)["state_dict"])
        scorer = AssignmentConditionedSetScorer()
        scorer.load_state_dict(torch.load(PHASE5R_CHECKPOINTS / f"model_c_seed{seed}.pt", map_location="cpu", weights_only=False)["state_dict"])
        head.eval()
        scorer.eval()
        heads[int(seed)] = head
        scorers[int(seed)] = scorer
    return heads, scorers


def _prepare_confirmation(frozen, matching) -> list[dict]:
    if any(not _checkpoint(int(seed)).is_file() for seed in SEEDS):
        raise RuntimeError("confirmation opened before checkpoints were saved")
    _opened_marker().write_text(json.dumps({"opened": True, "scenarios": list(CONFIRM_SCENARIOS)}), encoding="utf-8")
    prepared = []
    for scenario in CONFIRM_SCENARIOS:
        for moment in TIMES:
            _env, _movement, batch, diagnosis = replay_confirmation(int(scenario), int(moment))
            additive = _features(frozen, matching, batch)
            embedded = _edge_features(frozen, batch)
            logits = additive["logits"].detach().cpu().numpy()
            pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
            expanded = candidate_pool(logits, EXPANDED_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
            print(f"confirm pool {scenario} t={moment} model={pool['assignments'].shape[0]} expanded={expanded['assignments'].shape[0]}", flush=True)
            prepared.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(moment),
                    "additive_features": additive,
                    "embedded": embedded,
                    "pool": pool,
                    "expanded": expanded,
                    "diagnosis": diagnosis,
                }
            )
    return prepared


def _confirmation_labels(prepared: list[dict]) -> dict[tuple[int, int, str], float]:
    jobs = []
    for state in prepared:
        vectors = []
        seen = set()
        for assignments in (state["pool"]["assignments"], state["expanded"]["assignments"]):
            for row in assignments:
                vector = _vector(row)
                if vector in seen:
                    continue
                seen.add(vector)
                vectors.append(vector)
        jobs.append((int(state["scenario_id"]), int(state["time_step"]), vectors))
    cache = {}
    print(f"confirmation rollouts {sum(len(item[2]) for item in jobs)}", flush=True)
    with ProcessPoolExecutor(max_workers=int(WORKERS), initializer=_init_worker) as pool:
        futures = [pool.submit(roll_confirmation, job) for job in jobs]
        for future in as_completed(futures):
            scenario, moment, rows = future.result()
            for vector, value in rows:
                cache[(int(scenario), int(moment), vector)] = float(value)
            print(f"rolled {scenario} t={moment} n={len(rows)}", flush=True)
    return cache


def _view(state: dict, labels: np.ndarray) -> dict:
    return {
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "assignments": state["pool"]["assignments"],
        "candidate_mask": state["pool"]["candidate_mask"],
        "vectors": [_vector(row) for row in state["pool"]["assignments"]],
        "j_mappo": labels,
    }


def _edge_text(diagnosis: dict, vector: str) -> str:
    assignment = [int(item) for item in vector.split(";")]
    parts = [f"interference_sum={float(np.sum(diagnosis['interference'])):.6g}"]
    for uav, buoy in enumerate(assignment):
        if int(buoy) < 0:
            continue
        parts.append(
            "uav={0},buoy={1},distance_m={2:.4f},eta={3:.4f},channel_bps={4:.6e},queue={5:.4f},urgency={6:.4f}".format(
                uav,
                buoy,
                float(diagnosis["distance"][uav, buoy]),
                float(diagnosis["eta"][uav, buoy]),
                float(diagnosis["rate"][uav, buoy]),
                float(diagnosis["queue"][buoy]),
                float(diagnosis["urgency"][buoy]),
            )
        )
    return " | ".join(parts)


def _interaction_text(scorer, features: dict, vector: str) -> str:
    assignment = [int(item) for item in vector.split(";")]
    h_edge = features["h_edge"][0]
    parts = []
    with torch.no_grad():
        for left in range(len(assignment)):
            for right in range(left + 1, len(assignment)):
                if assignment[left] < 0 or assignment[right] < 0:
                    continue
                pair = torch.cat((h_edge[left, assignment[left]], h_edge[right, assignment[right]]), dim=-1).view(1, -1)
                value = scorer.interact(pair)
                parts.append(f"{left}-{right}:{float(torch.linalg.vector_norm(value)):.4f}")
    return " ".join(parts)


def _failure_class(scorer_regret: float, coverage_regret: float, top1: float, collapsed: float) -> str:
    if float(coverage_regret) >= float(PAYLOAD_BLOCK_BITS) and float(coverage_regret) >= float(scorer_regret):
        return "C_candidate_coverage"
    if float(top1) < 1.0 and float(collapsed) < 0.5:
        return "A_score_ordering_error"
    return "B_unseen_state_generalization"


def _state_row(model: str, seed: int, scenario: int, moment: int, result: dict, pool_oracle: float) -> dict:
    return {
        "model": model,
        "seed": int(seed),
        "scenario_id": int(scenario),
        "time_step": int(moment),
        "spearman": result.get("spearman", float("nan")),
        "kendall": result.get("kendall", float("nan")),
        "top1": result["top1"],
        "top3": result["top3"],
        "j_mappo": result["j_mappo"],
        "pool_oracle": float(pool_oracle),
        "regret": result["regret"],
        "normalized_regret": result["normalized_regret"],
        "score_std": result.get("score_std", float("nan")),
        "score_span": result.get("score_span", float("nan")),
        "collapsed": result.get("collapsed", float("nan")),
        "inference_ms": result.get("inference_ms", float("nan")),
        "assignment_vector": result["assignment_vector"],
    }


def _evaluate(prepared, cache, heads, phase5q, phase5s) -> dict:
    truth_rows = []
    state_rows = []
    paired = []
    regret_rows = []
    measured = {}
    coverage = {}
    for state in prepared:
        scenario = int(state["scenario_id"])
        moment = int(state["time_step"])
        pool_vectors = [_vector(row) for row in state["pool"]["assignments"]]
        expanded_vectors = {_vector(row) for row in state["expanded"]["assignments"]}
        pool_labels = np.asarray([cache[(scenario, moment, vector)] for vector in pool_vectors], dtype=np.float64)
        union_values = [cache[(scenario, moment, vector)] for vector in set(pool_vectors) | expanded_vectors]
        pool_oracle = float(np.max(pool_labels))
        expanded_oracle = float(np.max(union_values))
        coverage_regret = float(expanded_oracle - pool_oracle)
        coverage[(scenario, moment)] = {
            "pool_oracle": pool_oracle,
            "expanded_oracle": expanded_oracle,
            "coverage_regret": coverage_regret,
            "coverage_ratio": float(pool_oracle / (expanded_oracle + COVERAGE_EPS)),
        }
        for vector in sorted(set(pool_vectors) | expanded_vectors):
            truth_rows.append(
                {
                    "scenario_id": scenario,
                    "time_step": moment,
                    "assignment_vector": vector,
                    "j_mappo": float(cache[(scenario, moment, vector)]),
                    "in_model_pool": float(vector in set(pool_vectors)),
                    "in_expanded_pool": float(vector in expanded_vectors),
                }
            )
        view = _view(state, pool_labels)
        winners = np.flatnonzero(quantize(pool_labels) == np.max(quantize(pool_labels)))
        oracle = {
            "spearman": 1.0,
            "kendall": 1.0,
            "top1": 1.0,
            "top3": 1.0,
            "j_mappo": pool_oracle,
            "oracle_j_mappo": pool_oracle,
            "regret": 0.0,
            "normalized_regret": 0.0,
            "score_std": float("nan"),
            "score_span": float("nan"),
            "collapsed": 0.0,
            "inference_ms": float("nan"),
            "assignment_vector": pool_vectors[int(winners[0])],
            "n_optimal": int(winners.size),
        }
        measured[("pool_oracle", -1, scenario, moment)] = oracle
        state_rows.append(_state_row("pool_oracle", -1, scenario, moment, oracle, pool_oracle))
        for seed in SEEDS:
            started = time.perf_counter()
            additive = _pack(_eval_head(heads[int(seed)], state["additive_features"], view))
            additive["inference_ms"] = (time.perf_counter() - started) * 1.0e3
            started = time.perf_counter()
            single = _pack(_eval_scorer(phase5q[int(seed)], state["embedded"], view))
            single["inference_ms"] = (time.perf_counter() - started) * 1.0e3
            started = time.perf_counter()
            multi = _pack(_eval_scorer(phase5s[int(seed)], state["embedded"], view))
            multi["inference_ms"] = (time.perf_counter() - started) * 1.0e3
            for model, result in (("model_a", additive), ("phase5q_model_c", single), ("phase5s_model_c", multi)):
                measured[(model, int(seed), scenario, moment)] = result
                state_rows.append(_state_row(model, int(seed), scenario, moment, result, pool_oracle))
                regret_rows.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "scenario_id": scenario,
                        "time_step": moment,
                        "scorer_regret": float(result["regret"]),
                        "coverage_regret": coverage_regret,
                        "total_regret": float(result["regret"] + coverage_regret),
                    }
                )
            paired.append(
                {
                    "scenario_id": scenario,
                    "time_step": moment,
                    "seed": int(seed),
                    "delta_j_mappo": float(multi["j_mappo"] - single["j_mappo"]),
                    "delta_normalized_regret": float(multi["normalized_regret"] - single["normalized_regret"]),
                    "delta_top1": float(multi["top1"] - single["top1"]),
                    "delta_spearman": float(multi["spearman"] - single["spearman"]),
                }
            )
        print(f"scored {scenario} t={moment}", flush=True)
    failures = []
    for (model, seed, scenario, moment), result in measured.items():
        if model != "phase5s_model_c":
            continue
        state = next(item for item in prepared if int(item["scenario_id"]) == int(scenario) and int(item["time_step"]) == int(moment))
        oracle = measured[("pool_oracle", -1, scenario, moment)]
        coverage_regret = coverage[(scenario, moment)]["coverage_regret"]
        failures.append(
            {
                "scenario_id": int(scenario),
                "time_step": int(moment),
                "seed": int(seed),
                "regret": float(result["regret"]),
                "predicted_assignment": result["assignment_vector"],
                "pool_optimal_assignment": oracle["assignment_vector"],
                "n_optimal": oracle["n_optimal"],
                "predicted_score": float(result["scores"][int(result["chosen"])]),
                "j_mappo_predicted": float(result["j_mappo"]),
                "j_mappo_oracle": float(oracle["j_mappo"]),
                "predicted_edges": _edge_text(state["diagnosis"], result["assignment_vector"]),
                "oracle_edges": _edge_text(state["diagnosis"], oracle["assignment_vector"]),
                "selected_edge_interaction": _interaction_text(phase5s[int(seed)], state["embedded"], result["assignment_vector"]),
                "failure_class": _failure_class(float(result["regret"]), coverage_regret, float(result["top1"]), float(result["collapsed"])),
                "coverage_regret": coverage_regret,
            }
        )
    failures.sort(key=lambda row: (-float(row["regret"]), int(row["scenario_id"]), int(row["time_step"]), int(row["seed"])))
    return {
        "truth_rows": truth_rows,
        "state_rows": state_rows,
        "paired": paired,
        "regret_rows": regret_rows,
        "measured": measured,
        "coverage": coverage,
        "failures": failures[:10],
    }


def _answers(summary: dict) -> dict[str, str]:
    gate = summary["gate"]
    q1 = "是。" if gate == "S-A" else "没有达到预先写好的稳定提高标准。"
    q1 += f" mean ΔJ_MAPPO {_fmt(summary['delta_j']['mean'], '.4e')}，95% CI [{_fmt(summary['delta_j']['low'], '.4e')}, {_fmt(summary['delta_j']['high'], '.4e')}]。"
    q2 = q1
    q3 = f"scenario：更高 {summary['n_up']}，持平 {summary['n_tie']}，更低 {summary['n_down']}。seed：{summary['n_seeds_up']}/5 个平均差大于 0。"
    if summary["coverage_regret"] > summary["scorer_regret"]:
        q4 = f"COVERAGE_REGRET 更大，{_fmt(summary['coverage_regret'], '.4e')}，scorer regret {_fmt(summary['scorer_regret'], '.4e')}。"
    else:
        q4 = f"SCORER_REGRET 更大，{_fmt(summary['scorer_regret'], '.4e')}，coverage regret {_fmt(summary['coverage_regret'], '.4e')}。"
    if gate == "S-A":
        q5 = "开始优化 Stage-1 candidate generator。" if summary["coverage_regret"] > summary["scorer_regret"] else "继续扩大 scorer development data，不改结构。"
        q6 = "有。可以把冻结的 Phase-5L encoder 加上 Phase-5S non-additive scorer 作为当前 development 用的高层结构。现在仍不能部署。"
    elif gate == "S-B":
        q5 = "开始优化 Stage-1 candidate generator。"
        q6 = "scorer 架构可以保留，但还不能把更多数据训练当成已经解决 regret。"
    elif gate == "S-C":
        q5 = "继续扩大 development training scenarios，不增加网络复杂度。"
        q6 = "还不能正式固定。平均方向是正的，但还不稳。"
    else:
        q5 = "先检查 non-additive scorer 的 state representation 和 capacity，不能直接增加 HGNN layer。"
        q6 = "没有。fresh confirmation 没有证明把训练场景扩到 10 个已经解决问题。"
    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5, "q6": q6}


def _write_report(summary: dict) -> None:
    lines = [
        "# HTW Phase-5S Multi-Scenario Non-Additive Joint Scorer",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段只把训练场景从 310100 扩到 310100–310109。AssignmentConditionedSetScorer、冻结 encoder、loss、学习率、10 个 epoch 和 seed 0–4 都不变。310110–310117 在五个 checkpoint 保存之后才第一次生成标签，并且只评价一次。",
        "",
        "## 1. 训练",
        "",
        f"训练状态 30 个，标签全部来自已有的 Phase-5O / Phase-5R，没有为训练集重新 rollout。",
        f"第 10 个 epoch 的平均损失是 {_fmt(summary['final_loss'])}。平均 collapse ratio 是 {_fmt(summary['final_collapse'])}。平均训练 Top-1 是 {_fmt(summary['final_top1'])}。",
        "",
        "## 2. Fresh confirmation（310110–310117）",
        "",
        "比较的是 Phase-5S 减去只用 310100 训练的 Phase-5Q Model-C。统计以 scenario 为 cluster。",
        "",
        f"ΔJ_MAPPO mean {_fmt(summary['delta_j']['mean'], '.4e')}，median {_fmt(summary['delta_j_median'], '.4e')}，std {_fmt(summary['delta_j_std'], '.4e')}，positive fraction {_fmt(summary['positive_fraction'])}。",
        f"95% CI [{_fmt(summary['delta_j']['low'], '.4e')}, {_fmt(summary['delta_j']['high'], '.4e')}]。",
        f"Δnormalized regret mean {_fmt(summary['delta_regret']['mean'])}，95% CI [{_fmt(summary['delta_regret']['low'])}, {_fmt(summary['delta_regret']['high'])}]。",
        f"8 个 scenario：更高 {summary['n_up']}，持平 {summary['n_tie']}，更低 {summary['n_down']}。5 个 seed 里 {summary['n_seeds_up']} 个平均差大于 0。",
        "",
        "| scenario | ΔJ_MAPPO | Δregret | Phase-5S Top-1 | Phase-5Q Top-1 | scorer regret | coverage regret |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *summary["scenario_lines"],
        "",
        f"Phase-5S 平均 scorer regret {_fmt(summary['scorer_regret'], '.4e')}。平均 coverage regret {_fmt(summary['coverage_regret'], '.4e')}。",
        "",
        "## 3. Regret 最大的 10 个 state × seed",
        "",
        "| scenario | t | seed | regret | class |",
        "| --- | --- | --- | --- | --- |",
        *summary["failure_lines"],
        "",
        "## 4. 结论",
        "",
        f"Gate 是 **{summary['gate']}**。",
        "",
        summary["advice"],
        "",
        "Q1. 训练场景从 1 个扩到 10 个是否提高了 scorer？ " + summary["answers"]["q1"],
        "",
        "Q2. 新 confirmation 上是否稳定优于 Phase-5Q？ " + summary["answers"]["q2"],
        "",
        "Q3. seed 和 scenario 是否同向？ " + summary["answers"]["q3"],
        "",
        "Q4. 哪一类 regret 更大？ " + summary["answers"]["q4"],
        "",
        "Q5. 下一步做什么？ " + summary["answers"]["q5"],
        "",
        "Q6. 能否把 encoder 加 non-additive scorer 固定为当前高层结构？ " + summary["answers"]["q6"],
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    _freeze_protocol()
    _write_registry()
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    if any(not _checkpoint(int(seed)).is_file() for seed in SEEDS):
        states, signal_rows = _build_train_states(frozen, matching)
        _write_csv(OUTPUT / "phase5s_training_signal_stats.csv", signal_rows)
        _train_all(states)
    else:
        print("CHECKPOINTS_SAVED", flush=True)
        if not (OUTPUT / "phase5s_training_signal_stats.csv").is_file():
            _states, signal_rows = _build_train_states(frozen, matching)
            _write_csv(OUTPUT / "phase5s_training_signal_stats.csv", signal_rows)
    phase5s, curves = _load_phase5s()
    _write_csv(OUTPUT / "phase5s_training_curves.csv", curves)
    heads, phase5q = _load_phase5q()
    prepared = _prepare_confirmation(frozen, matching)
    cache = _confirmation_labels(prepared)
    evaluated = _evaluate(prepared, cache, heads, phase5q, phase5s)
    paired = evaluated["paired"]
    delta_j = cluster_bootstrap(paired, "delta_j_mappo", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    delta_regret = cluster_bootstrap(paired, "delta_normalized_regret", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED + 1)
    by_scenario = defaultdict(list)
    by_seed = defaultdict(list)
    for row in paired:
        by_scenario[int(row["scenario_id"])].append(float(row["delta_j_mappo"]))
        by_seed[int(row["seed"])].append(float(row["delta_j_mappo"]))
    scenario_means = {scenario: float(np.mean(values)) for scenario, values in by_scenario.items()}
    n_up = sum(value > 1.0 for value in scenario_means.values())
    n_down = sum(value < -1.0 for value in scenario_means.values())
    n_tie = len(scenario_means) - n_up - n_down
    n_seeds_up = sum(float(np.mean(values)) > 0.0 for values in by_seed.values())
    scorer_regret = _mean(row["scorer_regret"] for row in evaluated["regret_rows"] if row["model"] == "phase5s_model_c")
    phase5q_regret = _mean(row["scorer_regret"] for row in evaluated["regret_rows"] if row["model"] == "phase5q_model_c")
    coverage_regret = _mean(item["coverage_regret"] for item in evaluated["coverage"].values())
    gate = choose_gate(
        mean_delta_j=delta_j["mean"],
        delta_j_low=delta_j["low"],
        mean_delta_regret=delta_regret["mean"],
        n_scenarios_up=n_up,
        n_scenarios=len(CONFIRM_SCENARIOS),
        n_seeds_up=n_seeds_up,
        n_seeds=len(SEEDS),
        regret_phase5q=phase5q_regret,
        regret_phase5s=scorer_regret,
        coverage_regret=coverage_regret,
    )
    advice = {
        "S-A": "development-state diversity 进一步提高了 non-additive scorer。下一阶段冻结 Phase-5S scorer。",
        "S-B": "Phase-5S 与 Phase-5Q 基本持平，而且都接近候选池 oracle，coverage regret 更大。下一阶段优化 Stage-1 candidate coverage，不改 scorer。",
        "S-C": "平均收益为正，但置信区间或 scenario / seed 方向还不稳。继续扩大 development training scenarios，不增加网络复杂度。",
        "S-D": "fresh confirmation 没有证明多场景训练已经解决问题。下一步检查 scorer 的 state representation 和 capacity，不能直接增加 HGNN layer。",
    }[gate]
    scenario_lines = []
    for scenario in CONFIRM_SCENARIOS:
        rows = [row for row in paired if int(row["scenario_id"]) == int(scenario)]
        s_top = [float(item["top1"]) for (model, _seed, row_scenario, _moment), item in evaluated["measured"].items() if model == "phase5s_model_c" and int(row_scenario) == int(scenario)]
        q_top = [float(item["top1"]) for (model, _seed, row_scenario, _moment), item in evaluated["measured"].items() if model == "phase5q_model_c" and int(row_scenario) == int(scenario)]
        s_regret = [float(row["scorer_regret"]) for row in evaluated["regret_rows"] if row["model"] == "phase5s_model_c" and int(row["scenario_id"]) == int(scenario)]
        c_regret = [float(item["coverage_regret"]) for (row_scenario, _moment), item in evaluated["coverage"].items() if int(row_scenario) == int(scenario)]
        scenario_lines.append(
            "| " + " | ".join((
                str(scenario),
                _fmt(_mean(row["delta_j_mappo"] for row in rows), ".4e"),
                _fmt(_mean(row["delta_normalized_regret"] for row in rows)),
                _fmt(_mean(s_top)),
                _fmt(_mean(q_top)),
                _fmt(_mean(s_regret), ".4e"),
                _fmt(_mean(c_regret), ".4e"),
            )) + " |"
        )
    failure_lines = [
        f"| {row['scenario_id']} | {row['time_step']} | {row['seed']} | {_fmt(row['regret'], '.4e')} | {row['failure_class']} |"
        for row in evaluated["failures"]
    ]
    final_rows = [row for row in curves if row["row_type"] == "seed_epoch" and int(row["epoch"]) == int(EPOCHS)]
    seed_rows = []
    for seed in SEEDS:
        for model in ("model_a", "phase5q_model_c", "phase5s_model_c"):
            rows = [item for (name, row_seed, _scenario, _moment), item in evaluated["measured"].items() if name == model and int(row_seed) == int(seed)]
            seed_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "spearman": _mean(row["spearman"] for row in rows),
                    "top1": _mean(row["top1"] for row in rows),
                    "j_mappo": _mean(row["j_mappo"] for row in rows),
                    "regret": _mean(row["regret"] for row in rows),
                    "normalized_regret": _mean(row["normalized_regret"] for row in rows),
                    "collapsed": _mean(row["collapsed"] for row in rows),
                    "delta_j_mappo": _mean(by_seed[int(seed)]) if model == "phase5s_model_c" else "",
                }
            )
    summary = {
        "gate": gate,
        "advice": advice,
        "final_loss": _mean(row["loss"] for row in final_rows),
        "final_collapse": _mean(row["collapsed"] for row in final_rows),
        "final_top1": _mean(row["top1"] for row in final_rows),
        "delta_j": delta_j,
        "delta_j_median": _median(row["delta_j_mappo"] for row in paired),
        "delta_j_std": _std(row["delta_j_mappo"] for row in paired),
        "positive_fraction": _mean(float(row["delta_j_mappo"]) > 0.0 for row in paired),
        "delta_regret": delta_regret,
        "n_up": n_up,
        "n_tie": n_tie,
        "n_down": n_down,
        "n_seeds_up": n_seeds_up,
        "scorer_regret": scorer_regret,
        "coverage_regret": coverage_regret,
        "scenario_lines": scenario_lines,
        "failure_lines": failure_lines,
    }
    summary["answers"] = _answers(summary)
    _write_csv(OUTPUT / "phase5s_confirmation_ground_truth.csv", evaluated["truth_rows"])
    _write_csv(OUTPUT / "phase5s_state_metrics.csv", evaluated["state_rows"])
    _write_csv(OUTPUT / "phase5s_seed_metrics.csv", seed_rows)
    _write_csv(OUTPUT / "phase5s_paired_comparison.csv", paired)
    _write_csv(OUTPUT / "phase5s_regret_decomposition.csv", evaluated["regret_rows"])
    _write_csv(OUTPUT / "phase5s_failure_cases.csv", evaluated["failures"])
    payload = {
        "gate": gate,
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "delta_j": delta_j,
        "delta_regret": delta_regret,
        "n_scenarios_up": n_up,
        "n_scenarios_tie": n_tie,
        "n_scenarios_down": n_down,
        "n_seeds_up": n_seeds_up,
        "scorer_regret": scorer_regret,
        "phase5q_scorer_regret": phase5q_regret,
        "coverage_regret": coverage_regret,
    }
    (OUTPUT / "phase5s_multiscenario_training_gate.json").write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")
    _write_report(summary)
    print(json.dumps(_plain({"gate": gate, "delta_j": delta_j, "n_up": n_up, "n_seeds_up": n_seeds_up})), flush=True)
    return summary


if __name__ == "__main__":
    run()
