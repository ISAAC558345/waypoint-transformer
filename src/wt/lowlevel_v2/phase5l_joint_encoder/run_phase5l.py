"""Train the joint-utility encoder. The Phase-5K matching decoder stays frozen."""
from __future__ import annotations

import copy
import json

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.htw_phase943.evaluation import encode_env
from wt.htw_phase9.const import BUOY_DIM
from wt.lowlevel_v2.integration_phase1.stats import spearman
from wt.lowlevel_v2.phase5f_utility_rank.protocol import FORBIDDEN_INPUTS
from wt.lowlevel_v2.phase5f_utility_rank.score import new_hgnn
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5g_closed_loop.protocol import UTILITY_CKPT
from wt.lowlevel_v2.phase5h_execution_audit.rollout import trace_h15
from wt.lowlevel_v2.phase5j_joint_redesign.joint import conflict_rate, duplicate_count
from wt.lowlevel_v2.phase5j_joint_redesign.run_phase5j import replay_env
from wt.lowlevel_v2.phase5k_joint_decoder.dataset import load_states
from wt.lowlevel_v2.phase5k_joint_decoder.decoders import JointAssignmentDecoder
from wt.lowlevel_v2.phase5k_joint_decoder.loss import joint_rank_loss
from wt.lowlevel_v2.phase5k_joint_decoder.protocol import OUTPUT as PHASE5K_OUTPUT
from wt.round30.ppo import seed_everything

from .model import JointUtilityHGNN, append_queue_edge, batch_tensors
from .protocol import (
    DEPLOYMENT_VERIFIED,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT,
    PATIENCE,
    PLANNER_READY,
    REPORT,
    TEST_SCENARIO,
    TRAIN_SCENARIO,
    WEIGHT_DECAY,
    assert_phase5l_seed,
)


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _fmt(value: float, digits: str = ".4f") -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), digits)


def _top1(scores: list[float], target: list[float]) -> float:
    best = max(target)
    winners = {index for index, value in enumerate(target) if float(value) == float(best)}
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    return float(int(order[0]) in winners)


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _check_batch(batch: dict) -> None:
    leaked = set(batch).intersection(FORBIDDEN_INPUTS)
    if leaked:
        raise RuntimeError(f"encoder input contains label fields: {sorted(leaked)}")


def _load_matching() -> JointAssignmentDecoder:
    path = PHASE5K_OUTPUT / "matching_decoder.pt"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    head = JointAssignmentDecoder("matching")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    head.load_state_dict(payload["state_dict"])
    _freeze(head)
    return head


def _load_baseline():
    if not UTILITY_CKPT.is_file():
        raise FileNotFoundError(str(UTILITY_CKPT))
    model = new_hgnn()
    model.load_state_dict(torch.load(UTILITY_CKPT, map_location="cpu", weights_only=False))
    _freeze(model)
    return model


def _baseline_embeddings(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    _check_batch(batch)
    with torch.no_grad():
        encoded = encode_env(model, batch)
    return encoded["h_u"], encoded["h_b"]


def _proposed_embeddings(model: JointUtilityHGNN, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    _check_batch(batch)
    conditioned = append_queue_edge(batch)
    return model.condition(batch_tensors(conditioned))


def _scores(head: JointAssignmentDecoder, embeddings: tuple[torch.Tensor, torch.Tensor], assignments: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(assignments, dtype=torch.long)
    return head.score_assignments(embeddings[0], embeddings[1], tensor)


def _ranking(scores: list[float], joints: list[float]) -> dict[str, float]:
    return {"spearman": spearman(scores, joints), "top1": _top1(scores, joints)}


def _execute(seed: int, moment: int, assignment: np.ndarray, actor) -> dict[str, float]:
    env, movement = replay_env(int(seed), int(moment))
    traced = trace_h15(env, assignment, actor, movement, "mappo")
    count = int(env.config.num_uavs)
    return {
        "mappo": float(traced["payload"]),
        "duplicate_rate": float(duplicate_count(assignment) / float(count)),
        "conflict_rate": conflict_rate(assignment, int(env.config.num_buoys)),
    }


def train_encoder(model: JointUtilityHGNN, head: JointAssignmentDecoder, states: list[dict]) -> dict:
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise RuntimeError("matching decoder must stay frozen")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    decoder_ids = {id(parameter) for parameter in head.parameters()}
    if any(id(parameter) in decoder_ids for parameter in optimizer.param_groups[0]["params"]):
        raise RuntimeError("optimizer includes the matching decoder")
    best_score = -1.0e18
    best_epoch = 0
    wait = 0
    best_weights = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        measured = []
        for state in states:
            optimizer.zero_grad(set_to_none=True)
            embeddings = _proposed_embeddings(model, state["urgency_batch"])
            assignments = np.asarray(state["assignments"], dtype=np.int64)
            scores = _scores(head, embeddings, assignments)
            joints = torch.as_tensor(state["joint"], dtype=torch.float32)
            loss = joint_rank_loss(scores, joints)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError(f"joint encoder loss became non-finite at epoch {epoch}")
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            measured.append(_ranking([float(item) for item in scores.detach().cpu().numpy()], [float(item) for item in state["joint"]]))
        spearman_value = _mean([row["spearman"] for row in measured])
        history.append({"epoch": epoch, "loss": _mean(losses), "train_spearman": spearman_value})
        print(f"encoder epoch {epoch} loss={_mean(losses):.4f} train_spearman={spearman_value:.4f}", flush=True)
        if np.isfinite(spearman_value) and float(spearman_value) > float(best_score) + 1.0e-6:
            best_score = float(spearman_value)
            best_epoch = int(epoch)
            wait = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= PATIENCE and best_epoch > 0:
                break
    model.load_state_dict(best_weights)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {"best_epoch": best_epoch, "best_train_spearman": best_score, "history": history}


def evaluate(name: str, embeddings_of, head: JointAssignmentDecoder, states: list[dict], actor) -> dict:
    rows = []
    for state in states:
        assert_phase5l_seed(int(state["scenario_id"]))
        joints = [float(item) for item in state["joint"]]
        embeddings = embeddings_of(state["urgency_batch"])
        with torch.no_grad():
            scores = _scores(head, embeddings, np.asarray(state["assignments"], dtype=np.int64))
            assignment = head.decode(embeddings[0], embeddings[1])
        ranking = _ranking([float(item) for item in scores.detach().cpu().numpy()], joints)
        executed = _execute(int(state["scenario_id"]), int(state["time_step"]), assignment, actor)
        split = "test" if int(state["scenario_id"]) == int(TEST_SCENARIO) else "train"
        rows.append(
            {
                "model": name,
                "split": split,
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "assignment": [int(item) for item in assignment],
                **executed,
                **ranking,
            }
        )
        print(f"eval {name} {state['scenario_id']} t={state['time_step']} mappo={executed['mappo']:.3e} spearman={ranking['spearman']:.4f}", flush=True)
    packed = {}
    for split in ("train", "test"):
        chosen = [row for row in rows if row["split"] == split]
        packed[split] = {
            "n_states": float(len(chosen)),
            "spearman": _mean([row["spearman"] for row in chosen]),
            "top1": _mean([row["top1"] for row in chosen]),
            "mappo": _mean([row["mappo"] for row in chosen]),
            "duplicate_rate": _mean([row["duplicate_rate"] for row in chosen]),
            "conflict_rate": _mean([row["conflict_rate"] for row in chosen]),
        }
    packed["states"] = rows
    return packed


def _table(summary: dict, split: str) -> list[str]:
    lines = [
        "| model | Spearman vs J(A) | Top-1 | joint MAPPO | duplicate rate | conflict rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for title, key in (("Phase-5K Matching decoder", "baseline"), ("Joint utility HGNN", "proposed")):
        item = summary[key][split]
        lines.append(
            f"| {title} | {_fmt(item['spearman'])} | {_fmt(item['top1'])} | {_fmt(item['mappo'], '.4e')} | {_fmt(item['duplicate_rate'])} | {_fmt(item['conflict_rate'])} |"
        )
    return lines


def _state_table(summary: dict) -> list[str]:
    lines = [
        "| scenario | t | split | baseline MAPPO | proposed MAPPO | baseline Spearman | proposed Spearman |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for base, proposed in zip(summary["baseline"]["states"], summary["proposed"]["states"]):
        lines.append(
            f"| {base['scenario_id']} | {base['time_step']} | {base['split']} | {_fmt(base['mappo'], '.4e')} | {_fmt(proposed['mappo'], '.4e')} | {_fmt(base['spearman'])} | {_fmt(proposed['spearman'])} |"
        )
    return lines


def _report(summary: dict, training: dict, gained: bool) -> str:
    if gained:
        decision = "在未参与梯度的 310101 上，Spearman 和联合 MAPPO 都高于 Phase-5K Matching decoder。联合效用表示有提高。本阶段不把 encoder 训练扩展到 deployment，也不开始 hierarchical joint planner。"
    else:
        decision = (
            f"310101 上，Joint utility HGNN 的 Spearman 是 {_fmt(summary['proposed']['test']['spearman'])}，"
            f"Phase-5K Matching decoder 是 {_fmt(summary['baseline']['test']['spearman'])}。"
            f"两者的联合 MAPPO 都是 {_fmt(summary['proposed']['test']['mappo'], '.4e')}，duplicate rate 和 conflict rate 都是 0。"
            "Spearman 有提高，联合 MAPPO 没有提高。需要进一步设计 hierarchical joint planner。本阶段不开始该设计。"
        )
    return "\n".join(
        [
            "# HTW Phase-5L Joint Utility HGNN Training",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "本阶段重新训练 HGNN encoder。Phase-5K 的 Matching decoder 保持冻结，没有修改 Residual MAPPO、低层策略、reward 或 planner，没有调用 deployment `decide()`，没有使用 230000–230127。",
            "",
            "## 1. 为什么训练 encoder",
            "",
            "Phase-5K 中 OrderFreeSetDecoder、Global assignment head 和 Matching decoder 的 duplicate target rate 与 conflict rate 都是 0。Matching decoder 能给出合法的一对一分配，但与 J(A) 的 Spearman 只有 0.0803。瓶颈是联合效用表示，不是分配耦合。",
            "",
            "## 2. 联合表示",
            "",
            "UAV-UAV 边保持现有的距离和干扰。UAV-Buoy 边在现有距离、速率和 ETA 之外，附上该浮标的当前队列比例。这个队列已经在浮标节点上，不使用未来轨迹。",
            "",
            "`z_joint` 是 H_UAV 与 H_Buoy 的掩码平均，再拼上当前已被占用浮标的池化向量和原有 global 特征。它经过线性层加回每个 UAV 和每个浮标的 individual embedding，然后进入冻结的 Matching decoder。Hungarian 读出仍然是 one-to-one，decoder 参数不更新。",
            "",
            "## 3. 训练",
            "",
            "标签是 Phase-5J 的 J(A)，不是 assignment similarity。损失是 `-log sigmoid(s(A) - s(B))`，只在 `J(A) > J(B)` 时计算。s(A) 是冻结 Matching decoder 在联合条件嵌入上的配对分数和。",
            "",
            f"梯度只使用 {TRAIN_SCENARIO} 的 t=0、15、30。{TEST_SCENARIO} 的三个时刻不进入梯度，用来和 Phase-5K Matching decoder 比较。没有新的 scenario。选模使用 310100 上的 Spearman，最佳轮次是 {training['best_epoch']}，训练 Spearman 是 {_fmt(training['best_train_spearman'])}。",
            "",
            "## 4. 结果",
            "",
            "### 测试场景 310101",
            "",
            *_table(summary, "test"),
            "",
            "### 训练场景 310100",
            "",
            *_table(summary, "train"),
            "",
            "### 各时刻",
            "",
            *_state_table(summary),
            "",
            "Top-1 是分配池里分数最高的分配是否为 J(A) 最高的分配。联合 MAPPO 是 Hungarian 读出交给冻结 MAPPO 执行 15 步的 payload。",
            "",
            "## 5. 判断",
            "",
            decision,
            "",
            "本阶段不进入 deployment verification。",
            "",
            "`planner_ready = False`",
            "",
            "`deployment_verified = False`",
            "",
        ]
    )


def run() -> dict:
    states = load_states()
    for state in states:
        assert_phase5l_seed(int(state["scenario_id"]))
    train_states = [state for state in states if int(state["scenario_id"]) == int(TRAIN_SCENARIO)]
    if not train_states or any(int(state["scenario_id"]) == int(TEST_SCENARIO) for state in train_states):
        raise RuntimeError("phase-5l training split is empty or contains the comparison scenario")
    seed_everything(0)
    proposed = JointUtilityHGNN(buoy_dim=int(BUOY_DIM) + 1)
    baseline = _load_baseline()
    proposed.load_utility_encoder(baseline.state_dict())
    proposed.reset_readout()
    head = _load_matching()
    training = train_encoder(proposed, head, train_states)
    actor = load_frozen_actor()
    measured = {
        "baseline": evaluate("baseline", lambda batch: _baseline_embeddings(baseline, batch), head, states, actor),
        "proposed": evaluate("proposed", lambda batch: _proposed_embeddings(proposed, batch), head, states, actor),
    }
    base_test = measured["baseline"]["test"]
    new_test = measured["proposed"]["test"]
    gained = bool(
        np.isfinite(new_test["spearman"])
        and np.isfinite(base_test["spearman"])
        and float(new_test["spearman"]) > float(base_test["spearman"])
        and float(new_test["mappo"]) > float(base_test["mappo"])
    )
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "decide_called": False,
        "encoder_trained": True,
        "matching_decoder_frozen": True,
        "hierarchical_planner_started": False,
        "mappo_modified": False,
        "reward_modified": False,
        "joint_ability_gained": gained,
        "training": training,
        "models": measured,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": proposed.state_dict(), "training": training}, OUTPUT / "joint_utility_encoder.pt")
    (OUTPUT / "phase5l_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(_report(measured, training, gained), encoding="utf-8")
    print(json.dumps({"gained": gained, "test_baseline": base_test, "test_proposed": new_test}, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    run()
