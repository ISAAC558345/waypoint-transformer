"""Read-only objective audit and one offline payload-rank probe. No training."""
from __future__ import annotations

import json

import numpy as np

from wt.htw_phase945.protocol import LAMBDA_RANK, LAMBDA_REF, TAU
from wt.htw_phase945.utility_rollout import shadow_h15_payload
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman
from wt.lowlevel_v2.phase5_hgnn_urgency.features import observable_exclusive_assignment

from .protocol import AUDIT_SEEDS, DEPLOYMENT_VERIFIED, OUTPUT, PLANNER_READY, REPORT, assert_audit_seed


def objective_record() -> dict:
    return {
        "official_checkpoint": "final_artifacts/checkpoints/deepsets_util_seed0.pt",
        "official_input": [
            "UAV: normalized position, velocity, progress, assigned flag, assigned queue, energy slot fixed at 0",
            "Buoy: normalized position, queue/capacity, last arrival, progress, occupied, buoy-count ratio",
            "UAV-Buoy edge: distance, hypothetical rate, ETA",
            "UAV-UAV edge: distance, active-link interference ratio",
        ],
        "official_label": "softmax over a frozen candidate pool of H=15 fly-toward payload increments",
        "official_loss": f"L_util + {LAMBDA_RANK} L_rank + {LAMBDA_REF} L_ref",
        "official_tau_read_only": float(TAU),
        "official_reward": "none; the high-level has no reward and no payload term inside decode",
        "directly_maximizes_episode_payload": False,
        "phase5_hgnn_label": "observable exclusive assignment: queue * current hypothetical rate / (distance + 1)",
        "phase5_hgnn_loss": "order-free set NLL of that single assignment",
        "phase5_hgnn_is_payload": False,
    }


def feature_specification() -> list[dict[str, str]]:
    return [
        {
            "term": "expected_payload_gain",
            "alpha": "free, not fit in this audit",
            "source": "H=15 fly-toward payload increment of the candidate assignment. SINR is already inside this number.",
            "observable_now": "yes, by offline shadow, not as a network input",
        },
        {
            "term": "urgency",
            "beta": "free, not fit",
            "source": "q_k / (max q + 1e-8) only. Waiting time, AoI, and deadline are absent.",
            "observable_now": "queue ratio only",
        },
        {
            "term": "travel_cost",
            "gamma": "free, not fit",
            "source": "ETA already stored on the UAV-Buoy edge: distance / step length.",
            "observable_now": "yes",
        },
        {
            "term": "interference_cost",
            "delta": "0 if payload gain is the label; nonzero only as an extra penalty",
            "source": "Active-link interference is already folded into the H=15 payload.",
            "observable_now": "edge exists; value is 0 when no link is active",
        },
        {
            "term": "energy_cost",
            "eta": "0",
            "source": "L3-B does not expose energy. The UAV energy channel is a structural zero.",
            "observable_now": "no",
        },
    ]


def hgnn_input_table() -> list[dict[str, str]]:
    return [
        {"feature": "UAV-to-target distance", "current": "yes, UAV-Buoy edge channel 0", "needed": "keep", "note": "Do not add a second distance."},
        {"feature": "estimated travel time", "current": "yes, ETA on UAV-Buoy edge channel 2", "needed": "keep", "note": "distance / step length, current geometry only."},
        {"feature": "predicted communication gain", "current": "instantaneous hypothetical rate only", "needed": "use as the training label, not a new input feature", "note": "H=15 payload is the utility. Putting it into the input would leak the label."},
        {"feature": "interference graph", "current": "yes, UAV-UAV edge, active links only", "needed": "keep", "note": "Zero before service. Executed interference is already in the payload label."},
        {"feature": "energy consumption", "current": "slot filled with 0", "needed": "do not add", "note": "No environment variable to read."},
        {"feature": "queue", "current": "yes", "needed": "keep", "note": "Current queue."},
        {"feature": "channel", "current": "yes, rate_if_assigned", "needed": "keep", "note": "Current channel, not future channel."},
    ]


def _match_scores(env: L3BEnv) -> np.ndarray:
    info = env.info()
    rate = np.asarray(info["rate_if_assigned"], dtype=np.float64)
    dist = np.asarray(info["distance_uav_buoy"], dtype=np.float64)
    queue = np.asarray(env.queue, dtype=np.float64)
    return queue[None, :] * rate / (dist + 1.0)


def payload_rank_probe(seed: int) -> list[dict]:
    assert_audit_seed(int(seed))
    env = L3BEnv()
    info = env.reset(seed=int(seed))
    while int(env.t) < 15:
        assignment = observable_exclusive_assignment(env)
        info = env.step(fly_toward(env, assignment), assignment)
    _ = info
    base = observable_exclusive_assignment(env)
    scores = _match_scores(env)
    rows = []
    for focal in range(int(env.config.num_uavs)):
        taken = {int(base[uav]) for uav in range(int(env.config.num_uavs)) if uav != focal and int(base[uav]) >= 0}
        legal = [buoy for buoy in range(int(env.config.num_buoys)) if buoy not in taken]
        if len(legal) < 3:
            continue
        payload = []
        match = []
        for buoy in legal:
            trial = np.asarray(base, dtype=np.int64).copy()
            trial[focal] = int(buoy)
            payload.append(shadow_h15_payload(env, trial))
            match.append(float(scores[focal, buoy]))
        payload_a = np.asarray(payload, dtype=np.float64)
        match_a = np.asarray(match, dtype=np.float64)
        separated = float(np.max(payload_a) - np.min(payload_a)) >= 1.0e6 - 1.0
        rows.append(
            {
                "seed": int(seed),
                "focal": int(focal),
                "time": int(env.t),
                "n_candidates": len(legal),
                "spearman_match_vs_payload": spearman(match_a, payload_a),
                "kendall_match_vs_payload": kendall_tau(match_a, payload_a),
                "top1_same": float(int(np.argmax(match_a)) == int(np.argmax(payload_a))),
                "payload_span": float(np.max(payload_a) - np.min(payload_a)),
                "separated": separated,
            }
        )
    if int(env.t) != 15:
        raise RuntimeError("audit did not stop at the t=15 decision")
    return rows


def _mean(rows: list[dict], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def run() -> dict:
    for seed in AUDIT_SEEDS:
        assert_audit_seed(int(seed))
    rows = []
    for seed in AUDIT_SEEDS:
        print(f"payload rank seed={seed}", flush=True)
        rows.extend(payload_rank_probe(int(seed)))
    if not rows:
        raise RuntimeError("no candidate groups")
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "training": False,
        "decide_called": False,
        "joint_training": False,
        "mappo_retrained": False,
        "tau_modified": False,
        "holdout_used": False,
        "objective": objective_record(),
        "utility_interface": feature_specification(),
        "hgnn_inputs": hgnn_input_table(),
        "label_probe": {
            "seeds": list(AUDIT_SEEDS),
            "executor": "frozen fly-toward H=15, not MAPPO",
            "n_groups": len(rows),
            "spearman_match_vs_payload": _mean(rows, "spearman_match_vs_payload"),
            "kendall_match_vs_payload": _mean(rows, "kendall_match_vs_payload"),
            "top1_same": _mean(rows, "top1_same"),
            "mean_payload_span": _mean(rows, "payload_span"),
            "fraction_separated": _mean(rows, "separated"),
        },
        "route": "Option-2",
        "phase5d_started": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase5c_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(summary["route"], flush=True)
    return summary


def _report(summary: dict) -> str:
    obj = summary["objective"]
    probe = summary["label_probe"]
    lines = [
        "# HTW Phase-5C High-Level Utility Redesign Audit",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段没有训练，没有重训 MAPPO，没有调用 `decide()`，没有修改 B2、τ、Gate-1、Candidate-C 或 planner，没有使用 230000–230127，也没有进入联合训练。",
        "",
        "## 1. 当前 high-level 目标",
        "",
        "在线推理读取冻结的 DeepSets order-free 模型，输出联合贪心分配。这一步没有 loss，也没有 payload reward。",
        "",
        "该检查点的训练目标在 `utility_losses`：",
        "",
        f"- 输入：{'; '.join(obj['official_input'])}",
        f"- 标签：{obj['official_label']}",
        f"- loss：`{obj['official_loss']}`。τ = {obj['official_tau_read_only']}，本阶段只读取、不修改。",
        f"- reward：{obj['official_reward']}",
        "",
        "它不直接最大化整段 payload。`L_util` 只让候选集合上的对数概率去贴近已经算好的 15 步 fly-toward payload 分布。候选集合之外的目标不会被这个 loss 看见。解码时也不再计算 payload。",
        "",
        "Phase-5 的 HGNN 用的是另一套目标：标签是 `queue * 当前假设速率 / (distance + 1)` 的排他匹配，loss 是这个单一分配的 set NLL。这是 assignment similarity，不是 payload utility。",
        "",
        "## 2. Trajectory-aware utility 接口",
        "",
        "只定义接口，不写入模型。",
        "",
        "`U(assignment) = α * expected_payload_gain + β * urgency - γ * travel_cost - δ * interference_cost - η * energy_cost`",
        "",
        "系数本阶段不拟合。",
        "",
        "| term | 现在能否计算 | 说明 |",
        "| --- | --- | --- |",
    ]
    for row in summary["utility_interface"]:
        lines.append(f"| {row['term']} | {row['observable_now']} | {row['source']} |")
    lines.extend(
        [
            "",
            "若 `expected_payload_gain` 就是 15 步通信 payload，干扰已经进了这个数，`δ` 应为 0，否则干扰会被减两次。`η` 现在必须为 0，因为没有能耗变量。`urgency` 仍然只有队列比例。",
            "",
            "## 3. HGNN 输入",
            "",
            "没有改 encoder。",
            "",
            "| feature | current | needed |",
            "| --- | --- | --- |",
        ]
    )
    for row in summary["hgnn_inputs"]:
        lines.append(f"| {row['feature']} | {row['current']} | {row['needed']} |")
    lines.extend(
        [
            "",
            "距离、旅行时间和当前信道已经在 UAV–Buoy 边上。不要把 15 步 payload 再拼进输入，否则标签泄漏。干扰边保留。能耗不加。",
            "",
            "## 4. 新的训练标签",
            "",
            "旧标签是一条排他匹配。探针先用这条匹配和 fly-toward 走到 t=15，不调用 `decide()`。然后固定其他无人机，替换一名无人机的未占用浮标，再用冻结 fly-toward 走 15 步，记录 payload 增量。没有调用 MAPPO。",
            "",
            f"匹配分数和 15 步 payload 的 Spearman = {_fmt(probe['spearman_match_vs_payload'])}，Kendall = {_fmt(probe['kendall_match_vs_payload'])}，Top-1 相同比例 = {_fmt(probe['top1_same'])}，平均 payload 极差 = {_fmt(probe['mean_payload_span'])} bit，候选被拉开至少一个服务块的比例 = {_fmt(probe['fraction_separated'])}，组数 = {probe['n_groups']}。",
            "",
            "因此新标签应是每个状态下候选分配的 15 步 payload 排序，而不是匹配是否一致。正式训练时仍然不能把 shadow 的未来轨迹写进网络输入。",
            "",
            "## 5. 路线",
            "",
            "推荐 Option-2：HGNN 保持现有边，把监督从 assignment match 换成 trajectory-aware utility ranking。",
            "",
            "Option-1 只换 encoder。Phase-5 已经做过，HGNN 更像距离/队列匹配，payload 下降。",
            "",
            "Option-3 是高低层反馈对齐。Phase-5B 里 oracle 目标交给当前 MAPPO 仍能产生 payload，低层不是当前瓶颈。联合训练本阶段禁止开始。",
            "",
            "Phase-5C 停在规格。`planner_ready = False`，`deployment_verified = False`。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    run()
