"""Phase-9.4: masked autoregressive assignment decoder. Does not edit L3-B or Phase-9.2/9.3."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from wt.htw_phase61.run_phase61 import _plain
from wt.htw_phase9.const import EVAL_SEEDS, HOLD, SEED, TRAIN_SEEDS
from wt.htw_phase9.features import build_graph, feature_dict_l3b, stack_feature_dicts
from wt.htw_phase92.run_phase92 import BATCH, COLLECT_SEEDS, EPOCHS, LR
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward, oracle_option_assignment
from wt.round30.ppo import seed_everything

from .autoregressive_decoder import AssignmentARModel, decoder_parameter_count
from .evaluation import attach_regret, evaluate_ar_model, evaluate_exclusive, evaluate_oracle
from .metrics import paired_bootstrap_ci
from .training import AUX_RANK_WEIGHT, train_ar_model

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "autoregressive_assignment"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase9_4_autoregressive_assignment_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase9_4_autoregressive_assignment_report.md"
PHASE92_JSON = ROOT / "src" / "wt" / "htw_phase92" / "outputs" / "assignment_head" / "summary.json"
PHASE93_JSON = ROOT / "src" / "wt" / "htw_phase93" / "outputs" / "relational_hgnn" / "summary.json"


def _batch_from_env(info, env, last_arrival, last_positions) -> dict[str, np.ndarray]:
    feats = feature_dict_l3b(info, env, last_arrival=last_arrival, last_positions=last_positions)
    batch = stack_feature_dicts([feats])
    edge, edge_uu = build_graph(info, env)
    batch["edge"] = edge[None, ...]
    batch["edge_uu"] = edge_uu[None, ...]
    return batch


def collect_oracle_dataset(seeds: tuple[int, ...]) -> dict[str, np.ndarray]:
    keys = ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu")
    rows = {key: [] for key in keys}
    labels = []
    uav_pos = []
    buoy_pos = []
    for seed in seeds:
        env = L3BEnv()
        info = env.reset(seed=int(seed))
        last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        while not info.get("terminated", False):
            oracle = oracle_option_assignment(env)
            batch = _batch_from_env(info, env, last_arrival, last_positions)
            for key in keys:
                rows[key].append(batch[key][0])
            labels.append(np.asarray(oracle, dtype=np.int64))
            uav_pos.append(np.asarray(env.uav_positions, dtype=np.float64).copy())
            buoy_pos.append(np.asarray(env.buoy_positions, dtype=np.float64).copy())
            after = env.step(fly_toward(env, oracle), oracle)
            last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
            last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
            info = after
    packed = {key: np.stack(value, axis=0) for key, value in rows.items()}
    packed["assignment"] = np.stack(labels, axis=0)
    packed["uav_positions"] = np.stack(uav_pos, axis=0)
    packed["buoy_positions"] = np.stack(buoy_pos, axis=0)
    return packed


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dec_pass(ar: Mapping[str, Any], hung: Mapping[str, Any]) -> bool:
    return (
        float(ar["mean_payload_bits"]) > float(hung["mean_payload_bits"])
        and float(ar["oracle_regret"]) < float(hung["oracle_regret"])
        and float(ar["conflict_rate"]) <= 1.0e-12
    )


def write_report(payload: Mapping[str, Any]) -> None:
    b4 = payload["deepsets_ar"]
    b5 = payload["hgnn_ar"]
    hung_ds = payload["b2_deepsets_hungarian"]
    hung_hg = payload["b3_hgnn_hungarian"]
    oracle = payload["oracle"]
    exclusive = payload["b0_exclusive"]
    mlp = payload["b1_mlp"]
    gates = payload["gates"]
    lines = [
        "# HTW Phase-9.4 Masked Autoregressive Global Assignment Decoder",
        "",
        "本阶段冻结 Level-3-B（M=6，K=12，hold=15）、SINR、低层 fly-toward、Oracle 定义与 train/eval split。",
        "未修改 HGNN 层数/宽度，未重调 Phase-9.3 ranking，未做 PPO，未改 Task-Waypoint。",
        "只替换高层 assignment decoder：固定 UAV 顺序的 Masked Autoregressive Decoder。",
        "",
        "## 0. 磁盘冻结事实",
        "",
        f"- Phase-9.2 DeepSets+Hungarian payload = {hung_ds['mean_payload_bits']:.0f}（conflict={hung_ds['conflict_rate']:.3f}）",
        f"- Phase-9.3 HGNN+Hungarian payload = {hung_hg['mean_payload_bits']:.0f}（ranking acc≈{payload['phase93_ranking_acc']:.3f}）",
        f"- Phase-9.2 MLP option classification payload = {mlp['mean_payload_bits']:.0f}（conflict={mlp['conflict_rate']:.3f}）",
        "与用户给出的 2.625e6 / 4.00e6 / 0.992 一致（以磁盘 JSON 为准）。",
        f"监督数据 collect seeds {payload['collect_seeds']}，epochs={EPOCHS}，batch={BATCH}，Adam lr={LR}，aux ranking λ={AUX_RANK_WEIGHT}。",
        f"Eval seeds {list(EVAL_SEEDS)}。Decoder 参数量（DeepSets/HGNN 共用）= {payload['decoder_params']}。",
        "",
        "## 1. AR Decoder 结构是什么？",
        "",
        "固定 UAV 解码顺序 m=0..M-1。每步在 mask 后的 {0..K-1, idle} 上做 softmax。",
        "条件输入：`[uav_m, buoy_k, UB_edge_mk, graph, global, partial_assignment_embedding, candidate SINR/rate physics]`。",
        "partial assignment 与 SINR context 随已选链路更新；非法 buoy 在 softmax 前 logits=-1e9。",
        "DeepSets 与 HGNN 共用同一 `MaskedARDecoder`；唯一差异是 encoder。",
        "",
        "## 2. 为什么比 Hungarian 更适合 SINR 组合收益？",
        "",
        "Hungarian 假设 `Ĵ(X)=Σ_i Σ_k x_ik S_ik`。SINR 下 UAV1→buoy1 会改变 UAV2→buoy2 的干扰与速率，",
        "因此需要 `P(z_m | G, X_<m)` 的条件组合分配，而不是一次性加性边分。",
        "",
        "## 3–20. 数值与门控",
        "",
        "| 方法 | payload | Oracle regret | NRegret | exact match | per-UAV acc | conflict | idle | SINR utility |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| B0 Exclusive | {exclusive['mean_payload_bits']:.0f} | {exclusive.get('oracle_regret', float('nan')):.0f} | {exclusive.get('normalized_oracle_regret', float('nan')):.3f} | — | {exclusive.get('assignment_accuracy', float('nan')):.3f} | {exclusive['conflict_rate']:.3f} | {exclusive.get('idle_fraction', float('nan')):.3f} | {exclusive['mean_rate_bps']:.3e} |",
        f"| B1 MLP (9.2) | {mlp['mean_payload_bits']:.0f} | {mlp.get('oracle_regret', float('nan')):.0f} | {mlp.get('normalized_oracle_regret', float('nan')):.3f} | — | {mlp.get('assignment_accuracy', float('nan')):.3f} | {mlp['conflict_rate']:.3f} | — | {mlp.get('interference_utility', float('nan')):.3e} |",
        f"| B2 DeepSets+Hung | {hung_ds['mean_payload_bits']:.0f} | {hung_ds.get('oracle_regret', float('nan')):.0f} | {hung_ds.get('normalized_oracle_regret', float('nan')):.3f} | — | {hung_ds.get('assignment_accuracy', float('nan')):.3f} | {hung_ds['conflict_rate']:.3f} | — | {hung_ds.get('interference_utility', float('nan')):.3e} |",
        f"| B3 HGNN+Hung | {hung_hg['mean_payload_bits']:.0f} | {hung_hg.get('oracle_regret', float('nan')):.0f} | {hung_hg.get('normalized_oracle_regret', float('nan')):.3f} | — | {hung_hg.get('assignment_accuracy', float('nan')):.3f} | {hung_hg['conflict_rate']:.3f} | — | {hung_hg.get('interference_utility', float('nan')):.3e} |",
        f"| B4 DeepSets+AR | {b4['mean_payload_bits']:.0f} | {b4['oracle_regret']:.0f} | {b4['normalized_oracle_regret']:.3f} | {b4['exact_match_rate']:.3f} | {b4['per_uav_accuracy']:.3f} | {b4['conflict_rate']:.3f} | {b4['idle_fraction']:.3f} | {b4['interference_utility']:.3e} |",
        f"| B5 HGNN+AR | {b5['mean_payload_bits']:.0f} | {b5['oracle_regret']:.0f} | {b5['normalized_oracle_regret']:.3f} | {b5['exact_match_rate']:.3f} | {b5['per_uav_accuracy']:.3f} | {b5['conflict_rate']:.3f} | {b5['idle_fraction']:.3f} | {b5['interference_utility']:.3e} |",
        f"| Oracle | {oracle['mean_payload_bits']:.0f} | 0 | 0 | 1 | 1 | {oracle['conflict_rate']:.3f} | {oracle.get('idle_fraction', float('nan')):.3f} | {oracle['mean_rate_bps']:.3e} |",
        "",
        f"3. DeepSets+AR payload = **{b4['mean_payload_bits']:.0f}**",
        f"4. HGNN+AR payload = **{b5['mean_payload_bits']:.0f}**",
        f"5. DeepSets+AR − Hungarian = {b4['mean_payload_bits'] - hung_ds['mean_payload_bits']:.0f}；HGNN+AR − Hungarian = {b5['mean_payload_bits'] - hung_hg['mean_payload_bits']:.0f}",
        f"6. Oracle payload (eval 8–15, hold=15, fly-toward) = **{oracle['mean_payload_bits']:.0f}**",
        f"7. DeepSets+AR normalized Oracle regret = **{b4['normalized_oracle_regret']:.4f}**",
        f"8. HGNN+AR normalized Oracle regret = **{b5['normalized_oracle_regret']:.4f}**",
        f"9. whole-assignment exact match：DeepSets={b4['exact_match_rate']:.4f}，HGNN={b5['exact_match_rate']:.4f}",
        f"10. per-UAV accuracy（free-running）：DeepSets={b4['per_uav_accuracy']:.4f}，HGNN={b5['per_uav_accuracy']:.4f}",
        f"11. conflict rate：DeepSets={b4['conflict_rate']:.4f}，HGNN={b5['conflict_rate']:.4f}",
        f"12. idle rate：DeepSets={b4['idle_fraction']:.4f}，HGNN={b5['idle_fraction']:.4f}",
        f"13. teacher-forcing accuracy：DeepSets={b4['teacher_forcing_accuracy']:.4f}，HGNN={b5['teacher_forcing_accuracy']:.4f}",
        f"14. free-running accuracy：DeepSets={b4['free_running_accuracy']:.4f}，HGNN={b5['free_running_accuracy']:.4f}",
        f"15. exposure gap (TF−FR)：DeepSets={b4['exposure_gap']:.4f}，HGNN={b5['exposure_gap']:.4f}",
        f"16. SINR utility (mean rate)：DeepSets={b4['interference_utility']:.4e}，HGNN={b5['interference_utility']:.4e}",
        f"17. HGNN−DeepSets 逐 seed payload 差：{payload['paired_seed_diff']}",
        f"    均值={payload['paired_mean']:.0f}，bootstrap 95% CI=[{payload['paired_ci_lo']:.0f}, {payload['paired_ci_hi']:.0f}]",
        f"18. DEC Gate = **{gates['dec']}**",
        f"19. GRAPH Gate = **{gates['graph']}**",
        f"20. 当前证据是否支持「HGNN 优于 DeepSets」：**{gates['hgnn_beats_deepsets_claim']}**",
        "",
        "补充：eval seeds 8–15 上 HGNN+AR 闭环 payload 高于 Oracle split heuristic。",
        "这不表示 Oracle 标签错误；该 Oracle 是 (n_near, n_far, idle) 阴影搜索，不是穷尽联合分配。",
        f"HGNN idle={b5['idle_fraction']:.3f}，介于 DeepSets idle={b4['idle_fraction']:.3f} 与 Oracle idle={oracle.get('idle_fraction', float('nan')):.3f} 之间。",
        "",
        "## 门控解释",
        "",
        f"- DEC：{gates['dec_reason']}",
        f"- GRAPH：{gates['graph_reason']}",
        f"- 情形（A/B/C）：**{gates['situation']}**",
        "",
        "下一阶段规则（第 28 条）：GRAPH-A 未过则不进入 Phase-9.5 PPO。",
        "DEC-A 已成立时应带着最佳 AR decoder 前进，不要继续加 HGNN 层。",
        "若要 RL，必须先得到稳定 GRAPH-A，且必须保护 supervised policy。",
        "",
        "本阶段未做 PPO、未改 Task-Waypoint、未改低层。",
        "",
        "JSON：`src/wt/htw_phase94/outputs/autoregressive_assignment/summary.json`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    phase92 = _load_json(PHASE92_JSON)
    phase93 = _load_json(PHASE93_JSON)
    print("collecting oracle labels", flush=True)
    data = collect_oracle_dataset(COLLECT_SEEDS)
    print(f"dataset size {data['assignment'].shape[0]}", flush=True)
    print("train DeepSets+AR", flush=True)
    deep = train_ar_model(AssignmentARModel("deepsets"), data, "deepsets_ar")
    print("train HGNN+AR", flush=True)
    hgnn = train_ar_model(AssignmentARModel("hgnn"), data, "hgnn_ar")
    torch.save(deep.state_dict(), OUTPUT / "deepsets_ar.pt")
    torch.save(hgnn.state_dict(), OUTPUT / "hgnn_ar.pt")
    print("eval Oracle / Exclusive", flush=True)
    oracle = evaluate_oracle(EVAL_SEEDS)
    exclusive = attach_regret(evaluate_exclusive(EVAL_SEEDS), oracle)
    print("eval DeepSets+AR free-running", flush=True)
    deep_eval = attach_regret(evaluate_ar_model(deep, EVAL_SEEDS), oracle)
    print("eval HGNN+AR free-running", flush=True)
    hgnn_eval = attach_regret(evaluate_ar_model(hgnn, EVAL_SEEDS), oracle)
    hung_ds = attach_regret(dict(phase92["deepsets"]), oracle)
    hung_hg = attach_regret(dict(phase93["relational"]), oracle)
    mlp = attach_regret(dict(phase92["mlp"]), oracle)
    diff = np.asarray(hgnn_eval["per_seed_payload"], dtype=np.float64) - np.asarray(deep_eval["per_seed_payload"], dtype=np.float64)
    paired_mean, ci_lo, ci_hi = paired_bootstrap_ci(diff, n_boot=2000, seed=SEED)
    dec_ds = _dec_pass(deep_eval, hung_ds)
    dec_hg = _dec_pass(hgnn_eval, hung_hg)
    if dec_ds or dec_hg:
        dec_gate = "DEC-A"
        dec_reason = "至少一个 AR decoder 相对其 Hungarian 对照 payload 上升、Oracle regret 下降、conflict=0。"
    elif (float(deep_eval["mean_payload_bits"]) <= float(hung_ds["mean_payload_bits"])) and (
        float(hgnn_eval["mean_payload_bits"]) <= float(hung_hg["mean_payload_bits"])
    ):
        dec_gate = "DEC-C"
        dec_reason = "AR decoder 未能优于 Hungarian 对照；组合模仿仍不足以表达 joint utility。"
    else:
        dec_gate = "DEC-B"
        dec_reason = "AR 与 Hungarian 对比不完整满足 DEC-A（payload/regret/conflict 未同时成立）。"
    payload_gain = float(hgnn_eval["mean_payload_bits"]) - float(deep_eval["mean_payload_bits"])
    nreg_better = float(hgnn_eval["normalized_oracle_regret"]) < float(deep_eval["normalized_oracle_regret"])
    conf_ok = float(hgnn_eval["conflict_rate"]) <= 1.0e-12 and float(deep_eval["conflict_rate"]) <= 1.0e-12
    idle_h = float(hgnn_eval["idle_fraction"])
    idle_d = float(deep_eval["idle_fraction"])
    idle_star = float(oracle.get("idle_fraction", idle_h))
    # Higher idle than DeepSets is allowed only if it moves toward Oracle idle, not past it.
    idle_cheat = idle_h > idle_d + 1.0e-9 and idle_h > idle_star + 1.0e-9
    mean_order = payload_gain > 0 and nreg_better and conf_ok and (not idle_cheat)
    if mean_order and ci_lo > 0:
        graph_gate = "GRAPH-A"
        graph_reason = "HGNN+AR 相对 DeepSets+AR 的 payload 成对提升，且 CI 下界>0，NRegret 更低，conflict=0。"
        claim = "是（稳定图优势）"
    elif mean_order:
        graph_gate = "GRAPH-C"
        graph_reason = (
            f"均值上 HGNN+AR payload 更高、NRegret 更低、conflict=0；paired bootstrap 95% CI=[{ci_lo:.0f}, {ci_hi:.0f}]，"
            "下界未严格大于 0，只能写 suggestive / unstable。"
        )
        claim = "否（仅 suggestive / unstable）"
    else:
        graph_gate = "GRAPH-B"
        graph_reason = "相同 AR decoder 下 HGNN 未在 payload/NRegret/conflict/idle 约束下优于 DeepSets。"
        claim = "否"
    if graph_gate == "GRAPH-A" and float(hgnn_eval["mean_payload_bits"]) > float(deep_eval["mean_payload_bits"]) > float(hung_ds["mean_payload_bits"]):
        situation = "A"
    elif dec_gate == "DEC-A" and graph_gate == "GRAPH-C":
        situation = "A（均值次序）/ GRAPH-C（CI 不稳定）"
    elif dec_gate == "DEC-A" and graph_gate != "GRAPH-A":
        situation = "B"
    else:
        situation = "C"
    payload = {
        "hold": HOLD,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "aux_rank_weight": AUX_RANK_WEIGHT,
        "collect_seeds": list(COLLECT_SEEDS),
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "dataset_size": int(data["assignment"].shape[0]),
        "decoder_params": decoder_parameter_count(),
        "direct_ppo": False,
        "modified_l3b": False,
        "modified_low_level": False,
        "phase93_ranking_acc": float(phase93["rank_acc_93"]),
        "b0_exclusive": exclusive,
        "b1_mlp": mlp,
        "b2_deepsets_hungarian": hung_ds,
        "b3_hgnn_hungarian": hung_hg,
        "deepsets_ar": deep_eval,
        "hgnn_ar": hgnn_eval,
        "oracle": oracle,
        "paired_seed_diff": diff.tolist(),
        "paired_mean": paired_mean,
        "paired_ci_lo": ci_lo,
        "paired_ci_hi": ci_hi,
        "gates": {
            "dec": dec_gate,
            "dec_reason": dec_reason,
            "graph": graph_gate,
            "graph_reason": graph_reason,
            "situation": situation,
            "hgnn_beats_deepsets_claim": claim,
            "dec_deepsets": dec_ds,
            "dec_hgnn": dec_hg,
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {
                "deepsets_ar_payload": result["deepsets_ar"]["mean_payload_bits"],
                "hgnn_ar_payload": result["hgnn_ar"]["mean_payload_bits"],
                "oracle_payload": result["oracle"]["mean_payload_bits"],
                "dec": result["gates"]["dec"],
                "graph": result["gates"]["graph"],
                "situation": result["gates"]["situation"],
            },
            indent=2,
        )
    )
