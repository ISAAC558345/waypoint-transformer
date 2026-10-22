"""Phase-9.3 relational HGNN assignment ranking. No L3-B / PPO / hold edits."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from wt.htw_phase61.run_phase61 import _plain
from wt.htw_phase9.const import EVAL_SEEDS, HOLD, SEED
from wt.htw_phase92.models import HGNNAssignmentHead, IDLE_INDEX
from wt.htw_phase92.run_phase92 import COLLECT_SEEDS, EPOCHS, collect_oracle_dataset, evaluate_model
from wt.round30.ppo import seed_everything

from .losses import hybrid_loss, ranking_loss
from .models import RelationalHGNNAssignmentHead

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "relational_hgnn"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase9_3_relational_HGNN_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase9_3_relational_HGNN_report.md"
BATCH = 32
LR = 1.0e-3
PHASE92_JSON = Path(__file__).resolve().parents[1] / "htw_phase92" / "outputs" / "assignment_head" / "summary.json"


def _tensors(data: dict[str, np.ndarray], index: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    batch = {}
    for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
        dtype = torch.bool if key.endswith("mask") else torch.float32
        batch[key] = torch.as_tensor(data[key][index], dtype=dtype, device=device)
    batch["assignment"] = torch.as_tensor(data["assignment"][index], dtype=torch.int64, device=device)
    return batch


def train_relational(model: torch.nn.Module, data: dict[str, np.ndarray], name: str, *, ranking: bool) -> tuple[torch.nn.Module, dict[str, float]]:
    seed_everything(SEED)
    device = "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    count = int(data["assignment"].shape[0])
    last_acc = float("nan")
    model.train()
    for epoch in range(EPOCHS):
        order = np.random.permutation(count)
        losses, ranks = [], []
        for start in range(0, count, BATCH):
            index = order[start : start + BATCH]
            batch = _tensors(data, index, device)
            scores, idle = model(
                batch["uav"],
                batch["buoy"],
                batch["uav_mask"],
                batch["buoy_mask"],
                batch["global"],
                batch["edge"],
                batch["edge_uu"],
            )
            if ranking:
                loss, _ce, acc = hybrid_loss(scores, idle, batch["assignment"], batch["uav_mask"], batch["buoy_mask"])
            else:
                from wt.htw_phase92.run_phase92 import supervised_loss

                loss = supervised_loss(
                    model,
                    batch,
                    torch.where(batch["assignment"] >= 0, batch["assignment"], torch.full_like(batch["assignment"], IDLE_INDEX)),
                )
                _rank, acc = ranking_loss(scores, idle, batch["assignment"], batch["uav_mask"], batch["buoy_mask"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            ranks.append(float(acc.detach().cpu()))
        last_acc = float(np.mean(ranks))
        print(f"{name} epoch {epoch + 1}/{EPOCHS} loss={float(np.mean(losses)):.4f} rank_acc={last_acc:.3f}", flush=True)
    return model, {"train_ranking_accuracy": last_acc}


@torch.no_grad()
def dataset_ranking_accuracy(model: torch.nn.Module, data: dict[str, np.ndarray]) -> float:
    model.eval()
    device = "cpu"
    accs = []
    count = int(data["assignment"].shape[0])
    for start in range(0, count, BATCH):
        index = np.arange(start, min(start + BATCH, count))
        batch = _tensors(data, index, device)
        scores, idle = model(
            batch["uav"],
            batch["buoy"],
            batch["uav_mask"],
            batch["buoy_mask"],
            batch["global"],
            batch["edge"],
            batch["edge_uu"],
        )
        _loss, acc = ranking_loss(scores, idle, batch["assignment"], batch["uav_mask"], batch["buoy_mask"])
        accs.append(float(acc.detach().cpu()))
    return float(np.mean(accs))


def write_report(payload: Mapping[str, Any]) -> None:
    old = payload["phase92_hgnn"]
    deep = payload["phase92_deepsets"]
    rel = payload["relational"]
    beats = float(rel["mean_payload_bits"]) > float(deep["mean_payload_bits"]) + 1.0
    lines = [
        "# HTW Phase-9.3 Relational HGNN Assignment Ranking Report",
        "",
        "改进 9.2：HGNN **编码器不变**，边分改为 relation-aware，损失加入 pairwise ranking。",
        "未改 L3-B、SINR、低层 PPO、hold=15。未直接 PPO。Hungarian matching 仍用于解码。",
        "",
        "Relation-aware score 输入：UAV embedding、Buoy embedding、UAV–UAV interference embedding、global graph embedding、原始 global。",
        "Ranking：`-log sigmoid(score_pos - score_neg)`，含 UAV 内正负边、idle、以及同 buoy 上 UAV 间竞争。CE 权重 0.25。",
        "",
        "## 结果（eval seeds 8–15）",
        "",
        "| 方法 | payload | assignment accuracy | conflict | SINR utility | ranking accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Phase-9.2 HGNN edge score | {old['mean_payload_bits']:.0f} | {old['assignment_accuracy']:.3f} | {old['conflict_rate']:.3f} | {old['interference_utility']:.3e} | {payload['rank_acc_92']:.3f} |",
        f"| Phase-9.2 DeepSets + Hungarian | {deep['mean_payload_bits']:.0f} | {deep['assignment_accuracy']:.3f} | {deep['conflict_rate']:.3f} | {deep['interference_utility']:.3e} | — |",
        f"| Phase-9.3 relational HGNN | {rel['mean_payload_bits']:.0f} | {rel['assignment_accuracy']:.3f} | {rel['conflict_rate']:.3f} | {rel['interference_utility']:.3e} | {payload['rank_acc_93']:.3f} |",
        "",
        f"9.3 − 9.2 HGNN payload = {rel['mean_payload_bits'] - old['mean_payload_bits']:.0f}。",
        f"9.3 − DeepSets payload = {rel['mean_payload_bits'] - deep['mean_payload_bits']:.0f}。",
        "",
        f"**HGNN payload > DeepSets：{'成立' if beats else '不成立'}**",
        "",
        "JSON：`src/wt/htw_phase93/outputs/relational_hgnn/summary.json`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(PHASE92_JSON.read_text(encoding="utf-8"))
    seed_everything(SEED)
    print("collecting oracle labels", flush=True)
    data = collect_oracle_dataset(COLLECT_SEEDS)
    print(f"dataset size {data['assignment'].shape[0]}", flush=True)
    print("retrain 9.2 HGNN CE (same data, ranking metric)", flush=True)
    hgnn92, _meta92 = train_relational(HGNNAssignmentHead(), data, "hgnn92_ce", ranking=False)
    print("train 9.3 relational HGNN + ranking", flush=True)
    hgnn93, meta93 = train_relational(RelationalHGNNAssignmentHead(), data, "hgnn93_rank", ranking=True)
    rank92 = dataset_ranking_accuracy(hgnn92, data)
    rank93 = dataset_ranking_accuracy(hgnn93, data)
    print("eval 9.2 HGNN", flush=True)
    eval92 = evaluate_model(hgnn92, "hgnn", EVAL_SEEDS)
    print("eval 9.3 relational", flush=True)
    eval93 = evaluate_model(hgnn93, "hgnn", EVAL_SEEDS)
    eval93["train_ranking_accuracy"] = meta93["train_ranking_accuracy"]
    payload = {
        "hold": HOLD,
        "epochs": EPOCHS,
        "direct_ppo": False,
        "phase92_deepsets": prior["deepsets"],
        "phase92_hgnn_locked": prior["hgnn"],
        "phase92_hgnn": eval92,
        "relational": eval93,
        "rank_acc_92": rank92,
        "rank_acc_93": rank93,
        "hgnn_beats_deepsets": float(eval93["mean_payload_bits"]) > float(prior["deepsets"]["mean_payload_bits"]),
        "modified_l3b": False,
        "modified_low_level_ppo": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {
                "hgnn92_payload": result["phase92_hgnn"]["mean_payload_bits"],
                "hgnn93_payload": result["relational"]["mean_payload_bits"],
                "deepsets_payload": result["phase92_deepsets"]["mean_payload_bits"],
                "rank_acc_92": result["rank_acc_92"],
                "rank_acc_93": result["rank_acc_93"],
                "hgnn_beats_deepsets": result["hgnn_beats_deepsets"],
            },
            indent=2,
        )
    )
