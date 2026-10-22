"""Write the Phase-2 oracle hierarchical controller report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .experiment import run_experiment

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "oracle_controller"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase2_oracle_controller_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase2_oracle_controller_report.md"


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items() if key != "agent"}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def write_report(payload: Mapping[str, Any]) -> None:
    state = payload["results"]["state_only"]
    htw = payload["results"]["htw_oracle"]
    payload_gain = float(htw["mean_payload_bits"]) - float(state["mean_payload_bits"])
    track_drop = float(state["mean_waypoint_tracking_error"]) - float(htw["mean_waypoint_tracking_error"])
    rho_state = state["action_waypoint"].get("mean_rho")
    rho_htw = htw["action_waypoint"].get("mean_rho")
    rho_gain = (float(rho_htw) - float(rho_state)) if rho_state is not None and rho_htw is not None else float("nan")
    payload_up = payload_gain > 1.0
    track_down = track_drop > 1.0e-3
    rho_up = bool(np.isfinite(rho_gain) and rho_gain > 0.05)
    continue_htw = track_down and (rho_up or payload_up)
    lines = [
        "# HTW Phase-2 Oracle Hierarchical Controller Report",
        "",
        "最小原型：High-level oracle planner → TW_new → Low-level PPO（含 tracking 辅助目标）。",
        "未改 HGNN、Transformer、PPO GAE/clip/optimizer/loss、环境 reward；未开 Round。",
        "",
        f"- horizon H={payload['horizon']}",
        f"- tracking λ={payload['tracking_lambda']}",
        f"- mini updates={payload['updates']}（与 Phase-1 相同的 smoke PPO 预算）",
        "",
        "## 结构",
        "",
        "High-level：`independent_queue_rate` oracle shadow window，输出 `(goal, waypoint, horizon)`。",
        "Low-level actor 输入：`[state, TW_new]`（UAV dim 16）。输出：movement action。",
        "Controller 内部：`r_total = r_comm - λ * mean_i ||p_i - W_i||`（归一化坐标）。`r_comm` 仍来自未改动的 `env.step`。",
        "",
        "## 结果",
        "",
        "| 方法 | payload | env return | residual queue | tracking error | action-W cosine | TW in actor |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| State-only PPO | {state['mean_payload_bits']:.3f} | {state['mean_return']:.6f} | "
            f"{state['mean_residual_queue_bits']:.3f} | {state['mean_waypoint_tracking_error']:.6f} | "
            f"{rho_state} | {state['tw_enters_actor']} |"
        ),
        (
            f"| HTW-Oracle-PPO | {htw['mean_payload_bits']:.3f} | {htw['mean_return']:.6f} | "
            f"{htw['mean_residual_queue_bits']:.3f} | {htw['mean_waypoint_tracking_error']:.6f} | "
            f"{rho_htw} | {htw['tw_enters_actor']} |"
        ),
        "",
        "## 问题",
        "",
        f"1. Oracle TW + low-level controller 是否提升收益？ **{'是' if payload_up else '否'}**（Δpayload={payload_gain:.3f}）。",
        f"2. waypoint tracking error 是否下降？ **{'是' if track_down else '否'}**（Δerror={track_drop:.6f}）。",
        f"3. action relevance 是否提高？ **{'是' if rho_up else '否'}**（Δρ={rho_gain:.6f}）。",
        f"4. HTW 结构是否值得继续？ **{'值得继续' if continue_htw else '尚不充分'}**。",
        "",
        "判定规则：跟踪误差下降，且（ρ 提高或 payload 提高）则认为层次结构开始闭合，值得在不改 HGNN 的前提下继续；否则仍不能证明 hierarchical control 产生了控制价值。",
        "",
        f"JSON：`{OUTPUT.as_posix()}/summary.json`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = run_experiment()
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    print(json.dumps(_plain(result["results"]), indent=2))
