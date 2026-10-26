"""Write the E4 protocol review from committed records."""
from __future__ import annotations

import json

from .protocol import E3_GATE, GATE_BLOCKED, OUTPUT, REPORT
from .report import render_report
from .review import build_review
from .sources import load_review_inputs


def _schema(summary: dict) -> dict:
    candidate = summary["conditional_candidate"]
    return {
        "schema_version": "urgency_reward_protocol_review_v1",
        "e2_gate": summary["e2_gate"],
        "e3_gate": E3_GATE,
        "e3_gate_rewritten": False,
        "standards_loosened": False,
        "candidate_status": candidate["status"],
        "candidate_form": "B",
        "candidate_lambda_u": candidate["lambda_u"],
        "candidate_rho": candidate["rho"],
        "lambda_selected": None,
        "rho_selected": None,
        "lambda_h": "NOT_SELECTED",
        "approved": False,
        "training_authorized": False,
        "reran_environment": False,
        "resampled_tags": False,
        "trained": False,
        "formal_system_modified": False,
        "closed_loop_claim": False,
        "preferred_design": "B",
        "preferred_design_is_performance_claim": False,
        "formula_a": "r_base + lambda_u * urgency_term",
        "formula_b": "r_base + min(lambda_u * urgency_term, rho * payload_term)",
        "payload_term": "served_bits/1e6 for the current step, not cumulative payload_bits",
        "owner_decisions_pending": [
            "accept_stop_or_rollback_if_benchmark_payload_drops",
            "approve_lambda_0_0005_rho_0_001_as_only_first_lowlevel_candidate",
        ],
    }


def main() -> None:
    summary = build_review(load_review_inputs())
    if summary["gate"] == GATE_BLOCKED:
        summary["reason"] = "E2/E3 源文件、哈希或正式奖励字段不能复核"
    else:
        summary["reason"] = "审查完成，但首个低层训练候选尚未获准"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "schema.json").write_text(json.dumps(_schema(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(summary), encoding="utf-8")
    print(summary["gate"], summary["e3_gate"], summary["conditional_candidate"]["status"], len(summary["problems"]))


if __name__ == "__main__":
    main()
