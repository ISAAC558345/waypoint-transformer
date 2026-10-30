"""Write coverage artifacts and compare the historical unittest suite."""
from __future__ import annotations

import json
import sys

from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.run import parse_unittest

from .artifacts import write_json
from .prepare import prepare
from .protocol import GATE_BLOCKED, OUTPUT, ROOT
from .replay import replay
from .report import write_report


def _normalize(detail: str) -> str:
    text = str(detail)
    for root in (
        str(ROOT),
        str(ROOT).replace("\\", "/"),
        r"E:\srdp\wt_e18_baseline",
        "E:/srdp/wt_e18_baseline",
    ):
        text = text.replace(root, "<ROOT>")
    return text


def compare_logs(baseline_text: str, current_text: str) -> dict:
    baseline = parse_unittest(baseline_text)
    current = parse_unittest(current_text)
    baseline_ids = {(item["test_id"], item["kind"]): item for item in baseline["failures"]}
    new_items = []
    historical = []
    for item in current["failures"]:
        previous = baseline_ids.get((item["test_id"], item["kind"]))
        same = previous is not None and previous["exception"] == item["exception"] and _normalize(previous["detail"]) == _normalize(item["detail"])
        if same:
            historical.append(item)
        else:
            new_items.append(item)
    return {
        "baseline_ran": baseline["ran"],
        "repository_ran": current["ran"],
        "baseline_failure_count": len(baseline["failures"]),
        "repository_failure_count": len(current["failures"]),
        "historical_count": len(historical),
        "new_failure_count": len(new_items),
        "new_failures": new_items,
        "comparison_root": "E:/srdp/wt_e18_baseline",
    }


def attach_tests(summary: dict | None = None) -> dict:
    if summary is None:
        summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    comparison = compare_logs(
        (OUTPUT / "baseline_tests.log").read_text(encoding="utf-8", errors="replace"),
        (OUTPUT / "repository_tests.log").read_text(encoding="utf-8", errors="replace"),
    )
    related = (OUTPUT / "related_tests.log").read_text(encoding="utf-8", errors="replace")
    comparison["related_ok"] = related.strip().endswith("OK")
    summary["test_comparison"] = comparison
    if comparison["new_failure_count"] != 0 or not comparison["related_ok"]:
        summary["scientific_gate"] = GATE_BLOCKED
        summary["problems"] = list(summary.get("problems") or []) + ["new test failures or related tests did not pass"]
        summary["opportunity_published"] = False
        summary["opportunity_evidence"] = None
    write_json(OUTPUT / "test_comparison.json", comparison)
    write_json(OUTPUT / "summary.json", summary)
    write_report(summary)
    return summary


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: prepare | replay | attach-tests")
    command = sys.argv[1]
    if command == "prepare":
        prepare()
    elif command == "replay":
        replay()
    elif command == "attach-tests":
        attach_tests()
    else:
        raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main()
