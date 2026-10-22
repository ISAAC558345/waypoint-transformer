"""Read-only comparison of the low-level training tasks and scheme C."""
from __future__ import annotations

import json

from .audit import run_audit
from .protocol import OUTPUT, assert_phase_output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = run_audit()
    rows = {"dev": result.pop("dev_rows"), "train_probe": result.pop("train_rows")}
    assert_phase_output(OUTPUT / "summary.json").write_text(json.dumps(result), encoding="utf-8")
    assert_phase_output(OUTPUT / "scenario_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    print(json.dumps({"gates": result["gates"], "triggered": result["verdict"]["triggered"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
