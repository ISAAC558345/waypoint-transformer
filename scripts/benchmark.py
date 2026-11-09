"""Read saved comparison summaries. This does not rerun validation or reserved episodes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wt.common.resources import resolve_logical


SUMMARIES = (
    ("validation 322048-322063", "results/phase6iab_planner_stability/summary.json"),
    ("reserved test 322064-322079", "results/phase6iac_independent_holdout/summary.json"),
)


def main() -> int:
    for label, logical in SUMMARIES:
        path = resolve_logical(logical)
        if not path.is_file():
            print(f"{label}: saved summary is absent ({logical})")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        paired = payload.get("paired")
        if not isinstance(paired, dict) or "mean" not in paired:
            print(f"{label}: saved summary has no paired mean")
            continue
        print(
            f"{label}: mean={paired['mean']:.3f} bit "
            f"ci=[{paired.get('ci_low', float('nan')):.3f}, {paired.get('ci_high', float('nan')):.3f}] "
            f"positive={payload.get('positive_scenarios')} "
            f"tie={payload.get('tie_scenarios')} "
            f"negative={payload.get('negative_scenarios')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
