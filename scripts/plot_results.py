"""Plot saved tables when they exist. Does not rerun an evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    csv_path = ROOT / "src" / "wt" / "htw_phase106" / "outputs" / "untouched" / "phase106_final_results.csv"
    if csv_path.is_file():
        from plot_final_results import main as plot_saved

        plot_saved()
        return
    print("No Phase-10.6 table was found.")
    print("Closed-loop summaries:")
    print("results/phase6iab_planner_stability/summary.json")
    print("results/phase6iac_independent_holdout/summary.json")


if __name__ == "__main__":
    main()
