"""Run Round-7's non-learning audit and dataset-v2 construction."""
from __future__ import annotations
import argparse
from pathlib import Path
from wt.round7.round7_runner import run_round7
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true");p.add_argument("--run-id",default=None);p.add_argument("--output-dir",default=None);p.add_argument("--figures-dir",default=None);a=p.parse_args();root=Path(__file__).resolve().parents[1];rid=a.run_id or ("round07_smoke_20260809_1700" if a.smoke else "round07_formal_20260809_1730");r=run_round7(output_dir=Path(a.output_dir) if a.output_dir else root/"round_07"/"outputs"/rid,figures_dir=Path(a.figures_dir) if a.figures_dir else root/"round_07"/"figures"/rid,run_id=rid,smoke=a.smoke);print(f"episodes: {len(r['episodes'])}");print(f"retraining_data_ready: {r['readiness']['retraining_data_ready']}");print("training_performed: false");return 0
if __name__=="__main__":raise SystemExit(main())
