"""CLI for frozen Round-6 closed-loop evaluation."""
from __future__ import annotations
import argparse
from pathlib import Path
from wt.round6.closed_loop import ROOT, run_round6_evaluation, run_reproducibility_audit
from wt.round6.figures import generate_round6_figures

def main() -> int:
    p=argparse.ArgumentParser(); mode=p.add_mutually_exclusive_group(required=True); mode.add_argument('--smoke',action='store_true'); mode.add_argument('--formal',action='store_true'); mode.add_argument('--reproducibility-audit',action='store_true'); p.add_argument('--run-id',required=True); args=p.parse_args()
    out=ROOT/'round_06'/'outputs'/args.run_id
    if args.reproducibility_audit: print(run_reproducibility_audit(out)); return 0
    kind='smoke' if args.smoke else 'formal'; run_round6_evaluation(mode=kind,output_dir=out,config_path=ROOT/'round_06'/'configs'/'closed_loop_eval_v1.json'); generate_round6_figures(out,ROOT/'round_06'/'figures'/args.run_id); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
