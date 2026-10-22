"""Round-16 command-line runner."""
from __future__ import annotations
import argparse,json
from datetime import datetime
from pathlib import Path
from .audit import run_audit
from .reporting import create_figures
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("."));p.add_argument("--output",type=Path);p.add_argument("--figures",type=Path);a=p.parse_args();root=a.root.resolve();name=f"round16_dynamic_planning_audit_{datetime.now():%Y%m%d_%H%M}";out=(a.output or root/"round_16/outputs"/name).resolve();fig=(a.figures or root/"round_16/figures"/out.name).resolve();run=run_audit(root,out);run["figures"]=create_figures(out,fig);(out/"run_manifest.json").write_text(json.dumps(run,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8");print(json.dumps({"output":str(out),"figures":str(fig),"gate":run["gate"]},ensure_ascii=False))
if __name__=="__main__":main()
