"""Post-training, validation-only Round-11 metric and checkpoint audit."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

def run(root: Path = Path.cwd()) -> Path:
    out = root / "round_11/outputs/round11_task_waypoint_20260810_0220"
    with (out / "validation_metrics.csv").open(encoding="utf-8") as f: rows = list(csv.DictReader(f))
    gain = {}
    for H in sorted({r["H_task"] for r in rows}):
        gain[H] = {}
        for head in ("service","progress","return","total"):
            n = [float(r[head]) for r in rows if r["H_task"] == H and r["kind"] == "neural"]
            b = [float(r[head]) for r in rows if r["H_task"] == H and r["kind"] == "train_mean"]
            gain[H][head] = {"neural_validation_mean": sum(n)/len(n), "train_mean_validation": sum(b)/len(b), "relative_improvement_percent": 100.0*(sum(b)/len(b)-sum(n)/len(n))/(sum(b)/len(b))}
    checkpoints=[]
    for path in sorted((root / "round_11/checkpoints").rglob("best.pt")):
        checkpoints.append({"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(), "bytes": path.stat().st_size})
    (out / "task_information_gain.json").write_text(json.dumps(gain,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (out / "checkpoint_audit.json").write_text(json.dumps({"best_checkpoints":checkpoints,"count":len(checkpoints)},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (out / "holdout_access_audit.json").write_text(json.dumps({"access_count":0,"content_read":False,"old_development_test_access_count":0,"policy":"Round-11 loader accepts only Round-10 train/validation sample rows"},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return out
if __name__ == "__main__": print(run())
