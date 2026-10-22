"""Copy frozen checkpoint and dump configs/tables from live code + Phase-10.6 disk files."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

from . import _bootstrap  # noqa: F401

from wt.htw_phase9.const import BUOY_DIM, GLOBAL_DIM, HOLD, IDLE_INDEX, MAX_BUOYS, MAX_UAVS, UAV_DIM
from wt.htw_phase94.assignment_context import PHYS_DIM
from wt.htw_phase943.order_free_decoder import WIDTH, OrderFreeAssignmentModel
from wt.htw_phase101.protocol import HIGH_CKPT
from wt.htw_phase103.protocol import ENV_FILE
from wt.l3b.config import L3BConfig

from .constants import ARTIFACTS, EXPECTED_ENV_SHA, EXPECTED_HIGH_SHA, FINAL_CKPT, ROOT, sha256_bytes

PHASE106 = ROOT / "src" / "wt" / "htw_phase106" / "outputs" / "untouched"
STATUS106 = PHASE106 / "phase106_final_scientific_status.json"


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pack() -> dict:
    status = json.loads(STATUS106.read_text(encoding="utf-8"))
    required = {
        "architecture_finalized": True,
        "training_finished": True,
        "method_development_stopped": True,
        "final_untouched_consumed": True,
        "final_gate": "A",
        "final_system_confirmed": True,
    }
    for key, value in required.items():
        if status.get(key) != value:
            raise RuntimeError(f"scientific status mismatch: {key}={status.get(key)!r} expected {value!r}")
    ckpt_src = HIGH_CKPT
    digest = sha256_bytes(ckpt_src.read_bytes())
    if digest != EXPECTED_HIGH_SHA:
        raise RuntimeError("source checkpoint SHA is not the frozen Phase-10.5/10.6 file")
    env_sha = sha256_bytes(ENV_FILE.read_bytes())
    if env_sha != EXPECTED_ENV_SHA:
        raise RuntimeError("environment SHA is not the frozen L3-B file")
    FINAL_CKPT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_src, FINAL_CKPT)
    if sha256_bytes(FINAL_CKPT.read_bytes()) != digest:
        raise RuntimeError("copied checkpoint SHA changed")
    cfg = L3BConfig()
    model = OrderFreeAssignmentModel("deepsets")
    n_params = int(sum(p.numel() for p in model.parameters()))
    configs = ARTIFACTS / "configs"
    _dump(
        configs / "environment.json",
        asdict(cfg)
        | {
            "step_length_m": cfg.step_length_m,
            "M": cfg.num_uavs,
            "K": cfg.num_buoys,
            "T": cfg.episode_horizon,
            "hold": HOLD,
            "env_sha256": env_sha,
            "source": str(ENV_FILE),
        },
    )
    _dump(
        configs / "model.json",
        {
            "encoder": "deepsets",
            "checkpoint": str(FINAL_CKPT),
            "source_checkpoint": str(ckpt_src),
            "sha256": digest,
            "train_seed": 0,
            "train_phase": "9.4.5 utility-aligned, archived in 10.0, used in 10.5/10.6",
            "best_epoch": 60,
            "n_parameters": n_params,
            "input_schema": {
                "uav_dim": UAV_DIM,
                "buoy_dim": BUOY_DIM,
                "global_dim": GLOBAL_DIM,
                "max_uavs": MAX_UAVS,
                "max_buoys": MAX_BUOYS,
                "normalization": "median_mad",
                "norm_eps": 1.0,
            },
        },
    )
    _dump(
        configs / "decoder.json",
        {
            "name": "OrderFreeSetDecoder",
            "width": WIDTH,
            "phys_dim": PHYS_DIM,
            "idle_token": IDLE_INDEX,
            "env_idle": -1,
            "permutation_equivariant": True,
            "tie_atol": 1.0e-6,
        },
    )
    _dump(
        configs / "executor.json",
        {
            "movement": "fly-toward",
            "gate": "InRangeZeroMotionGate",
            "gate_rule": "non-idle AND valid AND queue>0 AND env._in_range -> movement (0,0)",
            "hold": HOLD,
            "not_ppo": True,
        },
    )
    evaluation = {
        "primary_metric": "episode cumulative payload",
        "hold": HOLD,
        "demo_default_seed": 8,
        "untouched_seeds_forbidden": list(range(106000, 106064)),
        "phase106_results": str(PHASE106 / "phase106_final_results.csv"),
        "recompute_untouched": False,
        "final_gate": "FINAL-A",
        "phase106_status_gate": status.get("final_gate"),
    }
    _dump(configs / "evaluation.json", evaluation)
    for name in ("environment", "model", "decoder", "executor", "evaluation"):
        src = configs / f"{name}.json"
        (configs / f"{name}.yaml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    ckpt_meta = {
        "path": str(FINAL_CKPT),
        "sha256": digest,
        "train_seed": 0,
        "train_phase": "HTW Phase-9.4.5 utility-aligned DeepSets, archived Phase-10.0",
        "epoch": 60,
        "encoder": "DeepSetsAssignmentHead",
        "decoder": "OrderFreeSetDecoder",
        "n_parameters": n_params,
        "selected_by_phase106": False,
        "matches_phase105_106": True,
    }
    _dump(ARTIFACTS / "checkpoints" / "checkpoint_manifest.json", ckpt_meta)

    src_rows = list(csv.DictReader((PHASE106 / "phase106_final_results.csv").open(encoding="utf-8")))
    out_csv = ARTIFACTS / "results" / "final_results_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method_id", "method", "payload", "completed_blocks", "residual_queue", "dropped", "flight_distance", "legality"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in src_rows:
            writer.writerow({key: row[key] for key in fields})
    shutil.copy2(out_csv, ROOT / "results/root_exports/final_results_table.csv")
    boot = json.loads((PHASE106 / "phase106_paired_bootstrap.json").read_text(encoding="utf-8"))
    pair_path = ARTIFACTS / "results" / "final_pairwise_statistics.csv"
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison", "mean_delta", "median_delta", "positive_fraction", "ci95_low", "ci95_high", "relative_gain"])
        writer.writeheader()
        for key in ("M3_minus_M0", "M3_minus_M1", "M3_minus_M2"):
            item = boot[key]
            writer.writerow(
                {
                    "comparison": key.replace("_minus_", "-"),
                    "mean_delta": item["mean_delta"],
                    "median_delta": item["median_delta"],
                    "positive_fraction": item["positive_scenario_frac"],
                    "ci95_low": item["ci95"]["low"],
                    "ci95_high": item["ci95"]["high"],
                    "relative_gain": item["relative_gain"],
                }
            )
    shutil.copy2(pair_path, ROOT / "results/root_exports/final_pairwise_statistics.csv")
    versions = {
        "python": sys.version.split()[0],
        "torch": __import__("torch").__version__,
        "numpy": __import__("numpy").__version__,
        "matplotlib": __import__("matplotlib").__version__,
    }
    req = ARTIFACTS / "requirements.txt"
    req.write_text(
        "\n".join(
            [
                f"# FINAL-HIER runtime recorded from the live interpreter. Do not upgrade blindly.",
                f"# python=={versions['python']}",
                f"numpy=={versions['numpy']}",
                f"torch=={versions['torch']}",
                f"matplotlib=={versions['matplotlib']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    tz = timezone(timedelta(hours=8))
    freeze = {
        "architecture_finalized": True,
        "training_finished": True,
        "method_development_stopped": True,
        "final_untouched_consumed": True,
        "final_method": "DeepSets-util + Order-Free Decoder + Task Option + fly-toward + InRangeZeroMotionGate",
        "high_level": "DeepSets",
        "HGNN_selected": False,
        "low_level_PPO_selected": False,
        "Transformer_selected": False,
        "hold": 15,
        "checkpoint_sha": digest,
        "code_freeze_timestamp": datetime.now(tz).isoformat(timespec="seconds"),
        "scientific_experiments_finished": True,
        "code_development_complete": True,
        "reproducibility_package_complete": True,
        "package_versions": versions,
    }
    ROOT.joinpath("FINAL_CODE_FREEZE.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _dump(ARTIFACTS / "FINAL_CODE_FREEZE.json", freeze)
    return freeze


if __name__ == "__main__":
    print(json.dumps(pack(), indent=2))
