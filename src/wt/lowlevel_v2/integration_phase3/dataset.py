"""Materialize the executor-aware HL dataset. Same frozen L=24 pool and H=15 U_LL as Phase-2."""
from __future__ import annotations

import json
import shutil
from typing import Any

import numpy as np

from wt.htw_phase941.protocol import sha256_bytes
from wt.lowlevel_v2.integration_phase2.utility_gen import generate_split_utilities, reconstruct_official_split

from .exec_loss import executor_targets
from .protocol import EXPECTED_UTIL_SHA, OUTPUT, PHASE2_OUTPUT, POOL_SIZE, TRAIN_NAME, VAL_NAME


def _label_stats(utilities: np.ndarray) -> dict[str, float]:
    u = np.asarray(utilities, dtype=np.float64).reshape(-1)
    return {
        "n": int(u.size),
        "mean": float(np.mean(u)),
        "std": float(np.std(u)),
        "min": float(np.min(u)),
        "max": float(np.max(u)),
        "n_nonpositive": int(np.sum(u <= 0.0)),
    }


def _save_split(data: dict, name: str) -> None:
    np.savez_compressed(
        OUTPUT / f"{name}.npz",
        **{key: data[key] for key in data if key not in {"meta", "hashes", "split", "seeds", "pool_size", "utility_per_seed", "reconstructed_npz"}},
    )
    (OUTPUT / f"{name}_manifest.json").write_text(
        json.dumps(
            {
                "split": name,
                "source_split": data.get("split"),
                "seeds": data["seeds"],
                "n": int(data["assignment"].shape[0]),
                "pool_size": data["pool_size"],
                "hashes": data["hashes"],
                "meta": data["meta"],
                "label": "U_LL_payload_increment",
                "q": "normalize(U_shift) if any(U<=0) else U/sum(U)",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _from_phase2_copy() -> bool:
    return (PHASE2_OUTPUT / "hl_rl_align_train.npz").is_file() and (PHASE2_OUTPUT / "hl_rl_align_val.npz").is_file()


def _rebuild_from_shards() -> dict[str, Any]:
    rebuilt = {}
    for src_name, dst_name, shard_split in (("phase945_train", TRAIN_NAME, "train"), ("phase945_val", VAL_NAME, "val")):
        print(f"rebuild executor split {dst_name} from official generator + Phase-2 U_LL shards", flush=True)
        data = reconstruct_official_split(src_name)
        generate_split_utilities(shard_split, data, workers=1)
        official = json.loads((PHASE2_OUTPUT / f"hl_rl_align_{shard_split}_manifest.json").read_text(encoding="utf-8"))
        if data["hashes"] != official["hashes"]:
            raise RuntimeError(f"candidate hashes drifted while rebuilding {dst_name}")
        _save_split(data, dst_name)
        rebuilt[dst_name] = data
    return rebuilt


def materialize_executor_dataset() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rerolled = False
    copied = False
    if (OUTPUT / f"{TRAIN_NAME}.npz").is_file() and (OUTPUT / f"{VAL_NAME}.npz").is_file():
        source = "existing Phase-3 npz"
    elif _from_phase2_copy():
        source_sha = sha256_bytes((PHASE2_OUTPUT / "hl_rl_align_train.npz").read_bytes() + (PHASE2_OUTPUT / "hl_rl_align_val.npz").read_bytes())
        if source_sha != EXPECTED_UTIL_SHA:
            raise RuntimeError(f"Phase-2 executor utility SHA mismatch: {source_sha}")
        for src_name, dst_name in (("hl_rl_align_train", TRAIN_NAME), ("hl_rl_align_val", VAL_NAME)):
            shutil.copy2(PHASE2_OUTPUT / f"{src_name}.npz", OUTPUT / f"{dst_name}.npz")
            man = json.loads((PHASE2_OUTPUT / f"{src_name}_manifest.json").read_text(encoding="utf-8"))
            man["split"] = dst_name
            man["label"] = "U_LL_payload_increment"
            man["q"] = "normalize(U_shift) if any(U<=0) else U/sum(U)"
            man["source_split"] = src_name
            man["source_utility_sha"] = source_sha
            (OUTPUT / f"{dst_name}_manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        copied = True
        source = "copied Phase-2 npz"
    else:
        _rebuild_from_shards()
        source = "reconstructed official pool + Phase-2 utility shards"
    all_u = []
    q_rows = []
    for dst_name in (TRAIN_NAME, VAL_NAME):
        packed = np.load(OUTPUT / f"{dst_name}.npz")
        u = np.asarray(packed["utility"], dtype=np.float64)
        if int(u.shape[1]) != int(POOL_SIZE):
            raise RuntimeError("candidate pool size is not L=24")
        all_u.append(u)
        for index in range(int(u.shape[0])):
            q = executor_targets(u[index])
            q_rows.append({"split": dst_name, "index": int(index), "q_sum": float(q.sum()), "q_min": float(q.min()), "q_max": float(q.max())})
    stacked = np.concatenate([arr.reshape(-1) for arr in all_u])
    stats = _label_stats(stacked)
    util_sha = sha256_bytes((OUTPUT / f"{TRAIN_NAME}.npz").read_bytes() + (OUTPUT / f"{VAL_NAME}.npz").read_bytes())
    payload = {
        "source": "Phase-2 frozen Low-Level H=15 RL-mean payload increment",
        "materialize": source,
        "candidate_generator": "unchanged official Phase-9.4.5 / Phase-2 pool",
        "pool_size": int(POOL_SIZE),
        "hold": 15,
        "rerolled": rerolled,
        "copied_phase2_npz": copied,
        "source_utility_sha": util_sha,
        "expected_utility_sha": EXPECTED_UTIL_SHA,
        "U_LL": stats,
        "q_sum_ok": all(abs(r["q_sum"] - 1.0) < 1.0e-8 for r in q_rows),
        "q_method": "U/sum(U); if any U<=0 then U-min(U)+eps then normalize",
        "artificial_q": False,
        "prefix_label": False,
        "candidate_conditional_q": False,
    }
    (OUTPUT / "HL_EXEC_DATASET.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "HL_EXEC_LABEL_STATS.json").write_text(json.dumps({"U_LL": stats, "per_state_q": q_rows[:8], "n_states": len(q_rows)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_exec_split(name: str) -> dict[str, Any]:
    packed = np.load(OUTPUT / f"{name}.npz")
    data = {key: packed[key] for key in packed.files}
    man = json.loads((OUTPUT / f"{name}_manifest.json").read_text(encoding="utf-8"))
    data["meta"] = man["meta"]
    data["hashes"] = man["hashes"]
    data["split"] = man["split"]
    data["seeds"] = man["seeds"]
    data["pool_size"] = man["pool_size"]
    return data


__all__ = ["load_exec_split", "materialize_executor_dataset"]
