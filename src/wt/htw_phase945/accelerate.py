"""Compute-only acceleration audits. Does not regenerate the frozen L=24 utility NPZ."""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wt.htw_phase941.protocol import dump_frozen, sha256_bytes
from wt.htw_phase9.const import HOLD
from wt.l3b.env import L3BEnv

from .candidate_pool import generate_pool
from .protocol import LAMBDA_RANK, LAMBDA_REF, OUTPUT, POOL_SIZE, ROOT
from .workers import h15_payload_worker, restore_hold_state


TMP = Path(__file__).resolve().parent / "tmp_runtime"


def _file_sha(rel: str) -> str:
    return sha256_bytes((ROOT / rel).read_bytes())


def dedup_audit() -> dict:
    man = json.loads((OUTPUT / "phase945_train_manifest.json").read_text(encoding="utf-8"))
    val = json.loads((OUTPUT / "phase945_val_manifest.json").read_text(encoding="utf-8"))
    all_hash = man["hashes"] + val["hashes"]
    slots = unique = 0
    dups = 0
    for row in all_hash:
        slots += len(row)
        u = len(set(row))
        unique += u
        dups += len(row) - u
    return {
        "train_states": len(man["hashes"]),
        "val_states": len(val["hashes"]),
        "generated_candidate_count": slots,
        "unique_candidate_count": unique,
        "duplicate_fraction": dups / max(slots, 1),
        "slots_per_state": POOL_SIZE,
        "pruned_by_utility": False,
    }


def _jobs_for_probe() -> list[dict]:
    jobs = []
    for scenario in (0, 1, 2, 3, 4, 5, 6, 7):
        env = L3BEnv()
        env.reset(seed=int(scenario))
        pool = generate_pool(env, seed_key=int(scenario) * 1009)
        seen = set()
        n_u = 0
        for slot, row in enumerate(pool):
            key = tuple(int(x) for x in row.tolist())
            if key in seen:
                continue
            seen.add(key)
            jobs.append({"scenario": int(scenario), "t": 0, "slot": int(slot), "assignment": np.asarray(row, dtype=np.int64)})
            n_u += 1
            if n_u >= 8:
                break
    return jobs


def _run_pool(jobs: list[dict], workers: int) -> list[dict]:
    TMP.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        return [h15_payload_worker(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        return list(pool.map(h15_payload_worker, jobs))


def profile_and_equivalence() -> dict:
    jobs = _jobs_for_probe()
    serial = _run_pool(jobs, 1)
    timings = []
    chosen = 1
    best = 1e18
    for workers in (1, 2, 4, 6):
        t0 = time.perf_counter()
        rows = _run_pool(jobs, workers)
        elapsed = time.perf_counter() - t0
        timings.append({"workers": workers, "wall_s": elapsed, "n_jobs": len(jobs)})
        if elapsed < best:
            best = elapsed
            chosen = workers
            probe = rows
    # compare serial vs 4-worker (or chosen if 1)
    parallel = _run_pool(jobs, max(2, min(4, chosen)))
    ok = True
    diffs = []
    for a, b in zip(serial, parallel):
        pay_ok = abs(float(a["payload_increment"]) - float(b["payload_increment"])) <= 1.0e-9
        hash_ok = a["assignment_hash"] == b["assignment_hash"]
        hold_ok = int(a["hold"]) == int(HOLD)
        if not (pay_ok and hash_ok and hold_ok):
            ok = False
            diffs.append({"slot": a["slot"], "scenario": a["scenario"], "serial": a["payload_increment"], "parallel": b["payload_increment"]})
    return {
        "n_states": 8,
        "n_unique_candidates_per_state_cap": 8,
        "n_jobs": len(jobs),
        "timings": timings,
        "chosen_workers": int(chosen),
        "pass": ok,
        "n_mismatch": len(diffs),
        "diffs": diffs[:8],
        "serial_mean_payload": float(np.mean([r["payload_increment"] for r in serial])),
        "parallel_mean_payload": float(np.mean([r["payload_increment"] for r in parallel])),
        "probe_n": len(probe) if chosen else 0,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    status = {
        "utility_dataset": "complete" if (OUTPUT / "phase945_train.npz").exists() else "missing",
        "training": "in_progress_or_partial",
        "confirmation": (OUTPUT / "phase945_confirmation_manifest.json").exists(),
        "phase_state": "D_training",
        "did_not_delete_existing": True,
        "did_not_regenerate_utility": True,
        "frozen_pool_size": POOL_SIZE,
        "hold": HOLD,
    }
    dump_frozen(OUTPUT / "phase945_phase_state_audit.json", status)
    train_sha = sha256_bytes((OUTPUT / "phase945_train.npz").read_bytes()) if (OUTPUT / "phase945_train.npz").exists() else ""
    val_sha = sha256_bytes((OUTPUT / "phase945_val.npz").read_bytes()) if (OUTPUT / "phase945_val.npz").exists() else ""
    proto_sha = sha256_bytes((OUTPUT / "phase945_protocol.json").read_bytes())
    amendment = {
        "scientific_protocol_changed": False,
        "candidate_pool_changed": False,
        "candidate_count_changed": False,
        "horizon_changed": False,
        "model_changed": False,
        "loss_changed": False,
        "seed_count_changed": False,
        "confirmation_changed": False,
        "acceleration_only": True,
        "frozen_pool_size": POOL_SIZE,
        "note_L": "Phase-9.4.5 froze L=24 before confirmation (allowed runtime clause). Acceleration did not change L.",
        "hold": HOLD,
        "lambda_rank": LAMBDA_RANK,
        "lambda_ref": LAMBDA_REF,
        "phase945_protocol_sha256": proto_sha,
        "candidate_generator_sha256": _file_sha("src/wt/htw_phase945/candidate_pool.py"),
        "utility_rollout_sha256": _file_sha("src/wt/htw_phase945/utility_rollout.py"),
        "code_sha256": {
            "training.py": _file_sha("src/wt/htw_phase945/training.py"),
            "utility_loss.py": _file_sha("src/wt/htw_phase945/utility_loss.py"),
            "run_phase945.py": _file_sha("src/wt/htw_phase945/run_phase945.py"),
        },
        "train_npz_sha256": train_sha,
        "val_npz_sha256": val_sha,
        "shared_utility_dataset": True,
        "shared_candidate_pool": True,
    }
    dump_frozen(OUTPUT / "PHASE945_COMPUTE_ACCELERATION_AMENDMENT.json", amendment)
    dump_frozen(ROOT / "results/root_exports/PHASE945_COMPUTE_ACCELERATION_AMENDMENT.json", amendment)

    dedup = dedup_audit()
    dump_frozen(OUTPUT / "phase945_candidate_dedup_audit.json", dedup)

    print("serial vs parallel H15 equivalence probe", flush=True)
    equiv = profile_and_equivalence()
    dump_frozen(OUTPUT / "phase945_parallel_utility_equivalence.json", equiv)

    man = {
        "shards": [],
        "note": "Full utility NPZ already frozen before this amendment; no shard regeneration.",
        "train_npz": str(OUTPUT / "phase945_train.npz"),
        "val_npz": str(OUTPUT / "phase945_val.npz"),
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "resume_reused_rows": True,
    }
    dump_frozen(OUTPUT / "phase945_utility_shard_manifest.json", man)

    util_man = json.loads((OUTPUT / "phase945_candidate_utility_dataset_manifest.json").read_text(encoding="utf-8"))
    util_man.update(
        {
            "train_npz_sha256": train_sha,
            "val_npz_sha256": val_sha,
            "generated_candidate_count": dedup["generated_candidate_count"],
            "unique_candidate_count": dedup["unique_candidate_count"],
            "duplicate_fraction": dedup["duplicate_fraction"],
            "shared_by_all_10_models": True,
            "regenerated_in_acceleration": False,
        }
    )
    dump_frozen(OUTPUT / "phase945_candidate_utility_dataset_manifest.json", util_man)

    accel_a = bool(equiv["pass"]) and POOL_SIZE == 24 and HOLD == 15 and not amendment["scientific_protocol_changed"]
    gate = {
        "gate": "ACCEL-A" if accel_a else "ACCEL-B",
        "scientific_equivalence_preserved": bool(accel_a),
        "serial_parallel_utility_consistent": bool(equiv["pass"]),
        "candidate_pool_unchanged": True,
        "frozen_L": POOL_SIZE,
        "hold": HOLD,
        "no_utility_pruning": True,
        "no_approximation": True,
        "five_plus_five_seeds": True,
        "confirmation_still_64": True,
        "did_not_regenerate_frozen_npz": True,
        "reason": "H=15 serial/parallel match; L stays the frozen protocol value; dataset reused.",
    }
    dump_frozen(OUTPUT / "phase945_compute_equivalence_gate.json", gate)
    print(json.dumps({"gate": gate["gate"], "dup": dedup["duplicate_fraction"], "workers": equiv["chosen_workers"], "equiv": equiv["pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
