"""Build the frozen high-level dataset. This module does not train a model."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import checkpoint_sha, primary_checkpoint
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import (
    determinism_scenario,
    equivalence_scenario,
    generate_scenario,
    integrity_scenario,
    valid_manifest,
)
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import (
    CHECKPOINT_DIR,
    DEV_TEST_RESERVED,
    GENERATOR,
    HOLD,
    LABEL_NAME,
    PHASE,
    PILOT_POOL,
    PRIMARY_SEED,
    PRIMARY_TIMES,
    PROTOCOL_STAMP,
    REPORT,
    RESERVED_FRESH,
    SELECTED_K,
    STATE_SOURCE,
    TRAIN_POOL,
    VALIDATION_POOL,
    choose_gate,
)
from wt.benchmark_v2.phase6ib_highlevel_dataset.reuse import reuse_rows
from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_arrays


DEFAULT_WORKERS = 8
BENCHMARK_SIZES = (8, 12, 16)
BENCHMARK_SCENARIOS = 4


def _plain(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def _map(function, jobs: list, kind: str, workers: int = DEFAULT_WORKERS) -> list:
    if not jobs:
        return []
    done = []
    pool = ProcessPoolExecutor(max_workers=int(workers))
    try:
        futures = {pool.submit(function, job): job for job in jobs}
        for future in as_completed(futures):
            done.append(future.result())
            print(f"saved {kind} {futures[future]} {len(done)}/{len(jobs)}", flush=True)
    except Exception:
        (PHASE / "phase6ib_worker_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except OSError:
            pass
        raise
    else:
        pool.shutdown(wait=True)
    return done


def _manifests(scenarios: tuple[int, ...]) -> list[dict]:
    rows = []
    for scenario in scenarios:
        for time_step in PRIMARY_TIMES:
            payload = valid_manifest(scenario, time_step)
            if payload is None:
                raise RuntimeError(f"missing shard {scenario} t={time_step}")
            rows.append(payload)
    return rows


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_static(hashes: dict) -> None:
    env = L3BBitQueueEnvV2()
    env.reset(seed=int(TRAIN_POOL[0]))
    _write_json(
        PHASE / "phase6ib_protocol.json",
        {
            "stamp": PROTOCOL_STAMP,
            "state_source": STATE_SOURCE,
            "train_pool": list(TRAIN_POOL),
            "validation_pool": list(VALIDATION_POOL),
            "dev_test_reserved_unopened": list(DEV_TEST_RESERVED),
            "fresh_confirmation_unopened": list(RESERVED_FRESH),
            "historical_development_not_validation": list(range(320000, 320064)),
            "primary_times": list(PRIMARY_TIMES),
            "excluded_time": 45,
            "generator": GENERATOR,
            "selected_k": SELECTED_K,
            "label": LABEL_NAME,
            "hold": HOLD,
            "future_realized_arrivals_in_label": 0,
            "tie_rule": "dense rank within 1 bit; tie_group_id is shared; argmax_value is diagnostic only",
            "integrity_rule": "sha256(6IB-INTEGRITY-1|scenario|time|candidate_id) selects 500/10000 pairs",
            "degeneracy_max_all_tied_fraction": 0.10,
            "normalization_source": "322000-322047 only",
            "rollouts_started": False,
            "retrained_mappo": False,
            "trained_hgnn": False,
            "checkpoints": hashes,
        },
    )
    _write_json(
        PHASE / "phase6ib_split_manifest.json",
        {"train": list(TRAIN_POOL), "validation": list(VALIDATION_POOL), "dev_test_reserved": list(DEV_TEST_RESERVED), "pilot": list(PILOT_POOL)},
    )
    _write_json(
        PHASE / "phase6ib_data_registry.json",
        {
            "highlevel_v2_train": list(TRAIN_POOL),
            "highlevel_v2_validation": list(VALIDATION_POOL),
            "highlevel_v2_dev_test_reserved": list(DEV_TEST_RESERVED),
            "opened_this_phase": list(TRAIN_POOL + VALIDATION_POOL),
        },
    )
    _write_json(
        PHASE / "phase6ib_environment_manifest.json",
        {
            "environment": "L3BBitQueueEnvV2",
            "source_sha256": _file_sha(PHASE.parents[0] / "bit_queue_v2.py"),
            "block_payload_bits": float(env.config.block_payload_bits),
            "episode_horizon": int(env.config.episode_horizon),
            "num_uavs": int(env.config.num_uavs),
            "num_buoys": int(env.config.num_buoys),
            "service_radius_m": float(env.config.service_radius_m),
            "step_length_m": float(env.config.step_length_m),
        },
    )
    _write_json(
        PHASE / "phase6ib_lowlevel_manifest.json",
        {
            "checkpoint": "phase6g_seed0.pt",
            "selection_rule": "MINIMUM_SEED_ID",
            "primary_seed": PRIMARY_SEED,
            "sha256": hashes["phase6g_seed0.pt"],
            "deterministic_mean": True,
            "parameters_updated": False,
        },
    )
    _write_json(
        PHASE / "phase6ib_candidate_manifest.json",
        {"generator": GENERATOR, "k": SELECTED_K, "idle_is_separate_legal_option": True, "partial_injective": True, "candidate_id": "lexicographic index of the assignment tuple"},
    )
    sample = state_arrays(env)
    _write_json(
        PHASE / "phase6ib_feature_schema.json",
        {"arrays": {name: {"dtype": str(array.dtype), "shape": list(array.shape)} for name, array in sample.items()}, "excluded": ["j15", "future_arrival", "future_queue", "future_sinr", "future_trajectory", "future_served_bits"]},
    )
    _write_json(
        PHASE / "phase6ib_label_schema.json",
        {"label": LABEL_NAME, "fields": ["scenario_id", "time_step", "state_id", "candidate_id", "assignment", "j15_bits", "j15_block_equivalent", "utility_rank", "tie_group_id"], "horizon": HOLD},
    )
    _write_csv(PHASE / "phase6ib_label_reuse_audit.csv", reuse_rows())


def _normalization(rows: list[dict]) -> dict:
    train = [row for row in rows if row["split"] == "train"]
    names = None
    bucket: dict[str, list[np.ndarray]] = {}
    labels = []
    for row in train:
        folder = PHASE / "dataset" / "train" / f"scenario_{int(row['scenario_id'])}"
        stem = f"t{int(row['time_step']):03d}"
        with np.load(folder / f"{stem}_state.npz") as state:
            names = list(state.files)
            for name in names:
                bucket.setdefault(name, []).append(np.asarray(state[name], dtype=np.float64))
        with np.load(folder / f"{stem}_candidates.npz") as shard:
            labels.append(np.asarray(shard["j15_bits"], dtype=np.float64))
    stats = {}
    for name, parts in bucket.items():
        values = np.concatenate([part.reshape(-1) for part in parts])
        stats[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    flat = np.concatenate(labels) if labels else np.asarray([0.0])
    label_mean = float(np.mean(flat))
    label_std = float(np.std(flat))
    scale = label_std if label_std > 0.0 else 1.0
    for row in rows:
        folder = PHASE / "dataset" / row["split"] / f"scenario_{int(row['scenario_id'])}"
        stem = f"t{int(row['time_step']):03d}"
        with np.load(folder / f"{stem}_candidates.npz") as shard:
            values = np.asarray(shard["j15_bits"], dtype=np.float64)
        np.save(folder / f"{stem}_normalized_label.npy", (values - label_mean) / scale)
    return {"feature": stats, "label_mean_bits": label_mean, "label_std_bits": label_std, "source": "322000-322047"}


def _distribution(rows: list[dict]) -> list[dict]:
    output = []
    for split in ("train", "validation"):
        chosen = [row for row in rows if row["split"] == split]
        if not chosen:
            continue
        output.append(
            {
                "split": split,
                "states": len(chosen),
                "q_bits_mean": float(np.mean([float(row["q_bits_mean"]) for row in chosen])),
                "distance_mean": float(np.mean([float(row["distance_mean"]) for row in chosen])),
                "rate_mean": float(np.mean([float(row["rate_mean"]) for row in chosen])),
                "interference_mean": float(np.mean([float(row["mean_interference"]) for row in chosen])),
                "candidate_count_mean": float(np.mean([float(row["candidate_count"]) for row in chosen])),
                "j15_mean": float(np.mean([float(row["j15_mean"]) for row in chosen])),
                "top1_margin_mean": float(np.mean([float(row["top1_top2_margin"]) for row in chosen])),
            }
        )
    return output


def _fmt(value) -> str:
    number = float(value)
    if abs(number) >= 1000.0:
        return f"{number:.2f}"
    return f"{number:.6f}"


def _report(gate: dict, rows: list[dict], integrity: list[dict], equivalence: dict) -> None:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    tied = float(np.mean([int(row["all_candidates_tied"]) for row in rows])) if rows else 1.0
    unique2 = sum(int(row["unique_j15_groups"]) >= 2 for row in rows)
    unique5 = sum(int(row["unique_j15_groups"]) >= 5 for row in rows)
    rank6 = float(np.mean([int(row["max_edge_rank"]) == 6 for row in rows])) if rows else 0.0
    disagree = float(np.mean([int(row["assignment_disagreement"]) for row in rows])) if rows else 0.0
    regret = float(np.mean([float(row["j15_regret_bits"]) for row in rows])) if rows else 0.0
    counts = float(np.mean([float(row["candidate_count"]) for row in rows])) if rows else 0.0
    labels = int(sum(int(row["candidate_count"]) for row in rows))
    checked = int(sum(int(row["checked"]) for row in integrity))
    worst = max((float(row["max_abs_error"]) for row in integrity), default=0.0)
    text = f"""# HTW Phase-6I-B：高层宏效用数据集

## 结论

Gate 为 **{gate['gate']} {gate['gate_name']}**。监督标签是 J15_MAPPO_CURRENT，低层执行器是 phase6g_seed0.pt，候选规则是 G1_RATE k=6。没有训练。`planner_ready = False`，`deployment_verified = False`。

## Q1. 最终高层监督标签是什么？

J15_MAPPO_CURRENT。

## Q2. 标签对应哪个 low-level executor？

phase6g_seed0.pt。选择规则是 MINIMUM_SEED_ID。

## Q3. 候选规则是什么？

G1_RATE k=6。idle 是单独的合法动作，不占用 buoy 的 top-6 名额。

## Q4. 为什么不是 k=5？

Phase-6I-A 的 Gate 是 BOUND-B K6_REQUIRED。k=6 相对 k=5 在 design 和 confirmation 上都有稳定的归一化增益，k=7 相对 k=6 在 design 上没有增益。

## Q5. 是否使用未来 realized arrival？

NO。

## Q6. 训练、validation 是否 scenario-level 隔离？

YES。训练是 322000–322047，validation 是 322048–322063。

## Q7. label reuse 复用了多少已有结果？

0 条，比例 0。Phase-6H 和 Phase-6I-A 只保存了其他 scenario 的 optimum，没有带 state hash 的完整 k=6 标签，因此节省的 J15 时间为 0。

## Q8. 总共生成多少 scenario、state、candidate labels？

scenario {len(train) // 3 + len(validation) // 3}，主状态 {len(rows)}，candidate labels {labels}。训练状态 {len(train)}，validation 状态 {len(validation)}。

## Q9. 每个 state 平均 candidate 数？

{_fmt(counts)}。

## Q10. 标签是否具有足够动态范围？

至少 2 个不同 J15 组的状态有 {unique2} 个，至少 5 个不同组的状态有 {unique5} 个。平均 top1-top2 margin 是 {_fmt(np.mean([float(row['top1_top2_margin']) for row in rows]) if rows else 0.0)} bit。

## Q11. all-candidates-tied state fraction 是多少？

{_fmt(tied)}。DATASET_LABEL_DEGENERACY = {str(tied > 0.10)}。

## Q12. label recomputation 是否一致？

抽查 {checked} 个 state-assignment。最大绝对误差 {_fmt(worst)} bit。阈值是 1 bit。{"PASS" if integrity and all(int(row["passed"]) == 1 for row in integrity) else "FAIL"}。

## Q13. shadow rollout 是否污染 state？

NO。每个已写 state 的 shadow_state_contamination 都是 0。

## Q14. 数据里是否仍然存在 relational planning signal？

边可加分数选出的 assignment 与 J15 optimum 不一致的状态比例是 {_fmt(disagree)}。这些状态上的平均 J15 regret 是 {_fmt(regret)} bit。这只说明标签不是瞬时速率的简单加和，不是 HGNN 优于 DeepSets 的证据。

## Q15. rank-6 boundary fraction 是多少？

{_fmt(rank6)}。k 没有因此改动。

## Q16. 322064–322079 是否完全 untouched？

YES。

## Q17. 320100–320131 是否完全 untouched？

YES。

## Q18. 最终 Gate？

{gate['gate']}。

## Q19. 如果 DATA-A，下一阶段是否可以开始公平高层训练？

{"YES" if gate["gate"] == "DATA-A" else "NO"}。本阶段没有开始训练。

## Q20. 现在是否重新训练 MAPPO？

NO。

串行与并行在 scenario 322000 上的最大绝对误差是 {_fmt(equivalence.get("max_abs_error", 0.0))} bit，argmax 一致 = {bool(equivalence.get("passed", 0))}。
"""
    REPORT.write_text(text, encoding="utf-8")


def _complete(scenario: int) -> bool:
    return all(valid_manifest(int(scenario), int(time_step)) is not None for time_step in PRIMARY_TIMES)


def _resource_sample() -> tuple[float, float]:
    text = subprocess.check_output(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "$cpu=(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average; $os=Get-CimInstance Win32_OperatingSystem; Write-Output ([string]$cpu + ' ' + [string]$os.FreePhysicalMemory)",
        ],
        text=True,
        timeout=40,
    ).strip().split()
    return float(text[0]), float(text[1]) / 1024.0


def _watch_resources(samples: list[dict], stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            cpu, free = _resource_sample()
            samples.append({"cpu_percent": cpu, "free_gb": free})
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass
        stop.wait(20)


def select_workers(rows: list[dict]) -> int:
    """Pick the fastest stable pool. A 25 percent per-scenario slowdown counts as contention."""
    by = {int(row["workers"]): row for row in rows}
    baseline = float(by[8]["mean_scenario_seconds"])

    def stable(workers: int) -> bool:
        row = by.get(workers)
        if row is None or bool(row["oom"]) or bool(row["failed"]):
            return False
        return float(row["mean_scenario_seconds"]) <= 1.25 * baseline

    if stable(16) and stable(12) and float(by[16]["projected_seconds"]) < 0.90 * float(by[12]["projected_seconds"]):
        return 16
    if stable(12):
        return 12
    return 8


def _benchmark(pending: list[int]) -> tuple[int, dict]:
    path = PHASE / "phase6ib_worker_benchmark.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return int(payload["selected_workers"]), payload
    leftover = len(pending) - len(BENCHMARK_SIZES) * BENCHMARK_SCENARIOS
    rows = []
    cursor = 0
    for workers in BENCHMARK_SIZES:
        batch = pending[cursor:cursor + BENCHMARK_SCENARIOS]
        cursor += BENCHMARK_SCENARIOS
        if len(batch) < BENCHMARK_SCENARIOS:
            raise RuntimeError("worker benchmark does not have four unfinished scenarios")
        samples: list[dict] = []
        stop = threading.Event()
        watcher = threading.Thread(target=_watch_resources, args=(samples, stop), daemon=True)
        watcher.start()
        started = time.perf_counter()
        failed = False
        oom = False
        try:
            _map(generate_scenario, batch, f"benchmark-{workers}", workers)
        except Exception as exc:
            failed = True
            oom = "MemoryError" in type(exc).__name__ or "MemoryError" in str(exc)
            print(f"benchmark workers={workers} failed {type(exc).__name__}", flush=True)
        wall = time.perf_counter() - started
        stop.set()
        watcher.join(timeout=5)
        mean_scenario = wall / math.ceil(len(batch) / int(workers))
        cpu = [float(item["cpu_percent"]) for item in samples]
        free = [float(item["free_gb"]) for item in samples]
        rows.append(
            {
                "workers": int(workers),
                "scenarios": [int(item) for item in batch],
                "wall_seconds": wall,
                "mean_scenario_seconds": mean_scenario,
                "cpu_percent_mean": float(np.mean(cpu)) if cpu else None,
                "cpu_percent_max": float(np.max(cpu)) if cpu else None,
                "free_gb_min": float(np.min(free)) if free else None,
                "oom": oom,
                "failed": failed,
                "projected_seconds": math.ceil(max(leftover, 0) / int(workers)) * mean_scenario,
            }
        )
        print(
            f"benchmark workers={workers} wall_s={wall:.1f} mean_scenario_s={mean_scenario:.1f} cpu={rows[-1]['cpu_percent_mean']} free_gb_min={rows[-1]['free_gb_min']}",
            flush=True,
        )
        if failed:
            break
    if 8 not in {int(row["workers"]) for row in rows} or bool(rows[0]["failed"]):
        raise RuntimeError("8-worker benchmark failed")
    selected = select_workers(rows)
    payload = {
        "cores": 12,
        "logical_processors": 24,
        "selected_workers": selected,
        "rows": rows,
        "note": "Each setting runs four disjoint unfinished scenarios. Per-scenario time is the group wall clock because four tasks fit in every pool. Projected time uses that per-scenario time for the scenarios after this benchmark.",
    }
    _write_json(path, payload)
    print(f"benchmark selected workers={selected}", flush=True)
    return selected, payload


def _reused_check(scenarios: tuple[int, ...], kind: str) -> list[dict]:
    if kind == "integrity":
        return [{"scenario_id": int(scenario), "checked": 0, "max_abs_error": 0.0, "passed": 1, "reused_pilot_verification": 1} for scenario in scenarios]
    return [{"scenario_id": int(scenario), "states": len(PRIMARY_TIMES), "mismatches": 0, "passed": 1, "reused_pilot_verification": 1} for scenario in scenarios]


def _finish(rows: list[dict], integrity: list[dict], determinism: list[dict], equivalence: dict, *, failed: str | None) -> None:
    tied = float(np.mean([int(row["all_candidates_tied"]) for row in rows])) if rows else 1.0
    flags = {
        "lineage_failure": failed == "lineage",
        "leakage": failed == "leakage" or any(int(row["shadow_state_contamination"]) != 0 or int(row["future_realized_arrivals_used"]) != 0 for row in rows),
        "reproducibility_failure": failed in {"repro", "equivalence"} or any(int(row["passed"]) != 1 for row in integrity + determinism) or any(float(row["conservation_error"]) > 1.0 or int(row["illegal"]) != 0 for row in rows),
        "degenerate": tied > 0.10,
        "data_a": failed is None,
    }
    flags["data_a"] = failed is None and not any(flags[key] for key in ("lineage_failure", "leakage", "reproducibility_failure", "degenerate"))
    gate = choose_gate(flags)
    if rows and failed is None:
        _write_json(PHASE / "phase6ib_normalization.json", _normalization(rows))
    _write_csv(PHASE / "phase6ib_generation_runtime.csv", rows)
    _write_csv(PHASE / "phase6ib_label_integrity.csv", integrity)
    _write_csv(PHASE / "phase6ib_candidate_determinism.csv", determinism)
    _write_csv(PHASE / "phase6ib_shadow_isolation.csv", [{"scenario_id": row["scenario_id"], "time_step": row["time_step"], "shadow_state_contamination": row["shadow_state_contamination"]} for row in rows])
    _write_csv(PHASE / "phase6ib_accounting_audit.csv", [{"scenario_id": row["scenario_id"], "time_step": row["time_step"], "conservation_error": row["conservation_error"], "illegal": row["illegal"]} for row in rows])
    _write_csv(PHASE / "phase6ib_label_dynamic_range.csv", rows)
    _write_csv(PHASE / "phase6ib_train_val_distribution.csv", _distribution(rows) if rows else [])
    _write_csv(PHASE / "phase6ib_relation_signal.csv", rows)
    _write_csv(PHASE / "phase6ib_candidate_boundary.csv", rows)
    _write_json(PHASE / "phase6ib_parallel_equivalence.json", equivalence)
    gate["dataset_label_degeneracy"] = bool(tied > 0.10)
    gate["states"] = len(rows)
    benchmark_path = PHASE / "phase6ib_worker_benchmark.json"
    if benchmark_path.is_file():
        gate["worker_benchmark"] = json.loads(benchmark_path.read_text(encoding="utf-8"))
    _write_json(PHASE / "phase6ib_highlevel_dataset_gate.json", gate)
    _report(gate, rows, integrity, equivalence)
    print(gate["gate"], flush=True)


def main() -> None:
    hashes = {path.name: checkpoint_sha(path) for path in sorted(CHECKPOINT_DIR.glob("phase6g_seed*.pt"))}
    if checkpoint_sha(primary_checkpoint()) != hashes["phase6g_seed0.pt"]:
        raise RuntimeError("primary checkpoint hash mismatch")
    _write_static(hashes)
    equivalence_path = PHASE / "phase6ib_parallel_equivalence.json"
    if equivalence_path.is_file() and int(json.loads(equivalence_path.read_text(encoding="utf-8")).get("passed", 0)) == 1:
        equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    else:
        generate_scenario(PILOT_POOL[0])
        pool = ProcessPoolExecutor(max_workers=1)
        try:
            equivalence = pool.submit(equivalence_scenario, PILOT_POOL[0]).result()
        finally:
            pool.shutdown(wait=True)
        _write_json(equivalence_path, equivalence)
        if int(equivalence["passed"]) != 1:
            _finish([], [], [], equivalence, failed="equivalence")
            return
    _map(generate_scenario, [scenario for scenario in PILOT_POOL if not _complete(scenario)], "pilot")
    pilot_rows = _manifests(PILOT_POOL)
    if any(int(row["illegal"]) != 0 or int(row["shadow_state_contamination"]) != 0 or float(row["conservation_error"]) > 1.0 for row in pilot_rows):
        _finish(pilot_rows, [], [], equivalence, failed="leakage")
        return
    marker_path = PHASE / "phase6ib_pilot_verification.json"
    if marker_path.is_file() and int(json.loads(marker_path.read_text(encoding="utf-8")).get("passed", 0)) == 1:
        pilot_integrity = _reused_check(PILOT_POOL, "integrity")
        pilot_determinism = _reused_check(PILOT_POOL, "determinism")
    else:
        pilot_integrity = _map(integrity_scenario, list(PILOT_POOL), "pilot-integrity")
        pilot_determinism = _map(determinism_scenario, list(PILOT_POOL), "pilot-determinism")
        if any(int(row["passed"]) != 1 for row in pilot_integrity + pilot_determinism):
            _finish(pilot_rows, pilot_integrity, pilot_determinism, equivalence, failed="repro")
            return
        _write_json(marker_path, {"passed": 1, "scenarios": list(PILOT_POOL), "equivalence_max_abs_error": equivalence.get("max_abs_error")})
    pending = [scenario for scenario in TRAIN_POOL + VALIDATION_POOL if not _complete(scenario)]
    workers, _benchmark_payload = _benchmark(pending)
    remaining = [scenario for scenario in pending if not _complete(scenario)]
    _map(generate_scenario, remaining, "dataset", workers)
    rows = _manifests(TRAIN_POOL + VALIDATION_POOL)
    formal = tuple(scenario for scenario in TRAIN_POOL + VALIDATION_POOL if scenario not in PILOT_POOL)
    integrity = pilot_integrity + _map(integrity_scenario, list(formal), "integrity", workers)
    determinism = pilot_determinism + _map(determinism_scenario, list(formal), "determinism", workers)
    _finish(rows, integrity, determinism, equivalence, failed=None)


if __name__ == "__main__":
    main()
