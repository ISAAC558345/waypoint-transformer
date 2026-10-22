"""Train remaining unfinished 9.4.5 seeds one process at a time. Frozen NPZ only."""
from __future__ import annotations

import os
import subprocess

from wt.htw_phase941.protocol import MODEL_SEEDS, dump_frozen
from wt.htw_phase945.protocol import OUTPUT, ROOT

PYTHON = (ROOT.parent / "decision-transformer-master" / ".venv-dt-gym" / "python.exe").resolve()


def remaining() -> list[tuple[str, int]]:
    jobs = []
    for encoder in ("deepsets", "hgnn"):
        for seed in MODEL_SEEDS:
            if not (OUTPUT / f"{encoder}_seed{seed}.pt").exists():
                jobs.append((encoder, int(seed)))
    return jobs


def _run(item: tuple[str, int]) -> dict:
    encoder, seed = item
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    proc = subprocess.run(
        [str(PYTHON), "-m", "wt.htw_phase945.train_one", encoder, str(seed)],
        cwd=str(ROOT),
        env=env,
    )
    return {"encoder": encoder, "seed": seed, "returncode": int(proc.returncode)}


def main() -> None:
    jobs = remaining()
    print(f"remaining jobs={jobs}", flush=True)
    workers = 1
    results = []
    if not jobs:
        dump_frozen(OUTPUT / "phase945_model_parallelism_audit.json", {"n_jobs": 0, "workers": 0, "note": "all checkpoints present"})
        return
    for job in jobs:
        row = _run(job)
        results.append(row)
        print(row, flush=True)
        if int(row["returncode"]) != 0:
            raise RuntimeError(str(row))
    dump_frozen(
        OUTPUT / "phase945_model_parallelism_audit.json",
        {
            "n_jobs": len(jobs),
            "workers": workers,
            "results": results,
            "shared_utility_dataset": True,
            "cpu_parallel": False,
            "serial_reason": "windows_pytorch_access_violation_on_two_cpu_jobs",
        },
    )


if __name__ == "__main__":
    main()
