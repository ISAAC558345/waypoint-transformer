"""Model-independent legal assignment candidate pool. Frozen size/seed before utilities."""
from __future__ import annotations

import hashlib

import numpy as np

from wt.htw_phase9.const import MAX_UAVS
from wt.htw_phase9.features import conflict_rate
from wt.l3b.env import L3BEnv
from wt.l3b.policies import assign_split, exclusive_matching, myopic_assignment, oracle_option_assignment

from .protocol import POOL_SEED, POOL_SIZE


def assignment_hash(assignment: np.ndarray) -> str:
    raw = np.asarray(assignment, dtype=np.int64).tobytes()
    return hashlib.sha256(raw).hexdigest()


def is_legal(assignment: np.ndarray, num_buoys: int) -> bool:
    values = np.asarray(assignment, dtype=np.int64)
    if values.shape[0] != MAX_UAVS:
        return False
    if np.any(values >= int(num_buoys)):
        return False
    return float(conflict_rate(values, int(num_buoys))) <= 1.0e-12


def legalize(assignment: np.ndarray, num_buoys: int) -> np.ndarray:
    out = np.asarray(assignment, dtype=np.int64).copy()
    taken: set[int] = set()
    for uav, buoy in enumerate(out):
        if int(buoy) < 0:
            continue
        if int(buoy) >= int(num_buoys) or int(buoy) in taken:
            out[uav] = -1
            continue
        taken.add(int(buoy))
    return out


def _pad(values: np.ndarray) -> np.ndarray:
    out = np.full(MAX_UAVS, -1, dtype=np.int64)
    n = min(MAX_UAVS, int(values.shape[0]))
    out[:n] = np.asarray(values, dtype=np.int64)[:n]
    return out


def _unique_add(pool: list[np.ndarray], item: np.ndarray, num_buoys: int) -> None:
    cand = legalize(_pad(item), num_buoys)
    if not is_legal(cand, num_buoys):
        return
    key = tuple(int(x) for x in cand.tolist())
    if key in {tuple(int(x) for x in row.tolist()) for row in pool}:
        return
    pool.append(cand)


def _one_edge(ref: np.ndarray, env: L3BEnv) -> list[np.ndarray]:
    rows = []
    taken = {int(b) for b in ref if int(b) >= 0}
    for uav in range(env.config.num_uavs):
        cur = np.asarray(ref, dtype=np.int64).copy()
        if int(cur[uav]) >= 0:
            nxt = cur.copy()
            nxt[uav] = -1
            rows.append(nxt)
        for buoy in range(env.config.num_buoys):
            if buoy in taken and buoy != int(cur[uav]):
                continue
            if buoy == int(cur[uav]):
                continue
            nxt = cur.copy()
            nxt[uav] = int(buoy)
            rows.append(nxt)
            break
    return rows


def _two_edge(ref: np.ndarray, env: L3BEnv) -> list[np.ndarray]:
    rows = []
    active = [i for i, b in enumerate(ref) if int(b) >= 0]
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            nxt = np.asarray(ref, dtype=np.int64).copy()
            a, b = active[i], active[j]
            nxt[a], nxt[b] = nxt[b], nxt[a]
            rows.append(nxt)
            if len(rows) >= 4:
                return rows
    return rows


def _idle_variants(ref: np.ndarray, env: L3BEnv, n_idle: int) -> np.ndarray:
    nxt = np.asarray(ref, dtype=np.int64).copy()
    active = [i for i, b in enumerate(nxt) if int(b) >= 0]
    for uav in active[:n_idle]:
        nxt[uav] = -1
    return nxt


def _fill_busy(ref: np.ndarray, env: L3BEnv) -> np.ndarray:
    nxt = np.asarray(ref, dtype=np.int64).copy()
    taken = {int(b) for b in nxt if int(b) >= 0}
    leftover = env.remaining_arrivals()
    for uav in range(env.config.num_uavs):
        if int(nxt[uav]) >= 0:
            continue
        best, score = -1, -1.0e18
        for buoy in range(env.config.num_buoys):
            if buoy in taken:
                continue
            work = float(max(env.queue[buoy], leftover[buoy]))
            if work <= 1.0e-9:
                continue
            dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[buoy]))
            val = work / (dist + 1.0)
            if val > score:
                best, score = buoy, val
        if best >= 0:
            nxt[uav] = best
            taken.add(best)
    return nxt


def _competition_alt(env: L3BEnv) -> np.ndarray:
    scores = []
    for uav in range(env.config.num_uavs):
        best, val = -1, -1.0e18
        for buoy in range(env.config.num_buoys):
            if env.queue[buoy] <= 1.0e-9:
                continue
            dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[buoy]))
            s = float(env.queue[buoy]) / (dist + 1.0)
            if s > val:
                best, val = buoy, s
        scores.append((uav, best, val))
    by_buoy: dict[int, list] = {}
    for uav, buoy, val in scores:
        if buoy < 0:
            continue
        by_buoy.setdefault(int(buoy), []).append((val, uav))
    out = np.full(env.config.num_uavs, -1, dtype=np.int64)
    taken: set[int] = set()
    for buoy, items in by_buoy.items():
        items.sort(reverse=True)
        winner = items[0][1]
        out[winner] = buoy
        taken.add(buoy)
        for _, uav in items[1:]:
            best, score = -1, -1.0e18
            for other in range(env.config.num_buoys):
                if other in taken or env.queue[other] <= 1.0e-9:
                    continue
                dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[other]))
                s = float(env.queue[other]) / (dist + 1.0)
                if s > score:
                    best, score = other, s
            if best >= 0:
                out[uav] = best
                taken.add(best)
    return out


def _interference_alt(env: L3BEnv, n_near: int, n_far: int) -> np.ndarray:
    return assign_split(env, n_near, n_far)


def _random_legal(env: L3BEnv, rng: np.random.Generator) -> np.ndarray:
    m, k = env.config.num_uavs, env.config.num_buoys
    out = np.full(m, -1, dtype=np.int64)
    buoys = rng.permutation(k).tolist()
    taken: set[int] = set()
    for uav in rng.permutation(m):
        if rng.random() < 0.30:
            continue
        for buoy in buoys:
            if buoy in taken:
                continue
            out[int(uav)] = int(buoy)
            taken.add(int(buoy))
            break
    return out


def generate_pool(env: L3BEnv, seed_key: int) -> np.ndarray:
    rng = np.random.default_rng(POOL_SEED + int(seed_key) * 10007)
    ref = _pad(oracle_option_assignment(env))
    pool: list[np.ndarray] = []
    _unique_add(pool, ref, env.config.num_buoys)
    _unique_add(pool, exclusive_matching(env), env.config.num_buoys)
    _unique_add(pool, myopic_assignment(env), env.config.num_buoys)
    for row in _one_edge(ref, env):
        _unique_add(pool, row, env.config.num_buoys)
        if len(pool) >= POOL_SIZE:
            break
    for row in _two_edge(ref, env):
        _unique_add(pool, row, env.config.num_buoys)
    _unique_add(pool, _competition_alt(env), env.config.num_buoys)
    _unique_add(pool, _interference_alt(env, 2, 1), env.config.num_buoys)
    _unique_add(pool, _interference_alt(env, 3, 1), env.config.num_buoys)
    _unique_add(pool, _interference_alt(env, 1, 2), env.config.num_buoys)
    _unique_add(pool, _idle_variants(ref, env, 2), env.config.num_buoys)
    _unique_add(pool, _idle_variants(ref, env, 4), env.config.num_buoys)
    _unique_add(pool, np.full(MAX_UAVS, -1, dtype=np.int64), env.config.num_buoys)
    _unique_add(pool, _fill_busy(ref, env), env.config.num_buoys)
    guard = 0
    while len(pool) < POOL_SIZE and guard < 400:
        _unique_add(pool, _random_legal(env, rng), env.config.num_buoys)
        guard += 1
    if len(pool) < POOL_SIZE:
        raise RuntimeError(f"could not fill candidate pool: {len(pool)}")
    return np.stack(pool[:POOL_SIZE], axis=0)


__all__ = ["assignment_hash", "generate_pool", "is_legal", "legalize"]
