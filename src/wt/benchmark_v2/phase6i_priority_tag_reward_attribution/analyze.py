"""Paired age/tag summaries. Thresholds from the source experiment stay untouched."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag.protocol import BETA_TAG, DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_reward.criteria import paired_interval
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import TRAIN_SEEDS


PAIRED_METRICS = (
    "payload_bits",
    "served_tagged_bits",
    "served_ordinary_bits",
    "age_service_bits",
    "age_service_ordinary_bits",
    "age_service_tagged_bits",
    "tag_increment_service_bits",
    "combined_urgency_served_bits",
    "dropped_bits",
    "safety_violations",
    "constraint_failures",
    "sinr_mean",
    "interference_mean",
    "tagged_delta_buoy_entry_exit",
    "tagged_delta_shared_volume",
    "tagged_delta_shared_mix",
)


def _select(rows: list[dict], variant: str, seed: int) -> list[dict]:
    selected = [row for row in rows if row["variant"] == variant and int(row["model_seed"]) == int(seed)]
    selected.sort(key=lambda row: int(row["scenario_id"]))
    return selected


def _values(rows: list[dict], metric: str) -> np.ndarray:
    return np.asarray([float(row[metric]) for row in rows], dtype=np.float64)


def _pool_age(rows: list[dict], mean_key: str, bits_key: str) -> float:
    bits = np.asarray([float(row[bits_key]) for row in rows], dtype=np.float64)
    means = np.asarray([float(row[mean_key]) for row in rows], dtype=np.float64)
    total = float(np.sum(bits))
    if total <= float(DUST_BITS):
        return 0.0
    return float(np.sum(means * bits) / total)


def _clipping(rows: list[dict]) -> dict:
    tagged = float(sum(float(row["served_tagged_bits"]) for row in rows))
    full = float(sum(float(row["tagged_served_fully_saturated_bits"]) for row in rows))
    below = float(sum(float(row["tagged_served_below_full_boost_bits"]) for row in rows))
    increment = float(sum(float(row["tag_increment_service_bits"]) for row in rows))
    potential = float(BETA_TAG) * tagged
    retained = None if potential <= float(DUST_BITS) else float(increment / potential)
    return {
        "tagged_served_bits": tagged,
        "full_saturation_fraction": None if tagged <= float(DUST_BITS) else float(full / tagged),
        "below_full_boost_fraction": None if tagged <= float(DUST_BITS) else float(below / tagged),
        "boost_retained_fraction": retained,
        "boost_suppressed_fraction": None if retained is None else float(1.0 - retained),
        "tag_draw_rate_mean": float(np.mean([float(row["tag_draw_rate"]) for row in rows])),
        "served_age_mean_ordinary": _pool_age(rows, "served_age_mean_ordinary", "served_ordinary_bits"),
        "served_age_mean_tagged": _pool_age(rows, "served_age_mean_tagged", "served_tagged_bits"),
        "age_service_bits": float(sum(float(row["age_service_bits"]) for row in rows)),
        "tag_increment_service_bits": increment,
        "combined_urgency_served_bits": float(sum(float(row["combined_urgency_served_bits"]) for row in rows)),
    }


def _share(numerator: float, denominator: float) -> float | None:
    if abs(float(denominator)) <= float(DUST_BITS):
        return None
    return float(numerator) / float(denominator)


def summarize_attribution(rows: list[dict]) -> dict:
    baseline = _select(rows, "L0", 0)
    groups = []
    for variant in ("L1", "L2"):
        for seed in TRAIN_SEEDS:
            current = _select(rows, variant, int(seed))
            if len(current) != len(baseline):
                raise RuntimeError(f"{variant} seed {seed} does not match the L0 scenario count")
            intervals = {}
            for metric in PAIRED_METRICS:
                intervals[metric] = paired_interval(_values(current, metric) - _values(baseline, metric))
            age_mean = float(intervals["age_service_bits"]["mean"])
            tag_mean = float(intervals["tag_increment_service_bits"]["mean"])
            combined_mean = float(intervals["combined_urgency_served_bits"]["mean"])
            groups.append(
                {
                    "variant": variant,
                    "model_seed": int(seed),
                    "age_delta_mean": age_mean,
                    "tag_increment_delta_mean": tag_mean,
                    "tagged_delta_mean": float(intervals["served_tagged_bits"]["mean"]),
                    "ordinary_age_delta_mean": float(intervals["age_service_ordinary_bits"]["mean"]),
                    "combined_delta_mean": combined_mean,
                    "ordinary_share_of_combined_delta": _share(float(intervals["age_service_ordinary_bits"]["mean"]), combined_mean),
                    "tag_share_of_combined_delta": _share(tag_mean, combined_mean),
                    "intervals": intervals,
                    "clipping": _clipping(current),
                    "mechanism_sums": {
                        "tagged_delta_buoy_entry_exit": float(np.sum(_values(current, "tagged_delta_buoy_entry_exit"))),
                        "tagged_delta_shared_volume": float(np.sum(_values(current, "tagged_delta_shared_volume"))),
                        "tagged_delta_shared_mix": float(np.sum(_values(current, "tagged_delta_shared_mix"))),
                        "tagged_delta_split_gap_max": float(np.max(np.abs(_values(current, "tagged_delta_split_gap")))),
                    },
                }
            )
    return {
        "l0_clipping": _clipping(baseline),
        "groups": groups,
        "max_decomposition_gap_bits": float(max(float(row["decomposition_gap_bits"]) for row in rows)),
        "max_saved_gap_bits": float(max(float(row["max_saved_gap_bits"]) for row in rows)),
        "max_origin_split_gap_bits": float(max(abs(float(row["tagged_delta_split_gap"])) for row in rows)),
    }
