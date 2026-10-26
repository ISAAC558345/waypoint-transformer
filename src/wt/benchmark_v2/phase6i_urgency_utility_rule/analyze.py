"""Pareto and lambda breakpoints on existing pilot labels. No new rollouts."""
from __future__ import annotations

from dataclasses import dataclass

from .protocol import EPS_BITS, PAYLOAD_SCALE


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    payload_bits: float
    urgent_bits: float
    assignment: str

    @property
    def payload_term(self) -> float:
        return float(self.payload_bits) / float(PAYLOAD_SCALE)

    @property
    def urgent_term(self) -> float:
        return float(self.urgent_bits) / float(PAYLOAD_SCALE)


def dominates(left: Candidate, right: Candidate, eps_bits: float = EPS_BITS) -> bool:
    """Left dominates right when it is no worse and strictly better beyond one bit."""
    payload_gap = float(left.payload_bits) - float(right.payload_bits)
    urgent_gap = float(left.urgent_bits) - float(right.urgent_bits)
    not_worse = payload_gap >= -float(eps_bits) and urgent_gap >= -float(eps_bits)
    strictly_better = payload_gap > float(eps_bits) or urgent_gap > float(eps_bits)
    return bool(not_worse and strictly_better)


def pareto_front(candidates: list[Candidate], eps_bits: float = EPS_BITS) -> list[Candidate]:
    front = []
    for candidate in candidates:
        if any(dominates(other, candidate, eps_bits) for other in candidates if other.candidate_id != candidate.candidate_id):
            continue
        front.append(candidate)
    return sorted(front, key=lambda item: (-item.payload_bits, -item.urgent_bits, item.candidate_id))


def payload_tied(candidates: list[Candidate], eps_bits: float = EPS_BITS) -> list[Candidate]:
    best = max(float(item.payload_bits) for item in candidates)
    return [item for item in candidates if abs(float(item.payload_bits) - best) <= float(eps_bits)]


def urgent_tied(candidates: list[Candidate], eps_bits: float = EPS_BITS) -> list[Candidate]:
    best = max(float(item.urgent_bits) for item in candidates)
    return [item for item in candidates if abs(float(item.urgent_bits) - best) <= float(eps_bits)]


def payload_first(candidates: list[Candidate], eps_bits: float = EPS_BITS) -> Candidate:
    """Max payload, then max urgent service, then the smaller candidate id."""
    pool = payload_tied(candidates, eps_bits)
    return max(pool, key=lambda item: (float(item.urgent_bits), -int(item.candidate_id)))


def payload_id_only(candidates: list[Candidate], eps_bits: float = EPS_BITS) -> Candidate:
    """Max payload with the smaller candidate id. Urgent service is ignored."""
    pool = payload_tied(candidates, eps_bits)
    return min(pool, key=lambda item: int(item.candidate_id))


def urgent_first(candidates: list[Candidate], eps_bits: float = EPS_BITS) -> Candidate:
    """Max urgent service, then the payload closest to the best, then the smaller id."""
    pool = urgent_tied(candidates, eps_bits)
    return max(pool, key=lambda item: (float(item.payload_bits), -int(item.candidate_id)))


def material_regret_bits(value_bits: float, reference_bits: float, eps_bits: float = EPS_BITS) -> float:
    """Return 0 when the two payloads match within one bit."""
    gap = float(value_bits) - float(reference_bits)
    if abs(gap) <= float(eps_bits):
        return 0.0
    return float(gap)


def _slope(left: Candidate, right: Candidate, eps_bits: float) -> float | None:
    urgent_gap = float(right.urgent_bits) - float(left.urgent_bits)
    payload_gap = float(left.payload_bits) - float(right.payload_bits)
    if urgent_gap <= float(eps_bits):
        return None
    if payload_gap <= float(eps_bits):
        return 0.0
    return float(payload_gap / urgent_gap)


def lambda_segments(candidates: list[Candidate], eps_bits: float = EPS_BITS) -> list[dict]:
    """Convex chain from the payload-first candidate toward higher urgent service."""
    front = pareto_front(candidates, eps_bits)
    start = payload_first(candidates, eps_bits)
    ordered = sorted(
        (item for item in front if float(item.urgent_bits) > float(start.urgent_bits) + float(eps_bits)),
        key=lambda item: (float(item.urgent_bits), -float(item.payload_bits), int(item.candidate_id)),
    )
    chain: list[Candidate] = [start]
    for candidate in ordered:
        while len(chain) >= 2:
            previous = _slope(chain[-2], chain[-1], eps_bits)
            proposed = _slope(chain[-1], candidate, eps_bits)
            if proposed is None:
                break
            if previous is None or proposed <= previous:
                chain.pop()
                continue
            break
        if _slope(chain[-1], candidate, eps_bits) is None:
            continue
        chain.append(candidate)
    best_payload = max(float(item.payload_bits) for item in candidates)
    payload_choice = start
    slopes = [float(_slope(left, right, eps_bits) or 0.0) for left, right in zip(chain, chain[1:])]
    bounds = [0.0, *slopes]
    segments = []
    for index, candidate in enumerate(chain):
        low = float(bounds[index])
        high = None if index == len(chain) - 1 else float(bounds[index + 1])
        regret = material_regret_bits(candidate.payload_bits, best_payload, eps_bits)
        urgent_gain = float(candidate.urgent_bits) - float(payload_choice.urgent_bits)
        segments.append(
            {
                "segment_index": int(index),
                "lambda_low": low,
                "lambda_high": high,
                "lambda_high_unbounded": int(high is None),
                "candidate_id": int(candidate.candidate_id),
                "assignment": candidate.assignment,
                "payload_bits": float(candidate.payload_bits),
                "urgent_service_bits": float(candidate.urgent_bits),
                "payload_term": float(candidate.payload_term),
                "urgent_service_term": float(candidate.urgent_term),
                "payload_regret_bits": float(regret),
                "urgent_gain_bits_vs_payload_first": float(urgent_gain),
                "material_payload_loss": int(regret < 0.0),
            }
        )
    return segments


def positive_cross_count(candidates: list[Candidate], eps_bits: float = EPS_BITS) -> int:
    """Count unordered pairs that trade a real payload loss for a real urgent gain."""
    count = 0
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            urgent_gap = float(right.urgent_bits) - float(left.urgent_bits)
            payload_gap = float(left.payload_bits) - float(right.payload_bits)
            if urgent_gap < 0.0:
                urgent_gap = -urgent_gap
                payload_gap = -payload_gap
            if urgent_gap > float(eps_bits) and payload_gap > float(eps_bits):
                count += 1
    return int(count)


def first_material_lambda(segments: list[dict]) -> float | None:
    for segment in segments:
        if int(segment["material_payload_loss"]) == 1:
            return float(segment["lambda_low"])
    return None


def state_analysis(candidates: list[Candidate], eps_bits: float = EPS_BITS) -> dict:
    if len(candidates) < 2:
        raise RuntimeError("a state needs at least two sampled candidates")
    front = pareto_front(candidates, eps_bits)
    best_payload = max(float(item.payload_bits) for item in candidates)
    best_urgent = max(float(item.urgent_bits) for item in candidates)
    tied = payload_tied(candidates, eps_bits)
    first = payload_first(candidates, eps_bits)
    id_only = payload_id_only(candidates, eps_bits)
    urgent = urgent_first(candidates, eps_bits)
    raw_regret = float(urgent.payload_bits) - float(best_payload)
    material = material_regret_bits(urgent.payload_bits, best_payload, eps_bits)
    free_gain = float(first.urgent_bits) - float(id_only.urgent_bits)
    tie_span = max(float(item.urgent_bits) for item in tied) - min(float(item.urgent_bits) for item in tied)
    pair_ties = 0
    pairs = 0
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            pairs += 1
            if abs(float(candidates[left].payload_bits) - float(candidates[right].payload_bits)) <= float(eps_bits):
                pair_ties += 1
    segments = lambda_segments(candidates, eps_bits)
    crosses = positive_cross_count(candidates, eps_bits)
    return {
        "candidate_count": int(len(candidates)),
        "pareto_count": int(len(front)),
        "pareto": front,
        "payload_tie_count": int(len(tied)),
        "payload_pair_tie_fraction": float(pair_ties / pairs),
        "payload_first_candidate_id": int(first.candidate_id),
        "payload_id_only_candidate_id": int(id_only.candidate_id),
        "urgent_first_candidate_id": int(urgent.candidate_id),
        "payload_first_urgent_bits": float(first.urgent_bits),
        "urgent_first_payload_bits": float(urgent.payload_bits),
        "best_payload_bits": float(best_payload),
        "best_urgent_bits": float(best_urgent),
        "raw_top1_payload_regret_bits": float(raw_regret),
        "material_top1_payload_regret_bits": float(material),
        "urgent_regret_of_payload_first_bits": float(first.urgent_bits) - float(best_urgent),
        "free_urgent_gain_bits": float(free_gain),
        "payload_tie_urgent_span_bits": float(tie_span),
        "urgent_opt_inside_payload_tie": int(abs(raw_regret) <= float(eps_bits)),
        "segments": segments,
        "segment_count": int(len(segments)),
        "material_segment_count": int(sum(int(item["material_payload_loss"]) for item in segments)),
        "first_material_lambda": first_material_lambda(segments),
        "positive_cross_count": int(crosses),
    }


def leave_one_out(records: list[dict], held_out: int) -> dict:
    """Describe one state without removing it from the primary sample."""
    kept = [item for item in records if int(item["scenario_id"]) != int(held_out)]
    if len(kept) == len(records):
        raise RuntimeError(f"held-out scenario {held_out} is not in the sample")
    return {
        "held_out_scenario": int(held_out),
        "removed_from_primary_gate": False,
        "full_count": int(len(records)),
        "without_count": int(len(kept)),
        "full_mean_material_regret_bits": _mean(item["material_top1_payload_regret_bits"] for item in records),
        "without_mean_material_regret_bits": _mean(item["material_top1_payload_regret_bits"] for item in kept),
        "full_material_loss_states": int(sum(float(item["material_top1_payload_regret_bits"]) < 0.0 for item in records)),
        "without_material_loss_states": int(sum(float(item["material_top1_payload_regret_bits"]) < 0.0 for item in kept)),
        "full_urgent_opt_inside_tie": int(sum(int(item["urgent_opt_inside_payload_tie"]) for item in records)),
        "without_urgent_opt_inside_tie": int(sum(int(item["urgent_opt_inside_payload_tie"]) for item in kept)),
    }


def _mean(values) -> float:
    pool = [float(item) for item in values]
    if not pool:
        raise RuntimeError("mean of an empty sample")
    return float(sum(pool) / len(pool))
