"""One fresh process. It loads the manifest itself and does not inherit a training process."""
from __future__ import annotations

import argparse
import json

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT
from wt.benchmark_v2.phase6i_priority_tag_reward.actor import load_formal_actor
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR, assert_dev_scenario
from wt.benchmark_v2.phase6i_priority_tag_replay_repro.trace import trace_episode

from .context import apply_manifest, code_hashes, file_sha256, manifest_is_explicit
from .protocol import FORMAL_ACTOR_SHA256, SCENARIO_ID


def _public_step(step: dict) -> dict:
    return {
        "t": int(step["t"]),
        "obs_sha256": step["obs_sha256"],
        "mu": [float(value) for value in step["mu"].reshape(-1)],
        "desired": [float(value) for value in step["desired"].reshape(-1)],
        "projected": [float(value) for value in step["projected"].reshape(-1)],
        "executed": [float(value) for value in step["executed"].reshape(-1)],
        "position": [float(value) for value in step["position"].reshape(-1)],
        "q_bits": [float(value) for value in step["q_bits"].reshape(-1)],
        "association": [int(value) for value in step["association"].reshape(-1)],
        "served": float(step["served"]),
        "dropped": float(step["dropped"]),
        "snap": float(step["snap"]),
        "payload": float(step["payload"]),
        "path_increment": float(step["path_increment"]),
        "ledger_gap": float(step["ledger_gap"]),
        "sinr": float(step["sinr"]),
        "constraint_status": step["constraint_status"],
    }


def _public_episode(episode: dict) -> dict:
    return {key: float(value) if key not in {"variant"} else value for key, value in episode.items()}


def run_child(manifest_path: str, output_path: str) -> None:
    manifest = json.loads(open(manifest_path, encoding="utf-8").read())
    if not manifest_is_explicit(manifest):
        raise RuntimeError("refusing a manifest that does not keep the historical seed role")
    assert_dev_scenario(int(SCENARIO_ID))
    current_hashes = code_hashes()
    if current_hashes != manifest["code_hashes"]:
        raise RuntimeError("code hashes changed after the manifest was written")
    if file_sha256(FORMAL_ACTOR) != FORMAL_ACTOR_SHA256:
        raise RuntimeError("formal checkpoint hash changed")
    apply_manifest(manifest)
    actor = load_formal_actor(FORMAL_ACTOR, LOG_STD_INIT)
    actor.eval()
    if actor.training:
        raise RuntimeError("actor left eval mode")
    applied = apply_manifest(manifest)
    if int(applied["torch_num_threads"]) != int(manifest["torch_num_threads"]):
        raise RuntimeError("torch thread count did not stick")
    if applied["deterministic_algorithms"] is not True:
        raise RuntimeError("deterministic algorithms did not stick")
    if int(applied["torch_initial_seed"]) != int(manifest["global_rng"]["applied_seed"]):
        raise RuntimeError("torch seed did not stick")
    trace = trace_episode(actor, instrumented=False)
    payload = {
        "applied": applied,
        "actor_training": bool(actor.training),
        "deterministic_actions": True,
        "reset": {
            "t": int(trace["reset"]["t"]),
            "payload": float(trace["reset"]["payload"]),
            "q_sha256": trace["reset"]["q_sha256"],
            "position_sha256": trace["reset"]["position_sha256"],
        },
        "steps": [_public_step(step) for step in trace["steps"]],
        "episode": _public_episode(trace["episode"]),
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_child(args.manifest, args.output)


if __name__ == "__main__":
    main()
