"""Registered formal references. This module uses only the standard library."""
from __future__ import annotations

import json
from pathlib import Path


REGISTERED = (322000, 322001)
POLICY = "scheme_c"
STOP_TIME = 48
PROCESS_SEED_OFFSET = 1_000_000
INSTRUCTION_COMMIT = "204dd3309115dfd369236c408f7e59306a0871d4"
EPISODE_FIELDS = (
    "scenario_id",
    "policy",
    "payload_bits",
    "served_bits",
    "steps",
    "process_seed",
    "reset_seed",
    "decision_times",
    "scheme_c_calls",
    "complete",
    "terminal_time",
)


class ReferenceRefusal(ValueError):
    """The scenario has no usable registered reference. Callers return exit code 2."""


def episode_path(scenario: int) -> str:
    return f"results/reproducibility/formal_scheme_c/episodes/{int(scenario)}_scheme_c.json"


def config_path(scenario: int) -> str:
    return f"results/reproducibility/formal_scheme_c/configs/{int(scenario)}_scheme_c.json"


def default_output(scenario: int) -> str:
    return f"results/reproducibility/scenario_{int(scenario)}"


def resolve(scenario: int) -> dict:
    """Return the shared binding. Unregistered scenarios carry no reference path."""
    scenario_id = int(scenario)
    accepted = scenario_id in REGISTERED
    return {
        "scenario_id": scenario_id,
        "policy": POLICY,
        "registered": accepted,
        "run_accepted": accepted,
        "process_seed": PROCESS_SEED_OFFSET + scenario_id,
        "reset_seed": scenario_id,
        "stop_time": STOP_TIME,
        "default_output": default_output(scenario_id),
        "reference_episode": episode_path(scenario_id) if accepted else None,
        "reference_config": config_path(scenario_id) if accepted else None,
        "reference_source": "results/reproducibility/formal_scheme_c",
    }


def require_reference(root: Path, scenario: int) -> tuple[dict, dict]:
    """Load one registered episode. A missing or mismatched file is refused before a run."""
    spec = resolve(scenario)
    if not spec["run_accepted"] or spec["reference_episode"] is None or spec["reference_config"] is None:
        raise ReferenceRefusal("scenario is not in the registered reference set")
    episode_file = Path(root) / spec["reference_episode"]
    config_file = Path(root) / spec["reference_config"]
    if not episode_file.is_file() or not config_file.is_file():
        raise ReferenceRefusal("registered reference file is missing")
    episode = json.loads(episode_file.read_text(encoding="utf-8"))
    missing = [key for key in EPISODE_FIELDS if key not in episode or episode[key] is None]
    if missing:
        raise ReferenceRefusal("registered reference is missing " + ",".join(missing))
    if episode["scenario_id"] != spec["scenario_id"] or episode["policy"] != spec["policy"]:
        raise ReferenceRefusal("registered reference identity does not match the scenario")
    if int(episode["process_seed"]) != int(spec["process_seed"]) or int(episode["reset_seed"]) != int(spec["reset_seed"]):
        raise ReferenceRefusal("registered reference seeds do not match the scenario")
    if len(episode["steps"]) != int(spec["stop_time"]):
        raise ReferenceRefusal("registered reference is not a complete 48-step episode")
    return spec, episode