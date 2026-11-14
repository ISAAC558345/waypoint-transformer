"""Load the frozen M1 contract."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "experiments" / "active_model_identification_m1.json"
M11_CONFIG_PATH = ROOT / "configs" / "experiments" / "active_model_identification_m11.json"
M12_CONFIG_PATH = ROOT / "configs" / "experiments" / "active_model_identification_m12.json"
M13_CONFIG_PATH = ROOT / "configs" / "experiments" / "active_model_identification_m13.json"


@lru_cache(maxsize=1)
def load_spec() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_m11_spec() -> dict:
    return json.loads(M11_CONFIG_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_m12_spec() -> dict:
    return json.loads(M12_CONFIG_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_m13_spec() -> dict:
    return json.loads(M13_CONFIG_PATH.read_text(encoding="utf-8"))


def decision_count(spec: dict | None = None) -> int:
    cfg = spec or load_spec()
    mission = int(cfg["time"]["mission_s"])
    step = int(cfg["time"]["decision_s"])
    if mission % step != 0:
        raise ValueError("mission length must be a multiple of the decision interval")
    return mission // step
