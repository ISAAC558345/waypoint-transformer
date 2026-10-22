"""Level-3-B SINR coupling config. Isolated from locked L3-A."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class L3BConfig:
    num_uavs: int = 6
    num_buoys: int = 12
    episode_horizon: int = 48
    service_slots: int = 3
    area_width: float = 160.0
    area_height: float = 80.0
    max_speed: float = 22.0
    delta_t: float = 1.0
    service_radius_m: float = 16.0
    queue_capacity_blocks: int = 4
    block_payload_bits: float = 1.0e6
    seed: int = 0
    option_hold: int = 15
    num_hot: int = 3
    bandwidth_hz: float = 1.0e6
    tx_power: float = 1.0
    noise_power: float = 1.0e-3
    path_loss_exp: float = 2.2
    min_link_m: float = 1.0
    kappa_same_cluster: float = 1.0
    kappa_cross_cluster: float = 0.08

    @property
    def step_length_m(self) -> float:
        return float(self.max_speed * self.delta_t)


__all__ = ["L3BConfig"]
