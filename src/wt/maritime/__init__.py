"""Versioned maritime environment, offline data, and Waypoint-label utilities."""

from .arrival import (
    ARRIVAL_TRACE_SCHEMA_VERSION,
    arrival_trace_id,
    derive_scenario_trace_seed,
    materialize_arrival_trace,
)
from .baselines import (
    BaselinePolicy,
    MaxQueuePolicy,
    NearestBuoyPolicy,
    RandomValidPolicy,
    QueueRateEnergyPolicy,
    UrgencyRateEnergyPolicy,
    available_baseline_policies,
    baseline_policy_metadata,
    create_baseline_policy,
)
from .channel import (
    ChannelMetrics,
    compute_channel_metrics,
    theoretical_max_rates,
    theoretical_maximum_rates,
    theoretical_max_single_buoy_rate_bps,
    theoretical_max_total_rate_bps,
)
from .config import MaritimeEnvConfig
from .dataset_io import (
    assert_roundtrip_exact,
    deterministic_episode_split,
    load_dataset,
    save_dataset,
    split_composition,
    validate_dataset,
    validate_saved_dataset,
)
from .env import MaritimeEnv
from .k_step_audit import (
    audit_k_steps,
    enumerate_k_step_pairs,
    get_k_step_state_pair,
)
from .metrics import compute_episode_metrics, summarize_episode_metrics
from .trajectory import EpisodeTrajectory, EpisodeTrajectoryBuilder
from .trajectory_v1 import MaritimeEpisodeBuilderV1, MaritimeEpisodeV1

__all__ = [
    "BaselinePolicy",
    "ARRIVAL_TRACE_SCHEMA_VERSION",
    "ChannelMetrics",
    "EpisodeTrajectory",
    "EpisodeTrajectoryBuilder",
    "MaxQueuePolicy",
    "MaritimeEnv",
    "MaritimeEnvConfig",
    "NearestBuoyPolicy",
    "RandomValidPolicy",
    "QueueRateEnergyPolicy",
    "MaritimeEpisodeBuilderV1",
    "MaritimeEpisodeV1",
    "UrgencyRateEnergyPolicy",
    "assert_roundtrip_exact",
    "audit_k_steps",
    "available_baseline_policies",
    "arrival_trace_id",
    "baseline_policy_metadata",
    "compute_channel_metrics",
    "compute_episode_metrics",
    "create_baseline_policy",
    "deterministic_episode_split",
    "derive_scenario_trace_seed",
    "enumerate_k_step_pairs",
    "get_k_step_state_pair",
    "load_dataset",
    "materialize_arrival_trace",
    "save_dataset",
    "split_composition",
    "summarize_episode_metrics",
    "theoretical_max_rates",
    "theoretical_maximum_rates",
    "theoretical_max_single_buoy_rate_bps",
    "theoretical_max_total_rate_bps",
    "validate_dataset",
    "validate_saved_dataset",
]
