"""HTW-C Phase-1 Task-Waypoint representation (goal, subgoal, horizon)."""

from .adapter import CONDITIONED_UAV_DIM, attach_task_waypoint
from .conditioned_actor import GoalConditionedActorCritic
from .contract import DEFAULT_HORIZON, TaskWaypointV2, validate_tw
from .labels import collect_planner_window, labels_from_planner_window
from .predictor import TaskWaypointV2Predictor, TWPredictorConfig, task_waypoint_v2_loss

__all__ = [
    "CONDITIONED_UAV_DIM",
    "DEFAULT_HORIZON",
    "GoalConditionedActorCritic",
    "TWPredictorConfig",
    "TaskWaypointV2",
    "TaskWaypointV2Predictor",
    "attach_task_waypoint",
    "collect_planner_window",
    "labels_from_planner_window",
    "task_waypoint_v2_loss",
    "validate_tw",
]
