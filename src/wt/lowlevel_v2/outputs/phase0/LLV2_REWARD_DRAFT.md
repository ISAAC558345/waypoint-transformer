# LL-V2 reward draft

Coefficients are **not** frozen. Auto-search is forbidden. One design freeze is required before any training.

{
  "primary": "actual payload increment / normalization scale",
  "r_progress": "service progress increment",
  "r_near": "soft near-collision cost",
  "r_project": "SafetyProjector intervention penalty",
  "r_residual": "residual action magnitude penalty",
  "formula": "r = r_payload + lp*r_progress - ln*r_near - lproj*r_project - lres*r_residual",
  "lambda_progress": null,
  "lambda_near": null,
  "lambda_project": null,
  "lambda_residual": null,
  "auto_search_forbidden": true,
  "design_freeze_required_before_training": true,
  "evaluation_uses_train_reward": false,
  "evaluation_metrics": [
    "payload",
    "completed_blocks",
    "collision_count",
    "minimum_separation",
    "flight_distance",
    "queue",
    "drop"
  ],
  "extra_interference_penalty": false,
  "reason_no_interference_penalty": "SINR->rate->payload already carries interference; do not add a second hand-designed penalty in v1"
}

Evaluation never uses the training reward. Flight distance is not energy.
