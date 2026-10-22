"""Post-training causal evaluation sequence for the already-frozen Round-22."""
from __future__ import annotations

import json

from .analysis import run_offline_analysis
from .closed_loop import run_fresh_development, run_validation_closed_loop
from .figures import render_figures
from .postprocess import run_postprocess


def run_post_training_sequence() -> dict:
    offline = run_offline_analysis()
    validation_gate = run_validation_closed_loop()
    fresh = run_fresh_development(validation_gate)
    post = run_postprocess(validation_gate, fresh is not None)
    figures = render_figures()
    return {"offline_checkpoint_count": len(offline["run_summary"]), "validation_gate": validation_gate["gate"], "fresh_dev_accessed": fresh is not None, "mu_wp_gate": post["mu_gate"]["gate"], "figure_file_count": len(figures)}


def main() -> None:
    print(json.dumps(run_post_training_sequence(), sort_keys=True))


if __name__ == "__main__":
    main()
