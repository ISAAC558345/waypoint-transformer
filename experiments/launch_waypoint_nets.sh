#!/usr/bin/env bash

config="experiments/config/d4rl/goal_network.cfg"
declare -a envs=("antmaze-umaze-v2" "antmaze-umaze-diverse-v2" "antmaze-medium-diverse-v2" "antmaze-medium-play-v2" "antmaze-large-diverse-v2" "antmaze-large-play-v2")
# declare -a envs=("kitchen-complete-v0" "kitchen-mixed-v0" "kitchen-partial-v0")
# declare -a envs=("hopper-medium-replay-v2" "halfcheetah-medium-replay-v2" "walker2d-medium-replay-v2" "hopper-medium-v2" "halfcheetah-medium-v2" "walker2d-medium-v2")
seeds=42
use_gpu=true
for env in "${envs[@]}"; do
	for seed in $(seq 42 $((seeds))); do
	      PYTHONPATH=src python -m wt.train --configs "$config" --env_name "$env" --seed "$seed" --use_gpu --train_goal_net &
	done
done
wait
