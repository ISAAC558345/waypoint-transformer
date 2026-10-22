"""
Run the training.

Source: RvS GitHub repository (https://github.com/scottemmons/rvs)
Edits by Anonymous Authors
"""

from __future__ import annotations

import json
import os
from typing import Optional, Union

import configargparse
from d4rl import offline_env
from gcsl import envs
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from wt import analyze_d4rl, dataset, policies, step, util, visualize, goal_net

args_filename = "args.json"
checkpoint_dir = "checkpoints"
rollout_dir = "env_steps_data"
wandb_project = "wt"


def log_args(
    args: configargparse.Namespace,
    wandb_logger: pl.loggers.wandb.WandbLogger,
) -> None:
    """Log arguments to a file in the wandb directory."""
    wandb_logger.log_hyperparams(args)

    args.wandb_entity = wandb_logger.experiment.entity
    args.wandb_project = wandb_logger.experiment.project
    args.wandb_run_id = wandb_logger.experiment.id
    args.wandb_path = wandb_logger.experiment.path

    out_directory = wandb_logger.experiment.dir
    args_file = os.path.join(out_directory, args_filename)
    with open(args_file, "w") as f:
        try:
            json.dump(args.__dict__, f)
        except AttributeError:
            json.dump(args, f)


def run_training(
    env: Union[step.GCSLToGym, offline_env.OfflineEnv],
    env_name: str,
    seed: int,
    wandb_logger: pl.loggers.wandb.WandbLogger,
    rollout_directory: Optional[str],
    unconditional_policy: bool,
    reward_conditioning: bool,
    cumulative_reward_to_go: bool,
    epochs: int,
    max_steps: int,
    train_time: str,
    hidden_size: int,
    depth: int,
    learning_rate: float,
    auto_tune_lr: bool,
    dropout_p: float,
    checkpoint_every_n_epochs: int,
    checkpoint_every_n_steps: int,
    checkpoint_time_interval: str,
    batch_size: int,
    val_frac: float,
    use_gpu: bool,
    train_goal_net: bool = False,
    waypoint_horizon: int = 30,
    context_length: int = 20,
    waypoint_context_length: int = 40,
    waypoint_hidden_size: int = 1024,
    reward_representation: Optional[str] = None,
    reward_episode_horizon: Optional[int] = None,
    goal_network_ckpt: Optional[str] = None,
) -> None:
    """Run the training with PyTorch Lightning and log to Weights & Biases."""
    if waypoint_horizon < 0:
        raise ValueError("waypoint_horizon must be non-negative.")
    if context_length <= 0 or waypoint_context_length <= 0:
        raise ValueError("Sequence context lengths must be positive.")
    if waypoint_hidden_size <= 0:
        raise ValueError("waypoint_hidden_size must be positive.")
    if reward_episode_horizon is not None and reward_episode_horizon <= 0:
        raise ValueError("reward_episode_horizon must be positive.")
    if (
        cumulative_reward_to_go
        and reward_representation not in (None, "cumulative")
    ):
        raise ValueError(
            "cumulative_reward_to_go conflicts with explicit "
            f"reward_representation={reward_representation!r}."
        )
    if cumulative_reward_to_go:
        reward_representation = "cumulative"
    if reward_conditioning:
        reward_representation = dataset.resolve_reward_representation(
            reward_representation,
            average_reward_to_go=not cumulative_reward_to_go,
        )

    data_module = dataset.create_data_module(
        env,
        env_name,
        rollout_directory,
        batch_size=batch_size,
        val_frac=val_frac,
        unconditional_policy=unconditional_policy,
        reward_conditioning=reward_conditioning,
        average_reward_to_go=not cumulative_reward_to_go,
        reward_representation=reward_representation,
        seed=seed,
        train_goal_net=train_goal_net,
        K=waypoint_context_length if train_goal_net else context_length,
        waypoint_horizon=waypoint_horizon,
        reward_episode_horizon=reward_episode_horizon,
    )
    if reward_conditioning and hasattr(
        data_module, "reward_representation"
    ):
        # Reward normalization is training-split metadata and must be known
        # before the policy checkpoint hyperparameters are captured.
        data_module.setup(stage="fit")

    if not train_goal_net:
        model = policies.RvS(
            env.observation_space,
            env.action_space,
            hidden_size=hidden_size,
            depth=depth,
            learning_rate=learning_rate,
            dropout_p=dropout_p,
            batch_size=batch_size,
            unconditional_policy=unconditional_policy,
            reward_conditioning=reward_conditioning,
            env_name=env_name,
            waypoint_horizon=waypoint_horizon,
            context_length=context_length,
            waypoint_hidden_size=waypoint_hidden_size,
            goal_network_ckpt=goal_network_ckpt,
            reward_condition_dim=(
                data_module.goal_dim
                if reward_conditioning and hasattr(data_module, "goal_dim")
                else None
            ),
            reward_representation=(
                data_module.reward_representation
                if reward_conditioning
                and hasattr(data_module, "reward_representation")
                else None
            ),
            reward_normalization_const=(
                data_module.reward_normalization_const
                if reward_conditioning
                and hasattr(data_module, "reward_normalization_const")
                else None
            ),
            reward_episode_horizon=(
                data_module.reward_episode_horizon
                if reward_conditioning
                and hasattr(data_module, "reward_episode_horizon")
                else None
            ),
        )
    else:
        required_metadata = ("state_dim", "goal_dim", "waypoint_dim")
        missing_metadata = [
            name for name in required_metadata
            if not hasattr(data_module, name)
        ]
        if missing_metadata:
            raise ValueError(
                "Waypoint training requires dataset dimensional metadata: "
                + ", ".join(missing_metadata)
            )
        if data_module.goal_dim != data_module.waypoint_dim:
            raise ValueError(
                "The official waypoint network requires goal_dim == waypoint_dim."
            )
        model = goal_net.KForwardGoalNetwork(
            obs_dim=data_module.state_dim,
            goal_dim=data_module.goal_dim,
            hidden_dim=waypoint_hidden_size,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_T=waypoint_horizon,
            reward=reward_conditioning,
            reward_representation=(
                data_module.reward_representation
                if reward_conditioning
                and hasattr(data_module, "reward_representation")
                else None
            ),
            reward_normalization_const=(
                data_module.reward_normalization_const
                if reward_conditioning
                and hasattr(data_module, "reward_normalization_const")
                else None
            ),
            reward_episode_horizon=(
                data_module.reward_episode_horizon
                if reward_conditioning
                and hasattr(data_module, "reward_episode_horizon")
                else None
            ),
        )
    wandb_logger.watch(model, log="all")

    monitor = "val_loss" if val_frac > 0 else "train_loss"
    checkpoint_dirpath = os.path.join(wandb_logger.experiment.dir, checkpoint_dir)
    checkpoint_filename = "gcsl-" + env_name + "-{epoch:03d}-{" + f"{monitor}" + ":.4e}"
    periodic_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        filename=checkpoint_filename,
        save_last=False,
        save_top_k=-1,
        every_n_epochs=checkpoint_every_n_epochs,
        every_n_train_steps=checkpoint_every_n_steps,
        train_time_interval=pd.Timedelta(checkpoint_time_interval).to_pytimedelta()
        if checkpoint_time_interval is not None
        else None,
    )
    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        monitor=monitor,
        filename=checkpoint_filename,
        save_last=True,  # save latest model
        save_top_k=1,  # save top model based on monitored loss
    )
    trainer = pl.Trainer(
        gpus=int(use_gpu),
        auto_lr_find=auto_tune_lr,
        max_epochs=epochs,
        max_steps=max_steps,
        max_time=train_time,
        logger=wandb_logger,
        progress_bar_refresh_rate=20,
        callbacks=[periodic_checkpoint_callback, val_checkpoint_callback],
        track_grad_norm=-1,  # logs the 2-norm of gradients
        limit_val_batches=1.0 if val_frac > 0 else 0,
        limit_test_batches=0,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description="Reinforcement Learning via Supervised Learning",
    )
    # configuration
    parser.add_argument(
        "--configs",
        default=None,
        required=False,
        is_config_file=True,
        help="path(s) to configuration file(s)",
    )
    # environment
    parser.add_argument(
        "--env_name",
        default="pointmass_rooms",
        type=str,
        choices=envs.env_names + step.gym_goal_envs + step.d4rl_env_names,
        help="an environment name",
    )
    # reproducibility
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="sets the random seed; if this is not specified, it is chosen randomly",
    )
    # conditioning
    conditioning_group = parser.add_mutually_exclusive_group()
    conditioning_group.add_argument(
        "--unconditional_policy",
        action="store_true",
        default=False,
        help="run vanilla behavior cloning without conditioning on goals",
    )
    conditioning_group.add_argument(
        "--reward_conditioning",
        action="store_true",
        default=False,
        help="condition behavior cloning on the reward-to-go",
    )
    parser.add_argument(
        "--cumulative_reward_to_go",
        action="store_true",
        default=False,
        help="if reward_conditioning, condition on cumulative (instead of average) "
        "reward-to-go",
    )
    # architecture
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="learning rate for each gradient step",
    )
    parser.add_argument(
        "--auto_tune_lr",
        action="store_true",
        default=False,
        help="have PyTorch Lightning try to automatically find the best learning rate",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        required=True,
        help="size of hidden layers in policy network",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=True,
        help="number of hidden layers in policy network",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        required=True,
        help="dropout probability",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="batch size for each gradient step",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=20,
        help="number of timesteps in each policy Transformer sequence",
    )
    parser.add_argument(
        "--waypoint_context_length",
        type=int,
        default=40,
        help="number of timesteps in each waypoint-network training sequence",
    )
    parser.add_argument(
        "--waypoint_horizon",
        type=int,
        default=30,
        help="future timestep offset used to construct waypoint targets",
    )
    parser.add_argument(
        "--waypoint_hidden_size",
        type=int,
        default=1024,
        help="hidden width of the learned waypoint network",
    )
    parser.add_argument(
        "--reward_representation",
        choices=("average", "cumulative", "both"),
        default=None,
        help="reward-to-go channels used by reward-conditioned datasets",
    )
    parser.add_argument(
        "--reward_episode_horizon",
        type=int,
        default=None,
        help=(
            "episode horizon used by average reward-to-go; defaults to the "
            "environment TimeLimit and is stored in checkpoints"
        ),
    )
    parser.add_argument(
        "--goal_network_ckpt",
        default=os.environ.get("GOAL_NETWORK_CKPT"),
        help=(
            "waypoint-network checkpoint path, or 'manual'; the legacy "
            "GOAL_NETWORK_CKPT environment variable is used only as a default"
        ),
    )
    # training
    train_time_group = parser.add_mutually_exclusive_group(required=True)
    train_time_group.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="the number of training epochs.",
    )
    train_time_group.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help="the number of training gradient steps per bootstrap iteration. ignored "
        "if --train_time is set",
    )
    train_time_group.add_argument(
        "--train_time",
        default=None,
        type=str,
        help="how long to train, specified as a DD:HH:MM:SS str",
    )
    checkpoint_frequency_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_frequency_group.add_argument(
        "--checkpoint_every_n_epochs",
        default=None,
        type=int,
        help="the period of training epochs for saving checkpoints",
    )
    checkpoint_frequency_group.add_argument(
        "--checkpoint_every_n_steps",
        default=None,
        type=int,
        help="the period of training gradient steps for saving checkpoints",
    )
    checkpoint_frequency_group.add_argument(
        "--checkpoint_time_interval",
        default=None,
        type=str,
        help="how long between saving checkpoints, specified as a HH:MM:SS str",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        required=True,
        help="fraction of data to use for validation",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="place networks and data on the GPU",
    )
    parser.add_argument("--which_gpu", default=0, type=int, help="which GPU to use")
    # GCSL
    parser.add_argument(
        "--rollout_directory",
        default=None,
        type=str,
        help="a directory containing the offline dataset to use for training",
    )
    parser.add_argument(
        "--total_steps",
        default=100000,
        type=int,
        help="if `rollout_directory` is not provided and the environment is from GCSL, "
        "generate an offline training dataset with this many environment steps",
    )
    parser.add_argument(
        "--max_episode_steps",
        default=50,
        type=int,
        help="the maximum number of steps in each episode",
    )
    discretization_group = parser.add_mutually_exclusive_group()
    discretization_group.add_argument(
        "--discretize",
        action="store_true",
        default=False,
        help="if the environment is from GCSL, discretize the environment's action "
        "space",
    )
    discretization_group.add_argument(
        "--discretize_rollouts_only",
        action="store_true",
        default=False,
        help="if the environment is from GCSL, sample discretized random rollouts in a "
        "continuous action space",
    )
    # analysis
    parser.add_argument(
        "--run_tag",
        default=None,
        type=str,
        help="a tag that's logged to help find the run later",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="visualize the hitting times of each bootstrap iteration's learned policy",
    )
    parser.add_argument(
        "--analyze_d4rl",
        action="store_true",
        default=False,
        help="analyze the learned policy in D4RL",
    )
    parser.add_argument(
        "--trajectory_samples",
        default=None,
        type=int,
        help="number of trajectory samples for --visualize and --analyzed4rl flags",
    )
    parser.add_argument(
        "--val_checkpoint_only",
        action="store_true",
        default=False,
        help="pass --val_checkpoint_only to analyze_d4rl script",
    )
    parser.add_argument(
        "--train_goal_net",
        action="store_true",
        default=False,
        help="pass --train_goal_net to train goal network",
    )
    parser.add_argument(
        "--d4rl_analysis",
        default="all",
        type=str,
        choices=[
            "input_interpolation",
            "weight_histograms",
            "kitchen_subtasks",
            "elite_goals",
            "length_goals",
            "reward_goals",
            "all",
        ],
        help="which analysis to run for --analyzed4rl",
    )

    args = parser.parse_args()
    if args.unconditional_policy and args.env_name not in step.d4rl_env_names:
        raise NotImplementedError

    args.seed = np.random.randint(2**31 - 1) if args.seed is None else args.seed
    if (
        args.cumulative_reward_to_go
        and args.reward_representation not in (None, "cumulative")
    ):
        parser.error(
            "--cumulative_reward_to_go conflicts with "
            f"--reward_representation={args.reward_representation}"
        )
    if args.cumulative_reward_to_go:
        args.reward_representation = "cumulative"
    elif args.reward_conditioning and args.reward_representation is None:
        args.reward_representation = dataset.resolve_reward_representation()
    util.set_seed(args.seed + 1)
    wandb_logger = pl.loggers.wandb.WandbLogger(project=wandb_project)
    device = util.configure_gpu(args.use_gpu, args.which_gpu)
    policy_env = step.create_env(
        args.env_name,
        args.max_episode_steps,
        args.discretize,
        seed=args.seed + 2,
    )
    if args.reward_conditioning and args.reward_episode_horizon is None:
        args.reward_episode_horizon = dataset.environment_episode_horizon(
            policy_env
        )
    log_args(args, wandb_logger)

    if args.discretize_rollouts_only and args.env_name not in step.d4rl_env_names:
        rollout_env = step.create_env(
            args.env_name,
            args.max_episode_steps,
            True,
            seed=args.seed + 3,
        )
    else:
        rollout_env = policy_env

    if args.rollout_directory is not None:
        rollout_directory = args.rollout_directory
    elif args.env_name in step.d4rl_env_names:
        rollout_directory = None
    else:
        rollout_directory = os.path.join(wandb_logger.experiment.dir, rollout_dir)
        step.generate_random_rollouts(
            rollout_env,
            rollout_directory,
            args.total_steps,
            args.max_episode_steps,
            use_base_actions=args.discretize_rollouts_only,
        )
    run_training(
        env=policy_env,
        env_name=args.env_name,
        seed=args.seed,
        wandb_logger=wandb_logger,
        rollout_directory=rollout_directory,
        unconditional_policy=args.unconditional_policy,
        reward_conditioning=args.reward_conditioning,
        cumulative_reward_to_go=args.cumulative_reward_to_go,
        epochs=args.epochs,
        max_steps=args.max_steps,
        train_time=args.train_time,
        hidden_size=args.hidden_size,
        depth=args.depth,
        learning_rate=args.learning_rate,
        auto_tune_lr=args.auto_tune_lr,
        dropout_p=args.dropout_p,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        checkpoint_time_interval=args.checkpoint_time_interval,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        use_gpu=args.use_gpu,
        train_goal_net=args.train_goal_net,
        waypoint_horizon=args.waypoint_horizon,
        context_length=args.context_length,
        waypoint_context_length=args.waypoint_context_length,
        waypoint_hidden_size=args.waypoint_hidden_size,
        reward_representation=args.reward_representation,
        reward_episode_horizon=args.reward_episode_horizon,
        goal_network_ckpt=args.goal_network_ckpt,
    )
    if args.visualize:
        visualize.visualize_performance(
            wandb_logger.experiment.dir,
            device,
            wandb_run=wandb_logger.experiment,
            trajectory_samples=2000
            if args.trajectory_samples is None
            else args.trajectory_samples,
        )
    if args.analyze_d4rl:
        analyze_d4rl.analyze_performance(
            wandb_logger.experiment.dir,
            device,
            wandb_run=wandb_logger.experiment,
            analysis=args.d4rl_analysis,
            trajectory_samples=200
            if args.trajectory_samples is None
            else args.trajectory_samples,
            last_checkpoints_too=not args.val_checkpoint_only,
        )
