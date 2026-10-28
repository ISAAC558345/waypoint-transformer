"""Twelve frozen rollouts. No parameter update and no thirteenth scenario."""
from __future__ import annotations

import copy
import json
import random

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT, PPO
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.actor import LookaheadActor
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.features import actor_features
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.protocol import HEAD_FEATURE_COUNT
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.rollout import collect
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.train import _workers, build_pair
from wt.benchmark_v2.phase6i_priority_tag_queue_lookahead.worker import QueueLookaheadWorker
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import WATCHED_PATHS
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import critic_dim
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _install, _restore
from wt.lowlevel_v2.contracts import env_target
from wt.lowlevel_v2.residual_policy import CentralizedCritic
from wt.round30.ppo import seed_everything

from .credit import static_gradients
from .protocol import (
    BIT_ABS,
    DIAGNOSTIC_ASSIGNMENT_SEED,
    DIAGNOSTIC_PROCESS_SEED,
    E12_OUTPUT,
    FLOAT32_ABS,
    GRAD_SUM_ABS,
    LOGPROB_ABS,
    LOOKAHEAD_ABS,
    MAX_REGISTERED_BATCHES,
    OUTPUT,
    RATIO_ABS,
    REWARD_ABS,
    ROOT,
    TERMINAL_SCENARIOS,
    TRAIN_SEEDS,
    VARIANTS,
)


SCALAR_FIELDS = (
    ("reward", float(REWARD_ABS)),
    ("base_reward", float(REWARD_ABS)),
    ("bonus_age", float(REWARD_ABS)),
    ("bonus_tag", float(REWARD_ABS)),
    ("motion_penalty", float(REWARD_ABS)),
    ("served_bits", float(BIT_ABS)),
    ("served_tagged_bits", float(BIT_ABS)),
)


def capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], np.asarray(numpy_state[1]).copy(), int(numpy_state[2]), numpy_state[3], numpy_state[4]),
        "torch": torch.get_rng_state().clone(),
    }


def restore_rng(saved: dict) -> None:
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"])


def same_rng(saved: dict) -> bool:
    current = capture_rng()
    numpy_same = (
        current["numpy"][0] == saved["numpy"][0]
        and np.array_equal(current["numpy"][1], saved["numpy"][1])
        and int(current["numpy"][2]) == int(saved["numpy"][2])
        and current["numpy"][3] == saved["numpy"][3]
        and float(current["numpy"][4]) == float(saved["numpy"][4])
    )
    return bool(current["python"] == saved["python"] and numpy_same and torch.equal(current["torch"], saved["torch"]))


def install_update_guard():
    originals = (torch.optim.Optimizer.step, torch.optim.Adam.step)

    def _refused(self, *args, **kwargs):
        raise RuntimeError("parameter update is forbidden in the credit audit")

    torch.optim.Optimizer.step = _refused
    torch.optim.Adam.step = _refused
    return originals


def restore_update_guard(originals) -> None:
    torch.optim.Optimizer.step, torch.optim.Adam.step = originals


def frozen_hashes() -> dict[str, str]:
    paths = [ROOT / relative for relative in list(WATCHED_PATHS)]
    paths.append(E12_OUTPUT / "scenario_rows.csv")
    paths.append(E12_OUTPUT / "summary.json")
    paths.extend(sorted((E12_OUTPUT / "checkpoints").glob("*.pt")))
    paths.extend(sorted((E12_OUTPUT / "training_logs").glob("*.json")))
    paths.extend(sorted((E12_OUTPUT / "training_samples").glob("*.json")))
    hashes = {}
    for path in paths:
        hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(path)
    return hashes


def _clone_state(module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _same_state(module, saved: dict[str, torch.Tensor]) -> bool:
    current = module.state_dict()
    if set(current) != set(saved):
        return False
    return all(torch.equal(current[key].detach().cpu(), saved[key]) for key in saved)


def _saved_update_one(variant: str, seed: int) -> list[dict]:
    rows = json.loads((E12_OUTPUT / "training_samples" / f"{variant.lower()}_seed{int(seed)}.json").read_text(encoding="utf-8"))
    selected = [row for row in rows if int(row["update"]) == 1 and int(row["env_index"]) == 0]
    selected.sort(key=lambda row: int(row["step"]))
    return selected


def match_saved_slice(samples: list[dict], saved: list[dict]) -> dict:
    if len(samples) != len(saved):
        return {"ok": False, "localized": True, "field": "sample_count", "got": len(samples), "expected": len(saved), "max_abs": {}}
    worst = {name: 0.0 for name, _limit in SCALAR_FIELDS}
    worst["lookahead_input"] = 0.0
    worst["lookahead_true"] = 0.0
    worst["reward_float32"] = 0.0
    first = None
    for step, (got, expected) in enumerate(zip(samples, saved)):
        for name, limit in SCALAR_FIELDS:
            gap = abs(float(got[name]) - float(expected[name]))
            worst[name] = max(worst[name], gap)
            if gap > limit and first is None:
                first = {"step": int(step), "field": name, "got": float(got[name]), "expected": float(expected[name]), "abs": gap, "limit": limit}
        for name in ("lookahead_input", "lookahead_true"):
            gap = float(np.max(np.abs(np.asarray(got[name], dtype=np.float64) - np.asarray(expected[name], dtype=np.float64))))
            worst[name] = max(worst[name], gap)
            if gap > float(LOOKAHEAD_ABS) and first is None:
                first = {"step": int(step), "field": name, "abs": gap, "limit": float(LOOKAHEAD_ABS)}
        gap = abs(float(np.float32(got["reward_float32"])) - float(np.float32(expected["reward_float32"])))
        worst["reward_float32"] = max(worst["reward_float32"], gap)
        if gap > float(FLOAT32_ABS) and first is None:
            first = {"step": int(step), "field": "reward_float32", "abs": gap, "limit": float(FLOAT32_ABS)}
    return {"ok": first is None, "localized": True, "first": first, "max_abs": worst, "checked_steps": len(samples), "checked_env_index": 0}


def _reshape_components(records: list[dict], captured: list[dict], horizon: int, n_env: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if len(records) != horizon * n_env or len(captured) != horizon * n_env:
        raise RuntimeError("probe capture does not match the rollout loop")
    base = np.zeros((horizon, n_env), dtype=np.float64)
    age = np.zeros_like(base)
    tag = np.zeros_like(base)
    n_uav = int(captured[0]["association"].shape[0])
    n_buoy = int(captured[0]["served_tagged_by_buoy"].shape[0])
    association = np.zeros((horizon, n_env, n_uav), dtype=np.int64)
    lookahead_true = np.zeros((horizon, n_env, n_uav, 2), dtype=np.float64)
    served = np.zeros((horizon, n_env, n_buoy), dtype=np.float64)
    for time_index in range(horizon):
        for env_index in range(n_env):
            cursor = time_index * n_env + env_index
            base[time_index, env_index] = records[cursor]["base"]
            age[time_index, env_index] = records[cursor]["age"]
            tag[time_index, env_index] = records[cursor]["tag"]
            association[time_index, env_index] = captured[cursor]["association"]
            lookahead_true[time_index, env_index] = captured[cursor]["lookahead_true"]
            served[time_index, env_index] = captured[cursor]["served_tagged_by_buoy"]
    return {"base": base, "age": age, "tag": tag}, {"association": association, "lookahead_true": lookahead_true, "served_tagged_by_buoy": served}


def collect_with_side(workers: list[QueueLookaheadWorker]):
    records: list[dict] = []
    originals = []
    for worker in workers:
        method = worker.step

        def _bind(bound):
            def _call():
                record = bound()
                records.append(
                    {
                        "base": float(record["base_reward"]),
                        "age": float(record["bonus_age"]),
                        "tag": float(record["bonus_tag"]),
                    }
                )
                return record

            return _call

        originals.append(method)
        worker.step = _bind(method)
    rollout, original = _install()
    captured: list[dict] = []
    installed = rollout.follow_step

    def _wrapped(env, ledger, movement, association, scenario_id):
        assoc = np.asarray(association, dtype=np.int64).copy()
        true = actor_features(ledger, int(env.t), assoc, int(env.config.num_buoys), use_lookahead=True)
        result = installed(env, ledger, movement, association, scenario_id)
        attr = getattr(ledger, "step_attribution", None)
        if not isinstance(attr, dict) or "served_tagged_by_buoy" not in attr:
            raise RuntimeError("probe did not capture the pre-service tag split")
        captured.append(
            {
                "association": assoc,
                "lookahead_true": np.asarray(true[:, int(HEAD_FEATURE_COUNT) :], dtype=np.float64).copy(),
                "served_tagged_by_buoy": np.asarray(attr["served_tagged_by_buoy"], dtype=np.float64).copy(),
            }
        )
        return result

    rollout.follow_step = _wrapped
    try:
        storage, metrics, samples = collect(workers, int(PPO.rollout_length))
    finally:
        _restore(rollout, original)
        for worker, method in zip(workers, originals):
            worker.step = method
    components, side = _reshape_components(records, captured, int(PPO.rollout_length), len(workers))
    return storage, metrics, samples, components, side


def _stored_reward_gap(storage: dict, components: dict) -> float:
    reconstructed = np.asarray(components["base"] + components["age"] + components["tag"], dtype=np.float32)
    return float(np.max(np.abs(np.asarray(storage["rewards"], dtype=np.float32) - reconstructed)))


def _class_masks(side: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    association = side["association"]
    n_buoy = int(side["served_tagged_by_buoy"].shape[-1])
    exposed = np.zeros(association.shape, dtype=bool)
    served = np.zeros(association.shape, dtype=bool)
    for time_index, env_index, uav in np.ndindex(association.shape):
        buoy = int(env_target(int(association[time_index, env_index, uav]), n_buoy))
        if buoy < 0:
            continue
        if float(side["lookahead_true"][time_index, env_index, uav, 0]) > 0.0:
            exposed[time_index, env_index, uav] = True
        if float(side["served_tagged_by_buoy"][time_index, env_index, buoy]) > float(DUST_BITS):
            served[time_index, env_index, uav] = True
    return {"exposed": exposed, "served": served}


def _mean_abs(values: np.ndarray, mask: np.ndarray):
    selected = np.abs(np.asarray(values, dtype=np.float64))[np.asarray(mask, dtype=bool)]
    if int(selected.size) == 0:
        return None
    return float(np.mean(selected))


def _public_credit(credit: dict, side: dict[str, np.ndarray]) -> dict:
    decomposed = credit["decomposed"]
    masks = _class_masks(side)
    advantage = np.asarray(decomposed["advantages_full"], dtype=np.float64)
    broadcast = np.repeat(advantage[:, :, None], side["association"].shape[-1], axis=2)
    public = {key: value for key, value in credit.items() if key != "decomposed"}
    public.update(
        {
            "float32_linear_gap": decomposed["float32_linear_gap"],
            "shared_scale_additive_gap": decomposed["shared_scale_additive_gap"],
            "float64_diagnostic_linear_gap": decomposed["float64_diagnostic_linear_gap"],
            "component_float32_sum_gap": decomposed["component_float32_sum_gap"],
            "tag_advantage_all": decomposed["tag_advantage_all"],
            "tag_advantage_positive_bonus": decomposed["tag_advantage_positive_bonus"],
            "tag_advantage_zero_bonus": decomposed["tag_advantage_zero_bonus"],
            "tag_advantage_zero_reward": decomposed["tag_advantage_zero_reward"],
            "separate_renorm_difference_all": decomposed["separate_renorm_difference_all"],
            "separate_renorm_difference_positive_bonus": decomposed["separate_renorm_difference_positive_bonus"],
            "separate_renorm_difference_zero_bonus": decomposed["separate_renorm_difference_zero_bonus"],
            "shared_tag_component_positive_bonus": decomposed["shared_tag_component_positive_bonus"],
            "shared_tag_component_zero_bonus": decomposed["shared_tag_component_zero_bonus"],
            "float32_distinguishable_steps": decomposed["float32_distinguishable_steps"],
            "positive_tag_steps": decomposed["positive_tag_steps"],
            "future_tag_advantage_on_zero_bonus": decomposed["future_tag_advantage_on_zero_bonus"],
            "normalization_note": decomposed["normalization_note"],
            "exposed_uav_steps": int(np.sum(masks["exposed"])),
            "served_uav_steps": int(np.sum(masks["served"])),
            "other_uav_steps": int(masks["exposed"].size - np.sum(masks["served"])),
            "mean_abs_advantage_exposed": _mean_abs(broadcast, masks["exposed"]),
            "mean_abs_advantage_not_exposed": _mean_abs(broadcast, ~masks["exposed"]),
            "mean_abs_advantage_served": _mean_abs(broadcast, masks["served"]),
            "mean_abs_advantage_not_served": _mean_abs(broadcast, ~masks["served"]),
            "advantage_broadcast": True,
            "team_credit_note": "one team advantage is broadcast; subset means select different steps and are not a within-step split",
            "clipping_bottleneck": "unconfirmed",
            "adam_update_inferred": False,
        }
    )
    return public, decomposed


def _save_batch(name: str, storage: dict, components: dict, side: dict, decomposed: dict | None) -> str:
    destination = OUTPUT / "batches" / f"{name}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key, value in storage.items():
        payload[f"storage_{key}"] = np.asarray(value)
    for key, value in components.items():
        payload[f"component_{key}"] = np.asarray(value)
    for key, value in side.items():
        payload[f"side_{key}"] = np.asarray(value)
    if decomposed is not None:
        for key in ("advantages_full", "advantages_base", "advantages_no_tag", "normalized_full", "normalized_no_tag_separate", "normalized_tag_component_shared"):
            payload[key] = np.asarray(decomposed[key])
    np.savez_compressed(destination, **payload)
    return str(destination.relative_to(ROOT)).replace("\\", "/")


def _terminal_workers(actor, critic, model_seed: int, use_lookahead: bool) -> list[QueueLookaheadWorker]:
    seed_everything(int(DIAGNOSTIC_PROCESS_SEED) + int(model_seed))
    workers = []
    for index in range(int(PPO.num_envs)):
        scenario = int(TERMINAL_SCENARIOS[index % 2])
        worker = QueueLookaheadWorker(
            int(model_seed),
            int(index),
            actor,
            critic,
            deterministic=False,
            scenario_id=scenario,
            use_lookahead=bool(use_lookahead),
        )
        worker.rng = np.random.default_rng(int(DIAGNOSTIC_ASSIGNMENT_SEED) + int(model_seed) * 100 + int(index))
        worker.reset()
        workers.append(worker)
    return workers


def _load_terminal(variant: str, seed: int):
    path = E12_OUTPUT / "checkpoints" / f"{variant.lower()}_seed{int(seed)}.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    actor = LookaheadActor(log_std_init=float(LOG_STD_INIT))
    critic = CentralizedCritic(critic_dim())
    actor.load_state_dict(blob["actor"])
    critic.load_state_dict(blob["critic"])
    return actor, critic


def _credit_or_gap(actor, critic, storage, components, side, *, publish: bool) -> tuple[dict, dict | None, list[str]]:
    gaps = []
    if not publish:
        return {"published": False, "reason": "historical attribution stopped"}, None, gaps
    credit = static_gradients(actor, critic, storage, components)
    public, decomposed = _public_credit(credit, side)
    public["published"] = True
    if not credit["parameters_unchanged"]:
        gaps.append("autograd changed a parameter or buffer")
    if float(credit["logprob_max_abs"]) > float(LOGPROB_ABS) or float(credit["ratio_max_abs_from_one"]) > float(RATIO_ABS):
        gaps.append("frozen log-probability does not match the stored action")
        public["published"] = False
    if float(credit["per_uav_gradient_sum_max_abs"]) > float(GRAD_SUM_ABS):
        gaps.append("per-UAV policy gradients do not sum to the team gradient")
    if float(credit["mask_min"]) != 1.0 or float(credit["mask_max"]) != 1.0:
        gaps.append("policy_active_mask is not identically one")
    return public, decomposed, gaps


def execute_probes(history: dict) -> dict:
    if int(MAX_REGISTERED_BATCHES) != 12:
        raise RuntimeError("the registered probe budget is not 12")
    torch.set_num_threads(1)
    outer_rng = capture_rng()
    guard = install_update_guard()
    before = frozen_hashes()
    batches = []
    gaps = []
    rng_check = {"second_execution_budgeted": False, "restore_returns_captured_state": None, "torch_rng_changed_during_batch": None}
    registered = 0
    try:
        expected = {(item["variant"], int(item["model_seed"])): item for item in history["checkpoints"]["checkpoints"]}
        for seed in TRAIN_SEEDS:
            bundle = build_pair(int(seed))
            for variant in VARIANTS:
                registered += 1
                print(f"initial {variant} seed {int(seed)}", flush=True)
                checkpoint = expected[(variant, int(seed))]
                init_ok = bundle["actor_init_sha256"] == checkpoint["actor_init_sha256"] and bundle["critic_init_sha256"] == checkpoint["critic_init_sha256"]
                group = bundle[variant]
                actor_state = _clone_state(group["actor"])
                critic_state = _clone_state(group["critic"])
                optimizer_before = len(group["agent"].optimizer.state)
                seed_everything(int(seed))
                workers = _workers(int(seed), group["actor"], group["critic"], use_lookahead=variant == "L4_Q")
                storage, _metrics, samples, components, side = collect_with_side(workers)
                stored_gap = _stored_reward_gap(storage, components)
                if stored_gap > float(REWARD_ABS):
                    gaps.append(f"{variant} seed {seed} initial stored reward does not match the ledger split")
                match = match_saved_slice(samples, _saved_update_one(variant, int(seed)))
                publish = bool(init_ok and match["ok"])
                if not init_ok:
                    gaps.append(f"{variant} seed {seed} rebuilt initialization hash does not match the checkpoint")
                    match = {"ok": False, "localized": True, "field": "initialization_sha256", "sample_match": match}
                public, decomposed, credit_gaps = _credit_or_gap(group["actor"], group["critic"], storage, components, side, publish=publish)
                gaps.extend(f"{variant} seed {seed} initial: {item}" for item in credit_gaps)
                path = _save_batch(f"initial_{variant.lower()}_seed{int(seed)}", storage, components, side, decomposed)
                unchanged = _same_state(group["actor"], actor_state) and _same_state(group["critic"], critic_state)
                if not unchanged or len(group["agent"].optimizer.state) != optimizer_before:
                    gaps.append(f"{variant} seed {seed} initial parameters or optimizer state changed")
                batches.append(
                    {
                        "role": "initial_reconstruction",
                        "source": "reconstructed_pre_update_batch_env0_checked" if publish else "initial_slice_not_historical",
                        "variant": variant,
                        "model_seed": int(seed),
                        "historical_gradient_published": bool(public.get("published", False)),
                        "initialization_matches_checkpoint": bool(init_ok),
                        "sample_match": match,
                        "batch_path": path,
                        "batch_sha256": file_sha256(ROOT / path),
                        "parameters_unchanged": bool(unchanged),
                        "optimizer_state_entries": len(group["agent"].optimizer.state),
                        "credit": public,
                    }
                )
            del bundle
        for seed in TRAIN_SEEDS:
            for variant in VARIANTS:
                registered += 1
                print(f"terminal {variant} seed {int(seed)}", flush=True)
                actor, critic = _load_terminal(variant, int(seed))
                actor_state = _clone_state(actor)
                critic_state = _clone_state(critic)
                workers = _terminal_workers(actor, critic, int(seed), variant == "L4_Q")
                captured = None
                if variant == "L4_C" and int(seed) == 0:
                    captured = capture_rng()
                    captured_generators = [copy.deepcopy(worker.rng.bit_generator.state) for worker in workers]
                storage, _metrics, _samples, components, side = collect_with_side(workers)
                stored_gap = _stored_reward_gap(storage, components)
                if stored_gap > float(REWARD_ABS):
                    gaps.append(f"{variant} seed {seed} terminal stored reward does not match the ledger split")
                if captured is not None:
                    changed = not same_rng(captured)
                    restore_rng(captured)
                    for worker, state in zip(workers, captured_generators):
                        worker.rng.bit_generator.state = copy.deepcopy(state)
                    rng_check = {
                        "second_execution_budgeted": False,
                        "restore_returns_captured_state": bool(same_rng(captured)),
                        "torch_rng_changed_during_batch": bool(changed),
                        "batch": "terminal_l4_c_seed0",
                    }
                    if not rng_check["restore_returns_captured_state"]:
                        gaps.append("RNG restore did not return the captured state")
                public, decomposed, credit_gaps = _credit_or_gap(actor, critic, storage, components, side, publish=True)
                gaps.extend(f"{variant} seed {seed} terminal: {item}" for item in credit_gaps)
                path = _save_batch(f"terminal_{variant.lower()}_seed{int(seed)}", storage, components, side, decomposed)
                unchanged = _same_state(actor, actor_state) and _same_state(critic, critic_state)
                if not unchanged:
                    gaps.append(f"{variant} seed {seed} terminal parameters changed")
                batches.append(
                    {
                        "role": "terminal_frozen_batch",
                        "source": "diagnostic_proxy",
                        "variant": variant,
                        "model_seed": int(seed),
                        "scenarios": [int(TERMINAL_SCENARIOS[index % 2]) for index in range(int(PPO.num_envs))],
                        "process_seed": int(DIAGNOSTIC_PROCESS_SEED) + int(seed),
                        "historical_gradient_published": False,
                        "diagnostic_gradient_published": bool(public.get("published", False)),
                        "batch_path": path,
                        "batch_sha256": file_sha256(ROOT / path),
                        "parameters_unchanged": bool(unchanged),
                        "credit": public,
                    }
                )
                del actor, critic
    except Exception as exc:
        gaps.append(f"{type(exc).__name__}: {exc}")
    finally:
        restore_update_guard(guard)
        restore_rng(outer_rng)
    after = frozen_hashes()
    if before != after:
        gaps.append("a frozen checkpoint, log, or formal file hash changed during the probe")
    if registered != int(MAX_REGISTERED_BATCHES):
        gaps.append(f"registered batch count is {registered}")
    return {
        "batches": batches,
        "registered_batches": registered,
        "verification_passes": 0,
        "rng_check": rng_check,
        "frozen_hashes_unchanged": before == after,
        "frozen_hash_count": len(before),
        "gaps": gaps,
        "outer_rng_restored": same_rng(outer_rng),
    }
