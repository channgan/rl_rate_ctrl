"""Complete-episode promotion checks with one isolated, frozen policy snapshot."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import pickle
import random

import numpy as np
import torch

from .backend import SimulatorError


def policy_fingerprint(policy) -> str:
    """Hash every state_dict entry, including value-network weights and buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            header = json.dumps([name, str(value.dtype), list(value.shape)]).encode()
            payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        else:
            header = json.dumps([name, "extra_state"]).encode()
            payload = pickle.dumps(value, protocol=5)
        for data in (header, payload):
            digest.update(len(data).to_bytes(8, "little"))
            digest.update(data)
    return digest.hexdigest()


def make_evaluation_seeds(training_seed: int, attempt: int, count: int) -> list[int]:
    """Use a separate deterministic seed namespace without advancing training RNGs."""
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer))
           for x in (training_seed, attempt, count)):
        raise ValueError("Evaluation seed arguments must be integers")
    if training_seed < 0 or attempt < 0 or count < 1:
        raise ValueError("Expected non-negative seed/attempt and a positive count")
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(training_seed), 0x4556414C, int(attempt)]))
    seeds = []
    used = {int(training_seed)}
    while len(seeds) < count:
        seed = int(rng.integers(0, 2**31 - 1))
        if seed not in used:
            seeds.append(seed)
            used.add(seed)
    return seeds


@contextmanager
def _preserve_global_rngs():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    # Do not initialise CUDA merely to evaluate a CPU policy.
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _run_episode(policy, env, seed):
    observation, reset_info = env.reset(seed=seed)
    config = env.unwrapped.config
    dt = float(config.dt)
    expected_steps = round(config.episode_seconds / dt)
    initial_sim_us = int(reset_info["sim_us"])
    saturation_steps = np.zeros(4, dtype=np.int64)
    streak = np.zeros(4, dtype=np.int64)
    longest_streak = np.zeros(4, dtype=np.int64)
    episode_return = 0.0
    steps = 0
    while True:
        action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        episode_return += float(reward)
        if info["sim_us"] != initial_sim_us + steps * round(dt * 1e6):
            raise SimulatorError("Evaluation simulation time did not advance by exactly dt")
        saturated = np.asarray(info["saturated_motors"], dtype=bool)
        if saturated.shape != (4,):
            raise ValueError("Expected saturation feedback for four motors")
        saturation_steps += saturated
        streak = np.where(saturated, streak + 1, 0)
        longest_streak = np.maximum(longest_streak, streak)
        if truncated:
            raise SimulatorError("Evaluation was truncated before a physical task outcome")
        if terminated:
            break
        if steps >= expected_steps:
            raise SimulatorError("Evaluation environment did not terminate at its task horizon")

    success = bool(info["is_success"])
    failure = info["failure"]
    if success and (failure is not None or steps != expected_steps):
        raise SimulatorError("Evaluation success did not complete the configured task horizon")
    if not success and failure is None:
        raise SimulatorError("Evaluation ended without success or a physical failure")
    if steps > expected_steps:
        raise SimulatorError("Evaluation exceeded the configured task horizon")
    axis_rmse = np.asarray(info["episode_axis_rate_rmse_deg_s"], dtype=float)
    settled_fraction = float(info["episode_settled_tracking_fraction"])
    if (axis_rmse.shape != (3,) or not np.all(np.isfinite(axis_rmse))
            or not np.isfinite(settled_fraction) or not 0 <= settled_fraction <= 1
            or not np.isfinite(episode_return)):
        raise ValueError("Invalid final evaluation metrics")
    transfer_qualified = bool(info["transfer_qualified"])
    # Record the actual backend reset seed, not a guessed RNG draw. This seed
    # controls Gazebo noise; deterministic policy inference leaves it enabled.
    noise_seed = int(reset_info["simulator_seed"])
    return {
        "seed": int(seed), "target_seed": int(seed), "noise_seed": int(noise_seed),
        "simulator_seed": noise_seed, "noise_seed_source": "reset_info",
        "qualified": bool(success and transfer_qualified),
        "is_success": success, "transfer_qualified": transfer_qualified,
        "failure": failure, "steps": steps,
        "survival_seconds": float((info["sim_us"] - initial_sim_us) / 1e6),
        "episode_return": episode_return,
        "axis_rate_rmse_deg_s": axis_rmse.tolist(),
        "settled_tracking_fraction": settled_fraction,
        "saturation_fraction_per_motor": (saturation_steps / steps).tolist(),
        "max_saturation_streak_ms_per_motor": (longest_streak * dt * 1000).tolist(),
        "initial_sim_us": initial_sim_us, "final_sim_us": int(info["sim_us"]),
        "training_stage": info.get("training_stage", reset_info.get("training_stage")),
    }


def evaluate_frozen_policy(policy, env_factory, seeds, checkpoint_id,
                           required_qualified=8, on_episode=None) -> dict:
    """Evaluate every seed on one snapshot; infrastructure errors invalidate the group.

    The factory must return an independent, non-vector Gym environment with no
    training/curriculum wrapper. Only the environment determines physical task
    termination. Deterministic actions do not disable simulator sensor noise.
    """
    seeds = list(seeds)
    if (not seeds or any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
                         or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Evaluation requires distinct non-negative integer seeds")
    if (isinstance(required_qualified, bool)
            or not isinstance(required_qualified, (int, np.integer))
            or not 1 <= required_qualified <= len(seeds)):
        raise ValueError("Required qualified episodes must be within the full group size")
    seeds = [int(seed) for seed in seeds]
    report = dict(valid=True, checkpoint_id=str(checkpoint_id), seeds=seeds,
                  completed_episodes=0, qualified_count=0, passed=False, episodes=[],
                  required_qualified=int(required_qualified))
    with _preserve_global_rngs():
        report["policy_fingerprint_before"] = policy_fingerprint(policy)
        # SB3 caches its last torch distribution. Following a gradient update,
        # its non-leaf tensors cannot be deep-copied. It is disposable inference
        # state: the next predict() constructs a distribution from the weights.
        # A deepcopy memo excludes the cache without touching the source object.
        cached_distribution = getattr(getattr(policy, "action_dist", None), "distribution", None)
        memo = {id(cached_distribution): None} if cached_distribution is not None else {}
        frozen = deepcopy(policy, memo)
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        report["frozen_policy_fingerprint_before"] = policy_fingerprint(frozen)
        env = None
        try:
            try:
                env = env_factory()
                with torch.no_grad():
                    for seed in seeds:
                        episode = _run_episode(frozen, env, seed)
                        report["episodes"].append(episode)
                        report["completed_episodes"] += 1
                        report["qualified_count"] += int(episode["qualified"])
                        if on_episode is not None:
                            on_episode(deepcopy(episode))
            finally:
                if env is not None:
                    env.close()
        except SimulatorError as exc:
            report["valid"] = False
            report["error"] = str(exc)
        finally:
            report["policy_fingerprint_after"] = policy_fingerprint(policy)
            report["frozen_policy_fingerprint_after"] = policy_fingerprint(frozen)
        report["policy_unchanged"] = (
            report["policy_fingerprint_before"] == report["policy_fingerprint_after"]
            == report["frozen_policy_fingerprint_before"]
            == report["frozen_policy_fingerprint_after"])
        if not report["policy_unchanged"]:
            report["valid"] = False
            report["error"] = "Policy state changed during frozen evaluation"
        report["passed"] = bool(report["valid"]
            and report["completed_episodes"] == len(seeds)
            and report["qualified_count"] >= required_qualified)
    return report
