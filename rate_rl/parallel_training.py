"""Independent native-torque workers and globally counted PPO updates.

Each subprocess owns its simulator, curriculum, Monitor and RNG.  Only worker
zero records trajectories.  The parent communicates through VecEnv methods;
it never accesses a subprocess's ``unwrapped`` environment.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from functools import partial
from multiprocessing.util import Finalize
import os
from pathlib import Path
import random
import signal
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import _stack_obs

from .backend import GazeboPX4Backend
from .curriculum import RigToFlightCurriculum
from .env import RateControlEnv
from .environment_contract import environment_contract
from .trajectory import EpisodeTrajectory, TrajectoryClock


class ParallelWorkerState(gym.Wrapper):
    """Expose checkpoint state inside its owning worker and close on failures."""

    def __init__(self, env, curriculum, *, worker_id, instance, seed, n_envs,
                 starting_steps, worker_state, trajectory_clock, monitor_path):
        super().__init__(env)
        self.curriculum = curriculum
        self.worker_id, self.instance = worker_id, instance
        self.worker_seed = int(worker_state.get("seed", seed)) if worker_state else seed
        self.n_parallel_envs = n_envs
        self.global_steps = starting_steps
        self.samples_seen = int(worker_state.get("samples_seen", 0)) if worker_state else 0
        self.trajectory_clock = trajectory_clock
        self.monitor_path = str(monitor_path)
        self.first_reset = True
        self.restored_rng = bool(worker_state and worker_state.get("np_rng_state") is not None)
        self._closed = False
        self._environment_contract = environment_contract(curriculum)
        base = curriculum.unwrapped
        base.np_random = np.random.default_rng(self.worker_seed)
        if self.restored_rng:
            base.np_random.bit_generator.state = deepcopy(worker_state["np_rng_state"])
        random.seed(self.worker_seed)
        np.random.seed(self.worker_seed)
        if worker_state and "python_rng_state" in worker_state:
            random.setstate(worker_state["python_rng_state"])
        if worker_state and "numpy_rng_state" in worker_state:
            np.random.set_state(worker_state["numpy_rng_state"])
        self.action_space.seed(self.worker_seed)

    def _close_after_failure(self):
        try:
            self.close()
        except BaseException:
            # Preserve the original infrastructure error.  The backend's close
            # is also called directly so a plotting/Monitor failure cannot skip it.
            try:
                self.curriculum.unwrapped.backend.close()
            except BaseException:
                pass

    def reset(self, *, seed=None, options=None):
        if self.first_reset:
            # PPO.set_random_seed() may have queued a seed in the VecEnv.  A
            # restored worker RNG takes precedence over that initialization seed.
            seed = None if self.restored_rng else self.worker_seed
        try:
            result = self.env.reset(seed=seed, options=options)
            self.first_reset = False
            return result
        except BaseException:
            self._close_after_failure()
            raise

    def step(self, action):
        try:
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.samples_seen += 1
            self.global_steps += self.n_parallel_envs
            info["parallel_worker_id"] = self.worker_id
            info["parallel_global_steps"] = self.global_steps
            return observation, reward, terminated, truncated, info
        except BaseException:
            self._close_after_failure()
            raise

    def parallel_worker_state(self):
        return dict(
            version=1, worker_id=self.worker_id, instance=self.instance,
            seed=self.worker_seed, samples_seen=self.samples_seen,
            global_steps=self.global_steps,
            curriculum_state=self.curriculum.state(),
            np_rng_state=deepcopy(self.curriculum.unwrapped.np_random.bit_generator.state),
            python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state(),
        )

    def parallel_worker_metadata(self):
        base = self.curriculum.unwrapped
        backend = base.backend
        return dict(
            worker_id=self.worker_id, instance=self.instance, seed=self.worker_seed,
            pid=os.getpid(), monitor_path=self.monitor_path,
            trajectory_enabled=self.trajectory_clock is not None,
            trajectory_step_stride=self.n_parallel_envs,
            simulator=deepcopy(backend.config),
            environment_contract=deepcopy(self._environment_contract),
            outer_loop=asdict(base.outer_loop.config),
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.env.close()
        finally:
            # A failing outer wrapper must not strand owned PX4/Gazebo groups.
            self.curriculum.unwrapped.backend.close()


def _build_worker(runtime, run_dir, config, promotion_options, curriculum_state,
                  worker_state, starting_steps, seed, base_instance, n_envs,
                  worker_id, backend_factory, monitor_fields):
    torch.set_num_threads(1)
    worker_dir = Path(run_dir) / "workers" / f"worker_{worker_id:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    instance = base_instance + worker_id
    backend = backend_factory(runtime, worker_dir / "episodes", instance=instance)
    env = None
    try:
        backend.set_mode("free_flight")
        curriculum = RigToFlightCurriculum(RateControlEnv(backend, config), **promotion_options)
        state = worker_state.get("curriculum_state", curriculum_state) if worker_state else curriculum_state
        if state is not None:
            if (state.get("training_stage") != "free_flight"
                    or not state.get("ready_for_air") or state.get("pending_evaluation") is not None):
                raise ValueError("Parallel training requires manual free-flight; rig/pending promotion is unsupported.")
            curriculum.restore(deepcopy(state))
        if not curriculum.ready_for_air or curriculum.pending_evaluation is not None:
            raise ValueError("Parallel workers cannot run rig promotion or evaluation.")
        clock = None
        wrapped = curriculum
        if worker_id == 0:
            clock = TrajectoryClock(
                Path(run_dir) / "trajectories", interval=40_000,
                label="Training episode (parallel worker 0)", stochastic_actions=True,
                step_stride=n_envs, initial_step=starting_steps, worker_id=worker_id,
            )
            wrapped = EpisodeTrajectory(wrapped, clock)
        monitor_path = worker_dir / "monitor.csv"
        wrapped = Monitor(wrapped, str(monitor_path), info_keywords=tuple(monitor_fields),
                          override_existing=not monitor_path.exists())
        env = ParallelWorkerState(
            wrapped, curriculum, worker_id=worker_id, instance=instance,
            seed=seed + worker_id, n_envs=n_envs, starting_steps=starting_steps,
            worker_state=worker_state, trajectory_clock=clock, monitor_path=monitor_path,
        )
        # multiprocessing's finalizers run on ordinary worker exits (including a
        # pipe EOF).  SIGTERM is explicit because simulators use new sessions.
        Finalize(env, env.close, exitpriority=10)

        def terminate_worker(signum, frame):
            env._close_after_failure()
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, terminate_worker)
        return env
    except BaseException:
        try:
            if env is not None:
                env.close()
        except BaseException:
            pass
        try:
            backend.close()
        except BaseException:
            pass
        raise


class ManagedSubprocVecEnv(SubprocVecEnv):
    """Spawn workers with peer-failure detection and bounded shutdown.

    SB3's standard close waits on pending replies indefinitely.  Here a dead
    worker or IPC failure closes every owned worker instead of leaving sibling
    simulators alive.  No process-name based or global simulator kill is used.
    """

    response_timeout = 240.0
    shutdown_timeout = 15.0

    def __init__(self, env_fns):
        try:
            super().__init__(env_fns, start_method="spawn")
        except BaseException:
            self.close()
            raise

    def _receive(self, remote):
        deadline = time.monotonic() + self.response_timeout
        while not remote.poll(0.2):
            failed = [index for index, process in enumerate(self.processes) if not process.is_alive()]
            if failed:
                raise RuntimeError(f"Parallel environment worker exited: {failed}; inspect worker episode logs.")
            if time.monotonic() >= deadline:
                raise TimeoutError("Parallel environment response timed out; closing all owned workers.")
        return remote.recv()

    def reset(self):
        try:
            for index, remote in enumerate(self.remotes):
                remote.send(("reset", (self._seeds[index], self._options[index])))
            results = [self._receive(remote) for remote in self.remotes]
            observations, self.reset_infos = zip(*results, strict=True)
            self._reset_seeds()
            self._reset_options()
            return _stack_obs(observations, self.observation_space)
        except BaseException:
            self.close()
            raise

    def step_async(self, actions):
        try:
            super().step_async(actions)
        except BaseException:
            self.close()
            raise

    def step_wait(self):
        try:
            results = [self._receive(remote) for remote in self.remotes]
            self.waiting = False
            observations, rewards, dones, infos, self.reset_infos = zip(*results, strict=True)
            return _stack_obs(observations, self.observation_space), np.stack(rewards), np.stack(dones), infos
        except BaseException:
            self.close()
            raise

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        try:
            remotes = self._get_target_remotes(indices)
            for remote in remotes:
                remote.send(("env_method", (method_name, method_args, method_kwargs)))
            return [self._receive(remote) for remote in remotes]
        except BaseException:
            self.close()
            raise

    def get_attr(self, attr_name, indices=None):
        try:
            remotes = self._get_target_remotes(indices)
            for remote in remotes:
                remote.send(("get_attr", attr_name))
            return [self._receive(remote) for remote in remotes]
        except BaseException:
            self.close()
            raise

    def close(self):
        if getattr(self, "closed", False):
            return
        self.closed = True
        remotes = getattr(self, "remotes", ())
        processes = getattr(self, "processes", ())
        for remote in remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        deadline = time.monotonic() + self.shutdown_timeout
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.terminate()  # Worker SIGTERM handler closes its backend.
        deadline = time.monotonic() + self.shutdown_timeout
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        for remote in remotes:
            remote.close()
        self.waiting = False


def resize_worker_states(states, n_envs, *, seed=0, legacy_rng=None):
    """Retain existing workers' RNGs; added workers receive distinct fresh seeds.

    Only a completed checkpoint may change worker count. Physical episodes are
    reset on resume, while policy/optimizer/global sample count are unaffected.
    Removed workers remain recoverable from the source checkpoint.
    """
    if n_envs not in (2, 4):
        raise ValueError('Supported parallel environment counts are 2 and 4')
    result = deepcopy(list(states or []))
    if not result and legacy_rng is not None:
        result = [dict(np_rng_state=deepcopy(legacy_rng), seed=seed)]
    for index, state in enumerate(result):
        if state is None or state.get('worker_id', index) != index:
            raise ValueError('Invalid saved worker identities')
    result = result[:n_envs]
    used = {state.get('seed', seed + index) for index, state in enumerate(result)}
    if len(used) != len(result):
        raise ValueError('Saved workers must have distinct seeds')
    while len(result) < n_envs:
        candidate = seed + len(result)
        while candidate in used:
            candidate += 1
        if not 0 <= candidate < 2**32:
            raise ValueError('Worker seeds must be nonnegative 32-bit integers')
        used.add(candidate)
        result.append(dict(seed=candidate))
    return result


def make_parallel_env(runtime, run_dir, config, promotion_options,
                      curriculum_state=None, worker_states=None, starting_steps=0,
                      seed=0, base_instance=41, n_envs=2, *, backend_factory=None,
                      monitor_fields=()):
    """Create isolated native-torque workers without resetting the simulators.

    ``worker_states`` is either None or one entry per worker (entries may be
    None when migrating a single-env checkpoint).  The shared curriculum is a
    fallback only; RNG states are never implicitly duplicated across workers.
    ``backend_factory`` exists for simulator-free integration tests.
    """
    if type(n_envs) is not int or n_envs < 2:
        raise ValueError("Parallel training requires at least two environments.")
    if type(base_instance) is not int or base_instance < 0 or base_instance + n_envs - 1 > 100:
        raise ValueError("Parallel PX4 instance range must remain within [0, 100].")
    if type(starting_steps) is not int or starting_steps < 0:
        raise ValueError("starting_steps must be a nonnegative global sample count.")
    if type(seed) is not int or seed < 0 or seed + n_envs - 1 >= 2**32:
        raise ValueError("Worker seeds must be distinct nonnegative 32-bit integers.")
    if not config.native_torque or not config.px4_native_outer:
        raise ValueError("Parallel training supports only the current native 9-observation/3-torque task.")
    states = [None] * n_envs if worker_states is None else list(worker_states)
    if len(states) != n_envs:
        raise ValueError("worker_states must have one entry for every parallel environment.")
    for index, state in enumerate(states):
        if state is not None and (not isinstance(state, dict) or state.get("worker_id", index) != index):
            raise ValueError("Worker checkpoint identities do not match their positions.")
        if state is not None and state.get("global_steps", starting_steps) != starting_steps:
            raise ValueError("Worker checkpoint step count differs from the model checkpoint.")
    factory = GazeboPX4Backend if backend_factory is None else backend_factory
    env_fns = [partial(
        _build_worker, str(runtime), str(run_dir), deepcopy(config), deepcopy(promotion_options),
        deepcopy(curriculum_state), deepcopy(states[index]), starting_steps, seed,
        base_instance, n_envs, index, factory, tuple(monitor_fields),
    ) for index in range(n_envs)]
    vec_env = ManagedSubprocVecEnv(env_fns)
    try:
        vec_env.parallel_metadata = vec_env.env_method("parallel_worker_metadata")
        contracts = [item["environment_contract"] for item in vec_env.parallel_metadata]
        if any(contract != contracts[0] for contract in contracts[1:]):
            raise ValueError("Parallel workers have different environment/reward contracts.")
        return vec_env
    except BaseException:
        vec_env.close()
        raise


def attach_parallel_env(model, vec_env, n_steps=2048, batch_size=1024, n_epochs=10):
    """Replace collection state without rebuilding policy, optimizer or KL state.

    Call only between completed updates, normally immediately after loading a
    checkpoint.  ``n_steps`` is per environment and the budget remains global.
    The previous environment is owned by the caller and is not closed here.
    """
    if vec_env.num_envs < 2:
        raise ValueError("attach_parallel_env requires multiple environments.")
    for name, value in (("n_steps", n_steps), ("batch_size", batch_size), ("n_epochs", n_epochs)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    samples = n_steps * vec_env.num_envs
    if samples <= 1 or batch_size <= 1 or samples % batch_size:
        raise ValueError("Global rollout samples must be divisible by batch_size, with both greater than one.")
    if getattr(model, "_old_policy_snapshot", None) is not None:
        raise ValueError("Cannot replace environments during an active PPO rollout/update.")
    check_for_correct_spaces(vec_env, model.observation_space, model.action_space)
    buffer_class = model.rollout_buffer_class or type(model.rollout_buffer)
    buffer = buffer_class(
        n_steps, model.observation_space, model.action_space,
        device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda,
        n_envs=vec_env.num_envs, **model.rollout_buffer_kwargs,
    )
    previous_n_envs = model.n_envs
    try:
        # SB3's public set_env checks equal counts, so update only the count
        # after all validation and buffer construction have already succeeded.
        model.n_envs = vec_env.num_envs
        model.set_env(vec_env, force_reset=True)
    except BaseException:
        model.n_envs = previous_n_envs
        raise
    model.n_steps, model.batch_size, model.n_epochs = n_steps, batch_size, n_epochs
    model.rollout_buffer = buffer
    model._last_episode_starts = np.ones(vec_env.num_envs, dtype=bool)
    if hasattr(model, "_last_original_obs"):
        model._last_original_obs = None
    settings = getattr(model, "active_stage_settings", None)
    if isinstance(settings, dict):
        settings["batch_size"] = batch_size
    return model


class ParallelTrainingCoordinator:
    """Checkpoint only complete PPO updates, counted across every environment."""

    def __init__(self, model, vec_env, metrics, run_dir):
        self.model, self.vec_env, self.metrics = model, vec_env, metrics
        self.run_dir = Path(run_dir)
        if model.n_envs != vec_env.num_envs or model.n_envs < 2:
            raise ValueError("The model and parallel VecEnv must have the same environment count.")
        if model.get_env() is not vec_env:
            raise ValueError("Attach the supplied parallel VecEnv to the model before coordinating training.")
        if getattr(metrics, "curriculum", None) is not None:
            raise ValueError("Parallel metrics must use curriculum=None; curriculum state is worker-owned.")
        metadata = vec_env.env_method("parallel_worker_metadata")
        contracts = [item["environment_contract"] for item in metadata]
        if any(contract != contracts[0] for contract in contracts):
            raise ValueError("Parallel worker environment contracts disagree.")
        self.environment_contract = contracts[0]
        saved_contract = getattr(model, "promotion_environment_contract", self.environment_contract)
        if saved_contract != self.environment_contract:
            raise ValueError("Checkpoint environment contract differs from parallel worker task/simulator assets.")
        model.promotion_environment_contract = deepcopy(self.environment_contract)
        previous = getattr(model, "parallel_training_state", getattr(model, "promotion_training_state", {}))
        self.state = dict(
            completed_rollouts=int(previous.get("completed_rollouts", 0)),
            last_checkpoint_step=int(previous.get("last_checkpoint_step", model.num_timesteps)),
            n_envs=model.n_envs, n_steps_per_env=model.n_steps,
            rollout_samples=model.n_steps * model.n_envs,
        )
        if hasattr(model, "training_rng_state"):
            rng = model.training_rng_state
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.random.set_rng_state(rng["torch_cpu"].cpu())

    def sync_state(self):
        workers = self.vec_env.env_method("parallel_worker_state")
        if len(workers) != self.model.n_envs:
            raise RuntimeError("Incomplete parallel worker checkpoint state.")
        if any(worker["global_steps"] != self.model.num_timesteps for worker in workers):
            raise RuntimeError("Model and worker global sample counts disagree; refusing a misleading checkpoint.")
        self.model.parallel_worker_states = deepcopy(workers)
        self.model.parallel_training_state = deepcopy(self.state)
        # Compatibility for task-contract code and explicit single-to-parallel
        # migration.  New parallel resumes must use the complete list above.
        self.model.curriculum_state = deepcopy(workers[0]["curriculum_state"])
        self.model.training_env_rng_state = deepcopy(workers[0]["np_rng_state"])
        self.model.training_rng_state = dict(
            python=random.getstate(), numpy=np.random.get_state(),
            torch_cpu=torch.random.get_rng_state(),
        )
        self.model.training_schedule_state = dict(
            first_success_step=getattr(self.metrics, "first_success_step", None),
            batch_size_switch_step=getattr(self.metrics, "batch_size_switch_step", None),
        )
        survival_gate = getattr(self.metrics, "survival_gate", None)
        if survival_gate is not None:
            self.model.exploration_survival_gate = survival_gate.state()

    def _save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.sync_state()
        self.model.save(path)

    def learn(self, total_timesteps, callback, stop_after_rollout=None):
        if total_timesteps is None and stop_after_rollout is None:
            raise ValueError("Unbounded training requires an explicit stop condition.")
        if total_timesteps is not None and (type(total_timesteps) is not int or total_timesteps <= 0):
            raise ValueError("total_timesteps must be a positive integer.")
        model = self.model
        end = None if total_timesteps is None else model.num_timesteps + total_timesteps
        self.state.pop("stop_reason", None)
        self.state.pop("stop_after_update_step", None)
        rollout_samples = model.n_steps * model.n_envs
        try:
            while end is None or model.num_timesteps < end:
                before_steps, before_updates = model.num_timesteps, model._n_updates
                # SB3's budget is global samples; n_steps is per environment.
                model.learn(total_timesteps=rollout_samples, callback=callback,
                            reset_num_timesteps=False, tb_log_name="PPO")
                if (model.num_timesteps != before_steps + rollout_samples
                        or model._n_updates <= before_updates or not model.rollout_buffer.full):
                    raise RuntimeError("PPO stopped before a complete parallel rollout and optimization.")
                self.state["completed_rollouts"] += 1
                model.set_logger(model.logger)
                model.logger.record("train/completed_rollouts", self.state["completed_rollouts"])
                model.logger.record("train/n_envs", model.n_envs)
                model.logger.record("train/steps_per_env", model.n_steps)
                model.logger.record("train/rollout_samples", rollout_samples)
                model.logger.dump(step=model.num_timesteps)
                self.sync_state()
                if model.num_timesteps - self.state["last_checkpoint_step"] >= 10_000:
                    self.state["last_checkpoint_step"] = model.num_timesteps
                    self._save(self.run_dir / "checkpoints" / f"ppo_{model.num_timesteps}_steps")
                if stop_after_rollout is not None and stop_after_rollout():
                    self.state["stop_reason"] = "stop_after_rollout"
                    self.state["stop_after_update_step"] = model.num_timesteps
                    break
            else:
                self.state["stop_reason"] = "step_budget"
                self.state["stop_after_update_step"] = model.num_timesteps
            self.sync_state()
            return model
        except BaseException:
            self.vec_env.close()
            raise
