"""Exercise parallel training with isolated synthetic native-torque backends."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, Logger
from torch.distributions import kl_divergence

from rate_rl.adaptive_ppo import AdaptiveKLPPO
from rate_rl.env import RateControlEnv, TaskConfig
from rate_rl.parallel_training import (
    ParallelTrainingCoordinator, attach_parallel_env, make_parallel_env, resize_worker_states,
)


class SyntheticNativeBackend:
    """No simulator, socket or instance lock is opened by this test backend."""

    hover = 0.7
    px4_hover_command = 0.7

    def __init__(self, runtime, log_dir=None, *, instance=43):
        self.instance = instance
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.config = dict(
            mode="free_flight", mass_kg=1.0, gravity_m_s2=9.8,
            max_rotor_rad_s=1000.0, thrust_coefficient=5e-6,
        )
        self.failure_step = (3 if instance % 2 else 5) if str(runtime) == "short" else (17 if instance % 2 else 29)
        self.marker = (instance - 40) * 0.03
        self.native_outer = self.native_torque = False

    def set_mode(self, mode):
        self.config["mode"] = mode

    def set_position_goal(self, position, heading):
        self.goal = np.array(position)
        self.heading = heading

    def reset(self, seed=None):
        self.last_seed = seed
        self.steps = 0
        self.t = 1_000_000
        self.torque = np.zeros(3, dtype=np.float32)
        return self.state()

    def state(self):
        rates = np.array([self.marker, -self.marker / 2, self.marker / 3])
        pwm = np.array([0.6, 0.65, 0.7, 0.75])
        altitude = 0.1 if self.steps >= self.failure_step else 3.0
        return dict(
            sim_us=self.t, sample_us=self.t, rates=rates, rates_true=rates,
            position_ned=[0.0, 0.0, -altitude], q_ned_frd=[1.0, 0.0, 0.0, 0.0],
            linear_velocity_ned=[0.0, 0.0, 0.0], rotor_speed_fraction=pwm.copy(),
            applied_pwm=pwm.copy(), applied_speed_fraction=0.15 + 0.85 * pwm,
            applied_torque=self.torque.copy(), px4_local_position=[0.0, 0.0, -3.0],
            px4_rate_target=rates.copy(), px4_thrust_body_z=-0.7,
            px4_reference_us=self.t, px4_now_us=self.t, px4_position_control=True,
            px4_attitude_target=[1.0, 0.0, 0.0, 0.0],
        )

    def step_torque(self, action, dt):
        self.steps += 1
        self.t += round(dt * 1e6)
        self.torque = np.asarray(action, dtype=np.float32).copy()
        return self.state()

    def close(self):
        pass


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def task_config():
    return TaskConfig(
        dt=0.01, episode_seconds=0.30, native_torque=True,
        px4_native_outer=True, waypoint_tracking=True, squared_error_reward=True,
        terminate_on_tilt=False,
    )


def make_env(path, *, runtime="long", **overrides):
    kwargs = dict(
        seed=321, base_instance=43, n_envs=2,
        backend_factory=SyntheticNativeBackend,
    )
    kwargs.update(overrides)
    return make_parallel_env(runtime, path, task_config(), {}, **kwargs)


def make_model(env, **overrides):
    kwargs = dict(
        n_steps=2048, batch_size=1024, n_epochs=10,
        kl_phase="late", learning_rate=5e-5,
        gamma=np.exp(-0.01 / 30), gae_lambda=np.exp(-0.01 / 0.5),
        ent_coef=0.01, clip_range=0.2,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.ReLU, log_std_init=-2.3,
        ),
        seed=321, device="cpu",
    )
    kwargs.update(overrides)
    return AdaptiveKLPPO("MlpPolicy", env, **kwargs)


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_state_equal(left, right)
    else:
        assert actual == expected


class CaptureRollout(BaseCallback):
    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buffer = self.model.rollout_buffer
        self.observation_shape = buffer.observations.shape
        self.action_shape = buffer.actions.shape
        self.episode_starts = buffer.episode_starts.copy()


class CaptureLogs(KVWriter):
    def __init__(self):
        self.dumps = []

    def write(self, key_values, key_excluded, step=0):
        self.dumps.append((step, deepcopy(key_values)))

    def close(self):
        pass


def test_two_worker_rollout_and_checkpoint_preserve_kl_and_optimizer(tmp_path):
    env = make_env(tmp_path / "original")
    try:
        model = make_model(env)
        before = deepcopy(model.policy.state_dict())
        old_policy = deepcopy(model.policy)
        callback = CaptureRollout()
        model.learn(4096, callback=callback)
        assert model.n_envs == 2
        assert model.num_timesteps == 4096
        assert callback.observation_shape == (2048, 2, 9)
        assert callback.action_shape == (2048, 2, 3)
        assert callback.episode_starts.shape == (2048, 2)
        assert callback.episode_starts[:, 0].sum() > callback.episode_starts[:, 1].sum()
        assert model.rollout_buffer.observations.shape == (4096, 9)
        assert model.batch_size == 1024 and model.n_epochs == 10
        assert 0 < model.logger.name_to_value["train/optimizer_steps"] <= 40
        assert np.isfinite(model.logger.name_to_value["train/full_kl"])
        with torch.no_grad():
            observations = torch.as_tensor(model.rollout_buffer.observations)
            old_distribution = old_policy.get_distribution(observations).distribution
            new_distribution = model.policy.get_distribution(observations).distribution
            expected_kl = kl_divergence(old_distribution, new_distribution).sum(dim=-1).mean().item()
        assert model.logger.name_to_value["train/full_kl"] == pytest.approx(expected_kl, abs=1e-7)
        assert any(not torch.equal(value, before[key]) for key, value in model.policy.state_dict().items())
        states = env.env_method("parallel_worker_state")
        assert [state["samples_seen"] for state in states] == [2048, 2048]
        assert [state["global_steps"] for state in states] == [4096, 4096]

        model._adapt_learning_rate(0.01, False)
        model._adapt_learning_rate(0.001, False)
        model._adapt_learning_rate(0.001, False)
        policy_state = deepcopy(model.policy.state_dict())
        optimizer_state = deepcopy(model.policy.optimizer.state_dict())
        model.parallel_worker_states = deepcopy(states)
        path = tmp_path / "parallel_checkpoint"
        model.save(path)
        next_rate = model.kl_learning_rate
    finally:
        env.close()

    resumed_env = make_env(
        tmp_path / "resumed", worker_states=states, starting_steps=4096,
    )
    try:
        loaded = AdaptiveKLPPO.load(path, env=resumed_env, device="cpu")
        assert loaded.n_envs == 2 and loaded.n_steps == 2048
        assert loaded.num_timesteps == 4096
        assert loaded.kl_phase == "late"
        assert loaded.kl_learning_rate == pytest.approx(next_rate)
        assert loaded.kl_low_kl_count == 2
        assert_state_equal(loaded.parallel_worker_states, states)
        assert_state_equal(loaded.policy.state_dict(), policy_state)
        assert_state_equal(loaded.policy.optimizer.state_dict(), optimizer_state)
        loaded.learn(4096, reset_num_timesteps=False)
        assert loaded.num_timesteps == 8192
        restored_states = resumed_env.env_method("parallel_worker_state")
        assert [state["samples_seen"] for state in restored_states] == [4096, 4096]
        assert [state["global_steps"] for state in restored_states] == [8192, 8192]
    finally:
        resumed_env.close()


def test_workers_have_independent_resets_rng_monitor_and_action_history(tmp_path):
    env = make_env(tmp_path, runtime="short")
    try:
        initial = env.reset()
        assert initial.shape == (2, 9)
        np.testing.assert_array_equal(initial[:, 6:], np.zeros((2, 3)))
        np.testing.assert_allclose(initial[:, 0], [0.09 / 5, 0.12 / 5])
        states_before = env.env_method("parallel_worker_state")
        assert [state["worker_id"] for state in states_before] == [0, 1]
        assert [state["instance"] for state in states_before] == [43, 44]
        assert [state["seed"] for state in states_before] == [321, 322]
        assert states_before[0]["np_rng_state"] != states_before[1]["np_rng_state"]
        first_simulator_seeds = [info["simulator_seed"] for info in env.reset_infos]
        assert first_simulator_seeds[0] != first_simulator_seeds[1]
        actions = np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]], dtype=np.float32)
        for step in range(1, 6):
            observations, _, dones, infos = env.step(actions)
            expected_dones = [step % 3 == 0, step % 5 == 0]
            np.testing.assert_array_equal(dones, expected_dones)
            np.testing.assert_allclose(observations[:, 0], initial[:, 0])
            for worker in range(2):
                expected_history = np.zeros(3) if dones[worker] else actions[worker]
                np.testing.assert_array_equal(observations[worker, 6:], expected_history)
                if dones[worker]:
                    np.testing.assert_array_equal(infos[worker]["terminal_observation"][6:], actions[worker])
                    assert infos[worker]["terminal_observation"][0] == initial[worker, 0]
                    assert infos[worker]["episode"]["l"] == (3 if worker == 0 else 5)
            if step == 3:
                states_after = env.env_method("parallel_worker_state")
                assert states_after[0]["np_rng_state"] != states_before[0]["np_rng_state"]
                assert states_after[1]["np_rng_state"] == states_before[1]["np_rng_state"]
                assert env.env_method("get_episode_lengths") == [[3], []]
        assert env.env_method("get_episode_lengths") == [[3], [5]]
        final_states = env.env_method("parallel_worker_state")
        assert [state["samples_seen"] for state in final_states] == [5, 5]
        assert [state["global_steps"] for state in final_states] == [10, 10]
    finally:
        env.close()


def test_worker_rng_restore_survives_vecenv_seed_and_preserves_local_counters(tmp_path):
    env = make_env(tmp_path / "first", runtime="short")
    try:
        first = env.reset()
        original_simulator_seeds = [info["simulator_seed"] for info in env.reset_infos]
        for _ in range(2):
            env.step(np.zeros((2, 3), dtype=np.float32))
        states = env.env_method("parallel_worker_state")
    finally:
        env.close()

    repeated = make_env(tmp_path / "repeat", runtime="short")
    try:
        np.testing.assert_array_equal(repeated.reset(), first)
        assert [info["simulator_seed"] for info in repeated.reset_infos] == original_simulator_seeds
    finally:
        repeated.close()

    expected_simulator_seeds = []
    for state in states:
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state["np_rng_state"])
        expected_simulator_seeds.append(int(rng.integers(0, 2**31 - 1)))
    restored = make_env(
        tmp_path / "restored", runtime="short", worker_states=states, starting_steps=4,
    )
    try:
        before_reset = restored.env_method("parallel_worker_state")
        for current, saved in zip(before_reset, states):
            assert current["np_rng_state"] == saved["np_rng_state"]
            assert current["curriculum_state"] == saved["curriculum_state"]
        # SB3 calls VecEnv.seed during setup/load; saved per-worker RNG wins.
        restored.seed(999)
        observations = restored.reset()
        assert [info["simulator_seed"] for info in restored.reset_infos] == expected_simulator_seeds
        np.testing.assert_array_equal(observations[:, 6:], np.zeros((2, 3)))
        restored.step(np.zeros((2, 3), dtype=np.float32))
        after = restored.env_method("parallel_worker_state")
        assert [state["samples_seen"] for state in after] == [3, 3]
        assert [state["global_steps"] for state in after] == [6, 6]
    finally:
        restored.close()


def test_single_worker_upgrade_and_coordinator_count_global_samples(tmp_path):
    single = RateControlEnv(SyntheticNativeBackend("short"), task_config())
    model = make_model(single, n_steps=32, batch_size=16, n_epochs=1)
    try:
        model.learn(32)
        model._adapt_learning_rate(0.01, False)
        model._adapt_learning_rate(0.001, False)
        policy = model.policy
        optimizer = policy.optimizer
        policy_state = deepcopy(policy.state_dict())
        optimizer_state = deepcopy(optimizer.state_dict())
        rate, low_count = model.kl_learning_rate, model.kl_low_kl_count
    finally:
        single.close()

    parallel = make_env(tmp_path / "parallel", starting_steps=32)
    try:
        attach_parallel_env(model, parallel)
        assert model.policy is policy
        assert model.policy.optimizer is optimizer
        assert_state_equal(model.policy.state_dict(), policy_state)
        assert_state_equal(model.policy.optimizer.state_dict(), optimizer_state)
        assert (model.kl_learning_rate, model.kl_low_kl_count) == (rate, low_count)
        assert model.n_envs == 2 and model.n_steps == 2048
        assert model.batch_size == 1024 and model.n_epochs == 10
        assert model.rollout_buffer.observations.shape == (2048, 2, 9)
        assert model.num_timesteps == 32

        captured = CaptureLogs()
        model.set_logger(Logger(folder=None, output_formats=[captured]))
        coordinator = ParallelTrainingCoordinator(model, parallel, SimpleNamespace(), tmp_path)
        callback = CaptureRollout()
        # A small global budget still finishes exactly one 2 x 2048 rollout.
        coordinator.learn(1, callback=callback)
        assert model.num_timesteps == 32 + 4096
        assert coordinator.state["completed_rollouts"] == 1
        assert model.parallel_training_state["rollout_samples"] == 4096
        assert model.parallel_training_state["n_steps_per_env"] == 2048
        states = model.parallel_worker_states
        assert [state["samples_seen"] for state in states] == [2048, 2048]
        assert [state["global_steps"] for state in states] == [4128, 4128]
        train_logs = [values for step, values in captured.dumps if step == 4128 and "train/full_kl" in values]
        assert len(train_logs) == 1
        assert train_logs[0]["train/n_envs"] == 2
        assert train_logs[0]["train/rollout_samples"] == 4096
        assert train_logs[0]["train/completed_rollouts"] == 1

        saved_policy = deepcopy(model.policy.state_dict())
        saved_optimizer = deepcopy(model.policy.optimizer.state_dict())
        checkpoint = tmp_path / "coordinator_checkpoint"
        coordinator._save(checkpoint)
        saved = AdaptiveKLPPO.load(checkpoint, device="cpu")
        assert saved.num_timesteps == 4128
        assert saved.kl_learning_rate == model.kl_learning_rate
        assert saved.kl_low_kl_count == model.kl_low_kl_count
        assert_state_equal(saved.parallel_worker_states, states)
        assert_state_equal(saved.policy.state_dict(), saved_policy)
        assert_state_equal(saved.policy.optimizer.state_dict(), saved_optimizer)
    finally:
        parallel.close()


def test_two_to_four_workers_preserves_optimizer_and_collects_4096(tmp_path):
    two = make_env(tmp_path / 'two')
    try:
        model = make_model(two, n_steps=16, batch_size=16, n_epochs=1)
        model.learn(32)
        states = two.env_method('parallel_worker_state')
        policy = deepcopy(model.policy.state_dict())
        optimizer = deepcopy(model.policy.optimizer.state_dict())
        kl = model.kl_schedule_metadata()
    finally:
        two.close()
    expanded = resize_worker_states(states, 4, seed=321)
    assert_state_equal(expanded[:2], states)
    assert [s['seed'] for s in expanded] == [321, 322, 323, 324]
    four = make_env(tmp_path / 'four', n_envs=4, worker_states=expanded, starting_steps=32)
    try:
        attach_parallel_env(model, four, n_steps=1024, batch_size=1024, n_epochs=10)
        assert_state_equal(model.policy.state_dict(), policy)
        assert_state_equal(model.policy.optimizer.state_dict(), optimizer)
        assert model.kl_schedule_metadata() == kl
        logs = CaptureLogs()
        model.set_logger(Logger(folder=None, output_formats=[logs]))
        callback = CaptureRollout()
        coordinator = ParallelTrainingCoordinator(model, four, SimpleNamespace(), tmp_path)
        coordinator.learn(4096, callback)
        assert model.num_timesteps == 4128
        assert callback.observation_shape == (1024, 4, 9)
        assert callback.action_shape == (1024, 4, 3)
        saved = model.parallel_worker_states
        assert [s['samples_seen'] for s in saved] == [1040, 1040, 1024, 1024]
        assert [s['global_steps'] for s in saved] == [4128] * 4
        assert len({str(s['np_rng_state']) for s in saved}) == 4
        assert_state_equal(resize_worker_states(saved, 2), saved[:2])
        final_logs = [v for _, v in logs.dumps if 'train/full_kl' in v][-1]
        assert final_logs['train/rollout_samples'] == 4096
        assert final_logs['train/n_envs'] == 4
        assert 0 < final_logs['train/optimizer_steps'] <= 40
    finally:
        four.close()


def test_added_worker_seeds_do_not_duplicate_saved_seeds():
    states = [dict(worker_id=0, seed=2), dict(worker_id=1, seed=3)]
    expanded = resize_worker_states(states, 4, seed=0)
    assert [s['seed'] for s in expanded] == [2, 3, 4, 5]
    assert_state_equal(states, [dict(worker_id=0, seed=2), dict(worker_id=1, seed=3)])
