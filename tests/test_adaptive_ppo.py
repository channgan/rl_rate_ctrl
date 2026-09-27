"""Adaptive PPO checks using a deterministic, simulator-free torque task."""

from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from torch.distributions import Independent, Normal, kl_divergence

from rate_rl.adaptive_ppo import AdaptiveKLPPO, diagonal_gaussian_kl


class SyntheticTorqueEnv(gym.Env):
    """Nine observations and three continuous actions, with no PX4 connection."""

    observation_space = gym.spaces.Box(-2.0, 2.0, (9,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

    def _observation(self):
        phase = self.elapsed * 0.17 + np.arange(9) * 0.31
        return np.sin(phase).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.elapsed = 0
        return self._observation(), {}

    def step(self, action):
        target = 0.4 * self._observation()[:3]
        reward = -float(np.square(np.asarray(action) - target).sum())
        self.elapsed += 1
        return self._observation(), reward, False, self.elapsed == 31, {}


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(model_type=AdaptiveKLPPO, **overrides):
    kwargs = dict(
        learning_rate=3e-4,
        n_steps=64,
        batch_size=16,
        n_epochs=2,
        target_kl=None,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.ReLU,
        ),
        seed=17,
        device="cpu",
    )
    kwargs.update(overrides)
    return model_type("MlpPolicy", SyntheticTorqueEnv(), **kwargs)


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_diagonal_gaussian_kl_matches_distribution_reference_and_sums_axes():
    old_mean = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.2, -0.5]], dtype=torch.float64)
    old_std = torch.tensor([[0.5, 0.8, 1.1], [0.7, 0.4, 1.2]], dtype=torch.float64)
    new_mean = torch.tensor([[0.3, 0.0, -0.1], [-0.2, 0.5, 0.4]], dtype=torch.float64)
    new_std = torch.tensor([[0.9, 0.6, 1.3], [0.5, 0.9, 0.8]], dtype=torch.float64)
    old = Independent(Normal(old_mean, old_std), 1)
    new = Independent(Normal(new_mean, new_std), 1)
    expected = kl_divergence(old, new)
    actual = diagonal_gaussian_kl(old_mean, old_std, new_mean, new_std)
    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, kl_divergence(new, old))
    torch.testing.assert_close(
        diagonal_gaussian_kl(old_mean, old_std, old_mean, old_std),
        torch.zeros(2, dtype=torch.float64),
    )


@pytest.mark.parametrize("phase, expected", [("early", 3e-4), ("late", 3e-4)])
def test_phase_default_initial_learning_rate(phase, expected):
    model = AdaptiveKLPPO("MlpPolicy", SyntheticTorqueEnv(), kl_phase=phase, device="cpu")
    assert model.kl_learning_rate == pytest.approx(expected)
    assert model.kl_low_kl_count == 0
    assert model.target_kl is None


def test_sampled_kl_guard_cannot_be_enabled_alongside_analytic_guard():
    with pytest.raises(ValueError, match="target_kl"):
        make_model(target_kl=0.02)


@pytest.mark.parametrize("hard_stop, final_kl", [(False, 0.021), (True, 0.001), (True, 0.021)])
def test_high_kl_or_hard_stop_decreases_once_and_clears_low_count(hard_stop, final_kl):
    model = make_model(learning_rate=3e-4)
    model._adapt_learning_rate(0.001, False)
    model._adapt_learning_rate(0.001, False)
    model._adapt_learning_rate(final_kl, hard_stop)
    assert model.kl_learning_rate == pytest.approx(2e-4)
    assert model.kl_low_kl_count == 0


def test_learning_rate_increase_needs_three_consecutive_normal_low_kl_updates():
    model = make_model(learning_rate=2e-4)
    for count in (1, 2):
        model._adapt_learning_rate(0.001, False)
        assert model.kl_learning_rate == pytest.approx(2e-4)
        assert model.kl_low_kl_count == count
    model._adapt_learning_rate(0.001, False)
    assert model.kl_learning_rate == pytest.approx(2.4e-4)
    assert model.kl_low_kl_count == 0


@pytest.mark.parametrize("interruption", [0.0, 0.005, 0.01, 0.02])
def test_zero_and_deadband_boundaries_reset_low_kl_streak(interruption):
    model = make_model(learning_rate=2e-4)
    model._adapt_learning_rate(0.001, False)
    model._adapt_learning_rate(0.001, False)
    model._adapt_learning_rate(interruption, False)
    assert model.kl_learning_rate == pytest.approx(2e-4)
    assert model.kl_low_kl_count == 0


@pytest.mark.parametrize("phase, lower, upper", [("early", 3e-5, 3e-4), ("late", 3e-5, 3e-4)])
def test_adaptation_obeys_phase_learning_rate_bounds(phase, lower, upper):
    model = make_model(kl_phase=phase, learning_rate=upper)
    for _ in range(6):
        model._adapt_learning_rate(0.001, False)
    assert model.kl_learning_rate == pytest.approx(upper)
    for _ in range(30):
        model._adapt_learning_rate(0.03, False)
    assert model.kl_learning_rate == pytest.approx(lower)


def test_phase_switch_clips_existing_rate_without_resetting_it_to_initial():
    model = make_model(learning_rate=2e-4)
    model._adapt_learning_rate(0.001, False)
    model.set_kl_phase("late")
    assert model.kl_phase == "late"
    assert model.kl_learning_rate == pytest.approx(2e-4)
    assert model.kl_low_kl_count == 0
    model._adapt_learning_rate(0.001, False)
    model.set_kl_phase("late")
    assert model.kl_low_kl_count == 1

    high = make_model(learning_rate=8e-4)
    high.set_kl_phase("late")
    assert high.kl_learning_rate == pytest.approx(3e-4)
    low = make_model(kl_phase="late", learning_rate=3e-5)
    low.set_kl_phase("early")
    assert low.kl_learning_rate == pytest.approx(3e-5)


def test_real_update_reports_full_rollout_old_to_new_analytic_kl():
    model = make_model(n_steps=4096, batch_size=2048, n_epochs=1)
    old_policy = deepcopy(model.policy)
    model.learn(4096)
    observations = torch.as_tensor(model.rollout_buffer.observations.reshape(-1, 9))
    with torch.no_grad():
        old = old_policy.get_distribution(observations).distribution
        new = model.policy.get_distribution(observations).distribution
        expected = kl_divergence(old, new).sum(dim=-1).mean().item()
    assert expected > 0
    assert np.isfinite(expected)
    assert model.logger.name_to_value["train/full_kl"] == pytest.approx(expected, abs=1e-7)
    assert model.logger.name_to_value["train/optimizer_steps"] == 2
    assert model.logger.name_to_value["train/actual_epochs"] == 1


def test_full_rollout_kl_does_not_advance_minibatch_shuffle_rng():
    model = make_model(n_epochs=1)
    model.learn(64)
    model._snapshot_old_policy()
    before = np.random.get_state()
    assert model._full_rollout_kl() == pytest.approx(0.0, abs=1e-12)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_regular_update_preserves_standard_ppo_losses_gradients_and_optimizer():
    baseline = make_model(PPO)
    baseline.learn(64)
    adaptive = make_model()
    adaptive.learn(64)
    assert_nested_equal(adaptive.policy.state_dict(), baseline.policy.state_dict())
    assert_nested_equal(adaptive.policy.optimizer.state_dict(), baseline.policy.optimizer.state_dict())
    for key in ("train/loss", "train/value_loss", "train/policy_gradient_loss", "train/entropy_loss"):
        assert adaptive.logger.name_to_value[key] == pytest.approx(baseline.logger.name_to_value[key])


def test_next_rollout_uses_adapted_rate_despite_sb3_schedule_writeback(monkeypatch):
    model = make_model(learning_rate=3e-4, n_epochs=1)
    monkeypatch.setattr(model, "_full_rollout_kl", lambda: 0.021)
    used_rates = []
    original_step = model.policy.optimizer.step

    def capture_step(*args, **kwargs):
        used_rates.append(model.policy.optimizer.param_groups[0]["lr"])
        return original_step(*args, **kwargs)

    monkeypatch.setattr(model.policy.optimizer, "step", capture_step)
    model.learn(64)
    assert used_rates == pytest.approx([3e-4] * 4)
    assert model.kl_learning_rate == pytest.approx(2e-4)
    assert model.logger.name_to_value["train/kl_learning_rate"] == pytest.approx(3e-4)
    assert model.logger.name_to_value["train/next_learning_rate"] == pytest.approx(2e-4)
    used_rates.clear()
    model.learn(64, reset_num_timesteps=False)
    assert used_rates == pytest.approx([2e-4] * 4)


def test_analytic_minibatch_stop_keeps_completed_update_and_dominates_low_full_kl(monkeypatch):
    model = make_model(learning_rate=3e-4, n_epochs=10)
    before = deepcopy(model.policy.state_dict())
    model._adapt_learning_rate(0.001, False)
    model._adapt_learning_rate(0.001, False)
    monkeypatch.setattr(model, "_full_rollout_kl", lambda: 0.001)
    evaluations = 0

    def trigger_after_first_step(observations):
        nonlocal evaluations
        evaluations += 1
        return 0.031 if evaluations == 2 else 0.0

    monkeypatch.setattr(model, "_mean_analytic_kl", trigger_after_first_step)
    model.learn(64)
    assert evaluations == 2
    assert model.logger.name_to_value["train/optimizer_steps"] == 1
    assert model.logger.name_to_value["train/kl_hard_stop"] == 1
    assert model.kl_learning_rate == pytest.approx(2e-4)
    assert model.kl_low_kl_count == 0
    assert any(not torch.equal(value, before[key]) for key, value in model.policy.state_dict().items())


def test_adaptive_checkpoint_restores_learning_rate_count_policy_and_optimizer(tmp_path):
    model = make_model(kl_phase="late", learning_rate=1.5e-4)
    model.learn(64)
    model._adapt_learning_rate(0.01, False)
    model._adapt_learning_rate(0.001, False)
    model._adapt_learning_rate(0.001, False)
    expected_policy = deepcopy(model.policy.state_dict())
    expected_optimizer = deepcopy(model.policy.optimizer.state_dict())
    path = tmp_path / "adaptive"
    model.save(path)
    loaded = AdaptiveKLPPO.load(path, env=SyntheticTorqueEnv(), device="cpu")
    assert loaded.kl_phase == "late"
    assert loaded.kl_learning_rate == pytest.approx(model.kl_learning_rate)
    assert loaded.kl_low_kl_count == 2
    assert_nested_equal(loaded.policy.state_dict(), expected_policy)
    assert_nested_equal(loaded.policy.optimizer.state_dict(), expected_optimizer)
    loaded._adapt_learning_rate(0.001, False)
    assert loaded.kl_learning_rate == pytest.approx(model.kl_learning_rate * 1.2)
    loaded.learn(64, reset_num_timesteps=False)
    assert loaded.num_timesteps == 128


def test_legacy_ppo_checkpoint_upgrades_without_resetting_policy_or_optimizer(tmp_path):
    legacy = make_model(PPO, learning_rate=3e-4, target_kl=None)
    legacy.learn(64)
    expected_policy = deepcopy(legacy.policy.state_dict())
    expected_optimizer = deepcopy(legacy.policy.optimizer.state_dict())
    path = tmp_path / "legacy"
    legacy.save(path)
    loaded = AdaptiveKLPPO.load(path, env=SyntheticTorqueEnv(), device="cpu", kl_phase="late")
    assert loaded.kl_phase == "late"
    assert loaded.kl_learning_rate == pytest.approx(3e-4)
    assert loaded.kl_low_kl_count == 0
    assert loaded.target_kl is None
    assert_nested_equal(loaded.policy.state_dict(), expected_policy)
    assert_nested_equal(loaded.policy.optimizer.state_dict(), expected_optimizer)
    loaded.learn(64, reset_num_timesteps=False)
    assert loaded.num_timesteps == 128
    assert any(not torch.equal(value, expected_policy[key]) for key, value in loaded.policy.state_dict().items())


@pytest.mark.parametrize("phase, rate", [("early", 2e-4), ("late", 5e-5)])
def test_legacy_phase_metadata_takes_precedence_over_rate_heuristics(tmp_path, phase, rate):
    legacy = make_model(PPO, learning_rate=rate)
    legacy.manual_training_phase = phase
    legacy.learn(64)
    path = tmp_path / phase
    legacy.save(path)
    loaded = AdaptiveKLPPO.load(path, env=SyntheticTorqueEnv(), device="cpu")
    assert loaded.kl_phase == phase
    assert loaded.kl_learning_rate == pytest.approx(rate)
    assert_nested_equal(loaded.policy.optimizer.state_dict(), legacy.policy.optimizer.state_dict())
