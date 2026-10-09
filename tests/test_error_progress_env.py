"""Integration checks: progress reaches PPO exactly once with correct history."""
from types import SimpleNamespace

import numpy as np
import pytest

from rate_rl.env import RateControlEnv
from rate_rl.reward_contract import mse_reward_interface_metadata, require_reward_interface
from test_native_torque_env import NativeTorqueBackend, task


def test_first_step_terminal_step_accounting_and_reset():
    backend = NativeTorqueBackend()
    backend.reference = np.deg2rad([5., 5., 5.])
    env = RateControlEnv(backend, task(episode_seconds=.02, error_progress_weight=.003))
    env.reset(seed=101)
    total = 0.
    for rate, expected_raw in [(1., .081), (2., .063)]:
        backend.true_rates = np.deg2rad([rate] * 3)
        # Only truth enters reward; noisy actor feedback is intentionally different.
        backend.rates = np.deg2rad([20.] * 3)
        _, r, terminated, truncated, info = env.step(np.zeros(3))
        assert info['error_progress_raw_reward'] == pytest.approx(expected_raw)
        assert info['error_progress_reward'] == pytest.approx(expected_raw * 7e-5)
        assert r == pytest.approx(sum(info['continuous_reward_components'])
            - info['saturation_penalty'] + info['success_bonus'] - info['failure_penalty']
            + info['error_progress_reward'])
        total += r
    assert terminated and not truncated
    assert info['episode_error_progress_reward'] / 7e-5 == pytest.approx(.144)
    assert total == pytest.approx(sum(info['episode_reward_components'])
        - info['episode_saturation_penalty'] - info['episode_failure_penalty']
        + info['episode_success_bonus'] + info['episode_error_progress_reward'])
    env.reset(seed=101)
    backend.true_rates = np.deg2rad([1.] * 3)
    assert env.step(np.zeros(3))[4]['error_progress_raw_reward'] == pytest.approx(.081)


def test_reference_change_compares_consecutive_reward_errors_not_next_reference():
    backend = NativeTorqueBackend()
    backend.reference = np.deg2rad([5., 0., 0.])
    env = RateControlEnv(backend, task(error_progress_weight=.003))
    env.reset(seed=3)
    backend.reference = np.deg2rad([10., 0., 0.])
    first = env.step(np.zeros(3))[4]
    assert first['error_progress_raw_reward'] == pytest.approx(0.)
    assert first['error_progress_current_sse'] == pytest.approx(25.)
    second = env.step(np.zeros(3))[4]
    assert second['error_progress_raw_reward'] == pytest.approx(.003 * (25.-100.))


def test_terminated_failure_keeps_progress_and_original_settlement():
    backend = NativeTorqueBackend()
    backend.reference = np.zeros(3)
    env = RateControlEnv(backend, task(episode_seconds=.01, error_progress_weight=.003))
    env.reset(seed=3)
    backend.true_rates = np.deg2rad([10., 0., 0.])
    _, _, done, _, info = env.step(np.zeros(3))
    assert done and info['failure'] == 'tracking_rmse'
    assert info['failure_penalty'] / 7e-5 == pytest.approx(30000.)
    assert info['episode_error_progress_reward'] / 7e-5 == pytest.approx(-.3)


def test_old_objective_cannot_be_silently_resumed_with_progress_enabled():
    old = mse_reward_interface_metadata(task())
    new = mse_reward_interface_metadata(task(error_progress_weight=.003))
    assert 'error_progress' not in old
    assert new['error_progress']['weight'] == .003
    with pytest.raises(ValueError, match='reward interface mismatch'):
        require_reward_interface(SimpleNamespace(reward_interface_metadata=old), new)


@pytest.mark.parametrize('gain', [7e-5, 1.])
def test_delta_cap_is_applied_before_gain_and_does_not_clip_history(gain):
    backend = NativeTorqueBackend()
    backend.reference = np.zeros(3)
    env = RateControlEnv(backend, task(error_progress_weight=.003,
        error_progress_delta_cap=1000., reward_gain=gain))
    env.reset(seed=101)
    for rates, raw in [([100., 0., 0.], -3.), ([0., 0., 0.], 3.), ([1., 0., 0.], -.003)]:
        backend.true_rates = np.deg2rad(rates)
        _, reward, _, _, info = env.step(np.zeros(3))
        assert info['error_progress_raw_reward'] == pytest.approx(raw)
        assert info['error_progress_reward'] == pytest.approx(raw * gain)
        assert info['error_progress_max_abs_raw_reward'] == 3.
        assert reward == pytest.approx(sum(info['continuous_reward_components'])
            - info['saturation_penalty'] + info['success_bonus'] - info['failure_penalty']
            + raw * gain)


def test_cap_is_bound_into_checkpoint_objective():
    old = mse_reward_interface_metadata(task(error_progress_weight=.003, reward_gain=1.))
    new = mse_reward_interface_metadata(task(error_progress_weight=.003, reward_gain=1.,
        error_progress_delta_cap=1000.))
    assert new['error_progress']['cap']['raw_reward_absolute_max'] == 3.
    with pytest.raises(ValueError, match='reward interface mismatch'):
        require_reward_interface(SimpleNamespace(reward_interface_metadata=old), new)


def test_current_recipe_final_progress_limit_is_ten_in_both_directions():
    from rate_rl.defaults import CURRENT
    backend = NativeTorqueBackend()
    backend.reference = np.zeros(3)
    env = RateControlEnv(backend, task(error_progress_weight=CURRENT.error_progress_weight,
        error_progress_delta_cap=CURRENT.error_progress_delta_cap, reward_gain=CURRENT.reward_gain))
    env.reset(seed=101)
    for rates, expected in [([100., 0., 0.], -10.), ([0., 0., 0.], 10.)]:
        backend.true_rates = np.deg2rad(rates)
        info = env.step(np.zeros(3))[4]
        assert info['error_progress_reward'] == pytest.approx(expected * CURRENT.reward_gain)
        assert info['error_progress_raw_reward'] == expected
