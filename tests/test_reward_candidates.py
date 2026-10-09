"""Staged progress reward: signs, units, temporal history and disabled mode."""
import numpy as np
import pytest

from rate_rl.reward_candidates import ErrorProgressReward


def test_improvement_deterioration_and_unchanged_error():
    reward = ErrorProgressReward(weight=1.)
    reward.reset([5., 5., 5.])
    improved = reward.step([4., 4., 4.])
    assert improved.previous_sse == 75.
    assert improved.current_sse == 48.
    assert improved.cost_delta == -27.
    assert improved.raw_reward == 27.
    assert improved.ppo_reward == pytest.approx(27. * 7e-5)
    assert reward.step([-4., 4., -4.]).raw_reward == 0.
    assert reward.step([5., 5., 5.]).raw_reward == -27.


def test_first_sample_without_reset_error_and_episode_reset():
    reward = ErrorProgressReward(weight=1.)
    assert reward.step([100., 0., 0.]).raw_reward == 0.
    assert reward.step([0., 0., 0.]).raw_reward == 10000.
    reward.reset()
    result = reward.step([3., 4., 0.])
    assert result.previous_sse is None
    assert result.raw_reward == 0.


def test_approved_candidate_weight_and_single_ppo_scaling():
    reward = ErrorProgressReward()
    reward.reset([5., 5., 5.])
    result = reward.step([4., 4., 4.])
    assert result.raw_reward == pytest.approx(.081)
    assert result.ppo_reward == pytest.approx(.081 * 7e-5)


def test_zero_weight_is_disabled_and_does_not_mutate_input():
    reward = ErrorProgressReward(weight=0.)
    error = np.array([3., 4., 0.])
    reward.reset(error)
    error[:] = 0.
    result = reward.step(error)
    assert result.previous_sse == 25.
    assert result.cost_delta == -25.
    assert result.raw_reward == result.ppo_reward == 0.
    np.testing.assert_array_equal(error, np.zeros(3))


def test_undiscounted_sum_telescopes_and_is_not_an_added_sse_penalty():
    reward = ErrorProgressReward(weight=.5)
    reward.reset([10., 0., 0.])
    total = sum(reward.step(e).raw_reward for e in
                ([20., 0., 0.], [1., 0., 0.], [4., 0., 0.]))
    assert total == .5 * (100. - 16.)


@pytest.mark.parametrize('value', [[], [1., 2.], [np.nan, 0., 0.],
                                  [np.inf, 0., 0.], [1e308, 0., 0.]])
def test_invalid_error_does_not_advance_history(value):
    reward = ErrorProgressReward(weight=1.)
    reward.reset([3., 4., 0.])
    with pytest.raises(ValueError):
        reward.step(value)
    assert reward.step([0., 0., 0.]).raw_reward == 25.


@pytest.mark.parametrize('weight', [-1., np.nan, np.inf])
def test_invalid_weight(weight):
    with pytest.raises(ValueError):
        ErrorProgressReward(weight)


def test_symmetric_raw_clip_before_gain_and_history_uses_actual_error():
    reward = ErrorProgressReward(reward_gain=7e-5)
    reward.reset([100.] * 3)
    result = reward.step([5.] * 3, max_abs_raw_reward=50.)
    assert result.unclipped_raw_reward == pytest.approx(89.775)
    assert result.raw_reward == 50.
    assert result.ppo_reward == pytest.approx(50. * 7e-5)
    result = reward.step([100.] * 3, max_abs_raw_reward=50.)
    assert result.raw_reward == -50.
    # Zero is a valid limit; history must still advance to the measured error.
    assert reward.step([5.] * 3, max_abs_raw_reward=0.).raw_reward == 0.
    assert reward.step([4.] * 3).raw_reward == pytest.approx(.081)


@pytest.mark.parametrize('limit', [-1., np.inf, np.nan])
def test_bad_clip_does_not_advance_history(limit):
    reward = ErrorProgressReward()
    reward.reset([5.] * 3)
    with pytest.raises(ValueError):
        reward.step([4.] * 3, max_abs_raw_reward=limit)
    assert reward.step([4.] * 3).raw_reward == pytest.approx(.081)
