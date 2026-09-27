"""Boundary and raw-unit checks of the pure transition calculations."""
from copy import deepcopy

import numpy as np
import pytest

from rate_rl.rewards import evaluate_outcome, evaluate_step_reward
from test_native_torque_env import task


GAIN = 7e-5


def reward(errors=(0., 0., 0.), torque=(0., 0., 0.), previous=(0., 0., 0.),
           failure=None):
    return evaluate_step_reward(task(tracking_bonus_threshold_deg_s=5.),
        error_deg_s=np.asarray(errors), error_scale_deg_s=5.,
        pwm_action=np.asarray([0., 1., .5, .7], dtype=np.float32),
        previous_pwm=np.full(4, .7, dtype=np.float32),
        accepted_torque=np.asarray(torque), previous_torque=np.asarray(previous),
        failure=failure)


def test_raw_component_values_sum_to_the_exact_scaled_transition():
    result = reward(errors=(3., 4., 0.), torque=(.1, 0., 0.))
    # SSE=25, B(10/s / 10/s)=1/2 on one of three axes. Four actual
    # motor outputs include saturation, but only torque limits gate income.
    raw = np.asarray([-.01 * 25., -.01 * (2000. / 49.) * (.5 / 3.), 0., 1.2])
    np.testing.assert_allclose(result.components / GAIN, raw, rtol=1e-14)
    assert result.motor_saturation_count == 2
    assert result.saturation_count == 0
    assert result.value == pytest.approx(raw.sum() * GAIN, abs=1e-18)


def test_each_signed_torque_boundary_adds_cost_and_blocks_tracking_income():
    result = reward(torque=(1., -1., 0.))
    assert result.saturation_count == 2
    assert result.saturation_penalty == .02
    assert result.diagnostics['tracking_bonus'] == 0.
    assert result.value / GAIN == pytest.approx(
        result.components.sum() / GAIN - 2. * .01 / GAIN)
    assert reward().saturation_count == 0


def test_slew_uses_first_action_from_zero_then_previous_accepted_action():
    first = reward(torque=(.1, -.2, .3))
    unchanged = reward(torque=(.1, -.2, .3), previous=(.1, -.2, .3))
    np.testing.assert_allclose(first.diagnostics['torque_rate_per_s'], [10., -20., 30.])
    assert first.components[1] < 0.
    assert unchanged.components[1] == 0.
    # Even the largest possible signed action reversal approaches but does
    # not reach the bounded per-step smoothness budget.
    largest = reward(torque=(-1., 1., -1.), previous=(1., -1., 1.))
    assert largest.components[1] / GAIN == pytest.approx(-.01 * (2000./49.) * 400./401.)
    assert abs(largest.components[1] / GAIN) < .01 * (2000./49.)


def test_sse_cap_is_shared_across_axes_and_does_not_rescale_other_components():
    result = reward(errors=(50., 50., 50.))
    assert result.diagnostics['tracking_cost'] == 5000.
    assert result.components[0] / GAIN == pytest.approx(-50.)
    np.testing.assert_array_equal(result.components[1:], [0., 0., 0.])


def state(**changes):
    return dict(q_ned_frd=np.asarray([1., 0., 0., 0.]),
                position_ned=np.asarray([0., 0., -5.]),
                rates_true=np.zeros(3), px4_position_control=True, **changes)


def outcome(error_deg, completed_steps, endpoint=None):
    error = np.deg2rad(error_deg)
    return evaluate_outcome(task(episode_seconds=20.48), endpoint or state(),
        on_rig=False, completed_steps=completed_steps, error_rad_s=error,
        previous_error_squared=float(np.mean(error**2)) * completed_steps,
        previous_axis_error_squared=error**2 * completed_steps)


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('error,success', [(5.-1e-8, True), (5., False), (5.+1e-8, False)])
def test_success_requires_strict_whole_episode_rmse_for_each_axis(axis, error, success):
    errors = np.zeros(3)
    errors[axis] = error
    before_end = outcome(errors, 2046)
    assert not before_end.terminated
    assert before_end.success_bonus == before_end.failure_penalty == 0.
    final = outcome(errors, 2047)
    assert final.terminated and final.success == success
    assert final.failure == (None if success else 'tracking_rmse')
    assert final.success_bonus / GAIN == pytest.approx(50000. if success else 0.)
    assert final.failure_penalty / GAIN == pytest.approx(0. if success else 30000.)


@pytest.mark.parametrize('step', [1, 2048])
def test_physical_failure_takes_priority_and_settles_executed_step_time(step):
    endpoint = state()
    endpoint['position_ned'][2] = -.2
    final = outcome(np.zeros(3), step - 1, endpoint)
    assert final.failure == 'ground' and final.terminated and not final.success
    assert final.success_bonus == 0.
    assert final.failure_penalty / GAIN == pytest.approx(30000.+3500.*(20.48-step*.01))
    terminal_reward = reward(failure=final.failure)
    assert terminal_reward.diagnostics['tracking_bonus'] == 0.
    total = terminal_reward.value + final.success_bonus - final.failure_penalty
    assert total / GAIN == pytest.approx(-30000.-3500.*(20.48-step*.01))


def test_pure_outcome_does_not_normalize_caller_attitude_array_in_place():
    endpoint = state()
    endpoint['q_ned_frd'] *= 2.
    snapshot = deepcopy(endpoint)
    result = outcome(np.zeros(3), 0, endpoint)
    assert result.tilt == 0.
    for key in endpoint:
        np.testing.assert_array_equal(endpoint[key], snapshot[key])
