"""Exercise rate-only transitions without hiding native allocation behind actions.

Slew and early-failure expectations use the already-approved .02/7 and
3500/s production values. They correct stale .01/7 and 2500/s assertions;
no runtime reward change accompanies this test correction. Explicit historical
2 deg/s threshold tests remain supported independently of the public 5 deg/s.
"""
from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from rate_rl.backend import SimulatorError
from rate_rl.env import RateControlEnv, TaskConfig
from rate_rl.reward_contract import mse_reward_interface_metadata, require_reward_interface
from rate_rl import torque_contract
from backend_fixture import TestBackend


class NativeTorqueBackend(TestBackend):
    config = dict(TestBackend.config, mode='free_flight')
    px4_hover_command = .7
    reference = np.zeros(3)
    thrust_z = -.7
    allocated_pwm = np.array([.6, .65, .7, .75], dtype=np.float32)
    accepted_override = None

    def set_position_goal(self, position, heading):
        self.goal = np.array(position)
        self.heading = heading

    def state(self):
        state = super().state()
        state.update(px4_local_position=[0., 0., -3.], px4_rate_target=self.reference.copy(),
                     px4_thrust_body_z=self.thrust_z, px4_reference_us=self.t,
                     px4_now_us=self.t, px4_position_control=True,
                     px4_attitude_target=[1., 0., 0., 0.])
        return state

    def step(self, action, dt):
        raise AssertionError('Native torque mode must not send direct motor commands')

    def step_torque(self, action, dt):
        self.received_torque = np.array(action)
        self.t += round(dt * 1e6)
        self.applied_speed_fraction = self.allocated_pwm.copy()
        state = self.state()
        state['applied_torque'] = (action.copy() if self.accepted_override is None
                                   else self.accepted_override)
        # Motor-speed scaling is distinct from the normalized allocator output.
        state['applied_speed_fraction'] = .15 + .85 * self.allocated_pwm
        return state


def task(**changes):
    return replace(TaskConfig(dt=.01, native_torque=True, px4_native_outer=True,
                              waypoint_tracking=True, squared_error_reward=True,
                              terminate_on_tilt=False), **changes)


def test_native_torque_gym_contract_has_nine_inputs_and_three_signed_actions():
    backend = NativeTorqueBackend()
    env = RateControlEnv(backend, task())
    assert backend.native_torque and backend.native_outer
    assert env.observation_space.shape == (9,)
    np.testing.assert_array_equal(env.observation_space.low[6:], [-1.] * 3)
    np.testing.assert_array_equal(env.observation_space.high[6:], [1.] * 3)
    assert env.action_space.shape == (3,)
    np.testing.assert_array_equal(env.action_space.low, [-1.] * 3)
    np.testing.assert_array_equal(env.action_space.high, [1.] * 3)
    check_env(env, warn=True)


@pytest.mark.parametrize('changes', [dict(px4_native_outer=False), dict(waypoint_tracking=False),
                                    dict(squared_error_reward=False), dict(air_outer_loop=False),
                                    dict(fixed_episode_rate_target=True)])
def test_native_torque_rejects_incompatible_task_before_reset(changes):
    with pytest.raises(ValueError, match='Native torque requires'):
        RateControlEnv(NativeTorqueBackend(), task(**changes))


def test_native_torque_rejects_rig():
    backend = NativeTorqueBackend()
    backend.config = dict(backend.config, mode='ball_rig')
    with pytest.raises(ValueError, match='Native torque requires'):
        RateControlEnv(backend, task())


def test_thrust_reference_and_delta_do_not_enter_actor_or_critic_observation():
    env = RateControlEnv(NativeTorqueBackend(), task())
    original, _ = env.reset(seed=2)
    env.thrust = 100.
    env.thrust_delta = -99.
    env.current_state['px4_thrust_body_z'] = -.1
    np.testing.assert_array_equal(env._obs(env.current_state), original)
    assert original.dtype == np.float32
    np.testing.assert_array_equal(original[6:], np.zeros(3))
    np.testing.assert_allclose(env.last_action, .7)


def test_native_transition_uses_old_reference_new_truth_and_actual_allocated_pwm():
    backend = NativeTorqueBackend()
    backend.reference = np.deg2rad([6., 8., 0.])
    env = RateControlEnv(backend, task())
    initial, _ = env.reset(seed=3)
    previous_pwm = env.last_action.copy()
    old_reference = env.target.copy()
    backend.true_rates = [0., 0., 0.]
    backend.rates = np.deg2rad([.1, -.2, .3])
    backend.reference = np.array([-.3, .2, .5])
    action = np.array([.2, -.1, .3], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    assert not done and not truncated
    np.testing.assert_array_equal(backend.received_torque, action)
    np.testing.assert_allclose(info['target_rad_s'], old_reference)
    np.testing.assert_allclose(info['error_deg_s'], [6., 8., 0.])
    np.testing.assert_allclose(obs[:3], backend.rates / 5.)
    np.testing.assert_allclose(obs[3:6], (backend.reference - backend.rates) / 5.)
    np.testing.assert_array_equal(initial[6:], np.zeros(3))
    np.testing.assert_array_equal(obs[6:], action)
    np.testing.assert_array_equal(env.last_torque, action)
    np.testing.assert_array_equal(env.last_action, backend.allocated_pwm)
    np.testing.assert_array_equal(info['current_pwm_command'], backend.allocated_pwm)
    np.testing.assert_array_equal(info['torque_command'], action)
    np.testing.assert_allclose(info['applied_speed_fraction'], .15 + .85 * backend.allocated_pwm)
    pwm_rate = (backend.allocated_pwm.astype(float) - previous_pwm.astype(float)) / .01
    np.testing.assert_allclose(info['pwm_rate_per_s'], pwm_rate)
    torque_rate = action.astype(float) / .01
    np.testing.assert_allclose(info['torque_rate_per_s'], torque_rate)
    assert info['slew_rate_source'] == 'accepted_normalized_torque'
    assert info['slew_rate_cost'] == pytest.approx(np.mean(torque_rate**2))
    slew = np.mean((torque_rate / 10.)**2 / (1. + (torque_rate / 10.)**2))
    assert info['continuous_reward_components'][0] == pytest.approx(-.01 * 100. * 7e-5)
    assert info['continuous_reward_components'][1] == pytest.approx(-.01 * (.02 / 7.) * slew)
    assert info['continuous_reward_components'][2] == 0.
    assert reward == pytest.approx(sum(info['continuous_reward_components']))
    assert env.observation_space.contains(obs)
    next_obs, _, _, _, next_info = env.step(np.zeros(3))
    np.testing.assert_array_equal(next_obs[6:], np.zeros(3))
    np.testing.assert_array_equal(next_info['pwm_rate_per_s'], np.zeros(4))
    np.testing.assert_allclose(next_info['torque_rate_per_s'], -torque_rate)
    assert next_info['continuous_reward_components'][1] == pytest.approx(-.01 * (.02 / 7.) * slew)


def test_reset_clears_policy_torque_history_but_keeps_native_motor_initialization():
    env = RateControlEnv(NativeTorqueBackend(), task())
    env.reset(seed=0)
    obs = env.step([.2, -.3, .4])[0]
    assert np.any(obs[6:] != 0.)
    reset_obs, _ = env.reset(seed=1)
    np.testing.assert_array_equal(reset_obs[6:], np.zeros(3))
    np.testing.assert_array_equal(env.last_torque, np.zeros(3))
    np.testing.assert_allclose(env.last_action, env.current_state['applied_pwm'], atol=1e-7)
    info = env.step([.1, -.2, .3])[4]
    np.testing.assert_allclose(info['torque_rate_per_s'], [10., -20., 30.])


def test_native_slew_ignores_collective_motor_change_when_torque_is_unchanged():
    backend = NativeTorqueBackend()
    env = RateControlEnv(backend, task())
    env.reset(seed=0)
    action = np.array([.1, -.2, .3], dtype=np.float32)
    env.step(action)
    backend.allocated_pwm = np.array([.8, .85, .9, .95], dtype=np.float32)
    info = env.step(action)[4]
    assert np.any(info['pwm_rate_per_s'] != 0.)
    np.testing.assert_array_equal(info['torque_rate_per_s'], np.zeros(3))
    assert info['slew_rate_cost'] == 0.
    assert info['continuous_reward_components'][1] == 0.


def test_native_slew_penalizes_torque_change_even_when_allocated_motors_do_not_change():
    env = RateControlEnv(NativeTorqueBackend(), task())
    env.reset(seed=0)
    env.step(np.zeros(3))
    info = env.step([.1, 0., 0.])[4]
    np.testing.assert_array_equal(info['pwm_rate_per_s'], np.zeros(4))
    np.testing.assert_allclose(info['torque_rate_per_s'], [10., 0., 0.])
    # Apply B per axis, then average over exactly three torque axes.
    assert info['continuous_reward_components'][1] == pytest.approx(-.01 * (.02 / 7.) * .5 / 3.)
    assert info['slew_rate_cost'] == pytest.approx(100. / 3.)


@pytest.mark.parametrize('dt,action', [(.01, [.1, -.2, .3]), (.02, [.2, -.4, .6])])
def test_native_slew_is_a_time_rate_and_integrates_the_cost_over_dt(dt, action):
    env = RateControlEnv(NativeTorqueBackend(), task(dt=dt))
    env.reset(seed=0)
    info = env.step(action)[4]
    np.testing.assert_allclose(info['torque_rate_per_s'], [10., -20., 30.])
    scaled_squared = np.array([1., -2., 3.])**2
    expected_mean = np.mean(scaled_squared / (1. + scaled_squared))
    assert info['continuous_reward_components'][1] == pytest.approx(-dt * (.02 / 7.) * expected_mean)


def test_headroom_income_removed_without_rescaling_tracking_reward():
    rewards = []
    for pwm in (.3, .9):
        backend = NativeTorqueBackend()
        backend.allocated_pwm = np.full(4, pwm, dtype=np.float32)
        env = RateControlEnv(backend, task(slew_weight=0.))
        env.reset(seed=0)
        _, reward, _, _, info = env.step(np.zeros(3))
        assert info['peak_pwm_penalty'] == 0.
        assert info['continuous_reward_components'][2] == 0.
        assert info['tracking_bonus'] == pytest.approx(.000084)
        rewards.append(reward)
    assert rewards == pytest.approx([.000084, .000084])


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('boundary', [-1., -.999, .999, 1.])
def test_each_signed_torque_limit_is_charged_each_step(axis, boundary):
    env = RateControlEnv(NativeTorqueBackend(), task())
    env.reset(seed=0)
    action = np.zeros(3, dtype=np.float32)
    action[axis] = boundary
    for _ in range(2):
        _, reward, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        assert info['saturation_source'] == 'accepted_normalized_torque'
        assert info['saturation_count'] == info['torque_saturation_count'] == 1
        np.testing.assert_array_equal(info['saturated_torque_axes'], np.arange(3) == axis)
        assert info['motor_saturation_count'] == 0
        assert info['saturated_motors'].shape == (4,)
        assert info['saturation_penalty'] == pytest.approx(.01)
        assert not info['tracking_eligible'] and info['tracking_bonus'] == 0.
        assert reward == pytest.approx(sum(info['continuous_reward_components']) - .01)


@pytest.mark.parametrize('action', [[0., 0., 0.], [-.9989, .9989, .5]])
def test_zero_and_interior_torque_get_no_limit_penalty_even_with_saturated_motors(action):
    backend = NativeTorqueBackend()
    backend.allocated_pwm = np.array([0., 1., 0., 1.], dtype=np.float32)
    env = RateControlEnv(backend, task())
    env.reset(seed=0)
    _, reward, _, _, info = env.step(action)
    assert info['saturation_count'] == info['torque_saturation_count'] == 0
    np.testing.assert_array_equal(info['saturated_torque_axes'], [False] * 3)
    np.testing.assert_array_equal(info['saturated_motors'], [True] * 4)
    assert info['motor_saturation_count'] == 4
    assert info['saturation_penalty'] == 0.
    assert info['tracking_eligible'] and info['tracking_bonus'] == pytest.approx(.000084)
    assert reward == pytest.approx(sum(info['continuous_reward_components']))


def test_allocated_motor_extremes_do_not_change_native_reward_or_bonus():
    outcomes = []
    for motors in ([.5] * 4, [0., 1., 0., 1.]):
        backend = NativeTorqueBackend()
        backend.allocated_pwm = np.array(motors, dtype=np.float32)
        env = RateControlEnv(backend, task())
        env.reset(seed=0)
        _, reward, _, _, info = env.step([.1, -.2, .3])
        outcomes.append((reward, info['tracking_bonus'], info['saturation_penalty']))
    assert outcomes[0] == pytest.approx(outcomes[1])


@pytest.mark.parametrize('dt', [.01, .02])
def test_torque_limits_keep_dt_scaling_and_add_all_three_axes_on_failure(dt):
    backend = NativeTorqueBackend()
    env = RateControlEnv(backend, task(dt=dt))
    env.reset(seed=0)
    backend.true_rates = [13., 0., 0.]
    _, reward, done, truncated, info = env.step([1., -1., 1.])
    assert done and not truncated and info['failure'] == 'rate'
    assert info['saturation_count'] == 3
    assert info['saturation_penalty'] == pytest.approx(3. * dt)
    assert info['episode_saturation_penalty'] == pytest.approx(3. * dt)
    assert info['episode_saturation_fraction'] == 1.
    assert info['episode_torque_saturation_fraction'] == 1.
    assert info['episode_motor_saturation_fraction'] == 0.
    assert info['tracking_bonus'] == 0.
    assert reward == pytest.approx(sum(info['continuous_reward_components'])
                                    - 3. * dt - info['failure_penalty'])


def test_torque_and_motor_episode_fractions_use_their_own_axis_counts_and_reset():
    backend = NativeTorqueBackend()
    backend.allocated_pwm = np.array([0., 1., .4, .5], dtype=np.float32)
    env = RateControlEnv(backend, task(episode_seconds=.02))
    env.reset(seed=0)
    assert not env.step([1., 0., 0.])[2]
    backend.allocated_pwm = np.array([0., 1., 0., 1.], dtype=np.float32)
    _, reward, done, _, info = env.step([1., -1., 1.])
    assert done and info['is_success']
    assert info['episode_saturation_penalty'] == pytest.approx(.04)
    assert info['episode_saturation_fraction'] == pytest.approx(4. / (3 * 2))
    assert info['episode_torque_saturation_fraction'] == pytest.approx(4. / (3 * 2))
    assert info['episode_motor_saturation_fraction'] == pytest.approx(6. / (4 * 2))
    assert reward == pytest.approx(sum(info['continuous_reward_components']) - .03 + 3.5)
    env.reset(seed=1)
    env.step(np.zeros(3))
    new_info = env.step(np.zeros(3))[4]
    assert new_info['episode_saturation_penalty'] == 0.
    assert new_info['episode_saturation_fraction'] == 0.
    assert new_info['episode_motor_saturation_fraction'] == 1.


def test_legacy_pwm_still_charges_each_zero_or_full_motor_and_gates_bonus():
    env = RateControlEnv(TestBackend(), TaskConfig(dt=.01, episode_seconds=.01,
        target_limit_rad_s=(0., 0., 0.), squared_error_reward=True))
    env.reset(seed=0)
    _, reward, done, _, info = env.step([0., .5, 1., 0.])
    assert done and info['is_success']
    assert info['saturation_source'] == 'allocated_motor'
    assert info['saturation_count'] == info['motor_saturation_count'] == 3
    np.testing.assert_array_equal(info['saturated_motors'], [True, False, True, True])
    assert 'saturated_torque_axes' not in info and 'torque_saturation_count' not in info
    assert info['saturation_penalty'] == pytest.approx(.03)
    assert info['tracking_bonus'] == 0.
    assert info['episode_saturation_fraction'] == .75
    assert info['episode_motor_saturation_fraction'] == .75
    assert 'episode_torque_saturation_fraction' not in info
    assert reward == pytest.approx(sum(info['continuous_reward_components']) - .03 + 3.5)


@pytest.mark.parametrize('bad_action', [[0, 0, 0, 0], [np.nan, 0, 0], [-1.001, 0, 0], [0, 1.001, 0]])
def test_invalid_torque_actions_do_not_advance_simulation(bad_action):
    env = RateControlEnv(NativeTorqueBackend(), task())
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(bad_action)
    assert env.backend.t == 1_000_000 and env.steps == 0


@pytest.mark.parametrize('bad_feedback', [[0, 0, 0], [np.nan, 0, 0], [0, 0], ['bad', 0, 0]])
def test_wrong_torque_acknowledgement_never_becomes_a_training_transition(bad_feedback):
    backend = NativeTorqueBackend()
    env = RateControlEnv(backend, task())
    env.reset(seed=0)
    initial = env.step([-.1, -.2, -.3])[0]
    previous_pwm = env.last_action.copy()
    previous_time = env.last_sim_us
    backend.accepted_override = bad_feedback
    with pytest.raises(SimulatorError):
        env.step([.1, .2, .3])
    assert env.needs_reset and env.current_state is None and env.steps == 1
    assert env.last_sim_us == previous_time
    np.testing.assert_array_equal(env.last_action, previous_pwm)
    np.testing.assert_array_equal(env.last_torque, initial[6:])


def test_native_reference_fault_requires_reset_instead_of_a_reusable_partial_transition():
    backend = NativeTorqueBackend()
    env = RateControlEnv(backend, task())
    env.reset(seed=0)
    initial = env.step([-.1, -.2, -.3])[0]
    previous_pwm = env.last_action.copy()
    previous_time = env.last_sim_us
    backend.reference = np.array([np.nan, 0, 0])
    with pytest.raises(SimulatorError, match='reference'):
        env.step(np.zeros(3))
    assert env.needs_reset and env.current_state is None
    np.testing.assert_array_equal(env.last_torque, initial[6:])
    np.testing.assert_array_equal(env.last_action, previous_pwm)
    assert env.last_sim_us == previous_time
    with pytest.raises(RuntimeError):
        env.step(np.zeros(3))


def test_native_torque_preserves_actual_waypoint_switch_without_bonus():
    env = RateControlEnv(NativeTorqueBackend(), task())
    env.reset(seed=4)
    loop = env.outer_loop
    loop.position_target_ned = np.asarray(env.current_state['position_ned'])
    old_target = loop.position_target_ned.copy()
    reached_before = loop.waypoints_reached
    loop.tracking_seconds = 4.99
    _, reward, _, _, info = env.step(np.zeros(3))
    assert info['waypoint_gate_transition']['switched']
    assert loop.waypoints_reached == reached_before + 1
    assert info['next_waypoints_reached'] == reached_before + 1
    assert not np.array_equal(loop.position_target_ned, old_target)
    np.testing.assert_array_equal(env.backend.goal, loop.position_target_ned)
    assert loop.tracking_seconds == 0.
    assert info['waypoint_switch_bonus'] == 0.
    assert info['episode_waypoint_bonus'] == 0.
    assert reward == pytest.approx(sum(info['continuous_reward_components']))
    assert env.step(np.zeros(3))[4]['waypoint_switch_bonus'] == 0.


@pytest.mark.parametrize('rates,success', [([0, 0, 0], True), ([.2, .2, .2], False)])
def test_native_torque_preserves_rmse_success_and_settlement(rates, success):
    backend = NativeTorqueBackend()
    env = RateControlEnv(backend, task(episode_seconds=.01))
    env.reset(seed=0)
    backend.true_rates = rates
    _, reward, done, _, info = env.step(np.zeros(3))
    assert done and info['is_success'] == success
    assert info['success_bonus'] == pytest.approx(3.5 if success else 0.)
    assert info['failure_penalty'] == pytest.approx(0. if success else 2.1)
    assert reward == pytest.approx(sum(info['continuous_reward_components'])
                                    + info['success_bonus'] - info['failure_penalty'])


def test_native_torque_reward_contract_cannot_load_legacy_headroom_objective():
    legacy = mse_reward_interface_metadata(TaskConfig(squared_error_reward=True))
    current = mse_reward_interface_metadata(task())
    assert current['version'].endswith('_v15')
    assert current['waypoint_switch_raw_bonus'] == 0.
    assert legacy['waypoint_switch_raw_bonus'] == 1000.
    assert current['continuous_weights']['mean_headroom'] == 0.
    assert current['continuous_weights']['rate'] == legacy['continuous_weights']['rate']
    assert current['raw_settlement']['success'] == legacy['raw_settlement']['success']
    assert current['raw_settlement']['early_failure_extra_max'] == -105000.
    assert current['horizon_success']['aggregation'].startswith('each axis separately')
    assert current['slew_scale_per_s'] == 10.
    assert legacy['slew_scale_per_s'] == 50.
    assert 'accepted requested normalized torque' in current['slew_input']
    assert 'three axes' in current['slew_aggregation']
    assert 'accepted normalized torque' in current['saturation_input']
    assert current['saturation_absolute_threshold'] == .999
    assert current['saturation_cost_per_axis_per_s'] == 1.
    assert 'no saturated torque axis' in current['tracking_bonus_gate']
    with pytest.raises(ValueError, match='reward interface mismatch'):
        require_reward_interface(SimpleNamespace(reward_interface_metadata=legacy), current)


def test_native_torque_rejects_previous_v13_pwm_slew_reward():
    current = mse_reward_interface_metadata(task())
    old = deepcopy(current)
    old['version'] = 'sse10000_native_torque_no_headroom_no_waypoint_bonus_v13'
    old['slew_input'] = '(current allocated applied_pwm - previous allocated applied_pwm) / dt'
    del old['slew_aggregation'], old['slew_history_initialization']
    with pytest.raises(ValueError, match='reward interface mismatch'):
        require_reward_interface(SimpleNamespace(reward_interface_metadata=old), current)


def test_native_torque_rejects_previous_v14_motor_saturation_objective():
    current = mse_reward_interface_metadata(task())
    old = deepcopy(current)
    old['version'] = 'sse10000_native_torque_slew_no_headroom_no_waypoint_bonus_v14'
    old['saturation_input'] = 'current allocated applied_pwm, each motor independently'
    for key in ('saturation_absolute_threshold', 'saturation_comparison',
                'saturation_cost_per_axis_per_s', 'motor_saturation_role', 'tracking_bonus_gate'):
        del old[key]
    with pytest.raises(ValueError, match='reward interface mismatch'):
        require_reward_interface(SimpleNamespace(reward_interface_metadata=old), current)


@pytest.mark.parametrize('old_identity', [
    'rate_only_allocated_pwm_no_headroom_no_waypoint_bonus_v2',
    'rate_only_torque_slew_no_headroom_no_waypoint_bonus_v3'])
def test_native_torque_rejects_previous_reward_identity_even_with_current_dimensions(old_identity):
    saved = SimpleNamespace(action_contract=torque_contract.ACTION_CONTRACT,
                            critic_contract=torque_contract.CRITIC_CONTRACT,
                            reward_contract=old_identity)
    with pytest.raises(ValueError, match='matching 9D observation / 3D torque checkpoint'):
        torque_contract.require_training_contract(saved)


@pytest.mark.parametrize('error_deg_s,expected_cost', [([3., 4., 12.], 169.),
                                                     ([100., 100., 100.], 5000.)])
def test_native_torque_retains_degree_error_sum_of_squares_and_shared_cap(error_deg_s, expected_cost):
    backend = NativeTorqueBackend()
    env = RateControlEnv(backend, task())
    env.reset(seed=0)
    backend.true_rates = -np.deg2rad(error_deg_s)
    info = env.step(np.zeros(3))[4]
    assert info['tracking_cost'] == pytest.approx(expected_cost)
    assert info['continuous_reward_components'][0] == pytest.approx(-.01 * expected_cost * 7e-5)


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('error,eligible', [(1.99, True), (2., False), (2.01, False), (-2., False), (4., False)])
def test_instant_tracking_bonus_strict_two_degrees_each_axis(axis, error, eligible):
    backend = NativeTorqueBackend()
    env = RateControlEnv(backend, task())
    env.reset(seed=0)
    errors = np.zeros(3)
    errors[axis] = error
    backend.true_rates = -np.deg2rad(errors)
    info = env.step(np.zeros(3))[4]
    assert info['tracking_eligible'] == eligible
    assert info['tracking_bonus'] == pytest.approx(.000084 if eligible else 0.)


def test_native_torque_retains_early_failure_settlement():
    backend = NativeTorqueBackend()
    env = RateControlEnv(backend, task())
    env.reset(seed=0)
    backend.true_rates = [13., 0., 0.]
    _, reward, terminated, truncated, info = env.step(np.zeros(3))
    assert terminated and not truncated and info['failure'] == 'rate'
    assert info['failure_penalty'] == pytest.approx((30000. + 3500. * (30. - .01)) * 7e-5)
    assert info['success_bonus'] == 0.
    assert reward == pytest.approx(sum(info['continuous_reward_components']) - info['failure_penalty'])


def test_tracking_income_is_ten_times_threshold_sse_and_slew_is_reduced():
    c = task()
    assert c.tracking_bonus_weight * c.dt / 7e-5 == pytest.approx(10 * c.dt * (3 * 2**2))
    assert c.slew_weight == pytest.approx(.02 / 7.)
    assert c.reward_slew_rate_scale_per_s == 10.
    expected = mse_reward_interface_metadata(c)
    old = deepcopy(expected)
    old['continuous_weights']['tracking_bonus'] = .1
    old['continuous_weights']['slew'] = .1
    with pytest.raises(ValueError):
        require_reward_interface(SimpleNamespace(reward_interface_metadata=old), expected)


@pytest.mark.parametrize('threshold,eligible', [(5., True), (2., False)])
def test_phase_threshold_changes_eligibility_without_changing_bonus_size(threshold, eligible):
    backend = NativeTorqueBackend()
    env = RateControlEnv(backend, task(tracking_bonus_threshold_deg_s=threshold))
    env.reset(seed=0)
    backend.true_rates = -np.deg2rad([3., 3., 3.])
    info = env.step(np.zeros(3))[4]
    assert info['tracking_eligible'] == eligible
    assert info['tracking_bonus'] / 7e-5 == pytest.approx(1.2 if eligible else 0.)


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('errors,success', [([4.9,4.9],True),([5.,5.],False),([7.,0.],True),([8.,0.],False)])
def test_horizon_requires_each_axis_full_episode_rmse(axis, errors, success):
    backend = NativeTorqueBackend()
    env = RateControlEnv(backend, task(episode_seconds=.02))
    env.reset(seed=0)
    for i, error in enumerate(errors):
        rates = np.zeros(3)
        rates[axis] = np.deg2rad(error)
        backend.true_rates = rates
        _, _, done, _, info = env.step(np.zeros(3))
        assert done == (i == 1)
    assert info['episode_rate_rmse_deg_s'] < 5.
    assert info['is_success'] == success
    assert info['success_bonus'] == pytest.approx(3.5 if success else 0.)
    assert info['failure_penalty'] == pytest.approx(0. if success else 2.1)


def test_native_default_slew_scale_changes_without_changing_legacy_or_explicit_values():
    assert task().reward_slew_rate_scale_per_s == 10.
    assert TaskConfig().reward_slew_rate_scale_per_s == 50.
    explicit = task(reward_slew_rate_scale_per_s=50.)
    current = mse_reward_interface_metadata(task())
    saved = mse_reward_interface_metadata(explicit)
    assert explicit.reward_slew_rate_scale_per_s == 50.
    with pytest.raises(ValueError, match='reward interface mismatch'):
        require_reward_interface(SimpleNamespace(reward_interface_metadata=saved), current)
