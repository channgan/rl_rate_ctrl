"""Real SITL interface diagnostic using a fixed PI rate controller, not an RL policy."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from rate_rl.backend import GazeboPX4Backend
from rate_rl.env import RateControlEnv, TaskConfig
from rate_rl.trajectory import EpisodeTrajectory, TrajectoryClock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', default=str(Path.home() / 'rl_rate/runtime.free_flight.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    backend = GazeboPX4Backend(args.runtime, args.out/'episodes', instance=42)
    config = TaskConfig(dt=.01, waypoint_tracking=True, px4_native_outer=True,
                        native_torque=True, squared_error_reward=True, terminate_on_tilt=False,
                        air_target_limit_rad_s=(1.,1.,1.))
    env = RateControlEnv(backend, config)
    recorder = EpisodeTrajectory(env, TrajectoryClock(args.out / 'trajectories', interval=args.steps,
        label='Fixed PI interface validation (not PPO)', stochastic_actions=False))
    report = dict(controller='fixed PI for interface validation; not a trained policy',
                  formal_training_started=False, passed=False)
    rows = []
    try:
        obs, info = recorder.reset(seed=2026)
        assert obs.shape == (9,) and env.action_space.shape == (3,)
        np.testing.assert_array_equal(obs[6:9], np.zeros(3))
        state = env.current_state
        initial_us = state['sim_us']
        snap1 = backend.snapshot()
        time.sleep(.05)
        snap2 = backend.snapshot()
        assert snap1['sim_us'] == snap2['sim_us'] == initial_us
        assert snap1['action_seq'] == snap2['action_seq'] == 0
        origin = np.array(state['position_ned'])
        goal = origin.copy()
        env.outer_loop.tracking_hold_seconds = 1e9  # Diagnostic goals are scheduled below.
        env.outer_loop._set_waypoint(goal, state['position_ned'])
        integral = np.zeros(3)
        for k in range(args.steps):
            if k in (1000, 2000):
                goal = origin.copy()
                goal[2] = -6. if k == 1000 else -4.
                env.outer_loop._set_waypoint(goal, env.current_state['position_ned'])
            error = env.target - np.asarray(env.current_state['rates'])
            integral = np.clip(integral + error * config.dt, -.3, .3)
            action = np.clip(np.array([.15,.15,.2])*error + np.array([.2,.2,.1])*integral, -1., 1.)
            previous_torque = obs[6:9].astype(float).copy()
            obs, reward, done, truncated, info = recorder.step(action)
            state = env.current_state
            assert state['sim_us'] == initial_us + (k+1)*10000
            assert state['seq'] == state['action_seq'] == k+1
            np.testing.assert_allclose(obs[6:9], action, atol=1e-7)
            assert info['current_pwm_command'].shape == (4,)
            assert info['waypoint_switch_bonus'] == 0.
            np.testing.assert_allclose(state['applied_torque'], action, atol=1e-7)
            expected_rate = (action.astype(np.float32).astype(float) - previous_torque) / config.dt
            np.testing.assert_allclose(info['torque_rate_per_s'], expected_rate, atol=1e-6)
            scaled_slew = (expected_rate / config.reward_slew_rate_scale_per_s) ** 2
            bounded_slew = float(np.mean(scaled_slew / (1. + scaled_slew)))
            np.testing.assert_allclose(info['continuous_reward_components'][1],
                -config.dt * config.slew_weight * bounded_slew, atol=1e-10)
            expected_limited = np.abs(action.astype(np.float32)) >= 1. - config.saturation_epsilon
            np.testing.assert_array_equal(info['saturated_torque_axes'], expected_limited)
            assert info['saturation_count'] == info['torque_saturation_count'] == int(expected_limited.sum())
            np.testing.assert_allclose(info['saturation_penalty'],
                config.dt * config.saturation_cost_per_motor_per_s * int(expected_limited.sum()), atol=1e-10)
            allocated = info['current_pwm_command']
            expected_motors = (allocated <= config.saturation_epsilon) | (allocated >= 1. - config.saturation_epsilon)
            np.testing.assert_array_equal(info['saturated_motors'], expected_motors)
            assert info['motor_saturation_count'] == int(expected_motors.sum())
            rows.append(dict(step=k+1, height=info['altitude_m'], target_height=-float(goal[2]),
                rate_error_deg_s=info['error_deg_s'].tolist(), torque=action.tolist(),
                torque_rate_per_s=info['torque_rate_per_s'].tolist(),
                torque_slew_scaled_reward=float(info['continuous_reward_components'][1]),
                pwm=info['current_pwm_command'].tolist(), speed=info['applied_speed_fraction'].tolist(),
                native_thrust_z=state['px4_thrust_body_z'], reward=reward))
            if (k+1) % 500 == 0:
                print('PROGRESS', k+1, 'height',info['altitude_m'],flush=True)
            if done or truncated:
                assert k+1 == 3000, f"Premature termination: {info['failure']}"
                break
        assert len(rows) == args.steps
        report.update(transitions=len(rows), observation_size=9, action_size=3,
                      observation_history='previous accepted policy torque, zero at reset',
                      torque_slew_formula_verified=True,
                      torque_limit_penalty_and_motor_diagnostics_verified=True,
                      torque_slew_scale_per_s=config.reward_slew_rate_scale_per_s,
                      slew_weight=config.slew_weight,
                      initial=initial_us, final_snapshot=state,
                      max_height=max(row['height'] for row in rows),
                      endpoint_height=rows[-1]['height'],
                      endpoint_rate_error_deg_s=rows[-1]['rate_error_deg_s'])
        if args.steps == 3000:
            for start, end, height in [(800,1000,5.),(1800,2000,6.),(2800,3000,4.)]:
                measured=float(np.mean([row['height'] for row in rows[start:end]]))
                assert abs(measured-height)<.5, f'Native altitude response off target: {measured} vs {height}'
            report['native_height_steps_5_6_4_passed'] = True
        report['passed'] = True
    finally:
        recorder.close()
        (args.out/'report.json').write_text(json.dumps(report,indent=2))
        (args.out/'trajectory.json').write_text(json.dumps(rows))
    print('COMPLETE',args.out,flush=True)


if __name__ == '__main__':
    main()
