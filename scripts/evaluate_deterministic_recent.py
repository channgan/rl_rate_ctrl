"""Read-only checkpoint evaluation on a separate PX4 instance and partition."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from stable_baselines3 import PPO

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from rate_rl.backend import GazeboPX4Backend
from rate_rl.env import RateControlEnv, TaskConfig
from rate_rl.waypoint import waypoint_gate_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--runtime', default=str(Path.home() / 'rl_rate/runtime.free_flight.json'))
    parser.add_argument('--suffix', default='')
    parser.add_argument('--instance', type=int, default=45,
                        help='Dedicated PX4 instance; defaults outside training instances 41-44')
    args = parser.parse_args()
    if not 0 <= args.instance <= 100:
        parser.error('instance must be between 0 and 100')
    run = args.run.resolve()
    source = args.checkpoint or max((run / 'checkpoints').glob('ppo_*_steps.zip'),
                 key=lambda p: int(p.stem.split('_')[1]))
    out = run / 'performance_analysis' / ('deterministic_' + source.stem + args.suffix)
    out.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    model = PPO.load(source, device='cpu')
    assert model.waypoint_gate_metadata == waypoint_gate_metadata(), 'Evaluation gate differs from checkpoint'
    config = TaskConfig(**json.loads((run / 'config.json').read_text())['task'])
    gain = config.reward_gain
    from rate_rl.reward_contract import mse_reward_interface_metadata, require_reward_interface
    require_reward_interface(model, mse_reward_interface_metadata(config))
    if config.native_torque:
        from rate_rl.torque_contract import require_training_contract
        require_training_contract(model)
    backend = GazeboPX4Backend(args.runtime, out / 'episodes', instance=args.instance)
    env = RateControlEnv(backend, config)
    report = dict(checkpoint=str(source), sha256=digest, steps=model.num_timesteps,
                  deterministic=True, runtime=args.runtime, gyro_noise=backend.config.get('gyro_noise'),
                  waypoint_gate=waypoint_gate_metadata(), instance=args.instance, episodes=[])
    print('CHECKPOINT', source, flush=True)
    try:
        for seed in [101, 202, 303]:
            started = time.time()
            obs, _ = env.reset(seed=seed)
            rows = []
            total = 0.
            streak = longest = 0.
            switches = []
            longest_gate_streak = 0.
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                err = info['error_deg_s']
                streak = streak + config.dt if np.all(np.abs(err) < 5) else 0.
                longest = max(longest, streak)
                gate = info.get('waypoint_gate_transition', {})
                longest_gate_streak = max(longest_gate_streak, gate.get('settled_seconds', 0.))
                if gate.get('switched'):
                    switches.append(dict(seconds=env.steps * config.dt, **gate))
                rows.append([env.steps * config.dt, *np.rad2deg(info['target_rad_s']),
                             *np.rad2deg(info['rates_true_rad_s']), *info['current_pwm_command'],
                             float(np.linalg.norm(env.outer_loop.position_target_ned - env.current_state['position_ned'])), info['altitude_m']])
                if config.native_torque:
                    rows[-1].extend(info['torque_command'])
                if env.steps % 1000 == 0:
                    print('PROGRESS', seed, env.steps, flush=True)
                if terminated or truncated:
                    break
            a = np.array(rows)
            result = dict(seed=seed, steps=env.steps, success=bool(info['is_success']),
                          outcome=info['failure'], rmse=info['episode_rate_rmse_deg_s'],
                          axis_rmse=info['episode_axis_rate_rmse_deg_s'].tolist(),
                          within5=float(np.mean(np.all(np.abs(a[:, 4:7]-a[:, 1:4]) < 5, axis=1))),
                          mean_abs_within5=float(np.mean(np.mean(np.abs(a[:, 4:7]-a[:, 1:4]), axis=1) < 5)),
                          longest_gate_streak_s=longest_gate_streak, waypoint_switches=switches,
                          longest_streak_s=longest, waypoints=env.outer_loop.waypoints_reached,
                          pwm_delta_rms=float(np.sqrt(np.mean(np.diff(a[:, 7:11], axis=0)**2))),
                          saturation_fraction=info['episode_saturation_fraction'],
                          saturation_source=info.get('saturation_source', 'allocated_motor'),
                          motor_saturation_fraction=info.get('episode_motor_saturation_fraction',
                                                             info['episode_saturation_fraction']),
                          scaled_return=total, raw_return=total / gain, wall_seconds=time.time()-started,
                          final_distance_m=float(a[-1, 11]), max_height_m=float(a[:, 12].max()))
            if config.native_torque:
                result['torque_saturation_fraction'] = info['episode_saturation_fraction']
                result['torque_delta_rms'] = float(np.sqrt(np.mean(np.diff(a[:, 13:16], axis=0)**2)))
            components = np.asarray(info['episode_reward_components']) / gain
            result['raw_reward_components'] = dict(rate=float(components[0]), slew=float(components[1]),
                headroom=float(components[2]), tracking=float(components[3]),
                saturation=-info['episode_saturation_penalty']/gain,
                failure=-info['episode_failure_penalty']/gain,
                success=info['episode_success_bonus']/gain)
            result['raw_reward_components']['error_progress'] = info.get('episode_error_progress_reward', 0.) / gain
            assert np.isclose(sum(result['raw_reward_components'].values()), result['raw_return'])
            with (out / f'seed_{seed}.csv').open('w') as f:
                writer = csv.writer(f)
                writer.writerow(['seconds','target_roll','target_pitch','target_yaw','roll','pitch','yaw',
                                 'pwm0','pwm1','pwm2','pwm3','distance_m','altitude_m']
                                + (['torque_roll','torque_pitch','torque_yaw'] if config.native_torque else []))
                writer.writerows(rows)
            report['episodes'].append(result)
            (out / 'report.json').write_text(json.dumps(report, indent=2))
            print('RESULT', json.dumps(result), flush=True)
    finally:
        env.close()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
