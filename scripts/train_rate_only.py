"""Explicit new-task entry point: native PX4 thrust/allocator, learned rate torque."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--phase', choices=['early', 'late'], default='late')
    p.add_argument('--resume', type=Path)
    p.add_argument('--runtime', default=str(Path.home() / 'rl_rate/runtime.free_flight.json'))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--n-envs', type=int, choices=[1, 2, 4], default=1)
    p.add_argument('--n-steps', type=int, help='Per-environment rollout length; default total samples=4096')
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--n-epochs', type=int, default=10)
    p.add_argument('--base-instance', type=int, default=41)
    p.add_argument('--episode-steps', type=int, default=2048)
    p.add_argument('--allow-episode-length-change', action='store_true')
    p.add_argument('--allow-failure2700-change', action='store_true')
    p.add_argument('--allow-axis-success-change', action='store_true')
    args = p.parse_args()
    args.run = args.run.resolve()
    if args.resume:
        args.resume = args.resume.resolve()
    if args.steps <= 0:
        p.error('steps must be positive')
    if args.episode_steps < 1:
        p.error('episode-steps must be positive')
    if args.allow_episode_length_change and not args.resume:
        p.error('Episode length migration requires --resume')
    n_steps = args.n_steps if args.n_steps is not None else (4096 // args.n_envs)
    if n_steps < 2 or args.batch_size < 2 or args.n_epochs < 1 or n_steps * args.n_envs % args.batch_size:
        p.error('Positive rollout/epochs required; batch-size must divide total samples')
    if args.run.exists():
        p.error('Use a new run directory to preserve prior models/logs')
    command = [sys.executable, '-B', '-u', '-m', 'rate_rl.train',
        '--run', str(args.run), '--steps', str(args.steps), '--runtime', args.runtime,
        '--training-phase', args.phase, '--stage', 'free_flight', '--native-torque',
        '--simple-critic', '--px4-native-outer', '--waypoint-tracking', '--squared-error-reward',
        '--control-dt', '.01', '--torque-slew-scale', '10', '--slew-weight', str(.02 / 7.),
        '--air-rate-limits', '1', '1', '1', '--no-tilt-termination', '--compact-logs',
        '--n-envs', str(args.n_envs), '--n-steps', str(n_steps),
        '--batch-size', str(args.batch_size), '--n-epochs', str(args.n_epochs),
        '--base-instance', str(args.base_instance), '--episode-steps', str(args.episode_steps)]
    if args.resume:
        command += ['--resume', str(args.resume), '--allow-phase-change', '--allow-tracking-tuning']
    if args.smoke:
        command += ['--smoke']
    if args.allow_episode_length_change:
        command += ['--allow-episode-length-change']
    if args.allow_axis_success_change:
        command += ['--allow-axis-success-change']
    if args.allow_failure2700_change:
        if not args.resume:
            p.error('Failure settlement migration requires --resume')
        command += ['--allow-failure2700-change']
    args.run.mkdir(parents=True)
    job_path = args.run / 'job.json'
    job = dict(status='starting', supervisor_pid=os.getpid(), phase=args.phase,
               requested_steps=args.steps, resume=str(args.resume) if args.resume else None,
               n_envs=args.n_envs, n_steps=n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
               episode_steps=args.episode_steps,
               started_utc=datetime.now(timezone.utc).isoformat(), command=command,
               automatic_phase_switch=False, publishers={})

    def save_job():
        temporary = job_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(job, indent=2))
        temporary.replace(job_path)

    def publish(script):
        with (args.run / (script + '.log')).open('w') as output:
            process = subprocess.Popen([sys.executable, '-u', str(PROJECT / 'scripts' / (script + '.py')),
                                        '--run', str(args.run)], cwd=PROJECT,
                                       stdout=output, stderr=subprocess.STDOUT)
        job['publishers'][script] = process.pid
        save_job()
        return process

    save_job()
    with (args.run / 'training.log').open('w') as output:
        process = subprocess.Popen(command, cwd=PROJECT, stdout=output, stderr=subprocess.STDOUT)
    job.update(status='running', training_pid=process.pid)
    save_job()
    publishers = [publish('tensorboard_trajectories')]
    while process.poll() is None and not (args.run / 'config.json').is_file():
        time.sleep(1)
    if (args.run / 'config.json').is_file():
        publishers.append(publish('tensorboard_raw_rewards'))
    returncode = process.wait()
    job.update(status='completed' if returncode == 0 else 'failed', returncode=returncode,
               ended_utc=datetime.now(timezone.utc).isoformat())
    save_job()
    # Publishers consume the last completed event/image before exiting.
    for publisher in publishers:
        try:
            publisher.wait(timeout=30)
        except subprocess.TimeoutExpired:
            publisher.terminate()
            publisher.wait(timeout=5)
    if returncode:
        raise SystemExit(returncode)


if __name__ == '__main__':
    main()
