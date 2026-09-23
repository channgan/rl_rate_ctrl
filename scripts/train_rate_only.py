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
    p.add_argument('--phase', choices=['early', 'late'], default='early')
    p.add_argument('--resume', type=Path)
    p.add_argument('--runtime', default=str(Path.home() / 'rl_rate/runtime.free_flight.json'))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--allow-failure2700-change', action='store_true')
    p.add_argument('--allow-axis-success-change', action='store_true')
    args = p.parse_args()
    args.run = args.run.resolve()
    if args.resume:
        args.resume = args.resume.resolve()
    if args.steps <= 0:
        p.error('steps must be positive')
    if args.run.exists():
        p.error('Use a new run directory to preserve prior models/logs')
    command = [sys.executable, '-B', '-u', '-m', 'rate_rl.train',
        '--run', str(args.run), '--steps', str(args.steps), '--runtime', args.runtime,
        '--training-phase', args.phase, '--stage', 'free_flight', '--native-torque',
        '--simple-critic', '--px4-native-outer', '--waypoint-tracking', '--squared-error-reward',
        '--control-dt', '.01', '--torque-slew-scale', '10', '--slew-weight', str(.01 / 7.),
        '--air-rate-limits', '1', '1', '1', '--no-tilt-termination', '--compact-logs']
    if args.resume:
        command += ['--resume', str(args.resume), '--allow-phase-change', '--allow-tracking-tuning']
    if args.smoke:
        command += ['--smoke']
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
