"""Finish the active exploration, then resume to a cumulative step budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--total', type=int, required=True)
    parser.add_argument('--phase', choices=['early', 'late'], default='early')
    args = parser.parse_args()
    source, run = args.source.resolve(), args.run.resolve()
    if run.exists() or args.total <= 0:
        parser.error('Continuation directory must be new and total positive')
    record_path = source / 'exploration_extension.json'
    record = dict(status='waiting_for_source', pid=os.getpid(), source=str(source),
                  continuation=str(run), total_steps=args.total, phase=args.phase,
                  automatic_phase_switch=args.phase == 'late')

    def save(**updates):
        record.update(updates)
        temporary = record_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(record_path)

    save()
    try:
        while True:
            job = json.loads((source / 'job.json').read_text())
            if job['status'] in ('failed', 'stopped', 'paused_by_user'):
                raise RuntimeError(f"Source training is {job['status']}; refusing automatic resume")
            if job['status'] == 'completed':
                break
            pid = job.get('training_pid')
            if pid and not Path(f'/proc/{pid}').exists():
                time.sleep(2)
                job = json.loads((source / 'job.json').read_text())
                if job['status'] != 'completed':
                    raise RuntimeError('Source trainer disappeared without successful completion')
                break
            time.sleep(10)
        from stable_baselines3.common.save_util import load_from_zip_file
        checkpoint = source / 'actor_critic.zip'
        data, _, _ = load_from_zip_file(checkpoint, device='cpu')
        start = int(data['num_timesteps'])
        remaining = args.total - start
        if remaining <= 0:
            save(status='completed', final_steps=start)
            return
        command = [sys.executable, '-u', str(PROJECT / 'scripts/train_rate_only.py'),
                   '--run', str(run), '--phase', args.phase, '--steps', str(remaining),
                   '--resume', str(checkpoint)]
        # Keep the completed exploration's collection/episode layout across the
        # phase boundary; future CLI defaults must not silently change its task.
        config = json.loads((source / 'config.json').read_text())
        command += ['--n-envs', str(config.get('n_envs', 1)),
                    '--n-steps', str(config['n_steps']),
                    '--batch-size', str(config['active_stage_settings']['batch_size']),
                    '--n-epochs', str(config['n_epochs']),
                    '--episode-steps', str(round(config['task']['episode_seconds']/config['task']['dt']))]
        source_command = job.get('command', [])
        if '--runtime' in source_command:
            command += ['--runtime', source_command[source_command.index('--runtime') + 1]]
        with (source / 'exploration_extension_training.log').open('a') as log:
            child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        save(status='continuing', starting_steps=start, additional_steps=remaining,
             expected_final_steps=start + ((remaining + 4095) // 4096) * 4096,
             supervisor_pid=child.pid, command=command)

        def link_display():
            # Keep the existing TensorBoard URL and its original history intact.
            for group in ('PPO_0', 'trajectories'):
                destination = source / 'tensorboard_raw' / group
                for event in (run / 'tensorboard_raw' / group).glob('events.out.tfevents.*'):
                    link = destination / ('events.out.tfevents.continuation.' + event.name)
                    if not link.exists():
                        link.symlink_to(event.resolve())

        while child.poll() is None:
            link_display()
            time.sleep(10)
        link_display()
        if child.returncode:
            raise RuntimeError(f'Continuation exited with code {child.returncode}')
        report = json.loads((run / 'training_report.json').read_text())
        save(status='completed', final_steps=report['timesteps'])
    except Exception as error:
        save(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
