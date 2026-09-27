"""Publish saved trajectory PNGs to TensorBoard without touching training."""
import argparse
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest_path = run / 'trajectory_image_manifest_unified_v4.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with SummaryWriter(str(run / 'tensorboard' / 'trajectories')) as writer:
        while True:
            for path in sorted((run / 'trajectories').glob('step_*.png')):
                if not path.stem.removeprefix('step_').isdigit():
                    continue
                # JSON is written last, after PNG/CSV completion.
                report = path.with_suffix('.json')
                if not report.exists():
                    continue
                signature = [path.stat().st_size, path.stat().st_mtime_ns]
                if manifest.get(path.name) == signature:
                    continue
                # PNGs already use adaptive axes when recorded. Republishing
                # history must not redraw/mutate every image and delay live data.
                step = int(path.stem.removeprefix('step_'))
                with Image.open(path) as picture:
                    pixels = np.array(picture.convert('RGB'))
                metadata = json.loads(report.read_text())
                tag = ('trajectories/angular_rates_torque_and_pwm' if metadata.get('torque_source')
                       else 'trajectories/angular_rates_and_pwm')
                writer.add_image(tag, pixels,
                                 global_step=step, dataformats='HWC')
                writer.flush()
                manifest[path.name] = signature
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(f'Published trajectory at step {step}: {path.name}', flush=True)
            job = run / 'job.json'
            try:
                state = json.loads(job.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                state = {}
            if state.get('status') in ('completed', 'failed', 'stopped', 'paused_by_user'):
                break
            time.sleep(10)


if __name__ == '__main__':
    main()
