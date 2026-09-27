"""Live display projection: unscale episode returns, preserve training metrics."""
import argparse
from contextlib import closing
import json
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.util import tensor_util


def raw_return_event(raw, gain):
    event = Event.FromString(raw)
    if event.HasField('summary'):
        for value in event.summary.value:
            if value.tag not in ('rollout/ep_rew_mean', 'eval/mean_reward'):
                continue
            if value.HasField('simple_value'):
                value.simple_value /= gain
            elif value.HasField('tensor'):
                value.tensor.CopyFrom(tensor_util.make_tensor_proto(
                    tensor_util.make_ndarray(value.tensor) / gain))
            value.metadata.summary_description = 'Raw episode return before the common PPO reward gain; display only.'
    return event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    gain = json.loads((run/'config.json').read_text())['reward_interface']['common_reward_gain']
    view = run/'tensorboard_raw'
    view.mkdir(exist_ok=True)
    for existing in view.glob('PPO_0/events.out.tfevents.*'):
        assert not any(Event.FromString(raw).HasField('summary')
                       for raw in RawEventFileLoader(str(existing)).Load()), 'View already contains summaries'
    trajectories = view/'trajectories'
    if not trajectories.exists():
        trajectories.symlink_to(run/'tensorboard/trajectories', target_is_directory=True)
    loaders = {}
    with closing(EventFileWriter(str(view/'PPO_0'))) as writer:
        while True:
            count = 0
            for path in sorted((run/'tensorboard/PPO_0').glob('events.out.tfevents.*')):
                loader = loaders.setdefault(path, RawEventFileLoader(str(path)))
                for raw in loader.Load():
                    event = raw_return_event(raw, gain)
                    if event.HasField('summary'):
                        writer.add_event(event)
                        count += 1
            writer.flush()
            if count:
                print(f'Published {count} events; return multiplier={1/gain}', flush=True)
            try:
                job = json.loads((run / 'job.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                job = {}
            if job.get('status') in ('completed', 'failed', 'stopped', 'paused_by_user'):
                break
            time.sleep(10)


if __name__ == '__main__':
    main()
