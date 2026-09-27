"""A recorded worker keeps local episode time and parallel global step labels."""
import csv
from copy import deepcopy
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from rate_rl.trajectory import EpisodeTrajectory, TrajectoryClock


class EpisodeEnv(gym.Env):
    def __init__(self, episode_length=3, worker_id=0):
        self.action_space = gym.spaces.Box(-1., 1., (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)
        self.config = SimpleNamespace(dt=.01, native_torque=True)
        self.current_state = {'q_ned_frd': np.array([1., 0., 0., 0.])}
        self.episode_length = episode_length
        self.worker_id = worker_id
        self.local_steps = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.local_steps = 0
        return np.zeros(9, dtype=np.float32), {}

    def step(self, action):
        self.local_steps += 1
        rate = np.full(3, .1 + self.worker_id)
        info = dict(
            training_stage='free_flight', sim_us=self.local_steps * 10000,
            target_rad_s=rate, rates_true_rad_s=rate, rates_rad_s=rate,
            current_pwm_command=np.full(4, .6), torque_command=np.asarray(action),
            position_ned=np.array([self.worker_id, 0., -5.]),
            position_target_ned=np.array([self.worker_id + 1., 0., -5.]),
            attitude_target_rpy_rad=np.zeros(3),
            waypoint_gate_transition=dict(settled_seconds=self.local_steps * .01,
                                          switched=self.local_steps == 2),
            failure=None, is_success=self.local_steps == self.episode_length,
        )
        return np.zeros(9, dtype=np.float32), 0., self.local_steps == self.episode_length, False, info

    def close(self):
        self.closed = True


def test_parallel_global_steps_do_not_double_episode_time(tmp_path, monkeypatch):
    # Plot rendering is unchanged; exercise the real CSV/JSON writer without
    # spending this timing/metadata test on rasterizing ten matplotlib axes.
    from matplotlib.figure import Figure
    titles = []
    original_title = Figure.suptitle

    def capture_title(self, title, **kwargs):
        titles.append(title)
        return original_title(self, title, **kwargs)

    monkeypatch.setattr(Figure, 'suptitle', capture_title)
    monkeypatch.setattr(Figure, 'savefig', lambda self, path, **kwargs: None)
    clock = TrajectoryClock(tmp_path, step_stride=2, initial_step=39997, worker_id=0)
    env = EpisodeTrajectory(EpisodeEnv(), clock)
    env.reset()
    for _ in range(3):
        env.step(np.array([.1, -.1, 0.]))

    report = json.loads((tmp_path / 'step_000040000.json').read_text())
    assert report['milestone'] == 40000  # Crossed between 39999 and 40001.
    assert (report['first_step'], report['last_step']) == (39999, 40003)
    assert report['duration_s'] == pytest.approx(.03)
    assert report['worker_id'] == 0
    assert report['global_step_stride'] == 2
    assert report['complete'] is True
    assert report['waypoint_switches'][0]['training_step'] == 40001
    assert report['waypoint_switches'][0]['episode_time_s'] == pytest.approx(.02)
    assert 'worker 0' in titles[0]
    with (tmp_path / 'step_000040000.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['training_step']) for row in rows] == [39999, 40001, 40003]
    assert [int(row['sim_us']) for row in rows] == [10000, 20000, 30000]
    with (tmp_path / 'step_000040000.state.csv').open() as stream:
        state_rows = list(csv.DictReader(stream))
    assert [int(row['training_step']) for row in state_rows] == [39999, 40001, 40003]
    assert all(float(row['north_m']) == 0. for row in state_rows)
    assert not env.rows and not env.milestones


def test_unrecorded_worker_never_buffers_or_overwrites_trajectory(tmp_path, monkeypatch):
    sentinel = tmp_path / 'step_000040000.json'
    sentinel.write_text('{"worker_id": 0}')
    clock = TrajectoryClock(tmp_path, step_stride=2, initial_step=39997,
                            worker_id=1, enabled=False)
    inner = EpisodeEnv(worker_id=1)
    env = EpisodeTrajectory(inner, clock)

    def unexpected_save(*args):
        pytest.fail('Disabled worker must not save a trajectory')

    monkeypatch.setattr(env, '_save', unexpected_save)
    env.reset()
    for _ in range(3):
        env.step(np.zeros(3))
    env.close()
    assert clock.steps == 40003
    assert not env.rows and not env.state_rows and not env.milestones
    assert sentinel.read_text() == '{"worker_id": 0}'
    assert list(tmp_path.iterdir()) == [sentinel]
    assert inner.closed


def test_default_clock_retains_single_environment_step_numbering(tmp_path, monkeypatch):
    clock = TrajectoryClock(tmp_path, interval=2)
    env = EpisodeTrajectory(EpisodeEnv(), clock)
    recorded = []
    monkeypatch.setattr(env, '_save', lambda outcome, complete: recorded.append(
        (deepcopy(env.rows), list(env.milestones), outcome, complete)))
    env.reset()
    for _ in range(3):
        env.step(np.zeros(3))
    rows, milestones, outcome, complete = recorded[0]
    assert [row[0] for row in rows] == [1, 2, 3]
    assert milestones == [2]
    assert outcome == 'success' and complete
    assert clock.steps == 3 and clock.step_stride == 1 and clock.enabled


def test_reset_separates_worker_episodes_without_resetting_global_clock(tmp_path, monkeypatch):
    clock = TrajectoryClock(tmp_path, interval=4, step_stride=2, initial_step=1)
    env = EpisodeTrajectory(EpisodeEnv(episode_length=2), clock)
    recorded = []
    monkeypatch.setattr(env, '_save', lambda outcome, complete: recorded.append(
        (deepcopy(env.rows), list(env.milestones), deepcopy(env.gate_events))))
    for _ in range(2):
        env.reset()
        env.step(np.zeros(3))
        env.step(np.zeros(3))
    assert [[row[0] for row in item[0]] for item in recorded] == [[3, 5], [7, 9]]
    assert [item[1] for item in recorded] == [[4], [8]]
    assert [item[2][0]['episode_time_s'] for item in recorded] == [.02, .02]


def test_clock_detects_all_crossed_milestones_and_allows_resume_assignment(tmp_path):
    clock = TrajectoryClock(tmp_path, interval=3, step_stride=10)
    assert list(clock.advance()) == [3, 6, 9]
    clock.steps = 22  # Existing single-worker resume interface remains valid.
    assert list(clock.advance()) == [24, 27, 30]


@pytest.mark.parametrize('settings', [
    {'interval': 0}, {'interval': 1.5}, {'step_stride': 0}, {'step_stride': True},
    {'initial_step': -1}, {'initial_step': 1.5}, {'worker_id': -1}, {'worker_id': True},
])
def test_invalid_global_clock_configuration_is_rejected(tmp_path, settings):
    with pytest.raises(ValueError):
        TrajectoryClock(tmp_path, **settings)
