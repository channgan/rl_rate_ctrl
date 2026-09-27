"""Golden transitions recorded from env.py before extracting reward functions.

The fixture is deliberately independent of the new reward implementation and
includes complete diagnostics, actor observations and terminal transitions.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from rate_rl.env import RateControlEnv, TaskConfig
from rate_rl.reward_contract import mse_reward_interface_metadata
from backend_fixture import TestBackend
from test_native_torque_env import NativeTorqueBackend, task


FIXTURE = Path(__file__).with_name('fixtures') / 'reward_transition_v15.json'


class TraceBackend(NativeTorqueBackend):
    overrides = None

    def state(self):
        state = super().state()
        state.update(self.overrides or {})
        return state


def as_json(value):
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def recorded_transitions(case):
    native = case.startswith('native')
    backend = TraceBackend() if native else TestBackend()
    config = (task(episode_seconds=.04, tracking_bonus_threshold_deg_s=5.,
                   slew_weight=.02 / 7.) if native else TaskConfig(
        dt=.01, episode_seconds=.04, target_limit_rad_s=(0., 0., 0.),
        squared_error_reward=case == 'legacy_sse'))
    env = RateControlEnv(backend, config)
    observation, initial = env.reset(seed=19)
    results = [dict(observation=observation, info=initial)]
    for index in range(4):
        if native:
            backend.allocated_pwm = np.asarray(
                [[0., 1., .4, .5], [.8, .8, .8, .8], [.2, .9, .1, .4], [.5]*4][index],
                dtype=np.float32)
            error = ([4., -3., 2.] if case == 'native_success' else [6., 0., 0.])
            backend.true_rates = env.target - np.deg2rad(error)
            backend.rates = np.asarray(backend.true_rates) + np.deg2rad([.1, -.2, .3])
            backend.reference = np.asarray([.1, -.2, .3]) * (index + 1)
            action = [[.1, -.2, .3], [.1, -.2, .3], [1., -1., .999], [0., 0., 0.]][index]
            if case == 'native_first_failure':
                backend.true_rates = [13., 0., 0.]
            if case == 'native_last_ground' and index == 3:
                backend.overrides = dict(position_ned=[0., 0., -.2])
        else:
            backend.rates = np.deg2rad([2., 3., 1.])
            action = [[.7]*4, [0., 1., .4, .5], [.2]*4, [.7]*4][index]
        observation, reward, terminated, truncated, info = env.step(action)
        results.append(dict(observation=observation, reward=reward,
                            terminated=terminated, truncated=truncated, info=info))
        if terminated or truncated:
            break
    env.close()
    return as_json(dict(transitions=results, reward_interface=(
        mse_reward_interface_metadata(config) if config.squared_error_reward else None)))


CASES = ['native_success', 'native_rmse', 'native_first_failure',
         'native_last_ground', 'legacy_bounded', 'legacy_sse']


@pytest.mark.parametrize('case', CASES)
def test_full_transitions_match_pre_refactor_golden(case):
    expected = json.loads(FIXTURE.read_text())[case]
    actual = recorded_transitions(case)
    # Exact equality guards the complete info schema and operation ordering,
    # not just a formula reimplemented in the assertion.
    assert actual == expected
