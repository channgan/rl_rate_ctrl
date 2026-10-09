from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rate_rl.env import RateControlEnv
from rate_rl.reward_contract import mse_reward_interface_metadata, require_reward_interface
from test_native_torque_env import NativeTorqueBackend, task


@pytest.mark.parametrize('success', [True, False])
def test_unscaled_environment_all_components_and_settlement(success):
    outputs = []
    for gain in [7e-5, 1.]:
        backend = NativeTorqueBackend()
        backend.reference = np.deg2rad([5., 5., 5.])
        env = RateControlEnv(backend, task(episode_seconds=.01,
            error_progress_weight=.003, reward_gain=gain))
        env.reset(seed=101)
        backend.true_rates = np.deg2rad([1., 1., 1.] if success else [-5., -5., -5.])
        _, r, done, _, info = env.step(np.array([.1, 0., 0.] if success else [1., -1., 0.]))
        assert done
        assert r == pytest.approx(sum(info['episode_reward_components'])
            + info['episode_error_progress_reward'] + info['episode_success_bonus']
            - info['episode_saturation_penalty'] - info['episode_failure_penalty'])
        outputs.append((r, info))
    scaled, raw = outputs
    assert raw[0] == pytest.approx(scaled[0] / 7e-5)
    for key in ['episode_reward_components', 'episode_error_progress_reward',
                'episode_success_bonus', 'episode_failure_penalty',
                'episode_saturation_penalty', 'tracking_bonus', 'saturation_penalty']:
        np.testing.assert_allclose(raw[1][key], np.asarray(scaled[1][key]) / 7e-5)
    assert raw[1]['episode_success_bonus' if success else 'episode_failure_penalty'] == pytest.approx(50000. if success else 30000.)
    assert raw[1]['episode_error_progress_reward'] == pytest.approx(.081 if success else -.675)


def test_gain_mismatch_rejects_checkpoint():
    old = task(error_progress_weight=.003)
    new = replace(old, reward_gain=1.)
    model = SimpleNamespace(reward_interface_metadata=mse_reward_interface_metadata(old))
    with pytest.raises(ValueError, match='reward interface mismatch'):
        require_reward_interface(model, mse_reward_interface_metadata(new))
