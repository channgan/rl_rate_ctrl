import numpy as np
import pytest
from rate_rl.env import RateControlEnv, TaskConfig
from backend_fixture import TestBackend


@pytest.mark.parametrize('errors,success', [([4.,4.],True),([5.,5.],False),([10.,0.],False)])
def test_horizon_uses_complete_episode_rmse_and_exclusive_settlement(errors, success):
    env = RateControlEnv(TestBackend(), TaskConfig(dt=.01,episode_seconds=.02,
                                                squared_error_reward=True))
    env.reset(seed=0)
    for index, error in enumerate(errors):
        env.target = np.deg2rad([error]*3)
        _, _, done, truncated, info = env.step([.7]*4)
        assert done == (index == 1)
        assert not truncated
    assert info['is_success'] == success
    assert info['success_bonus'] == pytest.approx(3.5 if success else 0)
    assert info['failure_penalty'] == pytest.approx(0 if success else 2.1)
    assert info['failure'] == (None if success else 'tracking_rmse')
    assert info['episode_rate_rmse_deg_s'] == pytest.approx(np.sqrt(np.mean(np.square(errors))))


@pytest.mark.parametrize('elapsed_steps', [1, 200, 500, 1500, 2999])
def test_early_failure_settlement_uses_elapsed_time(elapsed_steps):
    env = RateControlEnv(TestBackend(), TaskConfig(dt=.01,episode_seconds=30.,squared_error_reward=True))
    env.reset(seed=0)
    env.steps = elapsed_steps - 1
    env.backend.true_rates = [13.,0.,0.]
    _,_,done,_,info = env.step([.7]*4)
    assert done and info['failure']=='rate' and info['success_bonus']==0
    assert info['failure_penalty']/7e-5 == pytest.approx(105000-2500*elapsed_steps*.01)
