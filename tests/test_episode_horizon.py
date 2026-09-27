from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from rate_rl.episode_horizon import DEFAULT_NATIVE_EPISODE_STEPS, migrate_episode_horizon
from rate_rl.env import RateControlEnv
from rate_rl.reward_contract import mse_reward_interface_metadata, require_reward_interface
from test_native_torque_env import NativeTorqueBackend, task


@pytest.mark.parametrize('error_deg, success', [(0., True), (6., False)])
def test_exact_2048_step_settlement_and_axis_rmse(error_deg, success):
    backend = NativeTorqueBackend()
    config = task(episode_seconds=DEFAULT_NATIVE_EPISODE_STEPS * .01)
    env = RateControlEnv(backend, config)
    env.reset(seed=1)
    backend.true_rates = [np.deg2rad(error_deg), 0., 0.]
    for step in range(1, 2049):
        _, _, done, truncated, info = env.step(np.zeros(3))
        assert done == (step == 2048)
        assert not truncated
        if step < 2048:
            assert info['success_bonus'] == info['failure_penalty'] == 0.
    assert env.steps == 2048
    assert info['is_success'] == success
    assert info['success_bonus']/7e-5 == pytest.approx(50000 if success else 0)
    assert info['failure_penalty']/7e-5 == pytest.approx(0 if success else 30000)
    assert info['failure'] == (None if success else 'tracking_rmse')


@pytest.mark.parametrize('failure_step', [1, 1024, 2048])
def test_shorter_horizon_early_failure_uses_actual_remaining_seconds(failure_step):
    backend = NativeTorqueBackend()
    env = RateControlEnv(backend, task(episode_seconds=20.48))
    env.reset(seed=2)
    for _ in range(failure_step - 1):
        assert not env.step(np.zeros(3))[2]
    backend.true_rates = [13., 0., 0.]
    _, _, done, _, info = env.step(np.zeros(3))
    assert done and info['failure'] == 'rate'
    assert info['failure_penalty']/7e-5 == pytest.approx(30000 + 3500*(20.48-failure_step*.01))
    assert info['success_bonus'] == 0.


def checkpoint():
    old = task(episode_seconds=30.)
    new = task(episode_seconds=20.48)
    contract = dict(task=asdict(old), simulator=dict(identity='unchanged'))
    target = deepcopy(contract)
    target['task'] = asdict(new)
    model = SimpleNamespace(promotion_environment_contract=contract,
        reward_interface_metadata=mse_reward_interface_metadata(old),
        curriculum_state=dict(pending_evaluation=None),
        learning_rate=3e-4, kl_low_kl_count=2, optimizer_marker='unchanged')
    return model, target, new


def test_horizon_migration_is_explicit_and_preserves_other_fields():
    model, target, config = checkpoint()
    expected = mse_reward_interface_metadata(config)
    with pytest.raises(ValueError, match='reward interface mismatch'):
        require_reward_interface(model, expected)
    record = migrate_episode_horizon(model, target)
    assert record['previous_steps'] == 3000 and record['current_steps'] == 2048
    assert record['current_settlement']['early_failure_extra_max'] == -71680.
    assert require_reward_interface(model, expected) == expected
    assert model.promotion_environment_contract == target
    assert model.learning_rate == 3e-4 and model.kl_low_kl_count == 2
    assert model.optimizer_marker == 'unchanged'


@pytest.mark.parametrize('extra_change', ['dt', 'reward', 'simulator', 'pending', 'saved_reward'])
def test_horizon_migration_rejects_other_changes_without_mutation(extra_change):
    model, target, _ = checkpoint()
    if extra_change == 'dt': target['task']['dt'] = .02
    elif extra_change == 'reward': target['task']['slew_weight'] *= 2
    elif extra_change == 'simulator': target['simulator']['identity'] = 'changed'
    elif extra_change == 'pending': model.curriculum_state['pending_evaluation'] = dict(active=True)
    elif extra_change == 'saved_reward': model.reward_interface_metadata['raw_settlement']['success'] = 60000
    before = deepcopy(model.__dict__)
    with pytest.raises(ValueError):
        migrate_episode_horizon(model, target)
    assert model.__dict__ == before
