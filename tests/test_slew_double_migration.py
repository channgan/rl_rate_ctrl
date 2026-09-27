from dataclasses import replace
from types import SimpleNamespace
from copy import deepcopy
import pytest
from rate_rl.env import TaskConfig
from rate_rl.reward_contract import mse_reward_interface_metadata, require_reward_interface

def test_double_slew_requires_explicit_migration_and_keeps_other_reward_fields():
    old = TaskConfig(native_torque=True, squared_error_reward=True, slew_weight=.01/7.)
    saved = mse_reward_interface_metadata(old)
    expected = mse_reward_interface_metadata(replace(old, slew_weight=.02/7.))
    model = SimpleNamespace(reward_interface_metadata=saved)
    with pytest.raises(ValueError): require_reward_interface(model, expected)
    require_reward_interface(model, expected, allow_tracking_tuning=True)
    assert model.reward_interface_metadata == saved
    bad = deepcopy(expected)
    bad['raw_settlement']['failure_base'] = -1
    with pytest.raises(ValueError): require_reward_interface(model, bad, allow_tracking_tuning=True)
