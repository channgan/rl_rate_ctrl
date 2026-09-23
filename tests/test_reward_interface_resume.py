from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rate_rl.action_contract import REWARD_CONTRACT
from rate_rl.env import TaskConfig
from rate_rl.reward_contract import (
    mse_reward_interface_metadata, require_reward_interface,
)


def current():
    return mse_reward_interface_metadata(TaskConfig(squared_error_reward=True))


def test_previous_cap10000_native_model_requires_new_training():
    expected = mse_reward_interface_metadata(TaskConfig(squared_error_reward=True, native_torque=True))
    old = deepcopy(expected)
    old['mse_cap_deg_s_squared'] = 10000.
    with pytest.raises(ValueError):
        require_reward_interface(SimpleNamespace(reward_interface_metadata=old), expected)


def test_failure2700_migration_requires_explicit_flag_and_rejects_other_changes():
    expected = mse_reward_interface_metadata(TaskConfig(squared_error_reward=True, native_torque=True))
    # Exercise the historical migration without silently permitting 3000.
    expected['raw_settlement']['early_failure_extra_max'] = -81000.
    expected['raw_settlement']['failure_formula'] = '-30000-2700*episode_seconds*(1-survival_fraction)'
    old = deepcopy(expected)
    old['raw_settlement']['early_failure_extra_max'] = -75000.
    old['raw_settlement']['failure_formula'] = '-30000-2500*episode_seconds*(1-survival_fraction)'
    model = SimpleNamespace(reward_interface_metadata=old)
    with pytest.raises(ValueError):
        require_reward_interface(model, expected)
    assert require_reward_interface(model, expected, allow_failure2700_change=True) == old
    old['raw_settlement']['success'] = 60000.
    with pytest.raises(ValueError):
        require_reward_interface(model, expected, allow_failure2700_change=True)


def test_failure2500_rejects_2700_checkpoint():
    expected = mse_reward_interface_metadata(TaskConfig(squared_error_reward=True, native_torque=True))
    old = deepcopy(expected)
    old['raw_settlement']['early_failure_extra_max'] = -81000.
    old['raw_settlement']['failure_formula'] = '-30000-2700*episode_seconds*(1-survival_fraction)'
    with pytest.raises(ValueError):
        require_reward_interface(SimpleNamespace(reward_interface_metadata=old), expected,
                                 allow_tracking_tuning=True, allow_failure2700_change=True)


@pytest.mark.parametrize("mismatch", [
    "missing", "old_version", "survival_only", "old_settlement", "ungated_headroom",
    "slew_scale", "weight",
])
def test_same_legacy_contract_cannot_hide_a_changed_objective(mismatch):
    expected = current()
    model = SimpleNamespace(reward_contract=REWARD_CONTRACT)
    if mismatch != "missing":
        saved = deepcopy(expected)
        if mismatch == "old_version":
            saved["version"] = "old_mse_objective"
        elif mismatch == "survival_only":
            saved["horizon_success"].pop("whole_episode_rmse_strictly_below_deg_s")
        elif mismatch == "old_settlement":
            saved["raw_settlement"]["success"] = 40. / 7e-5
        elif mismatch == "ungated_headroom":
            saved["headroom_gate"]["strict_error_deg_s"] = None
        elif mismatch == "slew_scale":
            saved["slew_scale_per_s"] = 100.
        elif mismatch == "weight":
            saved["continuous_weights"]["slew"] = .2
        model.reward_interface_metadata = saved
    before = deepcopy(model.__dict__)
    with pytest.raises(ValueError, match="Checkpoint reward interface mismatch"):
        require_reward_interface(model, expected)
    assert model.__dict__ == before


def test_matching_objective_is_accepted_without_aliasing_saved_metadata():
    expected = current()
    model = SimpleNamespace(reward_interface_metadata=deepcopy(expected))
    accepted = require_reward_interface(model, expected)
    accepted["raw_settlement"]["success"] = -1
    assert model.reward_interface_metadata == expected


def test_metadata_tracks_configurable_reward_parameters():
    config = TaskConfig(squared_error_reward=True)
    changed = replace(config, slew_weight=.2, reward_slew_rate_scale_per_s=75.)
    old = mse_reward_interface_metadata(config)
    new = mse_reward_interface_metadata(changed)
    assert new["continuous_weights"]["slew"] == .2
    assert new["slew_scale_per_s"] == 75.
    with pytest.raises(ValueError, match="Checkpoint reward interface mismatch"):
        require_reward_interface(SimpleNamespace(reward_interface_metadata=old), new)


def test_explicit_sum_and_headroom_migration_preserves_other_rewards():
    expected = current()
    expected['version'] = 'sse10000_headroom2_tracking5_rmse5_failure2500persecond_v10'
    expected.pop('waypoint_switch_raw_bonus')
    old = deepcopy(expected)
    old['version'] = 'mse10000_tracking5_rmse5_failure2500persecond_v8'
    old.pop('rate_aggregation')
    old['continuous_weights']['mean_headroom'] /= 2
    model = SimpleNamespace(reward_interface_metadata=old)
    with pytest.raises(ValueError): require_reward_interface(model, expected)
    assert require_reward_interface(model, expected, allow_rate_sum_change=True) == old
    assert model.reward_interface_metadata == old
    wrong = deepcopy(old)
    wrong['raw_settlement']['failure_base'] = -9999
    with pytest.raises(ValueError):
        require_reward_interface(SimpleNamespace(reward_interface_metadata=wrong), expected, allow_rate_sum_change=True)
