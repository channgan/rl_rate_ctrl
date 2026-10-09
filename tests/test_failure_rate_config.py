"""Saved tasks must not inherit later changes to the public reward recipe."""
from dataclasses import asdict
from types import SimpleNamespace
import pytest
from rate_rl.env import TaskConfig
from rate_rl.reward_contract import early_failure_rate, mse_reward_interface_metadata, require_reward_interface


def test_saved_failure_rate_survives_serialization():
    old = TaskConfig(native_torque=True, squared_error_reward=True)
    assert early_failure_rate(old) == 3500.
    current = TaskConfig(native_torque=True, squared_error_reward=True, failure_rate_per_s=4000.)
    restored = TaskConfig(**asdict(current))
    assert early_failure_rate(restored) == 4000.
    saved = mse_reward_interface_metadata(old)
    model = SimpleNamespace(reward_interface_metadata=saved)
    require_reward_interface(model, saved)
    with pytest.raises(ValueError, match="reward interface mismatch"):
        require_reward_interface(model, mse_reward_interface_metadata(restored))
