"""Stage transitions must not reset the adaptive optimizer schedule."""
import gymnasium as gym
import pytest

from rate_rl.adaptive_ppo import AdaptiveKLPPO
from rate_rl.ppo_schedule import apply_stage_settings


def make_model(phase, learning_rate):
    return AdaptiveKLPPO(
        "MlpPolicy", gym.make("Pendulum-v1"), kl_phase=phase,
        learning_rate=learning_rate, n_steps=8, batch_size=4, n_epochs=1,
        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}}, device="cpu",
    )


def test_reapplying_late_stage_keeps_adapted_lr_and_low_kl_streak():
    model = make_model("late", 1e-4)
    try:
        model.kl_low_kl_count = 2
        optimizer = model.policy.optimizer
        apply_stage_settings(model, True)
        assert model.policy.optimizer is optimizer
        assert model.kl_learning_rate == pytest.approx(1e-4)
        assert model.learning_rate == pytest.approx(1e-4)
        assert model.kl_low_kl_count == 2
        assert model.active_stage_settings["learning_rate"] == pytest.approx(1e-4)
        assert model.batch_size == 2048
        assert model.ent_coef == .005
        assert model.clip_range(1.) == .2
    finally:
        model.get_env().close()


@pytest.mark.parametrize("old_lr, expected", [(1e-3, 3e-4), (1e-4, 1e-4)])
def test_entering_late_stage_clamps_without_raising_low_lr(old_lr, expected):
    model = make_model("early", old_lr)
    try:
        model.kl_low_kl_count = 2
        optimizer = model.policy.optimizer
        apply_stage_settings(model, True)
        assert model.kl_phase == "late"
        assert model.kl_low_kl_count == 0
        assert model.kl_learning_rate == pytest.approx(expected)
        assert model.active_stage_settings["learning_rate"] == pytest.approx(expected)
        assert model.policy.optimizer is optimizer
    finally:
        model.get_env().close()
