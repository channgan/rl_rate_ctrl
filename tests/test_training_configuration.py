"""Exercise command/configuration assembly without starting a simulator."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rate_rl.action_contract import reward_interface_metadata
from rate_rl.defaults import CURRENT
from rate_rl.env import TaskConfig
from rate_rl.training.cli import parse_args
from rate_rl.training.configuration import (
    build_monitor_fields, build_promotion_options, build_task_config,
)
from rate_rl.training.metadata import finalize_reward_metadata


NATIVE = ["--native-torque", "--simple-critic", "--px4-native-outer",
          "--waypoint-tracking", "--squared-error-reward", "--stage", "free_flight",
          "--training-phase", "late", "--control-dt", "0.01",
          "--no-tilt-termination", "--batch-size", "2048"]


def test_current_native_configuration_is_pure_and_preserves_task(tmp_path):
    run = tmp_path / "must-not-be-created"
    args = parse_args(NATIVE + ["--run", str(run)])
    config = build_task_config(args)
    assert not run.exists()
    assert (args.n_envs, args.n_steps, args.batch_size, args.n_epochs) == (1, 4096, 2048, 10)
    assert config.episode_seconds == 20.48
    assert config.dt == .01
    assert config.tracking_bonus_threshold_deg_s == 5.
    assert config.slew_weight == .02 / 7.
    assert config.reward_slew_rate_scale_per_s == 10.
    assert config.tracking_bonus_weight == .0084
    assert config.native_torque and config.px4_native_outer and config.waypoint_tracking
    assert not config.terminate_on_tilt
    assert config.target_limit_rad_s == (1., 1., 1.)
    assert args.ppo_stage == "free_flight"
    assert args.reward_scale_multiplier == 1.


def test_legacy_cli_defaults_remain_unchanged_and_runtime_path_did_not_move():
    args = parse_args([])
    config = build_task_config(args)
    assert Path(args.runtime) == Path(__file__).resolve().parents[1] / "runtime.local.json"
    assert args.training_phase is None and args.batch_size is None
    assert config.dt == .001 and config.episode_seconds == 30.
    assert not config.native_torque and config.slew_weight == .1
    assert config.reward_slew_rate_scale_per_s == 50.
    assert config.tracking_bonus_threshold_deg_s == 5.
    assert (args.n_steps, args.n_epochs) == (4096, 10)
    smoke = parse_args(["--smoke"])
    assert (smoke.n_steps, smoke.n_epochs) == (2048, 2)


@pytest.mark.parametrize("n_envs, per_env", [(2, 2048), (4, 1024)])
def test_parallel_internal_default_and_explicit_public_batch(n_envs, per_env):
    internal = [arg for arg in NATIVE[:-2]] + ["--n-envs", str(n_envs)]
    args = parse_args(internal)
    assert (args.n_steps, args.batch_size) == (per_env, 1024)
    explicit = parse_args(internal + ["--batch-size", "2048"])
    assert explicit.batch_size == 2048
    assert explicit.n_steps * explicit.n_envs == 4096


@pytest.mark.parametrize("options", [
    ["--episode-steps", "0"], ["--batch-size", "3000"],
    ["--n-epochs", "0"], ["--base-instance", "101"],
    ["--steps", "0"], ["--allow-episode-length-change"],
    ["--native-torque"], ["--n-envs", "4"],
    ["--pwm-slew-scale", "0"], ["--slew-weight", "nan"],
    ["--reward-scale-multiplier", "nan"],
    ["--waypoint-tracking", "--fixed-episode-rate-target"],
])
def test_validation_rejects_invalid_input_before_any_run_creation(options, tmp_path):
    run = tmp_path / "invalid"
    with pytest.raises(SystemExit) as exc:
        parse_args(options + ["--run", str(run)])
    assert exc.value.code == 2
    assert not run.exists()


def test_legacy_phase_scale_and_native_explicit_overrides():
    legacy = parse_args(["--training-phase", "early"])
    assert legacy.ppo_stage == "ball_rig" and legacy.reward_scale_multiplier == 2.
    assert build_task_config(legacy).reward_slew_rate_scale_per_s == 100.
    native = parse_args(NATIVE + ["--episode-steps", "1000", "--torque-slew-scale", "12",
                                 "--slew-weight", "0.03", "--air-error-scale-deg-s", "7"])
    config = build_task_config(native)
    assert config.episode_seconds == 10.
    assert config.slew_weight == .03 and config.reward_slew_rate_scale_per_s == 12.
    assert config.air_reward_error_scale_deg_s == 7.
    # In SSE mode, the legacy rate-scale flag does not silently change the bonus gate.
    assert config.tracking_bonus_threshold_deg_s == 5.


def test_episode_migration_flag_still_requires_explicit_native_resume():
    args = parse_args(NATIVE + ["--resume", "saved.zip", "--allow-episode-length-change"])
    assert args.resume == Path("saved.zip") and args.allow_episode_length_change
    with pytest.raises(SystemExit):
        parse_args(["--resume", "saved.zip", "--allow-episode-length-change"])


def test_monitor_fields_and_promotion_options_keep_contract():
    args = parse_args(NATIVE)
    fields = build_monitor_fields(args)
    assert "episode_torque_saturation_fraction" in fields
    assert "episode_motor_saturation_fraction" in fields
    assert "episode_saturation_fraction" not in fields
    assert build_monitor_fields(parse_args(NATIVE + ["--compact-logs"])) == ()
    assert build_promotion_options(args) == dict(
        window_episodes=5, required_qualified=3, evaluation_episodes=10,
        evaluation_required=8, cooldown_steps=4096 * 16)


def native_metadata(config):
    args = parse_args(NATIVE)
    metadata = dict(reward_interface=reward_interface_metadata(),
                    reward_convention="legacy text",
                    effective_reward_scales=dict(pwm_slew_per_s=config.reward_slew_rate_scale_per_s),
                    stage_hyperparameters=dict(early={}, late={}),
                    waypoint_reference={})
    model = SimpleNamespace()
    finalize_reward_metadata(model, metadata, args, config)
    return model, metadata


def test_native_metadata_describes_executed_formula_and_separates_owned_thrust():
    config = build_task_config(parse_args(NATIVE))
    model, metadata = native_metadata(config)
    text = metadata["reward_convention"]
    assert "<5 deg/s" in text and "<2 deg/s" not in text
    assert "min(sum(error_deg_s**2),5000)" in text
    assert "raw failure=-30000-3500*(20.48-T)" in text
    assert "success=50000" in text and "multiplied by 7e-05" in text
    assert "torque_rate/10" in text
    assert "no headroom or waypoint bonus" in text
    assert model.reward_interface_metadata == metadata["reward_interface"]
    assert model.reward_interface_metadata is not metadata["reward_interface"]
    assert "pwm_slew_per_s" not in metadata["effective_reward_scales"]
    assert metadata["effective_reward_scales"]["torque_slew_per_s"] == 10.
    assert "forward vehicle_rates_setpoint.thrust_body unchanged" in metadata["waypoint_reference"]["thrust_conversion"]


def test_native_metadata_derives_overrides_instead_of_stale_literals():
    config = replace(build_task_config(parse_args(NATIVE)), episode_seconds=10.,
                     tracking_bonus_threshold_deg_s=2., reward_slew_rate_scale_per_s=15.,
                     slew_weight=.03, saturation_epsilon=.002)
    _, metadata = native_metadata(config)
    text = metadata["reward_convention"]
    assert "<2 deg/s" in text and "<5 deg/s" not in text
    assert "3500*(10-T)" in text
    assert "torque_rate/15" in text and "abs(tau)>=0.998" in text


def test_old_callback_import_paths_still_resolve():
    from rate_rl import train
    from rate_rl.training import metrics
    assert train.RewardMetrics is metrics.RewardMetrics
    assert train.CompactMetrics is metrics.CompactMetrics
    assert train.CONTINUOUS_COMPONENT_NAMES is metrics.CONTINUOUS_COMPONENT_NAMES
