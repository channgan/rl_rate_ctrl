"""Pure task, sampling-gate and monitor configuration builders."""
from ..defaults import CURRENT
from ..env import TaskConfig


def build_task_config(args):
    defaults = TaskConfig(native_torque=args.native_torque)
    from ..episode_horizon import DEFAULT_NATIVE_EPISODE_STEPS
    episode_steps = args.episode_steps if args.episode_steps is not None else (
        DEFAULT_NATIVE_EPISODE_STEPS if args.native_torque else round(defaults.episode_seconds / args.control_dt))
    config = TaskConfig(
        episode_seconds=episode_steps * args.control_dt,
        tracking_bonus_threshold_deg_s=CURRENT.tracking_threshold_deg_s,
        squared_error_reward=args.squared_error_reward,
        slew_weight=args.slew_weight,
        error_progress_weight=CURRENT.error_progress_weight if args.native_torque else 0.,
        error_progress_delta_cap=CURRENT.error_progress_delta_cap if args.native_torque else None,
        reward_gain=CURRENT.reward_gain if args.native_torque else 7e-5,
        failure_rate_per_s=CURRENT.failure_rate_per_s if args.native_torque else 2500.,
        dt=args.control_dt, fixed_episode_rate_target=args.fixed_episode_rate_target,
        terminate_on_tilt=not args.no_tilt_termination,
        target_limit_rad_s=tuple(args.target_rate_limits),
        air_target_limit_rad_s=tuple(args.air_rate_limits), fixed_altitude_target_m=args.fixed_altitude,
        waypoint_tracking=args.waypoint_tracking,
        px4_native_outer=args.px4_native_outer,
        native_torque=args.native_torque,
        reward_error_scale_deg_s=defaults.reward_error_scale_deg_s * args.reward_scale_multiplier,
        air_reward_error_scale_deg_s=(args.air_error_scale_deg_s if args.air_error_scale_deg_s is not None else defaults.air_reward_error_scale_deg_s * args.reward_scale_multiplier),
        reward_slew_rate_scale_per_s=(args.pwm_slew_scale if args.pwm_slew_scale is not None else defaults.reward_slew_rate_scale_per_s * args.reward_scale_multiplier))
    return config

def build_promotion_options(args):
    promotion_options = dict(window_episodes=args.prescreen_window,
        required_qualified=args.prescreen_required, evaluation_episodes=args.promotion_window,
        evaluation_required=args.promotion_required, cooldown_steps=4096 * args.promotion_cooldown_rollouts)
    return promotion_options

def build_monitor_fields(args):
    monitor_fields = ("is_success", "training_stage", "promotion_window_size",
                                 "reward_error_scale_deg_s",
                                 "promotion_qualified_count", "promotion_window_full", "promotion_history",
                                 "episode_rate_rmse_rad_s", "episode_saturation_fraction",
                                 "episode_observed_rate_rmse_rad_s",
                                 "episode_rate_rmse_deg_s", "episode_observed_rate_rmse_deg_s",
                                 "episode_axis_rate_rmse_deg_s",
                                 "transfer_qualified", "episode_settled_tracking_fraction",
                                 "episode_axis_rate_rmse_rad_s", "episode_reward_components",
                                 "episode_saturation_penalty", "episode_failure_penalty", "episode_success_bonus")
    if args.native_torque:
        monitor_fields = tuple("episode_torque_saturation_fraction" if field == "episode_saturation_fraction"
                               else field for field in monitor_fields)
        monitor_fields += ("episode_motor_saturation_fraction",)
        monitor_fields += ("episode_error_progress_reward",)

    if args.compact_logs:
        monitor_fields = ()

    return monitor_fields
