"""Parse and validate training options without starting PX4 or creating run files.

Legacy flags remain here for checkpoint migration. The supported public native
torque command is assembled by scripts/train_rate_only.py.
"""
import argparse
import math
from pathlib import Path

from ..defaults import CURRENT


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", default=str(Path(__file__).resolve().parents[2] / "runtime.local.json"))
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--n-envs", type=int, choices=[1, 2, 4], default=CURRENT.n_envs)
    p.add_argument("--n-steps", type=int, help="Samples per environment per PPO rollout, not episode length")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--n-epochs", type=int)
    p.add_argument("--base-instance", type=int, default=CURRENT.base_instance)
    p.add_argument('--episode-steps', type=int, help='Maximum steps per episode; native torque default 2048')
    p.add_argument('--allow-episode-length-change', action='store_true',
                   help='Explicitly migrate only the saved episode duration and remaining-time settlement')
    p.add_argument('--allow-tracking-tuning', action='store_true')
    p.add_argument('--allow-axis-success-change', action='store_true')
    p.add_argument('--allow-failure2700-change', action='store_true', help='Explicitly migrate native torque early failure coefficient from 2500 to 2700')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", type=Path, default=Path("runs/ppo"))
    p.add_argument("--smoke", action="store_true", help="Short real-simulator rollout and actual gradient updates")
    p.add_argument("--promotion-window", type=int, default=10,
                   help="Complete evaluation episodes for one frozen checkpoint")
    p.add_argument("--promotion-required", type=int, default=8,
                   help="Qualified evaluation episodes required for promotion")
    p.add_argument("--prescreen-window", type=int, default=5,
                   help="Completed training episodes in the evaluation prescreen")
    p.add_argument("--prescreen-required", type=int, default=3,
                   help="Qualified training episodes needed to request evaluation")
    p.add_argument("--promotion-cooldown-rollouts", type=int, default=16,
                   help="Minimum full 4096-step rollouts after an unsuccessful evaluation")
    p.add_argument("--stage", choices=["ball_rig", "free_flight"],
                   help="Explicit starting stage; otherwise use checkpoint stage or runtime default")
    p.add_argument("--resume", type=Path, help="Continue an existing PPO checkpoint, preserving curriculum state")
    p.add_argument("--simple-critic", action="store_true", help="Ordinary PPO: same actor12 input and 64x64 architecture for critic")
    p.add_argument("--control-dt", type=float, default=.001)
    p.add_argument("--training-phase", choices=["early", "late"],
                   help="Select PPO settings; MSE phases share task and reward scales")
    p.add_argument('--explore-until-horizon', action='store_true',
                   help='Ignore --steps budget during early free flight; end after first full surviving episode and complete PPO update')
    p.add_argument("--allow-rate-sum-change", action="store_true",
                   help="Explicitly migrate v8 to sum-square cost and doubled headroom reward")
    p.add_argument("--allow-phase-change", action="store_true",
                   help="Permit only reward-scale contract changes during an explicit manual phase switch")
    p.add_argument("--allow-waypoint-gate-change", action="store_true",
                   help="Explicitly acknowledge a changed waypoint switching rule on resume")
    p.add_argument("--slew-weight", type=float, default=None)
    p.add_argument("--squared-error-reward", action="store_true",
                   help="Clipped raw degree MSE plus positive mean PWM headroom; common reward gain 7e-5")
    p.add_argument("--air-error-scale-deg-s", type=float, default=None,
                   help="Override free-flight error scale and tracking bonus threshold in deg/s")
    p.add_argument("--pwm-slew-scale", "--torque-slew-scale", dest="pwm_slew_scale", type=float, default=None,
                   help="Slew scale in s^-1: requested torque for --native-torque, otherwise PWM; independent of phase")
    p.add_argument("--allow-slew-weight-change", action="store_true",
                   help="Explicitly migrate only the slew reward weight when resuming")
    p.add_argument("--fixed-altitude", type=float, default=None)
    p.add_argument("--waypoint-tracking", action="store_true")
    p.add_argument("--px4-native-outer", action="store_true",
                   help="Use PX4 native position/velocity/attitude modules; Python sends waypoints only")
    p.add_argument('--native-torque', action='store_true',
                   help='Rate-only: actor9 with previous torque -> normalized torque3; PX4 owns thrust and allocation')
    p.add_argument("--air-rate-limits", type=float, nargs=3, default=[2., 2., 1.5])
    p.add_argument("--no-tilt-termination", action="store_true", help="Disable tilt failure; retain ground, rate and time limits")
    p.add_argument("--target-rate-limits", type=float, nargs=3, default=[1., 1., 1.],
                   metavar=("ROLL", "PITCH", "YAW"), help="Symmetric per-axis random target bounds in rad/s")
    p.add_argument("--fixed-episode-rate-target", action="store_true",
                   help="Sample independent early-range rates once per episode; altitude loop updates thrust only")
    p.add_argument("--compact-logs", action="store_true", help="Only core metrics, episode outcomes and reward contributions")
    p.add_argument("--reward-scale-multiplier", type=float, default=1.0,
                   help="Multiply rate-error scales, tracking eligibility threshold and PWM slew scale")
    p.add_argument("--ppo-stage", choices=["ball_rig", "free_flight"],
                   help="Explicit initial PPO settings, independent of physical starting stage")
    args = p.parse_args(argv)
    if args.episode_steps is not None and args.episode_steps < 1:
        p.error('episode-steps must be positive')
    if args.allow_episode_length_change and (not args.resume or not args.native_torque):
        p.error('Episode length migration requires --resume and --native-torque')
    rollout_steps = args.n_steps if args.n_steps is not None else (2048 if args.smoke and args.n_envs == 1 else CURRENT.rollout_steps // args.n_envs)
    epochs = args.n_epochs if args.n_epochs is not None else (2 if args.smoke else CURRENT.n_epochs)
    if args.n_envs > 1 and args.batch_size is None:
        args.batch_size = 1024
    if rollout_steps < 2 or epochs < 1 or not 0 <= args.base_instance <= 101 - args.n_envs:
        p.error("Invalid rollout length, epoch count or PX4 instance range")
    if args.batch_size is not None and (args.batch_size < 2 or
            rollout_steps * args.n_envs % args.batch_size):
        p.error("batch-size must divide the total rollout sample count")
    if args.n_envs > 1 and (not args.native_torque or not args.training_phase or
                           args.stage != 'free_flight' or args.explore_until_horizon):
        p.error("Parallel training requires native torque, manual phase and free flight; no survival promotion gate")
    if args.native_torque and not (args.simple_critic and args.px4_native_outer
                                  and args.waypoint_tracking and args.squared_error_reward
                                  and args.stage == 'free_flight'):
        p.error('--native-torque requires --simple-critic --px4-native-outer --waypoint-tracking '
                '--squared-error-reward --stage free_flight')
    if args.explore_until_horizon and (args.training_phase != 'early' or args.stage != 'free_flight' or args.smoke):
        p.error('--explore-until-horizon requires early free-flight training, without --smoke')
    if args.training_phase:
        # MSE exploration and refinement share the same task and rewards.
        args.reward_scale_multiplier = (1. if args.squared_error_reward else
                                        2. if args.training_phase == "early" else 1.)
        args.ppo_stage = "free_flight" if args.training_phase == "late" else "ball_rig"
    if args.steps <= 0 or args.promotion_cooldown_rollouts < 1:
        p.error("steps and promotion-cooldown-rollouts must be positive")
    if args.slew_weight is not None and (not math.isfinite(args.slew_weight) or args.slew_weight < 0):
        p.error("slew-weight must be finite and nonnegative")
    if args.air_error_scale_deg_s is not None and (not math.isfinite(args.air_error_scale_deg_s) or args.air_error_scale_deg_s <= 0):
        p.error("air-error-scale-deg-s must be finite and positive")
    if args.pwm_slew_scale is not None and (not math.isfinite(args.pwm_slew_scale) or args.pwm_slew_scale <= 0):
        p.error("slew scale must be finite and positive")
    if not math.isfinite(args.reward_scale_multiplier) or args.reward_scale_multiplier <= 0:
        p.error("reward-scale-multiplier must be finite and positive")
    if any(not math.isfinite(x) or x <= 0 for x in args.target_rate_limits):
        p.error("target-rate-limits must be finite and positive")
    if any(not math.isfinite(x) or x <= 0 for x in args.air_rate_limits):
        p.error("air-rate-limits must be finite and positive")
    if args.fixed_altitude is not None and (not math.isfinite(args.fixed_altitude) or args.fixed_altitude <= .3):
        p.error("fixed-altitude must be finite and above the ground threshold")
    if args.waypoint_tracking and (args.fixed_episode_rate_target or args.fixed_altitude is not None):
        p.error("waypoint tracking generates both altitude and rate references")
    args.n_steps = rollout_steps
    args.n_epochs = epochs
    return args
