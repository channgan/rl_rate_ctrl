import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from copy import deepcopy

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch

from .backend import GazeboPX4Backend
from .env import RateControlEnv, TaskConfig
from .curriculum import RigToFlightCurriculum
from .ppo_schedule import EARLY, AIR, apply_stage_settings
from .action_contract import (ACTION_CONTRACT, CRITIC_CONTRACT, REWARD_CONTRACT,
                              interface_metadata, critic_interface_metadata,
                              reward_interface_metadata, require_training_contract)
from .asymmetric_policy import AsymmetricActorCriticPolicy
from .privileged import PrivilegedCriticObservation
from .promotion_training import PromotionTraining
from .trajectory import EpisodeTrajectory, TrajectoryClock
from .reward_contract import mse_reward_interface_metadata, require_reward_interface


CONTINUOUS_COMPONENT_NAMES = ("rate", "slew_rate", "peak_pwm", "tracking_bonus")


class RewardMetrics(BaseCallback):
    def __init__(self, curriculum=None, previous=None, survival_gate=None):
        super().__init__()
        self.saturated_motor_steps = 0
        self.saturated_torque_axis_steps = 0
        self.successes = 0
        self.failures = 0
        self.first_success_step = None
        self.batch_size_switch_step = None
        self.curriculum = curriculum
        self.survival_gate = survival_gate
        if previous:
            self.first_success_step = previous.get("first_success_step")
            self.batch_size_switch_step = previous.get("batch_size_switch_step")

    def _on_step(self):
        for index, info in enumerate(self.locals["infos"]):
            if self.survival_gate is not None:
                self.survival_gate.observe(info, bool(self.locals['dones'][index]), self.num_timesteps)
            # Native policy limits are three signed torque axes. Allocated
            # motor saturation remains a separate four-channel diagnostic.
            motor_saturation_count = info.get("motor_saturation_count", info["saturation_count"])
            torque_saturation_count = info.get("torque_saturation_count", 0)
            self.saturated_motor_steps += motor_saturation_count
            self.saturated_torque_axis_steps += torque_saturation_count
            self.successes += int(info["is_success"])
            self.failures += int(info["failure"] is not None)
            if info["is_success"] and self.first_success_step is None:
                self.first_success_step = self.num_timesteps
            # Training episodes only feed the prescreen. Stage changes belong
            # to PromotionTraining, after optimization and a frozen exam.
            for name in ("tracking_cost", "rate_bounded_cost", "slew_rate_cost",
                         "upper_headroom", "peak_pwm_cost", "peak_pwm_penalty", "tracking_eligible", "tracking_bonus",
                         "saturation_penalty", "failure_penalty", "success_bonus"):
                self.logger.record_mean("reward/" + name, info[name])
            self.logger.record_mean("control/allocated_motor_saturation_count", motor_saturation_count)
            if "torque_saturation_count" in info:
                self.logger.record_mean("reward/torque_saturation_axis_count", torque_saturation_count)
            else:
                self.logger.record_mean("reward/motor_saturation_count", motor_saturation_count)
            self.logger.record_mean("tracking/error_abs_mean_deg_s",
                                    float(sum(abs(x) for x in info["error_deg_s"]) / 3))
            self.logger.record("reward/error_scale_deg_s", info["reward_error_scale_deg_s"])
            pwm_rates = info["pwm_rate_per_s"]
            pwm_prefix = "control/" if "torque_rate_per_s" in info else "reward/"
            self.logger.record_mean(pwm_prefix + "pwm_rate_abs_mean_per_s", float(sum(abs(x) for x in pwm_rates) / 4))
            self.logger.record_mean(pwm_prefix + "pwm_rate_max_abs_per_s", float(max(abs(x) for x in pwm_rates)))
            if "torque_rate_per_s" in info:
                torque_rates = info["torque_rate_per_s"]
                self.logger.record_mean("reward/torque_rate_abs_mean_per_s", float(sum(abs(x) for x in torque_rates) / 3))
                self.logger.record_mean("reward/torque_rate_max_abs_per_s", float(max(abs(x) for x in torque_rates)))
            components = info["continuous_reward_components"]
            if len(components) != len(CONTINUOUS_COMPONENT_NAMES):
                raise ValueError("Expected all four continuous reward components")
            for name, value in zip(CONTINUOUS_COMPONENT_NAMES, components):
                self.logger.record_mean("reward/weighted_" + name, float(value))
            if "episode_reward_components" in info:
                for name, value in zip(CONTINUOUS_COMPONENT_NAMES, info["episode_reward_components"]):
                    self.logger.record_mean("episode_reward/" + name, float(value))
                for name in ("saturation_penalty", "failure_penalty", "success_bonus"):
                    self.logger.record_mean("episode_reward/" + name, float(info["episode_" + name]))
                self.logger.record_mean("tracking/episode_rate_rmse_deg_s", info["episode_rate_rmse_deg_s"])
                if "episode_saturation_fraction" in info:
                    domain = "torque_axis" if "torque_saturation_count" in info else "motor"
                    self.logger.record_mean("episode_reward/" + domain + "_saturation_fraction",
                                            info["episode_saturation_fraction"])
                if "episode_motor_saturation_fraction" in info:
                    self.logger.record_mean("control/episode_allocated_motor_saturation_fraction",
                                            info["episode_motor_saturation_fraction"])
                self.logger.record_mean("curriculum/transfer_qualified", int(info["transfer_qualified"]))
        if self.curriculum is not None:
            self.curriculum.request_evaluation(self.num_timesteps)
            self.model.curriculum_state = self.curriculum.state()
            self.logger.record("curriculum/window_size", self.curriculum.window_size)
            self.logger.record("curriculum/qualified_count", self.curriculum.qualified_count)
            self.logger.record("curriculum/full", int(self.curriculum.full))
            self.logger.record("curriculum/history", json.dumps(list(self.curriculum.history)))
            self.logger.record("curriculum/free_flight", int(self.curriculum.ready_for_air))
            self.logger.record("curriculum/evaluation_attempts", self.curriculum.attempt_count)
            self.logger.record("curriculum/evaluation_due",
                               int(self.curriculum.evaluation_due(self.num_timesteps)))
        self.model.training_schedule_state = dict(first_success_step=self.first_success_step,
                                                  batch_size_switch_step=self.batch_size_switch_step)
        return True

    def _on_rollout_end(self):
        self.logger.record("train/batch_size", self.model.batch_size)
        self.logger.record("train/ent_coef", self.model.ent_coef)


class CompactMetrics(RewardMetrics):
    """Keep core diagnostics without repeated coordinate/unit variants."""
    def _on_step(self):
        result = super()._on_step()
        for key in list(self.logger.name_to_value):
            if key.startswith(("tracking/", "reward/", "curriculum/", "episode_reward/", "control/")):
                self.logger.name_to_value.pop(key, None)
                self.logger.name_to_count.pop(key, None)
                self.logger.name_to_excluded.pop(key, None)
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", default=str(Path(__file__).resolve().parents[1] / "runtime.local.json"))
    p.add_argument("--steps", type=int, default=100_000)
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
    args = p.parse_args()
    if args.native_torque and not (args.simple_critic and args.px4_native_outer
                                  and args.waypoint_tracking and args.squared_error_reward
                                  and args.stage == 'free_flight'):
        p.error('--native-torque requires --simple-critic --px4-native-outer --waypoint-tracking '
                '--squared-error-reward --stage free_flight')
    if args.native_torque:
        from . import torque_contract
    selected_action_contract = torque_contract.ACTION_CONTRACT if args.native_torque else ACTION_CONTRACT
    selected_reward_contract = torque_contract.REWARD_CONTRACT if args.native_torque else REWARD_CONTRACT
    selected_interface = torque_contract.interface_metadata if args.native_torque else interface_metadata
    check_training_contract = torque_contract.require_training_contract if args.native_torque else require_training_contract
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
    args.run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    defaults = TaskConfig(native_torque=args.native_torque)
    config = TaskConfig(
        tracking_bonus_threshold_deg_s=5. if args.training_phase == 'early' else 2.,
        squared_error_reward=args.squared_error_reward,
        slew_weight=args.slew_weight,
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
    trajectory_clock = TrajectoryClock(args.run / "trajectories")
    promotion_options = dict(window_episodes=args.prescreen_window,
        required_qualified=args.prescreen_required, evaluation_episodes=args.promotion_window,
        evaluation_required=args.promotion_required, cooldown_steps=4096 * args.promotion_cooldown_rollouts)
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

    if args.compact_logs:
        monitor_fields = ()

    def observation_wrapper(env):
        return env if args.simple_critic else PrivilegedCriticObservation(env)

    def training_factory(state=None, rng_state=None, session_index=0):
        backend = GazeboPX4Backend(args.runtime, args.run / "episodes")
        if args.px4_native_outer:
            backend.set_mode('free_flight')
        curriculum = RigToFlightCurriculum(RateControlEnv(backend, config), **promotion_options)
        if state is not None:
            curriculum.restore(state)
        if rng_state is not None:
            curriculum.unwrapped.np_random.bit_generator.state = deepcopy(rng_state)
        name = "monitor.csv" if session_index == 0 else f"monitor.session_{session_index:04d}.csv"
        env = Monitor(EpisodeTrajectory(observation_wrapper(curriculum), trajectory_clock), str(args.run / name),
                      info_keywords=monitor_fields, override_existing=not (args.run / name).exists())
        return env, curriculum

    def evaluation_factory():
        backend = GazeboPX4Backend(args.runtime, args.run / "evaluation_episodes")
        backend.set_mode('free_flight' if args.px4_native_outer else 'ball_rig')
        return observation_wrapper(RateControlEnv(backend, config))

    env, curriculum = training_factory()
    backend = curriculum.unwrapped.backend
    gamma = math.exp(-config.dt / 30.0)
    gae_lambda = math.exp(-config.dt / 0.5)
    metadata = dict(algorithm="PPO", task=asdict(config), simulator=backend.config, seed=args.seed,
                    gamma=gamma, gae_lambda=gae_lambda, n_steps=2048 if args.smoke else 4096,
                    batch_size_schedule=dict(initial=2048, after_survival_and_tracking=256,
                                             promotion_window=args.promotion_window,
                                             promotion_required=args.promotion_required,
                                             prescreen_window=args.prescreen_window,
                                             prescreen_required=args.prescreen_required,
                                             cooldown_training_steps=promotion_options["cooldown_steps"],
                                             gate_version=4,
                                             apply_at="after complete PPO update and frozen evaluation; before new air collection"),
                    outer_loop=asdict(curriculum.unwrapped.outer_loop.config), resume=str(args.resume) if args.resume else None,
                    stage_hyperparameters=dict(ball_rig=asdict(EARLY), free_flight=asdict(AIR)),
                    reward_convention="four continuous terms times dt: 70% rate cost, 10% PWM rate cost, 10% peak PWM cost, 10% eligible tracking bonus; saturation -1.0*dt/motor/step, failure -20-40*(1-survival_fraction) once and success +10 unscaled; no summed reward clipping",
                    reward_contract=REWARD_CONTRACT,
                    reward_interface=reward_interface_metadata(),
                    continuous_reward_components=list(CONTINUOUS_COMPONENT_NAMES),
                    reward_rate_feedback="Gazebo true body rate; reward error converted to deg/s and scaled by 10 in ball_rig / 5 in free_flight; noisy PX4 gyro for actor",
                    interface=interface_metadata(),
                    action_contract=ACTION_CONTRACT,
                    critic_contract=CRITIC_CONTRACT,
                    critic_interface=critic_interface_metadata(),
                    critic="35D privileged value network; independent actor12 and critic35 feature paths and weights",
                    reward_timing="true rate after a_t versus preceding rate target; PWM rate=(a_t-a_previous)/dt; references refresh after reward",
                    thrust_reference_role="actor/outer-loop reference only; no actual-thrust reward; tracking bonus uses the stage rate-error threshold without a thrust gate",
                    action="4 normalised ESC commands [0,1]; ideal simulator speed=command, no allocator; indices 0,1,2,3")
    if args.resume:
        # Validate before attaching the new Dict environment, so old ordinary
        # critics produce an explicit contract error instead of a shape error.
        model = PPO.load(args.resume, device="cpu")
        check_training_contract(model)
        if config.squared_error_reward:
            previous_reward = require_reward_interface(model, mse_reward_interface_metadata(config),
                                                      allow_rate_sum_change=args.allow_rate_sum_change,
                                                      allow_failure2700_change=args.allow_failure2700_change,
                                                      allow_tracking_tuning=args.allow_tracking_tuning,
                                                      allow_axis_success_change=args.allow_axis_success_change)
            if args.allow_axis_success_change:
                metadata['axis_success_migration'] = dict(source=str(args.resume), timestep=model.num_timesteps,
                    previous=previous_reward['horizon_success'], current=mse_reward_interface_metadata(config)['horizon_success'], optimizer_preserved=True)
            if args.allow_tracking_tuning:
                metadata['tracking_reward_migration'] = dict(source=str(args.resume), timestep=model.num_timesteps,
                    previous=previous_reward, current=mse_reward_interface_metadata(config), optimizer_preserved=True)
            if args.allow_failure2700_change:
                metadata['failure_settlement_migration'] = dict(source=str(args.resume), timestep=model.num_timesteps,
                    previous=previous_reward['raw_settlement'], current=mse_reward_interface_metadata(config)['raw_settlement'], optimizer_preserved=True)
            if args.allow_rate_sum_change:
                metadata['rate_sum_migration'] = dict(source=str(args.resume), timestep=model.num_timesteps,
                    previous=previous_reward, current=mse_reward_interface_metadata(config), optimizer_preserved=True)
        model.set_env(env)
        model.tensorboard_log = str(args.run / "tensorboard")
        if not hasattr(model, "curriculum_state"):
            raise ValueError("Checkpoint lacks the version 4 frozen-evaluation gate state")
        curriculum.restore(model.curriculum_state)
        if not hasattr(model, "promotion_environment_contract"):
            raise ValueError("Checkpoint lacks its frozen-evaluation environment contract")
        expected_n_steps = 2048 if args.smoke else 4096
        if model.n_steps != expected_n_steps:
            raise ValueError(f"This run requires n_steps={expected_n_steps}; checkpoint uses {model.n_steps}")
        if hasattr(model, "training_env_rng_state"):
            curriculum.unwrapped.np_random.bit_generator.state = deepcopy(model.training_env_rng_state)
    else:
        model = PPO("MlpPolicy" if args.simple_critic else AsymmetricActorCriticPolicy, env, seed=args.seed, learning_rate=EARLY.learning_rate,
                gamma=gamma, gae_lambda=gae_lambda, ent_coef=EARLY.ent_coef,
                n_steps=metadata["n_steps"], batch_size=2048,
                n_epochs=2 if args.smoke else 10, clip_range=EARLY.clip_range,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64] if args.simple_critic else [128, 128]),
                                   activation_fn=torch.nn.ReLU, log_std_init=-2.3),
                tensorboard_log=str(args.run / "tensorboard"), device="cpu", verbose=1)
        # Initialise only new networks; resume preserves learned actor weights.
        with torch.no_grad():
            model.policy.action_net.bias.fill_(0. if args.native_torque else backend.hover)
        model.action_contract = selected_action_contract
        model.critic_contract = (torque_contract.CRITIC_CONTRACT if args.native_torque else
            "ordinary_actor12_critic12_64x64_v1" if args.simple_critic else CRITIC_CONTRACT)
        model.reward_contract = selected_reward_contract
    trajectory_clock.steps = model.num_timesteps
    metadata["trajectory_recording"] = dict(interval_steps=40000, directory="trajectories",
        selection="complete training episode containing each global step milestone; partial on close")
    model.interface_metadata = selected_interface()
    metadata['interface'] = model.interface_metadata
    metadata['action_contract'] = selected_action_contract
    metadata['reward_contract'] = selected_reward_contract
    model.critic_interface_metadata = (dict(version=model.critic_contract, observation_size=9 if args.native_torque else 12,
        privileged=False, net_arch=dict(pi=[64,64],vf=[64,64])) if args.simple_critic else critic_interface_metadata())
    metadata["critic_contract"] = model.critic_contract
    metadata["critic_interface"] = model.critic_interface_metadata
    metadata["critic"] = "actor12 and critic12, independent 64x64 networks" if args.simple_critic else metadata["critic"]
    if args.native_torque:
        metadata['critic'] = 'actor9 and critic9, independent 64x64 networks'
    metadata["compact_logs"] = args.compact_logs
    if args.waypoint_tracking:
        from .waypoint import waypoint_gate_metadata
        previous_gate = deepcopy(getattr(model, 'waypoint_gate_metadata', None))
        current_gate = waypoint_gate_metadata()
        if args.resume and previous_gate != current_gate:
            if not args.allow_waypoint_gate_change or curriculum.pending_evaluation is not None:
                raise ValueError('Waypoint switching rule changed; explicit --allow-waypoint-gate-change required without pending evaluation')
            metadata['waypoint_gate_migration'] = dict(source=str(args.resume), timestep=model.num_timesteps,
                previous=previous_gate, current=current_gate, optimizer_preserved=True)
        model.waypoint_gate_metadata = current_gate
        metadata["waypoint_reference"] = dict(horizontal_half_width_m=5., vertical_half_width_m=1.,
            acceptance_radius_m=waypoint_gate_metadata()['distance_strictly_below_m'], distance="3D NED Euclidean", initial_separation_m=2.,
            tracking_gate=waypoint_gate_metadata(),
            position_p=1., velocity_d=1.8, horizontal_accel_limit_m_s2=3., heading="set bearing once for each new waypoint and hold throughout the leg; retain heading for vertical legs",
            switch="after current transition reward, before next observation; no episode reset")
        if args.px4_native_outer:
            metadata['waypoint_reference'].update(
                controller='PX4 mc_pos_control + mc_att_control, via goto_setpoint and vehicle_rates_setpoint',
                position_p=None, velocity_d=None, horizontal_accel_limit_m_s2=None,
                state_feedback='PX4 EKF local position and attitude',
                thrust_conversion='-vehicle_rates_setpoint.thrust_body[2] / MPC_THR_HOVER; HTE disabled',
                control_source_version='px4_native_goto_v1')
    model.reward_interface_metadata = reward_interface_metadata()
    if args.stage:
        curriculum.override_stage(args.stage)
    apply_stage_settings(model, args.ppo_stage == "free_flight" if args.ppo_stage else curriculum.ready_for_air)
    if args.training_phase:
        from stable_baselines3.common.utils import FloatSchedule
        learning_rate = 1e-3 if args.training_phase == "early" else 3e-4
        model.learning_rate = learning_rate
        model.lr_schedule = FloatSchedule(learning_rate)
        model.active_stage_settings["learning_rate"] = learning_rate
        model.manual_training_phase = args.training_phase
        metadata["manual_training_phase"] = args.training_phase
        metadata["stage_hyperparameters"] = dict(
            early=dict(batch_size=2048,learning_rate=1e-3,ent_coef=.02,clip_range=.3,
                       rate_scale_deg_s=10.,pwm_slew_scale_per_s=args.pwm_slew_scale if args.pwm_slew_scale is not None else 100.),
            late=dict(batch_size=256,learning_rate=3e-4,ent_coef=.01,clip_range=.2,
                      rate_scale_deg_s=5.,pwm_slew_scale_per_s=args.pwm_slew_scale if args.pwm_slew_scale is not None else 50.))
    metadata["reward_scale_multiplier"] = args.reward_scale_multiplier
    if args.training_phase and args.air_error_scale_deg_s is not None:
        metadata["stage_hyperparameters"][args.training_phase]["rate_scale_deg_s"] = config.air_reward_error_scale_deg_s
    if args.training_phase and config.squared_error_reward:
        for settings in metadata["stage_hyperparameters"].values():
            settings.update(rate_scale_deg_s=5.,
                            pwm_slew_scale_per_s=config.reward_slew_rate_scale_per_s)
    metadata["ppo_stage_override"] = args.ppo_stage
    metadata["effective_reward_scales"] = dict(
        ball_rig_deg_s=config.reward_error_scale_deg_s,
        free_flight_deg_s=config.air_reward_error_scale_deg_s,
        tracking_bonus_threshold="same as effective stage rate-error scale",
        pwm_slew_per_s=config.reward_slew_rate_scale_per_s)
    model.curriculum_state = curriculum.state()
    metadata["initial_curriculum"] = curriculum.state()
    metadata["simulator"] = dict(backend.config)
    if args.px4_native_outer:
        filters = dict(IMU_GYRO_CUTOFF=80., IMU_DGYRO_CUTOFF=40.)
        metadata['px4_gyro_filters'] = filters
        if args.resume:
            metadata['px4_filter_migration'] = dict(source=str(args.resume),
                previous=getattr(model, 'px4_gyro_filters', None), current=filters,
                note='Older checkpoints lack filter metadata; previous run ULog measured 40/20 Hz')
        model.px4_gyro_filters = filters
    metadata.update(n_steps=model.n_steps, n_epochs=model.n_epochs,
                    gamma=model.gamma, gae_lambda=model.gae_lambda)
    initial = {name: value.detach().clone() for name, value in model.policy.named_parameters()}
    from .survival_gate import FullEpisodeGate, exploration_step_budget, EXPLORATION_MAX_TOTAL_STEPS
    survival_gate = (FullEpisodeGate(round(config.episode_seconds/config.dt),
                                    getattr(model, 'exploration_survival_gate', None))
                     if args.explore_until_horizon else None)
    if survival_gate is not None:
        model.exploration_survival_gate = survival_gate.state()
        metadata['initial_exploration_gate'] = survival_gate.state()
        metadata['exploration_max_total_steps'] = EXPLORATION_MAX_TOTAL_STEPS
    metrics = (CompactMetrics if args.compact_logs else RewardMetrics)(
        curriculum, getattr(model, "training_schedule_state", None), survival_gate=survival_gate)
    metadata['exploration_exit'] = ('ten consecutive complete surviving episodes or 300000 total steps; finish rollout and optimizer update'
                                    if survival_gate else 'step budget')
    if args.allow_phase_change:
        if not args.resume or not args.training_phase or args.allow_slew_weight_change:
            raise ValueError("Phase migration requires resume and manual phase, without weight migration")
        from .environment_contract import environment_contract
        from .ppo_schedule import validate_phase_migration
        old = deepcopy(model.promotion_environment_contract)
        new = environment_contract(curriculum)
        changes = validate_phase_migration(old, new, curriculum.pending_evaluation, allow_tracking_tuning=args.allow_tracking_tuning)
        metadata["phase_migration"] = dict(source=str(args.resume), timestep=model.num_timesteps,
            changes=changes, optimizer_preserved=True, phase=args.training_phase)
        model.promotion_environment_contract = new
    metadata["active_stage_settings"] = deepcopy(model.active_stage_settings)
    if args.resume and args.allow_slew_weight_change:
        from .environment_contract import environment_contract
        old = deepcopy(model.promotion_environment_contract)
        new = environment_contract(curriculum)
        expected = deepcopy(old)
        expected["task"]["slew_weight"] = config.slew_weight
        if expected != new or curriculum.pending_evaluation is not None:
            raise ValueError("Explicit slew migration permits only a weight change and no pending evaluation")
        metadata["reward_migration"] = dict(source=str(args.resume),
            old_slew_weight=old["task"]["slew_weight"], new_slew_weight=config.slew_weight,
            timestep=model.num_timesteps, optimizer_preserved=True)
        model.promotion_environment_contract = new
    metadata["reward_interface"]["continuous_weights"]["slew_rate"] = config.slew_weight
    metadata["reward_interface"]["slew_rate"]["scale_per_s"] = config.reward_slew_rate_scale_per_s
    metadata["reward_convention"] = metadata["reward_convention"].replace("10% PWM rate cost", f"{100*config.slew_weight:g}% PWM rate cost")
    metadata["reward_interface"]["continuous_bounds"]["weighted_per_step_in_dt_units"]["slew_rate"] = [-config.slew_weight, 0.]
    model.reward_interface_metadata = deepcopy(metadata["reward_interface"])
    if config.squared_error_reward:
        metadata['effective_reward_scales'].update(
            primary_cost='sum of three squared degree/s errors, clipped at 10000; no error scale',
            tracking_bonus_threshold='all three absolute tracking errors strictly below 5 deg/s')
        metadata["reward_convention"] = "raw: -dt*min(sum(error_deg_s**2),10000), -dt*(10000/7)*bounded_slew, +dt*(20000/7)*mean(1-pwm), +dt*(10000/7)*tracking_eligible; saturation/failure and success=0.5*base_failure multiplied by 10000/.7; ALL terms scaled by 7e-5 for PPO"
        metadata["reward_rate_feedback"] = "post-action true degree/s error against preceding target; no error normalization scale; squared-error sum clipped after summing"
        metadata["reward_interface"] = mse_reward_interface_metadata(config)
        metadata["reward_convention"] += "; mean headroom reward applies immediately when all three post-action errors <5 deg/s and no failure; horizon success also requires whole-episode RMSE <5 deg/s"
        metadata["reward_convention"] = metadata["reward_convention"].replace(
            "saturation/failure and success=0.5*base_failure multiplied by 10000/.7",
            "saturation multiplied by 10000/.7; raw failure=-30000-2500*episode_seconds*(1-survival_fraction), raw success=50000")
        model.reward_interface_metadata = deepcopy(metadata["reward_interface"])
    if args.native_torque:
        metadata.update(
            continuous_reward_components=['rate', 'slew_rate', 'tracking_bonus'],
            outer_loop=dict(controller='PX4 native position, velocity and attitude controllers',
                            references='goto_setpoint position and fixed-leg heading'),
            action='3 normalized FRD torque commands [-1,1]; PX4 owns collective thrust and motor allocation',
            thrust_reference_role='PX4-only: forwarded unchanged to native allocator; excluded from actor and critic',
            reward_timing='post-action true rate versus issued reference; slew and signed-limit counts from accepted torque requests; allocated ESC saturation is diagnostic only',
            reward_convention='Rate SSE capped at 5000 (normalization denominator remains 10000); three-axis requested torque slew; signed torque saturation counts each axis independently when abs(tau)>=0.999; '
                              'no headroom reward; no waypoint bonus; instantaneous tracking bonus requires all three absolute rate errors <2 deg/s, '
                              'no failure and no torque-axis saturation, without a motor-saturation gate; raw failure=-30000-2500*(30-T), success=50000',
            monitor_saturation_fields=dict(
                episode_torque_saturation_fraction='saturated requested torque-axis steps / (3 * episode steps)',
                episode_motor_saturation_fraction='saturated allocated motor-channel steps / (4 * episode steps); diagnostic only'))
        metadata['effective_reward_scales']['primary_cost'] = 'sum of three squared degree/s errors, clipped at 5000; no error scale; denominator 10000 unchanged'
        metadata['effective_reward_scales']['tracking_bonus_threshold'] = 'all three absolute tracking errors strictly below 2 deg/s'
        metadata['effective_reward_scales']['tracking_bonus_threshold'] = f'all three absolute tracking errors strictly below {config.tracking_bonus_threshold_deg_s:g} deg/s'
        metadata['reward_convention'] = metadata['reward_convention'].replace('<2 deg/s', f'<{config.tracking_bonus_threshold_deg_s:g} deg/s')
        metadata['effective_reward_scales']['torque_slew_per_s'] = (
            metadata['effective_reward_scales'].pop('pwm_slew_per_s'))
        for settings in metadata.get('stage_hyperparameters', {}).values():
            settings['torque_slew_scale_per_s'] = settings.pop(
                'pwm_slew_scale_per_s', config.reward_slew_rate_scale_per_s)
        metadata['waypoint_reference']['thrust_conversion'] = (
            'none: forward vehicle_rates_setpoint.thrust_body unchanged to vehicle_thrust_setpoint')
        if args.training_phase:
            metadata['batch_size_schedule'] = dict(
                mode='manual phase', early=2048, late=256,
                automatic_promotion=False)
    coordinator = PromotionTraining(model, curriculum, metrics, training_factory, evaluation_factory, args.run)
    metadata["promotion_environment_contract"] = deepcopy(model.promotion_environment_contract)
    (args.run / "config.json").write_text(json.dumps(metadata, indent=2))
    if args.allow_phase_change:
        model.save(args.run / "phase_start")
    starting_timesteps, starting_updates, starting_batch_size = model.num_timesteps, model._n_updates, model.batch_size
    try:
        coordinator.learn(total_timesteps=exploration_step_budget(model.num_timesteps) if survival_gate else 2048 if args.smoke else args.steps,
                          callback=metrics, stop_after_rollout=(lambda: survival_gate.passed) if survival_gate else None)
        curriculum = coordinator.curriculum
        model.save(args.run / "actor_critic")
        delta = max((value.detach() - initial[name]).abs().max().item()
                    for name, value in model.policy.named_parameters()
                    if name.startswith(("mlp_extractor.policy_net", "action_net")))
        report = dict(algorithm="PPO", timesteps=model.num_timesteps, update_epochs=model._n_updates,
                      actor_max_parameter_change=delta, saturated_motor_steps=metrics.saturated_motor_steps,
                      saturated_torque_axis_steps=metrics.saturated_torque_axis_steps,
                      successes=metrics.successes, failures=metrics.failures,
                      timesteps_this_run=model.num_timesteps-starting_timesteps,
                      update_epochs_this_run=model._n_updates-starting_updates,
                      initial_batch_size=starting_batch_size, final_batch_size=model.batch_size,
                      first_success_step=metrics.first_success_step,
                      batch_size_switch_step=metrics.batch_size_switch_step,
                      curriculum=curriculum.state(),
                      promotion_training=coordinator.state,
                      active_stage_settings=model.active_stage_settings,
                      policy_performance_validated=False)
        if survival_gate:
            report['exploration_gate'] = survival_gate.state()
            report['exploration_gate']['post_update_step'] = model.num_timesteps
            report['exploration_gate']['exit_reason'] = coordinator.state['stop_reason']
            report['exploration_gate']['max_total_steps'] = EXPLORATION_MAX_TOTAL_STEPS
        (args.run / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        model.get_env().close()


if __name__ == "__main__":
    main()
