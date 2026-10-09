"""Human-readable run metadata, separate from checkpoint validation contracts."""
from copy import deepcopy
from dataclasses import asdict

from ..action_contract import (
    ACTION_CONTRACT, CRITIC_CONTRACT, REWARD_CONTRACT,
    interface_metadata, critic_interface_metadata, reward_interface_metadata,
)
from ..defaults import CURRENT
from ..ppo_schedule import EARLY, AIR
from ..reward_contract import mse_reward_interface_metadata
from .metrics import CONTINUOUS_COMPONENT_NAMES


def initial_metadata(args, config, curriculum, backend, promotion_options, rollout_steps, gamma, gae_lambda):
    return dict(algorithm="PPO", task=asdict(config), simulator=backend.config, seed=args.seed,
                    gamma=gamma, gae_lambda=gae_lambda, n_steps=rollout_steps,
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


def finalize_reward_metadata(model, metadata, args, config):
    metadata["reward_interface"]["continuous_weights"]["slew_rate"] = config.slew_weight
    metadata["reward_interface"]["slew_rate"]["scale_per_s"] = config.reward_slew_rate_scale_per_s
    metadata["reward_convention"] = metadata["reward_convention"].replace("10% PWM rate cost", f"{100*config.slew_weight:g}% PWM rate cost")
    metadata["reward_interface"]["continuous_bounds"]["weighted_per_step_in_dt_units"]["slew_rate"] = [-config.slew_weight, 0.]
    model.reward_interface_metadata = deepcopy(metadata["reward_interface"])
    if config.squared_error_reward and not args.native_torque:
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
        reward = mse_reward_interface_metadata(config)
        metadata["reward_interface"] = reward
        model.reward_interface_metadata = deepcopy(reward)
        gain = reward["common_reward_gain"]
        cap = reward["mse_cap_deg_s_squared"]
        threshold = reward["tracking_bonus_threshold_deg_s"]
        saturation_threshold = reward["saturation_absolute_threshold"]
        settlement = reward["raw_settlement"]
        early_failure_rate = -settlement["early_failure_extra_max"] / config.episode_seconds
        progress_expression = 'previous_error_sse-current_error_sse'
        if config.error_progress_delta_cap is not None:
            progress_expression = f'clip({progress_expression},-{config.error_progress_delta_cap:g},{config.error_progress_delta_cap:g})'
        metadata["reward_rate_feedback"] = (
            "post-action true degree/s error against preceding target; no error "
            "normalization scale; squared-error sum clipped after summing")
        metadata.update(
            continuous_reward_components=['rate', 'slew_rate', 'tracking_bonus'],
            outer_loop=dict(controller='PX4 native position, velocity and attitude controllers',
                            references='goto_setpoint position and fixed-leg heading'),
            action='3 normalized FRD torque commands [-1,1]; PX4 owns collective thrust and motor allocation',
            thrust_reference_role='PX4-only: forwarded unchanged to native allocator; excluded from actor and critic',
            reward_timing='post-action true rate versus issued reference; slew and signed-limit counts from accepted torque requests; allocated ESC saturation is diagnostic only',
            reward_convention=(
                f'raw per step: -dt*{config.tracking_weight / (10000. * 7e-5):g}*'
                f'min(sum(error_deg_s**2),{cap:g}); '
                f'-dt*{config.slew_weight / 7e-5:g}*mean(B(torque_rate/{config.reward_slew_rate_scale_per_s:g})); '
                f'+dt*{config.tracking_bonus_weight / 7e-5:g}*tracking_eligible; '
                f'-dt*{config.saturation_cost_per_motor_per_s / 7e-5:g}*saturated_torque_axis_count; '
                f'+{config.error_progress_weight:g}*({progress_expression}); '
                f'B(x)=x^2/(1+x^2); signed torque saturation when abs(tau)>={saturation_threshold:g}; '
                f'tracking requires each absolute true rate error <{threshold:g} deg/s, '
                'no failure and no torque-axis saturation; motor saturation is diagnostic only; '
                f'raw failure={settlement["failure_base"]:g}-{early_failure_rate:g}*({config.episode_seconds:g}-T), '
                f'success={settlement["success"]:g}; no headroom or waypoint bonus; '
                f'ALL terms multiplied by {gain:g} for PPO'),
            monitor_saturation_fields=dict(
                episode_torque_saturation_fraction='saturated requested torque-axis steps / (3 * episode steps)',
                episode_motor_saturation_fraction='saturated allocated motor-channel steps / (4 * episode steps); diagnostic only'))
        metadata['effective_reward_scales']['primary_cost'] = (
            f'sum of three squared degree/s errors, clipped at {cap:g}; '
            'no error scale; denominator 10000 unchanged')
        metadata['effective_reward_scales']['tracking_bonus_threshold'] = (
            f'all three absolute tracking errors strictly below {threshold:g} deg/s')
        metadata['effective_reward_scales']['torque_slew_per_s'] = (
            metadata['effective_reward_scales'].pop('pwm_slew_per_s'))
        for settings in metadata.get('stage_hyperparameters', {}).values():
            settings['torque_slew_scale_per_s'] = settings.pop(
                'pwm_slew_scale_per_s', config.reward_slew_rate_scale_per_s)
        metadata['waypoint_reference']['thrust_conversion'] = (
            'none: forward vehicle_rates_setpoint.thrust_body unchanged to vehicle_thrust_setpoint')
        if args.training_phase:
            metadata['batch_size_schedule'] = dict(
                mode='manual phase', early=args.batch_size or CURRENT.batch_size, late=args.batch_size or CURRENT.batch_size,
                automatic_promotion=False)
