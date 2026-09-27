import json
import math
from copy import deepcopy

from stable_baselines3 import PPO
import torch

from .adaptive_ppo import AdaptiveKLPPO
from .defaults import CURRENT
from .ppo_schedule import EARLY, AIR, apply_stage_settings
from .action_contract import (ACTION_CONTRACT, CRITIC_CONTRACT, REWARD_CONTRACT,
                              interface_metadata, critic_interface_metadata,
                              reward_interface_metadata, require_training_contract)
from .asymmetric_policy import AsymmetricActorCriticPolicy
from .promotion_training import PromotionTraining
from .trajectory import TrajectoryClock
from .reward_contract import mse_reward_interface_metadata, require_reward_interface
from .training.cli import parse_args
from .training.configuration import build_task_config, build_promotion_options, build_monitor_fields
from .training.environments import build_environment_factories
from .training.metadata import initial_metadata, finalize_reward_metadata
# Backward-compatible callback import paths for old consumers and serialized data.
from .training.metrics import RewardMetrics, CompactMetrics, CONTINUOUS_COMPONENT_NAMES



def main(argv=None):
    args = parse_args(argv)
    rollout_steps, epochs = args.n_steps, args.n_epochs
    if args.native_torque:
        from . import torque_contract
    selected_action_contract = torque_contract.ACTION_CONTRACT if args.native_torque else ACTION_CONTRACT
    selected_reward_contract = torque_contract.REWARD_CONTRACT if args.native_torque else REWARD_CONTRACT
    selected_interface = torque_contract.interface_metadata if args.native_torque else interface_metadata
    check_training_contract = torque_contract.require_training_contract if args.native_torque else require_training_contract
    args.run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    config = build_task_config(args)
    trajectory_clock = TrajectoryClock(args.run / "trajectories")
    promotion_options = build_promotion_options(args)
    monitor_fields = build_monitor_fields(args)

    training_factory, evaluation_factory = build_environment_factories(
        args, config, promotion_options, monitor_fields, trajectory_clock)

    env, curriculum = training_factory()
    backend = curriculum.unwrapped.backend
    gamma = math.exp(-config.dt / 30.0)
    gae_lambda = math.exp(-config.dt / 0.5)
    metadata = initial_metadata(args, config, curriculum, backend, promotion_options,
                                rollout_steps, gamma, gae_lambda)
    ppo_class = AdaptiveKLPPO if args.native_torque else PPO
    if args.resume:
        # Validate before attaching the new Dict environment, so old ordinary
        # critics produce an explicit contract error instead of a shape error.
        model = ppo_class.load(args.resume, device="cpu")
        check_training_contract(model)
        if args.allow_episode_length_change:
            from .episode_horizon import migrate_episode_horizon
            from .environment_contract import environment_contract
            migration = migrate_episode_horizon(model, environment_contract(curriculum))
            metadata['episode_length_migration'] = dict(source=str(args.resume),
                timestep=model.num_timesteps, **migration)
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
        if args.n_envs == 1:
            if model.n_envs != 1:
                raise ValueError('This checkpoint has parallel worker states; resume with --n-envs 2 or 4')
            model.set_env(env)
        model.tensorboard_log = str(args.run / "tensorboard")
        if not hasattr(model, "curriculum_state"):
            raise ValueError("Checkpoint lacks the version 4 frozen-evaluation gate state")
        curriculum.restore(model.curriculum_state)
        if not hasattr(model, "promotion_environment_contract"):
            raise ValueError("Checkpoint lacks its frozen-evaluation environment contract")
        old_sampling = dict(n_envs=model.n_envs, n_steps=model.n_steps,
                            batch_size=model.batch_size, n_epochs=model.n_epochs)
        if model.n_steps * model.n_envs != rollout_steps * args.n_envs:
            raise ValueError("Checkpoint and requested total rollout sizes differ")
        metadata['sampling_migration'] = dict(previous=old_sampling,
            current=dict(n_envs=args.n_envs, n_steps=rollout_steps,
                         batch_size=args.batch_size, n_epochs=epochs), optimizer_preserved=True)
        if hasattr(model, "training_env_rng_state"):
            curriculum.unwrapped.np_random.bit_generator.state = deepcopy(model.training_env_rng_state)
    else:
        adaptive_options = ({"kl_phase": args.training_phase or
                            ("late" if args.ppo_stage == "free_flight" else "early")}
                            if args.native_torque else {})
        model = ppo_class("MlpPolicy" if args.simple_critic else AsymmetricActorCriticPolicy, env, seed=args.seed, learning_rate=EARLY.learning_rate,
                gamma=gamma, gae_lambda=gae_lambda, ent_coef=EARLY.ent_coef,
                n_steps=metadata["n_steps"], batch_size=args.batch_size or CURRENT.batch_size,
                n_epochs=epochs, clip_range=EARLY.clip_range,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64] if args.simple_critic else [128, 128]),
                                   activation_fn=torch.nn.ReLU, log_std_init=-2.3),
                tensorboard_log=str(args.run / "tensorboard"), device="cpu", verbose=1,
                **adaptive_options)
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
    model.n_epochs = epochs
    if args.batch_size is not None:
        model.batch_size = args.batch_size
        model.active_stage_settings['batch_size'] = args.batch_size
    if args.training_phase:
        model.manual_training_phase = args.training_phase
        metadata["manual_training_phase"] = args.training_phase
        metadata["stage_hyperparameters"] = dict(
            early=dict(batch_size=EARLY.batch_size,learning_rate=EARLY.learning_rate,ent_coef=EARLY.ent_coef,clip_range=EARLY.clip_range,
                       rate_scale_deg_s=10.,pwm_slew_scale_per_s=args.pwm_slew_scale if args.pwm_slew_scale is not None else 100.),
            late=dict(batch_size=AIR.batch_size,learning_rate=AIR.learning_rate,ent_coef=AIR.ent_coef,clip_range=AIR.clip_range,
                      rate_scale_deg_s=5.,pwm_slew_scale_per_s=args.pwm_slew_scale if args.pwm_slew_scale is not None else 50.))
    metadata["reward_scale_multiplier"] = args.reward_scale_multiplier
    if args.training_phase and args.air_error_scale_deg_s is not None:
        metadata["stage_hyperparameters"][args.training_phase]["rate_scale_deg_s"] = config.air_reward_error_scale_deg_s
    if args.training_phase and config.squared_error_reward:
        for settings in metadata["stage_hyperparameters"].values():
            settings.update(rate_scale_deg_s=5.,
                            pwm_slew_scale_per_s=config.reward_slew_rate_scale_per_s)
    if args.batch_size is not None:
        for settings in metadata['stage_hyperparameters'].values():
            settings['batch_size'] = args.batch_size
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
    if isinstance(model, AdaptiveKLPPO):
        metadata["optimizer_schedule"] = model.kl_schedule_metadata()
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
        curriculum if args.n_envs == 1 else None,
        getattr(model, "training_schedule_state", None), survival_gate=survival_gate)
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
    finalize_reward_metadata(model, metadata, args, config)
    if args.n_envs > 1:
        from .parallel_training import make_parallel_env, attach_parallel_env, ParallelTrainingCoordinator, resize_worker_states
        from .environment_contract import environment_contract
        contract = environment_contract(curriculum)
        if getattr(model, 'promotion_environment_contract', contract) != contract:
            raise ValueError('Parallel task or simulator differs from the checkpoint contract')
        model.promotion_environment_contract = deepcopy(contract)
        worker_states = resize_worker_states(getattr(model, 'parallel_worker_states', None), args.n_envs,
            seed=args.seed, legacy_rng=getattr(model, 'training_env_rng_state', None))
        # The parent environment is used only to validate the task contract.
        # Physics and resets run exclusively inside the independent workers.
        env.close()
        vec_env = make_parallel_env(args.runtime, args.run, config, promotion_options,
            curriculum_state=curriculum.state(), worker_states=worker_states,
            starting_steps=model.num_timesteps, seed=args.seed,
            base_instance=args.base_instance, n_envs=args.n_envs, monitor_fields=monitor_fields)
        try:
            attach_parallel_env(model, vec_env, n_steps=rollout_steps,
                                batch_size=model.batch_size, n_epochs=epochs)
            coordinator = ParallelTrainingCoordinator(model, vec_env, metrics, args.run)
        except BaseException:
            vec_env.close()
            raise
        metadata.update(n_steps=model.n_steps, n_envs=model.n_envs,
            total_rollout_samples=model.n_steps * model.n_envs,
            parallel_workers=vec_env.env_method('parallel_worker_metadata'),
            trajectory_worker=0)
        metadata['batch_size_schedule'] = dict(mode='explicit override',
            early=model.batch_size, late=model.batch_size, automatic_promotion=False)
    else:
        coordinator = PromotionTraining(model, curriculum, metrics, training_factory, evaluation_factory, args.run)
        metadata.update(n_envs=1, total_rollout_samples=model.n_steps)
    metadata["promotion_environment_contract"] = deepcopy(model.promotion_environment_contract)
    starting_timesteps, starting_updates, starting_batch_size = model.num_timesteps, model._n_updates, model.batch_size
    try:
        if args.n_envs > 1:
            coordinator.sync_state()
        (args.run / "config.json").write_text(json.dumps(metadata, indent=2))
        if args.allow_phase_change or not args.resume:
            model.save(args.run / "phase_start")
        coordinator.learn(total_timesteps=exploration_step_budget(model.num_timesteps) if survival_gate else 2048 if args.smoke else args.steps,
                          callback=metrics, stop_after_rollout=(lambda: survival_gate.passed) if survival_gate else None)
        if args.n_envs == 1:
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
                      curriculum=model.curriculum_state,
                      parallel_workers=[{key: state[key] for key in
                          ('worker_id', 'instance', 'seed', 'samples_seen', 'global_steps')}
                          for state in getattr(model, 'parallel_worker_states', [])],
                      n_envs=model.n_envs, n_steps=model.n_steps,
                      promotion_training=coordinator.state,
                      active_stage_settings=model.active_stage_settings,
                      policy_performance_validated=False)
        if isinstance(model, AdaptiveKLPPO):
            report["optimizer_schedule"] = model.kl_schedule_metadata()
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
