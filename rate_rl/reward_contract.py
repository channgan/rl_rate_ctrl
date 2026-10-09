"""Describe the implemented MSE objective and reject incompatible resumes.

The legacy ``REWARD_CONTRACT`` predates several MSE objective revisions.  Its
unchanged identifier cannot establish that a checkpoint learned this reward.
"""
from copy import deepcopy

from .defaults import CURRENT


def early_failure_rate(config):
    return getattr(config, "failure_rate_per_s", 3500.) if getattr(config, "native_torque", False) else 2500.


def rate_error_cap(config):
    return CURRENT.rate_error_cap if getattr(config, 'native_torque', False) else 10000.


def mse_reward_interface_metadata(config):
    """Return the current objective, shared by config output and resume checks."""
    if not config.squared_error_reward:
        raise ValueError("MSE reward metadata requires squared_error_reward=True")
    metadata = dict(
        version="sse10000_headroom2_waypoint1000_noise5_v11",
        waypoint_switch_raw_bonus=1000.,
        rate_aggregation="sum of three squared degree/s errors, then cap",
        common_reward_gain=config.reward_gain,
        raw_settlement=dict(
            success=CURRENT.success_bonus, failure_base=-CURRENT.failure_base, early_failure_extra_max=-early_failure_rate(config)*config.episode_seconds,
            failure_formula=f"-30000-{early_failure_rate(config):g}*episode_seconds*(1-survival_fraction)"),
        headroom_gate=dict(
            strict_error_deg_s=5., consecutive_seconds=0., requires_no_failure=True),
        horizon_success=dict(
            whole_episode_rmse_strictly_below_deg_s=5.,
            aggregation="sqrt(mean over all steps and three axes of squared degree/s errors)",
            failure_reason="tracking_rmse", success_raw=CURRENT.success_bonus),
        mse_cap_deg_s_squared=rate_error_cap(config),
        tracking_bonus_threshold_deg_s=5.,
        continuous_weights=dict(
            rate=config.tracking_weight, slew=config.slew_weight,
            mean_headroom=2. * config.peak_pwm_weight,
            tracking_bonus=config.tracking_bonus_weight),
        slew_scale_per_s=config.reward_slew_rate_scale_per_s,
        legacy_component_slot_peak_pwm=(
            "positive mean headroom reward when current three-axis errors are all <5 deg/s"),
    )
    if getattr(config, 'native_torque', False):
        metadata.update(
            version='sse10000_native_torque_slew_limits_no_headroom_no_waypoint_bonus_v15',
            waypoint_switch_raw_bonus=0.,
            headroom_gate=None,
            legacy_component_slot_peak_pwm='disabled: collective thrust is controlled by PX4',
            motor_command_source='latest actual PX4 actuator_motors.control normalized ESC commands in applied_pwm',
            slew_input='(current accepted requested normalized torque - previous accepted requested normalized torque) / dt',
            slew_aggregation='mean over three axes of B(torque_rate / slew_scale_per_s), B(x)=x^2/(1+x^2)',
            slew_history_initialization='zero torque before the first accepted policy action of each episode',
            saturation_input='current accepted normalized torque, each of three signed axes independently',
            saturation_absolute_threshold=1. - config.saturation_epsilon,
            saturation_comparison='abs(torque) >= threshold; zero torque is not saturated',
            saturation_cost_per_axis_per_s=config.saturation_cost_per_motor_per_s,
            motor_saturation_role='diagnostic only; does not affect native torque reward or tracking bonus',
            tracking_bonus_threshold_deg_s=config.tracking_bonus_threshold_deg_s,
            tracking_bonus_gate=f'each true rate error strictly <{config.tracking_bonus_threshold_deg_s:g} deg/s; no failure; no saturated torque axis',
            collective_thrust_control='native PX4; no actor thrust observation or thrust action',
        )
        metadata['continuous_weights']['mean_headroom'] = 0.
        metadata['horizon_success']['aggregation'] = 'each axis separately: sqrt(mean over all episode steps of squared degree/s error); all three strictly below threshold'
    weight = getattr(config, 'error_progress_weight', 0.)
    if weight:
        if not getattr(config, 'native_torque', False):
            raise ValueError('Error progress reward requires native torque')
        metadata.update(
            version='native_torque_sse_consecutive_error_progress_v16',
            error_progress=dict(weight=weight,
                raw_formula='weight * (previous_true_error_sse_deg_s2 - current_true_error_sse_deg_s2)',
                history='reset true error seeds first action; each subsequent error uses its own action reference',
                reference_changes='included literally; no rebase or gate at waypoint changes',
                cap=None, dt_factor=False, terminal='included before independent outcome settlement',
                ppo_gain=config.reward_gain,
                accounting='separate error_progress_reward and episode_error_progress_reward; not in legacy four components'))
        if config.error_progress_delta_cap is not None:
            metadata['version'] = 'native_torque_sse_clipped_error_progress_v17'
            metadata['error_progress'].update(
                raw_formula='weight * clip(previous_true_error_sse_deg_s2 - current_true_error_sse_deg_s2, -delta_cap, delta_cap)',
                cap=dict(delta_cap_deg_s_squared=config.error_progress_delta_cap,
                         raw_reward_absolute_max=weight * config.error_progress_delta_cap,
                         history='unclipped actual SSE; clipping does not alter history'))
    return metadata


def require_reward_interface(model, expected, allow_rate_sum_change=False, allow_failure2700_change=False, allow_tracking_tuning=False, allow_axis_success_change=False):
    """Check the saved objective before train.py replaces model metadata.

    TaskConfig and SDF equality do not catch changes to Python reward formulas.
    A missing or different saved objective therefore requires an explicit
    migration decision; this function never repairs metadata in place.
    """
    saved = getattr(model, "reward_interface_metadata", None)
    if allow_axis_success_change and isinstance(saved, dict):
        candidate = deepcopy(saved)
        if candidate.get('horizon_success', {}).get('aggregation') == 'sqrt(mean over all steps and three axes of squared degree/s errors)':
            candidate['horizon_success']['aggregation'] = expected['horizon_success']['aggregation']
            if candidate == expected:
                return deepcopy(saved)
    if allow_tracking_tuning and isinstance(saved, dict):
        candidate = deepcopy(saved)
        for key in ('tracking_bonus_threshold_deg_s', 'tracking_bonus_gate'):
            candidate[key] = expected[key]
        for key in ('slew', 'tracking_bonus'):
            candidate['continuous_weights'][key] = expected['continuous_weights'][key]
        if (candidate == expected and expected['mse_cap_deg_s_squared'] == 5000
                and expected['tracking_bonus_threshold_deg_s'] in (2., 5.)
                and expected['continuous_weights']['slew'] in (.01 / 7., .02 / 7.)
                and expected['continuous_weights']['tracking_bonus'] == .0084):
            return deepcopy(saved)
    if allow_failure2700_change and isinstance(saved, dict):
        candidate = deepcopy(saved)
        settlement = candidate.get('raw_settlement', {})
        if (candidate.get('version') == 'sse10000_native_torque_slew_limits_no_headroom_no_waypoint_bonus_v15'
                and settlement.get('failure_formula') == '-30000-2500*episode_seconds*(1-survival_fraction)'
                and settlement.get('early_failure_extra_max') == -75000.):
            settlement['failure_formula'] = '-30000-2700*episode_seconds*(1-survival_fraction)'
            settlement['early_failure_extra_max'] = -81000.
            if candidate == expected:
                return deepcopy(saved)
    if allow_rate_sum_change and isinstance(saved, dict):
        candidate = deepcopy(saved)
        if candidate.get('version') == 'mse10000_tracking5_rmse5_failure2500persecond_v8':
            candidate['version'] = 'sse10000_headroom2_tracking5_rmse5_failure2500persecond_v10'
            candidate['rate_aggregation'] = 'sum of three squared degree/s errors, then cap'
            candidate['continuous_weights']['mean_headroom'] *= 2.
            if candidate == expected:
                return deepcopy(saved)
    if not isinstance(saved, dict) or saved != expected:
        previous = saved.get("version") if isinstance(saved, dict) else None
        current = expected.get("version")
        raise ValueError(
            "Checkpoint reward interface mismatch: "
            f"saved={previous!r}, expected={current!r}. "
            "The legacy reward_contract alone is insufficient; use a matching "
            "checkpoint or explicitly migrate the reward objective before resuming.")
    return deepcopy(saved)
