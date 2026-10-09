"""Pure reward and terminal-outcome calculations for an executed transition.

The environment owns simulator I/O, reference advancement and episode history.
These functions only read their inputs. In particular, ``error_rad_s`` belongs
to the reference that produced this action, never the next reference. Returned
rewards already use the PPO scale; the existing diagnostic contract is retained.
Legacy direct-PWM objectives remain explicit here for saved-task compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .defaults import CURRENT
from .reward_contract import early_failure_rate, rate_error_cap

if TYPE_CHECKING:
    from .env import TaskConfig


@dataclass(frozen=True)
class EpisodeOutcome:
    failure: str | None
    success: bool
    altitude: float
    tilt: float
    success_bonus: float
    failure_penalty: float

    @property
    def terminated(self) -> bool:
        return self.failure is not None or self.success


def evaluate_outcome(
    config: TaskConfig,
    state: Mapping[str, Any],
    *,
    on_rig: bool,
    completed_steps: int,
    error_rad_s: np.ndarray,
    previous_error_squared: float,
    previous_axis_error_squared: np.ndarray,
) -> EpisodeOutcome:
    """Classify the current endpoint, including it in full-episode RMSE.

    ``completed_steps`` and squared-error sums refer to steps *before* the
    current transition. Physical failures take priority over horizon tracking
    failure. The finite horizon is terminal, not a Gymnasium truncation.
    """
    c = config
    q = np.asarray(state['q_ned_frd'], dtype=float).copy()
    q /= np.linalg.norm(q)
    cos_tilt = 1 - 2 * (q[1]**2 + q[2]**2)
    tilt = np.arccos(np.clip(cos_tilt, -1, 1))
    altitude = -float(state['position_ned'][2])
    failure = None
    if not on_rig and altitude < .3:
        failure = 'ground'
    elif not on_rig and c.terminate_on_tilt and tilt > np.deg2rad(c.max_tilt_deg):
        failure = 'tilt'
    elif np.max(np.abs(state['rates_true'])) > c.max_rate_rad_s:
        failure = 'rate'
    elif c.px4_native_outer and not state.get('px4_position_control'):
        failure = 'px4_control_lost'

    steps = completed_steps + 1
    horizon_reached = steps >= round(c.episode_seconds / c.dt)
    horizon_rmse_deg_s = float(np.rad2deg(np.sqrt(
        (previous_error_squared + float(np.mean(error_rad_s ** 2))) / steps)))
    horizon_axis_rmse = np.rad2deg(np.sqrt(
        (previous_axis_error_squared + error_rad_s ** 2) / steps))
    tracking_failed = (bool(np.any((horizon_axis_rmse >= 5.) | np.isclose(
        horizon_axis_rmse, 5., rtol=0., atol=1e-12))) if c.native_torque
        else horizon_rmse_deg_s >= 5.)
    if failure is None and horizon_reached and tracking_failed:
        failure = 'tracking_rmse'

    success = failure is None and horizon_reached
    success_bonus = .5 * c.failure_cost if success else 0.
    survival_fraction = np.clip(steps * c.dt / c.episode_seconds, 0., 1.)
    failure_penalty = (c.failure_cost + c.early_failure_cost * (1. - survival_fraction)
                       if failure is not None else 0.)
    if c.squared_error_reward:
        success_bonus = CURRENT.success_bonus * c.reward_gain if success else 0.
        failure_penalty = ((CURRENT.failure_base + early_failure_rate(c) * c.episode_seconds
                            * (1. - survival_fraction)) * c.reward_gain
                           if failure is not None else 0.)
    return EpisodeOutcome(failure, bool(success), altitude, float(tilt),
                          success_bonus, failure_penalty)


@dataclass(frozen=True)
class StepReward:
    value: float
    components: np.ndarray
    saturation_count: int
    motor_saturation_count: int
    saturation_penalty: float
    diagnostics: dict[str, Any]


def evaluate_step_reward(
    config: TaskConfig,
    *,
    error_deg_s: np.ndarray,
    error_scale_deg_s: float,
    pwm_action: np.ndarray,
    previous_pwm: np.ndarray,
    accepted_torque: np.ndarray | None,
    previous_torque: np.ndarray,
    failure: str | None,
) -> StepReward:
    """Compute continuous income/cost and signed-action saturation penalty.

    Native mode charges accepted torque changes and torque boundaries, not
    collective motor output. Zero torque remains valid. The terminal step is
    charged normally; only its tracking bonus is gated by the outcome.
    """
    c = config
    axis_cost = (error_deg_s / error_scale_deg_s) ** 2
    tracking = float(np.mean(axis_cost))
    commanded_throttle = float(np.mean(pwm_action.astype(float)))
    pwm_rate = (pwm_action.astype(float) - previous_pwm.astype(float)) / c.dt
    if c.native_torque:
        if accepted_torque is None:
            raise ValueError('Native torque reward requires accepted torque feedback')
        penalized_rate = (accepted_torque - previous_torque.astype(float)) / c.dt
        slew_rate_source = 'accepted_normalized_torque'
    else:
        penalized_rate = pwm_rate
        slew_rate_source = 'applied_pwm'
    slew_rate = float(np.mean(penalized_rate ** 2))
    rate_normalised = float(np.mean(axis_cost / (1 + axis_cost)))
    if c.squared_error_reward:
        tracking = min(float(np.sum(error_deg_s ** 2)), rate_error_cap(c))
        rate_normalised = tracking / 10000.
    slew_scaled = (penalized_rate / c.reward_slew_rate_scale_per_s) ** 2
    slew_normalised = float(np.mean(slew_scaled / (1 + slew_scaled)))
    components = c.dt * np.array([-c.tracking_weight * rate_normalised,
                                 -c.slew_weight * slew_normalised, 0., 0.])
    reward = float(components.sum())

    saturated_motors = ((pwm_action <= c.saturation_epsilon)
                        | (pwm_action >= 1 - c.saturation_epsilon))
    motor_saturation_count = int(np.count_nonzero(saturated_motors))
    if c.native_torque:
        saturated_torque_axes = np.abs(accepted_torque) >= 1 - c.saturation_epsilon
        saturation_count = int(np.count_nonzero(saturated_torque_axes))
        saturation_source = 'accepted_normalized_torque'
    else:
        saturation_count = motor_saturation_count
        saturation_source = 'allocated_motor'
    saturation_penalty = c.saturation_cost_per_motor_per_s * c.dt * saturation_count
    reward -= saturation_penalty

    peak_pwm = float(np.max(pwm_action))
    upper_headroom = 1 - peak_pwm
    peak_pwm_penalty = c.dt * c.peak_pwm_weight * peak_pwm
    if c.squared_error_reward:
        upper_headroom = 1. - float(np.mean(pwm_action))
        tracking_now = failure is None and np.all(np.abs(error_deg_s) < 5.)
        peak_pwm_penalty = (-2. * c.dt * c.peak_pwm_weight * upper_headroom
                            if tracking_now else 0.)
    if c.native_torque:
        peak_pwm_penalty = 0.
    reward -= peak_pwm_penalty
    components[2] = -peak_pwm_penalty
    tracking_eligible = bool(
        failure is None and saturation_count == 0
        and (np.max(np.abs(error_deg_s)) < (c.tracking_bonus_threshold_deg_s
             if c.native_torque else 5.) if c.squared_error_reward
             else np.max(np.abs(error_deg_s)) <= error_scale_deg_s))
    tracking_bonus = c.dt * c.tracking_bonus_weight if tracking_eligible else 0.
    reward += tracking_bonus
    components[3] = tracking_bonus

    # Stored continuous coefficients use historical 7e-5 units. Convert all
    # monetary terms together, preserving their raw values and relative weights.
    if c.native_torque and c.reward_gain != 7e-5:
        factor = c.reward_gain / 7e-5
        reward *= factor
        components *= factor
        saturation_penalty *= factor
        peak_pwm_penalty *= factor
        tracking_bonus *= factor

    diagnostics = dict(
        tracking_cost=tracking, rate_bounded_cost=rate_normalised,
        reward_error_scale_deg_s=error_scale_deg_s, slew_rate_cost=slew_rate,
        slew_rate_source=slew_rate_source, pwm_rate_per_s=pwm_rate.copy(),
        continuous_reward_components=components.copy(),
        commanded_throttle=commanded_throttle, upper_headroom=upper_headroom,
        peak_pwm_cost=peak_pwm, peak_pwm_penalty=peak_pwm_penalty,
        tracking_eligible=tracking_eligible, tracking_bonus=tracking_bonus,
        saturated_motors=saturated_motors.copy(),
        motor_saturation_count=motor_saturation_count,
        saturation_count=saturation_count, saturation_source=saturation_source,
        saturation_penalty=saturation_penalty)
    if c.native_torque:
        diagnostics.update(torque_rate_per_s=penalized_rate.copy(),
                           saturated_torque_axes=saturated_torque_axes.copy(),
                           torque_saturation_count=saturation_count)
    return StepReward(float(reward), components, saturation_count,
                      motor_saturation_count, saturation_penalty, diagnostics)
