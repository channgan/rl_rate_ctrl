from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .outer_loop import AltitudeAttitudeReference, OuterLoopConfig
from .action_contract import esc_to_simulator_speed
from .backend import SimulatorError
from .reward_contract import early_failure_rate, rate_error_cap


@dataclass(frozen=True)
class TaskConfig:
    dt: float = 0.001
    episode_seconds: float = 30.0
    command_hold_seconds: float = 0.5
    target_limit_rad_s: tuple = (1.0, 1.0, 1.0)
    air_target_limit_rad_s: tuple = (2.0, 2.0, 1.5)
    # Equivalent observation divisor: 286.4788975654116 deg/s. Preserve the
    # existing network input values; only the reward sensitivity changes.
    rate_scale_rad_s: float = 5.0
    reward_error_scale_deg_s: float = 10.0
    air_reward_error_scale_deg_s: float = 5.0
    thrust_variation: float = 0.05
    tracking_weight: float = 0.70
    slew_weight: float | None = None
    peak_pwm_weight: float = 0.10
    tracking_bonus_weight: float | None = None
    tracking_bonus_threshold_deg_s: float = 2.
    reward_slew_rate_scale_per_s: float | None = None
    promotion_rate_rmse_rad_s: float = 0.25
    promotion_settled_tolerance_rad_s: float = 0.2
    promotion_settled_fraction: float = 0.85
    promotion_settling_seconds: float = 0.15
    failure_cost: float = 20.0
    early_failure_cost: float = 40.0
    # Historical field name: native torque mode charges each saturated torque
    # axis; legacy direct-PWM mode charges each saturated motor.
    saturation_cost_per_motor_per_s: float = 1.0
    saturation_epsilon: float = 0.001
    max_rate_rad_s: float = 12.0
    max_tilt_deg: float = 75.0
    terminate_on_tilt: bool = True
    air_outer_loop: bool = True
    fixed_episode_rate_target: bool = False
    fixed_altitude_target_m: float | None = None
    waypoint_tracking: bool = False
    squared_error_reward: bool = False
    px4_native_outer: bool = False
    native_torque: bool = False

    def __post_init__(self):
        if self.slew_weight is None:
            object.__setattr__(self, 'slew_weight', .02 / 7. if self.native_torque else .1)
        if self.tracking_bonus_weight is None:
            # Raw 120 per second = ten times SSE at three errors of 2 deg/s.
            object.__setattr__(self, 'tracking_bonus_weight', .0084 if self.native_torque else .1)
        if self.reward_slew_rate_scale_per_s is None:
            # Native torque and historical PWM commands use different scales.
            object.__setattr__(self, 'reward_slew_rate_scale_per_s',
                               10.0 if self.native_torque else 50.0)


class RateControlEnv(gym.Env):
    """Rate-control task with an explicitly selected actuator interface.

    Native torque mode: 9D observation [omega/scale (3), rate error/scale (3),
    last accepted torque request (3)] and three normalized torque actions.
    PX4 supplies collective thrust and allocates motors. Legacy mode retains
    its 12D observation (including thrust and delta) and four direct PWM actions.
    Simulator truth is used for rewards, termination and the reference outer
    loop. It is never appended to the actor observation.
    """
    metadata = {"render_modes": []}

    def __init__(self, backend, config: TaskConfig | None = None):
        super().__init__()
        self.backend = backend
        self.config = config or TaskConfig()
        c = self.config
        if c.dt <= 0 or c.episode_seconds < c.dt or c.command_hold_seconds < c.dt:
            raise ValueError("Invalid task timing")
        if c.failure_cost <= 0 or c.rate_scale_rad_s <= 0:
            raise ValueError("Costs and rate scales must be positive")
        if not np.isfinite(c.early_failure_cost) or c.early_failure_cost < 0:
            raise ValueError("Early failure cost must be finite and non-negative")
        error_scales = [c.reward_error_scale_deg_s, c.air_reward_error_scale_deg_s]
        if not np.all(np.isfinite(error_scales)) or min(error_scales) <= 0:
            raise ValueError("Reward angular-rate error scale must be finite and positive in deg/s")
        if not np.isfinite(c.reward_slew_rate_scale_per_s) or c.reward_slew_rate_scale_per_s <= 0:
            raise ValueError("Reward normalisation scales must be positive")
        if not 0 <= c.saturation_epsilon < 0.5:
            raise ValueError("Invalid saturation threshold")
        if not np.isfinite(c.saturation_cost_per_motor_per_s) or c.saturation_cost_per_motor_per_s < 0:
            raise ValueError("Saturation cost must be finite and non-negative")
        weights = [c.tracking_weight, c.slew_weight,
                   c.peak_pwm_weight, c.tracking_bonus_weight]
        if not np.all(np.isfinite(weights)) or min(weights) < 0:
            raise ValueError("Reward weights must be finite and non-negative")
        for seconds in (c.episode_seconds, c.command_hold_seconds):
            if not np.isclose(round(seconds / c.dt) * c.dt, seconds):
                raise ValueError("Task durations must be integer multiples of dt")
        physical = backend.config
        if c.native_torque and not (
                c.px4_native_outer and c.waypoint_tracking and c.squared_error_reward
                and c.air_outer_loop and not c.fixed_episode_rate_target
                and physical.get('mode') == 'free_flight'):
            raise ValueError('Native torque requires native PX4 free-flight waypoints, '
                             'dynamic outer-loop references and squared-error rewards')
        self.gravity = float(physical.get("gravity_m_s2", 9.8))
        self.mass = float(physical["mass_kg"])
        self.max_rotor_speed = float(physical["max_rotor_rad_s"])
        self.thrust_coefficients = np.broadcast_to(np.asarray(
            physical["thrust_coefficient"], dtype=float), (4,)).copy()
        if not np.all(np.isfinite([self.gravity, self.mass, self.max_rotor_speed])) or min(
                self.gravity, self.mass, self.max_rotor_speed) <= 0 or not np.all(
                np.isfinite(self.thrust_coefficients) & (self.thrust_coefficients > 0)):
            raise ValueError("Invalid motor thrust or vehicle mass parameters")
        # Physical upper bound for the reference outer loop, not thrust feedback
        # or a PWM-to-thrust reward proxy.
        self.max_thrust_ratio = float(np.sum(self.thrust_coefficients)
            * self.max_rotor_speed ** 2 / (self.mass * self.gravity))
        self.action_space = (spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
                             if c.native_torque else
                             spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32))
        history_low = [-1.] * 3 if c.native_torque else [0.] * 4 + [0., -np.inf]
        history_high = [1.] * 3 if c.native_torque else [1.] * 4 + [np.inf, np.inf]
        self.observation_space = spaces.Box(
            np.array([-np.inf] * 6 + history_low, dtype=np.float32),
            np.array([np.inf] * 6 + history_high, dtype=np.float32), dtype=np.float32)
        self.target = np.zeros(3)
        # Motor history remains separate for PWM diagnostics and legacy costs;
        # native-mode actor history and action costs use accepted torque requests.
        self.last_action = np.zeros(4, dtype=np.float32)
        self.last_torque = np.zeros(3, dtype=np.float32)
        self.thrust = 0.0
        self.thrust_delta = 0.0
        self.steps = 0
        self.needs_reset = True
        self.current_state = None
        from .waypoint import WaypointReference
        reference_class = WaypointReference if c.waypoint_tracking else AltitudeAttitudeReference
        self.outer_loop = reference_class(OuterLoopConfig(gravity_m_s2=self.gravity,
            fixed_altitude_target_m=c.fixed_altitude_target_m))
        if c.px4_native_outer:
            if not c.waypoint_tracking or physical['mode'] != 'free_flight':
                raise ValueError('Native PX4 outer loop requires free-flight waypoints')
            from .native_outer import NativePX4Reference
            backend.native_outer = True
            self.outer_loop = NativePX4Reference(OuterLoopConfig(gravity_m_s2=self.gravity), backend)
        backend.native_torque = c.native_torque

    @staticmethod
    def _validate_state(state):
        for key, size in (("rates", 3), ("rates_true", 3), ("position_ned", 3),
                          ("linear_velocity_ned", 3), ("q_ned_frd", 4),
                          ("rotor_speed_fraction", 4), ("applied_pwm", 4)):
            try:
                values = np.asarray(state[key], dtype=float)
            except (KeyError, TypeError, ValueError) as exc:
                raise SimulatorError(f"Missing or invalid {key} in simulator state") from exc
            if values.shape != (size,) or not np.all(np.isfinite(values)):
                raise SimulatorError(f"Invalid {key} in simulator state")
        if np.linalg.norm(state["q_ned_frd"]) < .5:
            raise SimulatorError("Invalid attitude in simulator state")
        if np.any(np.asarray(state["rotor_speed_fraction"]) < 0):
            raise SimulatorError("Negative rotor speed magnitude")
        if np.any(np.asarray(state["applied_pwm"]) < 0) or np.any(np.asarray(state["applied_pwm"]) > 1):
            raise SimulatorError("Invalid applied PWM feedback")

    @property
    def uses_outer_loop(self):
        return self.config.air_outer_loop and getattr(self.backend, "config", {}).get("mode") == "free_flight"

    @property
    def reward_error_scale_deg_s(self):
        # Use the physical stage, not the promotion-ready flag. The final rig
        # transition retains 10 deg/s; the next air episode starts at 5 deg/s.
        return (self.config.air_reward_error_scale_deg_s
                if self.backend.config["mode"] == "free_flight"
                else self.config.reward_error_scale_deg_s)

    def _air_command(self, state):
        try:
            target, self.thrust = self.outer_loop.command(
                state, self.max_thrust_ratio, self.config.air_target_limit_rad_s)
        except BaseException:
            self.needs_reset = True
            self.current_state = None
            raise
        if not self.config.fixed_episode_rate_target:
            self.target = target

    def _sample_command(self):
        limit = np.array(self.config.target_limit_rad_s)
        self.target = self.np_random.uniform(-limit, limit)
        # Hover is one vehicle weight; the target is not a PWM or F/Fmax fraction.
        self.thrust = float(self.np_random.uniform(
            1 - self.config.thrust_variation, 1 + self.config.thrust_variation))
        self.thrust = float(np.clip(self.thrust, 0, self.max_thrust_ratio))

    def _obs(self, state):
        rates = np.asarray(state["rates"])
        scale = self.config.rate_scale_rad_s
        if self.config.native_torque:
            return np.concatenate((rates / scale, (self.target - rates) / scale,
                                   self.last_torque)).astype(np.float32)
        return np.concatenate((rates / scale, (self.target - rates) / scale,
                               self.last_action, [self.thrust, self.thrust_delta])).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.needs_reset = True
        self.current_state = None
        simulator_seed = int(self.np_random.integers(0, 2**31 - 1))
        state = self.backend.reset(seed=simulator_seed)
        self._validate_state(state)
        self.steps = 0
        self.last_action = np.asarray(state["applied_pwm"], dtype=np.float32).copy()
        # No actor action has been accepted in this episode yet. Native
        # takeoff controller output is not a policy action-history sample.
        self.last_torque = np.zeros(3, dtype=np.float32)
        self.last_sim_us = state["sim_us"]
        self.episode_error_squared = 0.0
        self.episode_axis_error_squared = np.zeros(3)
        self.episode_settled_steps = 0
        self.episode_settled_good_steps = 0
        self.episode_reward_components = np.zeros(4)
        self.episode_observed_error_squared = 0.0
        self.episode_saturation_penalty = 0.0
        self.episode_failure_penalty = 0.0
        self.episode_success_bonus = 0.0
        self.episode_waypoint_bonus = 0.0
        self.episode_saturated_action_steps = 0
        self.episode_saturated_motor_steps = 0
        if self.uses_outer_loop:
            self.outer_loop.reset(state, self.np_random)
            self._air_command(state)
        else:
            self._sample_command()
        if self.config.fixed_episode_rate_target:
            limit = np.array(self.config.target_limit_rad_s)
            self.target = self.np_random.uniform(-limit, limit)
        self.thrust_delta = 0.0
        self.needs_reset = False
        self.current_state = state
        return self._obs(state), {"sim_us": state["sim_us"], "simulator_seed": simulator_seed,
                                 "target_rad_s": self.target.copy(),
                                 "thrust_demand": self.thrust,
                                 "reward_error_scale_deg_s": self.reward_error_scale_deg_s,
                                 "training_stage": self.backend.config["mode"]}

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("reset() is required before step()")
        action = np.asarray(action, dtype=np.float32)
        native_torque = self.config.native_torque
        expected_shape = (3,) if native_torque else (4,)
        minimum = -1.0 if native_torque else 0.0
        if (action.shape != expected_shape or not np.all(np.isfinite(action))
                or np.any(action < minimum) or np.any(action > 1)):
            raise ValueError('Expected three finite normalized torques in [-1,1]'
                             if native_torque else
                             'Expected four finite normalised ESC commands in [0,1]')
        try:
            if native_torque:
                state = self.backend.step_torque(action, self.config.dt)
            else:
                state = self.backend.step(esc_to_simulator_speed(action), self.config.dt)
            self._validate_state(state)
            if state["sim_us"] != self.last_sim_us + round(self.config.dt * 1e6):
                raise SimulatorError("Environment state did not advance by exactly dt")
            if native_torque:
                try:
                    accepted = np.asarray(state.get('applied_torque', []), dtype=float)
                except (TypeError, ValueError) as exc:
                    raise SimulatorError('Missing or invalid applied torque feedback') from exc
                if (accepted.shape != (3,) or not np.all(np.isfinite(accepted))
                        or not np.allclose(accepted, action, atol=1e-7, rtol=0)):
                    raise SimulatorError('Applied torque differs from the actor action')
            elif not np.allclose(state["applied_pwm"], action, atol=1e-7, rtol=0):
                raise SimulatorError("Applied command differs from the actor action")
            # Allocation can redistribute and clip motor commands. These four
            # actual outputs define motor history and motor diagnostics in both
            # modes. Native-mode slew and limit costs instead penalise accepted
            # torque requests, independently of collective motor commands.
            pwm_action = np.asarray(state['applied_pwm'], dtype=np.float32).copy()
            speed_action = np.asarray(state.get('applied_speed_fraction',
                                               esc_to_simulator_speed(pwm_action)), dtype=float).copy()
        except BaseException:
            self.needs_reset = True
            self.current_state = None
            raise
        c = self.config
        old_thrust = self.thrust
        error = self.target - np.asarray(state["rates_true"])
        observed_error = self.target - np.asarray(state["rates"])
        # Simulator/outer-loop states use rad/s. Express the endpoint error
        # explicitly in deg/s before applying the user-facing reward scale.
        error_deg_s = np.rad2deg(error)
        error_scale_deg_s = self.reward_error_scale_deg_s
        axis_cost = (error_deg_s / error_scale_deg_s) ** 2
        tracking = float(np.mean(axis_cost))
        commanded_throttle = float(np.mean(pwm_action.astype(float)))
        pwm_rate = (pwm_action.astype(float) - self.last_action.astype(float)) / c.dt
        if native_torque:
            torque_rate = (accepted - self.last_torque.astype(float)) / c.dt
            penalized_rate = torque_rate
            slew_rate_source = 'accepted_normalized_torque'
        else:
            penalized_rate = pwm_rate
            slew_rate_source = 'applied_pwm'
        slew_rate = float(np.mean(penalized_rate ** 2))
        # Bounded costs share a common [0,1) range. Weights define component
        # budgets, not the realised share of a trajectory's total reward.
        rate_normalised = float(np.mean(axis_cost / (1 + axis_cost)))
        if c.squared_error_reward:
            # Sum all three squared errors, then apply the unchanged shared cap.
            tracking = min(float(np.sum(error_deg_s ** 2)), rate_error_cap(c))
            rate_normalised = tracking / 10000.
        slew_scaled = (penalized_rate / c.reward_slew_rate_scale_per_s) ** 2
        slew_normalised = float(np.mean(slew_scaled / (1 + slew_scaled)))
        components = c.dt * np.array([-c.tracking_weight * rate_normalised,
            -c.slew_weight * slew_normalised, 0.0, 0.0])
        reward = float(components.sum())
        # Keep actual four-motor saturation for diagnostics. In native torque
        # mode only the actor's signed torque limits affect its cost and bonus;
        # zero torque is a valid request and is never a saturation boundary.
        saturated_motors = ((pwm_action <= c.saturation_epsilon)
                            | (pwm_action >= 1 - c.saturation_epsilon))
        motor_saturation_count = int(np.count_nonzero(saturated_motors))
        if native_torque:
            saturated_torque_axes = np.abs(accepted) >= 1 - c.saturation_epsilon
            saturation_count = int(np.count_nonzero(saturated_torque_axes))
            saturation_source = 'accepted_normalized_torque'
        else:
            saturation_count = motor_saturation_count
            saturation_source = 'allocated_motor'
        # Each axis/motor is charged independently on every executed step,
        # including the terminal step, with the existing coefficient and dt.
        saturation_penalty = c.saturation_cost_per_motor_per_s * c.dt * saturation_count
        reward -= saturation_penalty
        q = np.asarray(state["q_ned_frd"], dtype=float)
        q /= np.linalg.norm(q)
        cos_tilt = 1 - 2 * (q[1]**2 + q[2]**2)
        tilt = np.arccos(np.clip(cos_tilt, -1, 1))
        altitude = -float(state["position_ned"][2])
        failure = None
        on_rig = getattr(self.backend, "config", {}).get("mode") == "ball_rig"
        if not on_rig and altitude < 0.3: failure = "ground"
        elif not on_rig and c.terminate_on_tilt and tilt > np.deg2rad(c.max_tilt_deg): failure = "tilt"
        elif np.max(np.abs(state["rates_true"])) > c.max_rate_rad_s: failure = "rate"
        elif c.px4_native_outer and not state.get('px4_position_control'):
            failure = 'px4_control_lost'
        horizon_reached = self.steps + 1 >= round(c.episode_seconds / c.dt)
        horizon_rmse_deg_s = float(np.rad2deg(np.sqrt(
            (self.episode_error_squared + float(np.mean(error ** 2))) / (self.steps + 1))))
        horizon_axis_rmse = np.rad2deg(np.sqrt(
            (self.episode_axis_error_squared + error ** 2) / (self.steps + 1)))
        tracking_failed = (bool(np.any((horizon_axis_rmse >= 5.) | np.isclose(
            horizon_axis_rmse, 5., rtol=0., atol=1e-12))) if native_torque
            else horizon_rmse_deg_s >= 5.)
        if failure is None and horizon_reached and tracking_failed:
            failure = "tracking_rmse"
        # Penalise the peak command to favour upward PWM headroom. This is
        # always a cost and is not a collective-thrust tracking objective.
        peak_pwm = float(np.max(pwm_action))
        upper_headroom = 1 - peak_pwm
        peak_pwm_penalty = c.dt * c.peak_pwm_weight * peak_pwm
        if c.squared_error_reward:
            upper_headroom = 1. - float(np.mean(pwm_action))
            tracking_now = failure is None and np.all(np.abs(error_deg_s) < 5.)
            peak_pwm_penalty = (-2. * c.dt * c.peak_pwm_weight * upper_headroom
                                if tracking_now else 0.)
        if native_torque:
            # Collective thrust belongs to PX4. Rewarding lower collective
            # motor output would compete with the native altitude controller.
            peak_pwm_penalty = 0.
        reward -= peak_pwm_penalty
        components[2] = -peak_pwm_penalty
        # Sustained tracking earns this each qualifying step. There is no
        # unconditional survival reward or additional dwell-time requirement.
        tracking_eligible = bool(
            failure is None and saturation_count == 0
            and (np.max(np.abs(error_deg_s)) < (c.tracking_bonus_threshold_deg_s if native_torque else 5.) if c.squared_error_reward
                 else np.max(np.abs(error_deg_s)) <= error_scale_deg_s))
        tracking_bonus = c.dt * c.tracking_bonus_weight if tracking_eligible else 0.0
        reward += tracking_bonus
        components[3] = tracking_bonus
        self.episode_reward_components += components
        self.episode_axis_error_squared += error ** 2
        command_hold = self.outer_loop.config.attitude_hold_seconds if self.uses_outer_loop else c.command_hold_seconds
        command_step = self.steps % round(command_hold / c.dt)
        if command_step * c.dt >= c.promotion_settling_seconds:
            self.episode_settled_steps += 1
            self.episode_settled_good_steps += int(np.max(np.abs(error)) <= c.promotion_settled_tolerance_rad_s)
        self.steps += 1
        self.episode_error_squared += float(np.mean(error**2))
        self.episode_observed_error_squared += float(np.mean(observed_error**2))
        self.episode_saturated_action_steps += saturation_count
        self.episode_saturated_motor_steps += motor_saturation_count
        success = failure is None and self.steps >= round(c.episode_seconds / c.dt)
        terminated = failure is not None or success
        truncated = False  # Surviving the finite task horizon is an explicit success.
        success_bonus = .5 * c.failure_cost if success else 0.0
        survival_fraction = np.clip(self.steps * c.dt / c.episode_seconds, 0., 1.)
        failure_penalty = (c.failure_cost + c.early_failure_cost * (1. - survival_fraction)
                           if failure is not None else 0.0)
        if c.squared_error_reward:
            # Approved raw settlements, with the same global gain as every step reward.
            success_bonus = 50000. * 7e-5 if success else 0.
            failure_penalty = ((30000. + early_failure_rate(c) * c.episode_seconds * (1. - survival_fraction)) * 7e-5
                               if failure is not None else 0.)
        reward += success_bonus
        reward -= failure_penalty
        self.episode_saturation_penalty += saturation_penalty
        self.episode_failure_penalty += failure_penalty
        self.episode_success_bonus += success_bonus
        info = dict(sim_us=state["sim_us"], sample_us=state["sample_us"],
                    target_rad_s=self.target.copy(), rates_rad_s=np.array(state["rates"]),
                    rates_true_rad_s=np.array(state["rates_true"]), observed_error_rad_s=observed_error,
                    error_rad_s=error, error_deg_s=error_deg_s,
                    observed_error_deg_s=np.rad2deg(observed_error),
                    altitude_m=altitude, tilt_rad=float(tilt),
                    position_ned=np.array(state["position_ned"]),
                    tracking_cost=tracking, rate_bounded_cost=rate_normalised,
                    reward_error_scale_deg_s=error_scale_deg_s,
                    slew_rate_cost=slew_rate,
                    slew_rate_source=slew_rate_source,
                    pwm_rate_per_s=pwm_rate.copy(),
                    continuous_reward_components=components.copy(),
                    thrust_demand=self.thrust,
                    commanded_throttle=commanded_throttle, current_pwm_command=pwm_action.copy(),
                    applied_speed_fraction=speed_action.copy(),
                    upper_headroom=upper_headroom, peak_pwm_cost=peak_pwm,
                    peak_pwm_penalty=peak_pwm_penalty, tracking_eligible=tracking_eligible,
                    tracking_bonus=tracking_bonus,
                    rotor_speed_fraction=np.array(state["rotor_speed_fraction"]), failure=failure,
                    is_success=success, success_bonus=success_bonus,
                    saturated_motors=saturated_motors.copy(),
                    motor_saturation_count=motor_saturation_count,
                    saturation_count=saturation_count, saturation_source=saturation_source,
                    saturation_penalty=saturation_penalty, failure_penalty=failure_penalty)
        if native_torque:
            info['torque_command'] = action.copy()
            info['torque_rate_per_s'] = torque_rate.copy()
            info['saturated_torque_axes'] = saturated_torque_axes.copy()
            info['torque_saturation_count'] = saturation_count
        for key in ("source_sample_us", "truth_us", "action_seq", "snapshot_valid"):
            if key in state:
                info[key] = state[key]
        if "gyro_noise_flu" in state:
            info["gyro_noise_flu"] = state["gyro_noise_flu"]
        info["training_stage"] = self.backend.config["mode"]
        if terminated or truncated:
            axis_rmse = np.sqrt(self.episode_axis_error_squared / self.steps)
            settled_fraction = self.episode_settled_good_steps / max(1, self.episode_settled_steps)
            info.update(episode_axis_rate_rmse_rad_s=axis_rmse,
                        episode_axis_rate_rmse_deg_s=np.rad2deg(axis_rmse),
                        episode_settled_tracking_fraction=settled_fraction,
                        episode_reward_components=self.episode_reward_components.copy(),
                        transfer_qualified=bool(success
                            and np.all(axis_rmse <= c.promotion_rate_rmse_rad_s)
                            and self.episode_settled_steps > 0
                            and settled_fraction >= c.promotion_settled_fraction))
            info.update(episode_rate_rmse_rad_s=float(np.sqrt(self.episode_error_squared / self.steps)),
                        episode_rate_rmse_deg_s=float(np.rad2deg(np.sqrt(self.episode_error_squared / self.steps))),
                        episode_observed_rate_rmse_rad_s=float(np.sqrt(self.episode_observed_error_squared / self.steps)),
                        episode_observed_rate_rmse_deg_s=float(np.rad2deg(np.sqrt(self.episode_observed_error_squared / self.steps))),
                        episode_saturation_penalty=self.episode_saturation_penalty,
                        episode_failure_penalty=self.episode_failure_penalty,
                        episode_success_bonus=self.episode_success_bonus,
                        episode_saturation_fraction=(self.episode_saturated_action_steps
                            / ((3 if native_torque else 4) * self.steps)),
                        episode_motor_saturation_fraction=(self.episode_saturated_motor_steps
                            / (4 * self.steps)))
            if native_torque:
                info['episode_torque_saturation_fraction'] = info['episode_saturation_fraction']
        if self.uses_outer_loop:
            info.update(self.outer_loop.diagnostics())
            # These diagnostics describe the waypoint/reference used by this action.
            info['outer_diagnostics_timing'] = 'reference used for this transition, before next command'
        self.needs_reset = terminated or truncated
        info['waypoint_switch_bonus'] = 0.0
        # Reward belongs to the command in the preceding observation. The next
        # observation includes the new command when a command boundary is crossed.
        if not self.needs_reset:
            if self.uses_outer_loop:
                if not c.waypoint_tracking and not c.fixed_episode_rate_target and self.steps % round(self.outer_loop.config.attitude_hold_seconds / c.dt) == 0:
                    self.outer_loop.sample(self.np_random)
                reached_before = self.outer_loop.waypoints_reached if c.waypoint_tracking else 0
                self._air_command(state)
                if c.waypoint_tracking:
                    if (c.squared_error_reward and not native_torque
                            and self.outer_loop.waypoints_reached > reached_before):
                        bonus = 1000. * 7e-5
                        reward += bonus
                        self.episode_waypoint_bonus += bonus
                        info['waypoint_switch_bonus'] = bonus
                    info['waypoint_gate_transition'] = dict(self.outer_loop.gate_transition)
                    info['next_waypoints_reached'] = self.outer_loop.waypoints_reached
                    info['next_position_target_ned'] = self.outer_loop.position_target_ned.copy()
            elif not c.fixed_episode_rate_target and self.steps % round(c.command_hold_seconds / c.dt) == 0:
                self._sample_command()
        self.thrust_delta = self.thrust - old_thrust
        info['episode_waypoint_bonus'] = self.episode_waypoint_bonus
        # Commit histories only once the full transition (including the next
        # native reference) is valid. A transport/reference fault must not
        # expose an unreturned transition as an actor-history update.
        self.last_action = pwm_action.copy()
        if native_torque:
            self.last_torque = accepted.astype(np.float32).copy()
        self.last_sim_us = state["sim_us"]
        self.current_state = state
        return self._obs(state), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.backend.close()
        self.needs_reset = True
        self.current_state = None
