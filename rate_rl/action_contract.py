"""Version the physical meaning of policy inputs/outputs, not just their shapes."""
import math

ACTION_CONTRACT = "normalised_pwm_fdes_over_mg_delta_12d_v2"
CRITIC_CONTRACT = "privileged_critic_35d_v1"
REWARD_CONTRACT = "rate_pwm_tracking_bonus_deg10_5_sat_dt_early_failure40_v7"
OBSERVATION_SIZE = 12
ACTION_SIZE = 4
RATE_OBSERVATION_SCALE_RAD_S = 5.0
RATE_OBSERVATION_SCALE_DEG_S = math.degrees(RATE_OBSERVATION_SCALE_RAD_S)
OBSERVATION_DESCRIPTION = (
    "omega_frd_deg_s/(5*180/pi)[3], (rate_target_deg_s-omega_frd_deg_s)/(5*180/pi)[3] "
    "(equivalent to the unchanged source rad/s divided by 5), "
    "previous issued normalised PWM[4], Fdes/(mass*gravity)[1], "
    "current minus previous Fdes/(mass*gravity)[1]"
)


def interface_metadata():
    """Only actor-visible fields; these values are also the firmware interface."""
    return dict(
        version=ACTION_CONTRACT, observation_size=OBSERVATION_SIZE,
        action_size=ACTION_SIZE, dtype="float32", observation=OBSERVATION_DESCRIPTION,
        rate_frame="FRD", rate_units="deg/s", rate_divisor=RATE_OBSERVATION_SCALE_DEG_S,
        source_rate_units="rad/s", source_rate_divisor=RATE_OBSERVATION_SCALE_RAD_S,
        rate_preprocessing="deg/s divided by (5*180/pi) is equivalent to source rad/s divided by 5; runtime numbers are unchanged",
        previous_pwm_range=[0.0, 1.0], thrust_reference="Fdes/(mass_kg*gravity_m_s2)",
        thrust_reference_hover=1.0, thrust_reference_upper_bound=None,
        thrust_reference_delta="current minus preceding decision; not divided by dt",
        thrust_reference_delta_at_reset=0.0,
        action="4 normalised PWM/ESC commands clipped to [0,1], motor indices 0,1,2,3",
        history="last command actually issued, after action clipping",
        actor_requires_privileged_inputs=False,
    )


def critic_interface_metadata():
    """Training-only observation order; none of these extras enter actor export."""
    return dict(version=CRITIC_CONTRACT, observation_size=35, dtype="float32",
        privileged=True, deployed=False, independent_network=True,
        fields=[
            dict(indices="0:12", meaning="actor observation, identical to the current actor12"),
            dict(indices="12:15", meaning="true FRD angular rate", units="deg/s",
                 divisor=RATE_OBSERVATION_SCALE_DEG_S,
                 source_units="rad/s", source_divisor=RATE_OBSERVATION_SCALE_RAD_S,
                 numeric_preprocessing="unchanged source rad/s divided by 5, equivalent to deg/s divided by (5*180/pi)"),
            dict(indices="15:19", meaning="actual rotor angular speed / configured max rotor speed"),
            dict(indices="19:23", meaning="true FRD-to-NED unit quaternion wxyz, canonical sign"),
            dict(indices="23:26", meaning="true NED linear velocity", units="m/s", divisor=5.0),
            dict(indices="26", meaning="height above world zero", units="m", divisor=3.0),
            dict(indices="27", meaning="target height minus current height", units="m", divisor=3.0),
            dict(indices="28:31", meaning="shortest body-frame rotation vector from current to target attitude",
                 units="rad", divisor="pi"),
            dict(indices="31", meaning="outer loop low-pass filtered upward velocity", units="m/s", divisor=5.0),
            dict(indices="32", meaning="episode remaining seconds / episode duration"),
            dict(indices="33", meaning="command remaining hold seconds / stage command hold duration"),
            dict(indices="34", meaning="free flight stage flag (0=ball rig, 1=free flight)"),
        ], ball_rig_zero_indices=[27, 28, 29, 30, 31],
        terminal_zero_indices=[32, 33],
        timing="same decision time as actor12, after reference update; no future random targets")


def reward_interface_metadata():
    """Version the training objective independently of deployable actor inputs."""
    return dict(version=REWARD_CONTRACT,
        continuous_components=["rate", "slew_rate", "peak_pwm", "tracking_bonus"],
        continuous_weights=dict(rate=.70, slew_rate=.10, peak_pwm=.10, tracking_bonus=.10),
        continuous_integration="multiply each weighted component by dt in seconds",
        continuous_bounded=True,
        continuous_bounds=dict(normalised_cost="B(x)=x^2/(1+x^2), mathematically in [0,1) for finite x; no hard cost clipping",
            peak_pwm_cost="max(current applied PWM), in [0,1]", tracking_score="0 or 1",
            weighted_per_step_in_dt_units=dict(rate=[-.70, 0.], slew_rate=[-.10, 0.],
                                               peak_pwm=[-.10, 0.], tracking_bonus=[0., .10]),
            note="Conservative component bounds; multiply by dt. They do not clip the summed reward."),
        aggregate_reward_clipping=False,
        rate=dict(feedback="Gazebo true FRD angular rate after action execution, sourced in rad/s",
                  reference="rate target from the observation that caused this action",
                  conversion="error_deg_s = (target_rad_s - true_rate_rad_s) * 180 / pi",
                  units="deg/s", stage_error_scale_deg_s=dict(ball_rig=10.0, free_flight=5.0),
                  stage_selection="actual environment stage, independent of whether the outer loop is enabled",
                  stage_transition="next reset switches stages; the final ball_rig step still uses 10 deg/s",
                  normalised_error="x_i = error_deg_s_i / stage_error_scale_deg_s[actual_stage]",
                  cost="mean_i(x_i^2/(1+x_i^2))"),
        slew_rate=dict(input="(current applied PWM - previous applied PWM) / dt",
                       units="s^-1", scale_per_s=50., cost="mean_i(x_i^2/(1+x_i^2))"),
        peak_pwm=dict(cost="max(current applied PWM)", units="dimensionless",
                      contribution="-0.10 * dt * peak_pwm_cost; non-positive on every step, without an eligibility gate"),
        tracking_bonus=dict(score="1 if no failure, no saturated motor, and all three absolute true rate errors in deg/s <= the current stage error scale; otherwise 0",
                            tolerance_deg_s=dict(ball_rig=10.0, free_flight=5.0),
                            contribution="+0.10 * dt on each eligible step; otherwise 0",
                            accumulation="every eligible step contributes immediately; no minimum dwell time or 150 ms delay",
                            timing="same post-action true error and pre-action target as the rate cost"),
        positive_reward_sources=["eligible tracking steps", "30-second survival success"],
        survival_only_step_bonus=False,
        thrust_tracking_reward=False, thrust_reference_still_in_actor=True,
        saturation_cost_per_motor_per_s=1.0,
        saturation_formula="-1.0 * dt * saturated_motor_count",
        discrete_events=dict(saturated_motor_step=-.001, failure=-20., survival_success=10.,
                             failure_base=-20., early_failure_max=-40.,
                             failure_range=[-60., -20.],
                             failure_formula="-20 - 40 * (1 - clip(completed_episode_steps * dt / episode_seconds, 0, 1)); once on failure only",
                             reference_dt_s=.001, saturation_multiplied_by_dt=True,
                             saturation_per_motor_per_step=[0., -.001],
                             saturation_total_per_step=[0., -.001, -.002, -.003, -.004],
                             multiplied_by_dt=False,
                             saturation="each motor independently at PWM <= 0.001 or >= 0.999"),
        compatibility="old reward checkpoints cannot resume silently; actor12 export remains compatible")


def actor_observation_space(model):
    from gymnasium import spaces
    observation = getattr(model, "observation_space", None)
    return observation.spaces.get("actor") if isinstance(observation, spaces.Dict) else observation


def require_action_contract(model):
    import numpy as np
    from gymnasium import spaces
    if getattr(model, "action_contract", None) != ACTION_CONTRACT:
        raise ValueError("Checkpoint does not declare the 12D PWM + Fdes/(mg) + delta contract; "
                         "old 11D, F/Fmax, speed or throttle-reference policies cannot be resumed or exported")
    actor_space = actor_observation_space(model)
    if getattr(actor_space, "shape", None) != (OBSERVATION_SIZE,) or getattr(
            getattr(model, "action_space", None), "shape", None) != (ACTION_SIZE,):
        raise ValueError("Checkpoint spaces do not match the declared 12D observation / 4D PWM contract")
    action_space = model.action_space
    if not isinstance(actor_space, spaces.Box) or not isinstance(action_space, spaces.Box) or (
            actor_space.dtype != np.float32 or action_space.dtype != np.float32) or not (
            np.all(action_space.low == 0) and np.all(action_space.high == 1)):
        raise ValueError("Checkpoint spaces must use float32 observations and [0,1] PWM actions")


def require_training_contract(model):
    """PPO resume requires matching value inputs and reward, unlike actor export."""
    import numpy as np
    from gymnasium import spaces
    from .asymmetric_policy import AsymmetricActorCriticPolicy
    require_action_contract(model)
    if getattr(model, "critic_contract", None) == "ordinary_actor12_critic12_64x64_v1":
        observation = model.observation_space
        if not isinstance(observation, spaces.Box) or observation.shape != (12,):
            raise ValueError("Ordinary critic requires the same 12D observation as actor")
        if model.policy.net_arch != dict(pi=[64, 64], vf=[64, 64]):
            raise ValueError("Ordinary policy requires 64x64 actor and critic")
        if getattr(model, "reward_contract", None) != REWARD_CONTRACT:
            raise ValueError("Ordinary checkpoint reward contract mismatch")
        return
    if getattr(model, "critic_contract", None) != CRITIC_CONTRACT:
        raise ValueError("Checkpoint lacks the 35D privileged critic contract; ordinary critics cannot resume this training")
    observation = model.observation_space
    if not isinstance(observation, spaces.Dict) or set(observation.spaces) != {"actor", "critic"} or not (
            isinstance(observation["critic"], spaces.Box) and observation["critic"].shape == (35,)
            and observation["critic"].dtype == np.float32):
        raise ValueError("Training checkpoint requires Dict(actor12, critic35) float32 spaces")
    if not isinstance(model.policy, AsymmetricActorCriticPolicy):
        raise ValueError("Training checkpoint must use the isolated asymmetric policy")
    if getattr(model, "reward_contract", None) != REWARD_CONTRACT:
        raise ValueError("Checkpoint lacks the current stage-dependent 10/5 deg/s rate / PWM slew-rate / peak-PWM / tracking-bonus reward contract; "
                         "old reward models cannot resume this training")


def esc_to_simulator_speed(command):
    """Independent channel adapter, using an ideal linear speed response.

    A real ESC needs a calibrated command/speed/thrust model. Motor dynamics
    remain in Gazebo. This function never redistributes commands across motors.
    """
    import numpy as np
    return np.asarray(command, dtype=float).copy()
