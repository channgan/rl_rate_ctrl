"""Training-only, same-decision critic observations; the actor remains 12D."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .outer_loop import multiply


CRITIC_SIZE = 35
VELOCITY_SCALE_M_S = 5.0
HEIGHT_SCALE_M = 3.0


def canonical_quaternion(quaternion):
    """Normalise wxyz and give q and -q the same representation."""
    q = np.asarray(quaternion, dtype=np.float64).copy()
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < .5:
        raise ValueError("Expected a finite attitude quaternion")
    q /= np.linalg.norm(q)
    # At exactly pi, use the first nonzero axis to resolve the sign tie.
    first = next((x for x in q if abs(x) > 1e-12), 1.)
    if first < 0:
        q = -q
    return q


def attitude_error_rotvec(current, target):
    """Shortest target rotation in the current FRD frame, in radians."""
    q = canonical_quaternion(current)
    target = canonical_quaternion(target)
    error = canonical_quaternion(multiply(q * [1, -1, -1, -1], target))
    vector_norm = np.linalg.norm(error[1:])
    if vector_norm < 1e-12:
        return 2.0 * error[1:]
    angle = 2.0 * np.arctan2(vector_norm, max(float(error[0]), 0.))
    return error[1:] * (angle / vector_norm)


def critic_observation(env, actor_observation):
    """Read only the endpoint used for the actor observation, never a new sample.

    Call after reset/step has refreshed the next reference and its internal
    outer-loop state. Terminal endpoints retain the final reference; both
    clocks are zero because that episode has ended.
    """
    actor = np.asarray(actor_observation, dtype=np.float32)
    if actor.shape != (12,) or not np.all(np.isfinite(actor)):
        raise ValueError("Expected the current finite 12D actor observation")
    state = env.current_state
    if state is None:
        raise RuntimeError("reset() is required before reading critic observations")
    if state["sim_us"] != env.last_sim_us:
        raise RuntimeError("Critic state does not match the actor endpoint")
    result = np.zeros(CRITIC_SIZE, dtype=np.float32)
    result[:12] = actor
    result[12:15] = np.asarray(state["rates_true"]) / env.config.rate_scale_rad_s
    result[15:19] = state["rotor_speed_fraction"]
    result[19:23] = canonical_quaternion(state["q_ned_frd"])
    result[23:26] = np.asarray(state["linear_velocity_ned"]) / VELOCITY_SCALE_M_S
    altitude = -float(state["position_ned"][2])
    result[26] = altitude / HEIGHT_SCALE_M
    if env.uses_outer_loop:
        result[27] = (env.outer_loop.altitude_target - altitude) / HEIGHT_SCALE_M
        result[28:31] = attitude_error_rotvec(state["q_ned_frd"], env.outer_loop.attitude_target) / np.pi
        result[31] = env.outer_loop.velocity_up / VELOCITY_SCALE_M_S
    if not env.needs_reset:
        episode_steps = round(env.config.episode_seconds / env.config.dt)
        result[32] = np.clip((episode_steps - env.steps) / episode_steps, 0., 1.)
        hold = env.outer_loop.config.attitude_hold_seconds if env.uses_outer_loop else env.config.command_hold_seconds
        hold_steps = round(hold / env.config.dt)
        result[33] = (hold_steps - env.steps % hold_steps) / hold_steps
    result[34] = float(env.backend.config["mode"] == "free_flight")
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite critic observation")
    return result


class PrivilegedCriticObservation(gym.ObservationWrapper):
    """Use outside the curriculum wrapper and inside Monitor/DummyVecEnv.

    The dictionary is consumed by the asymmetric policy. Its actor branch
    reads only 'actor'; no privileged values are needed by the exported actor.
    """
    def __init__(self, env):
        super().__init__(env)
        if env.observation_space.shape != (12,):
            raise ValueError("Privileged wrapper requires the reviewed 12D actor environment")
        low = np.full(CRITIC_SIZE, -np.inf, dtype=np.float32)
        high = np.full(CRITIC_SIZE, np.inf, dtype=np.float32)
        low[:12], high[:12] = env.observation_space.low, env.observation_space.high
        low[15:19] = 0.
        low[19:23], high[19:23] = -1., 1.
        low[28:31], high[28:31] = -1., 1.
        low[32:35], high[32:35] = 0., 1.
        self.observation_space = spaces.Dict({
            "actor": env.observation_space,
            "critic": spaces.Box(low, high, dtype=np.float32),
        })

    def observation(self, observation):
        actor = np.asarray(observation, dtype=np.float32).copy()
        return {"actor": actor, "critic": critic_observation(self.env.unwrapped, actor)}
