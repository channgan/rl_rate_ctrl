"""Lazy simulator factories used by training and frozen evaluation."""
from copy import deepcopy

from stable_baselines3.common.monitor import Monitor

from ..backend import GazeboPX4Backend
from ..curriculum import RigToFlightCurriculum
from ..env import RateControlEnv
from ..privileged import PrivilegedCriticObservation
from ..trajectory import EpisodeTrajectory


def build_environment_factories(args, config, promotion_options, monitor_fields, trajectory_clock):
    def observation_wrapper(env):
        return env if args.simple_critic else PrivilegedCriticObservation(env)

    def training_factory(state=None, rng_state=None, session_index=0):
        backend = GazeboPX4Backend(args.runtime, args.run / "episodes", instance=args.base_instance)
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
        backend = GazeboPX4Backend(args.runtime, args.run / "evaluation_episodes", instance=args.base_instance)
        backend.set_mode('free_flight' if args.px4_native_outer else 'ball_rig')
        return observation_wrapper(RateControlEnv(backend, config))

    return training_factory, evaluation_factory
