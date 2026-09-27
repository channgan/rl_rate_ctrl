"""Single source for the current public native-torque training recipe.

Historical TaskConfig/checkpoint defaults remain supported; this immutable recipe
is applied by the public launcher. Importing it has no simulator or process effects.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingDefaults:
    phase: str = "late"
    n_envs: int = 1
    rollout_steps: int = 4096
    batch_size: int = 2048
    n_epochs: int = 10
    episode_steps: int = 2048
    control_dt: float = .01
    base_instance: int = 41
    learning_rate: float = 3e-4
    learning_rate_min: float = 3e-5
    ent_coef: float = .01
    clip_range: float = .2
    torque_slew_scale: float = 10.
    slew_weight: float = .02 / 7.
    tracking_threshold_deg_s: float = 5.
    tracking_bonus_weight: float = .0084
    failure_rate_per_s: float = 3500.
    reward_gain: float = 7e-5
    success_bonus: float = 50000.
    failure_base: float = 30000.
    rate_error_cap: float = 5000.
    gyro_white_stddev_rad_s: float = .0017453292
    gyro_bias_walk_rad_s_per_sqrt_s: float = .0002


CURRENT = TrainingDefaults()
