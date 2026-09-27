"""Reviewable stage settings, adapted from SB3 and RL Zoo reference ranges."""
from dataclasses import asdict, dataclass
from copy import deepcopy

from stable_baselines3.common.utils import FloatSchedule


@dataclass(frozen=True)
class PPOStageSettings:
    batch_size: int
    learning_rate: float
    ent_coef: float
    clip_range: float


EARLY = PPOStageSettings(2048, 3e-4, .01, .2)
AIR = PPOStageSettings(2048, 3e-4, .01, .2)


def validate_phase_migration(old, new, pending_evaluation, allow_tracking_tuning=False):
    """Allow reward scales only; retain all physical, observation and reward-weight checks."""
    expected = deepcopy(old)
    changes = {}
    if allow_tracking_tuning:
        for key in ('tracking_bonus_threshold_deg_s', 'tracking_bonus_weight', 'slew_weight'):
            changes[key] = dict(old=old['task'].get(key, 5.), new=new['task'][key])
            expected['task'][key] = new['task'][key]
    for key in ("reward_error_scale_deg_s", "air_reward_error_scale_deg_s",
                "reward_slew_rate_scale_per_s"):
        changes[key] = dict(old=old["task"][key], new=new["task"][key])
        expected["task"][key] = new["task"][key]
    if expected != new or pending_evaluation is not None:
        raise ValueError("Phase migration permits only reward scales and no pending evaluation")
    return changes


def apply_stage_settings(model, free_flight):
    settings = AIR if free_flight else EARLY
    model.batch_size = settings.batch_size
    if hasattr(model, "set_kl_phase"):
        # A phase change narrows the LR bounds; resuming the same phase must
        # preserve its adapted LR and consecutive-low-KL counter.
        model.set_kl_phase("late" if free_flight else "early")
    else:
        model.learning_rate = settings.learning_rate
        model.lr_schedule = FloatSchedule(settings.learning_rate)
    model.ent_coef = settings.ent_coef
    model.clip_range = FloatSchedule(settings.clip_range)
    model.active_stage_settings = asdict(settings)
    model.active_stage_settings["learning_rate"] = float(model.learning_rate)
    return settings
