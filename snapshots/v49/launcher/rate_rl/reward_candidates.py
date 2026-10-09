"""Consecutive angular-rate error improvement reward with episode-local history.

This module implements the requested consecutive-error SSE difference literally.
It does not rebase previous error onto a new reference: changing the reference
can therefore affect the score even without an improvement caused by the action.
The public native training recipe enables the approved weight 0.003. Historical
TaskConfig defaults remain zero; the reward contract distinguishes enablement.
"""
from dataclasses import dataclass

import numpy as np



def _sse(error_deg_s):
    error = np.asarray(error_deg_s, dtype=np.float64)
    if error.shape != (3,) or not np.all(np.isfinite(error)):
        raise ValueError('Expected three finite angular-rate errors in deg/s')
    with np.errstate(over='ignore'):
        value = float(np.sum(error ** 2))
    if not np.isfinite(value):
        raise ValueError('Angular-rate error SSE must be finite')
    return value


@dataclass(frozen=True)
class ErrorProgressResult:
    previous_sse: float | None
    current_sse: float
    cost_delta: float
    raw_reward: float
    ppo_reward: float
    unclipped_raw_reward: float
    max_abs_raw_reward: float | None


class ErrorProgressReward:
    """r_t = weight * (SSE_{t-1} - SSE_t), approved weight 0.003.

    Inputs are true, three-axis tracking errors in deg/s, using the same timing
    convention as the main reward. No mean, square root, dt factor or dt division
    is applied. An optional raw-unit limit clips both signs before the reward
    gain. Only ppo_reward includes the shared reward gain.

    Call reset at each episode boundary. Passing the reset-state error allows
    the first action to be scored; without it the first step only seeds history
    and earns zero. The terminal transition is scored normally, independently
    of the existing failure/success settlement. This is not an episode failure
    classifier and does not change the main SSE penalty.
    """

    def __init__(self, weight: float = .003, *, reward_gain: float = 7e-5):
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('Progress weight must be finite and non-negative')
        self.weight = float(weight)
        if not np.isfinite(reward_gain) or reward_gain <= 0:
            raise ValueError('Reward gain must be finite and positive')
        self.reward_gain = float(reward_gain)
        self._previous_sse = None

    def reset(self, error_deg_s=None):
        self._previous_sse = None if error_deg_s is None else _sse(error_deg_s)

    def step(self, error_deg_s, *, max_abs_raw_reward: float | None = None) -> ErrorProgressResult:
        if max_abs_raw_reward is not None and (
                not np.isfinite(max_abs_raw_reward) or max_abs_raw_reward < 0):
            raise ValueError('Progress reward limit must be finite and non-negative')
        current = _sse(error_deg_s)
        previous = self._previous_sse
        cost_delta = 0. if previous is None else current - previous
        raw = -self.weight * cost_delta
        if not np.isfinite(raw):
            raise ValueError('Progress reward must be finite')
        unclipped = raw
        if max_abs_raw_reward is not None:
            raw = float(np.clip(raw, -max_abs_raw_reward, max_abs_raw_reward))
        result = ErrorProgressResult(previous, current, cost_delta,
                                     raw, raw * self.reward_gain, unclipped, max_abs_raw_reward)
        self._previous_sse = current
        return result
