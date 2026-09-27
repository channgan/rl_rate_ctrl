"""SB3 PPO with a persisted, rollout-level Gaussian KL learning-rate controller.

The PPO objective and optimizer steps follow Stable-Baselines3 2.9.0.  The
controller measures KL(old || new) on the *unclipped* diagonal Gaussian policy,
summing action dimensions before averaging states.  It supports continuous Box
actions without gSDE or a squashed distribution; it does not change action
clipping, rewards, rollout collection, or PPO's loss coefficients.
"""

# PPO's loss/update loop is adapted from Stable-Baselines3 2.9.0 under MIT:
# Copyright (c) 2019 Antonin Raffin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from gymnasium import spaces
import numpy as np
import torch as th
from torch.nn import functional as F

from .defaults import CURRENT

from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.utils import FloatSchedule, explained_variance, update_learning_rate


def diagonal_gaussian_kl(
    old_mean: th.Tensor,
    old_std: th.Tensor,
    new_mean: th.Tensor,
    new_std: th.Tensor,
) -> th.Tensor:
    """Return per-state KL(old || new), summed over the last (action) axis."""
    # Double precision avoids cancellation when successive policies are close.
    old_mean, old_std = old_mean.double(), old_std.double()
    new_mean, new_std = new_mean.double(), new_std.double()
    per_action = (
        th.log(new_std / old_std)
        + (old_std.square() + (old_mean - new_mean).square()) / (2.0 * new_std.square())
        - 0.5
    )
    return per_action.sum(dim=-1).clamp_min(0.0)


class AdaptiveKLPPO(PPO):
    """PPO with one KL controller and a hard stop before a minibatch step.

    ``kl_learning_rate``, ``kl_low_kl_count`` and ``kl_phase`` are checkpointed.
    ``load`` also accepts ordinary PPO zip files and preserves their optimizer
    state while initializing the new controller.  New models default to early
    phase; pass ``kl_phase="late"`` explicitly for new late-phase models.
    """

    KL_PHASE_BOUNDS = {phase: (CURRENT.learning_rate_min, CURRENT.learning_rate)
                       for phase in ("early", "late")}
    KL_LOW = 0.005
    KL_HIGH = 0.02
    KL_HARD_STOP = 0.03
    KL_DECREASE_FACTOR = 1.5
    KL_INCREASE_FACTOR = 1.2
    KL_LOW_ROLLOUTS = 3

    def __init__(
        self,
        policy: Any,
        env: Any,
        learning_rate: float | None = None,
        *,
        kl_phase: str = "early",
        **kwargs: Any,
    ) -> None:
        self._validate_phase(kl_phase)
        if callable(learning_rate):
            raise ValueError("AdaptiveKLPPO owns the learning-rate schedule; use a numeric learning_rate.")
        if kwargs.get("target_kl") is not None:
            raise ValueError("AdaptiveKLPPO requires target_kl=None; its analytic KL guard is the sole guard.")
        if kwargs.get("use_sde", False):
            raise ValueError("AdaptiveKLPPO requires use_sde=False and an unsquashed diagonal Gaussian policy.")
        kwargs["target_kl"] = None
        initializing = kwargs.get("_init_setup_model", True)
        # BaseAlgorithm.load creates a shell first, then restores its attributes.
        # None lets ordinary PPO checkpoints infer their phase during setup.
        self.kl_phase: str | None = kl_phase if initializing else None
        self.kl_learning_rate: float | None = None
        self.kl_low_kl_count = 0
        self._kl_counter_phase: str | None = None
        self._kl_state_version = 0
        self._kl_legacy_load = False
        self._old_policy_snapshot = None
        initial_lr = self.KL_PHASE_BOUNDS[kl_phase][1] if learning_rate is None else float(learning_rate)
        if not math.isfinite(initial_lr) or initial_lr <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        super().__init__(policy, env, learning_rate=initial_lr, **kwargs)

    @classmethod
    def _validate_phase(cls, phase: str) -> None:
        if phase not in cls.KL_PHASE_BOUNDS:
            raise ValueError(f"kl_phase must be 'early' or 'late', got {phase!r}.")

    def _infer_legacy_phase(self) -> str:
        phase = getattr(self, "manual_training_phase", None)
        if phase in self.KL_PHASE_BOUNDS:
            return phase
        settings = getattr(self, "active_stage_settings", {})
        rate = settings.get("learning_rate") if isinstance(settings, dict) else None
        if rate is None:
            rate = self.learning_rate
        if callable(rate):
            rate = rate(getattr(self, "_current_progress_remaining", 1.0))
        return "early" if float(rate) > self.KL_PHASE_BOUNDS["late"][1] else "late"

    def _setup_model(self) -> None:
        self._kl_legacy_load = self._kl_state_version == 0
        if self.kl_phase is None:
            self.kl_phase = self._infer_legacy_phase()
        self._validate_phase(self.kl_phase)
        if self._kl_counter_phase != self.kl_phase:
            self.kl_low_kl_count = 0
        self._kl_counter_phase = self.kl_phase
        if self.kl_learning_rate is None:
            rate = self.learning_rate
            self.kl_learning_rate = float(rate(self._current_progress_remaining) if callable(rate) else rate)
        self._set_kl_learning_rate(self.kl_learning_rate)
        # Ordinary PPO checkpoints may have used SB3's sampled-KL early stop.
        self.target_kl = None
        self._old_policy_snapshot = None
        super()._setup_model()
        self._validate_policy()
        self._set_kl_learning_rate(self.kl_learning_rate)
        self._kl_state_version = 1

    def _validate_policy(self) -> None:
        if not isinstance(self.action_space, spaces.Box) or self.use_sde:
            raise ValueError("AdaptiveKLPPO supports only Box actions with use_sde=False.")
        if type(self.policy.action_dist) is not DiagGaussianDistribution or self.policy.squash_output:
            raise ValueError("AdaptiveKLPPO requires an unsquashed DiagGaussianDistribution.")

    def _set_kl_learning_rate(self, rate: float) -> None:
        if not math.isfinite(float(rate)) or rate <= 0:
            raise ValueError("The KL learning rate must be finite and positive.")
        low, high = self.KL_PHASE_BOUNDS[self.kl_phase]
        self.kl_learning_rate = float(np.clip(rate, low, high))
        # Keep SB3 metadata and newly constructed policies consistent, but the
        # overridden update below always uses kl_learning_rate as its source.
        self.learning_rate = self.kl_learning_rate
        self.lr_schedule = FloatSchedule(self.kl_learning_rate)
        settings = getattr(self, "active_stage_settings", None)
        if isinstance(settings, dict):
            settings["learning_rate"] = self.kl_learning_rate
        if hasattr(self, "policy"):
            update_learning_rate(self.policy.optimizer, self.kl_learning_rate)

    def _update_learning_rate(self, optimizers: Any) -> None:
        self.logger.record("train/learning_rate", self.kl_learning_rate)
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.kl_learning_rate)

    def set_kl_phase(self, phase: str) -> None:
        """Clip the existing LR into the new range; reset streak only on change."""
        self._validate_phase(phase)
        if phase != self.kl_phase:
            self.kl_low_kl_count = 0
        self.kl_phase = phase
        self._kl_counter_phase = phase
        self._set_kl_learning_rate(self.kl_learning_rate)

    def kl_schedule_metadata(self) -> dict[str, Any]:
        """Return JSON-compatible controller configuration and persisted state."""
        lower, upper = self.KL_PHASE_BOUNDS[self.kl_phase]
        return {
            "version": self._kl_state_version,
            "kl_direction": "old||new",
            "kl_reduction": "sum_actions_mean_states",
            "low_kl": self.KL_LOW,
            "high_kl": self.KL_HIGH,
            "hard_stop_kl": self.KL_HARD_STOP,
            "decrease_divisor": self.KL_DECREASE_FACTOR,
            "increase_multiplier": self.KL_INCREASE_FACTOR,
            "low_kl_rollouts_required": self.KL_LOW_ROLLOUTS,
            "phase_bounds": {key: {"min": value[0], "max": value[1]} for key, value in self.KL_PHASE_BOUNDS.items()},
            "phase": self.kl_phase,
            "learning_rate": self.kl_learning_rate,
            "min_learning_rate": lower,
            "max_learning_rate": upper,
            "low_kl_count": self.kl_low_kl_count,
            "target_kl": None,
        }

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_old_policy_snapshot", "_kl_legacy_load"]

    @classmethod
    def load(cls, *args: Any, **kwargs: Any) -> "AdaptiveKLPPO":
        model = super().load(*args, **kwargs)
        if model._kl_legacy_load:
            # Optimizer restoration happens after _setup_model.  Its current LR
            # is authoritative when upgrading a legacy scheduled PPO checkpoint.
            model.kl_low_kl_count = 0
            rate = model.policy.optimizer.param_groups[0]["lr"]
        else:
            rate = model.kl_learning_rate
        model._set_kl_learning_rate(rate)
        model._kl_legacy_load = False
        return model

    def _snapshot_old_policy(self) -> None:
        self._old_policy_snapshot = deepcopy(self.policy)
        self._old_policy_snapshot.set_training_mode(False)
        self._old_policy_snapshot.requires_grad_(False)
        # The frozen copy is inference-only; don't retain copied optimizer state.
        del self._old_policy_snapshot.optimizer

    def collect_rollouts(self, *args: Any, **kwargs: Any) -> bool:
        self._snapshot_old_policy()
        return super().collect_rollouts(*args, **kwargs)

    def _mean_analytic_kl(self, observations: Any) -> float:
        if self._old_policy_snapshot is None:
            raise RuntimeError("A frozen rollout policy is required for analytic KL.")
        with th.no_grad():
            old = self._old_policy_snapshot.get_distribution(observations).distribution
            new = self.policy.get_distribution(observations).distribution
            value = diagonal_gaussian_kl(old.mean, old.stddev, new.mean, new.stddev).mean().item()
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite analytic policy KL; refusing an optimizer update.")
        return float(value)

    def _full_rollout_kl(self) -> float:
        # Evaluate the entire rollout under the deployed (evaluation-mode) policy.
        # Do not call buffer.get(): its permutation would consume the training RNG.
        observations = self.rollout_buffer.observations
        if not self.rollout_buffer.generator_ready:
            if isinstance(observations, dict):
                observations = {
                    key: self.rollout_buffer.swap_and_flatten(value) for key, value in observations.items()
                }
            else:
                observations = self.rollout_buffer.swap_and_flatten(observations)
        total = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs
        was_training = self.policy.training
        self.policy.set_training_mode(False)
        weighted_kl, count = 0.0, 0
        try:
            for start in range(0, total, self.batch_size):
                stop = min(start + self.batch_size, total)
                if isinstance(observations, dict):
                    batch = {key: self.rollout_buffer.to_torch(value[start:stop]) for key, value in observations.items()}
                else:
                    batch = self.rollout_buffer.to_torch(observations[start:stop])
                size = stop - start
                weighted_kl += self._mean_analytic_kl(batch) * size
                count += size
        finally:
            self.policy.set_training_mode(was_training)
        if count == 0:
            raise RuntimeError("Cannot compute full-rollout KL from an empty rollout.")
        return weighted_kl / count

    def _adapt_learning_rate(self, final_kl: float, hard_stop: bool) -> float:
        if not math.isfinite(final_kl) or final_kl < 0:
            raise ValueError("final_kl must be finite and nonnegative.")
        rate = self.kl_learning_rate
        if hard_stop or final_kl > self.KL_HIGH:
            rate /= self.KL_DECREASE_FACTOR
            self.kl_low_kl_count = 0
        elif 0.0 < final_kl < self.KL_LOW:
            self.kl_low_kl_count += 1
            if self.kl_low_kl_count >= self.KL_LOW_ROLLOUTS:
                rate *= self.KL_INCREASE_FACTOR
                self.kl_low_kl_count = 0
        else:
            self.kl_low_kl_count = 0
        self._set_kl_learning_rate(rate)
        return self.kl_learning_rate

    def train(self) -> None:
        """Run SB3's PPO objective with the additional analytic KL controller."""
        self._validate_policy()
        if self.n_epochs < 1:
            raise ValueError("AdaptiveKLPPO requires n_epochs >= 1.")
        self.target_kl = None
        if self._old_policy_snapshot is None:
            raise RuntimeError("collect_rollouts() must freeze the behavior policy before train().")
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        rollout_lr = self.kl_learning_rate
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        hard_stop = False
        optimizer_steps = 0
        minibatches_per_epoch = math.ceil(self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs / self.batch_size)
        try:
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    advantages = rollout_data.advantages
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                    pg_losses.append(policy_loss.item())
                    clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())
                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())
                    entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                    entropy_losses.append(entropy_loss.item())
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_divs.append(th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy())

                    if self._mean_analytic_kl(rollout_data.observations) > self.KL_HARD_STOP:
                        hard_stop = True
                        if self.verbose >= 1:
                            print(f"Analytic KL hard stop before optimizer step in epoch {epoch + 1}.")
                        break
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    optimizer_steps += 1
                # Preserve SB3's epoch-counter semantics, including a stopped epoch.
                self._n_updates += 1
                if hard_stop:
                    break

            full_kl = self._full_rollout_kl()
            next_lr = self._adapt_learning_rate(full_kl, hard_stop)
            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/explained_variance", explained_var)
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)
            self.logger.record("train/kl_learning_rate", rollout_lr)
            self.logger.record("train/next_learning_rate", next_lr)
            self.logger.record("train/full_kl", full_kl)
            self.logger.record("train/kl_hard_stop", int(hard_stop))
            # Fractional epochs count only minibatches that actually took a step.
            self.logger.record("train/actual_epochs", optimizer_steps / minibatches_per_epoch)
            self.logger.record("train/optimizer_steps", optimizer_steps)
            self.logger.record("train/low_kl_count", self.kl_low_kl_count)
        finally:
            self._old_policy_snapshot = None
