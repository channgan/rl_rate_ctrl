"""Training callbacks, independent of CLI and environment construction."""
import json

from stable_baselines3.common.callbacks import BaseCallback


CONTINUOUS_COMPONENT_NAMES = ("rate", "slew_rate", "peak_pwm", "tracking_bonus")


class RewardMetrics(BaseCallback):
    def __init__(self, curriculum=None, previous=None, survival_gate=None):
        super().__init__()
        self.saturated_motor_steps = 0
        self.saturated_torque_axis_steps = 0
        self.successes = 0
        self.failures = 0
        self.first_success_step = None
        self.batch_size_switch_step = None
        self.curriculum = curriculum
        self.survival_gate = survival_gate
        if previous:
            self.first_success_step = previous.get("first_success_step")
            self.batch_size_switch_step = previous.get("batch_size_switch_step")

    def _on_step(self):
        for index, info in enumerate(self.locals["infos"]):
            if self.survival_gate is not None:
                self.survival_gate.observe(info, bool(self.locals['dones'][index]), self.num_timesteps)
            # Native policy limits are three signed torque axes. Allocated
            # motor saturation remains a separate four-channel diagnostic.
            motor_saturation_count = info.get("motor_saturation_count", info["saturation_count"])
            torque_saturation_count = info.get("torque_saturation_count", 0)
            self.saturated_motor_steps += motor_saturation_count
            self.saturated_torque_axis_steps += torque_saturation_count
            self.successes += int(info["is_success"])
            self.failures += int(info["failure"] is not None)
            if info["is_success"] and self.first_success_step is None:
                self.first_success_step = self.num_timesteps
            # Training episodes only feed the prescreen. Stage changes belong
            # to PromotionTraining, after optimization and a frozen exam.
            for name in ("tracking_cost", "rate_bounded_cost", "slew_rate_cost",
                         "upper_headroom", "peak_pwm_cost", "peak_pwm_penalty", "tracking_eligible", "tracking_bonus",
                         "saturation_penalty", "failure_penalty", "success_bonus"):
                self.logger.record_mean("reward/" + name, info[name])
            self.logger.record_mean("control/allocated_motor_saturation_count", motor_saturation_count)
            if "torque_saturation_count" in info:
                self.logger.record_mean("reward/torque_saturation_axis_count", torque_saturation_count)
            else:
                self.logger.record_mean("reward/motor_saturation_count", motor_saturation_count)
            self.logger.record_mean("tracking/error_abs_mean_deg_s",
                                    float(sum(abs(x) for x in info["error_deg_s"]) / 3))
            self.logger.record("reward/error_scale_deg_s", info["reward_error_scale_deg_s"])
            pwm_rates = info["pwm_rate_per_s"]
            pwm_prefix = "control/" if "torque_rate_per_s" in info else "reward/"
            self.logger.record_mean(pwm_prefix + "pwm_rate_abs_mean_per_s", float(sum(abs(x) for x in pwm_rates) / 4))
            self.logger.record_mean(pwm_prefix + "pwm_rate_max_abs_per_s", float(max(abs(x) for x in pwm_rates)))
            if "torque_rate_per_s" in info:
                torque_rates = info["torque_rate_per_s"]
                self.logger.record_mean("reward/torque_rate_abs_mean_per_s", float(sum(abs(x) for x in torque_rates) / 3))
                self.logger.record_mean("reward/torque_rate_max_abs_per_s", float(max(abs(x) for x in torque_rates)))
            components = info["continuous_reward_components"]
            if len(components) != len(CONTINUOUS_COMPONENT_NAMES):
                raise ValueError("Expected all four continuous reward components")
            for name, value in zip(CONTINUOUS_COMPONENT_NAMES, components):
                self.logger.record_mean("reward/weighted_" + name, float(value))
            if "episode_reward_components" in info:
                for name, value in zip(CONTINUOUS_COMPONENT_NAMES, info["episode_reward_components"]):
                    self.logger.record_mean("episode_reward/" + name, float(value))
                for name in ("saturation_penalty", "failure_penalty", "success_bonus"):
                    self.logger.record_mean("episode_reward/" + name, float(info["episode_" + name]))
                self.logger.record_mean("tracking/episode_rate_rmse_deg_s", info["episode_rate_rmse_deg_s"])
                if "episode_saturation_fraction" in info:
                    domain = "torque_axis" if "torque_saturation_count" in info else "motor"
                    self.logger.record_mean("episode_reward/" + domain + "_saturation_fraction",
                                            info["episode_saturation_fraction"])
                if "episode_motor_saturation_fraction" in info:
                    self.logger.record_mean("control/episode_allocated_motor_saturation_fraction",
                                            info["episode_motor_saturation_fraction"])
                self.logger.record_mean("curriculum/transfer_qualified", int(info["transfer_qualified"]))
        if self.curriculum is not None:
            self.curriculum.request_evaluation(self.num_timesteps)
            self.model.curriculum_state = self.curriculum.state()
            self.logger.record("curriculum/window_size", self.curriculum.window_size)
            self.logger.record("curriculum/qualified_count", self.curriculum.qualified_count)
            self.logger.record("curriculum/full", int(self.curriculum.full))
            self.logger.record("curriculum/history", json.dumps(list(self.curriculum.history)))
            self.logger.record("curriculum/free_flight", int(self.curriculum.ready_for_air))
            self.logger.record("curriculum/evaluation_attempts", self.curriculum.attempt_count)
            self.logger.record("curriculum/evaluation_due",
                               int(self.curriculum.evaluation_due(self.num_timesteps)))
        self.model.training_schedule_state = dict(first_success_step=self.first_success_step,
                                                  batch_size_switch_step=self.batch_size_switch_step)
        return True

    def _on_rollout_end(self):
        self.logger.record("train/batch_size", self.model.batch_size)
        self.logger.record("train/ent_coef", self.model.ent_coef)


class CompactMetrics(RewardMetrics):
    """Keep core diagnostics without repeated coordinate/unit variants."""
    def _on_step(self):
        result = super()._on_step()
        for key in list(self.logger.name_to_value):
            if key.startswith(("tracking/", "reward/", "curriculum/", "episode_reward/", "control/")):
                self.logger.name_to_value.pop(key, None)
                self.logger.name_to_count.pop(key, None)
                self.logger.name_to_excluded.pop(key, None)
        return result
