"""Training episodes request an exam; only a frozen-policy exam permits flight."""
from collections import deque
from copy import deepcopy

import gymnasium as gym


class RigToFlightCurriculum(gym.Wrapper):
    def __init__(self, env, window_episodes=5, required_qualified=3,
                 evaluation_episodes=10, evaluation_required=8, cooldown_steps=65536):
        super().__init__(env)
        for total, required in ((window_episodes, required_qualified),
                                (evaluation_episodes, evaluation_required)):
            if type(total) is not int or total < 1 or type(required) is not int or not 1 <= required <= total:
                raise ValueError("Episode windows and required counts must be positive integers with required <= window")
        self._check_step(cooldown_steps)
        self.window_episodes, self.required_qualified = window_episodes, required_qualified
        self.evaluation_episodes, self.evaluation_required = evaluation_episodes, evaluation_required
        self.cooldown_steps = cooldown_steps
        self.history = deque(maxlen=window_episodes)
        self.ready_for_air = env.backend.config["mode"] == "free_flight"
        self.stage_override = "free_flight" if self.ready_for_air else None
        self.attempt_count = 0
        self.last_evaluation_step = None
        self.last_evaluation = None
        self._pending_evaluation = None
        self.prescreen_requested = False

    @staticmethod
    def _check_step(value):
        if type(value) is not int or value < 0:
            raise ValueError("Training step counts must be non-negative integers")

    @property
    def window_size(self):
        return len(self.history)

    @property
    def qualified_count(self):
        return sum(self.history)

    @property
    def full(self):
        return self.window_size == self.window_episodes

    @property
    def pending_evaluation(self):
        return deepcopy(self._pending_evaluation)

    def _info(self):
        # Legacy log columns now describe only the weaker training prescreen.
        return dict(promotion_window_size=self.window_size,
                    promotion_qualified_count=self.qualified_count,
                    promotion_window_full=self.full, promotion_history=list(self.history),
                    promotion_window_role="training_prescreen_only",
                    curriculum_ready=self.ready_for_air,
                    evaluation_pending=self._pending_evaluation is not None,
                    prescreen_requested=self.prescreen_requested,
                    evaluation_attempt_count=self.attempt_count)

    def reset(self, **kwargs):
        if self.ready_for_air and self.env.backend.config["mode"] == "ball_rig":
            self.env.backend.set_mode("free_flight")
        obs, info = self.env.reset(**kwargs)
        info.update(self._info())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self.ready_for_air and info["training_stage"] == "ball_rig" and (terminated or truncated):
            # Failures occupy the window; interrupted resets and infrastructure
            # errors do not fabricate completed training episodes.
            self.history.append(bool(info["is_success"] and info["transfer_qualified"]))
        info.update(self._info())
        return obs, reward, terminated, truncated, info

    def evaluation_due(self, training_step):
        self._check_step(training_step)
        if self._pending_evaluation is not None:
            return True  # Resume the frozen checkpoint before further training.
        return bool(not self.ready_for_air
                    and (self.prescreen_requested or
                         (self.full and self.qualified_count >= self.required_qualified))
                    and (self.last_evaluation_step is None
                         or training_step - self.last_evaluation_step >= self.cooldown_steps))

    def request_evaluation(self, training_step):
        """Latch a qualifying window until the current PPO update is complete."""
        self._check_step(training_step)
        if not self.ready_for_air and self._pending_evaluation is None and self.evaluation_due(training_step):
            self.prescreen_requested = True
        return self.prescreen_requested

    def _check_identity(self, checkpoint_id, seeds):
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
            raise ValueError("An evaluation requires a non-empty checkpoint identity")
        if (not isinstance(seeds, list) or len(seeds) != self.evaluation_episodes
                or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError("An evaluation requires one distinct non-negative integer seed per episode")

    def begin_evaluation(self, training_step, checkpoint_id, seeds):
        self._check_step(training_step)
        if self._pending_evaluation is not None:
            raise ValueError("A pending evaluation cannot be overwritten")
        if not self.evaluation_due(training_step):
            raise ValueError("Evaluation prescreen or cooldown is not satisfied")
        self._check_identity(checkpoint_id, seeds)
        self.attempt_count += 1
        self.prescreen_requested = False
        self.stage_override = None
        self._pending_evaluation = dict(training_step=training_step,
            checkpoint_id=checkpoint_id, seeds=deepcopy(seeds), attempt_count=self.attempt_count)
        return self.pending_evaluation

    def _check_report(self, report):
        required = {"valid", "checkpoint_id", "seeds", "completed_episodes",
                    "qualified_count", "passed", "episodes"}
        if not isinstance(report, dict) or not required.issubset(report):
            raise ValueError("Incomplete evaluation report")
        self._check_identity(report["checkpoint_id"], report["seeds"])
        if type(report["valid"]) is not bool or type(report["passed"]) is not bool:
            raise ValueError("Evaluation validity and result must be booleans")
        count, qualified, episodes = report["completed_episodes"], report["qualified_count"], report["episodes"]
        if (type(count) is not int or not 0 <= count <= self.evaluation_episodes
                or type(qualified) is not int or not 0 <= qualified <= count
                or not isinstance(episodes, list) or len(episodes) != count):
            raise ValueError("Invalid evaluation episode counts")
        for episode, seed in zip(episodes, report["seeds"]):
            if (not isinstance(episode, dict) or type(episode.get("seed")) is not int
                    or episode["seed"] != seed or type(episode.get("qualified")) is not bool):
                raise ValueError("Evaluation episodes must match the ordered seed manifest")
        if qualified != sum(episode["qualified"] for episode in episodes):
            raise ValueError("Evaluation qualified count disagrees with its episodes")
        if report["valid"] and count != self.evaluation_episodes:
            raise ValueError("A valid evaluation must complete the entire episode window")
        expected = report["valid"] and qualified >= self.evaluation_required
        if report["passed"] != expected:
            raise ValueError("Evaluation result disagrees with its validity and qualified count")

    def record_evaluation(self, report, training_step):
        self._check_step(training_step)
        pending = self._pending_evaluation
        if pending is None:
            raise ValueError("No pending evaluation to complete")
        self._check_report(report)
        if (training_step != pending["training_step"]
                or report["checkpoint_id"] != pending["checkpoint_id"]
                or report["seeds"] != pending["seeds"]):
            raise ValueError("Evaluation must use the pending checkpoint, seeds and frozen training step")
        if not report["valid"]:
            pending["last_invalid_report"] = deepcopy(report)
            return False
        self.last_evaluation = dict(deepcopy(report), training_step=training_step,
                                    attempt_count=self.attempt_count)
        self.last_evaluation_step = training_step
        self._pending_evaluation = None
        self.prescreen_requested = False
        self.ready_for_air = report["passed"]
        return self.ready_for_air

    def override_stage(self, stage):
        if stage not in ("ball_rig", "free_flight"):
            raise ValueError("Unknown curriculum stage override")
        self.env.backend.set_mode(stage)
        self.history.clear()
        self.ready_for_air = stage == "free_flight"
        self.stage_override = stage
        self.attempt_count = 0
        self.prescreen_requested = False
        self.last_evaluation_step = self.last_evaluation = self._pending_evaluation = None

    def state(self):
        return deepcopy(dict(gate_version=4, window_episodes=self.window_episodes,
            required_qualified=self.required_qualified, evaluation_episodes=self.evaluation_episodes,
            evaluation_required=self.evaluation_required, cooldown_steps=self.cooldown_steps,
            history=list(self.history), window_size=self.window_size, qualified_count=self.qualified_count,
            full=self.full, ready_for_air=self.ready_for_air, training_stage=self.env.backend.config["mode"],
            stage_override=self.stage_override, attempt_count=self.attempt_count,
            prescreen_requested=self.prescreen_requested,
            last_evaluation_step=self.last_evaluation_step, last_evaluation=self.last_evaluation,
            pending_evaluation=self._pending_evaluation))

    def restore(self, state):
        if not isinstance(state, dict) or type(state.get("gate_version")) is not int or state["gate_version"] != 4:
            raise ValueError("Checkpoint lacks frozen-evaluation gate version 4; old training gates cannot certify flight")
        if not self.state().keys() <= state.keys():
            raise ValueError("Incomplete curriculum checkpoint state")
        for key in ("window_episodes", "required_qualified", "evaluation_episodes", "evaluation_required", "cooldown_steps"):
            if type(state[key]) is not int or state[key] != getattr(self, key):
                raise ValueError("Resume with the same prescreen, evaluation and cooldown configuration")
        history = state["history"]
        if not isinstance(history, list) or len(history) > self.window_episodes or any(type(x) is not bool for x in history):
            raise ValueError("Invalid curriculum history")
        if (type(state["window_size"]) is not int or state["window_size"] != len(history)
                or type(state["qualified_count"]) is not int or state["qualified_count"] != sum(history)
                or type(state["full"]) is not bool or state["full"] != (len(history) == self.window_episodes)):
            raise ValueError("Curriculum window summary disagrees with its history")
        stage, ready, override = state["training_stage"], state["ready_for_air"], state["stage_override"]
        if stage not in ("ball_rig", "free_flight") or type(ready) is not bool or override not in (None, "ball_rig", "free_flight"):
            raise ValueError("Invalid curriculum stage")
        requested = state["prescreen_requested"]
        if type(requested) is not bool:
            raise ValueError("Prescreen request must be a boolean")
        attempt, last, last_step, pending = (state[k] for k in
            ("attempt_count", "last_evaluation", "last_evaluation_step", "pending_evaluation"))
        self._check_step(attempt)
        if last is not None:
            self._check_report(last)
            if not last["valid"]:
                raise ValueError("Only a valid completed evaluation belongs in last_evaluation")
            self._check_step(last_step)
            if type(last.get("training_step")) is not int or last["training_step"] != last_step:
                raise ValueError("Evaluation completion step disagrees with cooldown state")
        elif last_step is not None:
            raise ValueError("Cooldown requires a completed evaluation")
        if pending is not None:
            if not isinstance(pending, dict) or not {"training_step", "checkpoint_id", "seeds", "attempt_count"} <= pending.keys():
                raise ValueError("Incomplete pending evaluation")
            self._check_step(pending["training_step"])
            self._check_identity(pending["checkpoint_id"], pending["seeds"])
            if type(pending["attempt_count"]) is not int or pending["attempt_count"] != attempt or attempt < 1:
                raise ValueError("Pending evaluation attempt mismatch")
            if last_step is not None and pending["training_step"] - last_step < self.cooldown_steps:
                raise ValueError("Pending evaluation violates cooldown")
            if "last_invalid_report" in pending:
                invalid = pending["last_invalid_report"]
                self._check_report(invalid)
                if invalid["valid"] or any(invalid[k] != pending[k] for k in ("checkpoint_id", "seeds")):
                    raise ValueError("Invalid report does not belong to pending evaluation")
        completed_attempt = attempt - int(pending is not None)
        if (last is None and completed_attempt != 0) or (last is not None and
                (type(last.get("attempt_count")) is not int or last["attempt_count"] != completed_attempt or completed_attempt < 1)):
            raise ValueError("Evaluation attempt history is inconsistent")
        passed = last is not None and last["passed"]
        if (ready != bool(passed or override == "free_flight")
                or (stage == "free_flight" and not ready)
                or (pending is not None and (ready or stage != "ball_rig"))
                or (override is not None and (attempt != 0 or override != stage))
                or (override == "free_flight" and history)):
            raise ValueError("Curriculum stage lacks matching evaluation or explicit override")
        if requested and (ready or pending is not None or stage != "ball_rig" or not state["full"]):
            raise ValueError("Prescreen request conflicts with the stage or pending evaluation")
        # Validate everything before mutating the live backend or gate.
        self.env.backend.set_mode(stage)
        self.history = deque(history, maxlen=self.window_episodes)
        self.ready_for_air, self.stage_override = ready, override
        self.attempt_count, self.last_evaluation_step = attempt, last_step
        self.prescreen_requested = requested
        self.last_evaluation, self._pending_evaluation = deepcopy(last), deepcopy(pending)
