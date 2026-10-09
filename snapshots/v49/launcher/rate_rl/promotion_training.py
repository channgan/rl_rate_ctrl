"""Run evaluation only between complete PPO updates, sharing one simulator slot."""
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import torch

from .backend import SimulatorError
from .evaluation import evaluate_frozen_policy, make_evaluation_seeds, policy_fingerprint
from .environment_contract import environment_contract
from .ppo_schedule import apply_stage_settings


def write_json(path, value):
    """Never leave a partially written report looking like a completed exam."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class PromotionTraining:
    """Factories create new Gym environments; neither factory may reset them.

    training_factory(curriculum_state, rng_state, session_index) returns
    (monitored_environment, curriculum). evaluation_factory() returns a plain
    independent evaluation environment, with no training curriculum or Monitor.
    Only one of those environments owns the PX4 process slot at any time.
    """

    def __init__(self, model, curriculum, metrics, training_factory, evaluation_factory,
                 run_dir, *, evaluator=evaluate_frozen_policy):
        if model.n_envs != 1:
            raise ValueError("Promotion evaluation currently supports one training environment")
        self.model = model
        self.curriculum = curriculum
        self.metrics = metrics
        self.training_factory = training_factory
        self.evaluation_factory = evaluation_factory
        self.run_dir = Path(run_dir)
        self.evaluator = evaluator
        self.environment_contract = environment_contract(curriculum)
        saved_contract = getattr(model, "promotion_environment_contract", self.environment_contract)
        if saved_contract != self.environment_contract:
            raise ValueError("Checkpoint evaluation environment contract differs from this task or simulator")
        model.promotion_environment_contract = deepcopy(self.environment_contract)
        self.state = deepcopy(getattr(model, "promotion_training_state", {
            "completed_rollouts": 0, "session_index": 0, "evaluation_resets": 0,
            "last_checkpoint_step": model.num_timesteps,
        }))
        if hasattr(model, "training_rng_state"):
            rng = model.training_rng_state
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.random.set_rng_state(rng["torch_cpu"].cpu())

    def sync_state(self):
        self.model.curriculum_state = self.curriculum.state()
        self.model.training_env_rng_state = deepcopy(self.curriculum.unwrapped.np_random.bit_generator.state)
        self.model.promotion_training_state = deepcopy(self.state)
        self.model.training_rng_state = dict(python=random.getstate(), numpy=np.random.get_state(),
                                            torch_cpu=torch.random.get_rng_state())
        self.model.training_schedule_state = dict(
            first_success_step=self.metrics.first_success_step,
            batch_size_switch_step=self.metrics.batch_size_switch_step)
        if self.metrics.survival_gate is not None:
            self.model.exploration_survival_gate = self.metrics.survival_gate.state()

    def _save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.sync_state()
        self.model.save(path)

    def _episode_progress(self, episode):
        print(f"Promotion evaluation: seed={episode['seed']} "
              f"qualified={episode['qualified']}", flush=True)

    def _evaluation_environment(self):
        env = self.evaluation_factory()
        try:
            if env.unwrapped.backend.config["mode"] != "ball_rig":
                raise ValueError("Promotion evaluation requires the ball rig environment")
            if environment_contract(env) != self.environment_contract:
                raise ValueError("Evaluation task or simulator assets changed since training")
        except BaseException:
            env.close()
            raise
        return env

    def evaluate_pending(self):
        """Evaluate the current post-update checkpoint, or retry its saved exam.

        On an infrastructure failure the pending request remains resumable and
        this method raises. No training resumes until the invalid exam is handled.
        """
        model, curriculum = self.model, self.curriculum
        if not curriculum.evaluation_due(model.num_timesteps):
            return False
        fingerprint = policy_fingerprint(model.policy)
        pending = curriculum.pending_evaluation
        if pending is None:
            seeds = make_evaluation_seeds(
                0 if model.seed is None else model.seed,
                curriculum.attempt_count + 1, curriculum.evaluation_episodes)
            curriculum.begin_evaluation(model.num_timesteps, fingerprint, seeds)
            pending = curriculum.pending_evaluation
        if pending["checkpoint_id"] != fingerprint or pending["training_step"] != model.num_timesteps:
            raise ValueError("Pending evaluation does not match this checkpoint and training step")
        exam_dir = self.run_dir / "evaluations" / f"attempt_{pending['attempt_count']:04d}"
        request_path = exam_dir / "request.json"
        if request_path.exists():
            previous = json.loads(request_path.read_text(encoding="utf-8"))
            if any(previous.get(key) != pending[key] for key in
                   ("checkpoint_id", "seeds", "training_step", "attempt_count")):
                raise ValueError("Existing evaluation artifacts belong to another candidate; use a new run directory")
        # Includes optimizer state and pending exam identity, so invalid exams
        # can be retried with the exact same policy rather than a newer policy.
        self._save(exam_dir / "candidate")
        write_json(request_path, pending)
        rng_state = deepcopy(model.training_env_rng_state)
        base = curriculum.unwrapped
        interrupted_steps = base.steps if not base.needs_reset else 0
        write_json(exam_dir / "training_pause.json", dict(
            training_step=model.num_timesteps, interrupted_episode_steps=interrupted_steps,
            reason="evaluation_reset", counted_as_success=False, counted_as_failure=False,
            completed_rollouts=self.state["completed_rollouts"]))
        # Closing releases the fixed PX4 instance lock. SB3's old observation
        # remains unused; set_env(force_reset=True) below discards it entirely.
        model.get_env().close()
        try:
            report = self.evaluator(model.policy, self._evaluation_environment,
                seeds=pending["seeds"], checkpoint_id=fingerprint,
                required_qualified=curriculum.evaluation_required,
                on_episode=self._episode_progress)
        except BaseException:
            self._save(exam_dir / "interrupted")
            raise
        report["environment_contract"] = deepcopy(self.environment_contract)
        write_json(exam_dir / "report.json", report)
        if policy_fingerprint(model.policy) != fingerprint:
            raise RuntimeError("Evaluation modified the training policy")
        curriculum.record_evaluation(report, model.num_timesteps)
        self._save(exam_dir / ("passed" if report["passed"] else "after_evaluation"))
        if not report["valid"]:
            raise SimulatorError(f"Promotion evaluation invalid; pending checkpoint: {exam_dir / 'after_evaluation.zip'}; "
                                 f"{report.get('error', 'infrastructure failure')}")
        next_stage = "free_flight" if curriculum.ready_for_air else "ball_rig"
        print(f"Promotion evaluation: {report['qualified_count']}/{report['completed_episodes']} "
              f"qualified; next training stage={next_stage}", flush=True)
        state = curriculum.state()
        self.state["session_index"] += 1
        self.state["evaluation_resets"] += 1
        new_env, new_curriculum = self.training_factory(state, rng_state, self.state["session_index"])
        try:
            if environment_contract(new_curriculum) != self.environment_contract:
                raise ValueError("Training task or simulator assets changed during evaluation")
            model.set_env(new_env, force_reset=True)
        except BaseException:
            new_env.close()
            raise
        self.curriculum = new_curriculum
        self.metrics.curriculum = new_curriculum
        if new_curriculum.ready_for_air and model.batch_size != 256:
            self.metrics.batch_size_switch_step = model.num_timesteps
        # No rig buffer remains to optimize. The first subsequent sample and
        # its next optimizer update both use the air stage's settings.
        apply_stage_settings(model, new_curriculum.ready_for_air)
        self.sync_state()
        self._save(exam_dir / "resume_training")
        return True

    def learn(self, total_timesteps, callback, stop_after_rollout=None):
        if total_timesteps is None and stop_after_rollout is None:
            raise ValueError('Unbounded training requires an explicit stop condition')
        if total_timesteps is not None and (type(total_timesteps) is not int or total_timesteps <= 0):
            raise ValueError("total_timesteps must be a positive integer")
        model = self.model
        end = None if total_timesteps is None else model.num_timesteps + total_timesteps
        # A resumed phase must not inherit the previous phase's stop reason.
        self.state.pop("stop_reason", None)
        self.state.pop("stop_after_update_step", None)
        # A saved invalid/pending exam must be resolved before any new update.
        self.evaluate_pending()
        while end is None or model.num_timesteps < end:
            before_steps, before_updates = model.num_timesteps, model._n_updates
            model.learn(total_timesteps=model.n_steps, callback=callback,
                        reset_num_timesteps=False, tb_log_name="PPO")
            if (model.num_timesteps != before_steps + model.n_steps
                    or model._n_updates <= before_updates or not model.rollout_buffer.full):
                # A callback stop is not a completed update and cannot certify
                # a candidate or be counted toward the retry cooldown.
                raise RuntimeError("PPO stopped before completing a full rollout and optimization")
            self.state["completed_rollouts"] += 1
            # _setup_learn otherwise replaces the logger on every short learn
            # call, discarding train() diagnostics recorded after SB3's dump.
            model.set_logger(model.logger)
            model.logger.record("train/completed_rollouts", self.state["completed_rollouts"])
            model.logger.dump(step=model.num_timesteps)
            self.sync_state()
            if stop_after_rollout is not None and stop_after_rollout():
                self.state['stop_reason'] = 'full_episode_survival_gate'
                self.state['stop_after_update_step'] = model.num_timesteps
                break
            self.evaluate_pending()
            if model.num_timesteps - self.state["last_checkpoint_step"] >= 10000:
                self.state["last_checkpoint_step"] = model.num_timesteps
                self._save(self.run_dir / "checkpoints" / f"ppo_{model.num_timesteps}_steps")
        else:
            self.state["stop_reason"] = "step_budget"
            self.state["stop_after_update_step"] = model.num_timesteps
        self.sync_state()
        return model
