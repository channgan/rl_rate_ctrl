"""Validated public run plans, independent of simulator and PPO imports.

Building a plan is read-only. Only RunSupervisor creates a run and processes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Sequence

from .defaults import CURRENT


@dataclass(frozen=True)
class RunConfig:
    run: Path
    steps: int
    phase: str = CURRENT.phase
    resume: Path | None = None
    runtime: str = field(default_factory=lambda: str(Path.home() / "rl_rate/runtime.free_flight.json"))
    smoke: bool = False
    n_envs: int = CURRENT.n_envs
    n_steps: int | None = None
    batch_size: int = CURRENT.batch_size
    n_epochs: int = CURRENT.n_epochs
    base_instance: int = CURRENT.base_instance
    episode_steps: int = CURRENT.episode_steps
    allow_episode_length_change: bool = False
    allow_failure2700_change: bool = False
    allow_axis_success_change: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "run", Path(self.run).resolve())
        if self.resume is not None:
            object.__setattr__(self, "resume", Path(self.resume).resolve())

    @property
    def per_environment_steps(self) -> int:
        return self.n_steps if self.n_steps is not None else CURRENT.rollout_steps // self.n_envs

    @property
    def rollout_samples(self) -> int:
        return self.n_envs * self.per_environment_steps

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.phase not in ("early", "late"):
            raise ValueError("phase must be early or late")
        if self.n_envs not in (1, 2, 4):
            raise ValueError("n-envs must be one of 1, 2, 4")
        if not 0 <= self.base_instance <= 101 - self.n_envs:
            raise ValueError("base-instance must keep every PX4 instance between 0 and 100")
        if self.episode_steps < 1:
            raise ValueError("episode-steps must be positive")
        if self.allow_episode_length_change and self.resume is None:
            raise ValueError("Episode length migration requires --resume")
        if self.allow_failure2700_change and self.resume is None:
            raise ValueError("Failure settlement migration requires --resume")
        if (self.per_environment_steps < 2 or self.batch_size < 2
                or self.n_epochs < 1 or self.rollout_samples % self.batch_size):
            raise ValueError("Positive rollout/epochs required; batch-size must divide total samples")
        if self.run.exists():
            raise ValueError("Use a new run directory to preserve prior models/logs")


@dataclass(frozen=True)
class RunPlan:
    config: RunConfig
    project: Path
    python: str
    command: tuple[str, ...]

    def publisher_command(self, script: str) -> tuple[str, ...]:
        return (self.python, "-u", str(self.project / "scripts" / f"{script}.py"),
                "--run", str(self.config.run))

    def as_dict(self) -> dict:
        """Display the settings actually passed to the trainer."""
        config = self.config
        return {
            "run": str(config.run), "project": str(self.project),
            "resume": str(config.resume) if config.resume else None,
            "phase": config.phase, "requested_steps": config.steps,
            "runtime": config.runtime, "n_envs": config.n_envs,
            "n_steps": config.per_environment_steps, "rollout_samples": config.rollout_samples,
            "batch_size": config.batch_size, "n_epochs": config.n_epochs,
            "episode_steps": config.episode_steps, "control_dt": CURRENT.control_dt,
            "torque_slew_scale": CURRENT.torque_slew_scale, "slew_weight": CURRENT.slew_weight,
            "automatic_phase_switch": False, "command": list(self.command),
        }


def build_plan(config: RunConfig, *, project: Path, python: str | None = None) -> RunPlan:
    config.validate()
    python = python or sys.executable
    command = [python, "-B", "-u", "-m", "rate_rl.train",
               "--run", str(config.run), "--steps", str(config.steps), "--runtime", config.runtime,
               "--training-phase", config.phase, "--stage", "free_flight", "--native-torque",
               "--simple-critic", "--px4-native-outer", "--waypoint-tracking", "--squared-error-reward",
               "--control-dt", str(CURRENT.control_dt),
               "--torque-slew-scale", str(CURRENT.torque_slew_scale),
               "--slew-weight", str(CURRENT.slew_weight),
               "--air-rate-limits", "1", "1", "1", "--no-tilt-termination", "--compact-logs",
               "--n-envs", str(config.n_envs), "--n-steps", str(config.per_environment_steps),
               "--batch-size", str(config.batch_size), "--n-epochs", str(config.n_epochs),
               "--base-instance", str(config.base_instance), "--episode-steps", str(config.episode_steps)]
    if config.resume is not None:
        command += ["--resume", str(config.resume), "--allow-phase-change", "--allow-tracking-tuning"]
    for name in ("smoke", "allow_episode_length_change", "allow_axis_success_change", "allow_failure2700_change"):
        if getattr(config, name):
            command.append("--" + name.replace("_", "-"))
    return RunPlan(config=config, project=Path(project).resolve(), python=python, command=tuple(command))


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Native PX4 thrust/allocator, learned rate torque.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--phase", choices=["early", "late"], default=CURRENT.phase)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--runtime", default=str(Path.home() / "rl_rate/runtime.free_flight.json"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-envs", type=int, choices=[1, 2, 4], default=CURRENT.n_envs)
    parser.add_argument("--n-steps", type=int,
                        help=f"Per-environment rollout length; default total samples={CURRENT.rollout_steps}")
    parser.add_argument("--batch-size", type=int, default=CURRENT.batch_size)
    parser.add_argument("--n-epochs", type=int, default=CURRENT.n_epochs)
    parser.add_argument("--base-instance", type=int, default=CURRENT.base_instance)
    parser.add_argument("--episode-steps", type=int, default=CURRENT.episode_steps)
    parser.add_argument("--allow-episode-length-change", action="store_true")
    parser.add_argument("--allow-failure2700-change", action="store_true")
    parser.add_argument("--allow-axis-success-change", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print validated command/settings without creating files or starting processes")
    return parser


def main(argv: Sequence[str] | None = None, *, project: Path | None = None) -> int:
    parser = argument_parser()
    arguments = vars(parser.parse_args(argv))
    dry_run = arguments.pop("dry_run")
    try:
        plan = build_plan(RunConfig(**arguments), project=project or Path(__file__).resolve().parents[1])
    except ValueError as error:
        parser.error(str(error))
    if dry_run:
        print(json.dumps(plan.as_dict(), indent=2))
        return 0
    from .run_manager import RunSupervisor
    return RunSupervisor(plan).run()


if __name__ == "__main__":
    raise SystemExit(main())
