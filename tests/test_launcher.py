"""The public CLI has a side-effect-free plan and preserves the current recipe."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from rate_rl.defaults import CURRENT
from rate_rl.launcher import RunConfig, build_plan, main


PROJECT = Path(__file__).resolve().parents[1]


def option(command, name):
    return command[command.index(name) + 1]


def test_plan_preserves_current_training_command(tmp_path):
    run = tmp_path / "new_run"
    plan = build_plan(RunConfig(run=run, steps=3500000), project=PROJECT, python="python-test")
    assert plan.command[:5] == ("python-test", "-B", "-u", "-m", "rate_rl.train")
    expected = {
        "--training-phase": "late", "--stage": "free_flight", "--control-dt": .01,
        "--torque-slew-scale": 10., "--slew-weight": .02 / 7., "--n-envs": 1,
        "--n-steps": 4096, "--batch-size": 2048, "--n-epochs": 10,
        "--base-instance": 41, "--episode-steps": 2048,
    }
    for name, value in expected.items():
        actual = option(plan.command, name)
        assert actual == value if isinstance(value, str) else float(actual) == value
    for flag in ("--native-torque", "--simple-critic", "--px4-native-outer",
                 "--waypoint-tracking", "--squared-error-reward", "--no-tilt-termination", "--compact-logs"):
        assert flag in plan.command
    assert "--resume" not in plan.command
    assert not plan.as_dict()["automatic_phase_switch"]
    assert not run.exists()


@pytest.mark.parametrize("n_envs,n_steps", [(1, 4096), (2, 2048), (4, 1024)])
def test_default_total_rollout_is_independent_of_environment_count(tmp_path, n_envs, n_steps):
    plan = build_plan(RunConfig(run=tmp_path / "run", steps=100, n_envs=n_envs), project=PROJECT)
    assert plan.config.per_environment_steps == n_steps
    assert plan.as_dict()["rollout_samples"] == CURRENT.rollout_steps


def test_all_existing_overrides_and_migration_flags_are_forwarded(tmp_path):
    config = RunConfig(run=tmp_path / "new", steps=31, phase="early", resume=tmp_path / "old.zip",
                       runtime="custom-runtime.json", smoke=True, n_envs=2, n_steps=32,
                       batch_size=16, n_epochs=3, base_instance=71, episode_steps=15,
                       allow_episode_length_change=True, allow_failure2700_change=True,
                       allow_axis_success_change=True)
    plan = build_plan(config, project=PROJECT)
    for flag in ("--allow-phase-change", "--allow-tracking-tuning", "--smoke",
                 "--allow-episode-length-change", "--allow-failure2700-change", "--allow-axis-success-change"):
        assert flag in plan.command
    assert option(plan.command, "--runtime") == "custom-runtime.json"
    assert option(plan.command, "--n-steps") == "32"
    assert option(plan.command, "--base-instance") == "71"
    assert option(plan.command, "--episode-steps") == "15"
    assert plan.as_dict()["rollout_samples"] == 64
    assert not config.run.exists()


@pytest.mark.parametrize("settings", [
    {"steps": 0}, {"episode_steps": 0}, {"batch_size": 300}, {"batch_size": 0},
    {"n_steps": 1}, {"n_epochs": 0}, {"n_envs": 0}, {"phase": "invalid"},
    {"base_instance": -1}, {"base_instance": 101}, {"n_envs": 4, "base_instance": 98},
    {"allow_episode_length_change": True}, {"allow_failure2700_change": True},
])
def test_invalid_plan_creates_nothing(tmp_path, settings):
    run = tmp_path / "run"
    with pytest.raises(ValueError):
        build_plan(RunConfig(**(dict(run=run, steps=100) | settings)), project=PROJECT)
    assert not run.exists()


def test_existing_directory_is_never_reused(tmp_path):
    with pytest.raises(ValueError, match="new run directory"):
        build_plan(RunConfig(run=tmp_path, steps=100), project=PROJECT)


@pytest.mark.parametrize("n_envs,base_instance", [(1, 100), (2, 99), (4, 97)])
def test_last_valid_instance_is_accepted(tmp_path, n_envs, base_instance):
    plan = build_plan(RunConfig(run=tmp_path / "run", steps=100,
                                n_envs=n_envs, base_instance=base_instance), project=PROJECT)
    assert option(plan.command, "--base-instance") == str(base_instance)


def test_dry_run_does_not_construct_supervisor(tmp_path, capsys, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Dry-run attempted to construct a process supervisor")
    monkeypatch.setattr("rate_rl.run_manager.RunSupervisor", forbidden)
    run = tmp_path / "dry_run"
    assert main(["--run", str(run), "--steps", "8192", "--dry-run"], project=PROJECT) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["requested_steps"] == 8192
    assert output["batch_size"] == 2048
    assert output["rollout_samples"] == 4096
    assert not run.exists()


def test_direct_script_dry_run_uses_its_own_checkout(tmp_path):
    run = tmp_path / "run"
    result = subprocess.run([sys.executable, str(PROJECT / "scripts/train_rate_only.py"),
                             "--run", str(run), "--steps", "1", "--dry-run"],
                            cwd=tmp_path, text=True, capture_output=True, check=True)
    output = json.loads(result.stdout)
    assert output["project"] == str(PROJECT)
    assert output["command"][0] == sys.executable
    assert not run.exists()


@pytest.mark.parametrize("status", ["failed", "stopped", "paused_by_user"])
def test_extension_refuses_unsuccessful_source_without_launching(tmp_path, status):
    source, run = tmp_path / "source", tmp_path / "continuation"
    source.mkdir()
    (source / "job.json").write_text(json.dumps({"status": status}))
    result = subprocess.run([sys.executable, str(PROJECT / "scripts/extend_rate_only_exploration.py"),
                             "--source", str(source), "--run", str(run), "--total", "100"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode != 0
    assert "refusing automatic resume" in result.stderr
    assert not run.exists()
    assert json.loads((source / "exploration_extension.json").read_text())["status"] == "failed"
