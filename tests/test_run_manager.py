"""Lifecycle checks use fake child processes; no PX4 or Gazebo is started."""
import json
from pathlib import Path
import signal
import subprocess

import pytest

from rate_rl.launcher import RunConfig, build_plan
from rate_rl.run_manager import RunSupervisor, StopRequested, stop_signals


class FakeProcess:
    def __init__(self, pid, returncode=None, stubborn=False):
        self.pid = pid
        self.returncode = returncode
        self.stubborn = stubborn
        self.signals = []
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake child", timeout)
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)
        if not self.stubborn:
            self.returncode = -int(signum)

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


def make_plan(tmp_path):
    return build_plan(RunConfig(run=tmp_path / "run", steps=4096), project=tmp_path)


@pytest.mark.parametrize("returncode,expected", [(0, "completed"), (17, "failed")])
def test_job_tracks_exit_and_only_starts_raw_publisher_after_config(tmp_path, returncode, expected):
    plan = make_plan(tmp_path)
    children = [FakeProcess(101), FakeProcess(102, returncode=0), FakeProcess(103, returncode=0)]
    commands = []

    def spawn(command, **kwargs):
        assert kwargs["cwd"] == plan.project
        commands.append(command)
        return children[len(commands) - 1]

    sleeps = 0
    def advance(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 1:
            (plan.config.run / "config.json").write_text("{}")
        else:
            children[0].returncode = returncode

    supervisor = RunSupervisor(plan, popen=spawn, sleep=advance, utc_now=lambda: "test-time", handle_signals=False)
    assert supervisor.run() == returncode
    job = json.loads((plan.config.run / "job.json").read_text())
    assert job["status"] == expected
    assert job["training_pid"] == 101
    assert job["publishers"] == {"tensorboard_trajectories": 102, "tensorboard_raw_rewards": 103}
    assert job["n_steps"] == 4096 and job["episode_steps"] == 2048
    assert job["command"] == commands[0] == list(plan.command)
    assert job["ended_utc"] == "test-time"
    assert not any(child.killed or child.terminated for child in children)


def test_trainer_failure_before_config_never_starts_raw_publisher(tmp_path):
    plan = make_plan(tmp_path)
    children = [FakeProcess(101, returncode=3), FakeProcess(102, returncode=0)]
    commands = []
    def spawn(command, **_kwargs):
        commands.append(command)
        return children[len(commands) - 1]
    supervisor = RunSupervisor(plan, popen=spawn, handle_signals=False)
    assert supervisor.run() == 3
    assert len(commands) == 2
    assert "tensorboard_raw_rewards" not in supervisor.job["publishers"]


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_stop_unwinds_trainer_and_stops_only_owned_publishers(tmp_path, signum):
    plan = make_plan(tmp_path)
    trainer, publisher = FakeProcess(101), FakeProcess(102)
    children = iter([trainer, publisher])
    def request_stop(_seconds):
        raise StopRequested(signum)
    supervisor = RunSupervisor(plan, popen=lambda *_args, **_kwargs: next(children),
                               sleep=request_stop, handle_signals=False)
    assert supervisor.run() == 128 + int(signum)
    assert trainer.signals == [signal.SIGINT]
    assert publisher.terminated
    assert not trainer.killed and not publisher.killed
    job = json.loads((plan.config.run / "job.json").read_text())
    assert job["status"] == "stopped"
    assert job["stop_signal"] == int(signum)


def test_failed_publisher_start_stops_trainer_and_records_failure(tmp_path):
    plan = make_plan(tmp_path)
    trainer = FakeProcess(101)
    def spawn(command, **_kwargs):
        if "rate_rl.train" in command:
            return trainer
        raise OSError("publisher launch failed")
    supervisor = RunSupervisor(plan, popen=spawn, handle_signals=False)
    with pytest.raises(OSError, match="publisher launch failed"):
        supervisor.run()
    assert trainer.signals == [signal.SIGINT]
    assert supervisor.job["status"] == "failed"
    assert supervisor.job["error"] == "publisher launch failed"


def test_nonresponsive_child_is_killed_after_bounded_grace_period():
    child = FakeProcess(101, stubborn=True)
    RunSupervisor._stop(child, graceful=True)
    assert child.signals == [signal.SIGINT]
    assert child.killed


def test_failed_trainer_cleanup_does_not_skip_owned_publishers(tmp_path, monkeypatch):
    plan = make_plan(tmp_path)
    trainer, publisher = FakeProcess(101), FakeProcess(102)
    children = iter([trainer, publisher])
    def request_stop(_seconds):
        raise StopRequested(signal.SIGTERM)
    def failed_signal(_signum):
        raise OSError("cannot signal trainer")
    monkeypatch.setattr(trainer, "send_signal", failed_signal)
    supervisor = RunSupervisor(plan, popen=lambda *_args, **_kwargs: next(children),
                               sleep=request_stop, handle_signals=False)
    assert supervisor.run() == 143
    assert publisher.terminated
    job = json.loads((plan.config.run / "job.json").read_text())
    assert job["trainer_cleanup_error"] == "cannot signal trainer"
    assert job["status"] == "stopped"


def test_failed_publisher_cleanup_does_not_skip_next_publisher(tmp_path, monkeypatch):
    plan = make_plan(tmp_path)
    supervisor = RunSupervisor(plan, handle_signals=False)
    plan.config.run.mkdir()
    first, second = FakeProcess(101), FakeProcess(102)
    def failed_terminate():
        raise OSError("cannot terminate first publisher")
    monkeypatch.setattr(first, "terminate", failed_terminate)
    supervisor.publishers = [first, second]
    supervisor.job["status"] = "stopped"
    supervisor._finish_publishers()
    assert second.terminated
    assert supervisor.job["publisher_cleanup_errors"] == [
        {"pid": 101, "error": "cannot terminate first publisher"}]


def test_signal_handlers_are_restored(monkeypatch):
    handlers = {signal.SIGINT: object(), signal.SIGTERM: object()}
    initial = handlers.copy()
    def install(signum, handler):
        old = handlers[signum]
        handlers[signum] = handler
        return old
    monkeypatch.setattr(signal, "signal", install)
    with pytest.raises(StopRequested):
        with stop_signals():
            handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert handlers == initial
