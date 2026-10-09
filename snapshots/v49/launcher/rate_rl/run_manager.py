"""Own one trainer and its log publishers; never signal another run's processes."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import signal
import subprocess
import threading
import time
from typing import Callable

from .launcher import RunPlan


class StopRequested(Exception):
    def __init__(self, signum: int):
        self.signum = signum


@contextmanager
def stop_signals(enabled: bool = True):
    """Translate supervisor SIGTERM into an orderly trainer shutdown."""
    if not enabled or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}

    def request_stop(signum, _frame):
        raise StopRequested(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_stop)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class RunSupervisor:
    def __init__(self, plan: RunPlan, *, popen: Callable = subprocess.Popen,
                 sleep: Callable[[float], None] = time.sleep,
                 utc_now: Callable[[], str] | None = None, handle_signals: bool = True):
        self.plan = plan
        self.popen = popen
        self.sleep = sleep
        self.utc_now = utc_now or (lambda: datetime.now(timezone.utc).isoformat())
        self.handle_signals = handle_signals
        self.trainer = None
        self.publishers = []
        config = plan.config
        self.job = dict(status="starting", supervisor_pid=os.getpid(), phase=config.phase,
                        requested_steps=config.steps, resume=str(config.resume) if config.resume else None,
                        n_envs=config.n_envs, n_steps=config.per_environment_steps,
                        batch_size=config.batch_size, n_epochs=config.n_epochs,
                        episode_steps=config.episode_steps, command=list(plan.command),
                        automatic_phase_switch=False, publishers={})

    def _save(self) -> None:
        path = self.plan.config.run / "job.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.job, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _start(self, command, log_name: str):
        with (self.plan.config.run / log_name).open("w", encoding="utf-8") as output:
            return self.popen(list(command), cwd=self.plan.project,
                              stdout=output, stderr=subprocess.STDOUT)

    def _publish(self, script: str) -> None:
        process = self._start(self.plan.publisher_command(script), f"{script}.log")
        self.publishers.append(process)
        self.job["publishers"][script] = process.pid
        self._save()

    @staticmethod
    def _stop(process, *, graceful: bool = False) -> None:
        if process.poll() is not None:
            return
        if graceful:
            # SIGINT unwinds train.py's finally block, closing its environment.
            # SIGTERM alone skips that cleanup in the single-environment trainer.
            process.send_signal(signal.SIGINT)
        else:
            process.terminate()
        try:
            process.wait(timeout=30 if graceful else 5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _finish_publishers(self) -> None:
        # Let publishers consume the final event/image, then bound shutdown.
        errors = []
        for publisher in self.publishers:
            try:
                try:
                    publisher.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self._stop(publisher)
            except Exception as error:
                errors.append({"pid": publisher.pid, "error": str(error)})
        if errors:
            self.job["publisher_cleanup_errors"] = errors
            self._save()

    def _stop_trainer(self) -> None:
        if self.trainer is not None:
            try:
                self._stop(self.trainer, graceful=True)
            except Exception as error:
                # Preserve the original interruption/launch error, and still
                # attempt every publisher cleanup in run's finally block.
                self.job["trainer_cleanup_error"] = str(error)

    def run(self) -> int:
        self.plan.config.validate()
        self.plan.config.run.mkdir(parents=True)
        self.job["started_utc"] = self.utc_now()
        self._save()
        try:
            with stop_signals(self.handle_signals):
                self.trainer = self._start(self.plan.command, "training.log")
                self.job.update(status="running", training_pid=self.trainer.pid)
                self._save()
                self._publish("tensorboard_trajectories")
                config_path = self.plan.config.run / "config.json"
                while self.trainer.poll() is None and not config_path.is_file():
                    self.sleep(1)
                if config_path.is_file():
                    self._publish("tensorboard_raw_rewards")
                while self.trainer.poll() is None:
                    self.sleep(1)
                returncode = self.trainer.wait()
                self.job.update(status="completed" if returncode == 0 else "failed", returncode=returncode)
        except (StopRequested, KeyboardInterrupt) as error:
            signum = error.signum if isinstance(error, StopRequested) else signal.SIGINT
            self.job.update(status="stopped", stop_signal=int(signum), returncode=128 + int(signum))
            self._stop_trainer()
        except Exception as error:
            self.job.update(status="failed", returncode=1, error=str(error))
            self._stop_trainer()
            raise
        finally:
            self.job["ended_utc"] = self.utc_now()
            try:
                self._save()
            finally:
                self._finish_publishers()
        return int(self.job["returncode"])
