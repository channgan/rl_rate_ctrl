"""Cleanup failures never skip another owned process or the instance lock."""
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rate_rl.backend import GazeboPX4Backend, SimulatorError


def empty_backend(tmp_path):
    backend = GazeboPX4Backend.__new__(GazeboPX4Backend)
    backend.processes = []
    backend.files = []
    backend.stream = backend.sock = backend.instance_lock = None
    backend.log_dir = tmp_path
    backend.instance = 99
    backend.episode_dir = None
    backend.config = {'px4': str(tmp_path / 'missing_px4')}
    backend.native_outer = False
    return backend


def test_reset_retries_startup_segfault_with_same_seed(tmp_path):
    backend = empty_backend(tmp_path)
    expected = dict(sim_us=10000)
    backend._reset_once = Mock(side_effect=[SimulatorError('Simulator process exited (-11); inspect episode'), expected])
    assert backend.reset(seed=17) is expected
    assert [call.args for call in backend._reset_once.call_args_list] == [(17,), (17,)]


def test_reset_segfault_retries_are_bounded(tmp_path):
    backend = empty_backend(tmp_path)
    backend._reset_once = Mock(side_effect=SimulatorError('Simulator process exited (-11); inspect episode'))
    with pytest.raises(SimulatorError, match='exited'):
        backend.reset(seed=19)
    assert backend._reset_once.call_count == 3


def test_reset_does_not_retry_unrecognized_process_error(tmp_path):
    backend = empty_backend(tmp_path)
    backend._reset_once = Mock(side_effect=SimulatorError('Simulator process exited (1); inspect episode'))
    with pytest.raises(SimulatorError, match='exited'):
        backend.reset(seed=19)
    assert backend._reset_once.call_count == 1


def test_resource_close_errors_do_not_skip_processes_logs_or_lock(tmp_path, monkeypatch):
    backend = empty_backend(tmp_path)
    first_error = OSError('stream close failed')
    stream = backend.stream = Mock()
    stream.close.side_effect = first_error
    sock = backend.sock = Mock()
    sock.close.side_effect = OSError('socket close failed')
    bad_log, other_log, lock = Mock(), Mock(), Mock()
    bad_log.close.side_effect = OSError('log close failed')
    backend.files = [bad_log, other_log]
    backend.instance_lock = lock
    processes = [Mock(pid=12301), Mock(pid=12302)]
    backend.processes = processes.copy()
    signals = Mock()
    monkeypatch.setattr('rate_rl.backend.os.killpg', signals)

    with pytest.raises(OSError) as caught:
        backend.close()
    assert caught.value is first_error
    assert signals.call_args_list == [((12302, signal.SIGTERM),), ((12301, signal.SIGTERM),)]
    for resource in (stream, sock, bad_log, other_log, lock):
        resource.close.assert_called_once_with()
    for process in processes:
        process.wait.assert_called_once_with(timeout=5)
    assert not backend.processes and not backend.files
    assert backend.stream is backend.sock is backend.instance_lock is None
    backend.close()  # Repeated cleanup does not repeat resource errors.


def test_process_timeout_and_signal_error_do_not_skip_other_process_or_lock(tmp_path, monkeypatch):
    backend = empty_backend(tmp_path)
    timed_out, other = Mock(pid=12301), Mock(pid=12302)
    timed_out.wait.side_effect = [subprocess.TimeoutExpired('px4', 5), 0]
    backend.processes = [timed_out, other]
    lock = backend.instance_lock = Mock()
    seen = []
    first_error = PermissionError('could not signal one process')

    def signal_owned(pid, signum):
        seen.append((pid, signum))
        if (pid, signum) == (12302, signal.SIGTERM):
            raise first_error

    monkeypatch.setattr('rate_rl.backend.os.killpg', signal_owned)
    with pytest.raises(PermissionError) as caught:
        backend.close()
    assert caught.value is first_error
    assert seen == [(12302, signal.SIGTERM), (12301, signal.SIGTERM), (12301, signal.SIGKILL)]
    assert timed_out.wait.call_count == 2
    other.wait.assert_called_once_with(timeout=5)
    lock.close.assert_called_once_with()
    assert backend.instance_lock is None


@pytest.mark.parametrize('failure_point', ['lock', 'mkdir', 'startup'])
def test_reset_cleans_lock_for_failures_before_spawning(tmp_path, monkeypatch, failure_point):
    fcntl = pytest.importorskip('fcntl')
    backend = empty_backend(tmp_path)
    lock = Mock()
    monkeypatch.setattr('rate_rl.backend.open', lambda *args: lock, raising=False)
    flock = Mock()
    monkeypatch.setattr(fcntl, 'flock', flock)
    monkeypatch.setattr('rate_rl.backend.uuid.uuid4', lambda: SimpleNamespace(hex='testepisode0'))
    original_error = OSError(f'{failure_point} failed')
    if failure_point == 'lock':
        flock.side_effect = original_error
    elif failure_point == 'mkdir':
        monkeypatch.setattr(Path, 'mkdir', Mock(side_effect=original_error))
    else:
        monkeypatch.setattr(Path, 'write_text', Mock(side_effect=original_error))
    monkeypatch.setattr('rate_rl.backend.subprocess.Popen', Mock(side_effect=AssertionError('must not spawn')))

    with pytest.raises(OSError) as caught:
        backend._reset_once(seed=17)
    assert caught.value is original_error
    lock.close.assert_called_once_with()
    assert backend.instance_lock is None


def test_reset_preserves_initialization_error_when_cleanup_also_fails(tmp_path, monkeypatch):
    fcntl = pytest.importorskip('fcntl')
    backend = empty_backend(tmp_path)
    lock = Mock()
    cleanup_error = OSError('lock close failed')
    lock.close.side_effect = cleanup_error
    monkeypatch.setattr('rate_rl.backend.open', lambda *args: lock, raising=False)
    monkeypatch.setattr(fcntl, 'flock', Mock())
    monkeypatch.setattr('rate_rl.backend.uuid.uuid4', lambda: SimpleNamespace(hex='testepisode0'))
    original_error = OSError('directory creation failed')
    monkeypatch.setattr(Path, 'mkdir', Mock(side_effect=original_error))

    with pytest.raises(OSError) as caught:
        backend._reset_once(seed=17)
    assert caught.value is original_error
    assert caught.value.__cause__ is cleanup_error
    assert backend.instance_lock is None


def test_existing_instance_lock_failure_keeps_specific_error_and_closes_handle(tmp_path, monkeypatch):
    fcntl = pytest.importorskip('fcntl')
    backend = empty_backend(tmp_path)
    lock = Mock()
    monkeypatch.setattr('rate_rl.backend.open', lambda *args: lock, raising=False)
    monkeypatch.setattr(fcntl, 'flock', Mock(side_effect=BlockingIOError('locked')))
    with pytest.raises(SimulatorError, match='Another training environment owns PX4 instance 99'):
        backend._reset_once()
    lock.close.assert_called_once_with()
    assert backend.instance_lock is None
