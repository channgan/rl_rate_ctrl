"""Offline fixture checks and replay of sealed full-chain evidence; never starts a payload."""
import ast
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from .startup_entry import AdmissionError, check, local_path, read, require_standard_training, sha, start

def run(policy):
    if os.name == 'nt':raise AdmissionError('Run complete offline self-check in WSL (fcntl/proc fixtures); no simulator is used')
    before = sha(policy['ledger']);runtime = local_path(policy['runtime']);results = []
    def passed(name, kind='offline_fixture'):
        results.append(dict(name=name, passed=True, kind=kind))
    manifest = read(runtime/'repair_manifest.json')
    assert sha(runtime/'repair_manifest.json') == policy['runtime_manifest_sha256']
    for name, digest in manifest.items():
        assert sha(runtime/name) == digest
        tree = ast.parse((runtime/name).read_text())
        if name != 'test_integration.py':
            assert not any(isinstance(n, ast.ImportFrom) and any(a.name == '*' for a in n.names) for n in ast.walk(tree)), name
    static = read(policy['evidence']['bindings']['path'])
    assert static['passed'] and not static['problems'] and len(static['files']) == 21
    passed('pinned_package_no_wildcard_or_callable_shadow', 'static_and_evidence_replay')
    evidence = read(policy['evidence']['chain']['path']);assert evidence['passed'] == 7
    names = set()
    for case in evidence['cases']:
        assert case['passed'];names.add(case['case']);root = local_path(case['root'])
        holder = read(root/'holder_identity.json');assert not holder['in_observer_job']
        records = [json.loads(line) for p in (root/'module_evidence').glob('*.jsonl') for line in p.read_text().splitlines()]
        for module in ['guardian.py', 'runner.py', 'fake_px4.py']:
            assert any(v['module'] == module and v['sha256'] == manifest[module] for v in records)
        events = {v['event']:v['monotonic'] for v in map(json.loads, (root/'holder_events.jsonl').read_text().splitlines())}
        assert read(root/'host_session.json')['host_seconds'] > events['preflight_exited']
        if case['case'] == 'observer_disappears':
            assert read(root/'observer_exit.json')['returncode'] == 91
            assert read(root/'holder_completed.json')['success'] and read(root/'matrix_exit.json')['returncode'] == 0
    assert names == {'normal','late_ready','owner_dead','read_failure','heartbeat_expired','windows_owner_exit','observer_disappears'}
    passed('actual_guardian_runner_payload_chain_7_cases', 'evidence_replay_no_new_processes')
    passed('preflight_excluded_and_observer_job_disappearance', 'evidence_replay_no_new_processes')
    for item in policy['evidence'].values():assert sha(item['path']) == item['sha256']
    assert read(policy['evidence']['budget']['path'])['passed']
    assert read(policy['evidence']['cap_stop']['path'])['passed'] == 1
    passed('original_cap_stop_export_settlement', 'evidence_replay_no_new_processes')
    assert check(policy)['start_allowed'] is False
    try:start(policy, Path('/nonexistent/policy'), '/nonexistent/run')
    except AdmissionError:pass
    else:raise AssertionError('blocked policy could launch')
    for entry in ['public launcher', 'direct trainer', 'legacy continuation']:
        try:require_standard_training(entry)
        except AdmissionError:pass
        else:raise AssertionError('legacy start bypass')
    passed('current_failed_gate_and_legacy_starts_fail_closed')
    with tempfile.TemporaryDirectory(prefix='drl-startup-offline-') as temporary:
        root = Path(temporary)
        def put(path, obj):path.write_text(json.dumps(obj))
        put(root/'host_session.json', dict(boot='fixture', owner='fixture'))
        stamp = dict(boot='fixture', owner='fixture', host_seconds=10., sequence=1, phase='ACTIVE', clock_fault=None)
        put(root/'host_clock.json', stamp)
        saved_path = sys.path[:];sys.path.insert(0, str(runtime))
        try:
            with patch.dict(os.environ, {'HOST_CLOCK_ROOT':str(root), 'SUPERVISION_METRICS_DIR':''}):
                alias = '_startup_frozen_clock_alias'
                spec = importlib.util.spec_from_file_location(alias, runtime/'host_clock.py')
                clock = importlib.util.module_from_spec(spec);sys.modules[alias] = clock;spec.loader.exec_module(clock)
                assert callable(clock.Deadline) and callable(clock.Bridge)
                machine = clock.Deadline(clock.Stamp(100., 'b', 'o'), duration=3, startup=1, export_grace=1)
                value = machine.poll(clock.Stamp(100.2, 'b', 'o'), ready=False)
                assert value['ready_upper'] is None
                value = machine.poll(clock.Stamp(100.3, 'b', 'o'), ready=True, stop_reason='fixture stop')
                assert value['ready_upper'] is None and value['late_ready']
                passed('alias_module_registered_before_exec')
                passed('explicit_ready_only_and_stop_before_late_ready')
                bridge = importlib.import_module('host_bridge');core = importlib.import_module('core')
                bridge._bridge = None;bridge._last = None
                broken = root/'half.json';broken.write_text('{')
                try:core.consistent_json(broken)
                except core.Pending:pass
                else:raise AssertionError('half JSON accepted')
                try:core.consistent_json(broken, True)
                except core.Invalid:pass
                else:raise AssertionError('invalid final JSON accepted')
                passed('half_json_pending_live_invalid_final')
                owner = importlib.import_module('windows_owner')
                class Log:
                    contention_logged=False;retries=0;max_retry_delay=0.
                    def event(self, *_args, **_kwargs):pass
                log = Log();real_replace = os.replace;attempts = []
                def replace_once(source, destination):
                    attempts.append(1)
                    if len(attempts) == 1:
                        error = PermissionError('injected sharing failure');error.winerror = 5;raise error
                    return real_replace(source, destination)
                with patch.object(owner.os, 'replace', replace_once):owner.atomic(root, 'atomic.json', {'complete':True}, log)
                assert read(root/'atomic.json')['complete'] and log.retries == 1
                passed('winerror5_bounded_atomic_replace_fixture')
                health = importlib.import_module('owner_health');saved = dict(pid=os.getpid(), start='fixture', state='S')
                with patch.object(health.host_bridge, 'snapshot', lambda:stamp), patch.object(health, 'identity', lambda _pid:saved):
                    with patch.object(health, 'consistent_json', lambda _p:{'monotonic':7.}):
                        assert health.evaluate(saved, broken, 2.5, {})['reason'] == 'owner_heartbeat_expired'
                    with patch.object(health, 'consistent_json', side_effect=OSError('injected read')):
                        assert health.evaluate(saved, broken, 2.5, {'last_valid':7.})['reason'] == 'owner_heartbeat_read_failure'
                    with patch.object(health, 'identity', lambda _pid:None):
                        assert health.evaluate(saved, broken, 2.5, {})['reason'] == 'owner_dead'
                passed('owner_dead_expired_read_failure_distinct')
                adapter = importlib.import_module('adapter');grant = read(runtime/'grant.json')
                job = dict(id='offline_budget', cap=113897, actor='A', profile='candidate', case='fixture', seed=1)
                put(root/'plan.json', {'jobs':[job]});put(root/'authorization.json', grant);put(root/'offline_only.json', {'offline_only':True})
                ledger = read(policy['ledger']);ledger.update(charged_native=1109747, attempts=[], reservations={})
                put(root/'ledger.json', ledger)
                put(root/'config.json', dict(backend='fake', authorization_probe=True, ledger=str(root/'ledger.json'), initial_charge=1109747, max_additional=113897, total_limit=1223644))
                put(root/'preflight_receipt.json', dict(passed=True, backend='fake', native_authorized=False, root=str(root), plan_sha256=sha(root/'plan.json'), config_sha256=sha(root/'config.json'), repair_manifest_sha256=sha(runtime/'repair_manifest.json'), package=manifest, synthetic_unit_receipt=True))
                ledger_api = adapter.Ledger(root)
                try:ledger_api.reserve(dict(job, cap=113898))
                except AssertionError:pass
                else:raise AssertionError('over-budget reservation accepted')
                ledger_api.reserve(job);(root/job['id']).mkdir();adapter.output(root, job).mkdir(parents=True)
                final = adapter.final_result(root, job, 'synthetic missing capture', 1)
                assert not final['counts_valid'] and final['charge'] == 113897 and final['actual_NN'] == 0
                ledger_api.settle(job, final);digest = sha(root/'ledger.json');ledger_api.settle(job, final)
                assert sha(root/'ledger.json') == digest and read(root/'ledger.json')['charged_native'] == 1223644
                try:ledger_api.reserve(job)
                except AssertionError:pass
                else:raise AssertionError('retry accepted')
                passed('budget_unknown_counts_conservative_idempotent_no_retry')
                with patch.object(bridge, 'snapshot', side_effect=OSError(22, 'injected invalid argument')):
                    bridge._last = 10.;assert bridge.host_now() == 10. and bridge.stopping()
                assert 'invalid argument' in (root/'stop').read_text().lower()
                passed('host_clock_errno22_fail_closed_no_invented_time')
        finally:sys.path[:] = saved_path
    assert sha(policy['ledger']) == before
    return dict(passed=True, checks=results, native_calls=0, new_payload_processes=0, new_long_tests=0,
                canonical_ledger_unchanged=True, training_enabled=False,
                limitation='Offline fixtures and evidence replay do not certify a repaired one-hour run or a training adapter.')
