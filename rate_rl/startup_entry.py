"""Single operational gateway over a pinned holder release; no second supervisor."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / 'runtime.supervision.json'

class AdmissionError(RuntimeError):
    pass

def local_path(value):
    value = str(value).replace('\\', '/')
    if os.name != 'nt' and len(value) > 2 and value[1:3] == ':/':
        return Path('/mnt/' + value[0].lower() + value[2:])
    if os.name == 'nt' and value.startswith('/mnt/') and len(value) > 7:
        return Path(value[5].upper() + ':' + value[6:])
    return Path(value)

def read(path):
    try:
        result = json.loads(local_path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise AdmissionError(f'Incomplete/unreadable JSON {path}: {exc!r}') from exc
    if not isinstance(result, dict):raise AdmissionError(f'Expected object: {path}')
    return result

def sha(path):
    digest = hashlib.sha256()
    with local_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):digest.update(block)
    return digest.hexdigest()

def require_standard_training(entry):
    """Retire unsupervised entry points; no environment-variable bypass."""
    raise AdmissionError(f'{entry}: legacy live training is disabled. Use '
        'scripts/supervise_training.py with the approved independent training protocol. '
        'Legacy launches lack the exact internal-budget admission.')

def check(policy, full_dependencies=False):
    if policy.get('schema') != 1:raise AdmissionError('Unsupported startup policy schema')
    errors, evidence = [], {}
    for name, item in policy['evidence'].items():
        try:
            if sha(item['path']) != item['sha256']:raise AdmissionError('evidence hash mismatch')
            evidence[name] = read(item['path'])
        except (AdmissionError, OSError) as exc:errors.append(f'{name}: {exc}')
    runtime = local_path(policy['runtime'])
    try:
        if sha(runtime/'repair_manifest.json') != policy['runtime_manifest_sha256']:
            raise AdmissionError('runtime manifest changed')
        for name, expected in read(runtime/'repair_manifest.json').items():
            if Path(name).name != name or sha(runtime/name) != expected:
                raise AdmissionError(f'runtime module changed: {name}')
    except (AdmissionError, OSError) as exc:errors.append(str(exc))
    ledger = read(policy['ledger'])
    if ledger['native_limit'] != policy['total_limit']:errors.append('ledger limit mismatch')
    if ledger['charged_native'] > ledger['native_limit']:errors.append('budget exceeded')
    if ledger['reservations']:errors.append('existing reservation: reconcile its owner; never start another run')
    if policy.get('enabled') is not True:errors.append('launch authorization disabled')
    if policy.get('training_adapter_qualified') is not True:
        errors.append('public 9-observation/3-torque training adapter is not qualified for this holder release')
    dependencies = None
    if full_dependencies:
        if os.name == 'nt':errors.append('full Linux dependency/link check requires the configured WSL environment')
        else:
            dep = read(runtime/'dependency_manifest.json');changed = []
            for name, expected in dep['sha256'].items():
                try:
                    if sha(name) != expected:changed.append(name)
                except OSError:changed.append(name)
            for name, target in dep['links'].items():
                if str(Path(name).resolve()) != target:changed.append(name)
            dependencies = dict(files=len(dep['sha256']), links=len(dep['links']), changed=changed)
            if changed:errors.append('frozen dependency drift; do not refresh hashes merely to pass')
    return dict(start_allowed=not errors, reasons=errors, ledger_charge=ledger['charged_native'],
        remaining=ledger['native_limit']-ledger['charged_native'], reservations=ledger['reservations'],
        formal_optimizer_updates=ledger['formal_optimizer_updates'], runtime=str(runtime),
        dependencies=dependencies, native_calls=0)

def status(policy):
    result = check(policy);result['runs'] = {}
    for name, path in policy['runs'].items():
        root = local_path(path)
        result['runs'][name] = {key: read(root/filename) if (root/filename).exists() else None
            for key, filename in [('host', 'host_completed.json'), ('holder', 'holder_completed.json')]}
        result['runs'][name]['first_stop'] = (root/'stop').read_text() if (root/'stop').exists() else None
    return result

def checked_root(policy, value):
    root = local_path(value).resolve()
    if root not in {local_path(p).resolve() for p in policy['runs'].values()}:
        raise AdmissionError('Run is outside the reviewed policy scope')
    return root

def stop(policy, value):
    root = checked_root(policy, value)
    if (root/'holder_completed.json').exists():return dict(already_closed=True, changed=False)
    job = read(root/'windows_config.json')['job_id']
    if Path(job).name != job:raise AdmissionError('Invalid job scope')
    for path in [root/'stop', root/job/'stop']:
        if path.parent.exists():
            try:
                with path.open('x', encoding='utf-8') as stream:stream.write('standard gateway cooperative stop requested')
            except FileExistsError:pass
    return dict(stop_requested=True, force_kill=False, must_wait_for_export_and_settlement=True)

def start(policy, policy_path, value):
    if policy.get('adapter_contract') == 'independent_fresh_batch_rp_v1':
        from .independent_batch import start as independent_start
        return independent_start(policy,policy_path,value)
    if policy.get('adapter_contract') in {'zero_nn_clock_regression_v1', 'zero_nn_host3600_once_v1'}:
        return start_fixture(policy, policy_path, value)
    verdict = check(policy)
    if not verdict['start_allowed']:raise AdmissionError('; '.join(verdict['reasons']))
    root = checked_root(policy, value)
    if any((root/name).exists() for name in ['launch_once.json', 'standard_launch_intent.json', 'holder_identity.json', 'stop']):
        raise AdmissionError('Run already used; no retry or existing-directory reuse')
    runtime = local_path(policy['runtime']);grant = read(runtime/'grant.json')
    cfg = read(root/'config.json');plan = read(root/'plan.json')
    if not grant.get('approved') or local_path(grant['root']).resolve() != root:
        raise AdmissionError('Runtime grant does not authorize this exact root')
    if local_path(cfg['ledger']).resolve() != local_path(policy['ledger']).resolve():
        raise AdmissionError('Canonical ledger mismatch')
    cap = sum(job['cap'] for job in plan['jobs'])
    if cap <= 0 or cap > verdict['remaining'] or cap > grant['maximum_additional']:
        raise AdmissionError('Insufficient explicitly approved native budget')
    if policy.get('adapter_contract') != 'qualified_prepared_holder_training_v1':
        raise AdmissionError('No qualified training phase adapter; configuration cannot invent one')
    if os.name != 'nt':raise AdmissionError('Live holder launch requires Windows')
    from ._holder_gateway import launch
    return launch(root, runtime, Path(policy_path).resolve())

def start_fixture(policy, policy_path, value):
    """Explicit zero-NN grant uses the SAME holder launch, never a training bypass."""
    verdict = check(policy)
    permitted = {'both original gates have not passed; training remains blocked',
        'launch authorization disabled',
        'public 9-observation/3-torque training adapter is not qualified for this holder release'}
    if any(reason not in permitted for reason in verdict['reasons']):
        raise AdmissionError('; '.join(verdict['reasons']))
    root = checked_root(policy, value)
    grant = policy.get('zero_nn_grant', {})
    if grant.get('approved') is not True or grant.get('native_calls') != 0:
        raise AdmissionError('Explicit zero-NN-only grant required')
    if str(root) not in {str(local_path(p).resolve()) for p in grant.get('roots', [])}:
        raise AdmissionError('Fixture root not granted')
    if any((root/n).exists() for n in ['standard_launch_intent.json','holder_identity.json','stop']):
        raise AdmissionError('Fixture already used; no retry')
    cfg=read(root/'config.json');plan=read(root/'plan.json');wc=read(root/'windows_config.json')
    if cfg.get('backend') != 'fake' or read(root/'offline_only.json') != {'offline_only':True}:
        raise AdmissionError('Fixture must use fake backend')
    if local_path(cfg['ledger']).resolve() != (root/'ledger.json').resolve() or local_path(cfg['ledger']).resolve() == local_path(policy['ledger']).resolve():
        raise AdmissionError('Fixture requires isolated ledger')
    cap=sum(j['cap'] for j in plan['jobs'])
    hour = policy.get('adapter_contract') == 'zero_nn_host3600_once_v1'
    if len(plan['jobs']) != 1 or not 0 < cap <= grant['maximum_fixture_records']:
        raise AdmissionError('Fixture cap exceeds explicit scope')
    if hour:
        job=plan['jobs'][0]
        if grant.get('once_only_host3600') is not True or len(grant['roots']) != 1:
            raise AdmissionError('Explicit single host3600 authorization required')
        if (wc['host_duration'],wc['host_startup'],wc['host_export_grace'],wc['require_duration']) != (3600,20,30,True):
            raise AdmissionError('Host3600 windows changed')
        if cap != 113897 or cfg['initial_charge'] != 0 or cfg['total_limit'] != 113897 or cfg['max_additional'] != 113897:
            raise AdmissionError('Isolated synthetic budget mismatch')
        if job.get('wall60_schedule') is not True or job['units'] != 100000 or job['tick'] != .05 or job['mode'] != 'normal':
            raise AdmissionError('Original normal/half-write/recovery schedule required')
        if (root/'fixture_clock_fault.json').exists():raise AdmissionError('No extra fault injection allowed')
    elif cap > verdict['remaining'] or wc['host_duration'] > 15 or wc['host_startup'] > 20 or wc['host_export_grace'] > 10:
        raise AdmissionError('Only short fixture windows authorized')
    runtime=local_path(policy['runtime'])
    if local_path(wc['owner_script']).resolve() != (runtime/'matrix_owner.py').resolve() or local_path(wc['linux_root']).resolve()!=root:
        raise AdmissionError('Fixture owner/root mismatch')
    for name,expected in grant['input_sha256'][root.name].items():
        if Path(name).name != name or sha(root/name)!=expected:raise AdmissionError('Fixture input changed: '+name)
    if os.name != 'nt':raise AdmissionError('Real Windows entry required')
    from ._holder_gateway import launch
    return launch(root,runtime,Path(policy_path).resolve())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '_observer':
        from ._holder_gateway import observer
        return observer(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status');sub.add_parser('self-check')
    child = sub.add_parser('check');child.add_argument('--full-dependencies', action='store_true')
    for name in ['start', 'stop']:
        child = sub.add_parser(name);child.add_argument('--run-root', required=True)
    child = sub.add_parser('plan');child.add_argument('arguments', nargs=argparse.REMAINDER)
    if 'plan' in argv:
        args = argv[argv.index('plan')+1:]
        if args[:1] == ['--']:args = args[1:]
        from .launcher import main as plan_main
        return plan_main(args + ([] if '--dry-run' in args else ['--dry-run']), project=PROJECT)
    args = parser.parse_args(argv)
    try:
        policy = read(args.config)
        if policy.get('adapter_contract') == 'independent_fresh_batch_rp_v1' and args.command in ['status','stop']:
            from . import independent_batch
            result = independent_batch.status(policy) if args.command=='status' else independent_batch.stop(policy,args.run_root)
        elif args.command == 'status':result = status(policy)
        elif args.command == 'check':result = check(policy, args.full_dependencies)
        elif args.command == 'stop':result = stop(policy, args.run_root)
        elif args.command == 'start':result = start(policy, args.config, args.run_root)
        else:
            from .startup_selfcheck import run
            result = run(policy)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2 if args.command == 'check' and not result['start_allowed'] else 0
    except (AdmissionError, OSError, KeyError, ValueError) as exc:
        print(json.dumps(dict(started=False, error=str(exc)), ensure_ascii=False), file=sys.stderr)
        return 2

if __name__ == '__main__':raise SystemExit(main())
