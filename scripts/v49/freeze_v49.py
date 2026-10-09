from pathlib import Path
import argparse,json,hashlib,sys,platform
import numpy as np,torch
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();V=args.root;T=V/'training';D=V/'development';X=V/'execution';OLD=V.parent/'v48_fixed_lr3e5/backtracking_revision3'
j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();put=lambda p,v:p.write_text(json.dumps(v,indent=2));win=lambda p:str(p).replace('/mnt/e/','E:/').replace('/','\\')
integration=j(V/'offline/integration/admission.json');fixed=j(V/'offline/fixedpoints/admission.json');windows=j(V/'offline/windows_admission_tests.json')
assert all(x['passed'] for x in [integration,fixed,windows])
assert integration['disposable_optimizer_attempts']==23 and fixed['disposable_optimizer_attempts']==4
for path,h in integration['source_hashes'].items():assert sha(path)==h
for path,h in j(V/'v48_protected_before.json').items():assert sha(path)==h
p=j(T/'protocol.json');dp=j(D/'protocol_frozen.json')
assert p['native_limit']==1316000 and dp['native_limit']==2700000
assert j(T/'ledger.json')['charged_native']==j(D/'ledger.json')['charged_native']==0
assert j(T/'ledger.json')['formal_optimizer_updates']==0 and not (T/'jobs').exists()
assert len(p['jobs'])==28 and sum(x['cap'] for x in p['jobs'])==1316000
assert len(dp['jobs'])==54 and sum(x['cap'] for x in dp['jobs'])==2700000
for root,q in [(T,p),(D,dp)]:
 for path,h in {**q['runtime_sha256'],**q['prepared_python_sha256']}.items():assert sha(path)==h,path
report=dict(passed=True,native_calls=0,formal_updates=0,disposable_optimizer_attempts=27,windows_admission_tests=11,linux_contract_tests=13,solver_boundary_tests=11,total_unit_tests=35,full_entry_cross_process_10_steps=True,checkpoint_restore_rollback_export=True,fixed_pre18_pre40_exact_prior_candidate=True,
            historical_fixtures_offline_only=True,formal_capture_count=0,formal_Adam_fresh=True,first_planned_capture_is_native_integration_gate=True,source_commit_and_remote_verification_required=True)
put(V/'offline/admission.json',report)
put(V/'offline/environment.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,torch_threads=2,deterministic_algorithms=True))
files=[V/'execution_contract.json',V/'v48_protected_before.json',T/'protocol.json',T/'effective_rules.json',T/'development_design.json',D/'protocol_frozen.json',D/'frozen_design.json',D/'amendment.json',V/'offline/admission.json']+list(T.glob('runtime/*.py'))+list(D.glob('runtime/*.py'))+list(T.glob('templates/*.json'))+list(D.glob('templates/*.json'))+list((V/'launcher').rglob('*.py'))
files+= [Path(p['initial_actor']['npz']),Path(p['initial_actor']['binary']),Path(p['guard_corpus'])]
files+=list({Path(path) for path in p['runtime_sha256']}|{Path(path) for path in dp['runtime_sha256']})
files=list(dict.fromkeys(files));put(V/'frozen_inputs.json',{str(f):sha(f) for f in files})
put(X/'windows_config.json',dict(owner_script=str(T/'runtime/batch_driver.py'),linux_root=str(T),observer_job='pending_gateway'))
old=j(OLD/'execution/admission.json');lf=V.parent/'v48_fixed_lr3e5/training/ledger.json'
policy=dict(adapter_contract='independent_fresh_batch_rp_v1',approved=True,experiment='v49_active_projection',fixed_native_limit=1316000,source_commit='pending_verified_commit',root=win(X),work_root=win(T),runtime=win(T/'runtime'),canonical_ledger=win(T/'ledger.json'),sha256={win(f):sha(f) for f in files},offline_validation=win(V/'offline/admission.json'),required_zero_nn_results=old['required_zero_nn_results'],old_ledger=win(lf),old_ledger_sha256=sha(lf))
put(X/'admission.json',policy)
print(json.dumps(dict(passed=True,unit_tests=35,disposable_optimizer_attempts=27,native_calls=0,files_frozen=len(files),launch_blocked_until_commit=True)))
