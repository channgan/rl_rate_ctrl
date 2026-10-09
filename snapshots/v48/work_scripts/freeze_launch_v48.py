from pathlib import Path
import json,hashlib,shutil,ast
B=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002');N=B/'native_fresh_batch_rp_v47';O=B/'v48_lr_offline_admission1';OLD=O/'prepared_training';V=B/'v48_fixed_lr3e5';T=V/'training';D=V/'development';X=V/'execution'
j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):p.write_text(json.dumps(v,indent=2))
def rep(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def win(p):return str(p).replace('/mnt/e/','E:/').replace('/','\\')
V.mkdir(exist_ok=False);shutil.copytree(OLD,T);D.mkdir();X.mkdir()
# Preserve immutable admission and all V47 ledger hashes.
shutil.copytree(O,V/'offline_evidence',ignore=shutil.ignore_patterns('prepared_training'))
original={str(p):sha(p) for p in [N/'ledger.json',N/'development_ledger.json']};put(V/'v47_ledgers_before.json',original)
p=json.loads((T/'protocol.json').read_text().replace(str(OLD),str(T)));p.update(authorized=True,version='v48-fixed-lr3e5-fresh28',authority='2026-10-09 delegated explicit authorization: fresh A/Adam,28 captures,40updates,final54; no retry')
p['learning_rate']=p['learning']['actor_lr']=3e-5;p['canonical_ledger']=str(T/'ledger.json')
p['objective_schedule_frozen']=dict(first_updates=[1,10],first_alpha=1,adaptive_from_update=11,alpha_initial=1,alpha_rule='min(previous, .099*min(track/smooth), 1); finite; zero-track=>global0; remeasure<=.1',not_inherit_v47_final_alpha=True,memory_diagnostic_from_update=21,first_two_batches_original_memory_gate=True,initial_coverage_proof='this run batch_0_coverage.json; not inherited')
put(T/'protocol.json',p)
# Development is prepared now; only candidate identity is bound after validated update40.
shutil.copytree(N/'development_recovery1/runtime',D/'runtime');shutil.copytree(N/'development_recovery1/templates',D/'templates')
dp=j(N/'development_recovery1/protocol.json');dp.pop('development_continuation',None);dp.update(authorized=True,canonical_ledger=str(D/'ledger.json'),training_ledger=str(T/'ledger.json'),training_complete=str(T/'training_complete.json'),sync_runtime_id='v48-qualified-v47-sync2',candidate_binding='only verified final40, pending',promotion_allowed=False)
design=j(T/'development_design.json');dp['jobs']=[dict(id=f"dev_{i+1:02d}",phase='development',case=x['case'],seed=x['seed'],label=x['label'],cap=56000 if x['label']=='pid' else 47000) for i,x in enumerate(design['jobs'])]
dp['actors']={k:p['initial_actor'] for k in ['pid','A']};dp['actor_sha256']={k:sha(Path(v['binary'])) for k,v in dp['actors'].items()};put(D/'protocol_frozen.json',dp);put(D/'frozen_design.json',design);put(D/'amendment.json',dict(runtime='qualified v47 sync2 unchanged',new_development_seeds=[480301,480302],frozen_at_start=True,original_metrics_unchanged=True))
put(D/'ledger.json',dict(authorized=True,native_limit=3000000,charged_native=0,formal_optimizer_updates=0,attempts=[],reservations={},stopped=False))
f=D/'runtime/batch_driver.py';s=f.read_text();s=rep(s,"and len(ledger['attempts'])==43 and ledger['charged_native']==1957486","and not ledger['attempts'] and ledger['charged_native']==0")
a=s.index("assert not protocol['promotion_allowed']");b=s.index("for p,h in protocol['runtime_sha256'].items()",a)
s=s[:a]+"assert not protocol['promotion_allowed'] and len(protocol['jobs'])==54\n"+s[b:]
s=rep(s,"completed=carried['completed'].copy()","completed=[]");s=rep(s,"len(ledger['attempts'])==55","len(ledger['attempts'])==54")
f.write_text(s)
f=D/'runtime/review54.py';s=f.read_text();s=rep(s,'seeds=[470301,470302]','seeds=[480301,480302]');f.write_text(s)
# Predeclared full-model validation and candidate binding, no selection.
final='''from pathlib import Path
import json,sys,hashlib,torch,numpy as np
T=Path(sys.argv[1]);D=T.parent/'development';sys.path.insert(0,str(T/'runtime'))
from v46_core import RPActor
from v46_offline_scheduler import binary_bytes
j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
done=j(T/'training_complete.json');assert done['completed'] and done['updates']==40
rows=[json.loads(line) for b in range(4) for line in (T/'training'/f'batch_{b}'/'updates.jsonl').read_text().splitlines()]
assert [r['global_update'] for r in rows]==list(range(1,41)) and all(r['accepted'] for r in rows)
actor=RPActor(done['actor']['npz']);A=RPActor(j(T/'protocol.json')['initial_actor']['npz']);cp=torch.load(T/'training/batch_3/recovery.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(v,cp['state']['actor'][k]) for k,v in actor.state_dict().items())
assert all(torch.equal(v,A.state_dict()[k]) for k,v in actor.named_buffers())
raw,mid=binary_bytes(actor);assert raw==Path(done['actor']['binary']).read_bytes() and mid==done['actor']['model_id']
assert sha(Path(done['actor']['binary']))==done['actor']['sha256']
ledger=j(T/'ledger.json');assert len(ledger['attempts'])==28 and all(a['status']=='audited' for a in ledger['attempts']) and not ledger['reservations'] and ledger['formal_optimizer_updates']==40
alphas=[r['smooth_objective']['cumulative_multiplier'] for r in rows];assert alphas[:10]==[1.]*10 and all(y<=x for x,y in zip(alphas,alphas[1:]))
for path,h in j(T.parent/'v47_ledgers_before.json').items():assert sha(Path(path))==h
report=dict(passed=True,updates=40,model_id=mid,sha256=sha(Path(done['actor']['binary'])),all_frozen_buffers_equal=True,new_native_calls=0)
(T/'final_model_verification.json').write_text(json.dumps(report,indent=2))
p=j(D/'protocol_frozen.json');p['actors']['candidate']=done['actor'];p['actor_sha256']['candidate']=report['sha256'];p['candidate_binding']='verified final40 only'
with (D/'protocol.json').open('x') as f:json.dump(p,f,indent=2)
print(json.dumps(report))
'''
(T/'runtime/finalize_and_bind.py').write_text(final)
f=T/'runtime/batch_driver.py';s=f.read_text();needle=" atomic(ROOT/'training_complete.json',dict(completed=True,actor=actor,updates=40,promotion_allowed=False,development_required=True));status(state='training_complete_not_promoted',updates=40,charged=ledger['charged_native'])"
addition=needle+"\n assert subprocess.run([sys.executable,'-B',str(R/'finalize_and_bind.py'),str(ROOT)]).returncode==0, 'Final model verification failed'\n dev=ROOT.parent/'development'\n status(state='development_running',updates=40,charged=ledger['charged_native'])\n with (dev/'driver.log').open('x') as stream:completed_dev=subprocess.run([sys.executable,'-B',str(dev/'runtime/batch_driver.py'),str(dev)],stdout=stream,stderr=subprocess.STDOUT)\n assert completed_dev.returncode==0, 'Development stopped; inspect separate ledger and failure'\n status(state='training_and_development_complete_not_promoted',updates=40,charged=ledger['charged_native'])"
s=rep(s,needle,addition);f.write_text(s)
# Entry check points to copied admission evidence and remains usable after authorization.
f=T/'entry.py';s=f.read_text().replace("R.parent/'admission.json'","R.parent/'offline_evidence/admission.json'");f.write_text(s)
p['prepared_python_sha256']={str(f):sha(f) for f in (T/'runtime').glob('*.py')};put(T/'protocol.json',p)
for f in list((T/'runtime').glob('*.py'))+list((D/'runtime').glob('*.py'))+[T/'entry.py']:ast.parse(f.read_text())
files=[T/'protocol.json',D/'protocol_frozen.json',D/'frozen_design.json',D/'amendment.json',T/'development_design.json',V/'offline_evidence/admission.json']+list((T/'runtime').glob('*.py'))+list((D/'runtime').glob('*.py'))+list((T/'templates').glob('*.json'))+list((D/'templates').glob('*.json'))
put(V/'frozen_inputs.json',{str(f):sha(f) for f in files})
put(X/'windows_config.json',dict(owner_script=str(T/'runtime/batch_driver.py'),linux_root=str(T),observer_job='pending_gateway'))
old=j(N/'execution_v47_memory1/admission.json')
ad=dict(adapter_contract='independent_fresh_batch_rp_v1',approved=True,root=win(X),work_root=win(T),runtime=win(T/'runtime'),canonical_ledger=win(T/'ledger.json'),sha256={win(f):sha(f) for f in files},offline_validation=win(V/'offline_evidence/admission.json'),required_zero_nn_results=old['required_zero_nn_results'],old_ledger=win(N/'ledger.json'),old_ledger_sha256=sha(N/'ledger.json'))
put(X/'admission.json',ad)
put(V/'execution_contract.json',dict(approved=True,training_limit=2000000,development_limit=3000000,training_captures=28,updates=40,development_episodes=54,first_capture_in_plan=True,no_retry=True,holdout_unused=[12701,12702],single_owner='Windows breakaway holder -> one locked training driver -> serial native child; after final verification one locked dev driver',cleanup='cooperative stop -> strict CADENCE_EXPORT when connected -> owned child cleanup -> verify no live owned process -> settle; uncertain count charged full reservation, unresolved ownership keeps reservation and blocks',objective_schedule=p['objective_schedule_frozen']))
print(json.dumps(dict(root=str(V),files_frozen=len(files),training_jobs=len(p['jobs']),development_jobs=len(dp['jobs']),native_calls=0)))
