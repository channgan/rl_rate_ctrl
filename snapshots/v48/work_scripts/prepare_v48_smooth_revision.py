from pathlib import Path
import json,hashlib,shutil,ast
V=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');OLD=V/'training';R=V/'smooth_start_revision1';T=R/'training';D=R/'development';X=R/'execution'
j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();put=lambda p,v:p.write_text(json.dumps(v,indent=2));win=lambda p:str(p).replace('/mnt/e/','E:/').replace('/','\\')
def rep(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
assert j(R/'offline/admission.json')['passed'];l=j(OLD/'ledger.json');assert l['stopped'] and l['charged_native']==303638 and l['formal_optimizer_updates']==0 and len(l['attempts'])==7 and not l['reservations']
put(R/'original_ledger_stopped.json',l);put(R/'original_failure.json',j(OLD/'failure.json'));put(R/'original_status.json',j(OLD/'status.json'));shutil.copyfile(V/'v47_ledgers_before.json',R/'v47_ledgers_before.json')
T.mkdir();D.mkdir();X.mkdir();shutil.copytree(OLD/'runtime',T/'runtime');shutil.copytree(OLD/'templates',T/'templates');shutil.copytree(V/'development/runtime',D/'runtime');shutil.copytree(V/'development/templates',D/'templates')
for name in ['frozen_design.json','amendment.json']:shutil.copyfile(V/'development'/name,D/name)
shutil.copyfile(OLD/'development_design.json',T/'development_design.json')
manifest=j(OLD/'batch_0_captures.json');put(T/'batch_0_captures.json',manifest)
paths=[]
for result in manifest['results']:
 d=Path(result).parent;paths.extend([f for f in d.iterdir() if f.is_file() and (f.suffix in ['.bin','.json'])]);paths.append(d.parents[2]/'attempt.json')
paths=[p for p in paths if p.exists()];carried=dict(manifest=manifest,sha256={str(p):sha(p) for p in paths},authorized_reuse_only=True,original_charge=303638,original_optimizer_steps=0)
put(T/'carried_batch0.json',carried)
p=j(OLD/'protocol.json');p.update(version='v48-smooth-start-revision1',authority='Explicit revision authorized: adaptive global smoothness starts at update1; preserve original failure/303638 charge, reuse exact7 A captures; LR3e-5; unchanged budgets/acceptance.',canonical_ledger=str(OLD/'ledger.json'),sync_runtime_id='v48-smooth-start1-same-sync2')
p['training_objective_revision']['start_global_update']=1;p['training_objective_revision']['first_ten_old_objective']=False;p['objective_schedule_frozen'].update(first_updates=[],first_alpha=1,adaptive_from_update=1)
p['revision_diff']=dict(original_protocol=str(OLD/'protocol.json'),original_protocol_sha256=sha(OLD/'protocol.json'),only_new_change='Move same global smooth adaptation from update11 to update1; not a strictly LR-only comparison with original V47 process',unchanged=['4000/s','all physical/scoring/PID/native sync gates','memory hard first2 batches,diagnostic from21','28total captures/40updates','final40-only54dev','all original10%/KL/ESS/projection/parameter guards'],new_native_captures=21,carried_native_captures=7)
p['preparation_notes']=['Explicit smooth-start timing revision from update1; original alpha1 failure retained.','Restart A/empty Adam at3e-5 and reuse exact current-run A batch0; no new capture0.','Remaining21 captures keep original frozen seeds/profiles and fresh behavior actor.','No promotion/holdout/extra samples/retry; all prior charge retained.']
# Make the sanctioned carry the only exception to fresh-root provenance.
f=T/'runtime/update_batch.py';s=f.read_text();s=rep(s,"assert len(m['results'])==7 and all(Path(v).resolve().is_relative_to((ROOT/'jobs').resolve()) for v in m['results']), 'Historical/external captures rejected'","assert len(m['results'])==7\nif batch==0:\n carried=json.loads((ROOT/'carried_batch0.json').read_text());assert m==carried['manifest']\n for path,h in carried['sha256'].items():assert digest(path)==h,path\nelse:\n assert all(Path(v).resolve().is_relative_to((ROOT/'jobs').resolve()) for v in m['results']), 'Historical/external captures rejected'")
s=rep(s,'validate_fresh(ROOT,batch,m,p)','if batch:validate_fresh(ROOT,batch,m,p)')
s=rep(s,"  if batch==0:\n   records=s.balance();balance=dict(coefficients=dict(legacy_torque=2/70,extra_RP_torque=.05,ESC=.05),zero_tracking_strata_axes=[],records=records)\n  else:balance=s.adapt()","  balance=s.adapt()")
f.write_text(s)
# Exact history admission; carry batch0 then collect only batches1..3.
f=T/'runtime/batch_driver.py';s=f.read_text();s=rep(s,"assert ledger['charged_native']==0 and ledger['formal_optimizer_updates']==0 and not ledger['attempts']","resume=read(ROOT/'runtime_replacement.json')\nassert ledger['charged_native']==resume['historical_charge']==303638 and ledger['formal_optimizer_updates']==0 and len(ledger['attempts'])==7\nassert hashlib.sha256(json.dumps(ledger['attempts'],sort_keys=True).encode()).hexdigest()==resume['historical_attempts_sha256']\nfor path,h in read(ROOT/'carried_batch0.json')['sha256'].items():assert sha(path)==h,path")
s=rep(s,"paths=[capture(job,actor) for job in protocol['jobs'] if job['batch']==batch]","paths=read(ROOT/'carried_batch0.json')['manifest']['results'] if batch==0 else [capture(job,actor) for job in protocol['jobs'] if job['batch']==batch]")
s=rep(s,"  with (ROOT/('batch_'+str(batch)+'_update.log')).open('x') as stream:done=", "  status(state='updating',batch=batch,charged=ledger['charged_native'],formal_optimizer_updates=ledger['formal_optimizer_updates'])\n  with (ROOT/('batch_'+str(batch)+'_update.log')).open('x') as stream:done=")
f.write_text(s)
f=T/'runtime/finalize_and_bind.py';s=f.read_text();s=rep(s,"ledger=j(T/'ledger.json')","ledger=j(Path(j(T/'protocol.json')['canonical_ledger']))")
s=rep(s,"assert alphas[:10]==[1.]*10 and all(y<=x for x,y in zip(alphas,alphas[1:]))","assert all(0<=x<=1 for x in alphas) and all(y<=x for x,y in zip(alphas,alphas[1:]))")
f.write_text(s)
history_hash=hashlib.sha256(json.dumps(l['attempts'],sort_keys=True).encode()).hexdigest();resume=dict(approved=True,historical_charge=303638,historical_optimizer_updates=0,historical_attempt_count=7,historical_attempts_sha256=history_hash,reason='smooth start from1, no native runtime change; exact7 audited captures carried')
put(T/'runtime_replacement.json',resume)
p['prepared_python_sha256']={str(f):sha(f) for f in (T/'runtime').glob('*.py')};put(T/'protocol.json',p)
dp=j(V/'development/protocol_frozen.json');dp.update(training_complete=str(T/'training_complete.json'),training_ledger=str(OLD/'ledger.json'),canonical_ledger=str(V/'development/ledger.json'),sync_runtime_id=p['sync_runtime_id']);put(D/'protocol_frozen.json',dp)
contract=dict(approved=True,revision='smooth_start_revision1',strictly_LR_only=False,adaptive_from_update=1,target_ratio=.099,unchanged_hard_ratio=.1,alpha_nonincreasing=True,initial_alpha_before_adaptation=1.,old_alpha1_failure_preserved=True,carried7=carried['manifest']['results'],charge_preserved=303638,remaining_captures=21,total_captures=28,formal_updates=40,LR=3e-5,train_limit=2000000,dev_limit=3000000,final40_only=True,dev54_seeds=[480301,480302],holdout_not_used=[12701,12702],no_extra_samples=True,no_automatic_retry=True,canonical_training_ledger=str(OLD/'ledger.json'),canonical_development_ledger=str(V/'development/ledger.json'))
put(R/'revision_contract.json',contract)
for f in [*T.glob('runtime/*.py'),*D.glob('runtime/*.py')]:ast.parse(f.read_text())
files=[T/'protocol.json',T/'runtime_replacement.json',T/'carried_batch0.json',T/'development_design.json',D/'protocol_frozen.json',D/'frozen_design.json',D/'amendment.json',R/'revision_contract.json',R/'offline/admission.json']+list((T/'runtime').glob('*.py'))+list((D/'runtime').glob('*.py'))+list((T/'templates').glob('*.json'))+list((D/'templates').glob('*.json'))
put(R/'frozen_inputs.json',{str(f):sha(f) for f in files});put(X/'windows_config.json',dict(owner_script=str(T/'runtime/batch_driver.py'),linux_root=str(T),observer_job='pending_gateway'))
old=j(V/'execution/admission.json');ad=dict(old);ad.update(root=win(X),work_root=win(T),runtime=win(T/'runtime'),runtime_replacement=True,canonical_ledger=win(OLD/'ledger.json'),sha256={win(f):sha(f) for f in files},offline_validation=win(R/'offline/admission.json'));put(X/'admission.json',ad)
put(R/'prepared.json',dict(passed=True,original_ledger_unmodified=True,carried_files=len(paths),remaining_native_reservation=21*47000,maximum_training_charge=303638+21*47000,native_calls=0))
print(json.dumps(j(R/'prepared.json')))
