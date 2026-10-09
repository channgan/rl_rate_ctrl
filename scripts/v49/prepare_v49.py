from pathlib import Path
import json,shutil,hashlib,ast
import argparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--scripts',type=Path,required=True);ap.add_argument('--project',type=Path,required=True);args=ap.parse_args()
 B=args.base;C=args.scripts;P=args.project;V48=B/'v48_fixed_lr3e5';OLD=V48/'backtracking_revision3';V=B/'v49_active_projection';T=V/'training';D=V/'development';X=V/'execution'
 j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();put=lambda p,v:p.write_text(json.dumps(v,indent=2))
 def replace(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
 V.mkdir(exist_ok=False);T.mkdir();D.mkdir();X.mkdir();(V/'offline').mkdir()
 protected=[V48/'training/ledger.json',V48/'development/ledger.json',OLD/'revision_contract.json',OLD/'frozen_inputs.json']+list((OLD/'training/runtime').glob('*.py'))+list((OLD/'development/runtime').glob('*.py'))+list((OLD/'training/training').glob('batch_*/*'))
 put(V/'v48_protected_before.json',{str(p):sha(p) for p in protected if p.is_file()})
 for dst,src in [(T,OLD/'training'),(D,OLD/'development')]:
  shutil.copytree(src/'runtime',dst/'runtime');shutil.copytree(src/'templates',dst/'templates')
 for name in ['bias_projection.py','bias_projection_active.py','v49_scheduler.py']:shutil.copyfile(C/name,T/'runtime'/name)
 p=j(OLD/'training/protocol.json')
 for key in ['resume_global_update','resume_step17','remaining_native_captures','historical_charge','additional_native_cap','maximum_new_captures']:p.pop(key,None)
 p.update(version='v49-active-projection',authorized=True,authority='Explicit delegated V49 execution approval; sole algorithm change active-set cost/bias intersection projection against V48 final effective rules',native_limit=1316000,development_limit=2700000,canonical_ledger=str(T/'ledger.json'),sync_runtime_id='v49-active-projection-same-sync2',historical_charge=0,maximum_new_captures=28,additional_native_cap=1316000,optimizer_seed_base=470901,sampler_initial_seed=460901)
 p['jobs']=[dict(id=f'train_b{b}_p{i}',phase='training',batch=b,profile_index=i,case=profile['case'],seed=490101+b*1000+i,cap=47000,label='A' if b==0 else 'candidate') for b in range(4) for i,profile in enumerate(p['entry_profiles'])]
 p['development'].update(seeds=[490301,490302],total_reserved=2700000,unused_reserve=0)
 p['learning'].pop('projection_passes',None);p['learning']['backtracking']=[2.**-i for i in range(13)]
 p['learning']['projection']=dict(method='feasible_active_set_SVD',max_steps=512,primal_stationarity_dual_complementarity_tolerance=1e-12,cast_normalized_tolerance=1e-10,strict_bias_limit=1e-4-1e-10)
 p['coverage_gate']['on_missing_coverage']='Velocity/history/physical failures stop V49. Historical memory quantiles diagnostic in every batch; finite/bounds remain hard. No replacement seeds or retry.'
 p['coverage_gate'].pop('all_five_max',None);p['coverage_gate']['all_seven_max']=143360
 rules=j(OLD/'training/effective_rules.json');rules.update(version='v49-final-effective-rules',training_native_limit=1316000,development_native_limit=2700000,projection=p['learning']['projection']);rules['development']['seeds']=[490301,490302]
 put(T/'effective_rules.json',rules);p.update(effective_rules_file=str(T/'effective_rules.json'),effective_rules_sha256=sha(T/'effective_rules.json'))
 design=j(OLD/'training/development_design.json')
 # Preserve the complete case/controller order, replacing only the declared seed pair.
 design=json.loads(json.dumps(design).replace('480301','490301').replace('480302','490302'))
 put(T/'development_design.json',design);put(D/'frozen_design.json',design);p['development_design_sha256']=sha(T/'development_design.json')
 for f in [T/'runtime/batch_driver.py',T/'runtime/runtime_support.py']:
  s=f.read_text().replace('==2000000','==1316000');f.write_text(s)
 f=T/'runtime/batch_driver.py';s=f.read_text();start=s.index("resume=read(ROOT/'runtime_replacement.json')");end=s.index("for p,h in protocol['runtime_sha256'].items()",start)
 s=s[:start]+"assert ledger['charged_native']==0 and ledger['formal_optimizer_updates']==0 and not ledger['attempts']\n"+s[end:]
 s=replace(s," actor=read(ROOT/'training/batch_0/export.json')\n for batch in range(1,4):"," actor=protocol['initial_actor']\n for batch in range(4):")
 s=replace(s,"paths=read(ROOT/'carried_inputs.json')['manifests']['1']['results'] if batch==1 else [capture(job,actor) for job in protocol['jobs'] if job['batch']==batch]","paths=[capture(job,actor) for job in protocol['jobs'] if job['batch']==batch]")
 s=replace(s,"  entry.update(status='audited',actual_native_NN=", "  if job['id']=='train_b0_p0':atomic(ROOT/'first_planned_capture_verification.json',dict(passed=True,job=job['id'],seed=job['seed'],model_id=actor['model_id'],steps=2048,native_calls=count,full_capture_audit=True,sync_audit=sync,extra_probe=False))\n  entry.update(status='audited',actual_native_NN=")
 f.write_text(s)
 f=T/'runtime/v48_configuration.py';s=f.read_text();s=s[:s.index('def validate_batch(')]+"def validate_batch(root,batch,manifest,protocol):\n assert batch in range(4)\n validate_fresh(root,batch,manifest,protocol)\n";s=s.replace('opt.load_state_dict(previous)','opt.load_state_dict(copy.deepcopy(previous))');f.write_text(s)
 f=T/'runtime/v46_offline_scheduler.py';s=f.read_text();s=replace(s,'delta=project_halfspaces(proposed,gradients,64)','delta=self.project_proposal(proposed,gradients)');f.write_text(s)
 f=T/'runtime/update_batch.py';s=f.read_text();s=s.replace('from adaptive_smooth import AdaptiveScheduler as Scheduler','from v49_scheduler import V49Scheduler')
 start=s.index('class FreshScheduler(');end=s.index('old=None',start)
 s=s[:start]+"s=V49Scheduler(actor,d,digest(ROOT/'protocol.json'),gh,gm,anchor,am)\n"+s[end:]
 s=replace(s,"out=ROOT/'training'/('batch_'+str(batch));resuming=(batch==1 and p.get('resume_global_update')==17);out.mkdir(parents=True,exist_ok=resuming)","out=ROOT/'training'/('batch_'+str(batch));out.mkdir(parents=True,exist_ok=False)")
 start=s.index('if resuming:');end=s.index('balance=[];',start);s=s[:start]+s[end:]
 s=s.replace('torch.manual_seed(470901+batch);np.random.seed(470901+batch);random.seed(470901+batch)',"torch.manual_seed(p['optimizer_seed_base']+batch);np.random.seed(p['optimizer_seed_base']+batch);random.seed(p['optimizer_seed_base']+batch)")
 s=replace(s,"s.save(out/'pre_update_recovery.pt')","s.save(out/'pre_update_recovery.pt')")
 s=s.replace("result=s.step();result['smooth_objective']", "result=s.step();result['projection_solver']=s.last_projection;result['smooth_objective']")
 f.write_text(s)
 f=T/'runtime/finalize_and_bind.py';s=f.read_text();s=s.replace("T.parent/'v47_ledgers_before.json'","T.parent/'v48_protected_before.json'")
 s=s.replace("all(r['accepted'] for r in rows)","all(r['accepted'] and r['parameter_delta_max']>=1e-10 and r['action_delta_previous_max']>0 and r['projection_solver']['passed'] for r in rows)")
 s=s.replace("and ledger['formal_optimizer_updates']==40","and ledger['formal_optimizer_updates']==40 and ledger['charged_native']<=1316000 and ledger['native_limit']==1316000")
 f.write_text(s)
 dp=j(OLD/'development/protocol_frozen.json');dp.update(authorized=True,native_limit=2700000,canonical_ledger=str(D/'ledger.json'),training_ledger=str(T/'ledger.json'),training_complete=str(T/'training_complete.json'),sync_runtime_id=p['sync_runtime_id'],candidate_binding='only verified final40 pending',promotion_allowed=False)
 dp['jobs']=[dict(id=f'dev_{i+1:02d}',phase='development',case=x['case'],seed=x['seed'],label=x['label'],cap=56000 if x['label']=='pid' else 47000) for i,x in enumerate(design['jobs'])]
 dp['actors']={key:p['initial_actor'] for key in ['pid','A']};dp['actor_sha256']={key:sha(Path(actor['binary'])) for key,actor in dp['actors'].items()}
 for f in [D/'runtime/batch_driver.py',D/'runtime/runtime_support.py']:f.write_text(f.read_text().replace('==3000000','==2700000'))
 f=D/'runtime/review54.py';f.write_text(f.read_text().replace('480301','490301').replace('480302','490302'))
 # Final rules are the baseline; no inherited update11/21 activation fields.
 assert p['learning_rate']==3e-5 and sum(x['cap'] for x in p['jobs'])==1316000
 assert sum(x['cap'] for x in dp['jobs'])==2700000 and len(dp['jobs'])==54
 for root,limit in [(T,1316000),(D,2700000)]:put(root/'ledger.json',dict(authorized=True,native_limit=limit,charged_native=0,formal_optimizer_updates=0,attempts=[],reservations={},stopped=False))
 put(D/'amendment.json',dict(version='V49',only_algorithm_change='cost and original bias intersection active-set SVD',same_sync2_native_and_locked_PID=True,development_seeds=[490301,490302],frozen_before_training=True,original_metrics_and_first3_unchanged=True))
 p['prepared_python_sha256']={str(f):sha(f) for f in (T/'runtime').glob('*.py')};dp['prepared_python_sha256']={str(f):sha(f) for f in (D/'runtime').glob('*.py')}
 put(T/'protocol.json',p);put(D/'protocol_frozen.json',dp)
 contract=dict(approved=True,version='V49',algorithm_baseline='V48 final effective rules, not historical11/21 switches',single_algorithm_change='deterministic feasible active-set SVD projection into original cost/bias intersection',start='A2428576135 plus fresh Adam',learning_rate=3e-5,trainable_RP_parameters=154,training_limit=1316000,development_limit=2700000,training_captures=28,updates=40,development_episodes=54,first_capture_in_plan=True,extra_probes=0,no_retry=True,no_worst_case_retry_margin=True,all_warmup_probe_failure_calls_charged=True,no_implicit_budget_extension=True,final40_only=True,holdout_unused=[12701,12702],effective_rules=rules)
 put(V/'execution_contract.json',contract)
 # Materialize the established public entry in a new isolated project; original worktree untouched.
 L=V/'launcher';(L/'scripts').mkdir(parents=True);shutil.copytree(P/'rate_rl',L/'rate_rl',ignore=shutil.ignore_patterns('__pycache__','*.pyc'));shutil.copyfile(P/'scripts/supervise_training.py',L/'scripts/supervise_training.py')
 f=L/'rate_rl/independent_batch.py';s=f.read_text();s=replace(s,"expected_limit=3000000 if policy.get('development_final40') else 2000000","expected_limit=3000000 if policy.get('development_final40') else 2000000\n if policy.get('experiment')=='v49_active_projection':\n  if policy.get('development_final40') or protocol.get('version')!='v49-active-projection':raise AdmissionError('V49 exact training entry required')\n  if policy.get('fixed_native_limit')!=1316000:raise AdmissionError('V49 fixed budget mismatch')\n  expected_limit=1316000\n  proof=read(root/'source_provenance.json')\n  if not proof.get('remote_verified') or proof['commit']!=policy.get('source_commit'):raise AdmissionError('V49 committed remote source proof required')\n  for name,digest in proof['live_files_sha256'].items():\n   if sha(local_path(name))!=digest:raise AdmissionError('V49 source changed after commit: '+name)")
 f.write_text(s)
 for file in list(T.glob('runtime/*.py'))+list(D.glob('runtime/*.py'))+list(L.rglob('*.py')):ast.parse(file.read_text())
 assert all(sha(Path(path))==h for path,h in j(V/'v48_protected_before.json').items())
 put(V/'preparation.json',dict(prepared=True,native_calls=0,formal_updates=0,root=str(V),source_baseline_commit='0d299fc56d38f96c6ab4856ac0a8446c4c3dd59b',offline_admission_pending=True,remote_commit_required_before_launch=True))
 print(json.dumps(dict(prepared=True,root=str(V),native_calls=0,training_cap=1316000,development_cap=2700000)))

if __name__=='__main__':main()
