from pathlib import Path
import json,hashlib,shutil,ast
V=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');PRE=V/'smooth_start_revision1';OLD=PRE/'training';R=V/'memory_gate_revision2';T=R/'training';D=R/'development';X=R/'execution'
j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();put=lambda p,v:p.write_text(json.dumps(v,indent=2))
def rep(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
l=j(V/'training/ledger.json');assert l['stopped'] and l['charged_native']==608641 and l['formal_optimizer_updates']==10 and len(l['attempts'])==14 and not l['reservations']
R.mkdir(exist_ok=False);T.mkdir();D.mkdir();X.mkdir();put(R/'original_ledger_stopped.json',l);put(R/'original_failure.json',j(OLD/'failure.json'));put(R/'original_status.json',j(OLD/'status.json'));shutil.copyfile(V/'v47_ledgers_before.json',R/'v47_ledgers_before.json')
for target,src in [(T,OLD),(D,PRE/'development')]:
 shutil.copytree(src/'runtime',target/'runtime');shutil.copytree(src/'templates',target/'templates')
for name in ['frozen_design.json','amendment.json']:shutil.copyfile(PRE/'development'/name,D/name)
shutil.copyfile(OLD/'development_design.json',T/'development_design.json');(T/'training').mkdir();shutil.copytree(OLD/'training/batch_0',T/'training/batch_0')
manifests={str(b):j(OLD/f'batch_{b}_captures.json') for b in [0,1]};files=[]
for m in manifests.values():
 for result in m['results']:
  d=Path(result).parent;files += [f for f in d.iterdir() if f.is_file() and f.suffix in ['.bin','.json']]
files += list((OLD/'training/batch_0').glob('*'));files=[f for f in files if f.is_file()]
carry=dict(authorized=True,manifests=manifests,sha256={str(f):sha(f) for f in files},paid_charge=608641,completed_updates=10,copy_batch0_not_reexecute=True)
put(T/'carried_inputs.json',carry)
for b,m in manifests.items():put(T/f'batch_{b}_captures.json',m)
rules=dict(schema=1,version='v48-effective-rules-memory2',learning_rate=3e-5,trainable_RP_parameters=154,smoothness=dict(mode='global_nonincreasing_adaptive_every_update',initial_alpha=1.,target_ratio=.099,hard_ratio=.1,zero_tracking='disable_global_smoothness',nonfinite='stop',rebuild='reward_returns_group_baseline_common_normalization'),memory_coverage=dict(mode='diagnostic_all_batches',historical_q05_q95_never_eligibility=True,record_original_gate_would_fail=True,finite_and_abs_le_one_hard=True,recurrence_and_128history_and_10ms_lookback_hard=True),velocity_coverage='unchanged_hard_q05_q95',phases=dict(first3_seconds=3,first3_sampling_mass=.5,later_sampling_mass=.5),batching=dict(batches=4,captures_per_batch=7,updates_per_batch=10,minibatch=2000,carry_actor_Adam_RNG_between_batches=True,new_permutations_for_each_new_behavior_dataset=True),total_updates=40,only_candidate='final_update40',training_native_limit=2000000,development_native_limit=3000000,development=dict(episodes=54,seeds=[480301,480302],holdout_sealed=[12701,12702],original_metrics_and_gates_unchanged=True),native_and_PID_unchanged=True,physical_scoring_unchanged=True,failure_penalty_raw_per_second=4000)
put(T/'effective_rules.json',rules)
p=j(OLD/'protocol.json')
for key in ['training_objective_revision','memory_sampling_revision','objective_schedule_frozen','revision_diff','preparation_notes']:p.pop(key,None)
p.update(version='v48-memory-gate-revision2',authority='Explicit user-authorized timeless final-rule contract; historical memory quantiles diagnostic in all batches; preserve every other hard guard.',sync_runtime_id='v48-memory2-same-sync2',effective_rules_file=str(T/'effective_rules.json'),effective_rules_sha256=sha(T/'effective_rules.json'),canonical_ledger=str(V/'training/ledger.json'),remaining_native_captures=14,resume_global_update=10)
helper='''from pathlib import Path
import json,numpy as np
def enforce_coverage(coverage,entries,batch,model_id,seeds):
 memory=np.asarray([e['memory'] for e in entries]);velocity=np.asarray([e['velocity'] for e in entries])
 assert np.isfinite(memory).all() and np.max(abs(memory))<=1, 'memory finite/bound guard failed'
 assert np.isfinite(velocity).all(), 'velocity finite guard failed'
 m=next(x for x in coverage if x['field']=='memory');v=next(x for x in coverage if x['field']=='velocity')
 diagnostic=dict(batch=batch,model_id=model_id,seeds=seeds,range_min=m['minimum'],range_max=m['maximum'],lower_tail_gap=np.maximum(0,np.asarray(m['minimum'])-m['required']['q05']).tolist(),upper_tail_gap=np.maximum(0,np.asarray(m['required']['q95'])-m['maximum']).tolist(),original_gate_would_fail=not m['passed'],diagnostic_only=True,no_cumulative_current_policy_coverage_claim=True)
 assert v['passed'], 'original velocity coverage gate failed'
 return diagnostic
'''
(T/'runtime/effective_rules.py').write_text(helper)
f=T/'runtime/update_batch.py';s=f.read_text();s=rep(s,'from v48_configuration import validate_fresh,configure_optimizer,validate_batch','from v48_configuration import validate_fresh,configure_optimizer,validate_batch\nfrom effective_rules import enforce_coverage')
a=s.index("revision=p.get('memory_sampling_revision'");b=s.index("guard=np.load",a)
s=s[:a]+"rules=json.loads(Path(p['effective_rules_file']).read_text());assert digest(p['effective_rules_file'])==p['effective_rules_sha256']\nassert rules['memory_coverage']['mode']=='diagnostic_all_batches' and rules['smoothness']['mode']=='global_nonincreasing_adaptive_every_update'\ndiagnostic=enforce_coverage(coverage,entries,batch,m['actor']['model_id'],[q['result']['seed'] for q in captures])\n(ROOT/('batch_'+str(batch)+'_memory_sampling.json')).write_text(json.dumps(diagnostic,indent=2))\n\n"+s[b:]
s=rep(s," rng_restore(old['state']['rng']);s.gen.set_state(old['state']['generator'])"," rng_restore(old['state']['rng']);s.gen.set_state(old['state']['generator'])\n s.initial_rng=copy.deepcopy(old['state']['rng']);s.initial_gen=old['state']['generator'].clone()")
f.write_text(s)
f=T/'runtime/v48_configuration.py';s=f.read_text();a=s.index('def validate_batch(');s=s[:a]+'''def validate_batch(root,batch,manifest,protocol):
 import hashlib
 root=Path(root)
 if batch>=2:validate_fresh(root,batch,manifest,protocol);return
 carry=json.loads((root/'carried_inputs.json').read_text())
 assert carry['authorized'] and carry['paid_charge']==608641 and carry['completed_updates']==10
 assert manifest==carry['manifests'][str(batch)] and len(manifest['results'])==len(set(manifest['results']))==7
 for path,h in carry['sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
''';f.write_text(s)
f=T/'runtime/batch_driver.py';s=f.read_text();s=rep(s,"resume['historical_charge']==303638 and ledger['formal_optimizer_updates']==0 and len(ledger['attempts'])==7","resume['historical_charge']==608641 and ledger['formal_optimizer_updates']==10 and len(ledger['attempts'])==14")
s=rep(s,"read(ROOT/'carried_batch0.json')['sha256']","read(ROOT/'carried_inputs.json')['sha256']")
s=rep(s," actor=protocol['initial_actor']\n for batch in range(4):"," actor=read(ROOT/'training/batch_0/export.json')\n for batch in range(1,4):")
s=rep(s,"paths=read(ROOT/'carried_batch0.json')['manifest']['results'] if batch==0 else","paths=read(ROOT/'carried_inputs.json')['manifests']['1']['results'] if batch==1 else")
f.write_text(s)
resume=dict(approved=True,historical_charge=608641,historical_optimizer_updates=10,historical_attempt_count=14,historical_attempts_sha256=hashlib.sha256(json.dumps(l['attempts'],sort_keys=True).encode()).hexdigest(),reason='historical patch chronology removed; memory quantiles always diagnostic; exact step10 continuation')
put(T/'runtime_replacement.json',resume)
p['prepared_python_sha256']={str(f):sha(f) for f in (T/'runtime').glob('*.py')};put(T/'protocol.json',p)
dp=j(PRE/'development/protocol_frozen.json');dp.update(training_complete=str(T/'training_complete.json'),sync_runtime_id=p['sync_runtime_id']);put(D/'protocol_frozen.json',dp)
audit=dict(historical_activation_rules_removed=['smooth adaptation start11 (already removed by revision1)','memory diagnostic start21/batch>=2 and initial coverage prerequisite'],genuine_algorithm_phases_retained=['first3s versus later scoring/sampling','4x7 captures,10updates each,2000 weighted samples','fresh optimizer at experiment start; carry actor/Adam/RNG between batches, reset dataset permutations','final40 only candidate;54development after model verification'],implementation_guards=['scheduler per-instance updates>=40 defensive bound remains; driver enforces10 per batch','batch>=2 in provenance means only two exact carried datasets allowed, not reward/memory stage'],historical_chronology_explanation='I incorrectly treated v47 patch introduction times as algorithm phases to preserve its historical sequence; final rule semantics should be independent of that chronology.',strict_LR_single_variable=False,physical_comparison_valid=True,original_contracts_preserved=True)
put(R/'inherited_rule_audit.json',audit)
for f in list((T/'runtime').glob('*.py'))+list((D/'runtime').glob('*.py')):ast.parse(f.read_text())
assert 'from_global_update' not in (T/'runtime/update_batch.py').read_text() and 'if batch>=2' not in (T/'runtime/update_batch.py').read_text()
print(json.dumps(dict(prepared=True,old_ledger_unchanged=True,carried_captures=14,remaining_captures=14,maximum_native_charge=608641+14*47000)))
