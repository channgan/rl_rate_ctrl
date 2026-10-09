from pathlib import Path
import json,hashlib,shutil,ast
V=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');PRE=V/'memory_gate_revision2';R=V/'backtracking_revision3';OLD=PRE/'training';T=R/'training';D=R/'development';X=R/'execution'
j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();put=lambda p,v:p.write_text(json.dumps(v,indent=2))
def rep(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
l=j(V/'training/ledger.json');assert l['stopped'] and l['charged_native']==608641 and l['formal_optimizer_updates']==17 and len(l['attempts'])==14 and not l['reservations']
assert j(PRE/'offline_backtracking_diagnosis3/report.json')['passed']
for path,h in j(PRE/'frozen_inputs.json').items():assert sha(Path(path))==h
R.mkdir(exist_ok=False);T.mkdir();D.mkdir();X.mkdir();put(R/'original_ledger_stopped.json',l);shutil.copyfile(V/'v47_ledgers_before.json',R/'v47_ledgers_before.json')
for target,src in [(T,OLD),(D,PRE/'development')]:
 shutil.copytree(src/'runtime',target/'runtime');shutil.copytree(src/'templates',target/'templates')
for name in ['frozen_design.json','amendment.json']:shutil.copyfile(PRE/'development'/name,D/name)
for name in ['development_design.json','carried_inputs.json','batch_0_captures.json','batch_1_captures.json']:shutil.copyfile(OLD/name,T/name)
(T/'training').mkdir();shutil.copytree(OLD/'training/batch_0',T/'training/batch_0');(T/'training/batch_1').mkdir();shutil.copyfile(OLD/'training/batch_1/updates.jsonl',T/'training/batch_1/updates.jsonl')
resume_source=OLD/'training/batch_1/pre_update_recovery.pt'
resume=dict(approved=True,source=str(resume_source),sha256=sha(resume_source),source_protocol=str(OLD/'protocol.json'),source_protocol_sha256=sha(OLD/'protocol.json'),source_last_accepted=str(OLD/'training/batch_1/recovery.pt'),source_last_accepted_sha256=sha(OLD/'training/batch_1/recovery.pt'),local_updates=7,global_updates=17,batch_anchor_global_update=10,keep_perms_positions=True)
put(T/'resume_step17.json',resume)
rules=j(OLD/'effective_rules.json');rules.update(version='v48-effective-rules-backtracking3',backtracking=dict(scales=[2.**-i for i in range(13)],minimum_scale=2.**-12,bias_original_limit=1e-4,bias_float64_acceptance=1e-4-1e-10,minimum_parameter_change=1e-10,nonzero_action_required=True,all_original_guards=True,exhaustion='stop_and_restore_batch_start',success_count='accepted_nonzero_only'))
put(T/'effective_rules.json',rules)
p=j(OLD/'protocol.json');p.update(version='v48-backtracking-revision3',sync_runtime_id='v48-backtracking3-same-sync2',effective_rules_file=str(T/'effective_rules.json'),effective_rules_sha256=sha(T/'effective_rules.json'),resume_global_update=17,resume_step17=str(T/'resume_step17.json'))
core=T/'runtime/v46_offline_scheduler.py';s=core.read_text()
s=rep(s,'self.opt.load_state_dict(s[\'optimizer\'])','self.opt.load_state_dict(copy.deepcopy(s[\'optimizer\']))')
s=rep(s,"previous_cost=np.asarray(self.diagnostics()['cost_delta']);before=self.snapshot();v=flat(self.actor);", "entry=self.diagnostics();assert entry['accepted'],'Infeasible starting actor';assert float((self.actor.rp_b.detach().double()-self.initial['rp_b'].double()).abs().max())<=1e-4-1e-10,'Infeasible strict batch bias';previous_cost=np.asarray(entry['cost_delta']);before=self.snapshot();v=flat(self.actor);previous_mean=self.means()[0].detach().clone();")
s=rep(s,'for scale in (1.,.5,.25,.125):','for scale in tuple(2.**-i for i in range(13)):')
s=rep(s,"   if report['accepted']:\n    self.updates", "   strict_bias=float((self.actor.rp_b.detach().double()-self.initial['rp_b'].double()).abs().max());delta_size=float((flat(self.actor).double()-v.double()).abs().max());action_delta=float((self.means()[0].detach().double()-previous_mean.double()).abs().max())\n   report.update(scale=scale,strict_batch_bias=strict_bias,parameter_delta_max=delta_size,action_delta_previous_max=action_delta)\n   report['accepted']=bool(report['accepted'] and strict_bias<=1e-4-1e-10 and delta_size>=1e-10 and action_delta>0)\n   if getattr(self,'folder',None):\n    with (self.folder/'proposals.jsonl').open('a') as stream:stream.write(json.dumps(dict(before_local_update=self.updates+1,**report))+'\\n')\n   if report['accepted']:\n    self.updates")
s=rep(s,'All four proposal scales failed; restored A, version stopped','All 13 fixed proposal scales rejected; restored batch-start actor/optimizer/RNG, version stopped; last accepted checkpoint preserved')
core.write_text(s)
f=T/'runtime/update_batch.py';s=f.read_text();s=rep(s,"  return report\ns=FreshScheduler", "  report['strict_A_bias']=float((b.double()-anchor.rp_b.double()).abs().max());report['accepted']=bool(report['accepted'] and report['strict_A_bias']<=1e-4-1e-10)\n  return report\ns=FreshScheduler")
s=rep(s,"out=ROOT/'training'/('batch_'+str(batch));out.mkdir(parents=True,exist_ok=False)","out=ROOT/'training'/('batch_'+str(batch));resuming=(batch==1 and p.get('resume_global_update')==17);out.mkdir(parents=True,exist_ok=resuming)")
needle="balance=[];(out/'inputs.json').write_text"
insert="""if resuming:
 spec=json.loads(Path(p['resume_step17']).read_text());assert digest(spec['source'])==spec['sha256'] and digest(spec['source_protocol'])==spec['source_protocol_sha256'] and digest(spec['source_last_accepted'])==spec['source_last_accepted_sha256']
 q=torch.load(spec['source'],map_location='cpu',weights_only=False);last=torch.load(spec['source_last_accepted'],map_location='cpu',weights_only=False)
 assert q['config_hash']==spec['source_protocol_sha256'] and q['guard_hash']==s.guard_hash
 assert all(torch.equal(v,s.initial[k]) for k,v in q['initial'].items()),'Batch anchor changed'
 assert all(torch.equal(v,last['state']['actor'][k]) for k,v in q['state']['actor'].items())
 s.restore(copy.deepcopy(q['state']));s.attempts=q['attempts'];s.forward_vectors=q['forward_vectors']
 assert s.updates==7 and not s.stopped and [int(v['step']) for v in s.opt.state_dict()['state'].values()]==[17,17]
 assert s.data_hash==q['data_hash'] and s.diagnostics()['accepted']
 assert torch.equal(torch.get_rng_state(),q['state']['rng']['torch']) and torch.equal(s.gen.get_state(),q['state']['generator'])
 assert s.positions==q['state']['positions'] and all(torch.equal(v,q['state']['perms'][k]) for k,v in s.perms.items())
 assert [json.loads(x)['global_update'] for x in (out/'updates.jsonl').read_text().splitlines()]==list(range(11,18))
 (out/'resume_verified.json').write_text(json.dumps(dict(global_update=17,local_update=7,Adam_steps=[17,17],batch_anchor=10,source_sha256=spec['sha256'],rng_and_sample_progress_exact=True,data_hash=s.data_hash),indent=2))
"""
s=rep(s,needle,insert+needle);s=rep(s,' for _ in range(10):',' for _ in range(s.updates,10):');f.write_text(s)
f=T/'runtime/batch_driver.py';s=f.read_text();s=rep(s,"ledger['formal_optimizer_updates']==10 and len(ledger['attempts'])==14","ledger['formal_optimizer_updates']==17 and len(ledger['attempts'])==14");f.write_text(s)
rr=j(OLD/'runtime_replacement.json');rr.update(historical_optimizer_updates=17,reason='Explicit fixed backtracking revision3; exact step17 actor/Adam/RNG/permutations, retain step10 anchor');put(T/'runtime_replacement.json',rr)
p['prepared_python_sha256']={str(f):sha(f) for f in (T/'runtime').glob('*.py')};put(T/'protocol.json',p)
dp=j(PRE/'development/protocol_frozen.json');dp.update(training_complete=str(T/'training_complete.json'),sync_runtime_id=p['sync_runtime_id']);put(D/'protocol_frozen.json',dp)
for f in list((T/'runtime').glob('*.py'))+list((D/'runtime').glob('*.py')):ast.parse(f.read_text())
assert j(V/'training/ledger.json')==l
print(json.dumps(dict(prepared=True,resume=17,batch_anchor=10,native_calls=0,ledger_unchanged=True)))
