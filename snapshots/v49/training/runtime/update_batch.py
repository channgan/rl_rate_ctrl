from pathlib import Path
import os,sys,json,copy,hashlib,random
import numpy as np
import torch
from v46_core import RPActor,read_capture,rotation
from v46_offline_scheduler import prepare,binary_bytes,digest,rng_restore
from v49_scheduler import V49Scheduler
from v48_configuration import validate_fresh,configure_optimizer,validate_batch
from effective_rules import enforce_coverage
ROOT=Path(sys.argv[1]);batch=int(sys.argv[2]);p=json.loads((ROOT/'protocol.json').read_text());m=json.loads((ROOT/('batch_'+str(batch)+'_captures.json')).read_text())
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(p['optimizer_seed_base']+batch);np.random.seed(p['optimizer_seed_base']+batch);random.seed(p['optimizer_seed_base']+batch)
assert p['authorized'] and p['learning_rate']==3e-5 and batch in range(4)
validate_batch(ROOT,batch,m,p)
actor=RPActor(m['actor']['npz']);anchor=RPActor(p['initial_actor']['npz']);captures=[];entries=[]
for path,profile in zip(m['results'],p['entry_profiles']):
 path=Path(path);audit=json.loads((path.parent/'combined_audit.json').read_text());assert audit['passed'] and audit['result_sha256']==digest(path)
 q=read_capture(path);assert q['success'] and q['result']['nn_final_audit']['model_id']==m['actor']['model_id'];captures.append(q)
 e=json.loads((path.parent/'v46_entry_snapshot.json').read_text());st=e['initial_state'];entries.append(dict(velocity=st['linear_velocity_ned'],memory=st['nn_audit']['memory_rp']))
assert len(captures)==len(p['entry_profiles'])==7
gate=p['coverage_gate'];coverage=[]
for key,field in [('entry_velocity_minmax_must_cover_development_q05_q95','velocity'),('RP_memory_minmax_must_cover_development_q05_q95','memory')]:
 values=np.asarray([e[field] for e in entries]);passed=bool(np.all(values.min(0)<=gate[key]['q05']) and np.all(values.max(0)>=gate[key]['q95']));coverage.append(dict(field=field,minimum=values.min(0).tolist(),maximum=values.max(0).tolist(),required=gate[key],passed=passed))
(ROOT/('batch_'+str(batch)+'_coverage.json')).write_text(json.dumps(coverage,indent=2))
rules=json.loads(Path(p['effective_rules_file']).read_text());assert digest(p['effective_rules_file'])==p['effective_rules_sha256']
assert rules['memory_coverage']['mode']=='diagnostic_all_batches' and rules['smoothness']['mode']=='global_nonincreasing_adaptive_every_update'
diagnostic=enforce_coverage(coverage,entries,batch,m['actor']['model_id'],[q['result']['seed'] for q in captures])
(ROOT/('batch_'+str(batch)+'_memory_sampling.json')).write_text(json.dumps(diagnostic,indent=2))

guard=np.load(p['guard_corpus']);gx=torch.tensor(guard['obs'],dtype=torch.float32);gdt=torch.tensor(guard['dt'],dtype=torch.float32)
with torch.no_grad():gh=actor.features(gx);gm=actor(gx,gdt);am=anchor(gx,gdt)
d=prepare(actor,captures,p['scenario_groups'])
s=V49Scheduler(actor,d,digest(ROOT/'protocol.json'),gh,gm,anchor,am)
old=None
if batch:
 old=torch.load(ROOT/'training'/('batch_'+str(batch-1))/'recovery.pt',map_location='cpu',weights_only=False)
s.initial_opt=configure_optimizer(s.opt,old['state']['optimizer'] if old else None,p['learning_rate'])
out=ROOT/'training'/('batch_'+str(batch));out.mkdir(parents=True,exist_ok=False)
s.attach(RPActor(m['actor']['npz']),captures,p['scenario_groups'],old['state'].get('objective_state',{}).get('alpha',1.) if old else 1.,out)
if old:
 rng_restore(old['state']['rng']);s.gen.set_state(old['state']['generator'])
 s.initial_rng=copy.deepcopy(old['state']['rng']);s.initial_gen=old['state']['generator'].clone()
 assert all(torch.equal(v,old['state']['actor'][k]) for k,v in actor.state_dict().items())
else:
 assert all(torch.equal(v,anchor.state_dict()[k]) for k,v in actor.state_dict().items())
balance=[];(out/'inputs.json').write_text(json.dumps(dict(batch=batch,actor=m['actor'],results=m['results'],balance=balance,data_hash=s.data_hash,coverage_entries=entries),indent=2))
try:
 for _ in range(s.updates,10):
  if (ROOT/'stop').exists():raise RuntimeError('explicit stop before update')
  balance=s.adapt()
  s.save(out/'pre_update_recovery.pt')
  result=s.step();result['projection_solver']=s.last_projection;result['smooth_objective']=dict(cumulative_multiplier=s.alpha,coefficients=balance['coefficients'],zero_tracking_strata_axes=balance['zero_tracking_strata_axes'],data_hash=s.data_hash);s.save(out/'recovery.pt')
  with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(batch=batch,global_update=batch*10+s.updates,**result))+'\n')
  ledger_path=Path(p['canonical_ledger']);ledger=json.loads(ledger_path.read_text());assert ledger['formal_optimizer_updates']==batch*10+s.updates-1
  ledger['formal_optimizer_updates']=batch*10+s.updates;tmp=ledger_path.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2));os.replace(tmp,ledger_path)
  print(json.dumps(dict(batch=batch,global_update=batch*10+s.updates,KL=result['KL'])),flush=True)
 assert s.updates==10 and s.diagnostics()['accepted'] and s.frozen_ok()
 np.savez(out/'actor.npz',**s.actor.canonical());raw,model_id=binary_bytes(s.actor);(out/'actor.bin').write_bytes(raw)
 (out/'export.json').write_text(json.dumps(dict(npz=str(out/'actor.npz'),binary=str(out/'actor.bin'),model_id=model_id,sha256=digest(out/'actor.bin'),batch=batch,global_updates=(batch+1)*10,promotion_allowed=False),indent=2))
except BaseException as error:
 (out/'failure_details.json').write_text(json.dumps(dict(error=repr(error),updates=s.updates,alpha=s.alpha,balances=s.balance_history,diagnostics=s.proposal_diagnostics),indent=2))
 s.save(out/'interrupted_recovery.pt');raise
