from pathlib import Path
import sys,json,copy,hashlib,traceback
import numpy as np,torch
V=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');PRE=V/'smooth_start_revision1/training';R=V/'memory_gate_revision2';T=R/'training';O=R/'offline';O.mkdir(exist_ok=False)
sys.path.insert(0,str(T/'runtime'))
from v46_core import RPActor,read_capture
import v46_offline_scheduler as core
from adaptive_smooth import AdaptiveScheduler
from effective_rules import enforce_coverage
from v48_configuration import validate_batch,configure_optimizer
j=lambda p:json.loads(p.read_text());put=lambda p,v:p.write_text(json.dumps(v,indent=2));torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
report=dict(passed=False,native_calls=0,formal_updates=0,disposable_optimizer_steps=0)
try:
 p=j(T/'protocol.json');m=j(T/'batch_1_captures.json');carry=j(T/'carried_inputs.json');validate_batch(T,1,m,p)
 assert len(set(carry['manifests']['0']['results']+carry['manifests']['1']['results']))==14
 for name in ['v46_core.py','v46_offline_scheduler.py','adaptive_smooth.py','native_case.py','runtime_support.py','sync_trace_audit.py']:assert core.digest(T/'runtime'/name)==core.digest(PRE/'runtime'/name)
 oldpath=PRE/'training/batch_0/recovery.pt';oldsha=core.digest(oldpath);old=torch.load(oldpath,map_location='cpu',weights_only=False)
 assert core.digest(T/'training/batch_0/recovery.pt')==oldsha and old['state']['updates']==10
 assert [json.loads(x)['global_update'] for x in (T/'training/batch_0/updates.jsonl').read_text().splitlines()]==list(range(1,11))
 actor=RPActor(m['actor']['npz']);A=RPActor(p['initial_actor']['npz']);assert all(torch.equal(v,old['state']['actor'][k]) for k,v in actor.state_dict().items())
 qs=[read_capture(x) for x in m['results']];assert len(qs)==7 and all(q['success'] and q['result']['nn_final_audit']['model_id']==m['actor']['model_id'] for q in qs)
 entries=[dict(velocity=j(Path(x).parent/'v46_entry_snapshot.json')['initial_state']['linear_velocity_ned'],memory=j(Path(x).parent/'v46_entry_snapshot.json')['initial_state']['nn_audit']['memory_rp']) for x in m['results']]
 coverage=j(PRE/'batch_1_coverage.json');diag=enforce_coverage(coverage,entries,1,m['actor']['model_id'],[q['result']['seed'] for q in qs]);assert diag['original_gate_would_fail'];put(O/'memory_diagnostic.json',diag)
 checks=0
 for b in range(4):assert enforce_coverage(coverage,entries,b,m['actor']['model_id'],[1]*7)['diagnostic_only'];checks+=1
 bad=copy.deepcopy(coverage);bad[0]['passed']=False
 try:enforce_coverage(bad,entries,1,1,[1]*7)
 except AssertionError:checks+=1
 else:raise AssertionError('velocity bypass')
 for value in [float('nan'),float('inf'),1.001,-1.001]:
  bad=copy.deepcopy(entries);bad[0]['memory'][0]=value
  try:enforce_coverage(coverage,bad,1,1,[1]*7)
  except AssertionError:checks+=1
  else:raise AssertionError('finite/bound bypass')
 report.update(rule_regression_checks=checks,source14_disjoint=True,model_exact_step10=True,copied_updates_exact_1_to_10=True,original_algorithms_and_guards_unchanged=True,memory_diagnostic=diag)
 g=np.load(p['guard_corpus']);gx=torch.tensor(g['obs'],dtype=torch.float32);dt=torch.tensor(g['dt'],dtype=torch.float32)
 with torch.no_grad():gh=actor.features(gx);gm=actor(gx,dt);am=A(gx,dt)
 data=core.prepare(actor,qs,p['scenario_groups'])
 class S(AdaptiveScheduler):
  @torch.no_grad()
  def diagnostics(self):
   r=super().diagnostics();r['global_A_weight_relative']=float(((self.actor.rp_w-A.rp_w).norm(dim=1)/A.rp_w.norm(dim=1)).max());r['global_A_bias_change']=float(abs(self.actor.rp_b-A.rp_b).max());mu=torch.clamp(torch.nn.functional.linear(gh,self.actor.rp_w,self.actor.rp_b),-1,1);r['global_A_guard_action_change']=float(abs(mu-am[:,:2]).max());r['accepted']=r['accepted'] and r['global_A_weight_relative']<=.01 and r['global_A_bias_change']<=1e-4 and r['global_A_guard_action_change']<=.002;return r
 s=S(actor,data,core.digest(T/'protocol.json'),gh,gm);s.initial_opt=configure_optimizer(s.opt,old['state']['optimizer'],p['learning_rate']);s.attach(RPActor(m['actor']['npz']),qs,p['scenario_groups'],old['state']['objective_state']['alpha']);core.rng_restore(old['state']['rng']);s.gen.set_state(old['state']['generator']);s.initial_rng=copy.deepcopy(old['state']['rng']);s.initial_gen=old['state']['generator'].clone()
 assert torch.equal(torch.get_rng_state(),old['state']['rng']['torch']) and torch.equal(s.gen.get_state(),old['state']['generator'])
 assert s.updates==0 and not s.perms and not s.positions
 for k,val in old['state']['optimizer']['state'].items():
  for key,v in val.items():assert torch.equal(v,s.opt.state_dict()['state'][k][key])
 assert [float(x['step']) for x in s.opt.state_dict()['state'].values()]==[10.,10.]
 before={k:v.clone() for k,v in s.data.items()};balance=s.adapt();assert all(torch.equal(v,s.data[k]) for k,v in before.items() if k not in ['adv','track_adv','smooth_adv'])
 s.save(O/'global11_pre_update.pt');results=[];batches=[];original=s.indices
 def indices():
  z=original();batches.append(hashlib.sha256(z.numpy().tobytes()+s.minibatch_weights.numpy().tobytes()).hexdigest());return z
 s.indices=indices
 result=s.step();report['disposable_optimizer_steps']+=1;expected=copy.deepcopy(s.snapshot());assert s.updates==1
 s.resume(O/'global11_pre_update.pt');r2=s.step();report['disposable_optimizer_steps']+=1
 assert all(torch.equal(v,expected['actor'][k]) for k,v in actor.state_dict().items()) and batches[0]==batches[1]
 for k,val in expected['optimizer']['state'].items():
  for key,v in val.items():assert torch.equal(v,s.opt.state_dict()['state'][k][key])
 assert all(torch.equal(v,A.state_dict()[k]) for k,v in actor.named_buffers())
 assert core.digest(oldpath)==oldsha and j(V/'training/ledger.json')==j(R/'original_ledger_stopped.json')
 report.update(passed=True,local_update1_means_global11=True,Adam_steps_after_probe=[float(x['step']) for x in s.opt.state_dict()['state'].values()],actual_lr=s.opt.param_groups[0]['lr'],alpha=balance['cumulative_multiplier'],first_proposed_update=result,recovery_actor_optimizer_minibatch_exact=True,all28_balance_pass=all(x['ratio']<=.1 for x in balance['actual']),frozen_buffers_equal_A=True,old_checkpoint_ledger_unchanged=True)
except BaseException as e:
 report.update(error=repr(e),traceback=traceback.format_exc());put(O/'admission.json',report);print(json.dumps(report,indent=2),flush=True);raise
put(O/'admission.json',report);print(json.dumps(report,indent=2),flush=True)
