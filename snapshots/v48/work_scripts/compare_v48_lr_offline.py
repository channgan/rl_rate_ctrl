from pathlib import Path
import sys,json,hashlib,copy
import numpy as np
import torch
N=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/native_fresh_batch_rp_v47');M=N/'memory_sampling_revision1'
O=N.parent/'v48_lr_offline_admission1';O.mkdir(exist_ok=False)
sys.path.insert(0,str(M/'runtime'))
import v46_offline_scheduler as sched
from adaptive_smooth import AdaptiveScheduler
from v46_core import RPActor,read_capture
j=lambda p:json.loads(p.read_text());torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
plan=dict(steps=[20,30,40],learning_rates=[3e-6,3e-5],minimum_guard_action_rms_ratio=5,native_calls=0,formal_updates=0,holdout_used=False,source='Historical behavior captures for one-step diagnostics only; not new online data')
(O/'plan.json').write_text(json.dumps(plan,indent=2));reports=[]
for b,root in [(1,N/'smooth_objective_revision1'),(2,M),(3,M)]:
 p=j(root/'protocol.json');manifest=j(root/f'batch_{b}_captures.json');anchor=RPActor(p['initial_actor']['npz']);q=[read_capture(f) for f in manifest['results']]
 guard=np.load(p['guard_corpus']);gx=torch.tensor(guard['obs'],dtype=torch.float32);gdt=torch.tensor(guard['dt'],dtype=torch.float32)
 with torch.no_grad():am=anchor(gx,gdt)
 cp=root/'training'/f'batch_{b}'/'pre_update_recovery.pt';saved=torch.load(cp,map_location='cpu',weights_only=False)
 pair=[]
 for lr in plan['learning_rates']:
  actor=RPActor(manifest['actor']['npz']);data=sched.prepare(actor,q,p['scenario_groups'])
  with torch.no_grad():gh=actor.features(gx);gm=actor(gx,gdt)
  class S(AdaptiveScheduler):
   @torch.no_grad()
   def diagnostics(self):
    r=super().diagnostics();r['global_A_weight_relative']=float(((self.actor.rp_w-anchor.rp_w).norm(dim=1)/anchor.rp_w.norm(dim=1)).max());r['global_A_bias_change']=float(abs(self.actor.rp_b-anchor.rp_b).max());mu=torch.clamp(torch.nn.functional.linear(gh,self.actor.rp_w,self.actor.rp_b),-1,1);r['global_A_guard_action_change']=float(abs(mu-am[:,:2]).max());r['accepted']=r['accepted'] and r['global_A_weight_relative']<=.01 and r['global_A_bias_change']<=1e-4 and r['global_A_guard_action_change']<=.002;return r
  s=S(actor,data,sched.digest(root/'protocol.json'),gh,gm);s.attach(RPActor(manifest['actor']['npz']),q,p['scenario_groups']);s.rebuild(saved['state']['objective_state']['alpha']);s.resume(cp)
  before=s.snapshot();v=sched.flat(actor).clone()
  with torch.no_grad():initial_out=actor(gx,gdt).clone()
  record=dict(step=(b+1)*10,lr=lr,start_actor_hash=sched.fingerprint(actor.state_dict()),data_hash=s.data_hash,checkpoint_sha=sched.digest(cp),alpha=s.alpha)
  for group in s.opt.param_groups:group['lr']=lr
  orig_indices=s.indices
  def indices():
   ix=orig_indices();record['minibatch_sha256']=hashlib.sha256(ix.numpy().tobytes()+s.minibatch_weights.numpy().tobytes()).hexdigest();return ix
  s.indices=indices
  orig_project=sched.project_halfspaces
  def project(delta,gradients,passes):
   z=orig_project(delta,gradients,passes);record.update(adam_proposed_norm=float(delta.norm()),projected_norm=float(z.norm()),projection_retained=float(z.norm()/delta.norm()));return z
  sched.project_halfspaces=project
  try:
   result=s.step();record['result']=result;record['passed_original_guards']=bool(result['accepted'])
  except Exception as e:record.update(error=repr(e),passed_original_guards=False)
  finally:sched.project_halfspaces=orig_project
  with torch.no_grad():diff=actor(gx,gdt)-initial_out
  record.update(step_parameter_L2=float((sched.flat(actor)-v).norm()),guard_action_RP_rms=float(torch.sqrt(torch.mean(diff[:,:2]**2))),guard_action_axis_rms=torch.sqrt(torch.mean(diff**2,dim=0)).tolist(),guard_action_max=float(diff.abs().max()),frozen_buffers_equal=all(torch.equal(x,before['actor'][k]) for k,x in actor.named_buffers()),finite_outputs=bool(torch.isfinite(diff).all()))
  if lr==3e-6:
   expected=torch.load(root/'training'/f'batch_{b}'/'recovery.pt',map_location='cpu',weights_only=False);record['original_actor_exact']=all(torch.equal(v,expected['state']['actor'][k]) for k,v in actor.state_dict().items())
  pair.append(record)
  (O/f'step_{(b+1)*10}_lr_{lr}.json').write_text(json.dumps(record,indent=2))
  print(json.dumps(record),flush=True)
 assert pair[0]['start_actor_hash']==pair[1]['start_actor_hash'] and pair[0]['data_hash']==pair[1]['data_hash'] and pair[0]['minibatch_sha256']==pair[1]['minibatch_sha256']
 ratio=pair[1]['guard_action_RP_rms']/pair[0]['guard_action_RP_rms']
 reports.append(dict(step=(b+1)*10,pairs=pair,action_rms_ratio=ratio,passed=all(x['passed_original_guards'] and x['finite_outputs'] and x['frozen_buffers_equal'] for x in pair) and pair[0]['original_actor_exact'] and ratio>=5))
final=dict(plan=plan,comparisons=reports,passed=all(r['passed'] for r in reports),disposable_optimizer_steps=6,native_calls=0,formal_updates=0)
(O/'admission.json').write_text(json.dumps(final,indent=2));print('FINAL',json.dumps(dict(passed=final['passed'],ratios=[r['action_rms_ratio'] for r in reports])),flush=True)
