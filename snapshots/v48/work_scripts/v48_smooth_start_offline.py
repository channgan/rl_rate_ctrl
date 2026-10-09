from pathlib import Path
import json,sys,copy,hashlib,math
import numpy as np,torch
V=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');T=V/'training';R=V/'smooth_start_revision1';O=R/'offline';O.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(T/'runtime'))
from v46_core import RPActor,read_capture
import v46_offline_scheduler as core
from adaptive_smooth import AdaptiveScheduler,choose_alpha
j=lambda p:json.loads(p.read_text());p=j(T/'protocol.json');m=j(T/'batch_0_captures.json');torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
plan=dict(lrs=[3e-6,3e-5],steps_per_lr=1,recovery_replay_per_lr=1,formal_updates=0,native_calls=0,adaptive_from_update=1,guard=.1,target=.099,source='seven current-run A behavior captures; no recollection')
(O/'plan.json').write_text(json.dumps(plan,indent=2));captures=[read_capture(x) for x in m['results']];A=RPActor(p['initial_actor']['npz']);guard=np.load(p['guard_corpus']);gx=torch.tensor(guard['obs'],dtype=torch.float32);dt=torch.tensor(guard['dt'],dtype=torch.float32)
with torch.no_grad():am=A(gx,dt)
cp=T/'training/batch_0/interrupted_recovery.pt';original_sha=core.digest(cp);reports=[]
def returns(reward,duration):
 out=np.zeros(len(reward));carry=0.
 for i in reversed(range(len(out))):carry=float(reward[i])+math.exp(-float(duration[i])/30)*carry;out[i]=carry
 return out
def independent_adv(alpha,weights):
 rewards=[];rr=[];times=[];reward_error=0.
 for q in captures:
  reward=q['bonus']-q['tracking']-alpha*(q['torque_smooth']+q['motor_smooth'])-q['saturation'];reward=reward.copy();reward[-1]+=q['terminal']
  reward_error=max(reward_error,float(abs(reward-(q['reward']+(1-alpha)*(q['torque_smooth']+q['motor_smooth']))).max()))
  rewards.append(reward);rr.append(returns(reward,q['transition_dt']));times.append((q['sample_us'][:-1]-q['sample_us'][0])*1e-6)
 adv=[];groups=p['scenario_groups']
 for i,t in enumerate(times):
  baselines=[np.mean([np.interp(t,times[k],rr[k]) for k,g in enumerate(groups) if g==other],axis=0) for other in range(5) if other!=groups[i]]
  adv.append(rr[i]-np.mean(baselines,axis=0))
 raw=np.concatenate(adv);mean=np.sum(weights*raw);sd=np.sqrt(np.sum(weights*(raw-mean)**2));return (raw-mean)/sd,reward_error,mean,sd
for lr in plan['lrs']:
 actor=RPActor(m['actor']['npz']);d=core.prepare(actor,captures,p['scenario_groups'])
 with torch.no_grad():gh=actor.features(gx);gm=actor(gx,dt)
 class S(AdaptiveScheduler):
  @torch.no_grad()
  def diagnostics(self):
   r=super().diagnostics();r['global_A_weight_relative']=float(((self.actor.rp_w-A.rp_w).norm(dim=1)/A.rp_w.norm(dim=1)).max());r['global_A_bias_change']=float(abs(self.actor.rp_b-A.rp_b).max());mu=torch.clamp(torch.nn.functional.linear(gh,self.actor.rp_w,self.actor.rp_b),-1,1);r['global_A_guard_action_change']=float(abs(mu-am[:,:2]).max());r['accepted']=r['accepted'] and r['global_A_weight_relative']<=.01 and r['global_A_bias_change']<=1e-4 and r['global_A_guard_action_change']<=.002;return r
 s=S(actor,d,core.digest(T/'protocol.json'),gh,gm);s.attach(RPActor(m['actor']['npz']),captures,p['scenario_groups']);s.resume(cp)
 assert s.updates==0 and not s.opt.state and all(torch.equal(v,A.state_dict()[k]) for k,v in actor.state_dict().items())
 for g in s.opt.param_groups:g['lr']=lr
 before={k:v.clone() for k,v in s.data.items()};balance=s.adapt();assert len(balance['actual'])==28 and max(x['ratio'] for x in balance['actual'])<=.099+2e-7
 assert all(torch.equal(v,s.data[k]) for k,v in before.items() if k not in ['adv','track_adv','smooth_adv'])
 expected,reward_error,mean,sd=independent_adv(s.alpha,s.data['weights'].numpy());adv_error=float(abs(expected-s.data['adv'].numpy()).max());assert reward_error<1e-12 and adv_error<1e-10
 saved=O/f'lr_{lr}_pre_step.pt';s.save(saved);indices=[];original=s.indices
 def ix():
  z=original();indices.append(hashlib.sha256(z.numpy().tobytes()+s.minibatch_weights.numpy().tobytes()).hexdigest());return z
 s.indices=ix
 result=s.step();state=copy.deepcopy(actor.state_dict());opt=copy.deepcopy(s.opt.state_dict())
 with torch.no_grad():delta=actor(gx,dt)-am
 s.resume(saved);replayed=s.step();assert all(torch.equal(v,state[k]) for k,v in actor.state_dict().items()) and indices[0]==indices[1]
 for key,values in opt['state'].items():
  for name,value in values.items():assert torch.equal(value,s.opt.state_dict()['state'][key][name])
 record=dict(lr=lr,alpha=s.alpha,balance=balance,independent_reward_max_error=reward_error,independent_return_baseline_normalization_max_error=adv_error,normalization_mean=mean,normalization_std=sd,unchanged_physical_and_behavior_tensors=True,minibatch_hash=indices[0],optimizer_lr=s.opt.param_groups[0]['lr'],result=result,guard_action_RP_rms=float(torch.sqrt(torch.mean(delta[:,:2]**2))),guard_action_axis_rms=torch.sqrt(torch.mean(delta**2,dim=0)).tolist(),frozen_buffers_equal=all(torch.equal(v,A.state_dict()[k]) for k,v in actor.named_buffers()),recovery_actor_optimizer_minibatch_exact=True)
 reports.append(record);(O/f'lr_{lr}_report.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='balance'}),flush=True)
assert reports[0]['alpha']==reports[1]['alpha'] and reports[0]['minibatch_hash']==reports[1]['minibatch_hash']
ratio=reports[1]['guard_action_RP_rms']/reports[0]['guard_action_RP_rms'];assert ratio>=5
# Boundary tests for existing adaptive rule, no optimizer calls.
assert choose_alpha([dict(stratum=0,axis=0,tracking_norm=1.,smooth_norm=10.)],.5)[0]==.0099
assert choose_alpha([dict(stratum=0,axis=0,tracking_norm=1.,smooth_norm=.01)],.2)[0]==.2
assert choose_alpha([dict(stratum=0,axis=0,tracking_norm=0.,smooth_norm=1.)],.5)[0]==0
try:choose_alpha([dict(stratum=0,axis=0,tracking_norm=float('nan'),smooth_norm=1.)],.5)
except ValueError:pass
else:raise AssertionError('nonfinite not rejected')
assert core.digest(cp)==original_sha
passed=all(r['result']['accepted'] and r['frozen_buffers_equal'] for r in reports)
out=dict(passed=passed,comparisons=reports,action_ratio=ratio,disposable_optimizer_steps=4,formal_updates=0,native_calls=0,original_checkpoint_unchanged=True,boundary_tests=4)
(O/'admission.json').write_text(json.dumps(out,indent=2));print('ADMISSION',json.dumps(dict(passed=passed,alpha=reports[0]['alpha'],ratio=ratio)),flush=True)
