"""Fixed-data constrained policy updates. No simulator, native collector, or online PPO.

Library API is also used by the deterministic offline tests. Formal CLI requires
an approved protocol and an independently audited five-capture input manifest.
Only trusted locally generated recovery journals may be loaded with torch.load.
"""
import copy, hashlib, json, math, random, struct
from pathlib import Path
import numpy as np
import torch
from v46_core import RPActor, project_halfspaces, discounted_returns, leave_one_out_baselines, read_capture, bounded
COUNTERS=dict(policy_mean_vectors=0,trunk_feature_vectors=0,optimizer_attempts=0)

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint(obj):
 h=hashlib.sha256()
 for k in sorted(obj):
  v=obj[k];h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes() if isinstance(v,torch.Tensor) else json.dumps(v,sort_keys=True).encode())
 return h.hexdigest()
def logpdf(raw,mean,std):
 return (-.5*((raw.double()-mean.double())/std.double()[:,None])**2-torch.log(std.double()[:,None])-.5*math.log(2*math.pi)).sum(1)
def rng_state():return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
def rng_restore(q):torch.set_rng_state(q['torch']);np.random.set_state(q['numpy']);random.setstate(q['python'])
def flat(a):return torch.cat([p.detach().flatten() for p in a.parameters()])
def assign(a,v):
 with torch.no_grad():
  i=0
  for p in a.parameters():p.copy_(v[i:i+p.numel()].reshape_as(p));i+=p.numel()
def gradflat(loss,a):
 return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,tuple(a.parameters()),retain_graph=True)])
def binary_bytes(actor):
 z=actor.canonical();keys=['w0','b0','w1','b1','w2','b2','gate_at_1ms'];payload=b''.join(np.asarray(z[k],dtype='<f4').tobytes() for k in keys);raw=b'NNM2'+struct.pack('<I',len(payload)//4)+payload;fnv=2166136261
 for byte in raw:fnv=((fnv^byte)*16777619)&0xffffffff
 return raw+struct.pack('<I',fnv),fnv

def group_baselines(times,returns,groups):
 assert len(times)==len(returns)==len(groups) and sorted(set(groups))==list(range(5))
 if len(groups)==5:return leave_one_out_baselines(times,returns)
 result=[]
 for i,t in enumerate(times):
  others=[]
  for group in range(5):
   if group==groups[i]:continue
   others.append(np.mean([np.interp(t,times[j],returns[j]) for j in range(len(groups)) if groups[j]==group],axis=0))
  result.append(np.mean(others,axis=0))
 return result

def grouped_weights(stratum,groups):
 assert sorted(set(groups))==list(range(5))
 weights=torch.zeros(len(stratum),dtype=torch.float64)
 for i,group in enumerate(groups):
  for phase in range(2):
   mask=stratum==2*i+phase;assert mask.any();weights[mask]=.1/(groups.count(group)*int(mask.sum()))
 assert abs(float(weights.sum())-1)<1e-12
 return weights

def grouped_counts(groups,update):
 counts={};mass={}
 for group in range(5):
  members=[i for i,g in enumerate(groups) if g==group];n=len(members);base,extra=divmod(200,n)
  for phase in range(2):
   for j,i in enumerate(members):
    count=base+int((j-update)%n<extra);counts[2*i+phase]=count;mass[2*i+phase]=.1/n
 assert sum(counts.values())==2000 and abs(sum(mass.values())-1)<1e-12
 return counts,mass

def prepare(actor,captures,groups=None):
 """Historical inputs are permitted for tests only; formal manifest checks are separate."""
 groups=list(range(len(captures))) if groups is None else list(groups);assert len(groups)==len(captures) and sorted(set(groups))==list(range(5))
 times=[(q['sample_us'][:-1]-q['sample_us'][0])*1e-6 for q in captures]
 rewards=[discounted_returns(q['reward'],q['transition_dt']) for q in captures]
 base=group_baselines(times,rewards,groups)
 tracks=[discounted_returns(-q['tracking'],q['transition_dt']) for q in captures]
 smooths=[discounted_returns(-q['torque_smooth']-q['motor_smooth'],q['transition_dt']) for q in captures]
 tb=group_baselines(times,tracks,groups);sb=group_baselines(times,smooths,groups)
 rows=[]
 for i,q in enumerate(captures):
  n=len(q['reward']);x=torch.as_tensor(q['obs'][:-1]);px=torch.as_tensor(q['previous_obs'][:-1]);dt=torch.as_tensor(q['dt'][:-1]);pdt=torch.as_tensor(q['previous_dt'][:-1])
  with torch.no_grad():
   h=actor.features(x);ph=actor.features(px);mu=actor(x,dt);pmu=actor(px,pdt)
  COUNTERS['policy_mean_vectors']+=2*n;COUNTERS['trunk_feature_vectors']+=4*n
  assert np.max(abs(mu.numpy()-q['mean'][:-1]))<2e-5
  rho=torch.as_tensor(q['rho'][:-1],dtype=torch.float64);std=torch.as_tensor(q['innovation_std'][:-1],dtype=torch.float64)
  raw=torch.as_tensor(q['raw'][:-1],dtype=torch.float64);pr=torch.as_tensor(q['previous_raw'][:-1],dtype=torch.float64)
  recorded_cm=torch.as_tensor(q['old_conditional_mean'][:-1],dtype=torch.float64)
  recomputed=mu.double()+rho[:,None]*(pr-pmu.double())
  assert float(abs(recomputed-recorded_cm).max())<2e-6
  behavior=torch.as_tensor(q['old_logp'][:-1],dtype=torch.float64)
  assert torch.isfinite(behavior).all() and float(abs(logpdf(raw,recorded_cm,std)-behavior).max())<2e-3
  e=q['native_error'];first=np.maximum(0,np.minimum(q['sample_us'][1:],q['sample_us'][0]+3000000)-q['sample_us'][:-1])*1e-6
  cruise=np.maximum(0,np.minimum(q['sample_us'][1:],q['sample_us'][0]+19000000)-np.maximum(q['sample_us'][:-1],q['sample_us'][0]+4000000))*1e-6
  cost=np.column_stack([q['constraint_costs'],.2*bounded(e[:,2]/.5)*first,.2*bounded(e[:,2]/.5)*cruise])
  returns=np.column_stack([discounted_returns(cost[:,j],q['transition_dt']) for j in range(9)])
  rows.append(dict(h=h,ph=ph,yaw=mu[:,2],pyaw=pmu[:,2],raw=raw,previous_raw=pr,rho=rho,std=std,behavior=behavior,old_cm=recorded_cm,initial_mean=mu,adv=torch.tensor(rewards[i]-base[i]),cost=torch.tensor(returns),track_adv=torch.tensor(tracks[i]-tb[i]),smooth_adv=torch.tensor(smooths[i]-sb[i]),stratum=torch.tensor(i*2+(times[i]>=3).astype(np.int64)),group=torch.full((n,),groups[i],dtype=torch.int64)))
 d={k:torch.cat([r[k] for r in rows]) for k in rows[0]}
 # Five original scenario groups retain equal total mass; split within each group and phase.
 weights=grouped_weights(d['stratum'],groups)
 d['weights']=weights;mean=(weights*d['adv']).sum();sd=torch.sqrt((weights*(d['adv']-mean)**2).sum())
 assert sd>1e-12;d['adv']=(d['adv']-mean)/sd
 # Cost baselines are fixed per-stratum constants; retain physical units.
 for s in range(2*len(groups)):
  mask=d['stratum']==s;d['cost'][mask]-=d['cost'][mask].mean(0)
 return d

class Scheduler:
 def __init__(self,actor,data,config_hash,guard_h=None,guard_mean=None):
  assert sum(p.numel() for p in actor.parameters())==154
  self.groups=[int(data['group'][data['stratum']==i][0]) for i in range(0,int(data['stratum'].max())+1,2)]
  self.strata_count=2*len(self.groups)
  self.actor=actor;self.data=data;self.initial=copy.deepcopy(actor.state_dict());self.config_hash=config_hash;self.data_hash=fingerprint(data)
  self.opt=torch.optim.Adam(actor.parameters(),lr=3e-6);self.initial_opt=copy.deepcopy(self.opt.state_dict())
  self.gen=torch.Generator().manual_seed(460901);self.perms={};self.positions={};self.updates=0;self.attempts=0;self.forward_vectors=0;self.stopped=False;self.reports=[]
  self.guard_h=guard_h;self.guard_mean=guard_mean
  self.guard_hash=fingerprint({'h':guard_h,'mean':guard_mean}) if guard_h is not None else 'test-only-no-guard'
  self.initial_rng=rng_state();self.initial_gen=self.gen.get_state().clone()
  self.base_ratio=torch.exp(self.likelihood()[0].detach()-data['behavior'])
  assert float(abs(torch.log(self.base_ratio)).max())<.002
 def means(self,idx=None):
  d=self.data;ix=slice(None) if idx is None else idx
  mu=torch.clamp(torch.nn.functional.linear(d['h'][ix],self.actor.rp_w,self.actor.rp_b),-1,1)
  pm=torch.clamp(torch.nn.functional.linear(d['ph'][ix],self.actor.rp_w,self.actor.rp_b),-1,1)
  self.forward_vectors+=2*len(mu)
  COUNTERS['policy_mean_vectors']+=2*len(mu)
  return torch.cat([mu,d['yaw'][ix,None]],1),torch.cat([pm,d['pyaw'][ix,None]],1)
 def likelihood(self,idx=None):
  d=self.data;ix=slice(None) if idx is None else idx;mu,pm=self.means(idx)
  cm=mu.double()+d['rho'][ix,None]*(d['previous_raw'][ix]-pm.double())
  return logpdf(d['raw'][ix],cm,d['std'][ix]),cm,mu
 def indices(self):
  out=[];weights=[];counts,mass=grouped_counts(self.groups,self.updates)
  for s in range(self.strata_count):
   ids=torch.where(self.data['stratum']==s)[0];left=counts[s]
   while left:
    if s not in self.perms or self.positions[s]==len(ids):self.perms[s]=ids[torch.randperm(len(ids),generator=self.gen)];self.positions[s]=0
    pos=self.positions[s];take=min(left,len(ids)-pos);out.append(self.perms[s][pos:pos+take]);weights.append(torch.full((take,),mass[s]/counts[s],dtype=torch.float64));self.positions[s]+=take;left-=take
  self.minibatch_weights=torch.cat(weights);assert len(self.minibatch_weights)==2000 and abs(float(self.minibatch_weights.sum())-1)<1e-12
  return torch.cat(out)
 def snapshot(self):
  return copy.deepcopy(dict(actor=self.actor.state_dict(),optimizer=self.opt.state_dict(),rng=rng_state(),generator=self.gen.get_state(),perms=self.perms,positions=self.positions,updates=self.updates,stopped=self.stopped,reports=self.reports,critic=None,critic_updates=0))
 def restore(self,s):
  assert s['critic'] is None and s['critic_updates']==0
  self.actor.load_state_dict(s['actor']);self.opt.load_state_dict(copy.deepcopy(s['optimizer']));rng_restore(s['rng']);self.gen.set_state(s['generator']);self.perms=copy.deepcopy(s['perms']);self.positions=copy.deepcopy(s['positions']);self.updates=s['updates'];self.stopped=s['stopped'];self.reports=copy.deepcopy(s['reports'])
 def frozen_ok(self):return all(torch.equal(v,self.initial[k]) for k,v in self.actor.named_buffers())
 @torch.no_grad()
 def diagnostics(self):
  d=self.data;lp,cm,mu=self.likelihood();ratio=torch.exp(lp-d['behavior']);w=d['weights'];klaxis=.5*((cm-d['old_cm'])/d['std'][:,None])**2
  ks=(w[:,None]*klaxis).sum(0);cost=(w[:,None]*(ratio-self.base_ratio)[:,None]*d['cost']).sum(0)
  ess=lambda r:float(r.sum().square()/(len(r)*r.square().sum()))
  es=[ess(ratio[d['stratum']==s]) for s in range(self.strata_count)];phase=[float(klaxis[d['stratum']==s].sum(1).mean()) for s in range(self.strata_count)]
  head=float(((self.actor.rp_w-self.initial['rp_w']).norm(dim=1)/self.initial['rp_w'].norm(dim=1)).max());bias=float(abs(self.actor.rp_b-self.initial['rp_b']).max());change=float(abs(mu-d['initial_mean']).max())
  if self.guard_h is not None:
   gm=torch.clamp(torch.nn.functional.linear(self.guard_h,self.actor.rp_w,self.actor.rp_b),-1,1);self.forward_vectors+=len(gm);change=max(change,float(abs(gm-self.guard_mean[:,:2]).max()))
  report=dict(KL=float(ks.sum()),axis_KL=ks.tolist(),phase_KL=phase,ESS_fraction=ess(ratio),stratum_ESS_fraction=es,ratio_min=float(ratio.min()),ratio_max=float(ratio.max()),cost_delta=cost.tolist(),weight_relative_change=head,bias_change=bias,action_change=change,frozen_ok=self.frozen_ok())
  finite=all(torch.isfinite(x).all().item() for x in (lp,ratio,klaxis,cost))
  report['group_weighted_ESS_fraction']=float((w*ratio).sum().square()/(w*ratio.square()).sum())
  report['accepted']=bool(finite and report['group_weighted_ESS_fraction']>=.98 and report['KL']<=.002 and max(report['axis_KL'][:2])<=.001 and max(phase)<=.004 and min(es)>=.98 and report['ESS_fraction']>=.98 and report['ratio_min']>=.5 and report['ratio_max']<=2 and float(cost.max())<=1e-6 and head<=.01 and bias<=1e-4 and change<=.002 and report['frozen_ok'])
  return report
 def balance(self):
  lp,_,_=self.likelihood();records=[]
  for s in range(self.strata_count):
   m=self.data['stratum']==s
   gt=gradflat((lp[m]*self.data['track_adv'][m]).mean(),self.actor);gs=gradflat((lp[m]*self.data['smooth_adv'][m]).mean(),self.actor)
   # Parameter flatten order: two weight rows, then two biases.
   for axis in range(2):
    ids=list(range(axis*76,(axis+1)*76))+[152+axis];tn=float(gt[ids].norm());sn=float(gs[ids].norm());records.append(dict(stratum=s,axis=axis,tracking_norm=tn,smooth_norm=sn,ratio=sn/max(tn,1e-30)))
  assert all(r['tracking_norm']>1e-12 and r['ratio']<=.1 for r in records),'Unusable or dominating smoothness learning signal'
  return records
 def stop(self):
  self.actor.load_state_dict(self.initial);self.opt.load_state_dict(self.initial_opt);rng_restore(self.initial_rng);self.gen.set_state(self.initial_gen);self.perms={};self.positions={};self.stopped=True
 def step(self):
  if self.stopped or self.updates>=40:raise RuntimeError('Stopped or final checkpoint already reached')
  try:return self._step()
  except Exception:
   if not self.stopped:self.stop()
   raise
 def _step(self):
  if self.stopped or self.updates>=40:raise RuntimeError('Stopped or final checkpoint already reached')
  if not self.frozen_ok():self.stop();raise RuntimeError('Frozen tensor altered')
  entry=self.diagnostics();assert entry['accepted'],'Infeasible starting actor';assert float((self.actor.rp_b.detach().double()-self.initial['rp_b'].double()).abs().max())<=1e-4-1e-10,'Infeasible strict batch bias';previous_cost=np.asarray(entry['cost_delta']);before=self.snapshot();v=flat(self.actor);previous_mean=self.means()[0].detach().clone();ix=self.indices();lp,_,_=self.likelihood(ix);r=torch.exp(lp-self.data['behavior'][ix]);adv=self.data['adv'][ix]
  loss=-(self.minibatch_weights*torch.minimum(r*adv,r.clamp(.95,1.05)*adv)).sum()
  self.opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.actor.parameters(),.5);self.opt.step();self.attempts+=1
  COUNTERS['optimizer_attempts']+=1
  proposed=flat(self.actor)-v;assign(self.actor,v)
  lp,_,_=self.likelihood();ratio=torch.exp(lp-self.data['behavior']);gradients=[gradflat((self.data['weights']*ratio*self.data['cost'][:,j]).sum(),self.actor) for j in range(9)]
  delta=self.project_proposal(proposed,gradients)
  for scale in tuple(2.**-i for i in range(13)):
   assign(self.actor,v+scale*delta)
   with torch.no_grad():report=self.diagnostics()
   report['cost_delta_previous']=(np.asarray(report['cost_delta'])-previous_cost).tolist()
   report['accepted']=report['accepted'] and max(report['cost_delta_previous'])<=1e-6
   strict_bias=float((self.actor.rp_b.detach().double()-self.initial['rp_b'].double()).abs().max());delta_size=float((flat(self.actor).double()-v.double()).abs().max());action_delta=float((self.means()[0].detach().double()-previous_mean.double()).abs().max())
   report.update(scale=scale,strict_batch_bias=strict_bias,parameter_delta_max=delta_size,action_delta_previous_max=action_delta)
   report['accepted']=bool(report['accepted'] and strict_bias<=1e-4-1e-10 and delta_size>=1e-10 and action_delta>0)
   if getattr(self,'folder',None):
    with (self.folder/'proposals.jsonl').open('a') as stream:stream.write(json.dumps(dict(before_local_update=self.updates+1,**report))+'\n')
   if report['accepted']:
    self.updates+=1;report.update(update=self.updates,scale=scale);self.reports.append(report);return report
  self.restore(before);self.stop();raise RuntimeError('All 13 fixed proposal scales rejected; restored batch-start actor/optimizer/RNG, version stopped; last accepted checkpoint preserved')
 def save(self,path):
  path=Path(path);tmp=path.with_suffix('.tmp');torch.save(dict(state=self.snapshot(),initial=self.initial,initial_optimizer=self.initial_opt,initial_rng=self.initial_rng,initial_generator=self.initial_gen,config_hash=self.config_hash,data_hash=self.data_hash,guard_hash=self.guard_hash,attempts=self.attempts,forward_vectors=self.forward_vectors),tmp);tmp.replace(path)
 def resume(self,path):
  q=torch.load(path,map_location='cpu',weights_only=False)
  assert q['config_hash']==self.config_hash and q['data_hash']==self.data_hash and q['guard_hash']==self.guard_hash,'Recovery input mismatch'
  assert fingerprint(q['initial'])==fingerprint(self.initial),'Recovery starting actor mismatch'
  assert all(torch.equal(q['state']['actor'][k],self.initial[k]) for k,_ in self.actor.named_buffers()),'Recovery modified frozen tensors'
  self.restore(q['state']);self.initial_rng=q['initial_rng'];self.initial_gen=q['initial_generator'];self.attempts=q['attempts'];self.forward_vectors=q['forward_vectors']
 def export(self,folder,test_only=False):
  if self.stopped or self.updates!=40 or not self.frozen_ok():raise RuntimeError('Only accepted update40 is exportable')
  with torch.no_grad():assert self.diagnostics()['accepted']
  out=Path(folder);out.mkdir(exist_ok=False);z=self.actor.canonical();np.savez(out/'final_actor.npz',**z)
  # Match the established full-actor format and FNV checksum exactly.
  raw,fnv=binary_bytes(self.actor)
  (out/'final_actor.bin').write_bytes(raw)
  (out/'export.json').write_text(json.dumps(dict(update=40,model_id=fnv,sha256=digest(out/'final_actor.bin'),config_hash=self.config_hash,data_hash=self.data_hash,test_only=test_only,promotion_allowed=False),indent=2))
