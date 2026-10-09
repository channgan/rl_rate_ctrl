import copy,json,math
from pathlib import Path
import numpy as np
import torch
from v46_offline_scheduler import Scheduler,prepare,gradflat,fingerprint

def choose_alpha(records,previous):
 if not math.isfinite(previous) or not 0<=previous<=1:raise ValueError('nonfinite/invalid multiplier')
 for r in records:
  if not all(math.isfinite(r[k]) and r[k]>=0 for k in ['tracking_norm','smooth_norm']):raise ValueError('nonfinite/invalid gradient: '+repr(r))
 zeros=[(r['stratum'],r['axis']) for r in records if r['tracking_norm']<=1e-12]
 if zeros:return 0.,zeros
 return min([previous]+[.099*r['tracking_norm']/r['smooth_norm'] for r in records if r['smooth_norm']>0]),zeros

class AdaptiveScheduler(Scheduler):
 def attach(self,behavior,captures,groups,alpha=1.,folder=None):
  self.behavior=behavior;self.captures=captures;self.capture_groups=groups;self.alpha=alpha;self.folder=Path(folder) if folder else None;self.raw_data_hash=self.data_hash;self.balance_history=[];self.proposal_diagnostics=[]
 def records(self):
  lp=self.likelihood()[0];result=[]
  for st in range(self.strata_count):
   mask=self.data['stratum']==st;gt=gradflat((lp[mask]*self.data['track_adv'][mask]).mean(),self.actor);gs=gradflat((lp[mask]*self.data['smooth_adv'][mask]).mean(),self.actor)
   for axis in range(2):
    ids=list(range(axis*76,(axis+1)*76))+[152+axis];tn=float(gt[ids].norm().detach());sn=float(gs[ids].norm().detach());result.append(dict(stratum=st,axis=axis,n=int(mask.sum()),tracking_norm=tn,smooth_norm=sn,ratio=sn/max(tn,1e-30)))
  return result
 def rebuild(self,alpha):
  captures=[]
  for q in self.captures:
   c=dict(q);c['reward']=q['reward']+(1-alpha)*(q['torque_smooth']+q['motor_smooth']);c['torque_smooth']=alpha*q['torque_smooth'];c['motor_smooth']=alpha*q['motor_smooth'];captures.append(c)
  d=prepare(self.behavior,captures,self.capture_groups)
  if not all(torch.isfinite(v).all() for v in d.values() if isinstance(v,torch.Tensor)):raise ValueError('nonfinite rebuilt training data')
  # Keep physical costs, behavior distribution, base_ratio, features and guards unchanged.
  for k in ['adv','track_adv','smooth_adv']:self.data[k]=d[k]
  self.data_hash=fingerprint(self.data);self.alpha=alpha
 def adapt(self):
  previous=self.alpha;self.rebuild(1.);raw=self.records()
  if self.folder:(self.folder/'pre_balance_raw.json').write_text(json.dumps(dict(previous=previous,records=raw),indent=2))
  alpha,zeros=choose_alpha(raw,previous);self.rebuild(alpha);actual=self.records()
  report=dict(before_update=self.updates+1,previous_multiplier=previous,cumulative_multiplier=alpha,relative_multiplier=alpha/previous if previous else 1.,coefficients=dict(legacy_torque=(2/70)*alpha,extra_RP_torque=.05*alpha,ESC=.05*alpha),zero_tracking_strata_axes=zeros,raw=raw,actual=actual,data_hash=self.data_hash,physical_scoring_unchanged=True)
  self.balance_history.append(report)
  if self.folder:(self.folder/'balance_diagnostics.json').write_text(json.dumps(self.balance_history,indent=2))
  if not all(math.isfinite(r['ratio']) and (r['ratio']<=.1 if r['tracking_norm']>1e-12 else alpha==0 and r['smooth_norm']==0) for r in actual):raise RuntimeError('remeasured original 10% balance failed; see balance_diagnostics.json')
  return report
 def snapshot(self):
  q=super().snapshot()
  if hasattr(self,'alpha'):q['objective_state']=dict(alpha=self.alpha,data_hash=self.data_hash,raw_data_hash=self.raw_data_hash,balance_history=copy.deepcopy(self.balance_history))
  return q
 def restore(self,q):
  super().restore(q)
  if 'objective_state' in q:
   v=q['objective_state'];assert v['raw_data_hash']==self.raw_data_hash;self.rebuild(v['alpha']);assert self.data_hash==v['data_hash'];self.balance_history=copy.deepcopy(v['balance_history'])
 def diagnostics(self):
  r=super().diagnostics()
  if hasattr(self,'proposal_diagnostics'):self.proposal_diagnostics.append(copy.deepcopy(r))
  return r
 def step(self):
  try:return super().step()
  except BaseException:
   if self.folder:(self.folder/'constraint_failure_details.json').write_text(json.dumps(self.proposal_diagnostics,indent=2))
   raise
