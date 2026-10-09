from pathlib import Path
import json,sys,numpy as np
N=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/native_fresh_batch_rp_v47');M=N/'memory_sampling_revision1';O=N.parent/'v48_lr_offline_admission1';sys.path.insert(0,str(M/'runtime'))
from v46_core import read_capture,bounded
j=lambda p:json.loads(p.read_text());rows=[]
for b,root in [(1,N/'smooth_objective_revision1'),(2,M),(3,M)]:
 p=j(root/'protocol.json');m=j(root/f'batch_{b}_captures.json')
 for i,path in enumerate(m['results']):
  q=read_capture(path);t=q['sample_us'];dt=q['transition_dt'];first=np.maximum(0,np.minimum(t[1:],t[0]+3000000)-t[:-1])*1e-6;late=dt-first;e=q['native_error'][:,1]
  for phase,dur,scale,weight in [('first3',first,.5,.7),('later',late,.2,.5)]:
   c=weight*bounded(e/scale);sensitivity=2*weight*abs(e)/(scale**2*(1+(e/scale)**2)**2)
   rows.append(dict(batch=b,profile=p['entry_profiles'][i]['id'],case=q['result']['case'],phase=phase,pitch_RMSE_deg_s=float(np.sqrt(np.sum(dur*e**2)/dur.sum())),pitch_cost_integral=float(np.sum(dur*c)),mean_abs_cost_derivative_per_deg_s=float(np.sum(dur*sensitivity)/dur.sum()),fraction_abs_error_above_scale=float(np.sum(dur*(abs(e)>=scale))/dur.sum()),sampling_mass=.1/p['scenario_groups'].count(p['scenario_groups'][i])))
points=[]
for e in [.05,.1,.2,.5,1.,2.]:
 vals={}
 for phase,s,w in [('first3',.5,.7),('later',.2,.5)]:vals[phase]=dict(cost=w*float(bounded(e/s)),derivative=2*w*e/(s*s*(1+(e/s)**2)**2))
 points.append(dict(error_deg_s=e,**vals))
r=dict(native_calls=0,formal_updates=0,rows=rows,analytic_points=points,first3_sampling_mass=.5,later_sampling_mass=.5,per_second_sampling_density_ratio=17.48/3,near_zero_cost_curvature_ratio=(.7/.5**2)/(.5/.2**2),caveat='Reward-to-go mixes phases and axes; these cost derivatives are diagnostics, not policy gradients. Sampling mass does not imply sufficient pitch credit.')
(O/'phase_diagnostic.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(rows=len(rows),density_ratio=r['per_second_sampling_density_ratio'],curvature_ratio=r['near_zero_cost_curvature_ratio'],points=points)),flush=True)
