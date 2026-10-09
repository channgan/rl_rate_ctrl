from pathlib import Path
import argparse,json,sys,math,collections,numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--revision',type=Path,required=True);p.add_argument('--v47',type=Path,required=True);a=p.parse_args();R=a.revision;T=R/'training';O=R/'offline_objective_diagnosis';O.mkdir(exist_ok=True);sys.path.insert(0,str(T/'runtime'))
from v46_core import read_capture,discounted_returns,bounded,RPActor
from v46_offline_scheduler import group_baselines
j=lambda p:json.loads(p.read_text());torch.set_num_threads(2)
curvature=[]
for phase,w,scale in [('first3',.7,.5),('later',.5,.2)]:
 for error in [0.,.01,.05,.1,.2,.5,1.,2.]:
  z=error/scale;curvature.append(dict(phase=phase,error_deg_s=error,cost_rate=w*z*z/(1+z*z),slope=2*w*error/scale**2/(1+z*z)**2,curvature=2*w/scale**2*(1-3*z*z)/(1+z*z)**3))
episodes=[];terminal_baseline_residuals=[]
for b in range(4):
 m=j(T/f'batch_{b}_captures.json');term_returns=[];times=[]
 for path in m['results']:
  q=read_capture(path);dt=q['transition_dt'];t=(q['sample_us'][1:]-q['sample_us'][0])*1e-6;first=t<=3.;e=q['native_error'];discount=np.exp(-np.r_[0,np.cumsum(dt[:-1])]/30.)
  entry=dict(batch=b,case=q['result']['case'],seed=q['result']['seed'],success=q['success'],terminal=q['terminal'],phases={})
  for name,mask,scale,w in [('first3',first,.5,.7),('later',~first,.2,.5)]:
   ee=e[mask,:2];dd=dt[mask];cost=bounded(ee/scale)*w
   entry['phases'][name]=dict(duration=float(dd.sum()),RP_rmse=np.sqrt((dd[:,None]*ee**2).sum(0)/dd.sum()).tolist(),bounded_cost_integral=(dd[:,None]*cost).sum(0).tolist(),squared_error_integral=(dd[:,None]*ee**2).sum(0).tolist(),fraction_abs_error_lt_0_02=(np.sum(dd[:,None]*(abs(ee)<.02),0)/dd.sum()).tolist(),fraction_negative_curvature=(np.sum(dd[:,None]*(abs(ee)>scale/math.sqrt(3)),0)/dd.sum()).tolist(),fraction_beyond_scale=(np.sum(dd[:,None]*(abs(ee)>scale),0)/dd.sum()).tolist())
  early=float((discount*q['tracking']*first).sum());late=float((discount*q['tracking']*(~first)).sum());entry['entry_tracking_return_later_fraction']=late/(early+late);entry['discounted_terminal_at_entry']=float(q['terminal']*discount[-1]);episodes.append(entry)
  tr=np.zeros(len(dt));tr[-1]=q['terminal'];term_returns.append(discounted_returns(tr,dt));times.append((q['sample_us'][:-1]-q['sample_us'][0])*1e-6)
 bases=group_baselines(times,term_returns,j(T/'protocol.json')['scenario_groups']);terminal_baseline_residuals.append(max(float(abs(x-y).max()) for x,y in zip(term_returns,bases)))
updates=[json.loads(x) for b in range(4) for x in (T/f'training/batch_{b}/updates.jsonl').read_text().splitlines()];reasons=collections.Counter();proposal_count=0;accepted_count=0
for b in range(1,4):
 for line in (T/f'training/batch_{b}/proposals.jsonl').read_text().splitlines():
  r=json.loads(line);proposal_count+=1;accepted_count+=int(r['accepted'])
  if not r['accepted']:
   if r['strict_batch_bias']>1e-4-1e-10:reasons['batch_bias']+=1
   if r['strict_A_bias']>1e-4-1e-10:reasons['global_A_bias']+=1
   if max(r['cost_delta'])>1e-6 or max(r['cost_delta_previous'])>1e-6:reasons['cost']+=1
   if r['KL']>.002 or max(r['axis_KL'][:2])>.001 or max(r['phase_KL'])>.004:reasons['KL']+=1
   if r['parameter_delta_max']<1e-10 or r['action_delta_previous_max']<=0:reasons['zero']+=1
protocol=j(T/'protocol.json');A=RPActor(protocol['initial_actor']['npz']);g=np.load(protocol['guard_corpus']);x=torch.tensor(g['obs'],dtype=torch.float32);dt=torch.tensor(g['dt'],dtype=torch.float32);batch_changes=[];previous=A
for b in range(4):
 F=RPActor(j(T/f'training/batch_{b}/export.json')['npz'])
 with torch.no_grad():delta=(F(x,dt)-A(x,dt)).numpy();weight=(F.rp_w-previous.rp_w).numpy();bias=(F.rp_b-previous.rp_b).numpy()
 batch_changes.append(dict(after_update=(b+1)*10,batch_weight_change_l2=np.linalg.norm(weight,axis=1).tolist(),batch_bias_delta=bias.tolist(),global_A_action_mean=delta.mean(0).tolist(),global_A_action_rms=np.sqrt((delta**2).mean(0)).tolist(),global_A_action_max=abs(delta).max(0).tolist()));previous=F
comparison={}
for name,path in [('v47',a.v47/'offline_next_revision_diagnosis1/diagnosis.json'),('v48',O/'diagnosis.json')]:
 if not path.exists():continue
 d=j(path);comparison[name]={}
 for label in ['pid','A','candidate']:
  records=[r for r in d['records'] if r['model']==label];comparison[name][label]={}
  for phase in ['first3','after3','full']:
   comparison[name][label][phase]={key:np.mean([r['phases'][phase][key] for r in records],axis=0).tolist() for key in ['reference_timing_MSE_delta','sampling_MSE_delta']}
 comparison[name]['max_algebra_residual']=max(r['error_decomposition_max_residual'] for r in d['records']);comparison[name]['max_truth_source_delta']=max(r['truth_source_delta_peak'] for r in d['records'])
report=dict(native_calls=0,optimizer_steps=0,near_zero_curvature=curvature,phase_sampling_mass=dict(first3=.5,later=.5),near_zero_effective_first_to_later_curvature_ratio=(.5/3*5.6)/(.5/17.48*25),episodes=episodes,terminal_baseline_max_abs_residual_by_batch=terminal_baseline_residuals,discount_horizon_s=30,first3_at_entry_later_tracking_fraction_range=[min(e['entry_tracking_return_later_fraction'] for e in episodes),max(e['entry_tracking_return_later_fraction'] for e in episodes)],alpha_by_update=[r['smooth_objective']['cumulative_multiplier'] for r in updates],accepted_scale_counts=dict(collections.Counter(str(r['scale']) for r in updates)),logged_proposals_from_update18=proposal_count,accepted_logged=accepted_count,rejected_reason_counts_nonexclusive=dict(reasons),guard_common_states=batch_changes,reference_decomposition=comparison,interpretation_limits=['A bounded objective has nonzero positive curvature near zero; no pitch dead zone is proven.','Later return contribution is expected for discounted return-to-go, not proof of incorrect causal assignment.','Reference-timing MSE delta also includes any measured truth-source difference; residual is checked.','No gradient sign claim is inferred from closed-loop two-seed outcome differences.'])
(O/'credit_and_constraints.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['episodes','near_zero_curvature','reference_decomposition']},indent=2))
