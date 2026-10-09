from pathlib import Path
import json,sys,hashlib,csv
import numpy as np
import torch
import argparse
p0=argparse.ArgumentParser();p0.add_argument('--revision',type=Path,required=True);args=p0.parse_args();N=args.revision;R=N/'development';M=N/'training';D=N/'offline_objective_diagnosis';D.mkdir(exist_ok=False)
sys.path.insert(0,str(M/'runtime'));from v46_core import RPActor
j=lambda p:json.loads(p.read_text());p=j(M/'protocol.json');A=RPActor(p['initial_actor']['npz']);F=RPActor(j(M/'training/batch_3/export.json')['npz']);torch.set_num_threads(2)
param={}
for axis in range(2):
 delta=(F.rp_w-A.rp_w)[axis].detach();param[['roll','pitch'][axis]]=dict(weight_delta_l2=float(delta.norm()),weight_relative=float(delta.norm()/A.rp_w[axis].norm()),weight_max_abs=float(abs(delta).max()),bias_delta=float((F.rp_b-A.rp_b)[axis].detach()))
records=[];outchanges=[]
for row in j(R/'completed.json')['completed']:
 path=Path(row['result_path']).parent;traj=j(path/'trajectory.json');ts=np.array([x['sim_us'] for x in traj]);err=np.array([x['error_deg_s'] for x in traj]);ref=np.array([x['metric_target'] for x in traj]);truth=np.array([x['true_rates'] for x in traj]);rr=np.frombuffer((path/'rate_reference.bin').read_bytes()[8:],np.dtype([('m','<u8',(11,)),('v','<f4',(10,))]));rt=rr['m'][:,0].astype(np.int64);rv=rr['v'].astype(float);physics=np.frombuffer((path/'physics_capture.bin').read_bytes()[8:],'<f8').reshape(-1,25);ix=np.searchsorted(rt,ts);assert np.array_equal(rt[ix],ts);pi=np.searchsorted(physics[:,2],ts);assert np.array_equal(physics[pi,2],ts);native100=np.rad2deg(rv[ix,:3]-physics[pi,22:25]);refdelta=np.rad2deg(ref-rv[ix,:3]);truthdelta=np.rad2deg(truth-physics[pi,22:25]);decomposition=float(abs(err-native100-refdelta+truthdelta).max());dt=np.diff(rt)*1e-6;u=rv[:,6:9];ni=np.searchsorted(physics[:,2],rt);assert np.array_equal(physics[ni,2],rt);ne=np.rad2deg(rv[:,:3]-physics[ni,22:25]);ph={}
 for name,lo,hi in [('first3',35000000,38000000),('after3',38000000,55480000),('full',35000000,55480000),('cruise',39000000,54000000)]:
  mask=(ts>lo)&(ts<=hi);left=np.r_[rt[0],rt[:-1]];ww=np.maximum(0,np.minimum(rt,hi)-np.maximum(left,lo))*1e-6;mo=(err[mask]**2).mean(0);mn100=(native100[mask]**2).mean(0);mn=(ww[:,None]*ne**2).sum(0)/ww.sum();ph[name]=dict(original_rmse=np.sqrt(mo).tolist(),native100_rmse=np.sqrt(mn100).tolist(),native_rmse=np.sqrt(mn).tolist(),native_bias=(ww[:,None]*ne).sum(0).tolist(),reference_timing_MSE_delta=(mo-mn100).tolist(),sampling_MSE_delta=(mn100-mn).tolist(),reference_delta_rmse=np.sqrt((refdelta[mask]**2).mean(0)).tolist(),reference_delta_peak=abs(refdelta[mask]).max(0).tolist())
  ph[name]['native_bias']=(np.array(ph[name]['native_bias'])/ww.sum()).tolist()
 score=(rt>35000000)&(rt<=55480000);entry=row['initial_state'];rec=dict(case=row['case'],seed=row['seed'],model=row['label'],phases=ph,error_decomposition_max_residual=decomposition,truth_source_delta_peak=float(abs(truthdelta).max()),torque_peak=abs(u[score]).max(0).tolist(),torque_saturation_fraction=(abs(u[score])>=.999).mean(0).tolist(),motor_upper_saturation=row['motor_upper_saturation_fraction'],motor_lower_saturation=row['motor_lower_saturation_fraction'],entry_true_rates=entry['rates_true'],entry_velocity=entry['linear_velocity_ned'],entry_position=entry['position_ned'],entry_memory=entry['nn_audit']['memory_rp'],entry_reference=entry['px4_rate_target'])
 if row['label']!='pid':
  nn=np.frombuffer((path/'nn_capture.bin').read_bytes()[8:],np.dtype([('m','<u8',(6,)),('v','<f4',(44,))]));mask=(nn['m'][:,0]>=35000000)&(nn['m'][:,0]<=55480000);v=nn['v'][mask];x=torch.tensor(v[:,1:21].copy());dtt=torch.tensor(v[:,0].copy());
  with torch.no_grad():aa=A(x,dtt).numpy();ff=F(x,dtt).numpy()
  change=ff-aa;outchanges.append(dict(case=row['case'],seed=row['seed'],trajectory_model=row['label'],count=len(x),rms=np.sqrt((change**2).mean(0)).tolist(),max_abs=abs(change).max(0).tolist()));obs=v[:,1:21];rec['observation_range']=dict(min=obs.min(0).tolist(),max=obs.max(0).tolist(),std=obs.std(0).tolist());rec['memory_range']=[obs[:,18:20].min(0).tolist(),obs[:,18:20].max(0).tolist()];rec['memory_clips']=int((abs(obs[:,18:20])>=1).sum());rec['measured_error_rad_s_rms']=np.sqrt((obs[:,3:6]*5)**2 .mean()) if False else np.sqrt(((obs[:,3:6]*5)**2).mean(0)).tolist()
 records.append(rec)
updates=[]
for b in range(4):updates.extend(json.loads(x) for x in (M/'training'/('batch_'+str(b))/'updates.jsonl').read_text().splitlines())
constraints=dict(updates=len(updates),backtracking_scales=sorted(set(x['scale'] for x in updates)),max_KL=max(x['KL'] for x in updates),min_ESS=min(x['ESS_fraction'] for x in updates),max_global_weight_relative=max(x['global_A_weight_relative'] for x in updates),max_global_bias_change=max(x['global_A_bias_change'] for x in updates),max_global_guard_action_change=max(x['global_A_guard_action_change'] for x in updates),ratio_min=min(x['ratio_min'] for x in updates),ratio_max=max(x['ratio_max'] for x in updates))
summary=dict(parameter_changes=param,common_state_output_changes=outchanges,records=records,constraints=constraints,native_calls=0,formal_updates=0)
(D/'diagnosis.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(parameter_changes=param,constraints=constraints,common_output_rms=np.sqrt(np.mean([np.array(x['rms'])**2 for x in outchanges],0)).tolist(),common_output_max=np.max([x['max_abs'] for x in outchanges],axis=0).tolist()),indent=2))
