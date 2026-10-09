from pathlib import Path
import json,csv,math,collections,numpy as np
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--development',type=Path,required=True);args=parser.parse_args();R=args.development;q=json.loads((R/'complete_review.json').read_text());OUT=R/'final_metrics';OUT.mkdir(exist_ok=False)
rows=[];episodes=[]
for r in q['records']:
 ep=dict(case=r['case'],seed=r['seed'],model=r['label'],coverage=r['coverage'],motor100Hz=r['motor'],original_result=r['result'])
 for typ,key in [('original100Hz','phases'),('native_consumed','native_weighted')]:
  full=np.asarray(r[key]['full']['rmse']);early=np.asarray(r[key]['first3']['rmse']);late=np.sqrt(np.maximum(0,(full**2*20.48-early**2*3)/17.48));phases={ph:np.asarray(v['rmse']) for ph,v in r[key].items()};phases['after3']=late
  for ph,a in phases.items():rows.append(dict(case=r['case'],seed=r['seed'],model=r['label'],reference=typ,phase=ph,roll=float(a[0]),pitch=float(a[1]),yaw=float(a[2]),RP=float(np.sqrt(np.mean(a[:2]**2)))))
 d=Path(r['result']['result_path']).parent;rr=np.frombuffer((d/'rate_reference.bin').read_bytes()[8:],np.dtype([('m','<u8',(11,)),('v','<f4',(10,))]));ts=rr['m'][:,0].astype(np.int64);u=rr['v'][:,6:9].astype(float);dt=np.diff(ts)*1e-6;mask=(ts[1:]>35000000)&(ts[1:]<=55480000);du=np.diff(u,axis=0);ep['native_torque_slew_RP_rms_per_s']=float(np.sqrt(np.sum(np.mean((du[mask,:2]/dt[mask,None])**2,axis=1)*dt[mask])/dt[mask].sum()));ep['native_torque_total_variation_RP']=float(np.sum(np.mean(abs(du[mask,:2]),axis=1)))
 ep.update({key:ep['original_result'][key] for key in ['height_rmse_m','height_p95_abs_m','height_min_m','position_error_p95_m','motor_upper_saturation_fraction','motor_lower_saturation_fraction']});ep.pop('original_result');episodes.append(ep)
cases=list(dict.fromkeys(r['case'] for r in q['records']));aggregates=[]
for case in cases+['ALL9_equal_episode']:
 for typ in ['original100Hz','native_consumed']:
  for phase in ['first3','after3','full','cruise','tail']:
   item=dict(case=case,reference=typ,phase=phase,models={})
   for label in ['pid','A','candidate']:
    rr=[r for r in rows if (case=='ALL9_equal_episode' or r['case']==case) and r['reference']==typ and r['phase']==phase and r['model']==label];item['models'][label]={axis:float(np.sqrt(np.mean([r[axis]**2 for r in rr]))) for axis in ['roll','pitch','yaw','RP']}
   item['candidate_vs_A_percent']={axis:100*(item['models']['candidate'][axis]/item['models']['A'][axis]-1) for axis in ['roll','pitch','yaw','RP']};item['candidate_vs_PID_percent']={axis:100*(item['models']['candidate'][axis]/item['models']['pid'][axis]-1) for axis in ['roll','pitch','yaw','RP']};aggregates.append(item)
sm=[]
for case in cases+['ALL9_equal_episode']:
 vals={label:{key:float(np.mean([e[key] for e in episodes if (case=='ALL9_equal_episode' or e['case']==case) and e['model']==label])) for key in ['motor100Hz','native_torque_slew_RP_rms_per_s','native_torque_total_variation_RP']} for label in ['pid','A','candidate']};sm.append(dict(case=case,models=vals,candidate_vs_A_motor_percent=100*(vals['candidate']['motor100Hz']/vals['A']['motor100Hz']-1),candidate_vs_PID_motor_percent=100*(vals['candidate']['motor100Hz']/vals['pid']['motor100Hz']-1)))
gates={label:dict(all9_main_pass=sum(x['passed'] for x in q['pairs'] if x['label']==label),original7_main_pass=sum(x['passed'] for x in q['pairs'] if x['label']==label and not x['case'].startswith('lateral_')),all4_low_pass=sum(x['passed'] for x in q['low_speed_pairs'] if x['label']==label),original2_low_pass=sum(x['passed'] for x in q['low_speed_pairs'] if x['label']==label and x['case'].startswith('low_'))) for label in ['A','candidate']}
report=dict(definition='Per phase axis RMSE pooled equally over two seeds; RP=sqrt(mean(roll_RMSE^2,pitch_RMSE^2)); ALL9 equal episode, descriptive not original acceptance aggregate. after3 derived from complete and first3 squared-error integrals,17.48s. Native torque derivative is supplementary normalized torque/s, not acceptance metric.',aggregates=aggregates,episodes=episodes,smoothness=sm,gates=gates,original_gate_flags={k:q[k] for k in ['original_PID_guards_pass','protected_tracking_no_mean_regression','protected_and_equal7_motor_mean_lower','motor_repeat_overlap_inconclusive','equal7_motor_mean_delta','performance_conditions_pass']},failed_candidate_main=[x for x in q['pairs'] if x['label']=='candidate' and not x['passed']],failed_candidate_low=[x for x in q['low_speed_pairs'] if x['label']=='candidate' and not x['passed']],protected_cases=json.loads((R/'frozen_design.json').read_text())['additional_protected_cases'],all_phase_axis_cross_repeat=q['all_phase_axis_cross_repeat'])
(OUT/'summary.json').write_text(json.dumps(report,indent=2))
with (OUT/'per_episode_phase_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (OUT/'per_episode_smoothness_coverage.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(episodes[0]));w.writeheader();w.writerows(episodes)
print(json.dumps(dict(gates=gates,flags=report['original_gate_flags'],overall=[a for a in aggregates if a['case']=='ALL9_equal_episode' and a['phase']=='full'],smoothness=sm[-1]),indent=2))
