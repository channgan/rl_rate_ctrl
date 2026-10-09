from pathlib import Path
import json,numpy as np
E=Path(__file__).resolve().parent;R=E.parent
PHASES={'first3':(35000000,38000000),'full':(35000000,55480000),'cruise':(39000000,54000000),'tail':(54000000,55480000)}
def review(s):
 assert len(s['completed'])==s['fully_audited_count']==54
 design=json.loads((E/'frozen_design.json').read_text());records=[]
 for r in s['completed']:
  D=Path(r['result_path']).parent;rows=json.loads((D/'trajectory.json').read_text());q=json.loads((D/'full_capture_contract.json').read_text());assert q['passed'];ts=np.array([x['sim_us'] for x in rows]);errors=np.array([x['error_deg_s'] for x in rows]);ph={}
  for name,(lo,hi) in PHASES.items():
   a=errors[(ts>lo)&(ts<=hi)];ph[name]=dict(rmse=np.sqrt(np.mean(a*a,0)).tolist(),bias=a.mean(0).tolist(),p99=np.percentile(abs(a),99,axis=0).tolist(),peak=np.max(abs(a),0).tolist())
  coverage=None
  if r['case'].startswith(('low_','lateral_')):
   axis=0 if r['case'].startswith('low_x') else 1;speed=.5 if axis==0 else .25;mask=np.array([x.get('mission_phase')=='cruise_forward' for x in rows]);vv=np.array([x['linear_velocity_ned'][axis] for x in rows])[mask];coverage=float(np.mean((vv>=.8*speed)&(vv<=1.2*speed))) if len(vv) else 0.
  # Native time weighted diagnostics supplement original100Hz metrics.
  a=np.frombuffer((D/'rate_reference.bin').read_bytes()[8:],np.dtype([('m','<u8',(11,)),('v','<f4',(10,))]));t=a['m'][:,0].astype(np.int64);p=np.frombuffer((D/'physics_capture.bin').read_bytes()[8:],'<f8').reshape(-1,25);ix=np.searchsorted(p[:,2],t);assert np.array_equal(p[ix,2],t);err=np.rad2deg(a['v'][:,:3]-p[ix,22:25]);left=np.r_[t[0],t[:-1]];weighted={}
  for name,(lo,hi) in PHASES.items():
   dt=np.maximum(0,np.minimum(t,hi)-np.maximum(left,lo))*1e-6;dur=dt.sum();weighted[name]=dict(duration_s=float(dur),ise=(dt[:,None]*err**2).sum(0).tolist(),rmse=np.sqrt((dt[:,None]*err**2).sum(0)/dur).tolist(),bias=((dt[:,None]*err).sum(0)/dur).tolist())
  records.append(dict(case=r['case'],seed=r['seed'],label=r['label'],phases=ph,native_weighted=weighted,coverage=coverage,motor=r['motor_variation_100hz'],survived=r['survived_2048'],result=r))
 lookup={(r['case'],r['seed'],r['label']):r for r in records};cases=list(dict.fromkeys(j['case'] for j in design['jobs']));seeds=[490301,490302];pairs=[];lowpairs=[];cross=[];protected=design['additional_protected_cases']
 for case in cases:
  for seed in seeds:
   p=lookup[case,seed,'pid']
   for label in ['A','candidate']:
    a=lookup[case,seed,label];key='axis_p99_abs' if 'waypoints' in case else 'axis_rmse';scale,offset=(1.1,.5) if 'waypoints' in case else (1.15,.02);margin=np.asarray(a['result'][key])-np.asarray(p['result'][key])*scale-offset;motor=a['motor']-p['motor']*1.15-.0001
    pairs.append(dict(case=case,seed=seed,label=label,metric=key,axis_margin=margin.tolist(),motor_margin=motor,passed=bool(a['survived'] and p['survived'] and np.all(margin<=0) and motor<=0)))
    if case.startswith(('low_','lateral_')):
     margins={ph:(np.array(a['phases'][ph]['rmse'][:2])-np.array(p['phases'][ph]['rmse'][:2])*1.15-.02).tolist() for ph in ['full','cruise']};cover=a['coverage']>=.8 and p['coverage']>=.8;lowpairs.append(dict(case=case,seed=seed,label=label,RP_margins=margins,motor_margin=motor,coverage_pass=cover,passed=bool(cover and motor<=0 and all(max(x)<=0 for x in margins.values()))))
  for phase in PHASES:
   for axis in range(3):
    aa=np.array([lookup[case,sd,'A']['phases'][phase]['rmse'][axis] for sd in seeds]);bb=np.array([lookup[case,sd,'candidate']['phases'][phase]['rmse'][axis] for sd in seeds]);delta=(bb[:,None]-aa[None,:]).ravel();cross.append(dict(case=case,phase=phase,axis=axis,A=aa.tolist(),candidate=bb.tolist(),mean_delta=float(bb.mean()-aa.mean()),all4_deltas=delta.tolist(),all_candidate_lower=bool(np.all(delta<0)),ranges_overlap=bool(max(aa.min(),bb.min())<=min(aa.max(),bb.max()))))
 motors=[]
 for case in cases:
  aa=np.array([lookup[case,sd,'A']['motor'] for sd in seeds]);bb=np.array([lookup[case,sd,'candidate']['motor'] for sd in seeds]);motors.append(dict(case=case,A=aa.tolist(),candidate=bb.tolist(),mean_delta=float(bb.mean()-aa.mean()),all4_deltas=(bb[:,None]-aa[None,:]).ravel().tolist(),ranges_overlap=bool(max(aa.min(),bb.min())<=min(aa.max(),bb.max()))))
 original=[x for x in cases if not x.startswith('lateral_')];delta={x['case']:x['mean_delta'] for x in motors};equal7=(sum(delta[c] for c in original if c!='low_y_025_goal100')+(delta['low_y_025_goal100']+delta['lateral_plus_goal100']+delta['lateral_minus_goal100'])/3)/7
 guards=all(p['passed'] for p in pairs if p['label']=='candidate' and p['case'] in original) and all(p['passed'] for p in lowpairs if p['label']=='candidate' and p['case'] in original)
 tracking=all(x['mean_delta']<=0 for x in cross if x['case'] in protected);smooth=equal7<0 and all(delta[c]<0 for c in protected);overlap=any(x['ranges_overlap'] for x in motors if x['case'] in protected)
 out=dict(runtime_compatibility=json.loads((E/'amendment.json').read_text()),complete=True,episodes=54,training_search_charged_native=json.loads(Path(json.loads((E/'protocol.json').read_text())['training_ledger']).read_text())['charged_native'],evaluation_NN=s['actual_native_nn_calls'],evaluation_PID=s['actual_native_pid_calls'],original_PID_guards_pass=guards,protected_tracking_no_mean_regression=tracking,protected_and_equal7_motor_mean_lower=smooth,motor_repeat_overlap_inconclusive=overlap,equal7_motor_mean_delta=equal7,performance_conditions_pass=bool(guards and tracking and smooth and not overlap),pairs=pairs,low_speed_pairs=lowpairs,all_phase_axis_cross_repeat=cross,motor_comparisons=motors,records=records,accepted=False,no_heldout=True,limitations=['Two fresh development seeds; same seed does not imply identical initial state.','Native weighted values supplementary; original100Hz and first3s retained.','Overlapping motor repeat ranges are inconclusive. No automatic promotion.'])
 (E/'complete_review.json').write_text(json.dumps(out,indent=2));return out
