from pathlib import Path
import argparse,json,csv,numpy as np
p=argparse.ArgumentParser();p.add_argument('--development',type=Path,required=True);a=p.parse_args();D=a.development;M=D/'final_metrics';q=json.loads((D/'complete_review.json').read_text());rows=list(csv.DictReader((M/'per_episode_phase_metrics.csv').open()));assert len(rows)==540 and len({(r['case'],r['seed'],r['model'],r['reference'],r['phase']) for r in rows})==540
index={(r['case'],int(r['seed']),r['model'],r['reference'],r['phase']):np.array([float(r[k]) for k in ['roll','pitch','yaw']]) for r in rows};details=[]
for record in q['records']:
 d=Path(record['result']['result_path']).parent;trows=json.loads((d/'trajectory.json').read_text());t=np.array([r['sim_us'] for r in trows]);err=np.array([r['error_deg_s'] for r in trows]);assert len(trows)==2048
 direct=np.sqrt(np.mean(err[(t>38000000)&(t<=55480000)]**2,axis=0));assert np.allclose(direct,index[record['case'],record['seed'],record['label'],'original100Hz','after3'],rtol=1e-10,atol=1e-10)
 ref=np.frombuffer((d/'rate_reference.bin').read_bytes()[8:],np.dtype([('m','<u8',(11,)),('v','<f4',(10,))]));ts=ref['m'][:,0].astype(np.int64);physics=np.frombuffer((d/'physics_capture.bin').read_bytes()[8:],'<f8').reshape(-1,25);ix=np.searchsorted(physics[:,2],ts);assert np.array_equal(physics[ix,2],ts);errors=np.rad2deg(ref['v'][:,:3]-physics[ix,22:25]);dt=np.maximum(0,np.minimum(ts,55480000)-np.maximum(np.r_[ts[0],ts[:-1]],38000000))*1e-6;assert abs(dt.sum()-17.48)<1e-9
 native=np.sqrt((dt[:,None]*errors**2).sum(0)/dt.sum());assert np.allclose(native,index[record['case'],record['seed'],record['label'],'native_consumed','after3'],rtol=1e-10,atol=1e-10)
 details.append(dict(case=record['case'],seed=record['seed'],model=record['label'],original100Hz=record['phases'],native_consumed=record['native_weighted']))
(M/'phase_axis_details.json').write_text(json.dumps(details,indent=2));result=dict(passed=True,unique_phase_rows=540,episodes=54,direct_after3_original_and_native_checks=108,native_after3_duration_s=17.48,no_native_calls=True);(M/'metric_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
