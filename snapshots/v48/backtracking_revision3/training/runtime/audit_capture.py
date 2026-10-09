from pathlib import Path
import sys,json,hashlib,types,traceback,argparse,os,importlib.util
import numpy as np
R=Path(__file__).parent;O=Path(os.environ['BATCH_TRAINING_ROOT']);P=R;E=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/native_roll_smooth_v44/evaluation_equivalence_continuation3')
sys.path.insert(0,str(E));sys.path.insert(0,str(R))
from v46_training_entry import verify_commands
from v46_core import rotation,read_capture
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
jr=Path(sys.argv[1]);pp=json.loads((jr/'protocol.json').read_text());profile=pp['v46_training_entry_profile'];D=jr/pp['audit_profile']/profile['case']/('seed_'+str(profile['seed']));rp=D/'result.json';r=json.loads(rp.read_text());r.update(result_path=str(rp),label=pp['audit_label'],v46_training=True);attempt={}
event=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/native_entry_coverage_v46/event55_offline_revision1/goal_event_semantics_v6.py')
assert sha(event)=='1debced4df595b6fa659b45968cb53fdcce49a6de1ca899195cc7921afef0f36'
spec=importlib.util.spec_from_file_location('goal_event_semantics',event);mod=importlib.util.module_from_spec(spec);sys.modules['goal_event_semantics']=mod;spec.loader.exec_module(mod)
code=(E/'check_stochastic_episode.py').read_text();needle="model=next(j for j in json.loads((R/'config.json').read_text())['jobs'] if (j['model_id']==int(n['m'][0,4])))";assert code.count(needle)==1
code=code.replace(needle,'model='+repr(dict(model_id=pp['head_model_id'],initial_actor=pp['actor_npz'])))
assert code.count("('nn_capture.bin',len(n),65536)")==1
code=code.replace("('nn_capture.bin',len(n),65536)","('nn_capture.bin',len(n),131072)")
mod=types.ModuleType('check_stochastic_episode');sys.modules['check_stochastic_episode']=mod;exec(compile(code,str(E/'check_stochastic_episode.py')+'[current batch actor]','exec'),mod.__dict__)
# Bind capture storage metadata to the unchanged validated v46 control binary.
reference_code=(E/'audit_rate_reference_revised2.py').read_text()
assert reference_code.count("integrity['capacity']==65536")==1
reference_code=reference_code.replace("integrity['capacity']==65536","integrity['capacity']==131072")
reference_module=types.ModuleType('audit_rate_reference_revised2');sys.modules['audit_rate_reference_revised2']=reference_module
exec(compile(reference_code,str(E/'audit_rate_reference_revised2.py')+'[131072 storage capacity only]','exec'),reference_module.__dict__)
try:
 if profile.get('vertical_extension'):
  entry=json.loads((D/'v46_entry_snapshot.json').read_text())['initial_state']
  assert entry['linear_velocity_ned'][2]*profile['velocity_goal_ned_m_s'][2]>0, 'vertical entry truth sign did not follow frozen goal'
 commands=verify_commands(D,profile,r['case'])
 # Retain every original capture/timing/coordinate/ownership check, substituting
 # only the deliberately changed training mission equality check.
 import runner_extension
 original_mission=runner_extension.verify_mission
 runner_extension.verify_mission=lambda rows,case:verify_commands(D,profile,case)
 source=(E/'audit_collect.py').read_text();source=source.replace("if r['case'].startswith('lateral_'):","if r['case'].startswith('lateral_') and not r.get('v46_training',False):")
 assert source.count("integ['capacity']==65536")==1;source=source.replace("integ['capacity']==65536","integ['capacity']==131072")
 module=types.ModuleType('v46_capture_audit');exec(compile(source,str(E/'audit_collect.py')+'[training mission adapter]','exec'),module.__dict__)
 report=module.check(r);runner_extension.verify_mission=original_mission
 max_history_error=None
 if pp['audit_label']!='pid':
  nn=np.frombuffer((D/'nn_capture.bin').read_bytes()[8:],np.dtype([('m','<u8',(6,)),('v','<f4',(44,))]));t=nn['m'][:,0].astype(np.int64);v=nn['v'];ids=np.flatnonzero((t>=35000000)&(t<=55480000));assert ids[0]>=128
  # Reconstruct each128-entry ring from the complete native executed-action trace.
  max_history_error=0.
  for i in ids:
   stamps=t[i-128:i];assert np.all(np.diff(stamps)>0) and stamps[-1]<t[i]
   j=int(np.searchsorted(stamps,t[i]-10000,side='right'))-1;assert j>=0
   selected=i-128+j;next_stamp=stamps[j+1] if j+1<128 else t[i];assert t[selected]<=t[i]-10000<next_stamp
   max_history_error=max(max_history_error,float(abs(v[i,7:10]-v[selected,21:24]).max()))
  assert max_history_error<2e-5
 mech=np.frombuffer((D/'mechanism.bin').read_bytes()[8:],np.dtype([('m','<u8',(12,)),('v','<f4',(28,))]));m=mech['m'];mv=mech['v'];cmds=json.loads((D/'v46_prefix_commands.json').read_text())+json.loads((D/'v46_score_commands.json').read_text())[:-1];entry_native=None
 requests={c['boundary_us']:c for c in json.loads((D/'goal_requests.json').read_text()) if c['boundary_us'] is not None};serialization_error=0.
 for c in cmds:
  b=c['sample_us'];lo=np.searchsorted(m[:,0],b,side='right');hi=np.searchsorted(m[:,0],b+10000,side='right');desired=np.asarray([*c['goal'],c['heading']]);sent=np.asarray(requests[b]['values']);serialization_error=max(serialization_error,float(abs(sent[:3]-desired[:3]).max()));assert serialization_error<=2e-6 and abs(sent[3]-desired[3])<=1e-7
  # Match the frozen bridge's nine-significant-digit wire representation, as
  # the original GOAL event audit does; never substitute a desired-only target.
  payload=sent.astype(np.float32)
  mask=(m[lo:hi,2]>=b)&(m[lo:hi,6]!=0)&np.all(mv[lo:hi,:4]==payload,axis=1);found=np.flatnonzero(mask);assert len(found),('Command lacks native consumption witness',b)
  if b==35000000:entry_native=mv[lo+int(found[0]),:3].astype(float)
 e=json.loads((D/'v46_entry_snapshot.json').read_text());jump=rotation(e['initial_state']['px4_heading']).T@(entry_native-np.asarray(e['old_goal_proof']['goal'][:3]));jump_error=float(abs(jump-profile['entry_jump_body_m']).max());assert jump_error<=2e-6
 if pp['audit_label']=='pid':
  assert r['nn_final_audit']['nn_cycles']==0 and r['nn_final_audit']['pid_cycles']>0
  audit=dict(passed=True,result_sha256=sha(rp),PID=True,actual_native_PID=r['nn_final_audit']['pid_cycles'],locked_parameters=True,GOAL_consumption_witnesses=len(cmds),actual_entry_jump_body=jump.tolist(),actual_entry_jump_error=jump_error,full_capture_report=report)
 else:
  q=read_capture(rp);assert q['success'];native=report['native']['nn_contract'];ref=report['reference'];latency=max(native['gyro_capture_cost_ns']['max'],native['nn_capture_cost_ns']['max'],ref['record_assembly_max_ns'])/1000.
  audit=dict(passed=True,result_sha256=sha(rp),checks={k:True for k in ['ownership','native_timing','consumed_reference','all128_history','lookback10ms','GOAL_commands','full_capture','locked_parameters']},native_max_gap_us=native['score_max_gap_us'],native_p99_gap_us=native['score_p99_gap_us'],reference_age_max_us=ref['reference_age_at_consumption_range_us'][1],native_cost_latency_max_us=latency,history_reconstruction_error=max_history_error,history_method='Reconstruct all128timestamps/actions from complete sequential native trace; verify actual selected10ms action; no hidden buffer dump claimed',GOAL_consumption_witnesses=len(cmds),actual_entry_jump_body=jump.tolist(),actual_entry_jump_error=jump_error,native_RMSE=q['native_rmse'].tolist(),original100Hz_RMSE=r['axis_rmse'],native_reward_sum=float(q['reward'].sum()),terminal=q['terminal'],native_transitions=len(q['reward']),AR_behavior=native.get('stochastic'),full_capture_report=report)
 (D/'combined_audit.json').write_text(json.dumps(audit,indent=2));attempt.update(status='audited',audit_path=str(D/'combined_audit.json'),audit_sha256=sha(D/'combined_audit.json'));print(json.dumps({k:v for k,v in audit.items() if k!='full_capture_report'},indent=2))
except BaseException as exc:
 (D/'audit_failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise
