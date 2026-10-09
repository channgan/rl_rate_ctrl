from pathlib import Path
import json,sys,importlib.util,shutil
B=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002');N=B/'native_fresh_batch_rp_v47';V=B/'v48_fixed_lr3e5';D=V/'development';O=V/'offline_review_adapter_test';O.mkdir(exist_ok=False)
j=lambda p:json.loads(p.read_text());q=j(N/'development_recovery1/complete_review.json')
for name in ['frozen_design.json','amendment.json']:shutil.copyfile(D/name,O/name)
(O/'protocol.json').write_text(json.dumps(j(D/'protocol_frozen.json')))
spec=importlib.util.spec_from_file_location('review_test',D/'runtime/review54.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.E=O
records=[]
for r in q['records']:
 x=dict(r['result']);x['seed']={470301:480301,470302:480302}[r['seed']];x['label']=r['label'];records.append(x)
r=m.review(dict(completed=records,fully_audited_count=54,actual_native_nn_calls=0,actual_native_pid_calls=0))
keys=['original_PID_guards_pass','protected_tracking_no_mean_regression','protected_and_equal7_motor_mean_lower','motor_repeat_overlap_inconclusive','equal7_motor_mean_delta','performance_conditions_pass']
assert all(r[k]==q[k] for k in keys)
assert len(r['pairs'])==36 and len(r['low_speed_pairs'])==16
(O/'result.json').write_text(json.dumps(dict(passed=True,native_calls=0,formal_updates=0,fixture='Archived54 relabeled in memory solely to test prospective seed binding; NOT V48 evaluation',original_gate_results_exact=True,main_pairs=36,low_pairs=16),indent=2));print('Review adapter offline fixture passed; no V48 evaluation data generated.')
