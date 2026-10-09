from pathlib import Path
import json,numpy as np,datetime
V=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');R=V/'backtracking_revision3';J=R/'training/jobs/train_b2_p0'
progress=json.loads((J/'progress.json').read_text());assert progress['stage']=='init_complete';facts=progress['facts']['native_counts'];assert facts['model_id']==330291573 and facts['nn_cycles']>0 and facts['pid_cycles']==0 and facts['fault']==0 and facts['history_valid']
p=next(J.rglob('nn_capture.bin'));dt=np.dtype([('m','<u8',(6,)),('v','<f4',(44,))]);count=(p.stat().st_size-8)//dt.itemsize
with p.open('rb') as f:f.read(8);data=np.fromfile(f,dtype=dt,count=count)
ids=np.unique(data['m'][:,4]).tolist();assert ids==[330291573]
out=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job='train_b2_p0',seed=482101,model_id=330291573,progress=progress,native_raw_path=str(p),complete_records=count,native_model_ids=ids,first_sample_us=int(data['m'][0,0]),last_sample_us=int(data['m'][-1,0]),full2048_and_audit_pending=True)
(R/'batch2_actual_actor_witness.json').write_text(json.dumps(out,indent=2))
doc=Path('@DRL_ROOT_LINUX@/rl_rate_ctrl_refactor/docs/V48_BACKTRACKING_REVISION3_20261009.md')
with doc.open('a',encoding='utf-8') as f:f.write('\nActual formal updates18..20 completed; model330291573 SHA25605c36b4117c5ff27eca606b80478cdb8e90887405391e14a028eda349d76ca9c. Actor/Adam/all proposals match offline replay exactly; updates1..20 contiguous. Third batch train_b2_p0 seed482101 native init and raw records confirm model330291573, NN selected, no fault, valid history. Full capture/audit still pending. Evidence formal_update20_verification.json and batch2_actual_actor_witness.json.\n')
print(json.dumps(out,indent=2))
