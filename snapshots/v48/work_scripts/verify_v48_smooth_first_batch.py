from pathlib import Path
import json,torch,sys,hashlib,numpy as np,datetime
V=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');R=V/'smooth_start_revision1';T=R/'training';sys.path.insert(0,str(T/'runtime'))
from v46_core import RPActor
from v46_offline_scheduler import binary_bytes
j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=j(T/'protocol.json');A=RPActor(p['initial_actor']['npz']);cp=torch.load(T/'training/batch_0/recovery.pt',map_location='cpu',weights_only=False);rows=[json.loads(x) for x in (T/'training/batch_0/updates.jsonl').read_text().splitlines()];bal=j(T/'training/batch_0/balance_diagnostics.json');ex=j(T/'training/batch_0/export.json');job=j(T/'jobs/train_b1_p0/protocol.json');actor=RPActor(ex['npz'])
assert len(rows)==10 and [x['global_update'] for x in rows]==list(range(1,11)) and cp['state']['updates']==10
assert all(torch.equal(v,cp['state']['actor'][k]) for k,v in actor.state_dict().items());raw,mid=binary_bytes(actor);assert mid==ex['model_id'] and raw==Path(ex['binary']).read_bytes() and sha(Path(ex['binary']))==ex['sha256']
assert all(torch.equal(v,cp['state']['actor'][k]) for k,v in A.named_buffers())
assert all(x['accepted'] and x['frozen_ok'] for x in rows)
assert all(len(x['actual'])==28 and all(v['ratio']<=.1 for v in x['actual']) for x in bal)
assert all(y['cumulative_multiplier']<=x['cumulative_multiplier'] for x,y in zip(bal,bal[1:]))
assert job['head_model_id']==mid and job['actor_sha256']==ex['sha256']
f=T/'jobs/train_b1_p0/candidate/noise200_hover_hold/seed_481101/nn_capture.bin';n=(f.stat().st_size-8)//224
with f.open('rb') as stream:stream.seek(8);buf=stream.read(n*224)
nr=np.frombuffer(buf,dtype=np.dtype([('m','<u8',(6,)),('v','<f4',(44,))]));assert len(nr)>0 and np.all(nr['m'][:,4]==mid)
assert all(sha(Path(path))==h for path,h in j(V/'frozen_inputs.json').items())
assert all(sha(Path(path))==h for path,h in j(R/'frozen_inputs.json').items())
assert all(sha(Path(path))==h for path,h in j(V/'v47_ledgers_before.json').items())
result=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),formal_updates=10,actual_lr=[x['lr'] for x in cp['state']['optimizer']['param_groups']],adam_steps=[float(x['step']) for x in cp['state']['optimizer']['state'].values()],frozen_buffers_equal_A=True,npz_binary_checkpoint_equal=True,all_10_updates_accepted=True,all_280_balance_checks_pass=True,max_KL=max(x['KL'] for x in rows),min_stratum_ESS=min(min(x['stratum_ESS_fraction']) for x in rows),scales=[x['scale'] for x in rows],alphas=[x['cumulative_multiplier'] for x in bal],maximum_remeasured_balance=max(v['ratio'] for x in bal for v in x['actual']),model_id=mid,binary_sha256=ex['sha256'],next_job='train_b1_p0',next_seed=481101,next_job_model_matches=True,native_observed_model_id=mid,native_record_count=n,native_latest_sample_us=int(nr['m'][-1,0]),formal_first_matches_offline_KL=rows[0]['KL']==j(R/'offline/lr_3e-05_report.json')['result']['KL'],canonical_ledger=j(V/'training/ledger.json'),original_contract_and_v47_ledgers_unchanged=True,new_native_calls_by_verification=0)
(R/'first_batch_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='canonical_ledger'},indent=2))
