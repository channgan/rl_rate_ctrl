from pathlib import Path
import json,sys,hashlib,torch,numpy as np
T=Path(sys.argv[1]);D=T.parent/'development';sys.path.insert(0,str(T/'runtime'))
from v46_core import RPActor
from v46_offline_scheduler import binary_bytes
j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
done=j(T/'training_complete.json');assert done['completed'] and done['updates']==40
rows=[json.loads(line) for b in range(4) for line in (T/'training'/f'batch_{b}'/'updates.jsonl').read_text().splitlines()]
assert [r['global_update'] for r in rows]==list(range(1,41)) and all(r['accepted'] and r['parameter_delta_max']>=1e-10 and r['action_delta_previous_max']>0 and r['projection_solver']['passed'] for r in rows)
actor=RPActor(done['actor']['npz']);A=RPActor(j(T/'protocol.json')['initial_actor']['npz']);cp=torch.load(T/'training/batch_3/recovery.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(v,cp['state']['actor'][k]) for k,v in actor.state_dict().items())
assert all(torch.equal(v,A.state_dict()[k]) for k,v in actor.named_buffers())
raw,mid=binary_bytes(actor);assert raw==Path(done['actor']['binary']).read_bytes() and mid==done['actor']['model_id']
assert sha(Path(done['actor']['binary']))==done['actor']['sha256']
ledger=j(Path(j(T/'protocol.json')['canonical_ledger']));assert len(ledger['attempts'])==28 and all(a['status']=='audited' for a in ledger['attempts']) and not ledger['reservations'] and ledger['formal_optimizer_updates']==40 and ledger['charged_native']<=1316000 and ledger['native_limit']==1316000
alphas=[r['smooth_objective']['cumulative_multiplier'] for r in rows];assert all(0<=x<=1 for x in alphas) and all(y<=x for x,y in zip(alphas,alphas[1:]))
for path,h in j(T.parent/'v48_protected_before.json').items():assert sha(Path(path))==h
report=dict(passed=True,updates=40,model_id=mid,sha256=sha(Path(done['actor']['binary'])),all_frozen_buffers_equal=True,new_native_calls=0)
(T/'final_model_verification.json').write_text(json.dumps(report,indent=2))
p=j(D/'protocol_frozen.json');p['actors']['candidate']=done['actor'];p['actor_sha256']['candidate']=report['sha256'];p['candidate_binding']='verified final40 only'
with (D/'protocol.json').open('x') as f:json.dump(p,f,indent=2)
print(json.dumps(report))
