from pathlib import Path
import json,torch,sys,hashlib
V=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');R=V/'backtracking_revision3';T=R/'training';B=T/'training/batch_1';O=R/'offline/first/training/batch_1';sys.path.insert(0,str(T/'runtime'))
from v46_core import RPActor
from v46_offline_scheduler import binary_bytes
j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
export=j(B/'export.json');assert export['global_updates']==20
rows=[json.loads(x) for b in [0,1] for x in (T/f'training/batch_{b}/updates.jsonl').read_text().splitlines()];assert [r['global_update'] for r in rows]==list(range(1,21))
assert (B/'proposals.jsonl').read_bytes()==(O/'proposals.jsonl').read_bytes()
assert sha(B/'actor.bin')==sha(O/'actor.bin')==export['sha256']
q=torch.load(B/'recovery.pt',map_location='cpu',weights_only=False);oq=torch.load(O/'recovery.pt',map_location='cpu',weights_only=False);old=torch.load(T/'training/batch_0/recovery.pt',map_location='cpu',weights_only=False)
actor=RPActor(export['npz']);A=RPActor(j(T/'protocol.json')['initial_actor']['npz'])
assert all(torch.equal(v,q['state']['actor'][k]) and torch.equal(v,oq['state']['actor'][k]) for k,v in actor.state_dict().items())
assert all(torch.equal(v,A.state_dict()[k]) for k,v in actor.named_buffers())
assert all(torch.equal(v,old['state']['actor'][k]) for k,v in q['initial'].items())
assert all(torch.equal(v,oq['state']['optimizer']['state'][k][key]) for k,x in q['state']['optimizer']['state'].items() for key,v in x.items())
assert {int(v['step']) for v in q['state']['optimizer']['state'].values()}=={20}
raw,mid=binary_bytes(actor);assert raw==(B/'actor.bin').read_bytes() and mid==export['model_id']
assert all(r['accepted'] for r in rows) and all(y['smooth_objective']['cumulative_multiplier']<=x['smooth_objective']['cumulative_multiplier'] for x,y in zip(rows,rows[1:]))
for p,h in j(R/'frozen_inputs.json').items():assert sha(Path(p))==h
for p,h in j(R/'v47_ledgers_before.json').items():assert sha(Path(p))==h
report=dict(passed=True,formal_updates=20,contiguous_unique_updates=True,actual_equals_offline_actor_Adam_all_proposals=True,model_id=mid,sha256=export['sha256'],batch_anchor=10,Adam_steps=20,scales=[r['scale'] for r in rows[-3:]],KL=rows[-1]['KL'],strict_batch_bias=rows[-1]['strict_batch_bias'],strict_A_bias=rows[-1]['strict_A_bias'],frozen_buffers_equal_A=True,old_contracts_unchanged=True,promotion_allowed=False)
(R/'formal_update20_verification.json').write_text(json.dumps(report,indent=2));active=j(V/'active_revision.json');active.update(launch_pending=False,holder_pid=5668,state='running');(V/'active_revision.json').write_text(json.dumps(active,indent=2));print(json.dumps(report,indent=2))
