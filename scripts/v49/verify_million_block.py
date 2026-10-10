"""Final block consistency only; no automatic performance evaluation."""
from pathlib import Path
import json,sys,hashlib
import torch
from v46_core import RPActor
from v46_offline_scheduler import binary_bytes

def verify(root):
    root=Path(root);j=lambda p:json.loads(p.read_text());p=j(root/'protocol.json');l=j(root/'ledger.json')
    assert len(l['attempts'])==490 and all(a['status']=='audited' for a in l['attempts']) and not l['reservations']
    assert l['formal_optimizer_updates']==740 and l['audited_effective_outer_steps']==1003520 and l['charged_native']<=23030000
    rows=[json.loads(s) for b in range(70) for s in (root/f'training/batch_{b}/updates.jsonl').read_text().splitlines()]
    assert [r['global_update'] for r in rows]==list(range(41,741))
    assert all(r['accepted'] and r['frozen_ok'] and r['projection_solver']['passed'] and r['parameter_delta_max']>=1e-10 and r['action_delta_previous_max']>0 for r in rows)
    alphas=[r['smooth_objective']['cumulative_multiplier'] for r in rows];assert alphas[0]<=.3767965169092372 and all(y<=x for x,y in zip(alphas,alphas[1:]))
    cp=torch.load(root/'training/batch_69/recovery.pt',map_location='cpu',weights_only=False);actor=RPActor(root/'training/batch_69/actor.npz');A=RPActor(p['initial_actor']['npz'])
    assert all(torch.equal(v,cp['state']['actor'][k]) for k,v in actor.state_dict().items())
    assert all(torch.equal(v,A.state_dict()[k]) for k,v in actor.named_buffers())
    assert {int(s['step']) for s in cp['state']['optimizer']['state'].values()}=={740}
    raw,mid=binary_bytes(actor);assert raw==(root/'training/batch_69/actor.bin').read_bytes()
    report=dict(passed=True,model_id=mid,sha256=hashlib.sha256(raw).hexdigest(),new_updates=700,effective_outer_steps=1003520,automatic_evaluation=False)
    (root/'block_verification.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':print(json.dumps(verify(sys.argv[1])))
