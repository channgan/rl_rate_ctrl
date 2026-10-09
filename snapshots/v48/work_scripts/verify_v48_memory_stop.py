from pathlib import Path
import json, torch
V=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');R=V/'memory_gate_revision2';B=R/'training/training/batch_1'
j=lambda p:json.loads(p.read_text())
q=torch.load(B/'recovery.pt',map_location='cpu',weights_only=False)
pre=torch.load(B/'pre_update_recovery.pt',map_location='cpu',weights_only=False)
inter=torch.load(B/'interrupted_recovery.pt',map_location='cpu',weights_only=False)
old=torch.load(R/'training/training/batch_0/recovery.pt',map_location='cpu',weights_only=False)
rows=[json.loads(x) for x in (B/'updates.jsonl').read_text().splitlines()]
assert [x['global_update'] for x in rows]==list(range(11,18))
assert q['state']['updates']==7 and not q['state']['stopped']
assert all(torch.equal(v,pre['state']['actor'][k]) for k,v in q['state']['actor'].items())
assert all(torch.equal(v,old['state']['actor'][k]) for k,v in inter['state']['actor'].items())
steps=sorted({int(v['step']) for v in q['state']['optimizer']['state'].values()});assert steps==[17]
ledger=j(V/'training/ledger.json');assert ledger['formal_optimizer_updates']==17 and ledger['charged_native']==608641 and ledger['stopped']
reports=j(B/'constraint_failure_details.json')[-4:]
out=dict(accepted_global_updates=list(range(11,18)),checkpoint_adam_steps=steps,checkpoint_matches_pre_update18=True,interrupted_checkpoint_rolls_back_to_step10_not_A=True,first_KL=rows[0]['KL'],last_KL=rows[-1]['KL'],last_bias_change=rows[-1]['bias_change'],last_global_A_bias_change=rows[-1]['global_A_bias_change'],failed_bias_changes=[r['bias_change'] for r in reports],bias_limit=1e-4,charged=608641,captures=14,development_cases=0,formal_updates=17,stopped=True,promotion_allowed=False)
(R/'stopped_update18_verification.json').write_text(json.dumps(out,indent=2))
active=j(V/'active_revision.json');active.update(launch_pending=False,state='stopped_failure',formal_optimizer_updates=17,holder_pid=42848);(V/'active_revision.json').write_text(json.dumps(active,indent=2))
print(json.dumps(out,indent=2))
