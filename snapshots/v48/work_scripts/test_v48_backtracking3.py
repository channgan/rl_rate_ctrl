from pathlib import Path
import json,hashlib,shutil,subprocess,sys,traceback,torch
V=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');PRE=V/'memory_gate_revision2';R=V/'backtracking_revision3';T=R/'training';O=R/'offline';O.mkdir(exist_ok=False)
j=lambda p:json.loads(p.read_text());put=lambda p,v:p.write_text(json.dumps(v,indent=2));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected=[V/'training/ledger.json',V/'development/ledger.json',PRE/'training/training/batch_1/recovery.pt',PRE/'training/training/batch_1/pre_update_recovery.pt',PRE/'training/training/batch_1/failure_details.json'];hashes={str(p):sha(p) for p in protected}
report=dict(passed=False,native_calls=0,formal_updates=0)
try:
 roots=[]
 for label in ['first','replay']:
  root=O/label;shutil.copytree(T,root);roots.append(root);put(root/'test_ledger.json',j(V/'training/ledger.json'));p=j(root/'protocol.json');p['canonical_ledger']=str(root/'test_ledger.json');put(root/'protocol.json',p)
  with (root/'offline_update.log').open('x') as log:result=subprocess.run([sys.executable,'-B',str(root/'runtime/update_batch.py'),str(root),'1'],stdout=log,stderr=subprocess.STDOUT)
  assert result.returncode==0,(label,(root/'offline_update.log').read_text()[-4000:])
  rows=[json.loads(x) for x in (root/'training/batch_1/updates.jsonl').read_text().splitlines()];assert [x['global_update'] for x in rows]==list(range(11,21))
  assert j(root/'test_ledger.json')['formal_optimizer_updates']==20 and j(root/'test_ledger.json')['charged_native']==608641
  q=torch.load(root/'training/batch_1/recovery.pt',map_location='cpu',weights_only=False);assert q['state']['updates']==10 and {int(v['step']) for v in q['state']['optimizer']['state'].values()}=={20}
  old=torch.load(T/'training/batch_0/recovery.pt',map_location='cpu',weights_only=False);assert all(torch.equal(v,old['state']['actor'][k]) for k,v in q['initial'].items())
  proposals=[json.loads(x) for x in (root/'training/batch_1/proposals.jsonl').read_text().splitlines()];assert sum(r['accepted'] for r in proposals)==3 and all(r['parameter_delta_max']>=1e-10 and r['action_delta_previous_max']>0 for r in proposals if r['accepted'])
  assert [x['scale'] for x in rows[-3:]][0]==1/32
  expected=j(PRE/'offline_backtracking_diagnosis3/report.json')['extended_update18'];assert rows[-3]['KL']==expected['KL'] and rows[-3]['strict_batch_bias']==expected['strict_batch_bias']
  report[label]=dict(resume=j(root/'training/batch_1/resume_verified.json'),updates=rows[-3:],proposals=len(proposals),export=j(root/'training/batch_1/export.json'))
 a,b=[torch.load(x/'training/batch_1/recovery.pt',map_location='cpu',weights_only=False) for x in roots]
 assert all(torch.equal(v,b['state']['actor'][k]) for k,v in a['state']['actor'].items())
 assert all(torch.equal(v,b['state']['optimizer']['state'][k][key]) for k,x in a['state']['optimizer']['state'].items() for key,v in x.items())
 assert (roots[0]/'training/batch_1/proposals.jsonl').read_bytes()==(roots[1]/'training/batch_1/proposals.jsonl').read_bytes()
 assert all(sha(Path(k))==v for k,v in hashes.items())
 report.update(passed=True,disposable_optimizer_updates=6,exact_actor_Adam_proposal_replay=True,formal_ledger_and_old_checkpoints_unchanged=True,step18_matches_diagnosis=True)
except BaseException as e:
 report.update(error=repr(e),traceback=traceback.format_exc());put(O/'admission.json',report);raise
put(O/'admission.json',report);print(json.dumps(dict(passed=True,scales=[x['scale'] for x in report['first']['updates']],native_calls=0,formal_updates=0,disposable_optimizer_updates=6,model=report['first']['export']),indent=2))
