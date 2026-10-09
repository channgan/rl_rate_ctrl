from pathlib import Path
import argparse,json,hashlib,shutil,importlib.util,sys,torch
p=argparse.ArgumentParser();p.add_argument('--revision',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();R=a.revision;V=R.parent;D=R/'development';T=R/'training';O=R/'independent_final_audit';O.mkdir(exist_ok=False)
j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tl=j(V/'training/ledger.json');dl=j(V/'development/ledger.json');protocol=j(D/'protocol.json');completed=j(D/'completed.json')['completed'];expected={(x['case'],x['seed'],x['label']) for x in protocol['jobs']}
assert len(expected)==54 and {(x['case'],x['seed'],x['label']) for x in completed}==expected and len(completed)==54
assert len(dl['attempts'])==54 and all(x['status']=='audited' for x in dl['attempts']) and not dl['reservations']
assert dl['charged_native']==sum(x['charge'] for x in dl['attempts'])==2397578<=dl['native_limit']==3000000
assert tl['formal_optimizer_updates']==40 and len(tl['attempts'])==28 and not tl['reservations'] and tl['charged_native']==1213571<=2000000
assert j(D/'status.json')['state']=='development_complete_not_promoted' and j(R/'execution/holder_completed.json')['returncode']==0
input_hashes={str(f):sha(f) for f in [V/'training/ledger.json',V/'development/ledger.json',D/'complete_review.json']};checks=[]
for x in dl['attempts']:
 d=Path(x['output']);q=j(d/'result.json');au=j(d/'combined_audit.json');sy=j(d/'sync_audit.json')
 assert q['survived_2048'] and q['steps']==2048 and q['physical_failure'] is None and q['nn_final_audit']['fault']==0
 assert au['passed'] and sy['passed'] and j(d/'export_status.json')['success'] and j(d/'full_capture_contract.json')['passed']
 assert sha(d/'result.json')==au['result_sha256'] and sha(d/'combined_audit.json')==x['audit_sha256'] and sha(d/'sync_audit.json')==x['sync_audit_sha256']
 if x['label']=='pid':assert q['nn_final_audit']['nn_cycles']==0
 else:assert q['nn_final_audit']['model_id']==protocol['actors'][x['label']]['model_id'] and q['nn_final_audit']['pid_cycles']==0
 checks.append(dict(id=x['id'],label=x['label'],case=x['case'],seed=x['seed'],steps=2048,all_audits=True,result_sha256=sha(d/'result.json')))
sys.path.insert(0,str(T/'runtime'));from v46_core import RPActor
from v46_offline_scheduler import binary_bytes
e=protocol['actors']['candidate'];actor=RPActor(e['npz']);anchor=RPActor(protocol['actors']['A']['npz']);cp=torch.load(T/'training/batch_3/recovery.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(v,cp['state']['actor'][k]) for k,v in actor.state_dict().items()) and all(torch.equal(v,anchor.state_dict()[k]) for k,v in actor.named_buffers())
raw,mid=binary_bytes(actor);assert raw==Path(e['binary']).read_bytes() and mid==251305383 and hashlib.sha256(raw).hexdigest()==e['sha256']=='46484f17d3cd42dd3326034d8d4ddf1484f2128fb726a01e52fdcdcfdbd18235'
rows=[json.loads(x) for b in range(4) for x in (T/f'training/batch_{b}/updates.jsonl').read_text().splitlines()];assert [x['global_update'] for x in rows]==list(range(1,41)) and all(x['accepted'] for x in rows) and {int(x['step']) for x in cp['state']['optimizer']['state'].values()}=={40}
for path,h in j(R/'frozen_inputs.json').items():assert sha(Path(path))==h
manifest=j(a.baseline/'snapshots/v48/source_manifest.json');matched=0
for item in manifest['files']:
 if item['path'].startswith('snapshots/v48/backtracking_revision3/'):
  source=Path(item['source'].replace('@DRL_ROOT_WINDOWS@','/mnt/e/drl'));assert sha(source)==item['source_sha256'] and sha(a.baseline/item['path'])==item['snapshot_sha256'];matched+=1
# Repeat the frozen review into a separate folder; no existing review/ledger overwrite.
for name in ['frozen_design.json','amendment.json','protocol.json']:shutil.copyfile(D/name,O/name)
spec=importlib.util.spec_from_file_location('frozen_review',D/'runtime/review54.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.E=O
replayed=module.review(dict(completed=completed,fully_audited_count=54,actual_native_nn_calls=sum(x['actual_native_NN'] for x in dl['attempts']),actual_native_pid_calls=sum(x['actual_native_PID'] for x in dl['attempts'])))
assert replayed==j(D/'complete_review.json'),'Frozen review replay differs'
assert all(sha(Path(k))==v for k,v in input_hashes.items())
report=dict(passed=True,episodes=54,exact_matrix=True,all_export_capture_sync_audits=True,all_full2048=True,failed_attempts=0,training_charge=1213571,development_charge=2397578,no_reservations=True,final_model_id=mid,final_model_sha256=e['sha256'],contiguous_updates=40,Adam_steps=40,baseline_commit='ff7aa38f21ffa8e888ce69fe6f3137f644410cb4',baseline_manifest_sources_matched=matched,normalized_snapshot_hashes_checked_separately=True,frozen_review_exact_replay=True,performance_conditions_pass=replayed['performance_conditions_pass'],promoted=False,heldout_used=False,native_calls_from_audit=0,checks=checks)
(O/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
