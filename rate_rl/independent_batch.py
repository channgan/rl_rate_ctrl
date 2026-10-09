"""Admission only: independent task holder, no remote heartbeat or observer lease."""
from pathlib import Path
import os
from .startup_entry import AdmissionError,read,sha,local_path
def start(policy,policy_path,value):
 root=local_path(value).resolve();runtime=local_path(policy['runtime']).resolve();work=local_path(policy.get('work_root',policy['root'])).resolve()
 if os.name!='nt':raise AdmissionError('Windows independent holder required')
 if policy.get('approved') is not True or root!=local_path(policy['root']).resolve():raise AdmissionError('Exact approved root required')
 if any((root/n).exists() for n in ['standard_launch_intent.json','holder_identity.json','stop']):raise AdmissionError('Used run; no implicit retry')
 for name,digest in policy['sha256'].items():
  p=local_path(name)
  if sha(p)!=digest:raise AdmissionError('Approved input changed: '+name)
 cfg=read(root/'windows_config.json');protocol=read(work/'protocol.json');ledger=read(local_path(policy.get('canonical_ledger',str(work/'ledger.json'))))
 if local_path(cfg['owner_script']).resolve()!=runtime/'batch_driver.py' or local_path(cfg['linux_root']).resolve()!=work:raise AdmissionError('Worker/root mismatch')
 expected_limit=3000000 if policy.get('development_final40') else 2000000
 if not protocol.get('authorized') or protocol['native_limit']!=expected_limit or ledger['native_limit']!=expected_limit:raise AdmissionError('Fixed internal budget required')
 if policy.get('development_final40'):
  completed=read(local_path(protocol['training_complete']));validated=read(local_path(protocol['validation_complete']))
  if protocol.get('phase')!='development' or len(protocol['jobs'])!=(12 if policy.get('development_continuation') else 54) or not validated.get('passed'):raise AdmissionError('Fixed development scope or runtime qualification missing')
  if not completed.get('completed') or completed.get('updates')!=40 or completed.get('actor')!=protocol['actors']['candidate']:raise AdmissionError('Final update40 only')
  if protocol.get('promotion_allowed') is not False or protocol.get('exploration_std')!=0:raise AdmissionError('Frozen deterministic development only')
  if policy.get('development_continuation'):
   validate_development_continuation(protocol,ledger,work)
  elif not ledger.get('authorized') or ledger['charged_native']!=0 or ledger['attempts']:raise AdmissionError('Fresh separate development ledger required')

 if ledger['stopped'] or ledger['reservations']:raise AdmissionError('Unreconciled ledger')
 if (ledger['attempts'] or ledger['charged_native']) and not policy.get('development_continuation'):
  if policy.get('runtime_replacement'):
   import hashlib,json
   resume=read(work/'runtime_replacement.json')
   if not resume['approved'] or ledger['charged_native']!=resume['historical_charge'] or ledger['formal_optimizer_updates']!=resume.get('historical_optimizer_updates',0) or len(ledger['attempts'])!=resume.get('historical_attempt_count',5):raise AdmissionError('Runtime replacement history mismatch')
   if hashlib.sha256(json.dumps(ledger['attempts'],sort_keys=True).encode()).hexdigest()!=resume['historical_attempts_sha256']:raise AdmissionError('Historical attempts changed')
   if local_path(protocol['canonical_ledger']).resolve()!=local_path(policy['canonical_ledger']).resolve():raise AdmissionError('Canonical budget mismatch')
   if ledger['runtime_replacements'][-1]['id']!=protocol['sync_runtime_id']:raise AdmissionError('Runtime replacement not recorded')
  else:
   if not policy.get('technical_continuation'):raise AdmissionError('Explicit existing-ledger continuation required')
   resume=read(work/'technical_resume.json')
   if not resume['approved'] or ledger['charged_native']!=47000 or ledger['formal_optimizer_updates']!=0 or [a['id'] for a in ledger['attempts']]!=['batch1_case1']:raise AdmissionError('Continuation scope mismatch')
   for saved in resume['completed'].values():
    if sha(saved['result'])!=saved['result_sha256'] or sha(saved['audit'])!=saved['audit_sha256'] or not read(saved['audit'])['passed']:raise AdmissionError('Existing capture revalidation mismatch')
 if not protocol.get('zero_nn_fixture'):
  if read(policy['offline_validation']).get('passed') is not True:raise AdmissionError('Offline training validation incomplete')
  for p in policy['required_zero_nn_results']:
   if read(p).get('passed') is not True:raise AdmissionError('Independent entry zero-NN validation incomplete')
  if sha(policy['old_ledger'])!=policy['old_ledger_sha256']:raise AdmissionError('Original ledger changed')
 from ._holder_gateway import launch
 return launch(root,runtime,Path(policy_path).resolve())

def status(policy):
 root=local_path(policy.get('work_root',policy['root']));result={'read_only':True,'monitor_degraded':False,'files':{}}
 for name in ['status.json','ledger.json','driver_identity.json','holder_completed.json','training_complete.json','failure.json']:
  try:result['files'][name]=read((local_path(policy['root']) if name=='holder_completed.json' else root)/name)
  except (AdmissionError,OSError) as e:
   result['files'][name]={'unavailable':str(e)}
   if name=='status.json':result['monitor_degraded']=True
 return result

def stop(policy,value):
 root=local_path(value).resolve()
 if root!=local_path(policy['root']).resolve():raise AdmissionError('Unapproved stop root')
 if (root/'holder_completed.json').exists():return {'already_closed':True,'changed':False}
 root=local_path(policy.get('work_root',policy['root']))
 try:
  with (root/'stop').open('x') as f:f.write('explicit user cooperative stop')
 except FileExistsError:pass
 return {'stop_requested':True,'force_kill':False,'await_internal_save_and_export':True}


def validate_development_continuation(protocol,ledger,work):
 import hashlib,json
 c=protocol.get('development_continuation',{})
 if not c.get('approved') or not c.get('once_only') or not ledger.get('authorized'):raise AdmissionError('Explicit once-only continuation required')
 if ledger['charged_native']!=1957486 or len(ledger['attempts'])!=43 or ledger['reservations']:raise AdmissionError('Exact development continuation history required')
 if hashlib.sha256(json.dumps(ledger['attempts'],sort_keys=True).encode()).hexdigest()!=c['historical_attempts_sha256']:raise AdmissionError('Historical attempts altered')
 if sum(x['status']=='audited' for x in ledger['attempts'])!=42 or ledger['attempts'][-1]['id']!='dev_43' or ledger['attempts'][-1]['charge']!=56000:raise AdmissionError('Original failure/42 results missing')
 original=c['original_jobs'];jobs=protocol['jobs']
 if len(original)!=54 or len(jobs)!=12 or jobs[0]['id']!='dev_43_recovery1':raise AdmissionError('Exact remaining matrix required')
 for i,(actual,expected) in enumerate(zip(jobs,original[42:])):
  q=dict(actual)
  if i==0:
   if q.pop('logical_id',None)!='dev_43' or q.pop('authorized_recovery_attempt',None)!=1:raise AdmissionError('Only one explicit recovery')
   q['id']='dev_43'
  if q!=expected:raise AdmissionError('Changed remaining case/model/seed/budget')
 if ledger['charged_native']+sum(j['cap'] for j in jobs)>3000000:raise AdmissionError('Development budget exceeded')
 carried=read(work/'carried42.json')
 if len(carried['completed'])!=42 or len({(r['case'],r['seed'],r['label']) for r in carried['completed']})!=42:raise AdmissionError('Carried matrix invalid')
 for path,h in carried['sha256'].items():
  if sha(local_path(path))!=h:raise AdmissionError('Carried result/audit changed')
