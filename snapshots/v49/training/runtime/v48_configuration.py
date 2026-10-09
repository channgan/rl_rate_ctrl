from pathlib import Path
import json,copy
def validate_fresh(root,batch,manifest,protocol):
 root=Path(root).resolve();paths=[Path(v).resolve() for v in manifest['results']]
 assert len(paths)==len(set(paths))==7, 'Exactly seven distinct fresh captures required'
 jobs=[j for j in protocol['jobs'] if j['batch']==batch]
 assert len(jobs)==7
 for path,job in zip(paths,jobs):
  jr=root/'jobs'/job['id'];assert path.is_relative_to(jr), 'Historical or wrong-batch capture rejected'
  attempt=json.loads((jr/'attempt.json').read_text());result=json.loads(path.read_text())
  assert attempt['status']=='audited' and attempt['batch']==batch and attempt['actor']['model_id']==manifest['actor']['model_id']
  assert result['seed']==job['seed'] and result['case']==job['case'] and result['nn_final_audit']['model_id']==manifest['actor']['model_id']
def configure_optimizer(opt,previous,lr):
 assert lr==3e-5
 if previous is not None:opt.load_state_dict(copy.deepcopy(previous))
 for group in opt.param_groups:group['lr']=lr
 return copy.deepcopy(opt.state_dict())

def validate_batch(root,batch,manifest,protocol):
 assert batch in range(4)
 validate_fresh(root,batch,manifest,protocol)
