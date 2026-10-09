from pathlib import Path
import json,hashlib,sys,subprocess,ast
O=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/v48_lr_offline_admission1');T=O/'prepared_training';R=T/'runtime'
helper='''from pathlib import Path
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
 if previous is not None:opt.load_state_dict(previous)
 for group in opt.param_groups:group['lr']=lr
 return copy.deepcopy(opt.state_dict())
'''
(R/'v48_configuration.py').write_text(helper)
f=R/'update_batch.py';x=f.read_text();x=x.replace("from adaptive_smooth import AdaptiveScheduler as Scheduler","from adaptive_smooth import AdaptiveScheduler as Scheduler\nfrom v48_configuration import validate_fresh,configure_optimizer")
x=x.replace("actor=RPActor(m['actor']['npz']);anchor=", "validate_fresh(ROOT,batch,m,p)\nactor=RPActor(m['actor']['npz']);anchor=")
x=x.replace("and revision['initial_seven_entry_gate_passed']","and all(c['passed'] for c in json.loads((ROOT/'batch_0_coverage.json').read_text()))")
x=x.replace(" s.opt.load_state_dict(old['state']['optimizer'])\n s.initial_opt=copy.deepcopy(s.opt.state_dict())\nfor group in s.opt.param_groups:group['lr']=p['learning_rate']\ns.initial_opt=copy.deepcopy(s.opt.state_dict())","s.initial_opt=configure_optimizer(s.opt,old['state']['optimizer'] if old else None,p['learning_rate'])")
f.write_text(x)
p=json.loads((T/'protocol.json').read_text());p['memory_sampling_revision']['initial_seven_entry_gate_passed']=False;p['prepared_python_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in R.glob('*.py')};(T/'protocol.json').write_text(json.dumps(p,indent=2))
tests='''import unittest,tempfile,json,sys,copy,subprocess
from pathlib import Path
import torch
T=Path(__file__).parent/'prepared_training';sys.path.insert(0,str(T/'runtime'))
from v48_configuration import validate_fresh,configure_optimizer
class Checks(unittest.TestCase):
 def fixture(self,root):
  jobs=[];paths=[]
  for i in range(7):
   job=dict(id=f'b0p{i}',batch=0,seed=100+i,case='hover');jobs.append(job);d=root/'jobs'/job['id'];d.mkdir(parents=True);p=d/'result.json';p.write_text(json.dumps(dict(seed=100+i,case='hover',nn_final_audit=dict(model_id=123))));(d/'attempt.json').write_text(json.dumps(dict(status='audited',batch=0,actor=dict(model_id=123))));paths.append(str(p))
  return dict(jobs=jobs),dict(results=paths,actor=dict(model_id=123))
 def test_fresh(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p,m=self.fixture(root);validate_fresh(root,0,m,p)
 def test_historical_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p,m=self.fixture(root);m['results'][0]='/historical/result.json'
   with self.assertRaises(AssertionError):validate_fresh(root,0,m,p)
 def test_duplicate_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p,m=self.fixture(root);m['results'][0]=m['results'][1]
   with self.assertRaises(AssertionError):validate_fresh(root,0,m,p)
 def test_wrong_model_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p,m=self.fixture(root);m['actor']['model_id']=124
   with self.assertRaises(AssertionError):validate_fresh(root,0,m,p)
 def test_wrong_seed_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p,m=self.fixture(root);p['jobs'][0]['seed']=1
   with self.assertRaises(AssertionError):validate_fresh(root,0,m,p)
 def test_new_adam_empty(self):
  x=torch.nn.Parameter(torch.ones(2));opt=torch.optim.Adam([x],lr=3e-6);q=configure_optimizer(opt,None,3e-5);self.assertEqual(q['state'],{});self.assertEqual(q['param_groups'][0]['lr'],3e-5)
 def test_adam_moments_retained_lr_overridden(self):
  x=torch.nn.Parameter(torch.ones(2));opt=torch.optim.Adam([x],lr=3e-6);x.sum().backward();opt.step();old=copy.deepcopy(opt.state_dict());new=torch.optim.Adam([x],lr=3e-6);q=configure_optimizer(new,old,3e-5);self.assertTrue(torch.equal(q['state'][0]['exp_avg'],old['state'][0]['exp_avg']));self.assertEqual(q['param_groups'][0]['lr'],3e-5)
 def test_unapproved_native_blocked(self):
  r=subprocess.run([sys.executable,'-B',str(T/'entry.py'),'--run'],capture_output=True,text=True);self.assertNotEqual(r.returncode,0);self.assertIn('not been authorized',r.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
'''
(O/'test_prepared.py').write_text(tests)
for f in [T/'entry.py',*R.glob('*.py')]:ast.parse(f.read_text())
v=subprocess.run([sys.executable,'-B',str(T/'entry.py')],capture_output=True,text=True);print(v.stdout,v.stderr);assert v.returncode==0
r=subprocess.run([sys.executable,'-B',str(O/'test_prepared.py')],capture_output=True,text=True);(O/'test_prepared.log').write_text(r.stdout+r.stderr);print(r.stdout+r.stderr);assert r.returncode==0
(O/'preparation.json').write_text(json.dumps(dict(root=str(T),authorized=False,native_calls=0,unit_tests=8,tests_passed=True,protocol_sha256=hashlib.sha256((T/'protocol.json').read_bytes()).hexdigest()),indent=2))
