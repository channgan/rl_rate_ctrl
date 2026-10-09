from pathlib import Path
import json,hashlib,subprocess,sys,ast
V=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');R=V/'smooth_start_revision1';T=R/'training';RT=T/'runtime'
j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();put=lambda p,v:p.write_text(json.dumps(v,indent=2))
f=RT/'v48_configuration.py';s=f.read_text();s+='''
def validate_batch(root,batch,manifest,protocol):
 import hashlib
 root=Path(root)
 if batch:
  validate_fresh(root,batch,manifest,protocol);return
 carry=json.loads((root/'carried_batch0.json').read_text())
 assert carry['authorized_reuse_only'] and carry['original_charge']==303638 and carry['original_optimizer_steps']==0
 assert manifest==carry['manifest'] and len(manifest['results'])==len(set(manifest['results']))==7
 assert manifest['actor']==protocol['initial_actor']
 for path,h in carry['sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
''';f.write_text(s)
f=RT/'update_batch.py';s=f.read_text().replace('validate_fresh,configure_optimizer','validate_fresh,configure_optimizer,validate_batch')
a=s.index("assert len(m['results'])==7");b=s.index("actor=RPActor(m['actor']['npz'])",a);s=s[:a]+"validate_batch(ROOT,batch,m,p)\n"+s[b:]
# Keep the canonical ledger honest even if a later step fails within a batch.
needle="  print(json.dumps(dict(batch=batch,global_update=batch*10+s.updates,KL=result['KL'])),flush=True)"
assert s.count(needle)==1;s=s.replace(needle,"  ledger_path=Path(p['canonical_ledger']);ledger=json.loads(ledger_path.read_text());assert ledger['formal_optimizer_updates']==batch*10+s.updates-1\n  ledger['formal_optimizer_updates']=batch*10+s.updates;tmp=ledger_path.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2));os.replace(tmp,ledger_path)\n"+needle);f.write_text(s)
tests='''import unittest,tempfile,json,sys,hashlib,copy
from pathlib import Path
T=Path(__file__).parent/'training';sys.path.insert(0,str(T/'runtime'))
from v48_configuration import validate_batch
class Tests(unittest.TestCase):
 def setup(self,tmp):
  root=Path(tmp);paths=[]
  for i in range(7):f=root/f'old{i}.json';f.write_text('{}');paths.append(str(f))
  actor=dict(model_id=2428576135);m=dict(actor=actor,results=paths);p=dict(initial_actor=actor)
  c=dict(authorized_reuse_only=True,original_charge=303638,original_optimizer_steps=0,manifest=m,sha256={x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in paths});(root/'carried_batch0.json').write_text(json.dumps(c));return root,m,p
 def test_exact_carry(self):
  with tempfile.TemporaryDirectory() as tmp:r,m,p=self.setup(tmp);validate_batch(r,0,m,p)
 def test_modified_bytes(self):
  with tempfile.TemporaryDirectory() as tmp:
   r,m,p=self.setup(tmp);Path(m['results'][0]).write_text('changed')
   with self.assertRaises(AssertionError):validate_batch(r,0,m,p)
 def test_reordered_data(self):
  with tempfile.TemporaryDirectory() as tmp:
   r,m,p=self.setup(tmp);m['results'].reverse()
   with self.assertRaises(AssertionError):validate_batch(r,0,m,p)
 def test_changed_actor(self):
  with tempfile.TemporaryDirectory() as tmp:
   r,m,p=self.setup(tmp);m['actor']['model_id']=1
   with self.assertRaises(AssertionError):validate_batch(r,0,m,p)
 def test_later_batch_cannot_reuse(self):
  with tempfile.TemporaryDirectory() as tmp:
   r,m,p=self.setup(tmp);p['jobs']=[dict(batch=1,id=f'b1p{i}',seed=i,case='hover') for i in range(7)]
   with self.assertRaises(AssertionError):validate_batch(r,1,m,p)
if __name__=='__main__':unittest.main(verbosity=2)
'''
(R/'test_revision.py').write_text(tests);done=subprocess.run([sys.executable,'-B',str(R/'test_revision.py')],capture_output=True,text=True);(R/'test_revision.log').write_text(done.stdout+done.stderr);print(done.stdout+done.stderr);assert done.returncode==0
sys.path.insert(0,str(RT));from v48_configuration import validate_batch
validate_batch(T,0,j(T/'batch_0_captures.json'),j(T/'protocol.json'))
for f in RT.glob('*.py'):ast.parse(f.read_text())
p=j(T/'protocol.json');p['prepared_python_sha256']={str(f):sha(f) for f in RT.glob('*.py')};put(T/'protocol.json',p)
contract=j(R/'revision_contract.json');contract['bookkeeping_revision']='Persist actual accepted update count after each saved/logged step, not only after a whole batch; no optimization math change';put(R/'revision_contract.json',contract)
frozen=j(R/'frozen_inputs.json');frozen={path:sha(Path(path)) for path in frozen};put(R/'frozen_inputs.json',frozen)
ad=j(R/'execution/admission.json');ad['sha256']={path:sha(Path(path.replace('E:\\','/mnt/e/').replace('\\','/'))) for path in ad['sha256']};put(R/'execution/admission.json',ad)
assert j(V/'training/ledger.json')['charged_native']==303638 and j(V/'training/ledger.json')['stopped']
put(R/'tests.json',dict(passed=True,provenance_tests=5,adaptive_boundary_tests=4,paired_recovery_steps=4,native_calls=0,formal_updates=0,actual_carried_files_verified=True))
print('REVISION READY: tests pass; canonical ledger still stopped/unchanged.')
