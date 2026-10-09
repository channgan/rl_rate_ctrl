"""Exercise actual public admission, mocking only the final process launch."""
from pathlib import Path
import argparse,sys,json,tempfile,hashlib,unittest
from unittest.mock import patch
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
sys.path.insert(0,str(args.root/'launcher'))
from rate_rl.independent_batch import start
from rate_rl.startup_entry import AdmissionError
from rate_rl import _holder_gateway
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
put=lambda p,v:p.write_text(json.dumps(v))

class Checks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.work=self.root/'work';self.runtime=self.work/'runtime';self.runtime.mkdir(parents=True)
  self.source=self.runtime/'batch_driver.py';self.source.write_text('# isolated admission unit fixture; never executed\n')
  self.p=dict(authorized=True,version='v49-active-projection',native_limit=1316000);self.ledger=dict(native_limit=1316000,stopped=False,reservations={},attempts=[],charged_native=0)
  put(self.work/'protocol.json',self.p);put(self.work/'ledger.json',self.ledger);put(self.root/'windows_config.json',dict(owner_script=str(self.source),linux_root=str(self.work)))
  self.validation=self.root/'validation.json';put(self.validation,dict(passed=True));self.old=self.root/'old.json';put(self.old,dict(unchanged=True))
  self.policy=dict(approved=True,experiment='v49_active_projection',fixed_native_limit=1316000,root=str(self.root),work_root=str(self.work),runtime=str(self.runtime),sha256={},source_commit='test-only-not-a-real-commit',offline_validation=str(self.validation),required_zero_nn_results=[str(self.validation)],old_ledger=str(self.old),old_ledger_sha256=sha(self.old))
  self.proof=dict(remote_verified=True,commit=self.policy['source_commit'],live_files_sha256={str(self.source):sha(self.source)})
  put(self.root/'source_provenance.json',self.proof)
 def run_start(self,valid=False):
  with patch.object(_holder_gateway,'launch',return_value={'unit_fixture_only':True}) as launch:
   if valid:self.assertTrue(start(self.policy,self.root/'policy.json',self.root)['unit_fixture_only']);launch.assert_called_once()
   else:
    with self.assertRaises(AdmissionError):start(self.policy,self.root/'policy.json',self.root)
    launch.assert_not_called()
 def test_exact_tighter_budget(self):self.run_start(True)
 def test_protocol_budget_increase_rejected(self):self.p['native_limit']=2000000;put(self.work/'protocol.json',self.p);self.run_start()
 def test_ledger_budget_increase_rejected(self):self.ledger['native_limit']=2000000;put(self.work/'ledger.json',self.ledger);self.run_start()
 def test_policy_budget_increase_rejected(self):self.policy['fixed_native_limit']=2000000;self.run_start()
 def test_unverified_remote_rejected(self):self.proof['remote_verified']=False;put(self.root/'source_provenance.json',self.proof);self.run_start()
 def test_wrong_commit_rejected(self):self.proof['commit']='wrong';put(self.root/'source_provenance.json',self.proof);self.run_start()
 def test_changed_source_rejected(self):self.source.write_text('changed');self.run_start()
 def test_repeat_launch_rejected(self):put(self.root/'holder_identity.json',{});self.run_start()
 def test_unsettled_reservation_rejected(self):self.ledger['reservations']={'a':{'cap':47000}};put(self.work/'ledger.json',self.ledger);self.run_start()
 def test_failed_offline_admission_rejected(self):put(self.validation,dict(passed=False));self.run_start()
 def test_old_ledger_change_rejected(self):put(self.old,dict(unchanged=False));self.run_start()

result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
report=dict(passed=result.wasSuccessful(),tests=result.testsRun,native_calls=0,process_launches=0,last_launch_mocked=True)
(args.root/'offline/windows_admission_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));raise SystemExit(0 if result.wasSuccessful() else 1)
