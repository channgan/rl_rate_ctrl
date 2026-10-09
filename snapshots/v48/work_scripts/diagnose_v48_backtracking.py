from pathlib import Path
import sys,json,copy,hashlib,traceback
import numpy as np,torch
V=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');R=V/'memory_gate_revision2';T=R/'training';B=T/'training/batch_1';O=R/'offline_backtracking_diagnosis3';O.mkdir(exist_ok=False)
sys.path.insert(0,str(T/'runtime'))
import v46_offline_scheduler as core
from v46_core import RPActor,read_capture
from adaptive_smooth import AdaptiveScheduler
from v48_configuration import configure_optimizer
j=lambda p:json.loads(p.read_text());put=lambda p,v:p.write_text(json.dumps(v,indent=2));torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
protected=[V/'training/ledger.json',V/'development/ledger.json',B/'recovery.pt',B/'pre_update_recovery.pt',B/'interrupted_recovery.pt',B/'failure_details.json',B/'constraint_failure_details.json']
hashes={str(p):core.digest(p) for p in protected}
report=dict(passed=False,native_calls=0,formal_updates=0)
try:
 p=j(T/'protocol.json');m=j(T/'batch_1_captures.json');old=torch.load(T/'training/batch_0/recovery.pt',map_location='cpu',weights_only=False)
 # Reuse the exact admitted construction, never its training/test main loop.
 setup=Path('@WORK_SCRIPTS_LINUX@/test_v48_memory_revision2.py').read_text()
 setup=setup[setup.index(" actor=RPActor(m['actor']['npz'])"):setup.index(' before={k:v.clone()')]
 setup=setup[:setup.index(' entries=')]+setup[setup.index(" g=np.load(p['guard_corpus'])"):]
 import textwrap
 exec(textwrap.dedent(setup),globals())
 cp=torch.load(B/'pre_update_recovery.pt',map_location='cpu',weights_only=False);s.restore(cp['state']);start=s.snapshot();s.proposal_diagnostics=[]
 before=s.diagnostics();assert before['accepted'] and s.updates==7
 strict=lambda a,b:float((a.detach().double()-b.detach().double()).abs().max())
 report['step17']=dict(diagnostics=before,batch_bias_float64=strict(s.actor.rp_b,s.initial['rp_b']),A_bias_float64=strict(s.actor.rp_b,A.rp_b),Adam_steps=[int(x['step']) for x in s.opt.state_dict()['state'].values()])
 try:s.step()
 except RuntimeError as e:report['original_failure']=str(e)
 else:raise AssertionError('original failure did not reproduce')
 replay=s.proposal_diagnostics[-4:];saved=j(B/'constraint_failure_details.json')[-4:]
 assert [r['bias_change'] for r in replay]==[r['bias_change'] for r in saved]
 assert all(torch.equal(v,old['state']['actor'][k]) for k,v in s.actor.state_dict().items())
 assert all(torch.equal(v,s.opt.state_dict()['state'][k][key]) for k,x in old['state']['optimizer']['state'].items() for key,v in x.items())
 report['original_replay_exact']=True;report['rollback_actor_and_Adam_equal_step10']=True
 source=(T/'runtime/v46_offline_scheduler.py').read_text();patched=source.replace('for scale in (1.,.5,.25,.125):','for scale in tuple(2.**-i for i in range(13)):')
 patched=patched.replace("previous_cost=np.asarray(self.diagnostics()['cost_delta']);before=self.snapshot();", "entry=self.diagnostics();assert entry['accepted'],'Infeasible starting actor';previous_cost=np.asarray(entry['cost_delta']);before=self.snapshot();")
 patched=patched.replace("if report['accepted']:\n    self.updates", "strict_bias=float((self.actor.rp_b.double()-self.initial['rp_b'].double()).abs().max())\n   nonzero=float((flat(self.actor).double()-v.double()).abs().max())>=1e-10\n   report['strict_batch_bias']=strict_bias;report['effective_nonzero']=nonzero\n   if report['accepted'] and strict_bias<=1e-4-1e-10 and nonzero:\n    self.updates")
 patched=patched.replace('All four proposal scales failed; restored A, version stopped','All fixed proposal scales rejected; restored batch-start actor/optimizer/RNG, version stopped; last accepted checkpoint preserved')
 (O/'v46_offline_scheduler_proposed.py').write_text(patched)
 method=patched[patched.index(' def _step(self):'):patched.index(' def save(self,path):')];ns={};exec(textwrap.dedent(method),core.__dict__,ns);core.Scheduler._step=ns['_step']
 # The global-A guard also gets a conservative float64 check, with no increased tolerance.
 original_diag=S.diagnostics
 def diagnostic(self):
  r=original_diag(self);r['strict_A_bias']=strict(self.actor.rp_b,A.rp_b);r['accepted']=r['accepted'] and r['strict_A_bias']<=1e-4-1e-10;return r
 S.diagnostics=diagnostic
 s.restore(copy.deepcopy(start));s.proposal_diagnostics=[];result=s.step();accepted=s.snapshot();assert result['accepted'] and result['effective_nonzero'] and s.updates==8
 assert float((core.flat(s.actor)-torch.cat([start['actor'][k].flatten() for k,_ in s.actor.named_parameters()])).abs().max())>0
 report['extended_update18']=result
 s.restore(copy.deepcopy(start));again=s.step();report['replay_metric_differences']={k:[result[k],again[k]] for k in result if result[k]!=again[k]}
 assert all(torch.equal(v,accepted['actor'][k]) for k,v in s.actor.state_dict().items())
 assert all(torch.equal(v,s.opt.state_dict()['state'][k][key]) for k,x in accepted['optimizer']['state'].items() for key,v in x.items())
 report['deterministic_actor_Adam_replay']=True
 # Reject a zero proposal: it must not increment accepted updates.
 s.restore(copy.deepcopy(start));project=core.project_halfspaces;core.project_halfspaces=lambda proposal,*args:torch.zeros_like(proposal)
 try:s.step()
 except RuntimeError as e:assert 'batch-start' in str(e)
 else:raise AssertionError('zero proposal accepted')
 finally:core.project_halfspaces=project
 assert s.updates==7 and s.stopped;report['zero_proposal_rejected_without_success_increment']=True
 assert all(torch.equal(v,old['state']['actor'][k]) for k,v in s.actor.state_dict().items());report['corrected_log_checkpoint_identity_pass']=True
 report.update(passed=True,fixed_scales=[2.**-i for i in range(13)],minimum_scale=2.**-12,bias_acceptance_limit=1e-4-1e-10,minimum_parameter_change=1e-10,protected_unchanged=all(core.digest(Path(k))==v for k,v in hashes.items()),proposal_only_not_installed=True)
 assert report['protected_unchanged']
except BaseException as e:
 report.update(error=repr(e),traceback=traceback.format_exc());put(O/'report.json',report);raise
put(O/'report.json',report);print(json.dumps(report,indent=2),flush=True)
