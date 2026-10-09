"""Full offline update_batch entry tests on clearly labelled historical fixtures."""
from pathlib import Path
import argparse,copy,json,os,shutil,subprocess,sys,traceback,hashlib
import numpy as np,torch

def same(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--v48',type=Path,required=True);a=ap.parse_args()
 V=a.root;T=V/'training';OLD=a.v48/'backtracking_revision3/training';O=V/'offline/integration';O.mkdir(exist_ok=False)
 j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();put=lambda p,v:p.write_text(json.dumps(v,indent=2))
 protected={str(p):sha(p) for p in [T/'ledger.json',V/'development/ledger.json',T/'protocol.json',*T.glob('runtime/*.py')]}
 report=dict(passed=False,native_calls=0,formal_updates=0,fixture_only=True,disposable_optimizer_attempts=0)
 try:
  manifests=j(OLD/'batch_0_captures.json');results=[]
  for label in ['first','replay']:
   root=O/label;root.mkdir();shutil.copytree(T/'runtime',root/'runtime');p=j(T/'protocol.json');p['canonical_ledger']=str(root/'ledger.json');p['offline_test_fixture']=True;p['effective_rules_file']=str(T/'effective_rules.json')
   paths=[]
   for i,source in enumerate(manifests['results']):
    source=Path(source);job=p['jobs'][i];dest=root/'jobs'/job['id']/'fixture';dest.mkdir(parents=True)
    # Hard-link immutable capture inputs only; updater writes exclusively outside jobs.
    for f in source.parent.iterdir():
     if f.is_file():os.link(f,dest/f.name)
    actual=j(source);job['seed']=actual['seed'];job['case']=actual['case']
    put(dest.parent/'attempt.json',dict(status='audited',batch=0,actor=manifests['actor']))
    paths.append(str(dest/'result.json'))
   put(root/'protocol.json',p);put(root/'batch_0_captures.json',dict(actor=manifests['actor'],results=paths));put(root/'ledger.json',dict(native_limit=1316000,charged_native=0,formal_optimizer_updates=0,attempts=[],reservations={},stopped=False,offline_test_fixture=True))
   with (root/'update.log').open('x') as log:run=subprocess.run([sys.executable,'-B',str(root/'runtime/update_batch.py'),str(root),'0'],stdout=log,stderr=subprocess.STDOUT)
   if run.returncode:
    report[label+'_failure_log']=(root/'update.log').read_text()[-8000:];raise AssertionError('Integrated entry failed: '+label)
   cp=torch.load(root/'training/batch_0/recovery.pt',map_location='cpu',weights_only=False)
   rows=[json.loads(line) for line in (root/'training/batch_0/updates.jsonl').read_text().splitlines()]
   assert [r['global_update'] for r in rows]==list(range(1,11)) and len(rows)==10
   assert cp['initial_optimizer']['state']=={} and cp['initial_optimizer']['param_groups'][0]['lr']==3e-5
   assert {int(v['step']) for v in cp['state']['optimizer']['state'].values()}=={10}
   assert all(r['accepted'] and r['parameter_delta_max']>=1e-10 and r['action_delta_previous_max']>0 and r['projection_solver']['passed'] for r in rows)
   alphas=[r['smooth_objective']['cumulative_multiplier'] for r in rows];assert alphas[0]<1 and all(y<=x for x,y in zip(alphas,alphas[1:]))
   assert j(root/'ledger.json')['formal_optimizer_updates']==10 and j(root/'ledger.json')['charged_native']==0
   results.append((root,cp,rows));report['disposable_optimizer_attempts']+=cp['attempts']
  assert same(results[0][1]['state'],results[1][1]['state'])
  assert results[0][2]==results[1][2]
  root,cp,rows=results[0]
  sys.path.insert(0,str(T/'runtime'))
  import v46_offline_scheduler as core
  import bias_projection_active
  from v46_core import RPActor,read_capture
  from v49_scheduler import V49Scheduler
  from v48_configuration import configure_optimizer
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
  p=j(root/'protocol.json');m=j(root/'batch_0_captures.json');actor=RPActor(m['actor']['npz']);A=RPActor(p['initial_actor']['npz']);qs=[read_capture(path) for path in m['results']]
  g=np.load(p['guard_corpus']);gx=torch.tensor(g['obs'],dtype=torch.float32);dt=torch.tensor(g['dt'],dtype=torch.float32)
  with torch.no_grad():gh=actor.features(gx);gm=actor(gx,dt);am=A(gx,dt)
  s=V49Scheduler(actor,core.prepare(actor,qs,p['scenario_groups']),core.digest(root/'protocol.json'),gh,gm,A,am)
  s.initial_opt=configure_optimizer(s.opt,None,3e-5);s.attach(RPActor(m['actor']['npz']),qs,p['scenario_groups'])
  pre=torch.load(root/'training/batch_0/pre_update_recovery.pt',weights_only=False,map_location='cpu')
  s.initial_rng=copy.deepcopy(pre['initial_rng']);s.initial_gen=pre['initial_generator'].clone();s.restore(copy.deepcopy(pre['state']))
  assert same(s.snapshot(),pre['state']) and s.updates==9
  r=s.step();report['disposable_optimizer_attempts']+=1
  assert same(s.actor.state_dict(),cp['state']['actor']) and same(s.opt.state_dict(),cp['state']['optimizer']) and same(core.rng_state(),cp['state']['rng']) and same(s.gen.get_state(),cp['state']['generator'])
  export=j(root/'training/batch_0/export.json');reloaded=RPActor(export['npz']);raw,mid=core.binary_bytes(reloaded)
  assert same(reloaded.state_dict(),cp['state']['actor']) and raw==Path(export['binary']).read_bytes() and mid==export['model_id']
  assert all(torch.equal(v,A.state_dict()[k]) for k,v in reloaded.named_buffers())
  with torch.no_grad():assert torch.equal(reloaded(gx,dt),s.actor(gx,dt))
  before=copy.deepcopy(pre['state']);method=s.project_proposal
  for mode in ['zero','solver_exhaustion']:
   s.restore(copy.deepcopy(before));s.initial_opt=copy.deepcopy(pre['initial_optimizer'])
   if mode=='zero':s.project_proposal=lambda proposal,gradients:torch.zeros_like(proposal)
   else:s.project_proposal=method;bias_projection_active.MAX_CYCLES=0
   try:s.step()
   except RuntimeError as e:assert ('All 13' if mode=='zero' else 'iteration limit') in str(e)
   else:raise AssertionError('Rejected update counted: '+mode)
   finally:bias_projection_active.MAX_CYCLES=512;s.project_proposal=method
   report['disposable_optimizer_attempts']+=1
   assert s.stopped and s.updates==9 and same(s.actor.state_dict(),s.initial) and same(s.opt.state_dict(),s.initial_opt) and same(core.rng_state(),s.initial_rng) and same(s.gen.get_state(),s.initial_gen) and not s.perms and not s.positions
  s.restore(copy.deepcopy(before));s.initial_opt=copy.deepcopy(pre['initial_optimizer']);buffer=next(s.actor.buffers())
  with torch.no_grad():buffer.flatten()[0]+=1
  count=core.COUNTERS['optimizer_attempts']
  try:s.step()
  except RuntimeError as e:assert 'Frozen tensor altered' in str(e)
  else:raise AssertionError('Frozen mutation admitted')
  assert core.COUNTERS['optimizer_attempts']==count and same(s.actor.state_dict(),s.initial)
  assert all(sha(Path(path))==h for path,h in protected.items())
  report.update(passed=True,full_entry_updates_per_replay=10,fresh_Adam=True,first_update_adaptive_alpha=rows[0]['smooth_objective']['cumulative_multiplier'],alpha_nonincreasing=True,
                exact_actor_Adam_RNG_permutations_cross_process=True,resume_saved_pre_update_exact=True,model_npz_binary_checkpoint_forward_exact=True,
                frozen_mutation_rejected_before_optimizer=True,zero_and_solver_exhaustion_rollback_exact=True,formal_files_unchanged=True,
                fixture_model_id=mid,scales=[r['scale'] for r in rows],source_hashes=protected)
 except BaseException as e:
  report.update(error=repr(e),traceback=traceback.format_exc());put(O/'admission.json',report);raise
 put(O/'admission.json',report);print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},indent=2))

if __name__=='__main__':main()
