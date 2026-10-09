"""Fixed two-checkpoint offline admission. Never imports a native launcher."""
from pathlib import Path
import argparse, copy, hashlib, json, traceback
import numpy as np
import torch
import sys
from bias_projection import project, ProjectionFailure, LIMIT, MAX_CYCLES, SOLVER_TOL, CAST_TOL

def same(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def synthetic():
    results={}
    z=torch.zeros(2); p=torch.tensor([1.,-1.,1.,-1.])
    gs=[torch.tensor([1.,0.,1.,0.]),torch.zeros(4)]
    d,r=project(p,gs,z,z,z)
    assert float(gs[0].double().dot(d.double()))<=CAST_TOL
    assert (d[-2:].double().abs()<=LIMIT).all()
    results['cost_box_intersection']=r
    d,_=project(torch.zeros(4),gs,z,z,z); assert torch.equal(d,torch.zeros(4)); results['zero_preserved']=True
    d,_=project(torch.tensor([.01,0.,0.,0.]),[],z,z,z); assert d[0]==.01; results['feasible_preserved']=True
    for name,args in [
        ('nan', (p*float('nan'),gs,z,z,z)),
        ('infinity',(p,[p*float('inf')],z,z,z)),
        ('empty',(p,gs,z,z,torch.ones(2))),
        ('infeasible_start',(p,gs,torch.ones(2),z,z))]:
        try: project(*args)
        except ValueError: results[name+'_rejected']=True
        else: raise AssertionError(name)
    bound=torch.tensor(LIMIT,dtype=torch.float32)
    if float(bound)>LIMIT: bound=torch.nextafter(bound,torch.tensor(-float('inf')))
    cur=torch.stack([bound,-bound]);d,r=project(p,[],cur,z,z)
    assert ((cur+d[-2:]).double().abs()<=LIMIT).all()
    results['float32_boundary']=r
    return results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--v48',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    V=a.v48;T=V/'backtracking_revision3/training';O=a.out;O.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(T/'runtime'))
    import v46_offline_scheduler as core
    from v46_core import RPActor,read_capture
    from adaptive_smooth import AdaptiveScheduler
    from v48_configuration import configure_optimizer
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    j=lambda p:json.loads(p.read_text())
    put=lambda p,q:p.write_text(json.dumps(q,indent=2,allow_nan=False))
    points=[('pre18',1,V/'memory_gate_revision2/training/training/batch_1/pre_update_recovery.pt',V/'memory_gate_revision2/training/protocol.json',17),
            ('pre40',3,T/'training/batch_3/pre_update_recovery.pt',T/'protocol.json',39)]
    protected=[V/'training/ledger.json',V/'development/ledger.json']+[q[2] for q in points]+list((T/'runtime').glob('*.py'))+list((T/'training').glob('batch_*/recovery.pt'))
    hashes={str(p.relative_to(V)):core.digest(p) for p in protected}
    plan=dict(fixed_points=[dict(name=n,batch=b,checkpoint=str(cp.relative_to(V)),sha256=core.digest(cp),accepted_global=g) for n,b,cp,pr,g in points],
              modes=['original','bias_intersection'],primary_disposable_steps=4,replay_disposable_steps=4,rejection_disposable_steps=2,
              limit=LIMIT,max_cycles=MAX_CYCLES,solver_tolerance=SOLVER_TOL,cast_normalized_tolerance=CAST_TOL,
              scales=[2.**-i for i in range(13)],native_calls=0,formal_updates=0,
              objective_and_guards='unchanged',failure_policy='report every point; no retuning or new checkpoints',
              next_native=dict(authorized=False,start='A plus fresh Adam',lr=3e-5,captures=28,updates=40,training_planned_cap=1316000,training_hard_cap=2000000,development_episodes=54,development_planned_cap=2700000,development_hard_cap=3000000,no_retries=True))
    put(O/'frozen_plan.json',plan)
    report=dict(native_calls=0,formal_updates=0,optimizer_attempts=0,points={},synthetic={},passed=False)
    original=core.project_halfspaces
    try:
      report['synthetic']=synthetic()
      for name,b,cpp,pr,global_step in points:
        item={};report['points'][name]=item
        try:
          p=j(T/'protocol.json');m=j(T/f'batch_{b}_captures.json');q=torch.load(cpp,map_location='cpu',weights_only=False)
          old=torch.load(T/f'training/batch_{b-1}/recovery.pt',map_location='cpu',weights_only=False)
          assert q['config_hash']==core.digest(pr)
          actor=RPActor(m['actor']['npz']);A=RPActor(p['initial_actor']['npz'])
          assert same(actor.state_dict(),old['state']['actor'])
          captures=[read_capture(x) for x in m['results']]
          g=np.load(p['guard_corpus']);gx=torch.tensor(g['obs'],dtype=torch.float32);dt=torch.tensor(g['dt'],dtype=torch.float32)
          with torch.no_grad():gh=actor.features(gx);gm=actor(gx,dt);am=A(gx,dt)
          data=core.prepare(actor,captures,p['scenario_groups'])
          class S(AdaptiveScheduler):
            @torch.no_grad()
            def diagnostics(self):
              r=super().diagnostics()
              r['global_A_weight_relative']=float(((self.actor.rp_w-A.rp_w).norm(dim=1)/A.rp_w.norm(dim=1)).max())
              r['global_A_bias_change']=float(abs(self.actor.rp_b-A.rp_b).max())
              mu=torch.clamp(torch.nn.functional.linear(gh,self.actor.rp_w,self.actor.rp_b),-1,1)
              r['global_A_guard_action_change']=float(abs(mu-am[:,:2]).max())
              r['strict_A_bias']=float((self.actor.rp_b.double()-A.rp_b.double()).abs().max())
              r['accepted']=bool(r['accepted'] and r['global_A_weight_relative']<=.01 and r['global_A_bias_change']<=1e-4 and r['global_A_guard_action_change']<=.002 and r['strict_A_bias']<=LIMIT)
              return r
          s=S(actor,data,core.digest(T/'protocol.json'),gh,gm)
          s.initial_opt=copy.deepcopy(configure_optimizer(s.opt,copy.deepcopy(old['state']['optimizer']),p['learning_rate']))
          s.attach(RPActor(m['actor']['npz']),captures,p['scenario_groups'],old['state']['objective_state']['alpha'])
          assert same(q['initial'],s.initial) and q['guard_hash']==s.guard_hash and same(q['initial_optimizer'],s.initial_opt)
          s.initial_rng=copy.deepcopy(q['initial_rng']);s.initial_gen=q['initial_generator'].clone()
          s.restore(copy.deepcopy(q['state']));start=copy.deepcopy(s.snapshot())
          assert same(start,q['state']) and s.updates==global_step-b*10
          assert all(int(v['step'])==global_step for v in s.opt.state_dict()['state'].values())
          assert s.diagnostics()['accepted'] and s.frozen_ok()
          item['restore_exact']=True;item['batch_anchor_global_update']=b*10;item['alpha']=s.alpha
          v=core.flat(actor).clone();mean=s.means()[0].detach().clone()
          original_indices=s.indices; minibatches=[]; indices_holder={}
          def indices():
              ix=original_indices();indices_holder['ix']=ix.clone();indices_holder['weights']=s.minibatch_weights.clone()
              minibatches.append(hashlib.sha256(ix.numpy().tobytes()+s.minibatch_weights.numpy().tobytes()).hexdigest());return ix
          s.indices=indices
          def surrogate():
              ix=indices_holder['ix'];r=torch.exp(s.likelihood(ix)[0]-s.data['behavior'][ix]);adv=s.data['adv'][ix]
              return float((indices_holder['weights']*torch.minimum(r*adv,r.clamp(.95,1.05)*adv)).sum().detach())
          outcomes={}
          for mode in ['original','bias_intersection']:
            runs=[];outcomes[mode]=runs
            for replay in range(2):
              s.restore(copy.deepcopy(start));s.attempts=q['attempts'];s.proposal_diagnostics=[]
              folder=O/name/f'{mode}_{replay}';folder.mkdir(parents=True);s.folder=folder;projection={}
              def candidate(proposal,gradients,passes):
                  try:delta,info=project(proposal,gradients,s.actor.rp_b.detach(),s.initial['rp_b'],A.rp_b.detach())
                  except ProjectionFailure as e:projection.update(e.diagnostics);raise
                  projection.update(info);return delta
              core.project_halfspaces=original if mode=='original' else candidate
              before_count=core.COUNTERS['optimizer_attempts'];error=None;result=None
              try: result=s.step()
              except Exception as e: error=str(e)
              report['optimizer_attempts']+=core.COUNTERS['optimizer_attempts']-before_count
              end=copy.deepcopy(s.snapshot());after=surrogate()
              actual=core.flat(actor).clone();action=s.means()[0].detach().clone()
              s.restore(copy.deepcopy(start));before=surrogate();s.restore(copy.deepcopy(end))
              run=dict(accepted=result is not None,error=error,diagnostics=result,projection=projection,
                       minibatch_hash=minibatches[-1],surrogate_before=before,surrogate_after=after,surrogate_improvement=after-before,
                       parameter_delta_max=float((actual.double()-v.double()).abs().max()),
                       weight_delta_norm=(actor.rp_w.detach().double()-start['actor']['rp_w'].double()).norm(dim=1).tolist(),
                       action_delta_max=float((action.double()-mean.double()).abs().max()),
                       action_delta_rms=torch.sqrt(((action.double()-mean.double())**2).mean(0)).tolist(),
                       actor_sha256=hashlib.sha256(core.binary_bytes(actor)[0]).hexdigest(),frozen_ok=s.frozen_ok(),
                       adam_steps=[int(x['step']) for x in s.opt.state_dict()['state'].values()])
              if error:
                  run['deltas_describe']='batch-start rollback, NOT an accepted update'
                  run['effective_accepted_parameter_delta_max']=0.
                  run['effective_accepted_action_delta_max']=0.
                  run['rollback_exact']=same(end['actor'],s.initial) and same(end['optimizer'],s.initial_opt) and same(end['rng'],s.initial_rng) and same(end['generator'],s.initial_gen) and end['perms']=={} and end['positions']=={} and end['updates']==start['updates']
              runs.append(run)
              if replay==0:expected=copy.deepcopy(end)
              else:run['deterministic_full_state_exact']=same(end,expected);assert same(end,expected) and same(runs[0],{k:v for k,v in run.items() if k!='deterministic_full_state_exact'})
            item[mode]=runs
          assert len(set(minibatches))==1
          item['identical_minibatch_all_four_runs']=True
          # Forced zero proposal must exhaust all 13 guards and restore batch state.
          s.restore(copy.deepcopy(start));s.folder=O/name/'zero_rejection';s.folder.mkdir();core.project_halfspaces=lambda *args:torch.zeros_like(args[0])
          before_count=core.COUNTERS['optimizer_attempts']
          try:s.step()
          except RuntimeError as e:assert 'All 13' in str(e)
          else:raise AssertionError('Zero update admitted')
          report['optimizer_attempts']+=core.COUNTERS['optimizer_attempts']-before_count
          end=s.snapshot()
          assert same(end['actor'],s.initial) and same(end['optimizer'],s.initial_opt) and same(end['rng'],s.initial_rng) and same(end['generator'],s.initial_gen) and not end['perms'] and not end['positions'] and end['updates']==start['updates'] and end['stopped']
          item['zero_rejection_batch_actor_adam_rng_rollback_exact']=True
          candidate=outcomes['bias_intersection'][0]
          item['passed']=bool(candidate['accepted'] and candidate['surrogate_improvement']>0 and candidate['parameter_delta_max']>=1e-10 and candidate['action_delta_max']>0 and candidate['frozen_ok'] and outcomes['bias_intersection'][1]['deterministic_full_state_exact'])
        except Exception as e:item.update(passed=False,error=repr(e),traceback=traceback.format_exc())
        finally:core.project_halfspaces=original
        put(O/'admission.json',report);print(name,json.dumps(item),flush=True)
      report['protected_unchanged']=all(core.digest(V/p)==h for p,h in hashes.items())
      report['protected_hashes']=hashes
      report['passed']=len(report['points'])==2 and all(x.get('passed',False) for x in report['points'].values()) and report['protected_unchanged']
    except Exception as e:report.update(error=repr(e),traceback=traceback.format_exc())
    finally:core.project_halfspaces=original;put(O/'admission.json',report)
    print(json.dumps(dict(passed=report['passed'],optimizer_attempts=report['optimizer_attempts'],output=str(O))),flush=True)

if __name__=='__main__':main()
