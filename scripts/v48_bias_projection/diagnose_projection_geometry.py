"""Capture two fixed offline proposals and audit their feasible intersections."""
from pathlib import Path
import argparse,copy,json,hashlib,sys
import torch,numpy as np
from bias_projection import LIMIT,MAX_CYCLES,SOLVER_TOL
from admit_bias_projection import same

def geometry(inputs):
    proposal,gradients,current,batch,anchor=inputs
    lower=torch.maximum(batch.double()-LIMIT,anchor.double()-LIMIT)
    upper=torch.minimum(batch.double()+LIMIT,anchor.double()+LIMIT)
    lo=lower.float();hi=upper.float()
    lo=torch.where(lo.double()<lower,torch.nextafter(lo,torch.full_like(lo,float('inf'))),lo)
    hi=torch.where(hi.double()>upper,torch.nextafter(hi,torch.full_like(hi,-float('inf'))),hi)
    lb=lo.double()-current.double();ub=hi.double()-current.double()
    G=torch.stack(gradients).double();norm=G.norm(dim=1);N=G/norm[:,None]
    eye=torch.eye(len(proposal),dtype=torch.float64)[-2:]
    A=torch.cat([N,eye,-eye]);b=torch.cat([torch.zeros(9,dtype=torch.float64),ub,-lb])
    assert (b>=0).all() # delta=0 explicitly witnesses all linear constraints
    singular=torch.linalg.svdvals(A)
    x=proposal.double().clone();cor=[torch.zeros_like(x) for _ in range(10)];history=[]
    for k in range(MAX_CYCLES):
        prev=x.clone()
        for i,g in enumerate(N):
            y=x+cor[i];x=y-g*torch.clamp(g.dot(y),min=0);cor[i]=y-x
        y=x+cor[-1];x=y.clone();x[-2:]=torch.maximum(lb,torch.minimum(ub,y[-2:]));cor[-1]=y-x
        residual=A@x-b;change=float((x-prev).abs().max())
        history.append(dict(cycle=k+1,signed_normalized_residual=residual.tolist(),raw_cost_dot=(G@x).tolist(),max_positive_residual=max(0.,float(residual.max())),max_change=change))
        if max(0.,float(residual.max()))<=SOLVER_TOL and change<=SOLVER_TOL:break
    return dict(current_bias=current.tolist(),absolute_lower=lower.tolist(),absolute_upper=upper.tolist(),increment_lower=lb.tolist(),increment_upper=ub.tolist(),
                constraint_labels=[f'cost_{i}' for i in range(9)]+['bias_roll_upper','bias_pitch_upper','bias_roll_lower','bias_pitch_lower'],
                proposal_norm=float(proposal.double().norm()),gradient_norms=norm.tolist(),cost_normal_cosines=(N@N.T).tolist(),
                all_normal_cosines=(A@A.T).tolist(),singular_values=singular.tolist(),condition_number=float(singular[0]/singular[-1]),
                zero_witness_signed_residual=(-b).tolist(),zero_witness_feasible=True,
                termination=dict(max_cycles=MAX_CYCLES,tolerance=SOLVER_TOL,requires_both='max positive normalized residual and last-cycle change',converged=history[-1]['max_positive_residual']<=SOLVER_TOL and history[-1]['max_change']<=SOLVER_TOL),
                history=history,final_delta=x.tolist())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--v48',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    V=a.v48;T=V/'backtracking_revision3/training';O=a.out;O.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(T/'runtime'))
    import v46_offline_scheduler as core
    from v46_core import RPActor,read_capture
    from adaptive_smooth import AdaptiveScheduler
    from v48_configuration import configure_optimizer
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    # The exact already-reviewed reconstruction block, used without its test loop.
    source=Path(__file__).with_name('admit_bias_projection.py').read_text()
    setup=source[source.index("          p=j(T/'protocol.json')"):source.index('          v=core.flat(actor).clone()')]
    import textwrap
    report=dict(native_calls=0,formal_updates=0,disposable_optimizer_attempts=0,points={})
    protected=[V/'training/ledger.json',V/'development/ledger.json']+list((T/'runtime').glob('*.py'))+list((T/'training').glob('batch_*/*.pt'))
    hashes={str(p.relative_to(V)):core.digest(p) for p in protected}
    points=[('pre18',1,V/'memory_gate_revision2/training/training/batch_1/pre_update_recovery.pt',V/'memory_gate_revision2/training/protocol.json',17),('pre40',3,T/'training/batch_3/pre_update_recovery.pt',T/'protocol.json',39)]
    for name,b,cpp,pr,global_step in points:
        item={};ctx=dict(globals(),**locals());ctx['j']=lambda p:json.loads(p.read_text())
        exec(textwrap.dedent(setup),ctx);s=ctx['s'];Aactor=ctx['A'];start=ctx['start'];q=ctx['q']
        captured=[]
        def capture(proposal,gradients,passes):
            captured.extend([proposal.clone(),[g.detach().clone() for g in gradients],s.actor.rp_b.detach().clone(),s.initial['rp_b'].clone(),Aactor.rp_b.detach().clone()])
            raise RuntimeError('Diagnostic capture only: no accepted update')
        project_original=core.project_halfspaces;core.project_halfspaces=capture
        try:s.step()
        except RuntimeError as e:assert str(e)=='Diagnostic capture only: no accepted update'
        finally:core.project_halfspaces=project_original
        report['disposable_optimizer_attempts']+=1
        assert same(s.actor.state_dict(),s.initial) and same(s.opt.state_dict(),s.initial_opt) and same(core.rng_state(),s.initial_rng)
        torch.save(captured,O/f'{name}_projection_inputs.pt')
        detail=geometry(captured);detail['checkpoint_sha256']=core.digest(cpp)
        detail['current_nonlinear_guards']=ctx['s'].diagnostics() # replaced below by exact pre-step restored state
        s.restore(copy.deepcopy(start));detail['current_nonlinear_guards']=s.diagnostics();assert detail['current_nonlinear_guards']['accepted']
        detail['exact_restore_and_capture_rollback']=True
        (O/f'{name}_geometry.json').write_text(json.dumps(detail,indent=2))
        report['points'][name]={k:v for k,v in detail.items() if k not in ['history','final_delta','all_normal_cosines','cost_normal_cosines']}
        report['points'][name]['last_iteration']=detail['history'][-1]
    report['protected_unchanged']=all(core.digest(V/p)==h for p,h in hashes.items());assert report['protected_unchanged']
    report['protected_hashes']=hashes
    (O/'diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
