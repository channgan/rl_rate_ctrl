"""Disposable final40 restore and exact pre40 logging replay; NEVER launches native."""
from pathlib import Path
import argparse, copy, hashlib, json, random, shutil, sys
import numpy as np
import torch

def same(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--v49',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    t=a.v49/'training';a.out.mkdir(parents=True,exist_ok=False)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    j=lambda p:json.loads(Path(p).read_text())
    protected=[t/'ledger.json',t/'training_complete.json',*t.glob('runtime/*.py'),*t.glob('training/batch_*/*.pt'),*t.glob('training/batch_*/actor.*')]
    before={str(p):sha(p) for p in protected}
    # Copy only Python source; instrument the disposable copy, never the live runtime.
    r=a.out/'runtime';r.mkdir()
    for p in (t/'runtime').glob('*.py'):shutil.copyfile(p,r/p.name)
    source=(r/'v46_offline_scheduler.py').read_text()
    old='self.opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.actor.parameters(),.5);self.opt.step();self.attempts+=1'
    new='self.opt.zero_grad();loss.backward();self.observe_raw(loss,ix,r,adv) if getattr(self,"telemetry_enabled",False) else None;torch.nn.utils.clip_grad_norm_(self.actor.parameters(),.5);self.opt.step();self.attempts+=1'
    assert source.count(old)==1;source=source.replace(old,new)
    old='delta=self.project_proposal(proposed,gradients)'
    new='delta=self.project_proposal(proposed,gradients)\n  if getattr(self,"telemetry_enabled",False):self.observe_projection(proposed,delta)'
    assert source.count(old)==1;source=source.replace(old,new)
    (r/'v46_offline_scheduler.py').write_text(source)
    sys.path.insert(0,str(r))
    import v46_offline_scheduler as core
    from v46_core import RPActor,read_capture
    from v49_scheduler import V49Scheduler
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    p=j(t/'protocol.json');m=j(t/'batch_3_captures.json')
    final=torch.load(t/'training/batch_3/recovery.pt',map_location='cpu',weights_only=False)
    pre=torch.load(t/'training/batch_3/pre_update_recovery.pt',map_location='cpu',weights_only=False)
    model=RPActor(t/'training/batch_3/actor.npz');anchor=RPActor(p['initial_actor']['npz'])
    assert same(model.state_dict(),final['state']['actor'])
    raw,mid=core.binary_bytes(model);assert raw==(t/'training/batch_3/actor.bin').read_bytes() and mid==4142227144
    assert all(torch.equal(v,anchor.state_dict()[k]) for k,v in model.named_buffers())
    assert {int(s['step']) for s in final['state']['optimizer']['state'].values()}=={40}
    assert final['state']['objective_state']['alpha']==0.3767965169092372
    torch.save(final,a.out/'roundtrip.pt');assert same(final,torch.load(a.out/'roundtrip.pt',weights_only=False))
    opt=torch.optim.Adam(model.parameters(),lr=3e-5);opt.load_state_dict(copy.deepcopy(final['state']['optimizer']))
    assert same(opt.state_dict(),final['state']['optimizer'])
    # Explicit new-stream admission fixture. Old RNG is retained in original checkpoint.
    fresh=copy.deepcopy(final['state']);fresh['perms']={};fresh['positions']={};fresh['updates']=0;fresh['reports']=[];fresh['stopped']=False
    g=torch.Generator().manual_seed(510901);fresh['generator']=g.get_state()
    torch.manual_seed(510902);np.random.seed(510902);random.seed(510902);fresh['rng']=core.rng_state()
    assert same(fresh['actor'],final['state']['actor']) and same(fresh['optimizer'],final['state']['optimizer'])
    assert same(fresh['objective_state'],final['state']['objective_state'])
    torch.save(fresh,a.out/'new_stream_state.pt');assert same(fresh,torch.load(a.out/'new_stream_state.pt',weights_only=False))
    g2=torch.Generator().manual_seed(510901);assert torch.equal(g.get_state(),g2.get_state())
    # Historical batch3 is ONLY a replay fixture, never continuation training data.
    captures=[read_capture(Path(x)) for x in m['results']]
    behavior=RPActor(m['actor']['npz']);data=core.prepare(behavior,captures,p['scenario_groups'])
    guard=np.load(p['guard_corpus']);gx=torch.tensor(guard['obs'],dtype=torch.float32);dt=torch.tensor(guard['dt'],dtype=torch.float32)
    with torch.no_grad():gh=behavior.features(gx);gm=behavior(gx,dt);am=anchor(gx,dt)
    class Observed(V49Scheduler):
        def observe_raw(self,loss,ix,ratio,adv):
            grad=torch.cat([p.grad.detach().flatten() for p in self.actor.parameters()]).clone()
            # Both norms are read-only; the original clip_grad_norm_ call remains unchanged.
            norms=torch.stack([p.grad.detach().norm(2) for p in self.actor.parameters()]);n=float(norms.norm(2))
            self.telemetry=dict(clipped_surrogate_loss=float(loss.detach()),unclipped_surrogate_loss=float(-(self.minibatch_weights*ratio.detach()*adv).sum()),raw_gradient=grad.tolist(),raw_gradient_norm=n,flat_gradient_norm=float(grad.norm()),clip_factor=min(1.,.5/(n+1e-6)),sample_index_sha256=hashlib.sha256(ix.cpu().numpy().tobytes()).hexdigest())
        def observe_projection(self,proposal,delta):
            self.telemetry.update(adam_proposal=proposal.detach().tolist(),projected_delta=delta.detach().tolist(),proposal_norm=float(proposal.norm()),projected_norm=float(delta.norm()),projection_correction_norm=float((proposal-delta).norm()))
        @torch.no_grad()
        def full_objective(self):
            lp=self.likelihood()[0];ratio=torch.exp(lp-self.data['behavior']);adv=self.data['adv'];w=self.data['weights']
            return dict(clipped_loss=float(-(w*torch.minimum(ratio*adv,ratio.clamp(.95,1.05)*adv)).sum()),unclipped_loss=float(-(w*ratio*adv).sum()))
    states=[];reports=[];exports=[];telemetry=None
    for enabled in [False,True]:
        s=Observed(RPActor(m['actor']['npz']),copy.deepcopy(data),core.digest(t/'protocol.json'),gh,gm,anchor,am)
        s.attach(RPActor(m['actor']['npz']),captures,p['scenario_groups']);s.restore(copy.deepcopy(pre['state']))
        s.telemetry_enabled=enabled
        before_obj=s.full_objective() if enabled else None
        report=s.step()
        if enabled:
            telemetry=s.telemetry;telemetry.update(full_before=before_obj,full_after=s.full_objective(),accepted=report,physical_tracking_sums=[float(q['tracking'].sum()) for q in captures],physical_torque_smooth_sums=[float(q['torque_smooth'].sum()) for q in captures],physical_motor_smooth_sums=[float(q['motor_smooth'].sum()) for q in captures])
        states.append(s.snapshot());reports.append(report);exports.append(core.binary_bytes(s.actor)[0])
    assert same(states[0],states[1]) and same(reports[0],reports[1]) and exports[0]==exports[1]==raw
    assert same(states[0]['actor'],final['state']['actor']) and same(states[0]['optimizer'],final['state']['optimizer'])
    assert abs(telemetry['raw_gradient_norm']-telemetry['flat_gradient_norm'])<1e-6
    # Independent autograd reconstruction in a separate disposable actor.
    s=Observed(RPActor(m['actor']['npz']),copy.deepcopy(data),core.digest(t/'protocol.json'),gh,gm,anchor,am)
    s.attach(RPActor(m['actor']['npz']),captures,p['scenario_groups']);s.restore(copy.deepcopy(pre['state']))
    ix=s.indices();lp=s.likelihood(ix)[0];ratio=torch.exp(lp-s.data['behavior'][ix]);adv=s.data['adv'][ix]
    loss=-(s.minibatch_weights*torch.minimum(ratio*adv,ratio.clamp(.95,1.05)*adv)).sum()
    grads=torch.autograd.grad(loss,tuple(s.actor.parameters()));independent=torch.cat([x.flatten() for x in grads])
    assert torch.equal(independent,torch.tensor(telemetry['raw_gradient'],dtype=independent.dtype))
    assert np.isfinite(np.asarray(telemetry['raw_gradient'])).all()
    assert before=={str(p):sha(p) for p in protected}
    (a.out/'telemetry.json').write_text(json.dumps(telemetry,indent=2,allow_nan=False))
    report=dict(passed=True,native_calls=0,formal_updates=0,disposable_optimizer_attempts=2,independent_gradient_checks=1,final40_restore_exact=True,adam_steps=40,alpha=final['state']['objective_state']['alpha'],new_stream_roundtrip=True,logging_on_off_bitwise_equal=True,pre40_replay_matches_final40=True,protected_files=len(protected),model_id=mid,model_sha256=sha(t/'training/batch_3/actor.bin'),limitations=['Historical pre40 fixture only; new-data continuation launcher not implemented/admitted.','Proposal contract/budget/stop-policy tests still required before launch.'])
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
