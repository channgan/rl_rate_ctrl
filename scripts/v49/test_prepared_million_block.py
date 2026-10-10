"""Zero-native tests of the actual generated runtime, historical replay only."""
from pathlib import Path
import argparse,copy,hashlib,json,sys
import numpy as np
import torch

def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--v49',type=Path,required=True);a=ap.parse_args()
    out=a.prepared;r=out/'runtime';t=a.v49/'training';j=lambda p:json.loads(p.read_text())
    p=j(out/'protocol.json');l=j(out/'ledger.json')
    assert not p['authorized'] and not l['authorized'] and l['formal_optimizer_updates']==40
    assert len(p['jobs'])==490 and len({x['seed'] for x in p['jobs']})==490
    assert sum(x['cap'] for x in p['jobs'])==23030000 and max(x['batch'] for x in p['jobs'])==69
    for f in r.glob('*.py'):compile(f.read_text(),str(f),'exec')
    sys.path.insert(0,str(r))
    from block_safety import require_grant,required_disk
    try:require_grant(out)
    except FileNotFoundError:pass
    else:raise AssertionError('unauthorized prepared runtime admitted')
    assert required_disk(490)==163745628160 and required_disk(0)==32212254720
    source=(r/'batch_driver.py').read_text()
    assert 'development_running' not in source and 'finalize_and_bind.py' not in source
    assert "for batch in range(70)" in source and "block_complete_await_decision" in source
    update=(r/'update_batch.py').read_text()
    assert '40+batch*10+s.updates' in update and 'resume_checkpoint' in update
    import v46_offline_scheduler as core
    from v46_core import RPActor,read_capture
    from v49_scheduler import V49Scheduler
    from v48_configuration import configure_optimizer
    from block_telemetry import full
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    original=j(t/'protocol.json');m=j(t/'batch_3_captures.json')
    qs=[read_capture(Path(x)) for x in m['results']];behavior=RPActor(m['actor']['npz']);anchor=RPActor(original['initial_actor']['npz'])
    data=core.prepare(behavior,qs,original['scenario_groups']);g=np.load(original['guard_corpus']);gx=torch.tensor(g['obs'],dtype=torch.float32);dt=torch.tensor(g['dt'],dtype=torch.float32)
    with torch.no_grad():gh=behavior.features(gx);gm=behavior(gx,dt);am=anchor(gx,dt)
    pre=torch.load(t/'training/batch_3/pre_update_recovery.pt',weights_only=False,map_location='cpu');final=torch.load(p['resume_checkpoint'],weights_only=False,map_location='cpu')
    # Verify exact final40 state transfer with fresh batch counters and explicit new stream.
    actor=RPActor(p['starting_actor']['npz']);opt=torch.optim.Adam(actor.parameters(),lr=3e-5)
    configure_optimizer(opt,final['state']['optimizer'],3e-5)
    assert same(actor.state_dict(),final['state']['actor']) and same(opt.state_dict(),final['state']['optimizer'])
    assert {int(x['step']) for x in opt.state_dict()['state'].values()}=={40}
    alpha=final['state']['objective_state']['alpha'];assert alpha==.3767965169092372
    streams=[torch.Generator().manual_seed(510901).get_state() for _ in range(2)];assert torch.equal(*streams)
    # Exact pre40 replay with the generated runtime logging hooks, on/off.
    results=[]
    for enabled in (False,True):
        s=V49Scheduler(RPActor(m['actor']['npz']),copy.deepcopy(data),core.digest(t/'protocol.json'),gh,gm,anchor,am)
        s.attach(RPActor(m['actor']['npz']),qs,original['scenario_groups']);s.restore(copy.deepcopy(pre['state']));s.telemetry_enabled=enabled
        if enabled:before=full(s)
        row=s.step()
        if enabled:telemetry=dict(before=before,after=full(s),**s.step_telemetry)
        results.append((s.snapshot(),row,core.binary_bytes(s.actor)[0]))
    assert same(results[0],results[1]);assert results[0][2]==Path(p['starting_actor']['binary']).read_bytes()
    assert same(results[0][0]['optimizer'],final['state']['optimizer'])
    assert all(np.isfinite(telemetry[k]) for k in ('raw_gradient_norm','adam_proposal_norm','projected_norm'))
    report=dict(passed=True,generated_source_syntax=True,unauthorized_start_blocked=True,exact_final40_actor_adam_alpha=True,global_update_range=[41,740],jobs=490,outer_steps=1003520,native_cap=23030000,automatic_evaluation_jobs=0,disk_required_bytes=required_disk(490),generated_logging_numerically_identical=True,disposable_optimizer_attempts=2,native_calls=0,formal_updates=0,limitations=['No new captures executed; fresh runtime behavior requires first planned capture gate at authorized launch.','No launch grant exists; final54 review and explicit launch decision still required.'])
    (out/'offline_admission.json').write_text(json.dumps(report,indent=2));(out/'replay_telemetry.json').write_text(json.dumps(telemetry,indent=2,allow_nan=False));print(json.dumps(report))
if __name__=='__main__':main()
