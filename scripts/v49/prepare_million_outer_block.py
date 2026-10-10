"""Prepare/start ONLY an explicitly granted 70-batch block. Prepare never launches."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys

CAP=23030000
MODEL='c0183c13bcf588c940a88f2a10aff59bc9256e8eb4b91066331ac9e4efe01855'
CHECKPOINT='492dea55be406cbb9aa36c1ae7dde7878854936c0d3cb9a95832cdcc1e37e4c8'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,v):Path(p).write_text(json.dumps(v,indent=2))
def replace(s,old,new):
    assert old in s,old
    return s.replace(old,new)

def prepare(v49,out):
    v49=Path(v49).resolve();out=Path(out).resolve();t=v49/'training'
    assert out!=v49 and not out.is_relative_to(v49) and not v49.is_relative_to(out)
    done=read(t/'training_complete.json');assert done['updates']==40 and done['completed']
    assert sha(done['actor']['binary'])==MODEL and sha(t/'training/batch_3/recovery.pt')==CHECKPOINT
    out.mkdir(parents=True,exist_ok=False);r=out/'runtime';shutil.copytree(t/'runtime',r)
    shutil.copytree(t/'templates',out/'templates')
    p=read(t/'protocol.json');p.update(authorized=False,authority='PREPARED ONLY; separate explicit launch grant required',version='million-outer-block-draft',native_limit=CAP,canonical_ledger=str(out/'ledger.json'),starting_actor=done['actor'],resume_checkpoint=str(t/'training/batch_3/recovery.pt'),resume_checkpoint_sha256=CHECKPOINT,optimizer_seed_base=510902,source_v49=str(v49),start_global_update=40,new_updates=700,block_count=1,effective_step_unit='10ms_env_step',automatic_development=False)
    p['jobs']=[dict(id=f'cont_b{b}_p{i}',phase='training',batch=b,profile_index=i,case=profile['case'],seed=1000000+10*b+i,cap=47000,label='candidate') for b in range(70) for i,profile in enumerate(p['entry_profiles'])]
    assert len(p['jobs'])==490 and sum(j['cap'] for j in p['jobs'])==CAP
    oldseeds={a['seed'] for a in read(t/'ledger.json')['attempts']}|{a['seed'] for a in read(v49/'development/ledger.json')['attempts']}|{12701,12702,520101,520102,520103,530101,530102}
    assert not oldseeds.intersection(j['seed'] for j in p['jobs'])
    for name in ('runtime_support.py','batch_driver.py'):
        s=(r/name).read_text();s=replace(s,'1316000','23030000');(r/name).write_text(s)
    s=(r/'v48_configuration.py').read_text();s=replace(s,'range(4)','range(70)');(r/'v48_configuration.py').write_text(s)
    s=(r/'update_batch.py').read_text();s=replace(s,'batch in range(4)','batch in range(70)')
    s=replace(s,'old=None','old=torch.load(p["resume_checkpoint"],map_location="cpu",weights_only=False)')
    s=replace(s,'if old:\n rng_restore','if batch:\n rng_restore')
    s=replace(s,"assert all(torch.equal(v,anchor.state_dict()[k]) for k,v in actor.state_dict().items())","assert all(torch.equal(v,old['state']['actor'][k]) for k,v in actor.state_dict().items())\n s.gen.manual_seed(510901)\n s.initial_gen=s.gen.get_state().clone()")
    s=replace(s,'global_update=batch*10+s.updates','global_update=40+batch*10+s.updates')
    s=replace(s,"==batch*10+s.updates-1","==40+batch*10+s.updates-1")
    s=replace(s,"']=batch*10+s.updates","']=40+batch*10+s.updates")
    s=replace(s,'global_updates=(batch+1)*10','global_updates=40+(batch+1)*10')
    # Preserve original global A anchor; all Adam moments and objective alpha carry forward.
    s=replace(s,'  result=s.step();','  from block_telemetry import full\n  objective_before=full(s)\n  result=s.step();')
    s=replace(s,"  with (out/'updates.jsonl').open('a') as f:","  with (out/'objective_telemetry.jsonl').open('a') as telemetry_file:telemetry_file.write(json.dumps(dict(global_update=40+batch*10+s.updates,full_before=objective_before,full_after=full(s),**s.step_telemetry),allow_nan=False)+'\\n')\n  with (out/'updates.jsonl').open('a') as f:")
    (r/'update_batch.py').write_text(s)
    s=(r/'v46_offline_scheduler.py').read_text()
    s=replace(s,'self.opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_', 'self.opt.zero_grad();loss.backward();__import__("block_telemetry").raw(self,loss,ix,r,adv);torch.nn.utils.clip_grad_norm_')
    s=replace(s,'delta=self.project_proposal(proposed,gradients)','delta=self.project_proposal(proposed,gradients)\n  __import__("block_telemetry").projected(self,proposed,delta)')
    (r/'v46_offline_scheduler.py').write_text(s)
    s=(r/'batch_driver.py').read_text()
    s=replace(s,"assert ledger['charged_native']==0 and ledger['formal_optimizer_updates']==0", "assert ledger['charged_native']==0 and ledger['formal_optimizer_updates']==40")
    s=replace(s,"actor=protocol['initial_actor']","actor=protocol['starting_actor']")
    s=replace(s,'for batch in range(4):','for batch in range(70):')
    s=replace(s,"ledger['formal_optimizer_updates']=(batch+1)*10","assert ledger['formal_optimizer_updates']==40+(batch+1)*10")
    s=replace(s,"if job['id']=='train_b0_p0'","if job['id']=='cont_b0_p0'")
    s=replace(s,"global ledger\n from sync_trace_audit", "global ledger\n from block_safety import require_disk\n require_disk(ROOT,490-len(ledger['attempts']))\n from sync_trace_audit")
    s=replace(s,"ledger['attempts'].append(entry);ledger['reservations'].pop(job['id']);", "ledger['attempts'].append(entry);ledger['audited_effective_outer_steps']=sum(2048 for a in ledger['attempts'] if a['status']=='audited');ledger['audited_scoring_native_transitions']=ledger['audited_effective_outer_steps']*10;ledger['reservations'].pop(job['id']);")
    start=s.index(" atomic(ROOT/'training_complete.json'")
    end=s.index('\nexcept BaseException as e:',start)
    s=s[:start]+"\n assert ledger['formal_optimizer_updates']==740 and ledger['audited_effective_outer_steps']==1003520 and not ledger['reservations']\n assert subprocess.run([sys.executable,'-B',str(R/'verify_block.py'),str(ROOT)]).returncode==0, 'block verification failed'\n atomic(ROOT/'training_complete.json',dict(completed=True,actor=actor,updates=740,new_updates=700,effective_outer_steps=1003520,scoring_native_transitions=10035200,promotion_allowed=False,automatic_development=False))\n status(state='block_complete_await_decision',updates=740,charged=ledger['charged_native'])"+s[end:]
    # Require a cryptographically bound grant at driver admission, not only CLI.
    s=replace(s,"protocol=read(ROOT/'protocol.json');", "from block_safety import require_grant\nrequire_grant(ROOT)\nprotocol=read(ROOT/'protocol.json');")
    (r/'batch_driver.py').write_text(s)
    shutil.copyfile(Path(__file__).with_name('million_block_safety.py'),r/'block_safety.py')
    shutil.copyfile(Path(__file__).with_name('verify_million_block.py'),r/'verify_block.py')
    shutil.copyfile(Path(__file__).with_name('million_block_telemetry.py'),r/'block_telemetry.py')
    # Immutable source hashes are re-checked before launch. No existing runtime is edited.
    p['prepared_runtime_sha256']={str(f):sha(f) for f in r.glob('*.py')}
    put(out/'protocol.json',p)
    put(out/'ledger.json',dict(authorized=False,native_limit=CAP,charged_native=0,formal_optimizer_updates=40,audited_effective_outer_steps=0,audited_scoring_native_transitions=0,attempts=[],reservations={},stopped=False))
    put(out/'preparation.json',dict(status='not_admitted',launch_enabled=False,protocol_sha256=sha(out/'protocol.json'),source_checkpoint_sha256=CHECKPOINT,batches=70,trajectories=490,new_updates=700,effective_outer_steps=1003520,native_limit=CAP,automatic_monitor_jobs=0))
    return out

def start(out):
    out=Path(out).resolve();sys.path.insert(0,str(out/'runtime'))
    from block_safety import require_grant,require_disk
    require_grant(out);require_disk(out,490)
    # Independent process; no restart/resume inference, no repeat start.
    with (out/'driver.log').open('x') as stream:
        child=subprocess.Popen([sys.executable,'-B',str(out/'runtime/batch_driver.py'),str(out)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    put(out/'launch_receipt.json',dict(pid=child.pid,started=True));return child.pid

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','start']);ap.add_argument('--v49');ap.add_argument('--out',required=True);a=ap.parse_args()
    print(prepare(a.v49,a.out) if a.action=='prepare' else start(a.out))
