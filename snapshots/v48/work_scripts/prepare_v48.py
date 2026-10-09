from pathlib import Path
import json,shutil,hashlib,ast
N=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/native_fresh_batch_rp_v47');M=N/'memory_sampling_revision1';O=N.parent/'v48_lr_offline_admission1';T=O/'prepared_training'
j=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert j(O/'admission.json')['passed'];T.mkdir(exist_ok=False);shutil.copytree(M/'runtime',T/'runtime');shutil.copytree(M/'templates',T/'templates')
def replace(text,a,b):assert text.count(a)==1,(a,text.count(a));return text.replace(a,b)
p=j(M/'protocol.json');p.update(version='v48-fixed-lr-3e-5-prepared-not-authorized',authorized=False,authority='Preparation only; no native launch before separate execution decision',canonical_ledger=str(T/'ledger.json'),learning_rate=3e-5,maximum_new_captures=28,additional_native_cap=1316000,historical_charge=0,selection='final update40 only; no evaluation-picked checkpoints',sync_runtime_id='v48-qualified-v47-sync2',validation_jobs=[])
p['development']['seeds']=[480301,480302];p['training_objective_revision']['first_ten_old_objective']=True
p['preparation_notes']=['Restart actor A and fresh Adam; no V47 optimizer continuation.','First10 original objective alpha1 and original balance guard; adaptation from11; memory diagnostic from21, matching V47 schedule.','All28 captures must originate under this root and from the current batch actor.','Training/native cap2m; 28x47000 reservations=1316000, unused capacity does not authorize extra captures.','54 frozen development cases; cap3m; holdout12701/12702 sealed.']
for key in ['old_ledger']:p.pop(key,None)
p['jobs']=[dict(id=f'train_b{b}_p{i}',phase='training',batch=b,profile_index=i,case=profile['case'],seed=480101+b*1000+i,cap=47000,label='A' if b==0 else 'candidate') for b in range(4) for i,profile in enumerate(p['entry_profiles'])]
# Avoid train/dev seed overlap by using 480101..480107,481101.. etc.
assert not set(x['seed'] for x in p['jobs'])&set(p['development']['seeds'])
rt=T/'runtime';f=rt/'batch_driver.py';x=f.read_text()
a="resume=read(ROOT/'runtime_replacement.json')";b="for p,h in protocol['runtime_sha256'].items():assert sha(p)==h,p"
start=x.index(a);end=x.index(b);x=x[:start]+"assert ledger['charged_native']==0 and ledger['formal_optimizer_updates']==0 and not ledger['attempts']\n"+x[end:]
x=replace(x," actor=read(ROOT/'training/batch_1/export.json')\n for path,h in read(ROOT/'carried_batch2.json')['sha256'].items():assert sha(path)==h,path\n for batch in range(2,4):"," actor=protocol['initial_actor']\n for batch in range(4):")
x=replace(x,"paths=read(ROOT/'batch_2_captures.json')['results'] if batch==2 else [capture(job,actor) for job in protocol['jobs'] if job['batch']==batch]","paths=[capture(job,actor) for job in protocol['jobs'] if job['batch']==batch]")
f.write_text(x)
f=rt/'update_batch.py';x=f.read_text();x=replace(x,"actor=RPActor(m['actor']['npz']);anchor=RPActor(p['initial_actor']['npz']);captures=[];entries=[]","assert p['authorized'] and p['learning_rate']==3e-5 and batch in range(4)\nassert len(m['results'])==7 and all(Path(v).resolve().is_relative_to((ROOT/'jobs').resolve()) for v in m['results']), 'Historical/external captures rejected'\nactor=RPActor(m['actor']['npz']);anchor=RPActor(p['initial_actor']['npz']);captures=[];entries=[]")
x=replace(x,"s=FreshScheduler(actor,d,digest(ROOT/'protocol.json'),gh,gm)","s=FreshScheduler(actor,d,digest(ROOT/'protocol.json'),gh,gm)\nold=None")
x=replace(x,"s.initial_opt=copy.deepcopy(s.opt.state_dict())","s.initial_opt=copy.deepcopy(s.opt.state_dict())")
x=replace(x,"out=ROOT/'training'/('batch_'+str(batch));out.mkdir(parents=True,exist_ok=False)","for group in s.opt.param_groups:group['lr']=p['learning_rate']\ns.initial_opt=copy.deepcopy(s.opt.state_dict())\nout=ROOT/'training'/('batch_'+str(batch));out.mkdir(parents=True,exist_ok=False)")
x=replace(x,"old['state'].get('objective_state',{}).get('alpha',1.),out)","old['state'].get('objective_state',{}).get('alpha',1.) if old else 1.,out)")
x=replace(x,"rng_restore(old['state']['rng']);s.gen.set_state(old['state']['generator'])\nassert all(torch.equal(v,old['state']['actor'][k]) for k,v in actor.state_dict().items())","if old:\n rng_restore(old['state']['rng']);s.gen.set_state(old['state']['generator'])\n assert all(torch.equal(v,old['state']['actor'][k]) for k,v in actor.state_dict().items())\nelse:\n assert all(torch.equal(v,anchor.state_dict()[k]) for k,v in actor.state_dict().items())")
x=replace(x,"balance=s.adapt();s.save(out/'pre_update_recovery.pt')","if batch==0:\n   records=s.balance();balance=dict(coefficients=dict(legacy_torque=2/70,extra_RP_torque=.05,ESC=.05),zero_tracking_strata_axes=[],records=records)\n  else:balance=s.adapt()\n  s.save(out/'pre_update_recovery.pt')")
f.write_text(x)
# Keep previously tested crash/export diagnostics, changing only training ledger cap.
x=(N/'development_recovery1/runtime/runtime_support.py').read_text().replace('==3000000','==2000000');(rt/'runtime_support.py').write_text(x)
p['prepared_python_sha256']={str(f):sha(f) for f in rt.glob('*.py')}
(T/'protocol.json').write_text(json.dumps(p,indent=2));(T/'ledger.json').write_text(json.dumps(dict(native_limit=2000000,charged_native=0,formal_optimizer_updates=0,attempts=[],reservations={},stopped=False),indent=2))
entry='''from pathlib import Path
import argparse,json,hashlib,subprocess,sys
R=Path(__file__).resolve().parent
def check():
 p=json.loads((R/'protocol.json').read_text());a=json.loads((R.parent/'admission.json').read_text())
 assert a['passed'] and p['learning_rate']==3e-5
 assert len(p['jobs'])==28 and len({j['seed'] for j in p['jobs']})==28
 assert sum(j['cap'] for j in p['jobs'])==1316000
 assert p['development']['episodes']==54 and p['development']['seeds']==[480301,480302]
 for path,h in {**p['runtime_sha256'],**p['prepared_python_sha256']}.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 return p
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');args=ap.parse_args();p=check()
 if args.run:
  if not p['authorized']:raise SystemExit('Prepared only: native execution has not been authorized.')
  raise SystemExit(subprocess.call([sys.executable,'-B',str(R/'runtime/batch_driver.py'),str(R)]))
 print(json.dumps(dict(preflight_passed=True,authorized=p['authorized'],native_calls=0,training_captures=28,updates=40)))
'''
(T/'entry.py').write_text(entry)
for f in [T/'entry.py',*rt.glob('*.py')]:ast.parse(f.read_text())
(O/'preparation.json').write_text(json.dumps(dict(root=str(T),authorized=False,native_calls=0,syntax_files=1+len(list(rt.glob('*.py'))),protocol_sha256=sha(T/'protocol.json')),indent=2))
print('Prepared',T,flush=True)
