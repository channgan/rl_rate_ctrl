from pathlib import Path
import json,struct
V=Path('@DRL_ROOT_WINDOWS@/experiments/robustness_v1_20261002/v48_fixed_lr3e5')
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return None
active=read(V/'active_revision.json')
active_root=Path(active['root']) if active else V
out={'active_root':str(active_root)}
for phase in ['training','development']:
 r=active_root/phase;s=read(r/'status.json');l=read(V/phase/'ledger.json');item={'status':s,'charged':l['charged_native'],'formal_updates':l.get('formal_optimizer_updates',0),'reserved':l['reservations'],'attempts':len(l['attempts']),'stopped':l['stopped']}
 if l['attempts']:item['last_attempt']={k:l['attempts'][-1].get(k) for k in ['id','status','charge','actual_native_NN','actual_native_PID']}
 if s and s.get('job'):
  jr=r/'jobs'/s['job'];item['progress']=read(jr/'progress.json');item['live_reference']=[]
  for f in jr.glob('*/*/seed_*/rate_reference.bin'):
   n=max(0,(f.stat().st_size-8)//128)
   if n:
    with f.open('rb') as stream:stream.seek(8+(n-1)*128);raw=stream.read(8)
    if len(raw)==8:item['live_reference'].append(dict(records=n,last_sim_us=struct.unpack('<Q',raw)[0]))
  if item['progress']:item['progress']={k:item['progress'].get(k) for k in ['stage','pid']}
 out[phase]=item
out['holder_completed']=read(active_root/'execution/holder_completed.json');print(json.dumps(out,indent=2))
