"""Execution-side guards only. No observer heartbeat or host-clock dependency."""
from pathlib import Path
import os,json,subprocess,sys,signal,struct,threading,hashlib
ROOT=Path(os.environ['BATCH_TRAINING_ROOT']);JOB=Path(os.environ['SUPERVISED_JOB_ROOT'])
def read(p):return json.loads(Path(p).read_text())
def atomic(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require_authorized(protocol):
 grant=read(ROOT/'protocol.json');ledger=read(grant.get('canonical_ledger',ROOT/'ledger.json'));job=protocol['v46_job_id']
 assert grant['authorized'] and grant['native_limit']==1316000 and ledger['native_limit']==1316000
 assert not ledger['stopped'] and ledger['reservations'][job]['cap']==protocol['v46_native_cap']
 assert sha(JOB/'protocol.json')==ledger['reservations'][job]['protocol_sha256']
 assert protocol['native_episode_tick_cap']==protocol['v46_native_cap']
 assert sha(protocol['full_path'])==protocol['actor_sha256']
def owned_spawn(b,args,env,name):
 log=(b.episode_dir/(name+'.log')).open('w');b.files.append(log)
 child=subprocess.Popen(args,cwd=b.episode_dir,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 b.processes.append(child);(JOB/'owned').mkdir(exist_ok=True)
 atomic(JOB/'owned'/(name+'.json'),dict(pid=child.pid,kind=name,argv=args,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]))
 child.diag_kind=name;child.diag_argv=args
def register_case_watchdog():
 import resource
 soft,hard=resource.getrlimit(resource.RLIMIT_CORE);cap=268435456 if hard==resource.RLIM_INFINITY else min(268435456,hard);resource.setrlimit(resource.RLIMIT_CORE,(cap,hard))
 atomic(JOB/'crash_capture_settings.json',dict(core_soft_limit=cap,core_hard_limit=hard,core_pattern=Path('/proc/sys/kernel/core_pattern').read_text().strip(),global_settings_changed=False))
 # Explicit stop signals only; assistant/reader lifetime is not a lease.
 def stop(sig,frame):
  try:(JOB/'stop').write_text('explicit signal '+str(sig))
  except OSError:raise RuntimeError('explicit stop could not be recorded')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def observer(b,out,line,data):
 if not line.startswith('STEP '):return
 scans=getattr(b,'diag_scans',{})
 for name,stride in [('rate_reference.bin',128),('nn_capture.bin',224)]:
  p=Path(out)/name
  if not p.exists():continue
  n=max(0,(p.stat().st_size-8)//stride);old,last=scans.get(name,(0,None))
  with p.open('rb') as f:f.seek(8+old*stride);raw=f.read((n-old)*stride)
  for off in range(0,len(raw)-stride+1,stride):
   stamp=struct.unpack_from('<Q',raw,off)[0]
   if stamp>35000000:
    if last is not None and stamp-last>4000:
     atomic(Path(out)/'first_failure.json',dict(from_us=last,to_us=stamp,gap_us=stamp-last));raise RuntimeError('actual control cadence >4ms')
    last=stamp
  scans[name]=(n,last)
 b.diag_scans=scans
def export_before_close(b,out):
 if getattr(b,'diag_export_attempted',False):return
 if not b.processes and b.episode_dir is None:return # pristine reset has no export to attempt
 if b.stream is None or b.sock is None:
  b.diag_export_attempted=True
  atomic(Path(out)/'export_status.json',dict(attempted=False,success=False,reason='bridge_not_connected',authoritative_counts_available=False))
  return # permit base cleanup; never claim successful capture or zero usage
 b.diag_export_attempted=True;report={'attempted':True}
 try:
  b.sock.settimeout(180);b.stream.write(b'CADENCE_EXPORT\n');reply=json.loads(b.stream.readline(8192));report['reply']=reply;assert reply=={'cadence_exported':True};report['success']=True
 except BaseException as e:report.update(success=False,error=repr(e))
 atomic(Path(out)/'export_status.json',report)
 # Never silently treat failed export as a completed capture.
 if not report.get('success'):raise RuntimeError('authoritative export failed; retain ownership for reconciliation')
def stage_emit(root,stage,**facts):
 # Advisory status may fail; raw capture and authoritative ledger may not.
 try:atomic(JOB/'progress.json',dict(stage=stage,pid=os.getpid(),facts=facts))
 except OSError as e:print('MONITOR_DEGRADED '+repr(e),file=sys.stderr,flush=True)


def record_child_exits(b):
 records=[]
 for child in b.processes:
  rc=child.poll()
  if rc is None:continue
  records.append(dict(pid=child.pid,kind=getattr(child,'diag_kind','unknown'),argv=getattr(child,'diag_argv',[]),returncode=rc,signal=-rc if rc<0 else None))
 if records:
  atomic(JOB/'child_exits.json',dict(children=records))
  q=subprocess.run(['dmesg'],capture_output=True,text=True,timeout=5)
  (JOB/'kernel_at_exit.txt').write_text(q.stdout+q.stderr)
 return records
