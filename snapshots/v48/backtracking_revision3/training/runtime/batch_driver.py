"""Existing independent batch execution: fixed sync probes then fresh collection/update."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,time,fcntl,traceback,signal
ROOT=Path(sys.argv[1]);R=Path(__file__).parent
def read(p):return json.loads(Path(p).read_text())
def atomic(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(**fields):
 try:atomic(ROOT/'status.json',fields)
 except OSError as e:print('MONITOR_DEGRADED '+repr(e),flush=True)
def log(event,**fields):print(json.dumps(dict(event=event,pid=os.getpid(),**fields)),flush=True)
def stop_handler(sig,frame):(ROOT/'stop').write_text('explicit signal '+str(sig))
signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
lock=(ROOT/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
protocol=read(ROOT/'protocol.json');LF=Path(protocol.get('canonical_ledger',ROOT/'ledger.json'));ledger=read(LF)
assert protocol['authorized'] and ledger['native_limit']==protocol['native_limit']==2000000 and not ledger['stopped'] and not ledger['reservations']
os.environ['BATCH_TRAINING_ROOT']=str(ROOT)
atomic(ROOT/'driver_identity.json',dict(pid=os.getpid(),ppid=os.getppid(),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
if protocol.get('zero_nn_fixture'):
 from local_guards import guard_next_native,guard_finite
 (ROOT/'status.json').mkdir();count=0;reason=''
 while True:
  if (ROOT/'stop').exists():reason='explicit_stop';break
  try:guard_next_native(count,1,32)
  except RuntimeError:reason='internal_budget';break
  guard_finite([count,0]);count+=1;atomic(ROOT/'checkpoint.json',dict(completed_actions=count,cap=32,fixture_only=True));status(state='fixture_running',count=count)
  if protocol['fixture_case']=='explicit_stop' and count==5:(ROOT/'stop').write_text('explicit fixture stop')
  time.sleep(.1)
 assert count==(5 if protocol['fixture_case']=='explicit_stop' else 32)
 try:guard_finite([float('nan')])
 except RuntimeError:pass
 else:raise AssertionError('nonfinite guard failed')
 atomic(ROOT/'fixture_result.json',dict(passed=True,count=count,reason=reason,native_calls=0,checkpoint=read(ROOT/'checkpoint.json'),host_clock_dependency=False,monitor_failure_did_not_stop=True));raise SystemExit(0)
resume=read(ROOT/'runtime_replacement.json')
assert ledger['charged_native']==resume['historical_charge']==608641 and ledger['formal_optimizer_updates']==17 and len(ledger['attempts'])==14
assert hashlib.sha256(json.dumps(ledger['attempts'],sort_keys=True).encode()).hexdigest()==resume['historical_attempts_sha256']
for path,h in read(ROOT/'carried_inputs.json')['sha256'].items():assert sha(path)==h,path
for p,h in protocol['runtime_sha256'].items():assert sha(p)==h,p
def capture(job,actor):
 global ledger
 from sync_trace_audit import check as sync_check
 jr=ROOT/'jobs'/job['id'];jr.mkdir(parents=True,exist_ok=False);pid=job.get('label')=='pid';label='baseline' if pid else 'candidate';cap=56000 if pid else 47000
 assert cap==job['cap'];p=read(ROOT/'templates'/('profile_'+str(job['profile_index'])+'.json'))
 p.update(v46_job_id=job['id'],v46_native_cap=cap,native_episode_tick_cap=cap,full_path=actor['binary'],head_model_id=actor['model_id'],actor_npz=actor['npz'],actor_sha256=sha(actor['binary']),continuous_score_steps=2048,audit_profile=label,audit_label='pid' if pid else 'A',sync_runtime_id=protocol['sync_runtime_id'])
 p['v46_training_entry_profile']['seed']=job['seed'];atomic(jr/'protocol.json',p)
 assert not ledger['reservations'] and ledger['charged_native']+cap<=ledger['native_limit']
 if job['phase']=='validation':assert sum(x['charge'] for x in ledger['attempts'] if x.get('phase')=='sync_validation' and x.get('sync_runtime_id')==protocol['sync_runtime_id'])+cap<=206000
 ledger['reservations'][job['id']]=dict(cap=cap,protocol_sha256=sha(jr/'protocol.json'));atomic(LF,ledger)
 out=jr/label/job['case']/('seed_'+str(job['seed']));env=os.environ.copy();env['SUPERVISED_JOB_ROOT']=str(jr);env['PX4_CADENCE_COLLECT']='0'
 entry=dict(id=job['id'],phase='sync_'+job['phase'],batch=job.get('batch'),seed=job['seed'],case=job['case'],label=job.get('label','A'),cap=cap,output=str(out),actor=actor,status='reserved',sync_runtime_id=protocol['sync_runtime_id']);child=None;started=time.monotonic();count=None
 try:
  with (jr/'runner.log').open('x') as stream:
   child=subprocess.Popen([sys.executable,'-B',str(R/'native_case.py'),'--protocol',str(jr/'protocol.json'),'--controller','pid','--profile',label,'--case',job['case'],'--seed',str(job['seed'])],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   entry['pid']=child.pid;atomic(jr/'attempt.json',entry);log('CAPTURE_STARTED',job=job,child_pid=child.pid,actor=actor);sent=False
   while child.poll() is None:
    if ((ROOT/'stop').exists() or time.monotonic()-started>1800) and not sent:(jr/'stop').write_text('explicit stop or local deadline');sent=True
    status(state='collecting',phase=job['phase'],job=job['id'],runner_pid=child.pid,charged=ledger['charged_native'],reserved=cap,formal_optimizer_updates=ledger['formal_optimizer_updates']);time.sleep(5)
  entry['returncode']=child.returncode
  if child.returncode!=0 or not (out/'result.json').exists():
   primary=out/'sync_failure.json' if (out/'sync_failure.json').exists() else out/'first_failure.json'
   entry['primary_failure']=read(primary) if primary.exists() else None
   entry['primary_failure_source']=str(primary) if primary.exists() else str(out/'failure.txt')
   entry['runner_log']=str(jr/'runner.log');entry['export_status']=read(out/'export_status.json') if (out/'export_status.json').exists() else None
   raise RuntimeError('native runner failed; primary='+repr(entry['primary_failure'])+'; see '+str(jr/'runner.log'))
  result=read(out/'result.json');export=read(out/'export_status.json');counts=result['nn_final_audit'];actual=counts['nn_cycles']+counts['pid_cycles']
  assert export['success'] and 0<actual<=cap and (counts['nn_cycles']==0 if pid else counts['pid_cycles']==0)
  assert result['steps']==2048 and result['survived_2048'] and result['physical_failure'] is None and not (ROOT/'stop').exists()
  with (jr/'audit.log').open('x') as stream:audit=subprocess.run([sys.executable,'-B',str(R/'audit_capture.py'),str(jr)],stdout=stream,stderr=subprocess.STDOUT)
  assert audit.returncode==0,'full original capture audit failed'
  sync=sync_check(out);assert sync['passed'];count=actual
  entry.update(status='audited',actual_native_NN=counts['nn_cycles'],actual_native_PID=counts['pid_cycles'],charge=count,audit_sha256=sha(out/'combined_audit.json'),sync_audit_sha256=sha(out/'sync_audit.json'));log('CAPTURE_AUDITED',job=job['id'],label=entry['label'],native_calls=count,steps=2048,sync=sync)
 except BaseException as e:
  entry.update(status='stopped_failure',error=repr(e),traceback=traceback.format_exc(),charge=cap if count is None else count);ledger.update(stopped=True,stop_reason=repr(e));raise
 finally:
  if child is not None and child.poll() is None:(jr/'stop').write_text('failure: export and stop');child.wait()
  living=[read(p) for p in (jr/'owned').glob('*.json') if Path('/proc/'+str(read(p)['pid'])).exists()]
  if living:atomic(ROOT/'blocked_ownership.json',dict(job=job['id'],owned=living,reservation_retained=True));raise RuntimeError('native ownership unresolved')
  entry.setdefault('charge',cap);entry['elapsed_seconds']=time.monotonic()-started;ledger['charged_native']+=entry['charge'];ledger['attempts'].append(entry);ledger['reservations'].pop(job['id']);atomic(jr/'attempt.json',entry);atomic(LF,ledger)
 return str(out/'result.json')
try:
 actor=read(ROOT/'training/batch_0/export.json')
 for batch in range(1,4):
  if (ROOT/'stop').exists():raise RuntimeError('explicit stop before batch')
  paths=read(ROOT/'carried_inputs.json')['manifests']['1']['results'] if batch==1 else [capture(job,actor) for job in protocol['jobs'] if job['batch']==batch]
  assert len(paths)==7
  atomic(ROOT/('batch_'+str(batch)+'_captures.json'),dict(actor=actor,results=paths));log('BATCH_CAPTURE_COMPLETE',batch=batch,paths=paths)
  status(state='updating',batch=batch,charged=ledger['charged_native'],formal_optimizer_updates=ledger['formal_optimizer_updates'])
  with (ROOT/('batch_'+str(batch)+'_update.log')).open('x') as stream:done=subprocess.run([sys.executable,'-B',str(R/'update_batch.py'),str(ROOT),str(batch)],stdout=stream,stderr=subprocess.STDOUT)
  assert done.returncode==0,'batch update failed'
  actor=read(ROOT/'training'/('batch_'+str(batch))/'export.json');ledger=read(LF);ledger['formal_optimizer_updates']=(batch+1)*10;atomic(LF,ledger);log('BATCH_UPDATED',batch=batch,updates=ledger['formal_optimizer_updates'],actor=actor)
 atomic(ROOT/'training_complete.json',dict(completed=True,actor=actor,updates=40,promotion_allowed=False,development_required=True));status(state='training_complete_not_promoted',updates=40,charged=ledger['charged_native'])
 assert subprocess.run([sys.executable,'-B',str(R/'finalize_and_bind.py'),str(ROOT)]).returncode==0, 'Final model verification failed'
 dev=ROOT.parent/'development'
 status(state='development_running',updates=40,charged=ledger['charged_native'])
 with (dev/'driver.log').open('x') as stream:completed_dev=subprocess.run([sys.executable,'-B',str(dev/'runtime/batch_driver.py'),str(dev)],stdout=stream,stderr=subprocess.STDOUT)
 assert completed_dev.returncode==0, 'Development stopped; inspect separate ledger and failure'
 status(state='training_and_development_complete_not_promoted',updates=40,charged=ledger['charged_native'])
except BaseException as e:
 ledger=read(LF);ledger.update(stopped=True,stop_reason=repr(e));atomic(LF,ledger);atomic(ROOT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));status(state='stopped_failure',error=repr(e),charged=ledger['charged_native'],reserved=ledger['reservations'],formal_optimizer_updates=ledger['formal_optimizer_updates']);log('STOPPED',error=repr(e));raise
