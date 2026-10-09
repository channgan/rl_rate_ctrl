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
assert protocol['authorized'] and ledger['authorized'] and ledger['native_limit']==protocol['native_limit']==2700000 and not ledger['stopped'] and not ledger['reservations'] and not ledger['attempts'] and ledger['charged_native']==0
os.environ['BATCH_TRAINING_ROOT']=str(ROOT)
atomic(ROOT/'driver_identity.json',dict(pid=os.getpid(),ppid=os.getppid(),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
training=read(Path(protocol['training_complete']));assert training['completed'] and training['updates']==40 and training['actor']==protocol['actors']['candidate']
assert not protocol['promotion_allowed'] and len(protocol['jobs'])==54
for p,h in protocol['runtime_sha256'].items():assert sha(p)==h,p
for label,actor in protocol['actors'].items():assert sha(actor['binary'])==protocol['actor_sha256'][label]
def capture(job,actor):
 global ledger
 from sync_trace_audit import check as sync_check
 jr=ROOT/'jobs'/job['id'];jr.mkdir(parents=True,exist_ok=False);pid=job.get('label')=='pid';label='baseline' if pid else 'candidate';cap=56000 if pid else 47000
 assert cap==job['cap'];p=read(ROOT/'templates'/('profile_'+job['case']+'.json'))
 p.update(v46_job_id=job['id'],v46_native_cap=cap,native_episode_tick_cap=cap,full_path=actor['binary'],head_model_id=actor['model_id'],actor_npz=actor['npz'],actor_sha256=sha(actor['binary']),continuous_score_steps=2048,audit_profile=label,audit_label=job['label'],development_seed=job['seed'],sync_runtime_id=protocol['sync_runtime_id'])
 atomic(jr/'protocol.json',p)
 assert not ledger['reservations'] and ledger['charged_native']+cap<=ledger['native_limit']
 if job['phase']=='validation':assert sum(x['charge'] for x in ledger['attempts'] if x.get('phase')=='sync_validation' and x.get('sync_runtime_id')==protocol['sync_runtime_id'])+cap<=206000
 ledger['reservations'][job['id']]=dict(cap=cap,protocol_sha256=sha(jr/'protocol.json'));atomic(LF,ledger)
 out=jr/label/job['case']/('seed_'+str(job['seed']));env=os.environ.copy();env['SUPERVISED_JOB_ROOT']=str(jr);env['PX4_CADENCE_COLLECT']='0'
 entry=dict(logical_id=job.get('logical_id',job['id']),id=job['id'],phase='sync_'+job['phase'],batch=job.get('batch'),seed=job['seed'],case=job['case'],label=job.get('label','A'),cap=cap,output=str(out),actor=actor,status='reserved',sync_runtime_id=protocol['sync_runtime_id']);child=None;started=time.monotonic();count=None
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
  with (jr/'audit.log').open('x') as stream:audit=subprocess.run([sys.executable,'-B',str(R/'audit_development.py'),str(jr)],stdout=stream,stderr=subprocess.STDOUT)
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
 completed=[]
 for job in protocol['jobs']:
  if (ROOT/'stop').exists():raise RuntimeError('explicit stop before development capture')
  rp=capture(job,protocol['actors'][job['label']]);r=read(rp);r.update(label=job['label'],result_path=rp);completed.append(r)
  atomic(ROOT/'completed.json',dict(completed=completed,fully_audited_count=len(completed)))
 assert len(completed)==54
 assert len({(x['case'],x['seed'],x['label']) for x in completed})==54
 assert sum(a['status']=='audited' for a in ledger['attempts'])==54 and len(ledger['attempts'])==54
 atomic(ROOT/'accounting_completeness.json',dict(actual_totals_are_known_lower_bounds=True,unknown_actual_attempts=[a['id'] for a in ledger['attempts'] if 'actual_native_NN' not in a or 'actual_native_PID' not in a],conservative_charged_total=ledger['charged_native'],all_failed_attempts_retained=True))
 import review54
 review54.E=ROOT
 report=review54.review(dict(completed=completed,fully_audited_count=54,actual_native_nn_calls=sum(x.get('actual_native_NN',0) for x in ledger['attempts']),actual_native_pid_calls=sum(x.get('actual_native_PID',0) for x in ledger['attempts'])))
 atomic(ROOT/'development_complete.json',dict(completed=True,episodes=54,promotion_allowed=False,performance_conditions_pass=report['performance_conditions_pass'],charged_native=ledger['charged_native']))
 ledger['stopped']=True;ledger['stop_reason']='completed54_not_promoted';atomic(LF,ledger);status(state='development_complete_not_promoted',episodes=54,charged=ledger['charged_native'])
except BaseException as e:
 ledger=read(LF);ledger.update(stopped=True,stop_reason=repr(e));atomic(LF,ledger);atomic(ROOT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));status(state='stopped_failure',error=repr(e),charged=ledger['charged_native'],reserved=ledger['reservations']);raise
