"""Gateway to the frozen Windows holder; supervision remains in that release."""
from pathlib import Path
import ctypes,json,os,subprocess,sys,time,uuid
R=None
def read(p):return json.loads(Path(p).read_text())
def put(p,v):Path(p).write_text(json.dumps(v,indent=2))
class BasicLimit(ctypes.Structure):
 _fields_=[('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),('flags',ctypes.c_uint32),('minimum',ctypes.c_size_t),('maximum',ctypes.c_size_t),('active',ctypes.c_uint32),('affinity',ctypes.c_size_t),('priority',ctypes.c_uint32),('scheduling',ctypes.c_uint32)]

class ExtendedLimit(ctypes.Structure):
 _fields_=[('basic',BasicLimit),('io',ctypes.c_ulonglong*6),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]

def observation_job(root):
 k=ctypes.WinDLL('kernel32',use_last_error=True);k.CreateJobObjectW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p];k.CreateJobObjectW.restype=ctypes.c_void_p;k.SetInformationJobObject.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint];k.AssignProcessToJobObject.argtypes=[ctypes.c_void_p,ctypes.c_void_p];k.TerminateJobObject.argtypes=[ctypes.c_void_p,ctypes.c_uint];k.CloseHandle.argtypes=[ctypes.c_void_p]
 name='Local\\CodexOfflineObserver_'+root.name;handle=k.CreateJobObjectW(None,name);assert handle
 limit=ExtendedLimit();limit.basic.flags=0x2000;assert k.SetInformationJobObject(handle,9,ctypes.byref(limit),ctypes.sizeof(limit))
 return k,handle,name

def start(root):
 command=subprocess.list2cmdline([str(Path(sys.executable).with_name('pythonw.exe')),'-B',str(R/'windows_holder.py'),str(root)])
 ps="$s=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32]16777216}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"+command.replace("'","''")+"';CurrentDirectory='@DRL_ROOT_WINDOWS@';ProcessStartupInformation=$s} | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress"
 p=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',ps],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
 if p.returncode:raise RuntimeError(p.stderr)
 result=json.loads(p.stdout);assert result['ReturnValue']==0,result;put(root/'wmi_launch.json',result);return result

def observer(arguments):
 global R
 assert os.name=='nt' and len(arguments)==3
 root=Path(arguments[0]);R=Path(arguments[1]);token=arguments[2]
 intent=read(root/'standard_launch_intent.json');assert intent['token']==token and intent['runtime']==str(R)
 cfg=read(root/'windows_config.json');k=ctypes.WinDLL('kernel32',use_last_error=True)
 k.OpenJobObjectW.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_wchar_p];k.OpenJobObjectW.restype=ctypes.c_void_p;k.GetCurrentProcess.restype=ctypes.c_void_p;k.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)];k.CloseHandle.argtypes=[ctypes.c_void_p]
 handle=k.OpenJobObjectW(4,False,cfg['observer_job']);assert handle
 inside=ctypes.c_int();assert k.IsProcessInJob(k.GetCurrentProcess(),handle,ctypes.byref(inside)) and inside.value;k.CloseHandle(handle)
 started=time.perf_counter()
 while not (root/'observer_go').exists():
  assert time.perf_counter()-started<30;time.sleep(.02)
 start(root)
 while True:time.sleep(.1)

def launch(root,runtime,policy_path):
 global R
 assert os.name=='nt';R=runtime
 kernel,job_handle,job_name=observation_job(root)
 cfg=read(root/'windows_config.json');cfg['observer_job']=job_name;put(root/'windows_config.json',cfg)
 token=uuid.uuid4().hex
 with (root/'standard_launch_intent.json').open('x') as f:json.dump(dict(token=token,runtime=str(runtime),policy=str(policy_path),no_retry=True),f)
 script=Path(__file__).resolve().parents[1]/'scripts/supervise_training.py'
 child=subprocess.Popen([sys.executable,'-B',str(script),'_observer',str(root),str(runtime),token],creationflags=subprocess.CREATE_NO_WINDOW)
 assert kernel.AssignProcessToJobObject(job_handle,int(child._handle));(root/'observer_go').write_text('standard gateway admitted')
 begin=time.perf_counter()
 while not (root/'holder_identity.json').exists() or not (root/'wmi_launch.json').exists():
  if time.perf_counter()-begin>30:raise RuntimeError('holder registration unresolved; do not retry')
  time.sleep(.05)
 identity=read(root/'holder_identity.json');assert not identity['in_observer_job']
 assert kernel.TerminateJobObject(job_handle,91);assert child.wait(timeout=10)==91
 put(root/'observer_exit.json',dict(pid=child.pid,returncode=91,host=time.perf_counter(),job_name=job_name,method='TerminateJobObject; independent holder persists'))
 kernel.CloseHandle(job_handle)
 return dict(started=True,holder_pid=identity['pid'],complete=False,root=str(root),note='Follow host_completed, holder_completed and settlement; launch is not success')
