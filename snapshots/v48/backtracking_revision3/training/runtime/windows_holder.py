from pathlib import Path
import os,sys,subprocess,time,json,ctypes
def read(p):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return None
def put(p,v):Path(p).write_text(json.dumps(v,indent=2))
def boot_identity():return 'independent-process-not-clock-qualification'
root=Path(sys.argv[1]);base=Path(__file__).parent;boot=boot_identity();inside=ctypes.c_int();kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p;kernel.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
assert kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(inside))
cfg=read(root/'windows_config.json');kernel.OpenJobObjectW.restype=ctypes.c_void_p;kernel.OpenJobObjectW.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_wchar_p];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
observer_job=kernel.OpenJobObjectW(4,False,cfg['observer_job']);assert observer_job,'cannot establish observing Job identity'
in_observer=ctypes.c_int();assert kernel.IsProcessInJob(kernel.GetCurrentProcess(),observer_job,ctypes.byref(in_observer));kernel.CloseHandle(observer_job)
put(root/'holder_identity.json',dict(pid=os.getpid(),ppid=os.getppid(),boot=boot,in_job=bool(inside.value),in_observer_job=bool(in_observer.value),observer_job=cfg['observer_job'],mechanism='WMI CreateFlags BREAKAWAY; membership tested against actual observation Job'))
assert not in_observer.value,'holder belongs to observation Job; refuse workload'
env=os.environ.copy();env.pop('HOST_CLOCK_ROOT',None)
command=['C:\\Windows\\System32\\wsl.exe','-d','Ubuntu-22.04','-u','cy','--','@LINUX_HOME@/rl_rate/.venv/bin/python','-B',cfg['owner_script'],cfg['linux_root']]
with (root/'training.log').open('x') as out,(root/'training.stderr').open('x') as err:
 child=subprocess.Popen(command,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
 put(root/'training_client.json',dict(pid=child.pid,holder_pid=os.getpid(),command=command));code=child.wait()
put(root/'holder_completed.json',dict(returncode=code,success=code==0,pid=os.getpid(),client_pid=child.pid,no_host_clock_lease=True))
