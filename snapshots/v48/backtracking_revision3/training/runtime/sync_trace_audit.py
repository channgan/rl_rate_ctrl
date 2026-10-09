from pathlib import Path
import json,numpy as np,hashlib
def check(out):
 out=Path(out);raw=(out/'closed_loop_sync.bin').read_bytes();assert raw[:8]==b'SYNC0001' and (len(raw)-8)%96==0
 a=np.frombuffer(raw[8:],'<u8').reshape(-1,12);assert len(a)>20480
 assert np.all(a[:,[1,3,4,5,6]]==a[:,0,None]) and np.all(a[:,8]==1)
 assert np.all(a[1:,9]==1) and np.all(a[1:,7]+1000==a[1:,0]),'plant consumed wrong/invalid preceding sample'
 assert np.all(np.diff(a[:,11].astype(np.int64))>=0),'wall record ordering'
 groups=[];last_end=None;last_seq=None
 for seq in np.unique(a[:,10]):
  b=a[a[:,10]==seq];assert 2<=len(b)<=101
  assert np.all(np.diff(b[:,0].astype(np.int64))==1000)
  assert b[0,2]==seq-1 and np.all(b[1:,2]==seq)
  if last_end is not None:assert b[0,0]==last_end and seq==last_seq+1
  last_end=int(b[-1,0]);last_seq=int(seq);groups.append(dict(seq=int(seq),ticks=len(b)-1,start=int(b[0,0]),end=last_end))
 r=json.loads((out/'result.json').read_text());initial=json.loads((out/'alignment.json').read_text())['bootstrap_initial']
 assert a[0,0]==initial['sim_us'] and a[-1,0]==55480000 and r['steps']==2048
 transitions=a[np.r_[False,a[1:,0]>a[:-1,0]]];assert len(transitions)==(55480000-int(a[0,0]))//1000
 rr=np.frombuffer((out/'rate_reference.bin').read_bytes()[8:],np.dtype([('m','<u8',(11,)),('v','<f4',(10,))]));t=rr['m'][:,0];ix=np.searchsorted(t,transitions[:,0]);assert np.all(ix<len(t)) and np.array_equal(t[ix],transitions[:,0])
 gyro=np.frombuffer((out/'gyro_audit.bin').read_bytes()[8:],np.dtype([('m','<u8',(10,)),('v','<f4',(26,))]));gt=gyro['m'][:,0];ix=np.searchsorted(gt,transitions[:,0]);assert np.all(ix<len(gt)) and np.array_equal(gt[ix],transitions[:,0])
 phys=np.frombuffer((out/'physics_capture.bin').read_bytes()[8:],'<f8').reshape(-1,25);pt=phys[:,2].astype(np.uint64);ix=np.searchsorted(pt,transitions[:,0]);assert np.all(ix<len(pt)) and np.array_equal(pt[ix],transitions[:,0]);assert np.array_equal(phys[ix,3].astype(np.uint64),transitions[:,2])
 report=dict(passed=True,post_INIT_transitions=len(transitions),trace_records=len(a),STEP_requests=len(groups),start_sim_us=int(a[0,0]),end_sim_us=int(a[-1,0]),exact_1ms=True,previous_boundary_motor_applied=True,all_current_filter_torque_allocator_motor_receipt_acks=True,raw_capture_correspondence=True,trace_sha256=hashlib.sha256(raw).hexdigest(),first_group=groups[0],last_group=groups[-1])
 (out/'sync_audit.json').write_text(json.dumps(report,indent=2));return report
