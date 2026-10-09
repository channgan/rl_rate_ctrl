"""Training-only GOAL conditioner. No simulator launch on import."""
from pathlib import Path
import json,hashlib
import numpy as np
from v46_core import prefix_goal,entry_goal,score_goal
ROOT=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/native_entry_coverage_v46')
def require_authorized(protocol):
 approval=ROOT/'approval.json';ledger=json.loads((ROOT/'ledger.json').read_text())
 if not approval.exists() or not ledger.get('native_authorized',False):raise PermissionError('v46 native execution is blocked pending parent protocol verification')
 a=json.loads(approval.read_text());digest=hashlib.sha256((ROOT/'protocol_revision2/protocol.json').read_bytes()).hexdigest()
 if not a.get('approved') or a.get('protocol_sha256')!=digest:raise PermissionError('v46 approval does not match frozen protocol')
 frozen=json.loads((ROOT/'protocol_revision2/protocol.json').read_text())
 for name,h in frozen['implementation_sha256'].items():
  if hashlib.sha256((ROOT/'protocol_revision2'/name).read_bytes()).hexdigest()!=h:raise PermissionError('Implementation changed after protocol review: '+name)
 job=protocol['v46_job_id'];reservation=ledger['reservations'].get(job)
 if reservation is None or len(ledger['reservations'])!=1:raise PermissionError('Canonical v46 budget reservation required before native launch')
 if ledger.get('stopped'):raise PermissionError('v46 stopped; no retries')
 if int(reservation['cap'])!=protocol['v46_native_cap']:raise PermissionError('Native cap mismatch')
def mechanism_goal(out,state):
 raw=(Path(out)/'mechanism.bin').read_bytes();assert raw[:8]==b'MECH0042' and (len(raw)-8)%208==0
 a=np.frombuffer(raw[8:],np.dtype([('m','<u8',(12,)),('v','<f4',(28,))]));m=a['m'][-1];v=a['v'][-1]
 assert m[0]==state['sim_us'] and m[6] and m[2]>0 and m[2]<=m[1] and v[26]==1 and np.isfinite(v[:4]).all()
 return v[:3].astype(float),float(v[3]),dict(sample_us=int(m[0]),latch_us=int(m[1]),goal_publication_us=int(m[2]),goal=v[:4].astype(float).tolist())
class Prefix:
 def __init__(self,profile,state,backend,out):
  assert state['sim_us']==30000000;self.profile=profile;self.backend=backend;self.out=Path(out);self.anchor,self.heading,self.proof=mechanism_goal(out,state);self.yaw=float(state['px4_heading']);self.commands=[]
  (self.out/'v46_prefix_start.json').write_text(json.dumps(dict(profile=profile,anchor=self.anchor.tolist(),heading=self.heading,yaw=self.yaw,initial_state=state,proof=self.proof)))
 def command(self,state):
  assert 30000000<=state['sim_us']<35000000;goal=prefix_goal(self.anchor,self.yaw,(state['sim_us']-30000000)*1e-6,self.profile);self.backend.set_position_goal(goal,self.heading);self.commands.append(dict(sample_us=state['sim_us'],goal=goal.tolist(),heading=self.heading))
 def save(self):(self.out/'v46_prefix_commands.json').write_text(json.dumps(self.commands))
class EntryMission:
 def __init__(self,case,profile,state,backend,out):
  assert state['sim_us']==35000000;self.case=case;self.backend=backend;self.out=Path(out);self.start=state['sim_us'];old,self.heading,proof=mechanism_goal(out,state);self.base_native=entry_goal(old,float(state['px4_heading']),profile);offset=np.asarray(state['px4_local_position'])-np.asarray(state['position_ned']);self.base_truth=self.base_native-offset
  # Match the existing trajectory metadata convention for the original low-speed helper.
  rel0=np.array([0.,0.,-5.]) if case.startswith('low_') else np.zeros(3);self.base_world=self.base_truth-rel0;self.position_target_ned=self.base_truth.copy();self.waypoints_reached=0;self.last_slot=-1;self.goal_hz=100;self.publications=0;self.commands=[]
  (self.out/'v46_entry_snapshot.json').write_text(json.dumps(dict(profile=profile,initial_state=state,old_goal_proof=proof,base_native=self.base_native.tolist(),base_truth=self.base_truth.tolist(),heading=self.heading,offset=offset.tolist(),entry_jump_body_m=profile['entry_jump_body_m'],history_reset=False)))
 def command(self,state,*unused):
  slot=(state['sim_us']-self.start)//10000
  if slot!=self.last_slot:
   assert slot==self.last_slot+1;elapsed=slot*.01;goal=score_goal(self.base_native,elapsed,self.case);self.position_target_ned=score_goal(self.base_truth,elapsed,self.case);self.backend.set_position_goal(goal,self.heading);self.commands.append(dict(slot=int(slot),sample_us=state['sim_us'],goal=goal.tolist(),heading=self.heading));self.last_slot=slot;self.publications+=1
  return np.asarray(state['px4_rate_target']),-state['px4_thrust_body_z']
 def save(self):(self.out/'v46_score_commands.json').write_text(json.dumps(self.commands))

def verify_commands(out,profile,case):
 out=Path(out);prefix=json.loads((out/'v46_prefix_start.json').read_text());commands=json.loads((out/'v46_prefix_commands.json').read_text());assert len(commands)==500
 for i,q in enumerate(commands):
  assert q['sample_us']==30000000+i*10000;expected=prefix_goal(prefix['anchor'],prefix['yaw'],i*.01,profile);np.testing.assert_allclose(q['goal'],expected,rtol=0,atol=1e-12)
 entry=json.loads((out/'v46_entry_snapshot.json').read_text());commands=json.loads((out/'v46_score_commands.json').read_text());assert len(commands)==2049
 for i,q in enumerate(commands):
  assert q['sample_us']==35000000+i*10000;np.testing.assert_allclose(q['goal'],score_goal(entry['base_native'],i*.01,case),rtol=0,atol=1e-12)
 return dict(passed=True,prefix_commands=500,score_commands=2049,history_reset=False)
