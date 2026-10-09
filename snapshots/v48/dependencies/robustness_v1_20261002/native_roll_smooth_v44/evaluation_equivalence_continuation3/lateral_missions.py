"""Frozen matched-entry GOAL diagnostic; no truth-derived reconstruction of old GOAL."""
import json, math
from pathlib import Path
import numpy as np
DTYPE=np.dtype([('m','<u8',(12,)),('v','<f4',(28,))])
CASES={'lateral_plus_goal100':1,'lateral_minus_goal100':-1}
def read_mechanism(path):
    raw=Path(path).read_bytes()
    assert raw[:8]==b'MECH0042' and (len(raw)-8)%208==0, 'Truncated mechanism capture'
    a=np.frombuffer(raw[8:],DTYPE);assert len(a)>0
    assert np.all(np.diff(a['m'][:,0].astype(np.int64))>0)
    return a
def one_leg(s,speed=.25):
    if s<=0:return 0.,0.
    if s<1:return .5*speed*(s-math.sin(math.pi*s)/math.pi),.5*speed*(1-math.cos(math.pi*s))
    if s<16:return speed*(s-.5),speed
    if s<17:
        u=s-16;return speed*15.5+.5*speed*(u+math.sin(math.pi*u)/math.pi),.5*speed*(1+math.cos(math.pi*u))
    return 16*speed,0.
def mission(name,seconds):
    assert name in CASES
    p=np.zeros(3);v=np.zeros(3);p[1],v[1]=one_leg(seconds-3)
    phase='cruise_forward' if 4<=seconds<19 else ('acceleration_or_braking' if 3<=seconds<4 or 19<=seconds<20 else 'hold')
    return p,v,phase
def entry_values(old_goal,yaw,sign):
    delta=sign*.04*np.array([-math.sin(yaw),math.cos(yaw),0.])
    return np.asarray(old_goal,dtype=float)+delta,delta
class NativeDiagnosticMission:
    def __init__(self,name,state,backend):
        self.name=name;self.backend=backend;self.start=state['sim_us'];assert self.start==35000000
        self.out=Path(backend.mechanism_output)
        a=read_mechanism(self.out/'mechanism.bin');m=a['m'][-1];v=a['v'][-1]
        assert m[0]==self.start and m[6] and 0<m[2]<=m[1] and np.isfinite(v[:4]).all() and v[26]==1
        self.offset=np.asarray(state['px4_local_position'])-np.asarray(state['position_ned'])
        self.base_native,self.delta=entry_values(v[:3],float(state['px4_heading']),CASES[name])
        self.base_world=self.base_native-self.offset;self.heading=float(v[3]);self.position_target_ned=self.base_world.copy()
        self.last_slot=-1;self.waypoints_reached=0;self.goal_hz=100;self.publications=0;self.commands=[]
        self.snapshot=dict(sample_us=int(m[0]),diagnostic_latch_us=int(m[1]),old_goal_timestamp_us=int(m[2]),old_goal_position=v[:3].astype(float).tolist(),old_goal_heading=self.heading,entry_yaw=float(state['px4_heading']),signed_body_lateral_m=CASES[name]*.04,delta_native=self.delta.tolist(),base_native=self.base_native.tolist(),coordinate_offset=self.offset.tolist(),base_world=self.base_world.tolist(),active_model_id=int(m[4]),heading_control=True,ordering='STEP at35s completes capture flush before first new GOAL; controller remains unchanged across sample35s; no simulator state reset')
        (self.out/'entry_snapshot.json').write_text(json.dumps(self.snapshot,indent=2))
    def command(self,state,*unused):
        assert state['px4_position_control'];slot=(state['sim_us']-self.start)//10000
        if slot!=self.last_slot:
            assert slot==self.last_slot+1
            rel,vel,phase=mission(self.name,slot*.01);goal=self.base_native+rel
            self.position_target_ned=self.base_world+rel
            ack=self.backend.set_position_goal(goal,self.heading)
            self.commands.append(dict(slot=int(slot),request_sim_us=int(state['sim_us']),native_goal=goal.tolist(),heading=self.heading))
            self.last_slot=slot;self.publications+=1
        return np.asarray(state['px4_rate_target']),-state['px4_thrust_body_z']
    def save(self):
        (self.out/'goal_commands.json').write_text(json.dumps(self.commands))
