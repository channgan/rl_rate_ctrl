"""Additional long-cruise diagnostic, retaining the completed v1 reversal benchmark."""
import math
import numpy as np
CASES={
    'low_x_025':dict(kind='translation',axis=0,speed=.25),
    'low_y_025':dict(kind='translation',axis=1,speed=.25),
    'low_x_050':dict(kind='translation',axis=0,speed=.5),
}
def one_leg(s,speed):
    if s<=0:return 0.,0.
    if s<1:return .5*speed*(s-math.sin(math.pi*s)/math.pi),.5*speed*(1-math.cos(math.pi*s))
    if s<16:return speed*(s-.5),speed
    if s<17:
        u=s-16
        return speed*15.5+.5*speed*(u+math.sin(math.pi*u)/math.pi),.5*speed*(1+math.cos(math.pi*u))
    return 16*speed,0.
def mission(name,seconds):
    spec=CASES[name];p=np.array([0.,0.,-5.]);v=np.zeros(3);p[spec['axis']],v[spec['axis']]=one_leg(seconds-3,spec['speed'])
    phase='cruise_forward' if 4<=seconds<19 else ('acceleration_or_braking' if 3<=seconds<4 or 19<=seconds<20 else 'hold')
    return p,v,phase
class NativeDiagnosticMission:
    def __init__(self,name,state,backend):
        assert name in CASES
        self.name=name;self.backend=backend;self.start=state['sim_us'];self.offset=np.asarray(state['px4_local_position'])-np.asarray(state['position_ned']);self.heading=float(state['px4_heading']);self.last_slot=-1;self.position_target_ned=np.array([0.,0.,-5.]);self.waypoints_reached=0
    def command(self,state,*unused):
        assert state['px4_position_control'];slot=(state['sim_us']-self.start)//100000
        if slot!=self.last_slot:
            self.position_target_ned,_,_=mission(self.name,slot*.1);self.backend.set_position_goal(self.position_target_ned+self.offset,self.heading);self.last_slot=slot
        return np.asarray(state['px4_rate_target']),-state['px4_thrust_body_z']
def tests():
    import importlib.util
    from pathlib import Path
    previous=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/low_speed_fixed_mission_diagnostic_v1/low_speed_missions.py')
    spec=importlib.util.spec_from_file_location('immutable_v1_mission',previous);v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)
    for name,spec in CASES.items():
        # Entire initial hold, ramp and first cruise segment exactly match v1.
        for t in np.arange(900)*.01:
            for a,b in zip(mission(name,t)[:2],v1.mission(name,t)[:2]):np.testing.assert_array_equal(a,b)
        for t in [0,3,4,19,20,20.48]:
            p,v,_=mission(name,t)
            for eps in [-1e-7,1e-7]:
                q,w,_=mission(name,t+eps);assert np.max(np.abs(q-p))<1e-6 and np.max(np.abs(w-v))<1e-6
        p,v,_=mission(name,20.48);expected=np.array([0.,0.,-5.]);expected[spec['axis']]=16*spec['speed'];np.testing.assert_array_equal(p,expected);np.testing.assert_array_equal(v,[0.,0.,0.])
        for t in [4.1,9,13,18.9]:
            p,v,_=mission(name,t);derivative=(mission(name,t+1e-5)[0]-mission(name,t-1e-5)[0])/2e-5;np.testing.assert_allclose(v,derivative,atol=1e-9);assert v[spec['axis']]==spec['speed']
        cruise=sum(mission(name,(i+1)*.01)[2].startswith('cruise') for i in range(2048));assert cruise==1500
    class Mock:
        def __init__(self):self.calls=[]
        def set_position_goal(self,p,h):self.calls.append((p.copy(),h))
    state=dict(sim_us=35000000,px4_local_position=[1,2,-4.7],position_ned=[0,0,-5],px4_heading=.12,px4_position_control=True,px4_rate_target=[.01,.02,.03],px4_thrust_body_z=-.6);b=Mock();o=NativeDiagnosticMission('low_x_025',state,b)
    for i in range(2049):
        state['sim_us']=35000000+i*10000;rates,thrust=o.command(state);np.testing.assert_array_equal(rates,[.01,.02,.03]);assert thrust==.6
    assert len(b.calls)==205
    return dict(passed=True,initial9s_identical_to_v1=True,continuous_position_velocity=True,analytic_velocity_checked=True,cruise_samples=1500,cruise_seconds=15,final_displacement_m=[4.,4.,8.],goal_10hz_count=205,native_rate_collective_unchanged=True,original_reversal_and_steps_results_preserved=True)
if __name__=='__main__':
    import json
    print(json.dumps(tests(),indent=2))
