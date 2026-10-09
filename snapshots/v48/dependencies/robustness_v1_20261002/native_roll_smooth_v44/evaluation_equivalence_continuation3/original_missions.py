"""Paired GOAL publication-cadence diagnostic: identical continuous motion reference."""
import math
import numpy as np
BASE_CASES={'low_x_025':dict(axis=0,speed=.25),'low_y_025':dict(axis=1,speed=.25),'low_x_050':dict(axis=0,speed=.5)}
CASES={f'{name}_goal{hz}':dict(kind='translation',base_case=name,goal_hz=hz,**spec) for name,spec in BASE_CASES.items() for hz in [10,100]}
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
        self.goal_hz=CASES[name]['goal_hz'];self.period_us=100000 if self.goal_hz==10 else 10000;self.period_s=.1 if self.goal_hz==10 else .01;self.publications=0
    def command(self,state,*unused):
        assert state['px4_position_control'];slot=(state['sim_us']-self.start)//self.period_us
        if slot!=self.last_slot:
            self.position_target_ned,_,_=mission(self.name,slot*self.period_s);self.backend.set_position_goal(self.position_target_ned+self.offset,self.heading);self.last_slot=slot;self.publications+=1
        return np.asarray(state['px4_rate_target']),-state['px4_thrust_body_z']
def tests():
    from pathlib import Path
    import importlib.util
    source=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/low_speed_long_cruise_diagnostic_v2/low_speed_missions.py');spec=importlib.util.spec_from_file_location('frozen_mission_v2',source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    class Mock:
        def __init__(self):self.calls=[]
        def set_position_goal(self,p,h):self.calls.append((p.copy(),h))
    for name,c in CASES.items():
        for i in range(2049):
            t=i*.01
            for a,b in zip(mission(name,t)[:2],old.mission(c['base_case'],t)[:2]):np.testing.assert_array_equal(a,b)
        state=dict(sim_us=35000000,px4_local_position=[1,2,-4.7],position_ned=[0,0,-5],px4_heading=.12,px4_position_control=True,px4_rate_target=[.01,.02,.03],px4_thrust_body_z=-.6);b=Mock();o=NativeDiagnosticMission(name,state,b)
        for i in range(2049):
            state['sim_us']=35000000+i*10000;rate,thrust=o.command(state);np.testing.assert_array_equal(rate,[.01,.02,.03]);assert thrust==.6
        expected=205 if c['goal_hz']==10 else 2049;assert o.publications==len(b.calls)==expected
        for j,(goal,yaw) in enumerate(b.calls):
            expected_goal=old.mission(c['base_case'],j*o.period_s)[0]+o.offset;np.testing.assert_array_equal(goal,expected_goal);assert yaw==.12
    return dict(passed=True,all2049_reference_samples_exactly_identical_to_v2=True,goal10_count=205,goal100_count=2049,goal_grid_exact=True,native_rate_collective_unchanged=True,models_and_physical_motion_limits_unchanged=True)
if __name__=='__main__':
    import json
    print(json.dumps(tests(),indent=2))
