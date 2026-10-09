"""Deterministic world-NED position missions; native PX4 retains all upper control."""
import math
import numpy as np

CASES = {
    'low_x_025': dict(kind='translation',axis=0,speed=.25),
    'low_y_025': dict(kind='translation',axis=1,speed=.25),
    'low_x_050': dict(kind='translation',axis=0,speed=.5),
    'fixed_hover_steps': dict(kind='steps'),
}

def one_leg(s, speed):
    if s <= 0:return 0.,0.
    if s < 1:return .5*speed*(s-math.sin(math.pi*s)/math.pi),.5*speed*(1-math.cos(math.pi*s))
    if s < 6:return speed*(s-.5),speed
    if s < 7:
        u=s-6
        return speed*5.5+.5*speed*(u+math.sin(math.pi*u)/math.pi),.5*speed*(1+math.cos(math.pi*u))
    return 6*speed,0.

def mission(name, seconds):
    spec=CASES[name];p=np.array([0.,0.,-5.]);v=np.zeros(3);phase='hold'
    if spec['kind']=='steps':
        goals=[[0.,0.,-5.],[.5,0.,-5.],[0.,.5,-5.],[-.5,0.,-5.],[0.,0.,-5.]]
        index=sum(seconds>=t for t in [0.,3.,8.,13.,18.])-1
        return np.array(goals[max(index,0)]),v,'step_'+str(max(index,0))
    forward,fv=one_leg(seconds-3,spec['speed']);backward,bv=one_leg(seconds-13,spec['speed'])
    p[spec['axis']]=forward-backward;v[spec['axis']]=fv-bv
    if 4<=seconds<9:phase='cruise_forward'
    elif 14<=seconds<19:phase='cruise_reverse'
    elif 3<=seconds<4 or 9<=seconds<10 or 13<=seconds<14 or 19<=seconds<20:phase='acceleration_or_braking'
    return p,v,phase

class NativeDiagnosticMission:
    def __init__(self,name,state,backend):
        assert name in CASES
        self.name=name;self.backend=backend;self.start=state['sim_us']
        self.offset=np.asarray(state['px4_local_position'])-np.asarray(state['position_ned'])
        # Retain measured initial native heading: no added yaw turn/reference controller.
        self.heading=float(state['px4_heading']);self.last_slot=-1
        self.position_target_ned=np.array([0.,0.,-5.]);self.waypoints_reached=0
    def command(self,state,*unused):
        assert state['px4_position_control']
        elapsed_us=state['sim_us']-self.start;slot=elapsed_us//100000
        if slot!=self.last_slot:
            self.position_target_ned,_,_=mission(self.name,slot*.1)
            self.backend.set_position_goal(self.position_target_ned+self.offset,self.heading)
            self.last_slot=slot
        return np.asarray(state['px4_rate_target']),-state['px4_thrust_body_z']

def tests():
    for name,spec in CASES.items():
        if spec['kind']=='steps':continue
        for t in [0,3,4,9,10,13,14,19,20,20.48]:
            p,v,_=mission(name,t);assert np.isfinite(p).all() and np.isfinite(v).all()
            for eps in [-1e-7,1e-7]:
                q,w,_=mission(name,t+eps)
                assert np.max(np.abs(q-p))<1e-6 and np.max(np.abs(w-v))<1e-6
        np.testing.assert_allclose(mission(name,20.48)[0],[0,0,-5],atol=1e-12)
        for t in [4.1,7,8.9,14.1,17,18.9]:
            p,v,_=mission(name,t);numeric=(mission(name,t+1e-5)[0]-mission(name,t-1e-5)[0])/2e-5
            np.testing.assert_allclose(v,numeric,atol=1e-9)
            assert abs(v[spec['axis']])==spec['speed']
    for t,goal in [(0,[0,0,-5]),(3,[.5,0,-5]),(8,[0,.5,-5]),(13,[-.5,0,-5]),(18,[0,0,-5])]:
        np.testing.assert_array_equal(mission('fixed_hover_steps',t)[0],goal)
    class Mock:
        def __init__(self):self.calls=[]
        def set_position_goal(self,p,h):self.calls.append((p.copy(),h))
    state=dict(sim_us=35000000,px4_local_position=[1,2,-4.7],position_ned=[0,0,-5],px4_heading=.12,px4_position_control=True,px4_rate_target=[.01,.02,.03],px4_thrust_body_z=-.6)
    backend=Mock();outer=NativeDiagnosticMission('low_x_025',state,backend)
    for i in range(2049):
        state['sim_us']=35000000+i*10000
        rates,collective=outer.command(state)
        np.testing.assert_array_equal(rates,[.01,.02,.03]);assert collective==.6
    assert len(backend.calls)==205 and all(h==.12 for p,h in backend.calls)
    np.testing.assert_allclose(backend.calls[0][0],[1,2,-4.7])
    return dict(passed=True,continuous_position_velocity=True,analytic_velocity_checked=True,return_to_origin=True,step_schedule_exact=True,goal_10hz_count=205,native_rate_and_collective_unchanged=True,EKF_translation_only=True)

if __name__=='__main__':
    import json
    print(json.dumps(tests(),indent=2))
