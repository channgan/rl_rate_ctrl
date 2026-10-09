"""Offline-testable v46 protocol primitives. No simulator launch or policy deployment."""
import copy, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn

NATIVE_CAP=1000000
SETTLEMENT_GAIN=.0007
FAILURE_RATE_RAW=4000.
GAMMA_TIME_CONSTANT=30.
RATE_SCALE_FIRST=np.array([.5,.5,.5])
RATE_SCALE_LATE=np.array([.2,.2,.5])
RATE_WEIGHT_FIRST=np.array([.7,.7,.2])
RATE_WEIGHT_LATE=np.array([.5,.5,.2])
LEGACY_SLEW=2./70.
EXTRA_TORQUE_SLEW=.05
EXTRA_MOTOR_SLEW=.05
SATURATION=.2
MAX_NEGATIVE_RATE=1.6+LEGACY_SLEW+EXTRA_TORQUE_SLEW+EXTRA_MOTOR_SLEW+SATURATION

def bounded(x):
 x=np.asarray(x,dtype=np.float64);return x*x/(1+x*x)
def rotation(yaw):
 c,s=math.cos(yaw),math.sin(yaw);return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
def integrated_startup(s):
 if s<=0:return 0.
 if s>=1:return s-.5
 return 2.5*s**4-3*s**5+s**6
def history_bump(s):
 # C2-continuous displacement pulse on seconds1..4, zero thereafter.
 u=(s-1.)/3.
 return 64*u**3*(1-u)**3 if 0<u<1 else 0.
def prefix_goal(anchor,yaw,elapsed,profile):
 assert 0<=elapsed<=5.
 return np.asarray(anchor,float)+np.asarray(profile['velocity_goal_ned_m_s'])*integrated_startup(elapsed)+rotation(yaw)@np.asarray(profile['history_pulse_body_m'])*history_bump(elapsed)
def entry_goal(old_goal,yaw,profile):
 return np.asarray(old_goal,float)+rotation(yaw)@np.asarray(profile['entry_jump_body_m'])
def leg(s,speed):
 if s<=0:return 0.
 if s<1:return .5*speed*(s-math.sin(math.pi*s)/math.pi)
 if s<16:return speed*(s-.5)
 if s<17:
  u=s-16;return speed*15.5+.5*speed*(u+math.sin(math.pi*u)/math.pi)
 return 16*speed
def score_goal(base,elapsed,case):
 p=np.asarray(base,float).copy()
 if case=='low_x_050_goal100':p[0]+=leg(elapsed-3,.5)
 elif case in ['low_y_025_goal100','lateral_plus_goal100']:p[1]+=leg(elapsed-3,.25)
 else:assert case in ['hover_transition','noise200_hover_hold']
 return p

class RPActor(nn.Module):
 def __init__(self,npz):
  super().__init__();z=np.load(npz)
  for key in ['w0','b0','w1','b1','gate_at_1ms']:self.register_buffer(key,torch.from_numpy(np.array(z[key],np.float32,copy=True)))
  self.rp_w=nn.Parameter(torch.from_numpy(z['w2'][:2].copy()));self.rp_b=nn.Parameter(torch.from_numpy(z['b2'][:2].copy()))
  self.register_buffer('yaw_w',torch.from_numpy(z['w2'][2:3].copy()));self.register_buffer('yaw_b',torch.from_numpy(z['b2'][2:3].copy()))
 def features(self,x):
  h=torch.relu(nn.functional.linear(torch.cat([x[...,:15],x[...,18:20]],-1),self.w0,self.b0));return torch.relu(nn.functional.linear(h,self.w1,self.b1))
 def forward(self,x,dt):
  h=self.features(x);rp=nn.functional.linear(h,self.rp_w,self.rp_b);yaw=nn.functional.linear(h,self.yaw_w,self.yaw_b).squeeze(-1);g=-torch.expm1(torch.log1p(-self.gate_at_1ms)*(dt/.001));return torch.clamp(torch.cat([rp,(g*yaw+(1-g)*x[...,17])[...,None]],-1),-1,1)
 def canonical(self):
  z={k:getattr(self,k).detach().cpu().numpy().copy() for k in ['w0','b0','w1','b1','gate_at_1ms']};z.update(w2=torch.cat([self.rp_w,self.yaw_w]).detach().cpu().numpy().copy(),b2=torch.cat([self.rp_b,self.yaw_b]).detach().cpu().numpy().copy());return z

def project_halfspaces(delta,gradients,passes=64):
 d=delta.clone()
 for _ in range(passes):
  for g in gradients:
   denom=torch.dot(g,g)
   if denom>1e-24:d-=torch.clamp(torch.dot(g,d),min=0)/denom*g
 return d

class BudgetLedger:
 """Every attempt reserves a cap before launch; no naming-based reset or retry."""
 def __init__(self):self.charged=0;self.actual_known=0;self.reservations={};self.closed={}
 def reserve(self,key,cap):
  if key in self.reservations or key in self.closed:raise ValueError('Attempt identifier already used; no retry')
  if cap<=0 or self.charged+sum(self.reservations.values())+cap>NATIVE_CAP:raise ValueError('Unified native budget exceeded')
  self.reservations[key]=cap
 def close(self,key,actual=None):
  cap=self.reservations[key]
  if actual is not None and not 0<=actual<=cap:raise ValueError('Runtime cap violated; stop entire version; reservation retained')
  self.reservations.pop(key)
  charge=cap if actual is None else actual;self.charged+=charge;self.actual_known+=0 if actual is None else actual;self.closed[key]=dict(actual=actual,charge=charge)

def interval_weights(t,lo,hi):return np.maximum(0,np.minimum(t[1:],hi)-np.maximum(t[:-1],lo))*1e-6
def integrate_arrays(t,reference,truth,u,previous_u,inference_dt,physics_t,esc,esc_before,*,physical_failure=None,original_rmse=None):
 """Native right-end consumed-reference quadrature; costs credited to preceding actions."""
 t=np.asarray(t,np.int64);dt=np.diff(t)*1e-6;start=int(t[0]);end=int(t[-1]);n=len(dt);assert n>0 and np.all(dt>0) and dt.max()<=.004
 assert np.asarray(reference).shape==np.asarray(truth).shape==(n+1,3)
 e=np.rad2deg(reference[1:]-truth[1:]);first=interval_weights(t,start,start+3000000);late=dt-first;cruise=interval_weights(t,start+4000000,start+19000000)
 first_axis=bounded(e/RATE_SCALE_FIRST)*RATE_WEIGHT_FIRST*first[:,None];late_axis=bounded(e/RATE_SCALE_LATE)*RATE_WEIGHT_LATE*late[:,None]
 tracking=(first_axis+late_axis).sum(1)
 udot=(u[:-1]-previous_u[:-1])/inference_dt[:-1,None];bu=bounded(udot/10.)
 torque=dt*(LEGACY_SLEW*bu.mean(1)+EXTRA_TORQUE_SLEW*bu[:,:2].mean(1))
 pt=np.asarray(physics_t,np.int64);assert len(pt)==(end-start)//1000 and np.array_equal(pt,np.arange(start+1000,end+1,1000));assert np.asarray(esc).shape==(len(pt),4)
 assert np.min(esc)>=-1e-6 and np.max(esc)<=1+1e-6
 dm=(esc-np.vstack([esc_before,esc[:-1]]))/.001;motor_physics=EXTRA_MOTOR_SLEW*bounded(dm/10.).mean(1)*.001;owners=np.searchsorted(t,pt,side='left')-1;assert owners.min()==0 and owners.max()==n-1
 motor=np.bincount(owners,weights=motor_physics,minlength=n);sat=SATURATION*np.mean(abs(u[:-1])>=.999,axis=1)*dt
 bonus=.084*np.all(abs(e)<5.,axis=1)*(np.max(abs(u[:-1]),axis=1)<.999)*dt
 native_rmse=np.sqrt(np.sum(e*e*dt[:,None],axis=0)/dt.sum());elapsed=(end-start)*1e-6;remaining=max(0.,20.48-elapsed)
 success=physical_failure is None and abs(elapsed-20.48)<1e-9 and np.all(native_rmse<5.) and (original_rmse is None or np.all(np.asarray(original_rmse)<5.))
 terminal=35. if success else -(30000.+4000.*remaining)*SETTLEMENT_GAIN
 reward=bonus-tracking-torque-motor-sat;reward[-1]+=terminal
 constraints=np.column_stack([first_axis[:,0],first_axis[:,1],bounded(e[:,0]/.2)*.5*cruise,bounded(e[:,1]/.2)*.5*cruise,(first_axis+late_axis)[:,0],(first_axis+late_axis)[:,1],(first_axis+late_axis)[:,2]])
 assert np.all(tracking+torque+motor+sat<=MAX_NEGATIVE_RATE*dt+1e-12)
 return dict(reward=reward,transition_dt=dt,constraint_costs=constraints,tracking=tracking,torque_smooth=torque,motor_smooth=motor,saturation=sat,bonus=bonus,terminal=terminal,native_rmse=native_rmse,success=bool(success),native_error=e,first_duration=float(first.sum()),cruise_duration=float(cruise.sum()))

def read_capture(result_path):
 D=Path(result_path).parent;r=json.loads(Path(result_path).read_text());nr=np.frombuffer((D/'nn_capture.bin').read_bytes()[8:],np.dtype([('m','<u8',(6,)),('v','<f4',(44,))]));start=int(r['initial_state']['sim_us']);end=int(r['nn_final_audit']['sample_us']);mask=(nr['m'][:,0]>=start)&(nr['m'][:,0]<=end);ix=np.flatnonzero(mask);assert ix[0]>0;nn=nr[ix];t=nn['m'][:,0].astype(np.int64);v=nn['v'].astype(float);assert t[0]==start and t[-1]==end
 rr=np.frombuffer((D/'rate_reference.bin').read_bytes()[8:],np.dtype([('m','<u8',(11,)),('v','<f4',(10,))]));ri=np.searchsorted(rr['m'][:,0],t);assert np.array_equal(rr['m'][ri,0],t);rv=rr['v'][ri].astype(float);assert np.array_equal(rv[:,6:9],v[:,21:24]);assert np.array_equal(v[:,21:24],np.clip(v[:,27:30],-1,1));assert np.max(abs(rv[:,:3]-5*(v[:,1:4]+v[:,4:7])))<2e-6
 p=np.frombuffer((D/'physics_capture.bin').read_bytes()[8:],'<f8').reshape(-1,25);pi=np.searchsorted(p[:,2],t);assert np.array_equal(p[pi,2],t);truth=p[pi,22:25];pm=(p[:,2]>start)&(p[:,2]<=end);j=int(np.searchsorted(p[:,2],start));assert p[j,2]==start;esc=(p[pm,4:8]-.15)/.85;esc_before=(p[j,4:8]-.15)/.85
 previous=nr['v'][ix-1,21:24].astype(float);assert np.array_equal(previous,v[:,16:19])
 q=integrate_arrays(t,rv[:,:3],truth,v[:,21:24],previous,v[:,0],p[pm,2].astype(np.int64),esc,esc_before,physical_failure=r['physical_failure'],original_rmse=r['axis_rmse'])
 mean_previous=nr['v'][ix-1,24:27].astype(float);mean_cost=np.sum(np.diff(t)*1e-6*(LEGACY_SLEW*bounded((v[:-1,24:27]-mean_previous[:-1])/v[:-1,0,None]/10).mean(1)+EXTRA_TORQUE_SLEW*bounded((v[:-1,24:26]-mean_previous[:-1,:2])/v[:-1,0,None]/10).mean(1)))
 q.update(sample_us=t,obs=v[:,1:21].astype(np.float32),dt=v[:,0].astype(np.float32),mean=v[:,24:27].astype(np.float32),actual_u=v[:,21:24],raw=v[:,27:30],previous_obs=nr['v'][ix-1,1:21].copy(),previous_dt=nr['v'][ix-1,0].copy(),rho=v[:,41],innovation_std=v[:,42],previous_raw=v[:,35:38],previous_mean=v[:,38:41],old_conditional_mean=v[:,32:35],old_logp=v[:,31],mean_only_torque_smooth_diagnostic=float(mean_cost),result=r)
 return q

def discounted_returns(reward,dt):
 out=np.zeros_like(reward,dtype=float);carry=0.
 for i in range(len(dt)-1,-1,-1):carry=reward[i]+math.exp(-dt[i]/GAMMA_TIME_CONSTANT)*carry;out[i]=carry
 return out
def leave_one_out_baselines(times,returns):
 assert len(times)>=2;return [np.mean([np.interp(t,times[j],returns[j]) for j in range(len(times)) if j!=i],axis=0) for i,t in enumerate(times)]
