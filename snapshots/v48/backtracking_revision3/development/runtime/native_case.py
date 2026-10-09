from runtime_support import stage_emit
from local_guards import guard_next_native,guard_finite
import sys
sys.path.insert(0,'@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/native_entry_coverage_v46/protocol_revision2')
import argparse,ctypes,json,re,sys,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,'@DRL_ROOT_LINUX@/rl_rate_ctrl_refactor')
sys.path.insert(0,'@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/native_roll_smooth_v44/evaluation_equivalence_continuation3');from repaired_backend import GazeboPX4Backend, SimulatorError
from rate_rl.native_outer import NativePX4Reference
from low_speed_missions import NativeDiagnosticMission,mission
from rate_rl.outer_loop import OuterLoopConfig
from v46_training_entry import Prefix, EntryMission
from runtime_support import require_authorized, observer, export_before_close, register_case_watchdog, owned_spawn, record_child_exits
p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--case',required=True);p.add_argument('--controller',choices=['pid','nn_cpp'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--profile',choices=['baseline','candidate'],required=True);a=p.parse_args()
protocol=json.loads(Path(a.protocol).read_text());spec=next(c for c in protocol['cases'] if c['name']==a.case)
require_authorized(protocol)
out=Path(a.protocol).parent/a.profile/a.case/f'seed_{a.seed}';out.mkdir(parents=True,exist_ok=False)
parameters=protocol['pid']['parameters'];mode=spec['mode'];rate_mode=False;assert mode in ['waypoints','hover_native','hover_hold','fixed_steps','low_speed_diagnostic']
class Backend(GazeboPX4Backend):
    def _spawn(self,args,env,name):
        env=env.copy();env['RATE_RL_CLOSED_LOOP_SYNC']='1';env['RATE_RL_SYNC_TRACE']=str(out/'closed_loop_sync.bin');env['RATE_RL_SYNC_DIAGNOSTIC']=str(out/'sync_failure.json');env['RATE_RL_SYNC_STOP_PATH']=str(Path(os.environ['SUPERVISED_JOB_ROOT'])/'stop');env['PX4_CADENCE_TRACE_PATH']=str(out/'cadence.bin');env['PX4_MECHANISM_CAPTURE_PATH']=str(out/'mechanism.bin');env['PX4_DIAG_SECOND_PATH']=protocol['second_path'];env['PX4_DIAG_POST_V39']=str(protocol['post_v39']);env['PX4_RATE_REFERENCE_CAPTURE_PATH']=str(out/'rate_reference.bin');env.pop('PX4_RL_INDEPENDENT_NN',None);env['PX4_NN_CAPTURE_PATH']=str(out/'nn_capture.bin');env['RATE_RL_PHYSICS_CAPTURE']=str(out/'physics_capture.bin');env['PX4_GYRO_AUDIT_PATH']=str(out/'gyro_audit.bin');env['RATE_RL_PITCH_BIAS_FRD']=str(protocol['injection_offset']);env['PX4_RL_WORLD_ELEVATION_MSL']='0'
        if a.profile!='baseline':env['PX4_RL_INDEPENDENT_NN']='1';env['PX4_NN_WEIGHTS']=a.profile;env['PX4_NN_HEAD_PATH']=protocol['head_path'];env['PX4_NN_FULL_PATH']=protocol['full_path'];env['PX4_NN_EXPLORATION_STD']=str(protocol['exploration_std']);env['PX4_NN_EXPLORATION_SEED']=str(a.seed);env['PX4_NN_MAX_CYCLES']=str(protocol['native_episode_tick_cap'])
        if rate_mode:env['PX4_RL_RATE_TEST']='1'
        if name=='px4':
            startup=self.episode_dir/'startup.sh'
            commands=''.join(f'param set {k} {v}\n' for k,v in parameters.items())
            commands+='param set THR_MDL_FAC 0\n'
            commands+='logger stop\nlogger start -f -t -r 1000 -b 2048 -m file -p independent_nn_status\n'
            for i in range(1,5):commands+=f'param set SIM_GZ_EC_MIN{i} 150\nparam set SIM_GZ_EC_MAX{i} 1000\n'
            commands+='param show MPC_YAWRAUTO_MAX\nparam show MPC_YAWRAUTO_ACC\nparam show MC_*\nparam show IMU_*\nparam show CA_*\nparam show THR_MDL_FAC\nparam show SIM_GZ_EC*\necho ready > rl_startup.ready'
            startup.write_text(startup.read_text().replace('echo ready > rl_startup.ready',commands))
        owned_spawn(self,args,env,name)
    def _check_processes(self):
        exits=record_child_exits(self)
        if exits:raise SimulatorError("Child process exited: "+json.dumps(exits))
    def close(self):
        export_before_close(self,out)
        return super().close()
    def reset(self,seed=None):
        return self._reset_once(seed) # preserve failed bootstrap, no automatic retries
    def _request(self,line):
        before=getattr(self,'last_observed',None)
        if line.startswith('STEP ') and (Path(os.environ['SUPERVISED_JOB_ROOT'])/'stop').exists():raise RuntimeError('diagnostic graceful stop requested')
        if line.startswith('STEP ') and before and before.get('nn_audit'):
            native_used=before['nn_audit']['nn_cycles']+before['nn_audit']['pid_cycles']
            guard_next_native(int(native_used),int(line.split()[2]),protocol['v46_native_cap'])
        data=super()._request(line)
        if line.startswith('GOAL '):
            ledger=getattr(self,'goal_requests',[])
            ledger.append(dict(command=line,values=[float(x) for x in line.split()[1:]],boundary_us=None if before is None else before.get('sim_us'),before_attitude=None if before is None else before.get('px4_attitude_target'),ack=data))
            self.goal_requests=ledger
            (out/'goal_requests.json').write_text(json.dumps(ledger))
        if 'sim_us' in data:
            guard_finite(data['rates_true']);self.last_observed=data.copy()
        audit=data.get('nn_audit')
        if audit:
            if audit['fault']:raise RuntimeError('Independent NN latched safety fault: '+json.dumps(audit))
            if a.profile!='baseline':
                assert audit['nn_selected'] and audit['pid_cycles']==0,'PID executed in independent NN run'
            else:assert not audit['nn_selected'] and audit['nn_cycles']==0
        observer(self,out,line,data)
        return data
    def rate(self,target,collective):
        raise AssertionError('Directrates override forbidden in nativeupper protocol')
        self.rate_sequence+=1
        command='RATE_TEST '+str(self.rate_sequence)+' '+' '.join(f'{v:.9g}' for v in [*target,collective])
        self._check_processes();self.stream.write((command+'\n').encode('ascii'));reply=json.loads(self.stream.readline(8192))
        assert reply=={'rate_test_accepted':self.rate_sequence},reply
    def advance(self,action):
        if self.native_torque:return self.step_torque(action,.01)
        seq=self.sequence+1;state=self._request(f'STEP {seq} 10 0 0 0 0')
        assert state['seq']==seq and state['sim_us']==self.last_sim_us+10000
        assert not state['px4_policy_owns_torque'] and not state['px4_policy_owns_motors']
        self.sequence=seq;self.last_sim_us=state['sim_us'];return state
backend=Backend(spec['runtime'],out/'episodes',instance=protocol['instances'][a.controller]);backend.mechanism_output=str(out);backend.native_outer=True;backend.native_baseline=a.controller=='pid';backend.native_torque=a.controller=='nn_cpp';backend.rate_sequence=0
import os
os.environ['PX4_NN_HEAD_PATH']=protocol['head_path'];os.environ['PX4_NN_FULL_PATH']=protocol['full_path']
lib=ctypes.CDLL(protocol['actor_libraries'][a.profile]);actor=lib.rate_actor_timed;ptr=ctypes.POINTER(ctypes.c_float);actor.argtypes=[ptr,ctypes.c_float,ptr];actor.restype=ctypes.c_bool
previous=np.zeros(3,dtype='float32')
def action_for(state,target):
    if a.controller=='pid':return np.zeros(3,dtype='float32')
    gyro=np.asarray(state['rates']);obs=np.concatenate([gyro/5,(target-gyro)/5,previous]).astype('float32');action=np.empty(3,dtype='float32');assert actor(obs.ctypes.data_as(ptr),action.ctypes.data_as(ptr));return action
def reference(k):
    v=np.zeros(3)
    for axis,start in enumerate([200,700,1200]):
        for offset,amplitude in [(0,.5),(200,1.)]:
            t=k-start-offset
            if 0<=t<25:v[axis]=amplitude
            elif 25<=t<75:v[axis]=-amplitude
            elif 75<=t<100:v[axis]=amplitude
    return v
register_case_watchdog()
rows=[];alignment=[];bootstrap_initial=None
try:
    rng=np.random.default_rng(a.seed);simseed=int(rng.integers(0,2**31-1));stage_emit(Path(os.environ['SUPERVISED_JOB_ROOT']).parent,'warmup_begin');state=backend.reset(simseed)
    stage_emit(Path(os.environ['SUPERVISED_JOB_ROOT']).parent,'init_complete',sim_us=state['sim_us'],native_counts=state.get('nn_audit'),first_inference_host_time_known=False)
    bootstrap_initial=state.copy();alignment=[];alignment_target_us=35000000
    assert state['sim_us']<alignment_target_us,'Bootstrap exceeded prospective fixed start; preserve failure, no retry'
    entry_profile=protocol.get('v46_training_entry_profile');prefix=None
    if entry_profile:assert state['sim_us']<30000000,'Bootstrap missed frozen conditioning start; no retry'
    while state['sim_us']<alignment_target_us:
        boundary=30000000 if entry_profile and state['sim_us']<30000000 else alignment_target_us
        if entry_profile and state['sim_us']>=30000000:
            if prefix is None:prefix=Prefix(entry_profile,state,backend,out)
            prefix.command(state)
        ticks=min(10,(boundary-state['sim_us'])//1000)
        assert ticks>=1 and state['sim_us']%1000==0
        seq=backend.sequence+1;previous_us=state['sim_us']
        state=backend._request(f'STEP {seq} {ticks} 0 0 0 0')
        assert state['seq']==seq and state['sim_us']==previous_us+ticks*1000
        assert not state['px4_policy_owns_torque'] and not state['px4_policy_owns_motors']
        backend.sequence=seq;backend.last_sim_us=state['sim_us'];alignment.append(state.copy())
        assert state['px4_position_control'] and -state['position_ned'][2]>.3 and max(abs(v) for v in state['rates_true'])<12,'Alignment physical/mode failure'
    assert state['sim_us']==alignment_target_us
    if prefix is not None:prefix.save()
    assert protocol['continuous_score_steps']==2048 and protocol['v46_native_cap']==(56000 if a.profile=='baseline' else 47000)
    assert state['nn_audit']['nn_cycles']+state['nn_audit']['pid_cycles']+20480<=protocol['v46_native_cap'], 'startup budget cannot support full2048; no shortened success'
    stage_emit(Path(os.environ['SUPERVISED_JOB_ROOT']).parent,'scoring_ready',sim_us=state['sim_us'],steps=2048)
    initial=state.copy()
    (out/'alignment.json').write_text(json.dumps(dict(bootstrap_initial=bootstrap_initial,states=alignment),default=lambda x:np.asarray(x).tolist()))
    raw_params=(backend.episode_dir/'px4.log').read_text(errors='replace')
    actual_params={k:float(v) for k,v in re.findall(r'\b([A-Z][A-Z0-9_]+)\s+\[[^\]]+\]\s*:\s*(-?[\d.]+)',raw_params)}
    expected=dict(parameters,THR_MDL_FAC=0,MPC_YAWRAUTO_MAX=60,MPC_YAWRAUTO_ACC=20)
    for i in range(1,5):expected.update({f'SIM_GZ_EC_MIN{i}':150,f'SIM_GZ_EC_MAX{i}':1000})
    for k,v in expected.items():assert k in actual_params and abs(actual_params[k]-v)<1e-4,(k,v,actual_params.get(k))
    (out/'verified_parameters.json').write_text(json.dumps(actual_params,indent=2))
    initial_z=float(state['position_ned'][2]);initial_position=np.asarray(state['position_ned']).copy();hover=backend.px4_hover_command;minimum=150/backend.config['max_rotor_rad_s'];warm=[]
    def collective(state,k=-1):
        q=np.asarray(state['q_ned_frd']);vertical=max(.5,1-2*(q[1]**2+q[2]**2))
        acc=np.clip(2*(state['position_ned'][2]-initial_z)+2*state['linear_velocity_ned'][2],-2.,2.)
        force_ratio=(1+acc/backend.config['gravity_m_s2'])/vertical
        if mode=='collective_switch' and k>=200:force_ratio*=.9 if (k//200)%2 else 1.1
        return float(np.clip(((hover*(1-minimum)+minimum)*np.sqrt(force_ratio)-minimum)/(1-minimum),.4,.95))
    if entry_profile:
        outer=EntryMission(a.case,entry_profile,state,backend,out);target,_=outer.command(state)
    elif mode=='low_speed_diagnostic':
        outer=NativeDiagnosticMission(a.case,state,backend);target,_=outer.command(state)
    elif mode=='fixed_steps':
        class FixedNativePositionSteps:
            # Fixed world-frame mission. Truth only maps mission coordinates to EKF frame,
            # never enters candidate policy inputs or synthesizes rates/thrust.
            def __init__(self,state):
                self.start=state['sim_us'];self.offset=np.asarray(state['px4_local_position'])-np.asarray(state['position_ned'])
                qw,qx,qy,qz=state['q_ned_frd'];truth_yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
                self.heading=np.pi/2+float(state['px4_heading'])-truth_yaw
                self.times=[0.,3.,8.,13.,18.];self.goals=np.array([[0.,0.,-5.],[.5,0.,-5.],[0.,.5,-5.],[-.5,0.,-5.],[0.,0.,-5.]])
                self.last_index=-1;self.position_target_ned=self.goals[0].copy();self.waypoints_reached=0
            def command(self,state,*unused):
                t=(state['sim_us']-self.start)*1e-6;index=sum(t>=v for v in self.times)-1
                if index!=self.last_index:
                    self.position_target_ned=self.goals[index].copy();backend.set_position_goal(self.position_target_ned+self.offset,self.heading);self.last_index=index
                assert state['px4_position_control']
                return np.asarray(state['px4_rate_target']),-state['px4_thrust_body_z']
        outer=FixedNativePositionSteps(state);target,_=outer.command(state)
    elif mode=='hover_hold':
        class HoldExistingNativeGoal:
            # No new GOAL or rates/thrust command: retain native takeoff hover task.
            position_target_ned=np.array([0.,0.,-5.]);waypoints_reached=0
            def command(self,state,*unused):
                assert state['px4_position_control']
                return np.asarray(state['px4_rate_target']),-state['px4_thrust_body_z']
        outer=HoldExistingNativeGoal();target,_=outer.command(state)
    elif rate_mode:
        for k in range(100):
            target=np.zeros(3);thrust=collective(state);last_packet=(target.copy(),thrust,state['sim_us']);backend.rate(target,thrust);action=action_for(state,target);state=backend.advance(action);previous=action.copy()
            warm.append(dict(sim_us=state['sim_us'],rate_mode=state['px4_rate_test_mode']))
            if state['px4_rate_test_mode'] and k>=49:break
        assert state['px4_rate_test_mode'],'Body-rate mode not acknowledged'
        queue=[last_packet] if spec.get('delay_steps',0) else []
        outer=None
    else:
        outer=NativePX4Reference(OuterLoopConfig(gravity_m_s2=backend.config['gravity_m_s2']),backend)
        if mode=='hover_native':
            def hold_initial(position):
                outer.position_target_ned=outer.origin.copy();outer.heading=float(state['px4_heading'])
                backend.set_position_goal(outer.position_target_ned+outer.local_offset,outer.heading)
            outer._next_waypoint=hold_initial;outer._update_arrival_gate=lambda state:False
        outer.reset(state,rng);target,_=outer.command(state,4.,(1.,1.,1.))
    failure=None;start_us=state['sim_us']
    for k in range(protocol['continuous_score_steps']):
        sample_us=state['sim_us']
        if rate_mode:
            raw_target=reference(k);raw_collective=collective(state,k);queue.append((raw_target.copy(),raw_collective,sample_us));target,thrust,source_us=queue.pop(0)
            assert sample_us-source_us==spec.get('delay_steps',0)*10000
            backend.rate(target,thrust);goal=initial_position
        else:
            raw_target=target.copy();source_us=sample_us;raw_collective=-state['px4_thrust_body_z'];goal=outer.position_target_ned.copy()
        action=action_for(state,target);state=backend.advance(action);previous=action.copy()
        assert state['sim_us']==start_us+(k+1)*10000
        assert not state.get('px4_rate_test_active',False),'Unexpected directrate mode'
        required=bool(state['px4_rate_test_mode']) if rate_mode else bool(state['px4_position_control'])
        if not required:failure='control_mode_lost'
        if rate_mode and required:
            assert 0<=state['px4_now_us']-state['px4_reference_us']<=50000
            np.testing.assert_allclose(state['px4_rate_target'],target,atol=1e-7)
            np.testing.assert_allclose(state['px4_thrust_body_z'],-thrust,atol=1e-6)
        audit=state['nn_audit'];assert state['sim_us']-audit['sample_us']<=20000
        if a.profile!='baseline':
            assert audit['model_id']==protocol['head_model_id'],'Wrong neuralhead in FC'
            assert audit['history_valid'],'10mslookback lacks bracketing real command samples'
            assert audit['sample_us']-audit['previous_action_age_us']<=audit['sample_us']-10000<audit['history_next_us']<=audit['sample_us']
            assert .000125<=audit['inference_dt_s']<=.020001
            assert abs(audit['inference_dt_s']-audit['previous_native_age_us']*1e-6)<1e-8
            obs=np.asarray(audit['obs']+audit.get('memory_rp',[]),dtype='float32');assert obs.shape==((20,) if 'memory_rp' in audit else (18,));pred=np.empty(3,dtype='float32');assert actor(obs.ctypes.data_as(ptr),audit['inference_dt_s'],pred.ctypes.data_as(ptr))
            assert np.isfinite(audit['output']).all() and np.max(np.abs(audit['output']))<=1 # stochastic means/densities verified from complete native trace
        metric_target=np.asarray(state['px4_rate_target']);error=np.rad2deg(metric_target-np.asarray(state['rates_true']));delivered_error=error.copy();assert np.isfinite(error).all()
        raw_pwm=np.asarray(state['applied_pwm']);esc=(raw_pwm-minimum)/(1-minimum) if a.controller=='pid' else raw_pwm
        assert np.min(esc)>=-1e-6 and np.max(esc)<=1+1e-6;esc=np.clip(esc,0,1)
        rows.append(dict(goal_update_hz=getattr(outer,'goal_hz',None),goal_publications=getattr(outer,'publications',None),linear_velocity_ned=state['linear_velocity_ned'],mission_reference_position=(outer.base_world+mission(a.case,(state['sim_us']-start_us)*1e-6)[0]).tolist() if mode=='low_speed_diagnostic' else None,mission_reference_velocity=mission(a.case,(state['sim_us']-start_us)*1e-6)[1].tolist() if mode=='low_speed_diagnostic' else None,mission_phase=mission(a.case,(state['sim_us']-start_us)*1e-6)[2] if mode=='low_speed_diagnostic' else None,gyro_noise_flu=state.get('gyro_noise_flu'),metric_target=metric_target.tolist(),reference_us=state['px4_reference_us'],reference_age_us=state['px4_now_us']-state['px4_reference_us'],nn_audit=audit,step=k+1,sim_us=state['sim_us'],command_generated_us=source_us,command_age_us=sample_us-source_us,raw_target=raw_target.tolist(),delivered_target=target.tolist(),actual_px4_target=state['px4_rate_target'],true_rates=state['rates_true'],filtered_rates=state['rates'],error_deg_s=error.tolist(),legacy_mislabeled_delivered_error_deg_s=delivered_error.tolist(),position_ned=state['position_ned'],position_target=np.asarray(goal).tolist(),quaternion=state['q_ned_frd'],collective=-state['px4_thrust_body_z'],raw_collective=raw_collective,normalized_esc=esc.tolist(),backend_raw_pwm=raw_pwm.tolist(),waypoints=outer.waypoints_reached if outer else None,nn_torque=action.tolist() if a.controller=='nn_cpp' else None))
        if -state['position_ned'][2]<.3:failure='ground'
        elif np.max(np.abs(state['rates_true']))>12:failure='rate'
        if failure:break
        if outer:target,_=outer.command(state,4.,(1.,1.,1.))
    error=np.array([r['error_deg_s'] for r in rows]);esc=np.array([r['normalized_esc'] for r in rows]);heights=np.array([r['position_ned'][2]-r['position_target'][2] for r in rows]);positions=np.array([np.asarray(r['position_ned'])-r['position_target'] for r in rows])
    result=dict(candidate_id=protocol['candidate_id'],profile=a.profile,nn_final_audit=state["nn_audit"],controller=("locked_pid" if a.profile=="baseline" else "independent_nn"),native_transport=a.controller,case=a.case,seed=a.seed,simulator_seed=simseed,steps=len(rows),physical_failure=failure,survived_2048=len(rows)>=2048 and failure is None,survived_target=len(rows)==protocol['continuous_score_steps'] and failure is None,axis_rmse=np.sqrt(np.mean(error**2,axis=0)).tolist(),axis_bias=np.mean(error,axis=0).tolist(),axis_p95_abs=np.percentile(np.abs(error),95,axis=0).tolist(),axis_p99_abs=np.percentile(np.abs(error),99,axis=0).tolist(),axis_peak_abs=np.max(np.abs(error),axis=0).tolist(),axis_rmse_after3s=np.sqrt(np.mean(error[300:]**2,axis=0)).tolist() if len(rows)>300 else None,legacy_mislabeled_delivered_axis_rmse=np.sqrt(np.mean(np.array([r['legacy_mislabeled_delivered_error_deg_s'] for r in rows])**2,axis=0)).tolist(),height_rmse_m=float(np.sqrt(np.mean(heights**2))),height_p95_abs_m=float(np.percentile(np.abs(heights),95)),height_min_m=min(-r['position_ned'][2] for r in rows),position_error_p95_m=float(np.percentile(np.linalg.norm(positions,axis=1),95)),motor_upper_saturation_fraction=float(np.mean(esc>=.9999)),motor_lower_saturation_fraction=float(np.mean(esc<=.0001)),motor_variation_100hz=float(np.mean(np.abs(np.diff(esc,axis=0)))) if len(rows)>1 else None,waypoints=outer.waypoints_reached if outer else None,initial_state=initial,mode_transition=warm,log_dir=str(backend.episode_dir))
    if entry_profile or mode=='low_speed_diagnostic':outer.save()
    (out/'trajectory.json').write_text(json.dumps(rows,default=lambda x:np.asarray(x).tolist()));(out/'result.json').write_text(json.dumps(result,indent=2,default=lambda x:np.asarray(x).tolist()));print(json.dumps({k:v for k,v in result.items() if k not in ['initial_state','mode_transition']}),flush=True)
except Exception:
    (out/'alignment_failure.json').write_text(json.dumps(dict(bootstrap_initial=bootstrap_initial,states=alignment),default=lambda x:np.asarray(x).tolist()));(out/'failure.txt').write_text(traceback.format_exc());(out/'partial_trajectory.json').write_text(json.dumps(rows));raise
finally:backend.close()
