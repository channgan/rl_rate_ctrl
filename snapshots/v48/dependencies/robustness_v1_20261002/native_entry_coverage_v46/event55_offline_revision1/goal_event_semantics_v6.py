from pathlib import Path
import json,numpy as np

def validate_mechanism_clocks(mm):
 # Sensor acquisition precedes diagnostic copy. GOAL publication precedes its
 # successful copy, bounded above by the post-copy hrt latch. Sample is NOT
 # the GOAL consumption clock; wall_ns assembly fields are another timebase.
 for name,bad in [('sample_after_latch',mm[:,0]>mm[:,1]),('goal_after_latch',(mm[:,6]!=0)&(mm[:,2]>mm[:,1]))]:
  ids=np.flatnonzero(bad)
  if len(ids):
   k=int(ids[0]);raise AssertionError(dict(rule=name,index=k,sample_us=int(mm[k,0]),latch_us=int(mm[k,1]),goal_us=int(mm[k,2])))

def validate_goal_witness(mm,vv,k,stamp,b,payload):
 detail=dict(rule='native_GOAL_post_copy_latch',index=int(k),command_us=int(b),sample_us=int(mm[k,0]),latch_us=int(mm[k,1]),goal_us=int(mm[k,2]),expected_goal_us=int(stamp),payload=vv[k,:4].tolist(),expected_payload=payload.tolist())
 assert int(mm[k,0])<=int(mm[k,1]) and b<=stamp<=int(mm[k,1]) and int(mm[k,2])==stamp and mm[k,6]!=0 and vv[k,26]==1 and np.array_equal(vv[k,:4],payload),detail

def causal_heading_index(t,h,publication,output,precommand_witnesses):
 strict=np.flatnonzero(t<publication);assert len(strict);idx=int(strict[-1]);interval=np.flatnonzero((t>=publication)&(t<=output))
 for j in interval:
  if h[j]==h[idx]:continue
  # A position message predates a heading change: only a uniquely matched already-published attitude can prove consumer-before-command.
  assert int(t[j]) in precommand_witnesses and precommand_witnesses[int(t[j])]==output,'Goal changed in unobserved consumer interval'
 return idx

SOURCE_CLOCK_HASHES = {'@LINUX_HOME@/rl_rate/PX4-head-switch-v42/src/modules/simulation/gz_bridge/RLBridge.cpp': '1732f624ce1b61e067db06b4ea549400c0e960e9a4fb0f18f7f1f922481d51ee', '@LINUX_HOME@/rl_rate/PX4-head-switch-v42/src/modules/mc_pos_control/GotoControl/GotoControl.cpp': 'd2c95bc74c3ab718f4509196999f369d60df65f3a1cc320fcd784d88858f2a85', '@LINUX_HOME@/rl_rate/PX4-head-switch-v42/src/modules/mc_pos_control/MulticopterPositionControl.cpp': '15da8b46899f2ca28194ebdab535a98d7d70e5ba1cdaa0c490a077ff81c02f78', '@LINUX_HOME@/rl_rate/PX4-head-switch-v42/platforms/common/uORB/uORBDeviceNode.cpp': 'f383fdd0f4fac33d529cfead6eadff0146e679797ddc348e4cae5a7edace2a59', '@LINUX_HOME@/rl_rate/PX4-head-switch-v42/platforms/common/uORB/uORBManager.cpp': '01c0149fefcbf47c0a9ca3665a02f36f52836df313f2e5e4454e6e2f5dc90877', '@LINUX_HOME@/rl_rate/PX4-head-switch-v42/platforms/common/uORB/Publication.hpp': 'ddd7a343d6514ba43de1170f53ad137e24536e33306085eee2192f2fd89b90da', '@LINUX_HOME@/rl_rate/PX4-head-switch-v42/platforms/common/uORB/Subscription.hpp': '0d350258d6f3d2833b91183a36b3ea4efe0d318e077b79e13efcfb613bbf6d04', '@LINUX_HOME@/rl_rate/PX4-head-switch-v42/build/px4_sitl_default/px4_boardconfig.h': '7c224ac8943a8d9afd9cc82717feff64f6d2d50f7876d73080e42ea2a15b1b90'}
_source_checked=False
def verify_repeat_source():
 global _source_checked
 if not _source_checked:
  import hashlib
  for p,h in SOURCE_CLOCK_HASHES.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,('repeat source changed',p)
  _source_checked=True

def equivalent_ack_repeat(go,mm,vv,b,stamp,payload,publication,output,context):
 # This proves identical INPUT STATES for either post-ACK publication order.
 # It never compares candidate outputs. No ULog event or timestamp is inserted.
 detail=dict(rule='post_ACK_identical_publication_class',command_us=b,heartbeat_us=stamp,position_publication_us=int(publication),consumer_output_us=int(output),payload=payload.tolist())
 assert context is not None,detail
 c=context['command'];ack=c['ack']
 assert c['boundary_us']==b and ack['sim_us']==ack['px4_now_us']==b and ack['px4_position_control'] and not ack['px4_rate_test_active'],detail
 assert b<int(publication)<=int(output)<context['next_boundary'] and stamp<context['next_boundary'],detail
 # Every captured event in this interval must carry the complete same command.
 gp=np.column_stack([go[f'position[{i}]'] for i in range(3)]+[go['heading']]);ids=np.flatnonzero((go['timestamp']>=b)&(go['timestamp']<=max(stamp,int(output))))
 assert len(ids) and np.all(gp[ids]==payload),dict(detail,reason='different full target in consumer interval')
 flags={'flag_control_heading':1,'flag_set_max_heading_rate':0,'flag_set_max_horizontal_speed':0,'flag_set_max_vertical_speed':0,'max_heading_rate':0.,'max_horizontal_speed':0.,'max_vertical_speed':0.}
 for key,value in flags.items():assert key in go and np.all(go[key][ids]==value),dict(detail,reason='different control flag/limit',field=key)
 # A pre-command independent publication proves the fixed typed publisher was initialized.
 prior=np.flatnonzero(go['timestamp']<b);assert len(prior),detail;j=int(prior[-1]);prior_stamp=int(go['timestamp'][j]);native=np.flatnonzero((mm[:,2]==prior_stamp)&(mm[:,6]!=0)&np.all(vv[:,:4]==gp[j],axis=1));assert len(native),dict(detail,reason='no initialized publisher witness')
 # Existing heading input selection must also be generation invariant; this
 # avoids synthesizing an unrecorded event to repair a changing-heading case.
 assert gp[j,3]==payload[3],dict(detail,reason='changing heading requires independent ordered event')
 sample=int(context['position_samples'][int(output)]);assert sample<min(b,stamp)+500000 and prior_stamp>0 and b-prior_stamp<500000,dict(detail,reason='timeout class differs')
 verify_repeat_source()
 return dict(detail,source_verified=True,ACK_after_synchronous_publication=True,prior_ULog_index=j,prior_GOAL_us=prior_stamp,prior_native_index=int(native[0]),position_sample_us=sample,full_control_fields=flags,identical_input_class=True,no_generation_identified=True,no_synthetic_event=True)

def corroborated_heartbeat(go,mm,vv,b,payload,next_boundary,publications,outputs,precommand_output,context=None):
 # Choose the earliest actual publication matching both independent record streams, never an output-fit candidate.
 gp=np.column_stack([go[f'position[{i}]'] for i in range(3)]+[go['heading']])
 possible=np.flatnonzero((go['timestamp']>b)&(go['timestamp']<=b+10000)&(go['timestamp']<next_boundary)&np.all(gp==payload,axis=1))
 found=[]
 for j in possible:
  stamp=int(go['timestamp'][j]);ids=np.flatnonzero((mm[:,2]==stamp)&np.all(vv[:,:4]==payload,axis=1)&(mm[:,6]!=0)&(vv[:,26]==1))
  if len(ids):found.append((stamp,int(ids[0])))
 assert found,dict(rule='no_independent_heartbeat',command_us=int(b),next_boundary=int(next_boundary),payload=payload.tolist(),ulog_candidate_indices=possible.tolist())
 stamp,k=min(found)
 # A consumer could execute anywhere between its position publication and output; reject overlap unless the existing pre-command publication uniquely proves completion.
 overlap=(np.asarray(outputs)>=b)&(np.asarray(publications)<=stamp)
 for oi in np.flatnonzero(overlap):
  output=int(np.asarray(outputs)[oi])
  if precommand_output is not None and output==precommand_output and output<=b:continue
  if int(np.asarray(publications)[oi])==output==b and context is not None and context.get('all_requests'):
   eq=earlier_ack_stable_input(go,mm,vv,b,stamp,payload,b,output,context)
  else:
   eq=equivalent_ack_repeat(go,mm,vv,b,stamp,payload,int(np.asarray(publications)[oi]),output,context)
  context.setdefault('equivalent_consumers',[]).append(eq)
 validate_goal_witness(mm,vv,k,stamp,b,payload)
 return stamp,k

def resolve_episode_heading(D,groups,output_times,position_samples):
 D=Path(D);go=groups['goto_setpoint',0];lp=groups['vehicle_local_position',0];att=groups['vehicle_attitude_setpoint',0]
 assert np.all(go['flag_control_heading']) and not np.any(go['flag_set_max_heading_rate'])
 requests=json.loads((D/'goal_requests.json').read_text()) if (D/'goal_requests.json').exists() else []
 raw=(D/'mechanism.bin').read_bytes();assert raw[:8]==b'MECH0042';a=np.frombuffer(raw[8:],np.dtype([('m','<u8',(12,)),('v','<f4',(28,))]));mm=a['m'];vv=a['v'];validate_mechanism_clocks(mm);aq=np.column_stack([att[f'q_d[{i}]'] for i in range(4)])
 additions=[];witnesses={};proof=[]
 lpt0=lp['timestamp'].astype(np.int64);li0=np.searchsorted(lpt0,output_times,side='right')-1;assert np.all(li0>=0)
 publications=lpt0[li0];assert np.array_equal(lp['timestamp_sample'][li0],position_samples)
 boundaries=[int(c['boundary_us']) for c in requests if c['boundary_us'] is not None]
 assert all(x<=y for x,y in zip(boundaries,boundaries[1:]))
 for c in requests:
  b=c['boundary_us']
  if b is None or b<35000000 or b>=int(max(output_times)):continue
  assert c['ack']['sim_us']==b and c['ack']['px4_now_us']==b
  payload=np.array(c['values'],np.float32);match=(mm[:,2]==b)&np.all(vv[:,:4]==payload,axis=1)&(mm[:,6]!=0);ids=np.flatnonzero(match)
  before=np.asarray(c['before_attitude']);hit=np.flatnonzero(abs(aq-before).max(1)<1e-9) if before.shape==(4,) else np.array([],int)
  precommand_output=int(att['timestamp'][hit[0]]) if len(hit)==1 else None
  if not len(ids):
   next_boundary=min([x for x in boundaries if x>b]+[int(max(output_times))+1])
   context=dict(command=c,all_requests=requests,next_boundary=next_boundary,position_samples={int(t):int(p) for t,p in zip(output_times,position_samples)})
   stamp,k=corroborated_heartbeat(go,mm,vv,b,payload,next_boundary,publications,output_times,precommand_output,context)
   proof.append(dict(command_us=b,native_goal_timestamp=stamp,native_first_observation_sample=int(mm[k,0]),native_first_latch=int(mm[k,1]),native_record_index=k,payload_f32=payload.tolist(),sensor_sample_is_not_command_clock=True,payload_exact=True,missing_ULog_publication_added=False,immediate_publication_generation_not_captured=True,existing_heartbeat_used=True,no_unobserved_heading_consumer=not bool(context.get('equivalent_consumers')),equivalent_consumer_publications=context.get('equivalent_consumers',[]),precommand_output=precommand_output))
   continue
  k=int(ids[0]);validate_goal_witness(mm,vv,k,b,b,payload)
  existing=(go['timestamp']==b)&np.all(np.column_stack([go[f'position[{i}]'] for i in range(3)]+[go['heading']])==payload,axis=1);new_index=int(np.searchsorted(go['timestamp'],b,side='right')-1);assert new_index>=0
  if not np.any(existing):additions.append((b,payload,new_index))
  before=np.asarray(c['before_attitude']);hit=np.flatnonzero(abs(aq-before).max(1)<1e-9) if before.shape==(4,) else np.array([],int)
  if len(hit)==1:witnesses[b]=int(att['timestamp'][hit[0]])
  proof.append(dict(command_us=b,native_goal_timestamp=int(mm[k,2]),native_first_observation_sample=int(mm[k,0]),native_first_latch=int(mm[k,1]),native_record_index=k,payload_f32=payload.tolist(),sensor_sample_is_not_command_clock=True,payload_exact=True,missing_ULog_publication_added=not bool(np.any(existing))))
 if additions:
  merged={k:list(v) for k,v in go.items()}
  for b,payload,j in additions:
   for key in merged:
    value=go[key][j]
    if key=='timestamp':value=b
    elif key=='heading':value=payload[3]
    elif key.startswith('position['):value=payload[int(key[9])]
    merged[key].append(value)
  order=np.argsort(np.array(merged['timestamp']),kind='stable');old={k:v.dtype for k,v in go.items()};go.clear();go.update({k:np.asarray(v,dtype=old[k])[order] for k,v in merged.items()})
 t=go['timestamp'].astype(np.int64);h=go['heading'];lpt=lp['timestamp'].astype(np.int64);chosen=[];boundaries=[]
 for stamp,sample in zip(output_times,position_samples):
  stamp=int(stamp);sample=int(sample);li=np.searchsorted(lpt,stamp,side='right')-1;assert li>=0 and int(lp['timestamp_sample'][li])==sample;publication=int(lpt[li]);idx=causal_heading_index(t,h,publication,stamp,witnesses);chosen.append(idx);boundaries.append([sample,publication,stamp,int(t[idx])])
 return np.array(chosen),dict(clock_semantics=dict(sensor_sample='gyro acquisition; integration and cadence only',mechanism_latch='hrt after GOAL copy; upper bound of native diagnostic consumption',goal_timestamp='hrt publication',position_publication='lower bound of heading consumer execution',trajectory_output='upper bound of heading consumer execution',position_sample='integration dt only',assembly_ns='separate steady wall clock; duration only'),method='ULog goal events augmented only by exact synchronousGOAL payload/timestamp corroborated independently in nativeMECH. Input selection uses position PUBLICATION before consumer, never sampletime or outputfit. Changes inside unobserved consumer interval reject unless exact pre-command attitude proves prior publication.',native_command_witnesses=proof,augmented_goal_events=len(additions),position_publication_boundaries=boundaries,limits='Only original command API position+heading; flags independently checked. Immediate generation may be absent in sampled captures: actual independently corroborated heartbeat accepted with no intervening consumer, or source-verified post-ACK full-control-field identity and invariant timeout/heading class, no synthetic publication inserted for this case. Generation remains unidentified; command effect/order supported, not complete publication history.')

def tests():
 t=np.array([10,20]);h=np.array([0.,1.]);assert causal_heading_index(t,h,21,22,{})==1
 try:causal_heading_index(t,h,19,22,{})
 except AssertionError:pass
 else:raise AssertionError('ambiguous consumer input accepted')
 assert causal_heading_index(t,h,19,22,{20:22})==0
 return dict(passed=True,publication_order_accepted=True,ambiguous_interval_rejected=True,precommand_published_attitude_witness=True)

def earlier_ack_stable_input(go,mm,vv,b,stamp,payload,publication,output,context):
 detail=dict(rule='earlier_ACK_complete_input_equivalence',command_us=b,heartbeat_us=stamp,position_publication_us=publication,consumer_output_us=output)
 assert context is not None and publication==output==b,detail
 req=context.get('all_requests',[]);current=context['command'];assert current in req,detail
 ci=req.index(current);assert ci>0,detail
 anchor=req[ci-1];ab=anchor['boundary_us'];assert ab is not None and ab==b-10000,detail
 for c in [anchor,current]:
  a=c['ack'];cb=c['boundary_us']
  assert a['sim_us']==a['px4_now_us']==cb and a['px4_position_control'] and not a['px4_rate_test_active'],detail
  assert np.array_equal(np.asarray(c['values'],np.float32),payload),detail
 assert ab<publication and b<stamp<context['next_boundary'],detail
 gp=np.column_stack([go[f'position[{i}]'] for i in range(3)]+[go['heading']])
 prior=np.flatnonzero((go['timestamp']>ab)&(go['timestamp']<publication));assert len(prior),detail
 j=int(prior[-1]);ps=int(go['timestamp'][j]);ids=np.flatnonzero((go['timestamp']>=ps)&(go['timestamp']<=stamp))
 assert len(ids)>=2 and np.all(gp[ids]==payload),detail
 flags={'flag_control_heading':1,'flag_set_max_heading_rate':0,'flag_set_max_horizontal_speed':0,'flag_set_max_vertical_speed':0,'max_heading_rate':0.,'max_horizontal_speed':0.,'max_vertical_speed':0.}
 for key,value in flags.items():assert key in go and np.all(go[key][ids]==value),dict(detail,field=key)
 native=np.flatnonzero((mm[:,2]==ps)&(mm[:,6]!=0)&(vv[:,26]==1)&np.all(vv[:,:4]==payload,axis=1)&(mm[:,1]<publication));assert len(native),detail
 k=int(native[0]);validate_goal_witness(mm,vv,k,ps,ab,payload)
 sample=int(context['position_samples'][output]);assert 0<ps<=sample<=publication and sample<ps+500000,detail
 verify_repeat_source()
 return dict(detail,anchor_ACK_us=ab,anchor_GOAL_us=ps,anchor_native_index=k,anchor_latch_us=int(mm[k,1]),position_sample_us=sample,full_control_fields=flags,source_verified=True,identical_input_class=True,no_generation_identified=True,no_synthetic_event=True,proof='Earlier synchronous acknowledged command and independently recorded publication strictly precede consumer. Adjacent repeated command and all observed publications have identical complete inputs; timestamp alternatives are initialized and fresh. Source update ignores generation and timestamp except initialization/timeout; no output fitting.')
