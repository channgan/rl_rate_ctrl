from original_missions import mission,CASES
def extend_runner(t):
    needle="from rate_rl.native_outer import NativePX4Reference"
    assert needle in t
    t=t.replace(needle,needle+"\nfrom low_speed_missions import NativeDiagnosticMission,mission")
    assert "'hover_hold','fixed_steps']" in t
    t=t.replace("'hover_hold','fixed_steps']","'hover_hold','fixed_steps','low_speed_diagnostic']")
    needle="    if mode=='fixed_steps':"
    assert t.count(needle)==1
    t=t.replace(needle,"    if mode=='low_speed_diagnostic':\n        outer=NativeDiagnosticMission(a.case,state,backend);target,_=outer.command(state)\n    elif mode=='fixed_steps':")
    t=t.replace("rows.append(dict(gyro_noise_flu=","rows.append(dict(goal_update_hz=getattr(outer,'goal_hz',None),goal_publications=getattr(outer,'publications',None),linear_velocity_ned=state['linear_velocity_ned'],mission_reference_position=mission(a.case,(state['sim_us']-start_us)*1e-6)[0].tolist() if mode=='low_speed_diagnostic' else None,mission_reference_velocity=mission(a.case,(state['sim_us']-start_us)*1e-6)[1].tolist() if mode=='low_speed_diagnostic' else None,mission_phase=mission(a.case,(state['sim_us']-start_us)*1e-6)[2] if mode=='low_speed_diagnostic' else None,gyro_noise_flu=")
    t=t.replace("obs=np.asarray(audit['obs'],dtype='float32');assert obs.shape==(18,);","obs=np.asarray(audit['obs']+audit.get('memory_rp',[]),dtype='float32');assert obs.shape==((20,) if 'memory_rp' in audit else (18,));")
    return t
def verify_mission(rows,name):
    import numpy as np
    if name not in CASES:return
    for i,x in enumerate(rows):
        assert x['sim_us']==35000000+(i+1)*10000
        ideal,velocity,phase=mission(name,(i+1)*.01)
        np.testing.assert_allclose(x['mission_reference_position'],ideal,rtol=0,atol=1e-12)
        np.testing.assert_allclose(x['mission_reference_velocity'],velocity,rtol=0,atol=1e-12)
        assert x['mission_phase']==phase and x['goal_update_hz']==100 and x['goal_publications']==i+1
        np.testing.assert_allclose(x['position_target'],mission(name,i*.01)[0],rtol=0,atol=1e-12)
