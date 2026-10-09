from pathlib import Path
import os,sys,json,copy,importlib.util
import pytest,numpy as np,torch
V=Path(os.environ['V49_TEST_ROOT']);T=V/'training';D=V/'development';sys.path.insert(0,str(T/'runtime'))
from v48_configuration import validate_batch,configure_optimizer
from effective_rules import enforce_coverage
from local_guards import guard_next_native
from adaptive_smooth import choose_alpha
j=lambda p:json.loads(p.read_text())

def test_frozen_scope_and_budget():
 p=j(T/'protocol.json');d=j(D/'protocol_frozen.json');r=j(T/'effective_rules.json')
 assert len(p['jobs'])==28 and len({x['seed'] for x in p['jobs']})==28
 assert sum(x['cap'] for x in p['jobs'])==p['native_limit']==1316000
 assert len(d['jobs'])==54 and sum(x['cap'] for x in d['jobs'])==d['native_limit']==2700000
 assert len({(x['case'],x['seed'],x['label']) for x in d['jobs']})==54
 assert {x['seed'] for x in d['jobs']}=={490301,490302}
 assert not {12701,12702}&{x['seed'] for x in p['jobs']+d['jobs']}
 assert r['smoothness']['mode']=='global_nonincreasing_adaptive_every_update'
 assert r['memory_coverage']['mode']=='diagnostic_all_batches'
 assert not {'resume_global_update','resume_step17'}&p.keys()
 assert p['initial_actor']['model_id']==2428576135 and p['learning_rate']==3e-5
 assert r['backtracking']['scales']==[2.**-i for i in range(13)]

@pytest.mark.parametrize('batch',range(4))
def test_all_batches_reject_historical_captures(batch,tmp_path):
 m=dict(results=['/historical/'+str(i) for i in range(7)],actor=dict(model_id=1))
 p=dict(jobs=[dict(id=str(i),batch=batch) for i in range(7)])
 with pytest.raises(AssertionError,match='Historical'):validate_batch(tmp_path,batch,m,p)

@pytest.mark.parametrize('batch',range(4))
def test_memory_quantiles_diagnostic_but_velocity_hard(batch):
 entries=[dict(memory=[0,0],velocity=[0,0,0])]*7
 cov=[dict(field='memory',minimum=[0,0],maximum=[0,0],required=dict(q05=[-1,-1],q95=[1,1]),passed=False),dict(field='velocity',passed=True)]
 assert enforce_coverage(cov,entries,batch,1,list(range(7)))['original_gate_would_fail']
 cov[1]['passed']=False
 with pytest.raises(AssertionError):enforce_coverage(cov,entries,batch,1,list(range(7)))

def test_budget_exact_last_tick_then_stop():
 for cap in [47000,56000,1316000,2700000]:
  guard_next_native(cap-1,1,cap)
  with pytest.raises(RuntimeError,match='budget'):guard_next_native(cap,1,cap)

def test_fresh_adam_and_nonincreasing_alpha():
 v=torch.nn.Parameter(torch.ones(2));opt=torch.optim.Adam([v],lr=3e-6);state=configure_optimizer(opt,None,3e-5)
 assert state['state']=={} and state['param_groups'][0]['lr']==3e-5
 records=[dict(stratum=0,axis=0,tracking_norm=1.,smooth_norm=2.)]
 a,_=choose_alpha(records,1.);b,_=choose_alpha(records,a);assert a==b==.0495
 records[0]['tracking_norm']=0;assert choose_alpha(records,a)[0]==0
 records[0]['tracking_norm']=float('nan')
 with pytest.raises(ValueError):choose_alpha(records,a)

@pytest.mark.parametrize('area,limit',[('training',1316000),('development',2700000)])
def test_runtime_budget_checks(area,limit,tmp_path,monkeypatch):
 root=tmp_path/area;job=root/'job';job.mkdir(parents=True)
 protocol=dict(v46_job_id='fixed',v46_native_cap=47000,native_episode_tick_cap=47000,full_path=str(job/'actor.bin'))
 (job/'actor.bin').write_bytes(b'unit-fixture-only')
 import hashlib
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 protocol['actor_sha256']=sha(job/'actor.bin');(job/'protocol.json').write_text(json.dumps(protocol))
 grant=dict(authorized=True,native_limit=limit);ledger=dict(native_limit=limit,stopped=False,reservations={'fixed':dict(cap=47000,protocol_sha256=sha(job/'protocol.json'))})
 (root/'protocol.json').write_text(json.dumps(grant));(root/'ledger.json').write_text(json.dumps(ledger))
 monkeypatch.setenv('BATCH_TRAINING_ROOT',str(root));monkeypatch.setenv('SUPERVISED_JOB_ROOT',str(job))
 spec=importlib.util.spec_from_file_location('runtime_budget_'+area,V/area/'runtime/runtime_support.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.require_authorized(protocol)
 ledger['native_limit']+=1;(root/'ledger.json').write_text(json.dumps(ledger))
 with pytest.raises(AssertionError):module.require_authorized(protocol)
