from pathlib import Path
import sys,os,json,hashlib,types,importlib.util
R=Path(__file__).parent;E=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/native_roll_smooth_v44/evaluation_equivalence_continuation3')
sys.path.insert(0,str(E));sys.path.insert(0,str(R))
jr=Path(sys.argv[1]);pp=json.loads((jr/'protocol.json').read_text());case=pp['cases'][0]['name'];D=jr/pp['audit_profile']/case/('seed_'+str(pp['development_seed']));rp=D/'result.json';r=json.loads(rp.read_text());r.update(result_path=str(rp),label=pp['audit_label'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
event=Path('@DRL_ROOT_LINUX@/experiments/robustness_v1_20261002/native_entry_coverage_v46/event55_offline_revision1/goal_event_semantics_v6.py')
assert sha(event)=='1debced4df595b6fa659b45968cb53fdcce49a6de1ca899195cc7921afef0f36'
spec=importlib.util.spec_from_file_location('goal_event_semantics',event);mod=importlib.util.module_from_spec(spec);sys.modules['goal_event_semantics']=mod;spec.loader.exec_module(mod)
code=(E/'check_stochastic_episode.py').read_text();needle="model=next(j for j in json.loads((R/'config.json').read_text())['jobs'] if (j['model_id']==int(n['m'][0,4])))";assert code.count(needle)==1;code=code.replace(needle,'model='+repr(dict(model_id=pp['head_model_id'],initial_actor=pp['actor_npz'])))
assert code.count("('nn_capture.bin',len(n),65536)")==1;code=code.replace("('nn_capture.bin',len(n),65536)","('nn_capture.bin',len(n),131072)")
mod=types.ModuleType('check_stochastic_episode');sys.modules['check_stochastic_episode']=mod;exec(compile(code,str(E/'check_stochastic_episode.py')+'[frozen dev actor/capacity]','exec'),mod.__dict__)
code=(E/'audit_rate_reference_revised2.py').read_text();assert code.count("integrity['capacity']==65536")==1;code=code.replace("integrity['capacity']==65536","integrity['capacity']==131072")
mod=types.ModuleType('audit_rate_reference_revised2');sys.modules['audit_rate_reference_revised2']=mod;exec(compile(code,str(E/'audit_rate_reference_revised2.py')+'[capacity]','exec'),mod.__dict__)
code=(E/'audit_collect.py').read_text();assert code.count("integ['capacity']==65536")==1;code=code.replace("integ['capacity']==65536","integ['capacity']==131072")
mod=types.ModuleType('development_capture_audit');exec(compile(code,str(E/'audit_collect.py')+'[capacity only; original missions unchanged]','exec'),mod.__dict__)
report=mod.check(r);result=dict(passed=True,result_sha256=sha(rp),original_missions_unchanged=True,full_capture_report=report)
(D/'combined_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
