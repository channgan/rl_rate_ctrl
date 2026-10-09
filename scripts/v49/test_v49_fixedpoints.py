from pathlib import Path
import argparse,sys,json,copy,hashlib,textwrap
import torch,numpy as np
from admit_bias_projection import same
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--v48',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);args=ap.parse_args()
V=args.v48;T=V/'backtracking_revision3/training';O=args.root/'offline/fixedpoints';O.mkdir(exist_ok=False)
sys.path.insert(0,str(args.root/'training/runtime'))
import v46_offline_scheduler as core
from v46_core import RPActor,read_capture
from v49_scheduler import V49Scheduler
from v48_configuration import configure_optimizer
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
j=lambda p:json.loads(p.read_text());LIMIT=1e-4-1e-10
source=Path(__file__).with_name('admit_bias_projection.py').read_text();setup=source[source.index("          p=j(T/'protocol.json')"):source.index('          v=core.flat(actor).clone()')]
start=setup.index('          class S(');end=setup.index('          s.initial_opt=',start)
setup=setup[:start]+"          s=V49Scheduler(actor,data,core.digest(T/'protocol.json'),gh,gm,A,am)\n"+setup[end:]
ref=j(args.reference);report=dict(passed=False,native_calls=0,formal_updates=0,disposable_optimizer_attempts=0,points={})
for name,b,cpp,pr,global_step in [('pre18',1,V/'memory_gate_revision2/training/training/batch_1/pre_update_recovery.pt',V/'memory_gate_revision2/training/protocol.json',17),('pre40',3,T/'training/batch_3/pre_update_recovery.pt',T/'protocol.json',39)]:
 item={};ctx=dict(globals(),**locals());exec(textwrap.dedent(setup),ctx);s=ctx['s'];initial=copy.deepcopy(s.snapshot());results=[]
 for replay in range(2):
  s.restore(copy.deepcopy(initial));result=s.step();report['disposable_optimizer_attempts']+=1
  expected=ref['points'][name]['bias_intersection'][0];assert result==expected['diagnostics']
  assert hashlib.sha256(core.binary_bytes(s.actor)[0]).hexdigest()==expected['actor_sha256']
  assert s.last_projection==dict(passed=True,**expected['projection']) and s.frozen_ok()
  results.append(copy.deepcopy(s.snapshot()))
 assert same(*results)
 report['points'][name]=dict(passed=True,diagnostics=result,exact_prior_offline_candidate=True,full_state_replay_exact=True)
report['passed']=True;(O/'admission.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(passed=True,points=list(report['points']),disposable_optimizer_attempts=4,native_calls=0)))
