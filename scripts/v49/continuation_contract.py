"""Pure planning/admission functions. No subprocess, simulator or native launcher."""
from pathlib import Path
import argparse, hashlib, json, math

CASES=('noise200_hover_hold','low_x_050_goal100','low_y_025_goal100')
MONITOR_SEEDS=(520101,520102,520103)
FINAL_SEEDS=(530101,530102)
FINAL_HASH='c0183c13bcf588c940a88f2a10aff59bc9256e8eb4b91066331ac9e4efe01855'
LIMIT=8340000

def plan():
    jobs=[]
    def monitor(stage):
        for case in CASES:
            for seed in MONITOR_SEEDS:
                jobs.append(dict(id=f'monitor_{stage}_{case}_{seed}',kind='monitor',stage=stage,case=case,seed=seed,cap=47000))
    monitor(0)
    for b in range(12):
        for p in range(7):jobs.append(dict(id=f'train_{b}_{p}',kind='training',batch=b,profile=p,seed=610101+1000*b+p,cap=47000))
        if (b+1)%4==0:monitor((b+1)//4)
    for case in ('hover_hold','hover_transition','waypoints','combined_native_waypoints','noise200_hover_hold','low_x_050_goal100','low_y_025_goal100','lateral_plus_goal100','lateral_minus_goal100'):
        for seed in FINAL_SEEDS:
            for label in ('pid','A','candidate'):
                jobs.append(dict(id=f'final_{case}_{seed}_{label}',kind='final',case=case,seed=seed,label=label,cap=56000 if label=='pid' else 47000))
    assert len(jobs)==174 and sum(j['cap'] for j in jobs)==LIMIT
    return jobs

def validate_seed_partition(previous_seeds):
    proposed={j['seed'] for j in plan()}
    assert not proposed.intersection({12701,12702}|set(previous_seeds)), 'seed collision/sealed holdout'
    groups=[{j['seed'] for j in plan() if j['kind']==k} for k in ('training','monitor','final')]
    assert all(not x.intersection(y) for i,x in enumerate(groups) for y in groups[i+1:])

class Budget:
    def __init__(self):self.charged=0;self.reserved={};self.attempted=set();self.stopped=False
    def reserve(self,job):
        assert not self.stopped and job['id'] not in self.attempted, 'stopped or retry'
        canonical={j['id']:j for j in plan()};assert job==canonical[job['id']], 'unknown/changed job'
        assert self.charged+sum(self.reserved.values())+job['cap']<=LIMIT
        self.attempted.add(job['id']);self.reserved[job['id']]=job['cap']
    def settle(self,job_id,actual,failed=False):
        assert job_id in self.reserved and isinstance(actual,int) and actual>=0
        cap=self.reserved.pop(job_id);self.charged+=actual
        if failed or actual>cap or self.charged>LIMIT:self.stopped=True
        assert actual<=cap and self.charged<=LIMIT, 'overrun retained and stopped'

def new_batch_manifest(manifest,root,batch,expected_model_id):
    """Pure validation of future fresh captures; never accepts historical replay fixtures."""
    assert 0<=batch<12 and len(manifest)==7
    root=Path(root).resolve();paths=[]
    for p,row in enumerate(manifest):
        path=Path(row['result']).resolve();paths.append(path)
        assert path.is_relative_to(root/'jobs'/f'train_{batch}_{p}'), 'historical/wrong capture path'
        assert row['seed']==610101+1000*batch+p and row['model_id']==expected_model_id
        assert row['audit_passed'] and row['sync_passed'] and row['steps']==2048 and not row['physical_failure']
    assert len(set(paths))==7

def monitor_decision(base,previous,current):
    """Rows keyed by case,seed. J and each ref/axis phase metric are deg/s."""
    expected={(c,s) for c in CASES for s in MONITOR_SEEDS}
    maps=[{(r['case'],r['seed']):r for r in rows} for rows in (base,previous,current)]
    assert all(len(rows)==9 and set(m)==expected for rows,m in zip((base,previous,current),maps))
    for k in expected:
        r=maps[2][k];assert all(math.isfinite(v) and v>=0 for v in [r['J'],r['yaw'],r['motor'],*r['axes'].values()])
        if not r['safe']:return 'protection_failure'
        for ref in maps[:2]:
            a=ref[k]
            if r['yaw']>a['yaw']+max(.005,.05*a['yaw']) or r['motor']>1.05*a['motor']:return 'protection_failure'
    # Axis protection is per-case mean, both reference definitions and all fixed phases.
    keys={f'{ref}_{axis}_{phase}' for ref in ('native','requested') for axis in ('roll','pitch') for phase in ('full','first3','cruise')}
    assert all(set(r['axes'])==keys for rows in (base,previous,current) for r in rows)
    for c in CASES:
        for key in keys:
            value=sum(maps[2][c,s]['axes'][key] for s in MONITOR_SEEDS)/3
            for ref in maps[:2]:
                a=sum(ref[c,s]['axes'][key] for s in MONITOR_SEEDS)/3
                if value>a+max(.005,.02*a):return 'protection_failure'
    for ref in maps[:2]:
        a=sum(r['J'] for r in ref.values())/9;b=sum(r['J'] for r in maps[2].values())/9
        if a<.005 or b>.95*a:return 'insufficient_evidence'
        if any(sum(maps[2][c,s]['J']-ref[c,s]['J'] for c in CASES)>=0 for s in MONITOR_SEEDS):return 'insufficient_evidence'
    return 'improvement'

def stage_action(stage,decision,previous_insufficient):
    assert stage in (1,2,3) and decision in ('improvement','insufficient_evidence','protection_failure')
    count=0 if decision=='improvement' else previous_insufficient+1
    if decision=='protection_failure':return 'stop_diagnose_no_final',count
    return ('terminal_final_only' if stage==3 or count>=2 else 'continue'),count

def prepare(root,out):
    """Plan-only entry. Completion/launch permission is deliberately not manufactured."""
    raise RuntimeError('Superseded frequent-monitor plan: do not prepare or launch. Select a million-step unit and use block_plan for review only.')

def historical_prepare_disabled(root,out):
    """Archived implementation only, always blocked before filesystem access."""
    raise RuntimeError('Historical plan disabled')
    root=Path(root);out=Path(out);j=lambda p:json.loads(p.read_text())
    done=j(root/'training/training_complete.json');assert done['completed'] and done['updates']==40
    actor=Path(done['actor']['binary']);assert hashlib.sha256(actor.read_bytes()).hexdigest()==FINAL_HASH
    old=j(root/'training/ledger.json');assert old['formal_optimizer_updates']==40 and not old['reservations']
    previous=[a['seed'] for a in old['attempts']]
    previous += [a['seed'] for a in j(root/'development/ledger.json')['attempts']]
    validate_seed_partition(previous)
    report=dict(status='prepared_not_admitted',native_launch_enabled=False,requires_complete_v49_54=True,requires_launch_decision=True,requires_fresh_data_launcher_integration=True,start_global_update=40,max_global_update=160,initial_actor=done['actor'],monitor_cases=CASES,monitor_seeds=MONITOR_SEEDS,final_seeds=FINAL_SEEDS,budget_limit=LIMIT,jobs=plan())
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(report,f,indent=2)
    return report

def block_plan(unit,blocks=3):
    """Review-only arithmetic, no sampling. Unit must be explicitly selected."""
    assert unit in ('native_scoring_1ms','legacy_outer_10ms'), 'step unit must be selected'
    assert type(blocks) is int and 1<=blocks<=3
    batches=7 if unit=='native_scoring_1ms' else 70
    trajectories=batches*7;native=trajectories*20480;outer=trajectories*2048
    training_cap=trajectories*47000*blocks
    # Optional boundary-only monitor ceiling: one baseline + one after each block.
    monitor_cap=9*47000*(blocks+1)
    return dict(status='review_only_not_admitted',native_launch_enabled=False,unit=unit,blocks=blocks,batches_per_block=batches,trajectories_per_block=trajectories,native_scoring_per_block=native,legacy_outer_per_block=outer,optimizer_updates_per_block=batches*10,gradient_rows_per_block=batches*10*2000,score_seconds_per_block=trajectories*20.48,training_native_cap=training_cap,optional_boundary_monitor_cap=monitor_cap,one_final54_cap=2700000,total_ceiling=training_cap+monitor_cap+2700000,inside_block_high_cost_monitor_jobs=0)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--v49',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    r=prepare(a.v49,a.out);print(json.dumps({k:r[k] for k in ('status','native_launch_enabled','budget_limit')}))
