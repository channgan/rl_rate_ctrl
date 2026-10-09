from pathlib import Path
import json,hashlib,datetime
V=Path('@DRL_ROOT_WINDOWS@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');R=V/'smooth_start_revision1';T=R/'training';P=Path('@DRL_ROOT_WINDOWS@/rl_rate_ctrl_refactor');j=lambda p:json.loads(p.read_text());local=lambda p:Path(str(p).replace('/mnt/e/','E:/'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=j(R/'offline/admission.json');assert a['passed'] and j(R/'tests.json')['passed']
for path,h in j(R/'frozen_inputs.json').items():assert sha(local(path))==h
for path,h in j(V/'frozen_inputs.json').items():assert sha(local(path))==h
lf=V/'training/ledger.json';l=j(lf);old=j(R/'original_ledger_stopped.json');assert l==old and l['charged_native']==303638 and l['formal_optimizer_updates']==0 and l['stopped']
alpha=a['comparisons'][1]['alpha'];maxratio=max(x['ratio'] for x in a['comparisons'][1]['balance']['actual'])
text=f'''# V48 smooth_start_revision1 — explicit authorized revision

Original V48 stopped before its first optimizer step: all seven trajectories passed full2048/export/sync/velocity/memory checks, but lateral+ first3 roll smooth/track gradient ratio0.1408974 exceeded0.1 at alpha1. Original protocol, failure, checkpoint and charge303638 are preserved. This revision is **not strictly an LR-only comparison against the original V47 process**: adaptive smooth initialization now begins at update1 instead of11.

Global alpha=min(previous,0.099*min(track/smooth),1), never increases; zero tracking globally disables smooth; nonfinite gradients stop. Rebuild reward-to-go, grouped leave-one-group-out baselines and common normalization, then remeasure all28 case-phase-axis ratios under the original0.1 hard gate. Physical reward/4000-s failure, acceptance, native binaries/PID, phase weights and memory schedule are unchanged.

Offline paired first-step results share A, empty Adam, the exact7 behavior-A trajectories and the same weighted2000 minibatch. Both produce alpha **{alpha:.12f}**, maximum remeasured ratio **{maxratio:.12f}**. LR3e-6/3e-5 KL values2.363889e-7/2.363928e-5; action RP RMS3.143594e-6/3.143637e-5, ratio{a['action_ratio']:.8f}. All original guards pass without backtracking. Independent reward error7.11e-15; independent return/baseline/normalization error7.54e-14. Frozen parameters/physical costs/behavior distribution unchanged. Actor, optimizer and minibatch restore exactly. Four disposable offline steps, zero formal updates/native calls. Five provenance tests and four adaptive-rule boundary tests pass.

Resume reuses the paid seven trajectories from the current V48 A behavior, not new sampling. Fresh A/Adam at3e-5; remaining21 original captures, four batches total,10 updates each, final40 only. Training cap2m includes original303638; remaining21 reservations total987000, worst total1290638 without extra samples/retries. Separate54-case development cap3m, seeds480301/480302, sealed holdout12701/12702. First two batches retain hard memory/velocity coverage; memory diagnostic from21 remains unchanged. A remains current; no automatic promotion.

Canonical ledgers stay at original V48 training/ledger.json and development/ledger.json. This revision's new status/model/log files are under smooth_start_revision1/training and development. Original stopped status remains historical evidence. Accepted-step count is now journaled after every checkpoint/logged update to avoid underreporting a partial failed batch; optimizer math is unchanged.

Sources: revision_contract.json, frozen_inputs.json, offline/admission.json, tests.json, original_ledger_stopped.json. New unique admission: execution/admission.json. Existing holder is complete, no live old owned process. One independent holder and serial driver; original ownership/export/cleanup/budget checks retained. No change of old seed, no fee reset, no automatic retry.
'''
(P/'docs/V48_SMOOTH_START_REVISION1_20261009.md').write_text(text,encoding='utf-8')
note=f'\n\n2026-10-09 authorized V48 smooth_start_revision1: original alpha1 pre-update failure and303638 charge retained; adaptive global smooth rule starts at update1 (previously11), original0.1 hard gate unchanged. Offline paired first-step3e-6/3e-5 passed with shared alpha{alpha:.12f}, ratio{a["action_ratio"]:.8f}; four disposable steps excluded. Exact7 paid A trajectories reused, empty Adam3e-5,21remaining captures,40total updates/final40only54dev; original2m/3m budgets,4000/s,PID/sync/phase weights/memory schedule unchanged. Not strictly LR-only versus original V47 process. New status/runtime smooth_start_revision1; original canonical ledgers preserved. Every accepted formal update now journaled to ledger. See docs/V48_SMOOTH_START_REVISION1_20261009.md.\n'
with (P/'TRAINING_PARAMETERS.md').open('a',encoding='utf-8') as f:f.write(note)
with (P/'docs/TRAINING_STARTUP_RUNBOOK.md').open('a',encoding='utf-8') as f:f.write(note)
id=j(T/'protocol.json')['sync_runtime_id'];l.setdefault('runtime_replacements',[]).append(dict(id=id,approved=True,reason='smooth start objective timing revision only; native sync2 unchanged',historical_charge=303638,historical_optimizer_updates=0,historical_attempt_count=7,original_stop_reason=l.get('stop_reason'),original_failure=str(V/'training/failure.json'),revision_contract=str(R/'revision_contract.json')));l['stopped']=False;l['stop_reason']='explicit smooth_start_revision1 continuation authorized; original failure retained'
tmp=lf.with_suffix('.tmp');tmp.write_text(json.dumps(l,indent=2));tmp.replace(lf)
(V/'active_revision.json').write_text(json.dumps(dict(revision='smooth_start_revision1',root=str(R),status=str(T/'status.json'),canonical_training_ledger=str(lf),canonical_development_ledger=str(V/'development/ledger.json'),launch_pending=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
print(json.dumps(dict(ready=True,charge=l['charged_native'],formal_updates=l['formal_optimizer_updates'],remaining_captures=21,admission=str(R/'execution/admission.json'))))
