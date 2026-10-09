from pathlib import Path
import json,hashlib,datetime,ast
V=Path('@DRL_ROOT_WINDOWS@/experiments/robustness_v1_20261002/v48_fixed_lr3e5');PRE=V/'smooth_start_revision1';R=V/'memory_gate_revision2';T=R/'training';D=R/'development';X=R/'execution';P=Path('@DRL_ROOT_WINDOWS@/rl_rate_ctrl_refactor')
j=lambda p:json.loads(p.read_text());local=lambda p:Path(str(p).replace('/mnt/e/','E:/'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();put=lambda p,v:p.write_text(json.dumps(v,indent=2));linux=lambda p:str(p).replace('E:/','/mnt/e/').replace('E:\\','/mnt/e/').replace('\\','/');win=lambda p:str(p).replace('/','\\')
a=j(R/'offline/admission.json');assert a['passed'] and a['formal_updates']==0 and a['disposable_optimizer_steps']==2
lf=V/'training/ledger.json';l=j(lf);assert l==j(R/'original_ledger_stopped.json') and l['stopped'] and l['charged_native']==608641 and l['formal_optimizer_updates']==10 and not l['reservations']
for root in [V,PRE]:
 for name,h in j(root/'frozen_inputs.json').items():assert sha(local(name))==h
for name,h in j(T/'carried_inputs.json')['sha256'].items():assert sha(local(name))==h
p=j(T/'protocol.json');p.update(historical_charge=608641,maximum_new_captures=14,additional_native_cap=658000)
p['reward']['smoothness']['gradient_balance_gate']='Every update: global nonincreasing alpha targets0.099 across all28 case-phase-RP gradients; rebuild full training reward/returns/group baselines/common normalization; remeasure original0.1 hard gate. No activation step.'
put(T/'protocol.json',p)
contract=dict(approved=True,version='memory_gate_revision2',effective_rules=linux(T/'effective_rules.json'),effective_rules_sha256=sha(T/'effective_rules.json'),historical_activation_steps_removed=True,strict_LR_single_variable=False,physical_acceptance_unchanged=True,source_actor_update=10,source_actor_model_id=3433787511,carried_captures=14,reuse_batch1_only_for_updates11_to20=True,never_repeat_updates1_to10=True,remaining_native_captures=14,total_captures=28,total_updates=40,only_candidate='update40',training_limit=2000000,development_limit=3000000,charge_preserved=608641,max_training_charge_with_remaining_planned_caps=1266641,no_extra_qualification=True,no_automatic_retry=True,original_failures=[linux(V/'training/failure.json'),linux(PRE/'training/failure.json')],canonical_training_ledger=linux(lf),canonical_development_ledger=linux(V/'development/ledger.json'))
put(R/'revision_contract.json',contract)
for f in [*T.glob('runtime/*.py'),*D.glob('runtime/*.py')]:ast.parse(f.read_text())
assert 'from_global_update' not in (T/'runtime/update_batch.py').read_text() and 'for batch in range(1,4)' in (T/'runtime/batch_driver.py').read_text()
files=[T/'protocol.json',T/'effective_rules.json',T/'runtime_replacement.json',T/'carried_inputs.json',T/'development_design.json',D/'protocol_frozen.json',D/'frozen_design.json',D/'amendment.json',R/'revision_contract.json',R/'inherited_rule_audit.json',R/'offline/admission.json']+list(T.glob('runtime/*.py'))+list(D.glob('runtime/*.py'))+list(T.glob('templates/*.json'))+list(D.glob('templates/*.json'))+list((T/'training/batch_0').glob('*'))
files=[f for f in files if f.is_file()];put(R/'frozen_inputs.json',{linux(f):sha(f) for f in files})
put(X/'windows_config.json',dict(owner_script=linux(T/'runtime/batch_driver.py'),linux_root=linux(T),observer_job='pending_gateway'))
ad=j(PRE/'execution/admission.json');ad.update(root=win(X),work_root=win(T),runtime=win(T/'runtime'),canonical_ledger=win(lf),sha256={win(f):sha(f) for f in files},offline_validation=win(R/'offline/admission.json'));put(X/'admission.json',ad)
text='''# V48 memory_gate_revision2 — 最终有效规则与恢复合同

此前把 V47 第11步引入平滑修订、第21步引入记忆统计修订的历史时点，当成了算法阶段。为复制历史顺序而保留这些条件，未正确落实“沿用最终规则”，导致两次重复撞旧门槛。本版移除这些时间条件，独立 effective_rules.json 是有效规则来源；旧合同和失败完整保留。

所有批次：历史记忆 q05/q95 仅诊断，记录 model/seed/min/max/lower-upper gap/original_gate_would_fail，不能用跨策略累计范围宣称当前策略覆盖。记忆有限性、[-1,1]边界、递推重建、128动作历史和10ms选取仍为硬检查；速度覆盖、物理、同步、优化及验收门槛不变。每次更新均应用全局只降不升平滑alpha，目标0.099、复测原0.1硬门槛，重建回报/基线/共同归一化，无第11/21步启用条件。

保留的真正阶段：轨迹首3秒/后段的原权重；4批×7条、每批10次、2000加权样本；跨批保留actor/Adam/RNG，针对新行为数据重建采样排列；最终40唯一候选、完整54项开发后判定。来源校验中的batch>=2只表示前两批有获授权的精确历史数据，不是奖励或记忆门槛阶段。scheduler40防御界限不改变driver每批10次。

现有测试已实际完成：9项全批次记忆诊断/速度/有限性/边界回归；14条来源互不重复，第二批全为模型3433787511；step10 actor与checkpoint逐位一致，Adam步数10与moments/RNG准确恢复；复制日志精确1..10且不重复执行。两次一次性第11步离线重放，actor/Adam/minibatch一致，KL3.869595920326284e-6，所有原守卫通过，正式更新与native调用均0。旧checkpoint和账本未动。

从准确step10接11..20，复用第二批七条；先前14条全部计入原28，不重新采样或更新1..10。608641费用保留，剩余14条×47000=658000，计划最坏总计1266641<2m；开发独立3m、原54项/480301与480302、holdout12701/12702封存。A仍为当前模型，不自动晋级。

此版本已不等价于严格LR单变量实验；完整物理对比及原验收规则仍有效，但不得将效果全部因果归给学习率。所有旧合同/失败保留在原目录；新运行在memory_gate_revision2，训练与开发继续使用原V48两本账本。测试通过后已获授权自主恢复，无额外资格采样或自动重试。
'''
(P/'docs/V48_MEMORY_GATE_REVISION2_20261009.md').write_text(text,encoding='utf-8')
note='\n\n2026-10-09 authorized V48 memory_gate_revision2: historical memory q05/q95 diagnostic for ALL batches, not an update21 algorithm stage. Keep min/max/gaps/model/seeds/original_gate_would_fail; velocity, memory finite/bounds/reconstruction, physical/sync/optimization and acceptance unchanged. Global adaptive smooth every update remains effective, no historical11/21 switches. Independent effective_rules.json plus inherited_rule_audit.json distinguish true phases. Restore exact step10 actor3433787511/Adam/RNG, reuse only its7 batch1 captures for11..20; previous14 paid captures counted in28,608641 preserved,14future captures only.9 rule tests and2 disposable recovery steps passed, excluded from formal counts. Original two failures retained. Not strict LR-only; final40-only54 physical evaluation remains unchanged. See docs/V48_MEMORY_GATE_REVISION2_20261009.md.\n'
for f in [P/'TRAINING_PARAMETERS.md',P/'docs/TRAINING_STARTUP_RUNBOOK.md']:
 with f.open('a',encoding='utf-8') as stream:stream.write(note)
l.setdefault('runtime_replacements',[]).append(dict(id=p['sync_runtime_id'],approved=True,historical_charge=608641,historical_optimizer_updates=10,historical_attempt_count=14,original_stop_reason=l.get('stop_reason'),reason='Memory historical quantiles diagnostic all batches; same native runtime',contract=linux(R/'revision_contract.json')));l['stopped']=False;l['stop_reason']='Explicit memory_gate_revision2 continuation; original failures retained';tmp=lf.with_suffix('.tmp');put(tmp,l);tmp.replace(lf)
put(V/'active_revision.json',dict(revision='memory_gate_revision2',root=str(R),status=str(T/'status.json'),canonical_training_ledger=str(lf),canonical_development_ledger=str(V/'development/ledger.json'),launch_pending=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print(json.dumps(dict(ready=True,charge=l['charged_native'],updates=l['formal_optimizer_updates'],files_frozen=len(files),no_native_calls=True)))
