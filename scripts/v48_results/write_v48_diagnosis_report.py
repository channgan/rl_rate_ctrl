from pathlib import Path
import argparse,json,numpy as np
p=argparse.ArgumentParser();p.add_argument('--revision',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();R=a.revision;O=R/'offline_objective_diagnosis';q=json.loads((O/'credit_and_constraints.json').read_text());d=json.loads((O/'diagnosis.json').read_text())
assert q['native_calls']==q['optimizer_steps']==0 and len(q['episodes'])==28 and len(d['records'])==54
assert all(e['success'] and e['terminal']==35 for e in q['episodes']) and max(q['terminal_baseline_max_abs_residual_by_batch'])==0
# Independent finite-difference check of the analytical near-zero curvature.
for w,s,expected in [(.7,.5,5.6),(.5,.2,25.)]:
 h=1e-6;f=lambda e:w*(e/s)**2/(1+(e/s)**2);numeric=(f(h)-2*f(0)+f(-h))/h**2;assert abs(numeric-expected)<1e-7
for name,r in q['reference_decomposition'].items():assert r['max_algebra_residual']<1e-10 and r['max_truth_source_delta']<1e-7
checks=dict(passed=True,training_captures=28,development_trajectories=54,terminal_baseline_cancellation=True,curvature_finite_difference=True,reference_identity_v47_v48=True,native_calls=0,optimizer_steps=0)
(O/'verification.json').write_text(json.dumps(checks,indent=2))
text='''# V48 零 native 诊断与下一版最小方案

最终评测结果已提交并推送16596122c37b066a08b0748e5a5b771e19a7b99d，远端哈希一致。V48不晋升，A保留。本诊断只读取28条训练采集、54条开发轨迹及既有v47诊断；native调用0、optimizer步骤0，没有运行新试验。

## 可核验事实

目标的单轴连续跟踪代价为 w*(e/s)^2/(1+(e/s)^2)，e单位deg/s。R/P前3秒w=.7,s=.5，后段w=.5,s=.2。零点斜率为0是对称误差代价的正常极小值，并非死区；零点二阶导分别5.6、25，已用有限差分核验。R/P的曲率相同，没有pitch特有近零平坦区证据。|e|超过s/sqrt(3)后曲率变负，大误差梯度逐渐衰减；验收RMSE/SSE及P99不会同样饱和。因此训练目标与验收有真实口径差异，但不能凭此认定其是此次失败的唯一原因。

28条训练轨迹的等回合平均：pitch处于|e|<.02deg/s的时间比例，前3秒8.24%、后段9.02%；处于负曲率区的比例分别15.66%、50.63%，超过尺度s的比例4.50%、25.31%。这些是误差分布诊断，不能直接等同参数空间Hessian或策略梯度信噪比。

首3秒与后段采样质量各0.5。考虑每秒采样密度后，近零曲率的前/后有效比约1.305，而不是简单“前3秒权重太小”。实际30秒时间常数return-to-go中，入场时跟踪成本回报有81.82%–91.55%来自3秒以后。此为长时信用分配的事实，不是索引错误证据。奖励结算使用右端消费参考/真值，成本归前一个动作；优势用其它场景组的时间对齐基线，没有critic。全部28条训练采集成功、终奖35，其共同折扣曲线在分组基线中精确抵消，四批残差均0；失败4000/s在这些成功轨迹中没有提供区分梯度。不能因为存活就声称这40步学会了存活。

154参数确实发生更新：roll/pitch权重相对A变化0.10317%/0.07565%；bias变化+3.23946e-5/-9.99969e-5。同一批开发状态输入下，最终策略相对A输出RMS为3.84963e-5/1.07768e-4，最大绝对变化1.33598e-4/2.02314e-4，yaw严格0。pitch主要呈负向偏移；这不证明闭环误差需要该方向，因误差到动作响应还经过动力学和上层环。yaw闭环指标变化来自状态/参考轨迹变化，不是yaw参数学习。

40步有31步接受scale1。revision3覆盖18–40的61个完整提案中23个接受、38个拒绝：18次批内bias越界、20次全局A bias越界，计数为非排他分类；这些记录没有KL/cost/零步拒绝。最终pitch bias已用尽约99.9969%的1e-4预算，最后一步1/256。全程最大KL0.0001931远低于0.002，权重相对变化0.0010317远低于0.01，guard最大动作变化0.00020536远低于0.002。证据支持bias约束是已记录后半程的直接回溯瓶颈，不支持笼统说所有守卫都太紧；1–17未保存同等完整提案日志，不能把后半程计数外推到全部40步。

alpha从.7026389降至.2781231、第4步降至.1101477611，之后保持；最终约为原平滑系数的11.01%，因而平滑优化信号被明显缩小。原0.1梯度比例门槛始终保留。它可能影响电机改善，但没有做隔离反事实，不能据此归因。Adam每个接受提案推进一次，参数回溯比例不同比例缩放Adam状态；这是当前既定实现，不凭现有结果宣称其必然错误。

## 两种参考口径差异

对每一条相同轨迹，先在原100Hz时间戳重算native消费参考误差，再比较native全频时间加权值，拆为“参考时序差”和“采样/加权差”。恒等式e_original-e_native100 = reference_delta-truth_delta成立：v47/v48最大残差约2.3e-15/2.0e-15deg/s，真值源差最大约2.9e-9deg/s；可排除真值坐标差作为主要来源。

V48候选全段R/P/Y的MSE差（原100Hz减native，单位deg²/s²）分别由参考项[.01444484,.00384917,-.01186369]和采样项[.00228480,.00741001,.01788911]组成。v47候选对应参考项[.01581715,.00438608,-.01022306]、采样项[-.00812582,.00285163,.00498373]。参考更新时序和采样相位都足以改变小幅收益的符号。v47/v48开发种子不同、入场状态不同、协议修订历史不同，不能据两个种子做学习率因果结论，也不能选择对候选更有利的口径替代原验收。

## 假说与最小下一版（仅提案）

最直接、最小可检验假说：目前先仅投影九项成本半空间，再因一个bias维度越界缩小整个154维提案，压缩了仍有可行余量的权重方向。先只修改“可行提案构造”，不再尝试同样的LR放大，不同时改奖励/alpha/状态接口。

候选构造：在现有成本线性半空间与批内/全局A两套bias盒的交集上投影Adam提案。bias增量下界=max(batch_anchor-limit-current,A-limit-current)，上界=min(batch_anchor+limit-current,A+limit-current)，limit保持1e-4-1e-10。从合法当前状态出发，用固定有界迭代和显式残差检查；不收敛即拒绝，不靠调大容差。投影后仍逐档执行原13档回溯及全部非线性KL/ESS/cost/weight/action/冻结参数守卫；非零参数和非零输出均为计数条件。禁止事后clip绕过门槛，禁止把第17步当新比较锚点。此方案尚未实现或运行，不能承诺会改善闭环性能。

先做零native离线准入：在已有精确pre18和pre40恢复点，各比较旧/新构造共4个一次性提案；原actor/Adam/RNG/样本进度和batch/global锚点一致。验证可行起点、角点/NaN/无解/全零/浮点边界、冻结张量、全部原守卫、回滚身份、完整提案记录、确定性恢复。新方向须在相同minibatch上有非零、有限的surrogate改善且全部原守卫通过；否则否决，不靠凑满更新。之后可离线完整重放检查每步守卫和数值稳定，仍不得声称固定旧数据可替代新闭环验证。

目标曲率/早期信用问题作为第二个独立假说保留：先离线比较同一状态/同一扰动下有界代价与原SSE/P99变化、前3秒局部与全段return梯度方向一致性；若证实系统性冲突，再另行预注册奖励/信用分配修订，不能和投影改动混成一次试验。当前证据不足以直接将pitch惩罚调大或宣布原目标错误。

若离线准入通过并获得新native授权，建议新独立版本从A+新Adam开始，保留当前LR、奖励、sync2、4000/s、2048优先、原门槛及holdout封存，仅测试上述提案构造。沿用4×7采集/每批10步/最终40唯一候选；训练计划上限28×47000=1316000，硬上限2000000。完整54开发上限18×56000+36×47000=2700000，硬上限3000000；不额外做收费pilot/重试以突破预算。原V48的1213571/2397578账单保持不变，新版本预算须独立明确授权。两种子仍只能形成有限开发证据，不能据此作广泛因果结论。现在只提交方案，不启动新native。

## 证据与限制

本目录diagnosis.json含54条同轨迹参考分解和同状态输出比较；credit_and_constraints.json含28条目标/信用分配、曲率、逐步alpha/scale、提案拒绝分类及批次动作方向；verification.json核验曲率有限差分、终奖基线抵消、两版参考恒等式。所有native及optimizer调用计数均0。原记录、正式账本、模型、冻结源码未修改。
'''
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text,encoding='utf-8');print(json.dumps(checks))
