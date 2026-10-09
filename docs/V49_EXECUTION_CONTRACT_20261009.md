# V49 冻结执行合同与准入 — 2026-10-09

用户已明确授权准备并启动V49。先通过全部集成准入、提交并核验远端，再生成实际运行源码对应证据，最后由既有独立Windows holder启动。首条计划内正常采集train_b0_p0为实际native端到端验证，失败即停；没有额外资格样本。

唯一新算法改动是确定性可行活动集+SVD，将原批次/global-A bias边界纳入原九项成本可行集。固定512步、原数值标准、13档有限回溯及所有KL/ESS/成本/权重/动作/冻结/非零守卫保持。基线明确采用V48最终有效规则：从首步开始global平滑alpha只减不增，历史记忆分位在所有批次仅诊断；速度、记忆有限/界限/历史、物理与采集同步仍为硬门槛。不存在历史11/21激活时间点。

起点A2428576135，新Adam，LR3e-5，仅154个RP输出参数，trunk/yaw/gate/history冻结。4批×7条全新轨迹、每批10次有效更新，总40；只能使用final40进入54项独立开发，PID仅作比较，原PX4上游与独立网络结构不变。候选不自动晋级。

训练硬上限1,316,000=28×47000；开发硬上限2,700,000=18×56000+36×47000。所有warmup、探针、失败调用计费；不知道实际值的失败按预留上限收费。原计划无最坏情况重试余量，实际节余也不授权增加回合。预算不足即停，不隐含扩容、补采、种子搜索或重试。

训练种子每批490101+b×1000+i（b=0..3，i=0..6）；开发490301/490302，9任务×PID/A/final40。holdout12701/12702继续封存。优化器随机种子470901+batch、初始minibatch生成器460901与基线一致；跨批保存actor/Adam/RNG，新数据重建采样排列。原first3与全部失败回合保留，原100Hz和native-consumed两参考口径同时报告，不换验收口径。

准入：35项单元测试通过（Windows入口11、冻结合同/预算/规则13、投影边界11）。真实update_batch入口在明确标记的历史离线夹具上完成两次10步；actor/Adam/RNG/排列完全一致，首步alpha=0.702638918807，10次均scale1非零。预更新checkpoint恢复、冻结张量篡改拒绝、零提案13档拒绝、求解预算耗尽回滚、NPZ/binary/checkpoint/同状态前向一致均通过。实际集成类在固定pre18/pre40与已提交候选逐项相同，并重放一致。总27次一次性离线optimizer尝试，不计正式更新，0 native。

V49使用独立实验目录和空正式账本。旧V48模型、账本、运行时、失败和报告不变。旧入口硬编码2m/3m仅在隔离V49入口/运行时缩紧至本合同值，没有增加权限或替换原工作树。准入入口还要求remote_verified和实际源码SHA与提交一致，未提供提交证明会拒绝启动。

合同和源码快照在snapshots/v49，source_manifest区分原始运行文件SHA与仅路径标记/LF转换后的Git快照SHA；模型、原始轨迹、checkpoint和账本不入Git。运行时source_provenance.json须在远端核验后生成。机器特定路径标记需使用私有映射还原，快照不应直接当作当前机器可运行文件。

复现准入：scripts/v49/test_v49_contract.py（设置V49_TEST_ROOT）、test_v49_windows_admission.py --root、test_v49_integration.py --root --v48、test_v49_fixedpoints.py --root --v48 --reference。集成输出目录必须不存在；历史数据仅用于测试，正式validate_batch在所有批次拒绝外部/历史路径。

正式入口：V49/launcher/scripts/supervise_training.py --config V49/execution/admission.json start --run-root V49/execution。启动后以execution/holder_identity.json、training/driver_identity.json、首条采集attempt及审计为证据，启动不等于训练完成。训练和开发各自status/ledger记录当前阶段和保守计费；任何守卫失败停止，无自动重启。
