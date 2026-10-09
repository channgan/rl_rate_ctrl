# 偏置交集投影零 native 准入 — 2026-10-09

结论：**不通过，不启动新一轮 native 训练。** 预声明 pre18 通过、pre40 失败；两个恢复点全部报告，没有筛选成功样本。只实现隔离的候选提案投影，未安装到 V48 运行时，也未改变正式 checkpoint、账本或当前模型。

## 固定方案与唯一算法修改

基于已提交 `05e497a4f17a9a3b36b64f9021a314104d4ee86c` 方案，固定 pre18（已接受17，批次起点10）和 pre40（已接受39，批次起点30）。精确读取原 actor、Adam、RNG、minibatch permutations/positions、alpha、数据及 guard hash；没有把比较锚点重置为17或39。

候选用 Dykstra 投影求 Adam 增量与九个线性成本半空间、批次/全局 A 两组 bias 边界的交集。bias 限值仍为 `1e-4-1e-10`。绝对边界先向内舍入到 float32 可表示值，避免转换后越界。固定最多512轮，float64归一化成本残差及相邻迭代变化阈值1e-12；转换残差阈值1e-10仅用于数值检查，不改变任何非线性成本守卫。首轮运行前已写 frozen_plan.json；失败后没有改算法常数或选择新恢复点。

仍调用原 scheduler step，唯一替换是 proposal projection 函数。保留全部13档 `2^-i, i=0..12`、KL、ESS、ratio、九项成本及相对前一步成本、weight/bias/action、冻结张量和非零参数/动作守卫。目标、alpha、LR3e-5和Adam逻辑不变。投影不收敛直接拒绝并执行原批次起点回滚，不返回未收敛增量。

## 全部固定对照

surrogate 为相同2000条加权 minibatch 的 clipped surrogate，正差为离线代理改善；action为相同历史状态上的策略输出差。单位为控制器归一化动作，不是角速度误差。

| 恢复点 | 构造 | 结果 | 接受scale | surrogate增量 | 接受更新动作最大变化 |
| --- | --- | --- | --- | --- | --- |
| pre18 | original | PASS | 0.03125 | 1.138148020564042e-06 | 5.234032869338989e-07 |
| pre18 | bias_intersection | PASS | 1.0 | 2.7876129527213203e-05 | 5.041365511715412e-06 |
| pre40 | original | PASS | 0.00390625 | 4.776423936578289e-08 | 1.862645149230957e-08 |
| pre40 | bias_intersection | FAIL | none | not accepted | 0 |

pre18 候选两轮即收敛，九项原始成本梯度点积均为负，归一化正残差为0。相对原投影，接受scale从1/32变为1，代理改善约24.49倍；这不代表闭环收益。候选实际两行权重变化L2为7.92816e-5/6.78729e-5，动作RMS为7.89424e-7/3.68192e-6/0（R/P/Y）；冻结张量和yaw输出未变，所有原守卫通过。

pre40 候选达到固定512轮仍不满足收敛要求：归一化正成本残差 `2.278952456443028e-10`，末轮最大变化 `1.653070979899102e-10`，bias盒残差 `0.0`。未进入13档回溯、没有接受更新。确定性回滚到原第30步actor/Adam/RNG，成功计数未增加。JSON中该失败行的普通参数/动作差及surrogate_after描述的是回滚到批次起点的差异，**绝不是有效学习更新**；补充的effective_accepted字段明确为0。pre40原投影仍按1/256通过，输出二进制SHA与原V48 final40完全一致。

## 测试、计数与原记录保护

- 两个恢复点 × 原/新 × 同进程精确重放 = 8次一次性optimizer尝试，另两次强制零提案验证13档全拒绝及回滚，共10次。
- 在第二个独立进程复验相同10次，总计 **20次一次性离线optimizer尝试、0次正式更新、0次native调用**。第二次仅增加失败残差诊断和回滚字段标注，没有改动算法、常数或恢复点。两进程actor二进制SHA、Adam步数、minibatch hash、完整诊断和surrogate结果逐项一致；每个进程内部完整actor/Adam/RNG/生成器/permutations/positions/报告状态重放精确相等。
- 7个单元测试通过：成本与双bias交集、零提案/零梯度、NaN/Inf/不可行输入拒绝、两侧相邻float32边界、固定迭代耗尽拒绝、输入不变和确定性。初次pytest误扫描父目录而遇到无关受限文件，限定rootdir/confcutdir后通过；未更改系统访问权限。
- 强制零提案在两个恢复点均不增加成功计数；原批次actor、Adam、RNG、生成器恢复精确，permutations/positions清空，stopped为真。
- 所有原运行时Python文件、原两个pre-update checkpoint、四批recovery及正式训练/开发账本SHA前后一致。原V48失败及最终记录保留；未创建或发布新模型。

仅pre18有离线代理改善，pre40失败意味着本候选整体不具备准入资格。没有新的闭环轨迹，不能声称稳定性、跟踪精度或电机平滑度改善。后续若研究更适合交集的固定求解器，应作为新的预声明离线版本评审，不能在本轮不断调迭代数至通过。

## 新一轮配置、预算及终止规则（未获启动资格）

若未来完整离线准入通过并另获native授权，建议按原方案独立版本从A+fresh Adam开始，仅改变提案投影，LR3e-5、154个RP参数、冻结其它参数、既有目标/alpha/采集及所有物理验收保持。4×7新采集，每批10更新，总40，唯一final40候选；完整54开发矩阵，holdout12701/12702继续封存。

训练计划上界28×47000=1,316,000，硬上限2,000,000；开发计划上界18×56000+36×47000=2,700,000，硬上限3,000,000。所有warmup、失败和实际调用计费，无额外pilot、种子搜索、补采或重试。保留旧V48费用1,213,571/2,397,578，不能清零或混账。本轮不分配新的native预算。

采集/同步/入口/物理/数据审计失败即停；非有限值、不可行起点、求解512轮未收敛或转换后越界即停；全部13档拒绝即回滚批次起点并停止，保留最近已接受checkpoint，不把零步/拒绝计为成功。到40步停止更新，只有全部开发验收通过才讨论晋级；预算达限即停，不自动扩容。

## 复现

在含torch/numpy/pytest的离线Python环境中，运行：

```sh
python -B -m pytest -q -p no:cacheprovider --rootdir=scripts/v48_bias_projection --confcutdir=scripts/v48_bias_projection scripts/v48_bias_projection/test_bias_projection.py
python -B scripts/v48_bias_projection/admit_bias_projection.py --v48 /path/to/v48_fixed_lr3e5 --out /new/empty/output_directory
```

依赖原本地审计后的capture和可信checkpoint，不可从本仓库快照独立重建原轨迹。输出目录必须不存在，防止覆盖。完整首轮/复验准入、冻结方案、逐档审计及交叉核验位于 `snapshots/v48/bias_projection_admission1/`。原模型/原始轨迹/账本不入Git。
