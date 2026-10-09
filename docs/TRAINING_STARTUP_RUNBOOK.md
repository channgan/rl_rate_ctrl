> **当前流程更新（2026-10-08）：** 用户已授权解除辅助监控耦合后直接训练。新训练使用独立后台执行、内部硬预算和自动保存；助理周期只读日志。旧一小时验证仅是辅助诊断，下面历史“因hour失败禁止训练”的条目不再是新训练门槛。原失败记录不改写。新唯一启动/状态/停止命令见第11节。

# 强化学习训练与仿真启动运行手册

更新：2026-10-08。适用项目：`@DRL_ROOT_WINDOWS@\rl_rate_ctrl_refactor`。本手册只涉及本地仿真、离线训练管理与证据保存，不授权实机操作。

## 1. 当前结论：训练仍锁止

唯一项目操作入口为 **`scripts/supervise_training.py`**。其状态、预检、计划与协作停止命令已经实现；实际启动必须通过全部门禁。入口委托已冻结的 Windows 持有者 / WSL 监督链，不另造一套监督器。

当前不得启动或恢复训练，原因是原定两项验收只有一项通过，而且没有新的训练调用授权：

| 项目 | 已核验结果 |
| --- | --- |
| A / seed 469901 / OFF，连续 90 秒 | 通过：9000 步；无物理失败；NN 111,883，PID 0；完整导出与计数审计通过 |
| 真实评分采样间隔 | 最大 3 ms；>4 ms 的间隔为 0 |
| 实际消费参考信号三轴 RMSE | 0.089563 / 0.093307 / 0.046690 °/s；不能用旧 `original_latest` 标签代替 |
| 高度 | RMSE 0.023976 m；最低 4.975882 m；电机上下限饱和比例均为 0 |
| 真实任务主机时长 | 预检 30.067893 s（不占启动窗口）；启动至有效 INIT 31.779444 s；有效阶段至关闭 4151.651283 s |
| 一小时零 NN 监督验证 | **失败**：覆盖下界 1880.778375 s，上界 1880.871038 s，未达 3600 s |
| 一小时任务终态 | 协作停止；完整导出、计数有效、结算完成；无未清预留；主机观测间隙 0 |
| 失败触发信息 | WSL 时钟桥报告 `OSError(22, 'Invalid argument')`，随后关闭；不是本次持有者消失 |
| 真实预算 | 原记账 1,109,747；本次新增 111,883；当前 1,221,630 / 1,223,644；剩余 2,014 |
| 策略更新 | 仍为 40；本次没有新增策略更新，没有新训练 checkpoint |

零 NN 夹具生成了 21,899 条合成记录。其局部账本中 `actual_NN` 是模拟 ABI 计数字段，**不是 21,899 次真实推理**；真实 NN 调用为 0，主账本不受它影响。

一次固定种子存活 9000 步超过了“先活到 2048 步”的阶段目标，但不证明多种子稳定、训练收敛或可实飞。训练配方仍按 `TRAINING_PARAMETERS.md` 与批准协议执行，失败剩余时间惩罚 4000/s，不在线改奖励或缩短回合来制造成功。

## 2. 环境与文件约定

- Windows PowerShell；已使用 Python：`@WINDOWS_HOME@\AppData\Local\Programs\Python\Python312\python.exe`。
- WSL：`Ubuntu-22.04`，用户 `cy`；Python：`@LINUX_HOME@/rl_rate/.venv/bin/python`。
- 公开训练接口仍为 9 维观测 / 3 路归一化力矩，控制步长 0.01 s、物理步长 0.001 s、常规回合 2048 步。历史 6 入 / 4 电机模型不能直接套用。
- 本次 v46 冻结实验是内部 18 特征 native NN 诊断链。它不等同于公开 9/3 训练器，不能把诊断通过冒充公开训练适配完成。
- `runtime.supervision.json` 是本机配置，已被 `runtime*.json` 规则排除于 Git；它包含运行包、主账本、结果与哈希路径，当前 `enabled=false`。
- 项目代码、手册与入口在 Git 工作树；模型、运行日志、授权、主账本、机器路径配置和证据放在项目外。不要覆盖用户修改或旧 checkpoint。
- Windows 的状态/计划检查只需标准库。环境类按需导入，仍保留 `rate_rl.RateControlEnv`、`rate_rl.TaskConfig` 原名称；需要实际环境时才加载 Gym 等依赖。

本次证据根目录（下文称 `EVIDENCE`）：

```text
@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_entry_coverage_v46\stage_holder_execution_revision4
```

主账本：`@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_entry_coverage_v46\ledger.json`。

主要记录：`EVIDENCE\completion_report.json`、`production\host_completed.json`、`host3600\host_completed.json`、各任务的 `holder_completed.json`、`matrix_exit.json`、`matrix_reaped.json`、`settled.json`。第一次停止原因在 `host3600\stop`，不要只看后来传播的 `task stop requested`。

## 3. 可直接执行的管理命令

### Windows：状态与门禁

```powershell
$DrlPython = '@WINDOWS_HOME@\AppData\Local\Programs\Python\Python312\python.exe'
$DrlProject = '@DRL_ROOT_WINDOWS@\rl_rate_ctrl_refactor'
& $DrlPython -X utf8 -B "$DrlProject\scripts\supervise_training.py" --config "$DrlProject\runtime.supervision.json" status
& $DrlPython -X utf8 -B "$DrlProject\scripts\supervise_training.py" --config "$DrlProject\runtime.supervision.json" check
```

`status` 只读。`check` 当前应打印 `start_allowed=false` 并返回非零状态（Python 退出码 2）；这是正确的拒绝，不是可以绕过的错误。它核对结果哈希、固定运行包、主账本、残留预留与授权。完整 Linux 依赖检查使用下面的 WSL 命令。

### WSL：完整依赖预检与离线自检

```powershell
wsl -d Ubuntu-22.04 -u cy -- @LINUX_HOME@/rl_rate/.venv/bin/python -B @DRL_ROOT_LINUX@/rl_rate_ctrl_refactor/scripts/supervise_training.py check --full-dependencies
wsl -d Ubuntu-22.04 -u cy -- @LINUX_HOME@/rl_rate/.venv/bin/python -B @DRL_ROOT_LINUX@/rl_rate_ctrl_refactor/scripts/supervise_training.py self-check
```

本轮入口固化后，完整预检明确发现 `rate_rl/__init__.py`、`rate_rl/launcher.py`、`rate_rl/train.py` 三项与旧冻结清单不符，分别来自按需导入及旧启动锁止；其原文件已备份，旧清单没有刷新。其余 1592 个文件和 627 个链接核对一致。这个预期代码差异同样阻止旧运行包再次启动，不能用此前的真实测试替新代码签发验收。结果在 `EVIDENCE\standardization\full_dependency_check.json`。

完整预检按冻结清单读取 1595 个依赖文件和 627 个符号链接。若出现漂移，记录具体文件与用途，不能为了变绿直接刷新哈希。自检只创建临时文件、模拟可重复的错误并回放已有全链路证据；不启动 PX4/Gazebo、不产生 NN 调用、不新增一小时测试，也不授权训练。

### WSL：只读训练计划

```powershell
wsl -d Ubuntu-22.04 -u cy -- @LINUX_HOME@/rl_rate/.venv/bin/python -B @DRL_ROOT_LINUX@/rl_rate_ctrl_refactor/scripts/supervise_training.py plan --run @DRL_ROOT_LINUX@/experiments/startup_preview_never_launched --steps 4096 --episode-steps 2048 --runtime @LINUX_HOME@/rl_rate/runtime.free_flight.json
```

`plan` 强制走现有公开配方的 `--dry-run`，只显示参数和底层命令，不创建 run、不加载模型、不启动进程。示例 run 路径必须不存在；若已有目录，另选新目录，不删除已有成果。它不是一次训练授权，也不说明底层训练命令现在可以直接执行。

### 实际启动：当前明确不可用

已实现的命令形式是：

```text
scripts/supervise_training.py --config <审核后的本机配置> start --run-root <已准备且获批的新任务目录>
```

这只是参数说明，**当前配置不能启动**。门禁在创建持有者之前拒绝失败验收、未授权预算、未清预留、已使用目录、错误运行包或未合格训练适配器。不能把 `enabled` 或 `training_adapter_qualified` 改成 true 充当验收/授权。当前 9/3 训练器尚未在此 holder 发布包中合格接入，不能把旧 `RunSupervisor` 当作备用启动路径。

将来确需训练，也必须使用这个入口及同一监督实现；先有真实修复、匹配的阶段协议/适配器回归、完整验收和明确预算。已支持阶段的参数通过配置管理；尚未实现或未验证的能力不得伪装成“改一个配置就能运行”。此次没有执行新入口的 live 分支，不宣称其训练链已获新验收。

### 协作停止

```powershell
& $DrlPython -X utf8 -B "$DrlProject\scripts\supervise_training.py" --config "$DrlProject\runtime.supervision.json" stop --run-root '@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_entry_coverage_v46\stage_holder_execution_revision4\host3600'
```

此示例目录已关闭，命令应返回 `already_closed=true`、`changed=false`，不会改旧记录。对于当前配置中登记的活跃任务，它只独占创建协作停止标记，不按进程名杀进程；必须继续等导出、真实退出证据和账本结算。

## 4. 唯一启动顺序与阶段协议

1. **只读盘点与授权**：核对现有任务身份、日志和预留；连接列表的 `connected` 不等于模型在运行。确定新任务目录、模型/协议哈希、允许的 NN 调用数、阶段目标和停止条件。预算不足或计数不明时拒绝启动。
2. **独立持有者与完整预检**：唯一入口沿用已执行的 WMI/Job 启动原语，启动独立 Windows holder；它先校验依赖、链接、运行包、配置和授权，再创建 Windows 时钟会话。预检耗时不占 startup 窗口。仅创建持有者不执行 NN。
3. **WSL 监督与原子预算预留**：Windows holder 持有 Windows owner 和 WSL 客户端句柄；WSL matrix holder 持有 matrix owner 的真实进程句柄。matrix owner 在集成账本锁下原子预留，随后才允许 guardian → runner → payload。预算可用性在进程前预检查，实际原子预留由原监督 owner 完成；不得再添加一个重复预留者。
4. **初始化与预热**：注册 PX4/Gazebo 身份，执行 INIT/预热，保留模型 ID、native NN/PID 计数、故障位和模拟时刻。旧 checkpoint 与失败启动不可覆盖，不自动重新 bootstrap。
5. **真实就绪**：当前诊断使用主线程成功 INIT 的 `init_complete` 标记结束 startup；随后对齐到仿真 35 s，再发 `scoring_ready` 开始正式评分。普通文件存在、捕获文件长度、首次刷盘或线程心跳都不能替代真实就绪。首次可见记录不等于首次推理时刻。
6. **采集/训练阶段**：只执行已注册、获批且已验证的阶段。当前验证的是采集/评分，不是新增优化更新。记录真实计数、固定步长、采样间隔和完整性；评分中 >4 ms 的间隔触发停止。
7. **停止、导出、退出、结算**：STOP 一旦生效不可再转 READY，不重置时限；协作停止 → 完整导出/最终完整性审计 → scoped 子进程退出 → Windows/WSL OS 退出回执 → 原子结算。尚有导出义务或未知存活子进程时保持阻塞，不删除预留、不开始下一项。

该顺序保持已验证的 holder/owner/guardian/runner 分工，没有复制一份临时拼接的监督器。新入口的 Windows Job/WMI 原语来源与原文复用证明在 `EVIDENCE\standardization\holder_primitive_provenance.json`。

## 5. 时钟、预算与停止门禁

- 有效覆盖只用同一 Windows boot / owner 下的 `time.perf_counter()`。WSL monotonic、日志 wall time、仿真时间只能分别作诊断或物理区间指标，不跨时钟相减。
- 记录 `ready_lower/upper`、`stop_host`、`ack_host` 与 `closed_host`。一小时下界是 `stop_host - ready_upper`；不能拿进程总寿命、预检耗时或文件修改时间凑足 3600 秒。
- 当前真实任务：startup 120 s、有效阶段上限 5400 s、导出宽限 600 s，目标完整 9000 步 / 90 s。零 NN：startup 20 s、覆盖下界至少 3600 s、导出宽限 30 s。预检独立计时。
- holder 必须证明不属于实际观察 Job，并持有子进程句柄记录真实退出码。`in_job=true` 可以属于 WMI 服务的其他 Job；不得声称它脱离所有 Job，也不保证主机重启、断电或 WMI 服务被杀后仍能善后。
- 旧扣费保留，不清零、不退款、不将未知实际计数写成已证实的零。本轮真实硬闸 113,897，主账本总上限 1,223,644；包括 bootstrap 与所有实际 native NN/PID 调用。
- native 二进制在推理前检查 `PX4_NN_MAX_CYCLES`；Python 还有 pre-step 与完整目标预算检查。捕获无效/计数不全时按预留 cap 保守结算；相同结算幂等，冲突结算拒绝。
- >4 ms、真实安全故障、时钟读取/验证错误、owner 死亡/过期/读失败等按冻结协议停止。不能退回 PID、借 WSL 时间继续计时或压低目标来制造成功。

## 6. 已确认缺陷与可重复防回归

执行 `self-check` 可重复以下离线夹具/证据核对。区分“重新执行离线夹具”和“回放原全链路证据”；后者不是新的一小时或 native 运行。

| 缺陷或边界 | 防复发检查 | 当前依据与范围 |
| --- | --- | --- |
| alias/打包导出错误 | 受控别名先注册 `sys.modules` 再执行，检查必要导出可调用；固定包哈希 | `alias_module_registered_before_exec`；不是任意打包的形式化证明 |
| JSON 半写 | live 读为 Pending，final 读为 Invalid；不得接受半个对象 | `half_json_pending_live_invalid_final`；真实半写阶段已执行但一小时最终未通过 |
| Windows WinError 5/共享冲突 | 临时文件、原子替换、仅有界重试；最终文件保持完整 | `winerror5_bounded_atomic_replace_fixture`；本次真实记录有 49,833 次已恢复重试，最大约 0.051722 s，未触发 Windows 首异常 |
| callable 被变量覆盖、通配导入 | AST/绑定检查及运行包逐文件哈希 | 原 20 模块通过；本次增加授权模块后 21 模块通过，无通配导入 |
| 只测外壳，未覆盖 guardian | 核对真实 Windows → WSL → guardian → runner → fake payload 的模块哈希与 OS 退出回执 | 原 7 项全链路通过；自检回放这些证据，不另启链路 |
| 用文件缓冲判断 READY | 明确阶段标记；文件大小仅可表示导出义务，不能决定就绪 | `explicit_ready_only_and_stop_before_late_ready`；迟到 READY 被拒绝 |
| owner dead / expired / read failure 混淆 | 保存进程身份、主机 stamp/age、阈值与读取错误；三种原因分别判定 | `owner_dead_expired_read_failure_distinct`，并有对应真实链路夹具 |
| 观察会话退出带走任务 | 独立 WMI holder、观察 Job 成员关系验证、OS 句柄退出回执 | 原观察 Job 整体终止夹具通过；真实 90 s 运行期间 exec-server key 变化后同一 PID 继续完成 |
| 超预算/未知计数/重复结算 | 113,898 拒绝，113,897 可预留；未知捕获全额记账；幂等与禁止重试 | `budget_unknown_counts_conservative_idempotent_no_retry`；另有上限停止完整导出夹具 |
| 本次时钟桥 errno 22 | 保持安全停止，不创造经过时间、不借 WSL 时钟继续、不自动重试 | `host_clock_errno22_fail_closed_no_invented_time` 只证明停止语义；**未证明底层读错误已修复** |
| 旧入口绕过 | 公共实际启动、直接训练模块和旧续训的启动分支拒绝；只读计划保留 | `current_failed_gate_and_legacy_starts_fail_closed`；不提供 magic 环境变量绕过 |

原七项场景为：normal、late_ready、owner_dead、read_failure、heartbeat_expired、windows_owner_exit、observer_disappears。预留/上限夹具使用隔离账本；不占真实预算。

## 7. 失败原因：哪些确定，哪些仍未知

已确定：此前完整依赖预检重复计入 startup，压缩真实启动时间；捕获写入缓冲使“首个可见文件数据”无法代表首次推理；旧状态机可在 STOP 后接受迟到 READY；旧观察会话与工作进程生命周期耦合且退出证据不足；owner 消失与租约/读取故障曾共用一个原因；打包别名、名称覆盖/通配导入和 Windows 原子替换共享冲突已有对应证据与修复/回归。WinError 5 在真实入口可复现，但最早两次缺少原始堆栈，不能倒推它就是每一次历史退出的唯一原因。

本次确定的是：主机时钟桥报告 errno 22 后触发安全停止；Windows holder/owner 和 WSL 监督仍能记录退出、导出和结算。host coverage 仅 1880.778375 s。Windows 没有记录时钟回退或发布观测间隙。半写入夹具与 errno 22 发生在同一时段，**不能据此断言二者有因果关系**。

仍未知：errno 22 的底层成因与具体首次报错读取者（现有 stop 文本缺少完整堆栈/PID）；旧 8 ms 间隔的完整发生机制；旧异常退出/所谓“关机”的发起者及先后顺序。历史 Windows/WSL 时钟、网络日志与退出相邻也不构成因果证明。不能声称全部根因已解决或保证永不再错。

发生任何新失败：先保留 first-stop、first-exception、所有阶段/心跳/OS 身份、捕获与完整性文件、账本和退出回执；完成导出结算再分析。若底层读错误仍未知，不放宽门禁、不偷偷重试一小时、不重新训练来掩盖旧失败。

## 8. 维护与禁止事项

- 未来操作仍从 `scripts/supervise_training.py` 进入；配置、阶段协议和已支持参数变更对应做离线回归，必要的真实验收须有新的明确授权。不能复制旧脚本临时拼接另一套监督器。
- 不执行旧 README 的 live 命令、旧 `python -m rate_rl.train`、续训脚本或临时实验 launcher 绕过门禁。旧快照保留为证据，不改其封存内容；其冻结依赖检查会拒绝本轮入口固化产生的源文件差异。项目级门禁不是操作系统沙箱，不能阻止拥有写权限的人故意改代码或手动执行裸二进制。
- 不复用已启动目录，不覆盖 checkpoint/失败日志，不修改旧实验 3500/s 配置冒充 4000/s，不把一次存活当作全部验收。
- 不按进程名批量杀 Python/PX4/ROS/VSCode；不改变系统时间、服务、网络设置或购买算力；不操作实机、解锁电机或部署策略。
- 只有新门禁完整通过、预算明确且阶段适配器匹配，才讨论有限恢复训练。本次剩余 2,014 不能被当成一次新的完整训练预算。

## 9. 交付核对

1. `status` 与 `completion_report.json` 一致：真实项通过、一小时项失败、训练锁止。
2. `check` 拒绝启动；`plan` 不创建目录；已关闭任务的 `stop` 不改旧文件。
3. `self-check` 给出逐项结果，明确 `native_calls=0`、`new_payload_processes=0`、`new_long_tests=0`。
4. 原 checkpoint、冻结 native 二进制、旧 959 文件/2 链接证据和离线修复 935 文件证据校验通过；既有用户修改保留。
5. 本机入口配置、离线自检结果与具体阻断原因记录在 `EVIDENCE\standardization`；最终封存清单在 `EVIDENCE\execution_seal.json`。离线自检通过不改变失败的一小时结论。


## 10. 2026-10-08 时钟读取故障专项结论（revision5）

**原一小时仍失败，底层 EINVAL 根因未查明，不恢复训练。** 原 `host_bridge.py` 通过 `Path.read_text()` 读取 Windows 发布的 `host_session.json` / `host_clock.json`；此路径不调用 `clock_gettime`，也不传 clockid。Windows 生产者使用 `time.perf_counter()`；原结果没有 Windows clock_fault。原异常处理只写 repr，未保存首发栈、PID、线程及 open/read/close 区分，因此原始这些信息不可恢复。现存 matrix traceback 是后续 `attempt failed; stop matrix`，不能冒充首发错误。旧 lease lost 与本次同属监督依赖链，但没有同一底层原因的证据。

最小修复位于独立 `clock_read_revision5/runtime/host_bridge.py`：分开记录 open/read/close/JSON/校验阶段，保留完整首发 traceback、路径、参数、errno、PID/线程、WSL/Windows 来源和最后有效宿主时间；故障锁存 STOP。恢复文件读取不能重新开始计时；最后有效时间只用于退出导出记账，不回退到 WSL 时钟。stderr 与每进程 `clock_fault_<pid>.json` 保留证据。原 revision4 及其清单保持封存。

通过真正 `scripts/supervise_training.py ... start` Windows 入口完成两个短程零 NN 全链回归：正常产生 32 条合成记录；读阶段注入 errno22 产生 2 条后停止。二者均经原 `_holder_gateway` → WMI holder → Windows owner → WSL matrix/guardian/runner/fake 链路，脱离观察 Job；导出、结算、OS 退出均核验。故障例 Windows owner / matrix 退出码均 1，覆盖不合格。注入栈明确标识 injected=true，只证明处理路径，不证明原 EINVAL 的底层原因。另一个隔离测试证明文件恢复后时间仍冻结、停止仍锁存。

统一入口没有新增监督实现：原 `_holder_gateway.py` 未修改，冻结运行包 25 个 Python 模块中只变更 host_bridge.py。新 `zero_nn_clock_regression_v1` 是独立、一次性短测授权：fake 后端、独立账本、确切目录/输入哈希、最多 64 条合成记录、活动窗口最多 15 秒，不能用作训练授权。新离线发布仅显式复核前次入口修改的 3 项依赖差异，不刷新旧清单，不赋予 native 资格。原训练 policy 仍 disabled，公共 9/3 适配仍未合格。

本轮实际执行命令（目录已使用，**再次执行应被拒绝**，不是重测许可）：

```powershell
& '@WINDOWS_HOME@\AppData\Local\Programs\Python\Python312\python.exe' -X utf8 -B '@DRL_ROOT_WINDOWS@\rl_rate_ctrl_refactor\scripts\supervise_training.py' --config '@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_entry_coverage_v46\clock_read_revision5\policy.json' start --run-root '@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_entry_coverage_v46\clock_read_revision5\read_errno22'
```

证据：`@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_entry_coverage_v46\clock_read_revision5` 下的 `original_incident_analysis.json`、`verified_normal.json`、`verified_read_errno22.json`、`latch_regression.json`、`entry_rejections.json`、`report.json`。原账保持 1,221,630 / 1,223,644，余额 2,014；没有 native 调用、策略更新、模型替换或长时重测。之后任何实跑都仍需明确授权；不能用本次注入结果宣布真实故障已修复。


## 11. 简单独立训练流程：v47

用户已批准200万训练/搜索真实native上限、300万独立开发评估上限。v46旧账1221630/1223644封存不重置；新实验明确改进为4批×5新轨迹、每批10次约束更新，按批在线采集、批内固定数据优化。从A开始，v46第40次权重保留但不晋级。详见 NEXT_TRAINING_PROTOCOL.md 和 `native_fresh_batch_rp_v47/protocol.json`。

唯一入口仍为 `scripts/supervise_training.py`，委托原WMI/Job启动原语持有独立Windows进程，再持有一个WSL训练执行进程。训练不使用host-clock文件或助理心跳作lease；普通状态写入/只读日志失败降级，不杀任务。内部native cap、STEP前调用余额、非有限/物理失败、实际控制捕获间隔>4ms审计、原始捕获与checkpoint完整性仍是必要条件。即时采样扫描受已提交缓冲数据可见性限制；完整控制审计必须在任何下一回合/优化前通过，不能声称对每个1ms故障零延迟发现。关键保存/预算/所有权不明停止并保留证据，不自动重试。

```powershell
$DrlPython='@WINDOWS_HOME@\AppData\Local\Programs\Python\Python312\python.exe'
$DrlEntry='@DRL_ROOT_WINDOWS@\rl_rate_ctrl_refactor\scripts\supervise_training.py'
$DrlRun='@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_fresh_batch_rp_v47'
& $DrlPython -X utf8 -B $DrlEntry --config "$DrlRun\admission.json" start --run-root $DrlRun
& $DrlPython -X utf8 -B $DrlEntry --config "$DrlRun\admission.json" status
# 仅明确要求停止时执行；普通日志读取失败不执行stop：
& $DrlPython -X utf8 -B $DrlEntry --config "$DrlRun\admission.json" stop --run-root $DrlRun
```

start为一次性，目录已用则拒绝。助理定期读取training.log、ledger.json、jobs/*/attempt.json和training/batch_*/updates.jsonl；读取失败下次重读，不改预算/控制文件。每个接受更新后保存完整recovery.pt；actor/Adam/RNG/采样位置/数据协议哈希均保留，原始失败/日志不可删除。固定最后第40次才进入开发评估，不按中间分数挑模型。

同真实入口零NN短测：观察Job退出、普通status写入持续故障时执行32次后由内部预算停止并保存；明确stop在5次后停止保存；非有限输入拒绝。旧数据丢弃式离线检查证明A二进制一致、下一次恢复更新及Adam精确一致，原完整捕获审计适配通过。它们不证明新策略已优于PID。

单位更正：前述90秒实际消费参考三轴RMSE数值0.089563/0.093307/0.046690的代码单位为**度/秒（°/s）**；先前报告误标rad/s。数值和原证据未改写。


### v47 首回合容量适配修复与续接

首回合正常完成2048步，原审计使用旧65536容量而当前已验证二进制为131072，故停止。仅绑定NN/reference/mechanism三类容量到现有二进制，未改变47000调用cap、4ms门槛、性能或完整性标准；现有数据离线全审计通过，实际NN41469、PID0、最大间隔2ms。原失败与47000保守计费保留，不退款、不重采。只继续原计划剩余19回合，账本仍是同一个native_fresh_batch_rp_v47/ledger.json。

当前执行记录在 `native_fresh_batch_rp_v47/execution_v47_capacity1`；读取该目录的training.log/training.stderr/holder_completed.json，公共ledger/status/checkpoint仍在父目录。唯一入口的 --config 使用当前执行目录的admission.json，start/stop的 --run-root 使用当前执行目录。已启动目录禁止重复start；status只读，stop是明确协作停止而非读日志失败后的动作。


### 原一小时零NN验证的唯一修复后恢复结果

2026-10-08 04:00:36 UTC有效计时开始，05:00:38 UTC全部关闭。本次通过：Windows单调覆盖下界3600.0172145秒、上界3600.1541121秒，正常/半写/正常恢复三阶段完成，观察间隙0、clock_fault为空，导出/结算完成，退出导出等待2.125秒<30秒。真实native调用0；40532条是合成ABI记录，不是NN推理。Windows owner退出0，WSL matrix按期限协作停止退出1，符合本次覆盖判据，并非要求合成100000条全部完成。

原一小时失败保持失败，原EINVAL底层根因仍未知，不因本次通过追认已解决。旧v46账1221630/1223644、余额2014不变。本结果仅是辅助监督验证，不参与v47训练生命线，也不是新策略性能资格。证据在 `native_entry_coverage_v46/clock_read_revision5_host3600_once/result.json`。


2026-10-08 v47 event-driven STEP continuation (approved): one bridge worker owns simulation advancement. For boundary t, wait for physics t and exact filter/torque/allocator/Gazebo motor-receipt sample t; then request only one 1 ms step. That integration consumes motor sample t; its resulting sensor is t+1 ms. Confirm the applied source remains t and wait for the t+1 ms control-chain acknowledgements before advancing again. Callbacks only notify a condition variable; they never wait for their own callback, recursively advance physics, or update actor weights. Duplicate/late notifications cannot satisfy timestamp/sequence predicates. A 5 s STEP deadline and explicit-stop checks (at most100 ms event-wait cancellation latency, plus any bounded in-flight transport request) stop progress; the4 ms audit is unchanged. Existing unpaused INIT is unchanged; every explicit post-INIT STEP, including conditioning and scoring, uses this gate. Initial boundary ACK is required before the first such step.
New shared runtime: synchronization_revision1/event_revision1/runtime_package. New work: synchronization_revision1/sync_execution; execution_v47_sync1/admission.json remains the unique supervise_training.py entry. Actual-entry zeroNN32-budget/5-stop save tests passed. Ordered validation: lateral_plus_goal100 seed470105 PID,A; noise200_hover_hold seed470101 PID,A. Stop on first real failure. Only all4 full2048/original audit/exact1ms ACK successes permit automatic20 fresh training captures,4x5+10updates. Old4 successful captures stay archived, never mixed. Original canonical ledger218574 retained; probes<=206000 plus20x47000 keeps worst-case1364574<=2000000. Final-only54 development<=3000000, same runtime/PID parameters and heldout12701/2 unchanged.

2026-10-08 fixed synchronization probe1 STOPPED: lockedPID lateral_plus_goal100/470105 completed13613 post-INIT causal1ms integrations and210 scoring steps, then STEP1363 ACK acquisition timed out (~5.179s since preceding record). Read rejection stage was not recorded, so its precise cause is unresolved. No NN calls, no actor updates, no other probes/training started. Secondary CADENCE_EXPORT failure traced to incorrect replacement-source baseline: original archived RLBridge.cpp is native_entry_coverage_v46/supervisor_integrated_revision1/native/RLBridge.cpp, not the original build-tree copy. Old-binary exact relink alone did not validate replacement interface equivalence. Normal SIGINT shutdown recovered all gyro/NN/RREF/mechanism buffers with integrity counts equal; PID actual29258, charge56000 retained. Canonical ledger274574/2000000, reservations0. This runtime is NOT qualified; do not resume or retry before source/interface correction and bounded ACK diagnosis. Evidence: native_fresh_batch_rp_v47/synchronization_revision1/sync_execution/probe1_failure_report.json.


## 2026-10-08 synchronization offline revision2 — NOT runtime qualified

Offline correction only; no new simulator/PID/NN calls, no actor updates, no retry. Canonical v47 ledger remains stopped at 274574/2000000 with no reservations. Reward4000/s and full2048 survival priority are unchanged.

Unique source baseline: native_entry_coverage_v46/supervisor_integrated_revision1/native/RLBridge.cpp, SHA256 4bacb7fe2b40b6ffa41037d1137c3360df7b97fefa9ca717d4b42dc41468261a. All three original source/object pairs and original full binary reproduce byte-exactly. The corrected bridge preserves all seven commands (CADENCE_EXPORT, GOAL, INIT, RATE_TEST, STATE, STEP, STEP_TORQUE), original state/data formats, EOF stop/export, and controller arithmetic. Source/object/archive/final-binary mapping is sealed in synchronization_offline_revision2/source_object_binary_mapping.json; original runtime/checkpoints remain untouched.

Single bridge worker advances physics. Callbacks only notify. Before each 1ms advance require current clock, physics, STEP sequence, filter, torque, allocator and motor receive; require armed/rates-enabled after existing INIT. Initial/current motor application is not required before advancement. After advancing require actual applied previous motor sample and next-boundary observations. The ACK service never waits for a future step. Event wake-ups include bounded100ms requery for missed notification/query races; elapsed time never grants a physics advance. STEP deadline remains5s plus an in-flight existing request bound, with cancellation checks. Each of ten condition diagnostics records required/available/ready, expected and observed values, available sequence/generation metadata, source-error code, cancellation/timeout and RPC details. Unknown sequence provenance is null. Diagnostic path is mandatory when sync is enabled; diagnostic Python caller patches remain proposed, not installed.

Offline results:23 step checks,5 event checks,19 fault checks plus cancellation-race regression;13 actual-dispatch scenarios/126 command exchanges;20 actual synchronization-method scenarios using fake IO/injected clock. Missing-source, late/future/duplicate, cancellation, transport/service/schema/value/exception and never-ready paths covered. Replay of14975 valid saved records/13613 integrations passes recorded timing/sequence/application checks. Original failed STEP1363 produced no further physics; its specific read-rejection branch was not recorded and remains unknown. Lost-requery was reproduced synthetically, not established as the unique real failure cause. The historical trace cannot validate newly added mode/source-availability gates.

Final isolated PX4 SHA256 6bf92a950c87f04668e5d313100876a83be1b57236df2ee4c007f53313dd025e; plugin SHA256 16faecad2cbe2d587f4a0b663a656a715db0ffd7b91eceb9007eb2801d672ed1. Linking and dependency checks passed. This is offline qualification only; no live stability or2048 survival claim.

Future proposal only, not scheduled: PID/A lateral_plus_goal100 seed470105 then PID/A noise200_hover_hold seed470101, cap206000, stop at first failure without replacement. Only all-pass would permit20 fresh captures (940000 cap) and existing4x10 updates; old-runtime captures cannot be mixed. Projected total1420574 leaves579426 of2m. Separate54-episode development cap3m and sealed127xx holdouts unchanged. No automatic retry is authorized by this offline correction.


2026-10-08 approved real integration of synchronization_offline_revision2: Python diagnostic patches installed in its isolated sync_execution/runtime; exact PX4 SHA256 6bf92a950c87f04668e5d313100876a83be1b57236df2ee4c007f53313dd025e verified from running /proc/exe. Unique execution_v47_sync2 admission launched through supervise_training.py, holder30168. Canonical history274574 and six attempts preserved, current four-validation cap206000 counted separately from the previous failed probe while all charges remain under the same2m total. Fixed PID/A lateral_plus_goal100/470105 then PID/A noise200_hover_hold/470101. First real failure stops all subsequent jobs with diagnostic/export preservation; no automatic retry. Only four full2048/original-audit/exact1ms successes permit20 entirely fresh captures and4x10 RP updates. Initialization/arming and ACK gates were not bypassed. Current qualification is pending actual validation, not granted by launch; authoritative state is sync_execution/status.json and canonical ledger.json. Existing models and old runtime data remain untouched. Final-only54 development/3m and sealed127xx unchanged.


2026-10-08 synchronization revision2 real fixed-four qualification PASSED: both PID/A on lateral_plus_goal100 seed470105 and noise200_hover_hold seed470101 completed full2048, original audits, exact1ms causal acknowledgements/raw-capture correspondence and explicit CADENCE_EXPORT. Four-validation charge 182044/206000; old274574 retained, after-validation total 456618/2000000. Scope is this fixed runtime qualification only, not general policy acceptance. Automatic fresh4x5 capture/10updates-per-batch continuation is now enabled by actual results, with no old captures reused, same4000/s/full2048 and final-update40-only rules. Evidence: synchronization_offline_revision2/real_validation.json and sync_execution/validation_complete.json. Live updates and captures remain authoritative in the canonical ledger; no model promotion or heldout use.


2026-10-08 authorized v47 seven-entry collection revision: original five same-sync2-runtime trajectories all survived2048 and passed capture/1ms audits, but original batch FAILED its unchanged entry-velocity coverage before any update. Historical charge673045 retained. Offline analysis disproves zero endpoint command: original dynamic GOAL vz stays+/-0.025m/s before35s, while actual ascending truth response was insufficient. Coverage q05=-0.012510112529149998 derives unchanged from24 historical A/candidate development entries; it is an operating-state coverage guard for collective/attitude/motor coupling, not a privileged actor input. No controller, reward4000/s, PID,4ms, native binary or history-reset changes.
Freeze two added noise200_hover_hold variants with pure NED vz=-0.10/+0.10m/s: C2 ramp30..31s, constant goal slope31..35s, maximum GOAL displacement0.45m, no entry position jump, hold goal during normal2048 scoring. New-only simulation height guard3.5..6.5m, original physical checks retained. Actual entry sign and original velocity/RP-memory minmax-vs-q05/q95 gate required; another failure stops with no further sampling/retry. Seeds470501..470508 fixed in pairs across four batches. Batch1 carries only the five already-audited same-runtime/same-A captures and adds two; subsequent batches use five original profiles plus two vertical variants under the current frozen behavior actor. Never claim original five-entry batch passed.
Five original mission groups retain0.2 each; quiet group split equally among its three variants, first3/later split0.5 each. Fixed2000-transition minibatches use normalized group-phase weights and deterministic rotating66/67 allocation; leave-one-group-out baseline averages trajectories within each other group then averages four groups. Original loss/clip/Adam3e-6/154RP-only/frozen tensors/projection/KL/cost guards retained, individual case-phase checks expanded to14 and group-weighted ESS additionally checked. Ten updates per batch, forty total, only final40 for development. Offline18 command/causal/boundary/weight/budget checks and actual-saved-data duplicate invariance passed, no optimizer steps/native calls; these do not certify future empirical coverage.
Maximum23 new captures x47000=1081000; worst total1754045<2000000, margin245955. No extra probes, seed searches or old-runtime data. Independent54dev cap3m and holdouts12701/12702 remain unchanged. Evidence and new admission: vertical_entry_revision1 and execution_v47_vertical1; original sync_execution failure/logs/ledger snapshot retained.


2026-10-08 authorized v47 smooth-objective revision from update11: preserve original second-batch balance failure (ratios .3197663454/.1389845772), first10 old-objective updates and1061838 native charge. Reuse all seven already-audited batch2 trajectories, no recollection. Before each update, finite original per-stratum/axis tracking and smooth gradients determine a single global cumulative smooth multiplier alpha_next=min(alpha_previous, .099*min(track/smooth),1); never increase. Zero tracking globally disables smooth alpha0 with reasons; nonfinite hard stop. Rebuild training reward/returns/group baselines/normalization consistently and remeasure <= original .1; never select by evaluation reward. First multiplier .3096010615804205 multiplies original coefficients: legacy torque 2/70 -> .008845744616583443, extra RP torque .05 -> .015480053079021026, ESC .05 -> .015480053079021026. Physical scoring, failure4000/s, all constraints/KL/ESS/PID/4ms unchanged. Save alpha/actual coefficients, actor/Adam/RNG/data hash each step and detailed failure metrics. Restore/minibatch consistency and8 boundary tests passed; two disposable offline test steps excluded from formal10. Then remaining14 captures max658000 => total<=1719838/2000000, no retry/probes. Final40 only for independent54 development physical evaluation, holdouts sealed. This is a mid-training preregistered objective revision, not unchanged-objective40. Runtime/evidence smooth_objective_revision1, admission execution_v47_smooth1.


2026-10-08 authorized v47 memory sampling eligibility revision from update21 only. Preserve original batch3 failure: pitch-memory min8.42073423e-5 exceeded historical q05 6.99082877e-5. All7 physical/sync/history/behavior audits valid; empirical q05/q95 from24 mixed A/candidate entries is not a physical bound and7 entries per evolving-policy batch do not guarantee tail coverage. Initial7 original gate pass retained. Subsequent batch3/4 historical RP-memory coverage becomes explicit diagnostic with range/tail gaps/modelID/seeds/original_gate_would_fail; no cumulative-range claim of current-policy coverage. Original velocity coverage, recurrence/128-action/10ms history,finite/bounds,physical/sync,budget and every optimization constraint remain hard. No reward/alpha/PID/4ms/holdout/final40/development acceptance change. Reuse seven current-policy batch3 results and original update20 actor/Adam/RNG/alpha,then only7 scheduled batch4 captures; historical1364092 retained,total maximum1693092/2m. Zero-native boundary/recovery regression passed; offline disposable steps excluded. This revision preserves earlier update11 objective revision and all old failures. New runtime memory_sampling_revision1; admission execution_v47_memory1.


2026-10-08 v47 development startup recovery1 authorized once: 42 valid audited results preserved. dev43 lockedPID/lowX470302 failed before bridge connection/scoring; kernel identifies original PX4 pid583403 SIGSEGV inside libzmq.so.5.2.4, deeper cause unknown. Original failure charge56000 retained,total1957486. No orphan instances; available memory/disk not exhausted. Isolated Python lifecycle fix: pristine pre-reset close does not latch export; absent socket/stream records unsuccessful/unavailable export and permits base cleanup, never claims valid capture or zero counts. Connected export strict unchanged. Child exit identity/returncode/signal and kernel diagnostics saved; runner-local core limit256MiB with existing WSL core handler, no global core setting change. Seven cleanup and7 continuation tests pass,zero native. Exactly dev43_recovery1 same frozen model/PID/sync2/seed,then original44..54; recurrence stops, no further retry. Original42 reused only as evaluation results; matrix54unique required,55attempts incl original failure. Remainingcaps600000,totalmax2557486<=3m. Controller/dynamics/metrics/acceptance unchanged. Accounting known actual lower bounds distinguished from conservative failed-call charge. Runtime development_recovery1; unique independent execution_v47_dev_recovery1.


2026-10-09 v47 postprocess reconciliation: all54 unique development results completed2048 and passed export/full/sync audits;55 attempts include retained dev43 startup failure56000. Total development charge2498802/3m,no reservations;training1671267/2m. Recovery collector exited1 only after capture completion because frozen_design.json/amendment.json were not carried to recovery root. Restored byte-exact original admission-bound metadata and reran unchanged offline review,0new native calls. Original holder/failure/status/ledger-before preserved; final_evaluation_verification.json and complete_review.json authoritative for recovered postprocessing. Performance conditions FAILED; no promotion/heldout. Future evaluation continuations must bind and carry frozen_design and amendment before launch.


## 2026-10-09 V48 first scheduled capture passed

Independent holder2380 (outside observation Job), locked driver624109, first train_b0_p0/noise200_hover_hold/480101/A2428576135 completed2048; explicit export, full original audit, exact1ms synchronization and ownership cleanup passed. NN43457/PID0 including warmup settled to fresh training ledger; second scheduled train_b0_p1 automatically started with47000 reserved. Formal updates0, development0; this is collection-chain validation, not trained-policy acceptance. Original100Hz R/P/Y RMSE0.222708828/0.207432659/0.200250980 deg/s; native0.222720002/0.207432726/0.199611494. Maxgap1000us,history reconstruction error0. Running PX4 /proc/exe hash matched qualified sync2. V47 ledgers unchanged. Evidence: experiments/robustness_v1_20261002/v48_fixed_lr3e5/first_capture_verification.json. Frozen28/40/54 plan continues autonomously; any actual failure stops without retry. No model promoted or holdout accessed.


2026-10-09 authorized V48 smooth_start_revision1: original alpha1 pre-update failure and303638 charge retained; adaptive global smooth rule starts at update1 (previously11), original0.1 hard gate unchanged. Offline paired first-step3e-6/3e-5 passed with shared alpha0.702638918807, ratio10.00013627; four disposable steps excluded. Exact7 paid A trajectories reused, empty Adam3e-5,21remaining captures,40total updates/final40only54dev; original2m/3m budgets,4000/s,PID/sync/phase weights/memory schedule unchanged. Not strictly LR-only versus original V47 process. New status/runtime smooth_start_revision1; original canonical ledgers preserved. Every accepted formal update now journaled to ledger. See docs/V48_SMOOTH_START_REVISION1_20261009.md.

2026-10-09 smooth_start_revision1 real batch0 completed ten accepted formal updates at Adam3e-5. First update alpha0.7026389188070779/KL2.3639283532540662e-5 matches paired offline result; alpha monotonically decreased to0.11014776110586012 by update10. New actor3433787511, SHA2565f6a724e2a95556ce23ff8f439bf70c59b6da1423fc1c4104e83767c141e84dd. Next scheduled train_b1_p0/seed481101 uses this actor; actual native INIT audit records model3433787511,11172NN calls,fault0,not merely a configured path. Ten updates and Adam state/frozen buffers verified; original7 trajectories and303638 charge retained,47000 reserved for new capture. Overall28/40/54 work remains running; A stays current. New authoritative status smooth_start_revision1/training/status.json; canonical ledger remains original training/ledger.json. Evidence first_batch_verification.json and revision_contract.json. No old contract or V47 ledger changed.


2026-10-09 authorized V48 memory_gate_revision2: historical memory q05/q95 diagnostic for ALL batches, not an update21 algorithm stage. Keep min/max/gaps/model/seeds/original_gate_would_fail; velocity, memory finite/bounds/reconstruction, physical/sync/optimization and acceptance unchanged. Global adaptive smooth every update remains effective, no historical11/21 switches. Independent effective_rules.json plus inherited_rule_audit.json distinguish true phases. Restore exact step10 actor3433787511/Adam/RNG, reuse only its7 batch1 captures for11..20; previous14 paid captures counted in28,608641 preserved,14future captures only.9 rule tests and2 disposable recovery steps passed, excluded from formal counts. Original two failures retained. Not strict LR-only; final40-only54 physical evaluation remains unchanged. See docs/V48_MEMORY_GATE_REVISION2_20261009.md.


2026-10-09 authorized V48 backtracking_revision3: fixed scales 2^-i for i=0..12 (minimum1/4096), all original guards retained; additional float64 batch/A bias <=1e-4-1e-10, nonzero actual parameter/action change required; every proposal recorded and only accepted nonzero steps counted. Exhaustion stops without retry. Resume exact17 actor/Adam/RNG/permutations/positions from protected pre18 checkpoint; retain original batch-start10 actor/optimizer/RNG and comparison anchor, never reset allowance at17. Existing1..17 and14captures copied/carried, not rerun;608641 charge retained. Integrated isolated18..20 plus exact replay passed (six disposable updates, zero native/formal); step18 matches diagnosis. Remaining14 captures and18..40, final40-only54 evaluation,2m/3m,holdouts unchanged. Rejection log now accurately says batch-start rollback, while latest accepted checkpoint remains separate. Three explicit revisions: smooth adaptation from1, memory diagnostic all batches, fixed extended backtracking; no strict LR-only causal attribution. Native binaries/physical acceptance unchanged. See backtracking_revision3/offline/admission.json and revision_contract.json.
