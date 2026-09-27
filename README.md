# rl_rate_ctrl

使用 Stable-Baselines3 PPO 学习多旋翼角速度内环，运行于 PX4 v1.17.0 + Gazebo Harmonic。

PX4原生位置、速度和姿态环生成角速度与总推力需求；策略只输出三轴归一化力矩。总推力指令原样交给PX4原生控制分配器，分配后的电机指令驱动Gazebo。此仓库不含完整PX4源码、旧实验入口、模型或训练日志。

## 当前接口

- 观测9维：当前带噪角速度3维、目标减实际角速度3维、上一步已执行的三轴力矩3维。角速度及误差用rad/s除以5。
- 动作3维：FRD坐标系归一化力矩，每轴范围[-1,1]，不是N·m。
- Actor/Critic为独立64×64 ReLU网络，共用同样的9维观测，无特权Critic。
- 策略步长10ms，Gazebo物理步长1ms。回合最长2048步/20.48s，先由原生PX4起飞并稳定在5m，再接管内环。
- 前后期奖励、航点和成功条件详见 [TRAINING_PARAMETERS.md](TRAINING_PARAMETERS.md)。

安装和测试不会自动开始训练，运行状态应检查进程及日志。训练表现尚未验证为可部署的实机控制器。

## 安装

推荐Ubuntu 22.04（包括WSL2），预先安装Gazebo Harmonic、PX4构建工具、CMake、Python 3.10+及venv。PX4基线固定为`d6f12ad1c4f70ad3230afd7d86e971421e02fef4`（v1.17.0）。安装脚本创建独立PX4工作目录并应用所需桥接改动，不修改其他PX4仓库。

```bash
git clone git@github.com:channgan/rl_rate_ctrl.git
cd rl_rate_ctrl
bash scripts/setup.sh
```

默认运行目录为`$HOME/rl_rate`，可以通过`RL_RUNTIME_DIR`覆盖。运行目录必须与已有训练环境隔离。`requirements.lock.txt`记录来源环境的依赖快照；安装入口以`pyproject.toml`为准。

```bash
PY="$HOME/rl_rate/.venv/bin/python"
"$PY" -m pytest -q
"$PY" scripts/validate_native_torque.py --runtime "$HOME/rl_rate/runtime.free_flight.json" --out validation/interface --steps 100
```

第二个命令会启动仿真，使用固定PI控制器验证接口，不是PPO训练。仓库打包验证仅执行Python回归和脚本语法检查，未在全新机器重新构建整个PX4/Gazebo工具链。

## 预览与启动

先预览实际命令。`--dry-run`只解析参数并输出计划，不创建运行目录、不加载模型、不启动仿真：

```bash
"$PY" scripts/train_rate_only.py --run runs/experiment --steps 3500000 --dry-run
"$PY" scripts/train_rate_only.py --run runs/experiment --steps 3500000
```

新训练不传`--resume`；旧目录会被拒绝。当前默认单环境，每轮采样4096步，batch2048，最多10个epochs；每回合2048步/20.48秒，可跨采样轮次。学习率按KL在3e-5～3e-4之间调整，熵0.01、clip0.2，前后期优化参数和跟踪奖励门槛相同，不自动转阶段。总步数按完整rollout向上取整。

默认配方集中在`rate_rl/defaults.py`；实际启动配置保存在各run的`config.json`，进程状态在`job.json`。公式与当前约定见[训练参数](TRAINING_PARAMETERS.md)，模块职责见[架构说明](docs/ARCHITECTURE.md)。不要用参数文档推断训练是否正在运行。

续训需要显式指定检查点与新的运行目录，`--steps`表示追加采样步数：

```bash
"$PY" scripts/train_rate_only.py --run runs/continued --steps 1000000 \
  --resume runs/experiment/checkpoints/ppo_1757184_steps.zip --dry-run
```

确认计划后移除`--dry-run`执行。加载时仍校验观测、动作、奖励和仿真资产契约；旧检查点若奖励不匹配，不能仅改元数据绕过校验。旧30秒模型缩短到20.48秒须显式`--allow-episode-length-change`，且仅允许该时长及对应剩余时间惩罚上限变化。续训恢复策略、优化器与保存的训练随机状态，仿真重新起飞，不恢复空中的物理状态。

历史并行实现仍支持`--n-envs 2`或`4`，相应每环境采样2048或1024步，默认不启用；2/4环境检查点之间可转换。训练通常使用实例41（并行41～44），独立评估使用45。历史阶段衔接脚本`extend_rate_only_exploration.py`仅在显式运行时生效，并传递源任务的采样布局与runtime路径；当前统一参数训练无需启动它。

## 监控与评估

```bash
"$PY" scripts/serve_tensorboard_local.py --logdir runs/experiment/tensorboard_raw --host 127.0.0.1 --port 6006 --load_fast=false
"$PY" scripts/evaluate_deterministic_recent.py --run runs/experiment --runtime "$HOME/rl_rate/runtime.free_flight.json"
```

TensorBoard地址为<http://127.0.0.1:6006/>，奖励曲线还原为原始奖励。每40000步保存包含该步的完整回合，绘制三轴角速度、三轴力矩和四电机分配输出，并标记实际航点切换。确定性评估关闭策略动作采样噪声，保留传感器噪声与电机动态，结果写入所选run的`performance_analysis/`。

`rate_rl`保留当前训练器依赖的内部接口检查和共享组件；公开训练入口固定启用当前9维/3力矩方案。没有打包旧PWM训练入口、历史实验文档、旧模型或旧Git提交历史。
