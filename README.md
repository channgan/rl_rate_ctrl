# rl_rate_ctrl

使用 Stable-Baselines3 PPO 学习多旋翼角速度内环，运行于 PX4 v1.17.0 + Gazebo Harmonic。

PX4原生位置、速度和姿态环生成角速度与总推力需求；策略只输出三轴归一化力矩。总推力指令原样交给PX4原生控制分配器，分配后的电机指令驱动Gazebo。此仓库不含完整PX4源码、旧实验入口、模型或训练日志。

## 当前接口

- 观测9维：当前带噪角速度3维、目标减实际角速度3维、上一步已执行的三轴力矩3维。角速度及误差用rad/s除以5。
- 动作3维：FRD坐标系归一化力矩，每轴范围[-1,1]，不是N·m。
- Actor/Critic为独立64×64 ReLU网络，共用同样的9维观测，无特权Critic。
- 策略步长10ms，Gazebo物理步长1ms。回合最长30s，先由原生PX4起飞并稳定在5m，再接管内环。
- 前后期奖励、航点和成功条件详见 [TRAINING_PARAMETERS.md](TRAINING_PARAMETERS.md)。

当前训练未运行；安装和测试不会自动开始训练。训练表现尚未验证为可部署的实机控制器。

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

## 显式开始训练

```bash
"$PY" scripts/train_rate_only.py --run runs/early --phase early --steps 300000
```

从零探索；已有目录会被拒绝，避免覆盖日志。完成后按完整4096步rollout预计停在303104步。手动续训示例：

```bash
"$PY" scripts/train_rate_only.py --run runs/late --phase late --steps 3000000 --resume runs/early/actor_critic.zip
```

也可以在前期训练运行时，在另一个终端执行以下管理器，等待前期成功结束再自动启动后期：

```bash
"$PY" scripts/extend_rate_only_exploration.py --source runs/early --run runs/late --total 3303104 --phase late
```

使用默认运行目录时，预计累计3305472步结束。自定义运行路径请为训练入口传入`--runtime`；自动管理器目前使用默认运行目录，定制路径场景请使用手动续训。

## 监控与评估

```bash
"$PY" scripts/serve_tensorboard_local.py --logdir runs/early/tensorboard_raw --host 127.0.0.1 --port 6006 --load_fast=false
"$PY" scripts/evaluate_deterministic_recent.py --run runs/late --runtime "$HOME/rl_rate/runtime.free_flight.json"
```

TensorBoard地址为<http://127.0.0.1:6006/>，奖励曲线还原为原始奖励。每40000步保存包含该步的完整回合，绘制三轴角速度、三轴力矩和四电机分配输出，并标记实际航点切换。确定性评估关闭策略动作采样噪声，保留传感器噪声与电机动态，结果写入所选run的`performance_analysis/`。

`rate_rl`保留当前训练器依赖的内部接口检查和共享组件；公开训练入口固定启用当前9维/3力矩方案。没有打包旧PWM训练入口、历史实验文档、旧模型或旧Git提交历史。
