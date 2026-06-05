# 第 4.5 节真实实验代码说明

本文件夹整理了论文第 4.5 节计划使用的四组真实机械臂验证实验代码。当前代码来自已经调通的 live 实验脚本，目的是方便切换到 Ubuntu 系统后迁移到真实机械臂平台上运行。

重要说明：这些脚本目前仍保留 CoppeliaSim Remote API 的通信接口。真实实验前，必须把仿真通信层替换为 Ubuntu 端真实机械臂驱动接口。不能把 CoppeliaSim 仿真结果当作真实机械臂实验结果使用。

## 实验总结构

第 4.5 节按以下四组实验展开：

1. clean repetitive tracking 实机主实验；
2. with vs without drift-free constraint 实机消融；
3. real-time feasibility 实时性验证；
4. mild robustness 轻微扰动鲁棒性验证。

这四组实验共同支撑一个核心结论：JDF-DLCCZNN 在采样控制链路中能够同时保持 position tracking accuracy 和 joint drift-free behavior。

## 1. Clean Repetitive Tracking

文件夹：

```text
01_clean_repetitive_tracking/
```

主脚本：

```text
run_clean_repetitive_tracking.py
```

参数文件：

```text
final_live_best_tau0005.json
```

实验目的：

- 在真实机械臂上验证 clean 条件下的重复位置跟踪；
- 记录 desired 和 measured end-effector trajectory；
- 记录 position tracking error、joint angles、joint velocities；
- 记录相对初始关节角的 joint drift。

推荐主轨迹：

```text
circle
```

原因是前面第 4.2、4.3、4.4 节主要围绕 circle 展开，4.5 继续使用 circle 可以保持实验逻辑统一。除非后续明确要补其他轨迹，否则先用 circle 做主实验。

## 2. With vs Without Drift-Free Constraint

文件夹：

```text
02_drift_free_ablation/
```

脚本：

```text
run_with_drift_free.py
run_without_drift_free.py
```

参数文件：

```text
final_live_best_tau0005.json
```

实验目的：

- 对比加入 drift-free constraint 的 JDF-DLCCZNN 与去掉 drift-free 机制后的版本；
- 尽量保持同一轨迹、同一初始关节角、同一采样周期和相同控制参数；
- 重点观察 final joint drift、accumulated joint drift norm 和 mean position error。

这组实验是第 4.5 节最重要的消融实验。它要证明 drift-free constraint 不是装饰项，而是在重复跟踪运动中抑制关节漂移的关键机制。

建议先低速、小幅度运行 `run_with_drift_free.py`，确认真实机械臂运行稳定后，再运行 `run_without_drift_free.py`。不要一开始就直接做完整幅值的 no-drift-free 实验。

## 3. Real-Time Feasibility

文件夹：

```text
03_real_time_feasibility/
```

脚本：

```text
benchmark_step_time.py
```

实验目的：

- 在 Ubuntu 真实实验机器上测量每个 control cycle 的 computation time；
- 报告 mean、median、95th percentile、99th percentile、maximum computation time；
- 统计 control deadline miss count；
- 验证控制器更新是否基本能在采样周期内完成。

目前默认采样周期为：

```text
tau = 0.005 s
```

对应控制周期 deadline 为：

```text
5 ms per control cycle
```

计时范围应该只包住 controller update 本身，不应包含绘图、文件写入、结果后处理或人工等待时间。论文中可以把这一组结果写成 sampled real-time control feasibility 的证据。

## 4. Mild Robustness

文件夹：

```text
04_mild_robustness/
```

主脚本：

```text
run_mild_disturbance.py
```

参数文件：

```text
final_live_best_tau0005.json
```

实验目的：

- 验证真实控制链路中加入轻微扰动后，JDF-DLCCZNN 是否仍能保持可接受的位置跟踪精度和关节漂移抑制效果；
- 扰动建议优先采用软件层面的轻微扰动，例如 desired velocity disturbance、command-level disturbance 或 measurement-noise injection；
- 不建议一开始加入危险的物理外力扰动。

推荐第一版扰动：

```text
n(t) = 0.5t
```

如果真实机械臂上这个扰动幅值偏大，应先缩小幅值。安全优先，论文中只需要 mild robustness，不需要证明极端扰动下仍稳定。

## Ubuntu 迁移步骤

### 1. 建立 Python 环境

在 Ubuntu 端进入本文件夹后，建议先建立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy
```

如果真实机械臂 SDK 需要额外依赖，例如 ROS、RTDE、厂商 Python SDK，请按真实平台要求安装。

### 2. 替换 CoppeliaSim 通信层

每个脚本中需要重点搜索并替换以下函数或调用：

```text
connect_to_coppeliasim
simxSetJointTargetPosition
read_joint_positions_fast
read_object_position_fast
reset_simulation_with_toolbar_equivalent
```

真实机械臂端应替换为类似接口：

```text
connect_to_robot
send_joint_position_command 或 send_joint_velocity_command
read_joint_positions
read_end_effector_position
move_to_initial_configuration
```

如果真实机械臂只能可靠执行关节位置命令，则优先保持当前位置控制口径。若后续要改成速度控制，需要同步检查关节速度限制、积分更新和安全保护。

### 3. 保持实验指标定义一致

四组实验中的指标定义必须保持一致：

```text
position error = measured end-effector position - desired end-effector position
joint drift = measured joint position - initial measured joint position
computation time = one controller update 的墙钟耗时
```

其中 joint drift 必须相对真实机械臂开始实验后的初始测量关节角计算，而不是相对仿真或理论初始值计算。

### 4. 推荐运行顺序

建议按以下顺序进行：

```text
1. 短时 clean circle 低速试运行；
2. 完整 clean circle 主实验；
3. with drift-free constraint 实验；
4. without drift-free constraint 消融实验；
5. real-time computation timing；
6. mild robustness 实验。
```

不要一开始就运行 no-drift-free 或扰动实验。先确认 clean drift-free 实验安全稳定。

## 每次实验必须保存的数据

每次真实实验至少保存以下原始数据：

```text
time
desired_position
measured_position
position_error
joint_position
joint_velocity 或 commanded_joint_increment
joint_drift
computation_time_per_cycle
controller_parameters
initial_joint_configuration
terminal_joint_configuration
```

建议同时保存：

```text
robot_model
control_frequency
sampling_period
trajectory_name
trajectory_period
total_duration
disturbance_type
disturbance_parameters
experiment_timestamp
```

这些信息后续用于论文表格、图注和可复现性说明。

## 安全注意事项

- 真实机械臂实验先使用小幅值、低速度轨迹。
- 运行前确认关节限位、速度限位和工作空间边界。
- 保持急停按钮可用。
- no-drift-free 消融实验可能导致更明显的关节漂移，必须先低风险测试。
- mild robustness 实验优先使用软件扰动，不要直接施加不可控物理外力。
- 如果出现关节接近限位、速度突增或末端轨迹异常，应立即停止实验。

## 第 4.5 节写作逻辑

真实实验完成后，第 4.5 节建议按以下顺序写：

1. 说明真实实验平台、采样周期、轨迹、初始关节角和安全约束；
2. 展示 clean repetitive tracking 主结果；
3. 展示 with vs without drift-free constraint 消融结果；
4. 展示 real-time feasibility 计算耗时结果；
5. 展示 mild robustness 结果。

这一节不要再重复第 4.4 节的 solver comparison。4.5 的重点是证明 JDF-DLCCZNN 在真实控制链路中的可执行性、跟踪精度和关节漂移抑制能力。
