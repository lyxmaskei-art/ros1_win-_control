# UR3e 实机验证代码收尾说明

本目录包含 4 个用于 UR3e 实机验证的 live 控制脚本。当前版本已经把默认实验配置改成这次最终确认的 14 cm 圆轨迹参数，并修复了旧 ROS 位置 topic 传输口径导致的实机抖动风险。

## 当前结论

推荐真机主实验使用：

```text
clean 或 with_drift_free
```

不推荐把 `without_drift_free` 作为主方案。它在 CoppeliaSim live 中末端误差接近，但 final joint drift 明显变大，适合作为对照组。

`mild_disturbance` 用于扰动鲁棒性验证，不作为 clean 精度主结果。

## 四个脚本

```text
run_clean_repetitive_tracking_servo_like.py
run_with_drift_free_servo_like.py
run_without_drift_free_servo_like.py
run_mild_disturbance_servo_like.py
```

四个脚本默认均使用 14 cm 普通圆轨迹：

```text
trajectory_name = circle
heart_scale = 0.0175
circle radius = 4 * heart_scale = 0.07 m
circle diameter = 0.14 m
duration = 20 s
trajectory_period = 10 s
tau = 0.005 s
```

## 最终控制参数

三个带 drift-free 的实验默认参数：

```text
method = method2_dlccznn
task_gain = 160
drift_gain = 10
solver_gamma = 4352
activation_power = 0.8
activation_exp_clip = 4.0
drift_feedback_mode = nonlinear
dlccznn_inner_steps = 60
```

`run_without_drift_free_servo_like.py` 强制保持：

```text
drift_gain = 0
```

即使命令行误传 `--drift-gain 10`，该脚本也不会启用 drift-free 项，避免对照实验被污染。

`run_mild_disturbance_servo_like.py` 默认使用：

```text
internal_disturbance = linear
disturbance_scale = 1.0
```

即文献式线性扰动口径。

## 真机传输口径

默认 live 后端为：

```text
--live-backend real_ur3e
```

默认真机命令后端为：

```text
--real-command-backend servoj
```

URScript `servoj` 默认参数：

```text
ur3e_servoj_t = 0.005
ur3e_servoj_lookahead_time = 0.05
ur3e_servoj_gain = 500
feedback_wait_timeout = 0.03
```

这里 `servoj_t=0.005` 与控制周期 `tau=0.005` 对齐，目标是每 5 ms 单点发送一次目标关节角。

旧的 ROS position topic 方式仍保留为兼容选项：

```text
--real-command-backend topic_position
```

但它不建议作为实机精度证据路径，因为旧写法等价于：

```text
q_target = q_feedback + qdot * tau
publish q_target to /pos_joint_group_controller/command
```

该路径是异步位置 topic，容易受到旧反馈、队列延迟、控制器插值和 ROS 调度影响，不等价于 CoppeliaSim synchronous live，也不等价于 URScript `servoj`。

## 推荐真机运行命令

先替换 `<UR3E_IP>` 为 UR 控制柜 IP。

clean 主实验：

```bash
python run_clean_repetitive_tracking_servo_like.py \
  --ur3e-robot-ip <UR3E_IP> \
  --skip-plots
```

with drift-free 主实验：

```bash
python run_with_drift_free_servo_like.py \
  --ur3e-robot-ip <UR3E_IP> \
  --skip-plots
```

without drift-free 对照实验：

```bash
python run_without_drift_free_servo_like.py \
  --ur3e-robot-ip <UR3E_IP> \
  --skip-plots
```

mild disturbance 扰动实验：

```bash
python run_mild_disturbance_servo_like.py \
  --ur3e-robot-ip <UR3E_IP> \
  --skip-plots
```

如果要显式写全参数，等价命令为：

```bash
python run_clean_repetitive_tracking_servo_like.py \
  --live-backend real_ur3e \
  --experiment single \
  --method method2_dlccznn \
  --trajectory-name circle \
  --duration 20 \
  --offline-duration 20 \
  --trajectory-period 10 \
  --heart-scale 0.0175 \
  --tau 0.005 \
  --task-gain 160 \
  --drift-gain 10 \
  --solver-gamma 4352 \
  --activation-power 0.8 \
  --activation-exp-clip 4.0 \
  --drift-feedback-mode nonlinear \
  --dlccznn-inner-steps 60 \
  --real-command-backend servoj \
  --ur3e-robot-ip <UR3E_IP> \
  --ur3e-servoj-t 0.005 \
  --ur3e-servoj-lookahead-time 0.05 \
  --ur3e-servoj-gain 500 \
  --feedback-wait-timeout 0.03 \
  --skip-plots
```

## CoppeliaSim live 验证

如果只想在 CoppeliaSim ZMQ live 中复核，不连接真机：

```bash
python run_clean_repetitive_tracking_servo_like.py \
  --live-backend coppelia_zmq \
  --skip-plots
```

CoppeliaSim ZMQ 端口默认为：

```text
23000
```

本地完整 20 s live 验证结果中，clean / with drift-free 的典型结果为：

```text
mean position error ~= 5.61e-4 m
final joint drift ~= 2.18e-3 rad
```

without drift-free 的末端误差接近，但 final joint drift 约为：

```text
7.70e-1 rad
```

因此不建议作为实机主方案。

## 实机运行前检查

1. UR 控制柜处于 Remote Control 模式。
2. 机器人初始关节角接近：

```text
(0, -90, 90, 0, 90, 0) deg
```

3. ROS 能持续收到 `/joint_states`。
4. 确认 6 个关节名顺序为：

```text
shoulder_pan_joint
shoulder_lift_joint
elbow_joint
wrist_1_joint
wrist_2_joint
wrist_3_joint
```

如实际 joint name 不同，用环境变量覆盖：

```bash
export UR3E_JOINT_NAMES="joint1,joint2,joint3,joint4,joint5,joint6"
```

5. 第一次真机运行建议先保留急停旁站，观察一圈后再继续完整 20 s。

## 诊断字段

每次运行的 summary 中会记录：

```text
real_transport_diagnostics.command_backend
real_transport_diagnostics.command_dt_s
real_transport_diagnostics.feedback_age_s
real_transport_diagnostics.feedback_wait_timeouts
real_transport_diagnostics.target_or_velocity_step_norm
real_transport_diagnostics.script_reconnects
```

如果真机仍出现抖动，优先检查：

```text
command_dt_s 是否稳定接近 0.005
feedback_age_s 是否明显滞后
feedback_wait_timeouts 是否非零
target_or_velocity_step_norm 是否存在突增
```

这几个字段可以区分抖动来自控制参数，还是来自 ROS/网络/URScript 传输时序。
