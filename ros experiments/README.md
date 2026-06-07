# ROS 实机 UR3e 实验排查与运行说明

这个目录是 `ros-experiments` 分支里用于 Ubuntu/ROS1 实机验证 UR3e 轨迹跟踪的代码集合。当前重点不是继续盲目调参，而是把“仿真效果好、实机抖动且精度下降”的原因拆开定位。

## 当前判断

根据 2026-06-07 上传的 UR3e 实机结果，`/joint_states` 预检中位频率约 `452-459 Hz`，ROS stamp age p95 约 `0.76-0.87 ms`，正式运行 5 ms 外环 deadline miss 基本为 `0%`。因此本轮数据里，主因不是 UR3e 反馈频率不够，而是命令接口和底层执行语义不匹配，再叠加命令加速度/jerk 偏尖。

实机效果比 Windows/CoppeliaSim live 仿真差一个数量级，最可能不是单一参数问题，而是下面几类因素叠加：

1. **控制接口语义不匹配**

   算法内部更接近“速度/增量控制”：先算 `theta_dot_next`，再积分得到 `theta_next`。原 ROS 实机路径把高频 `theta_next` 位置数组发给 `/pos_joint_group_controller/command`。仿真同步步进可以吃下这种命令，但真实 UR 控制器还有插值、伺服动态、通信延迟和安全限幅，200 Hz 密集位置流容易表现为抖动。

2. **控制器执行层滞后**

   结果中 `position_array` / `joint_trajectory` 的命令速度 p95 约 `0.89 rad/s`，但反馈速度 p95 只有 `0.08-0.12 rad/s`，差了约 `7-11` 倍；`velocity_array` 则基本同量级。这说明脚本没掉频，是真实控制器没有按 5 ms 单点位置/轨迹目标同步跟上。

3. **ROS 闭环时序不稳定**

   原 `03_real_time_feasibility/benchmark_step_time.py` 主要测 `controller.step`，没有测完整闭环周期。真实抖动可能来自 `/joint_states` 反馈过旧、ROS1/Python 调度抖动、publish 耗时、`rospy.Rate` sleep 误差或 deadline miss。

4. **命令速度/加速度/jerk 尖峰**

   `task_gain`、`solver_gamma`、`drift_gain`、`theta_dot_limit`、`max_command_step`、`max_command_accel` 是耦合的。某些组合会让单步位置变化不大，但关节速度变化很急，真机上就会抖。

5. **TCP / frame / FK 模型误差**

   如果只用 `/joint_states` 加本地 DH/FK 计算 TCP，日志里可能显示“跟踪还行”，但真实工具端因为 TCP offset、base/tool frame 或工具安装误差而偏离。需要可选真实 TCP pose 来交叉验证。

6. **反馈频率撑不起控制频率**

   `--tau 0.005` 对应 200 Hz。如果 `/joint_states` 实际频率明显低于 200 Hz，控制器就是在用旧状态闭环，继续调增益没有意义。

## 已做的代码改动

主要改动集中在：

- `_ros_real_ur3e.py`
- `03_real_time_feasibility/benchmark_step_time.py`
- `README.md`

新增工具：

- `ros_real_preflight_check.py`
- `analyze_ros_real_diagnostics.py`
- `compare_ros_real_runs.py`
- `collect_ros_real_artifacts.py`
- `ROS_REAL_UR3E_JITTER_AUDIT.md`

### 1. 实机安全预设

新增 `--real-safe-preset`，会为未显式指定的参数填入低抖动默认值：

```bash
--tcp-offset 0,0,0.145
--task-gain 240
--solver-gamma 30
--drift-gain 2
--theta-dot-limit 0.3
--max-command-accel 8
--max-command-jerk 80
```

你仍然可以在命令行显式覆盖其中任意值。

### 2. 三种命令接口 A/B

新增 `--command-mode`：

```bash
--command-mode position_array
--command-mode joint_trajectory
--command-mode velocity_array
```

含义：

- `position_array`：保留原来的 `std_msgs/Float64MultiArray` 位置数组接口。
- `joint_trajectory`：发布单点 `trajectory_msgs/JointTrajectory`，保持逐周期实时控制口径。
- `velocity_array`：直接发布受限后的关节速度命令，更接近算法输出，但必须确认机器人上有对应 velocity controller。

`velocity_array` 模式会在正常结束、异常栈展开、ROS shutdown 和进程退出时尝试发送零速度。

### 3. 非运动 preflight

新增：

```bash
python3 ros_real_preflight_check.py
```

它不会让机器人运动，会检查：

- `/joint_states` 是否可用
- `/joint_states` 实际到达频率
- ROS stamp age 是否 stale
- controller manager 里哪些 controller loaded/running
- command topic 是否有 subscriber
- topic 类型是否匹配当前 command mode
- `preflight_verdict.status` 是否为 `pass` / `warn` / `fail`

正式运动 run 在 `preflight_verdict.status == fail` 时会拒绝启动。

### 4. 整周期诊断

每次 ROS real run 会保存：

```text
ros_real_cycle_diagnostics.csv
ros_real_cycle_diagnostics.npz
ros_real_cycle_diagnostics_summary.json
```

诊断字段包括：

- `cycle_period_s`
- `cycle_work_s`
- `state_read_s`
- `state_age_s`
- `controller_step_s`
- `publish_s`
- `sleep_s`
- `command_step_norm_rad`
- `command_step_max_abs_rad`
- `command_velocity_norm_rad_s`
- `command_accel_norm_rad_s2`
- `feedback_velocity_norm_rad_s`
- `deadline_miss`
- `period_overrun`

如果启用真实 TCP pose，还会记录：

- `tcp_pose_used`
- `tcp_pose_used_ratio`
- `tcp_pose_age_s`
- `tcp_fk_error_norm_m`

### 5. 可选真实 TCP pose

如果 UR driver、外部 tracker 或其它节点能发布真实 TCP 位姿，可以传：

```bash
--tcp-pose-topic /tool_pose
```

要求该 topic 是：

```text
geometry_msgs/PoseStamped
```

启用后：

- `actual_positions` 和 `mean_position_error_m` 使用真实 TCP pose
- `fk_positions` 和 `mean_fk_position_error_m` 仍然保存本地 FK 结果
- analyzer 会检查 `Measured TCP and local FK disagree`

如果 `tcp_fk_error_norm_m` 很大，先检查 TCP offset、base/tool frame 和测量 topic 坐标系，不要先调增益。

## Ubuntu/ROS 环境

示例：

```bash
source /home/lyx/文档/catkin_ws/devel_isolated/setup.bash
export ROS_IP=192.168.126.11
export ROS_HOSTNAME=
export ROS_MASTER_URI=http://192.168.126.11:11311
```

请根据你的实际 catkin workspace 路径和网络配置调整。

## 推荐实机流程

### Step 1：先跑非运动 preflight

优先检查 trajectory controller：

```bash
python3 ros_real_preflight_check.py \
  --real-safe-preset \
  --command-mode joint_trajectory \
  --trajectory-command-topic /scaled_pos_joint_traj_controller/command \
  --require-pass
```

重点看输出里的：

- `preflight_verdict.status`
- `preflight_verdict.issues`
- `command_mode_recommendations`
- `joint_state_rate_probe.median_rate_hz`
- `joint_state_rate_probe.ros_stamp_age_s`

如果 `preflight_verdict.status == fail`，不要让机器人运动，先按 `issues` 里的 action 修。

如果 `joint_state_rate_probe.median_rate_hz` 明显低于 200 Hz，不要直接跑 `--tau 0.005`，先提高反馈频率或测试：

```bash
--tau 0.01
```

### Step 2：跑 5 秒短时 A/B

先跑原始位置数组接口：

```bash
python3 01_clean_repetitive_tracking/run_clean_repetitive_tracking.py \
  --experiment single \
  --method method2_dlccznn \
  --trajectory-name circle \
  --duration 5 \
  --real-safe-preset \
  --command-mode position_array \
  --tau 0.005
```

再跑 velocity 接口。2026-06-07 的结果里，`velocity_array` 是唯一让命令速度和反馈速度进入同一量级的模式：

```bash
python3 01_clean_repetitive_tracking/run_clean_repetitive_tracking.py \
  --experiment single \
  --method method2_dlccznn \
  --trajectory-name circle \
  --duration 5 \
  --real-safe-preset \
  --command-mode velocity_array \
  --velocity-command-topic /joint_group_vel_controller/command \
  --theta-dot-limit 0.25 \
  --max-command-accel 6 \
  --max-command-jerk 60 \
  --tau 0.005
```

如果只能使用 trajectory controller，再跑单点 trajectory 接口；这仍然是实时控制，但要重点看命令速度/反馈速度比：

```bash
python3 01_clean_repetitive_tracking/run_clean_repetitive_tracking.py \
  --experiment single \
  --method method2_dlccznn \
  --trajectory-name circle \
  --duration 5 \
  --real-safe-preset \
  --command-mode joint_trajectory \
  --trajectory-command-topic /scaled_pos_joint_traj_controller/command \
  --trajectory-command-duration 0.02 \
  --tau 0.005
```

### Step 3：分析单次 run

```bash
python3 analyze_ros_real_diagnostics.py <run-directory>
```

输出：

```text
ros_real_diagnostic_analysis.json
ros_real_diagnostic_analysis.md
```

优先处理 high severity：

- `Whole-loop deadline misses`
- `ROS loop period jitter`
- `Stale joint-state feedback`
- `Command acceleration spikes`

如果 timing 和 command smoothness 都正常，但机器人仍然抖，优先怀疑 command interface 或 TCP/model mismatch。

### Step 4：比较多个接口

```bash
python3 compare_ros_real_runs.py \
  <position_array_run> \
  <joint_trajectory_run> \
  <velocity_array_run>
```

输出：

```text
ros_real_ab_comparison.csv
ros_real_ab_comparison.md
```

比较指标包括：

- high/medium finding 数量
- deadline miss ratio
- period overrun ratio
- state age
- command acceleration
- TCP-FK error
- mean tracking error
- final joint drift

### Step 5：打包证据

跑完 preflight 和短时 A/B 后，打包轻量诊断证据：

```bash
python3 collect_ros_real_artifacts.py \
  <position_array_run> \
  <joint_trajectory_run> \
  <velocity_array_run> \
  --output-zip ros_real_triage.zip
```

zip 里包含：

- preflight report
- summary JSON
- cycle diagnostics
- analyzer 输出
- A/B comparison
- manifest

只有需要 `.npz` history 或图片时才加：

```bash
--include-heavy
```

## 如何解释典型结果

### 1. deadline miss / period jitter 高

说明完整 ROS 闭环周期不稳。先不要调增益，优先：

- 测 `--tau 0.01`
- 减少其它 ROS 节点负载
- 检查 `/joint_states` 发布频率
- 检查网络和 driver 负载

### 2. state age 高

说明反馈旧。外环用旧状态控制会明显放大抖动。先修反馈频率或降低控制频率。

### 3. command acceleration 高

说明命令变化太急。优先：

```bash
--max-command-accel 6
--max-command-jerk 60
--theta-dot-limit 0.25
--task-gain 160
--solver-gamma 20
--drift-gain 1
```

再逐步回到 `--real-safe-preset`。

### 4. command velocity 远高于 feedback velocity

如果 `Command velocity greatly exceeds feedback velocity` 是 high finding，说明脚本输出的速度型命令远快于底层控制器实际执行速度。此时不要继续只调高/调低增益，优先：

- 使用 `velocity_array`
- 降低 `--theta-dot-limit`
- 降低 `--max-command-accel`
- 加 `--max-command-jerk`
- 如果必须用 trajectory controller，坚持单点发送并检查 `Command velocity greatly exceeds feedback velocity`；若仍滞后，说明这个控制器接口不适合当前实时速度型算法

### 5. position_array 抖，joint_trajectory 稳

说明原高频位置数组接口很可能是主要问题。后续应优先使用 trajectory controller，或者进一步接入更合适的 velocity/servo 接口。

### 6. TCP-FK error 大

说明本地 FK 和真实 TCP 不一致。先修：

- TCP offset
- base frame
- tool frame
- 外部 pose topic 坐标系

不要先调控制器增益。

## 推荐低风险初始 run

如果第一次上真机不放心，可以比 `--real-safe-preset` 更保守：

```bash
python3 01_clean_repetitive_tracking/run_clean_repetitive_tracking.py \
  --experiment single \
  --method method2_dlccznn \
  --trajectory-name circle \
  --duration 5 \
  --real-safe-preset \
  --task-gain 160 \
  --solver-gamma 20 \
  --drift-gain 1 \
  --command-mode velocity_array \
  --velocity-command-topic /joint_group_vel_controller/command \
  --theta-dot-limit 0.25 \
  --max-command-accel 6 \
  --max-command-jerk 60 \
  --tau 0.005
```

只有在诊断显示 timing 稳、反馈新、命令加速度平滑后，再逐步提高参数。

## 当前最重要的结论

这套代码现在的目标不是“直接证明某个参数最好”，而是让实机实验能回答下面几个问题：

1. 抖动是不是 ROS 闭环周期/反馈 stale 导致？
2. 抖动是不是高频 position array 接口导致？
3. 抖动是不是命令加速度尖峰导致？
4. 精度下降是不是 TCP/FK/frame mismatch 导致？
5. 哪个 command mode 在真机上最稳定？

拿到 `ros_real_triage.zip` 后，就可以按证据继续做下一轮参数或接口修改。
