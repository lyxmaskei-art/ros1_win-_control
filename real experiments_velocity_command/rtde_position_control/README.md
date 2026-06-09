# RTDE servoJ angle-command real UR3e experiments

This folder contains the RTDE position/angle-command version of the four real UR3e live experiments.

Control semantics:

```text
ZNN computes qdot online every control period.
The adapter integrates q_cmd = q_cmd + qdot * dt.
The robot receives q_cmd through ur_rtde RTDEControlInterface.servoJ(...).
```

This matches the best CoppeliaSim live simulation command semantics more closely than direct speed control, because the simulation executed integrated joint-angle targets.

Files:

- `run_clean_repetitive_tracking_rtde_position.py`
- `run_with_drift_free_rtde_position.py`
- `run_without_drift_free_rtde_position.py`
- `run_mild_disturbance_rtde_position.py`
- `rtde_realtime_adapter.py`

Important implementation details:

- No raw `30002` URScript streaming is used in this folder.
- The old socket-based `servoj(...)` transport has been removed from these scripts.
- `rtde_realtime_adapter.py` uses `RTDEControlInterface` and `RTDEReceiveInterface`.
- Each command period calls `initPeriod()`, sends `servoJ(...)`, then waits in `waitPeriod(...)` through the script's synchronous-trigger wrapper.
- `q_cmd` is kept continuous inside the adapter; it is not reset to feedback at every step.
- `servoStop()` is called when stopping the run.

Default real-control parameters:

```text
real_command_backend = rtde_servoj
trajectory_name = circle
heart_scale = 0.0175      # circle radius = 4 * heart_scale = 0.07 m, diameter = 14 cm
tau / servoj_t = 0.005 s
servoj_lookahead_time = 0.05 s
servoj_gain = 500
solver_gamma = 4352
activation_power = 0.8
dlccznn_inner_steps = 60
command_filter_alpha = 1.0
```

Ubuntu setup:

```bash
python3 -m pip install ur_rtde
export UR3E_ROBOT_IP=192.168.126.10
```

Recommended first validation before a full circle:

```bash
python3 run_clean_repetitive_tracking_rtde_position.py \
  --live-backend ros_real_ur3e \
  --ur3e-robot-ip "$UR3E_ROBOT_IP" \
  --duration 2 \
  --skip-plots
```

Then run the 20 s circle:

```bash
python3 run_clean_repetitive_tracking_rtde_position.py \
  --live-backend ros_real_ur3e \
  --ur3e-robot-ip "$UR3E_ROBOT_IP" \
  --duration 20
```

Use this folder first when comparing against the previous CoppeliaSim live results.
