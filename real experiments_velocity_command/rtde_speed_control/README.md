# RTDE speedJ velocity-command real UR3e experiments

This folder contains the RTDE pure velocity-command version of the four real UR3e live experiments.

Control semantics:

```text
ZNN computes qdot online every control period.
The robot receives qdot directly through ur_rtde RTDEControlInterface.speedJ(...).
```

This is the cleanest velocity-level implementation of the ZNN controller, but it is not the same command semantics as the best CoppeliaSim live runs, which integrated qdot into joint-angle targets before sending.

Files:

- `run_clean_repetitive_tracking_rtde_speed.py`
- `run_with_drift_free_rtde_speed.py`
- `run_without_drift_free_rtde_speed.py`
- `run_mild_disturbance_rtde_speed.py`
- `rtde_realtime_adapter.py`

Important implementation details:

- No raw `30002` URScript streaming is used in this folder.
- The old socket-based `speedj(...)` transport has been removed from these scripts.
- `rtde_realtime_adapter.py` uses `RTDEControlInterface` and `RTDEReceiveInterface`.
- Each command period calls `initPeriod()`, sends `speedJ(...)`, then waits in `waitPeriod(...)` through the script's synchronous-trigger wrapper.
- `speedStop()` is called when stopping the run.

Default real-control parameters:

```text
real_command_backend = rtde_speedj
trajectory_name = circle
heart_scale = 0.0175      # circle radius = 4 * heart_scale = 0.07 m, diameter = 14 cm
tau / speedj_t = 0.005 s
speedj_accel = 0.5 rad/s^2
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

Recommended transport sanity check before running the ZNN circle:

```python
import time
import rtde_control
import rtde_receive

ip = "192.168.126.10"
c = rtde_control.RTDEControlInterface(ip)
r = rtde_receive.RTDEReceiveInterface(ip)
q0 = r.getActualQ()
c.speedJ([0, 0.05, 0, 0, 0, 0], 0.2, 2.0)
time.sleep(2.1)
c.speedStop(0.2)
q1 = r.getActualQ()
print(q0)
print(q1)
```

Joint 2 should move by roughly `0.1 rad`. If it does not, fix RTDE/program-mode/network setup before running the full ZNN experiment.

Full 20 s circle:

```bash
python3 run_clean_repetitive_tracking_rtde_speed.py \
  --live-backend ros_real_ur3e \
  --ur3e-robot-ip "$UR3E_ROBOT_IP" \
  --duration 20
```
