# ROS UR driver velocity-control checkout for UR3e

This folder name is historical. The current code does **not** use bare `ur_rtde`, `speedJ`, `servoJ`, raw sockets, or port `30002`.

Current command path:

```text
ZNN computes qdot
-> hard safety limits in rtde_realtime_adapter.py
-> std_msgs/Float64MultiArray
-> /joint_group_vel_controller/command
-> Universal Robots official ROS driver + External Control URCap
-> UR3e
```

The scripts are locked by default. They refuse to publish nonzero commands unless:

```bash
export UR3E_ENABLE_ROS_DRIVER_EXPERIMENT_I_ACCEPT_RISK=1
```

Do not set that variable until:

- UR ROS driver is running.
- External Control is active on the teach pendant.
- `/joint_group_vel_controller/command` has exactly one expected subscriber.
- Robot speed slider is reduced.
- Emergency stop has been tested.
- The workspace is clear.

Safety defaults:

```text
duration = 2 s
tau = 0.005 s
circle radius = 5 mm
max_abs_qdot = 0.03 rad/s
max_qdot_delta = 0.01 rad/s per cycle
```

The default `max_abs_qdot = 0.03 rad/s` is only a checkout clamp for a tiny 2 s / 5 mm motion. It is intentionally too small for the formal 14 cm validation trajectory. If the measured or expected joint speed is around `0.3 rad/s`, a formal run must raise the clamp explicitly. Use `0.30` only when the measured peak stays below that value; use `0.35` when the peak is around `0.3` and you need margin against harmless clipping.

```bash
export UR3E_SAFE_MAX_ABS_QDOT=0.35
export UR3E_SAFE_MAX_QDOT_DELTA=0.01
```

`UR3E_SAFE_MAX_QDOT_DELTA=0.01` means each joint command can change by at most `0.01 rad/s` every `5 ms`, equivalent to about `2 rad/s^2`. That is normally already above the smooth 20 s two-circle reference demand, while still blocking sharp command jumps. If the diagnostic report shows repeated clipping during a stable checkout, raise it gradually instead of removing it.

After each live run, check the adapter diagnostic report:

```text
raw_velocity_max_abs  raw ZNN velocity peak before safety clipping
abs_clip_count        number of cycles clipped by UR3E_SAFE_MAX_ABS_QDOT
delta_clip_count      number of cycles clipped by UR3E_SAFE_MAX_QDOT_DELTA
```

For a formal tracking run, `abs_clip_count` should normally be zero. If it is not zero, the absolute velocity clamp is still below the controller demand and the trajectory is being distorted by the transport layer.

Run only after the ROS driver and controller are already up:

```bash
source ~/catkin_ws/devel/setup.bash
export UR3E_ENABLE_ROS_DRIVER_EXPERIMENT_I_ACCEPT_RISK=1
python3 run_clean_repetitive_tracking_rtde_speed.py --live-backend ros_real_ur3e
```

Full 14 cm, 20 s experiments are **not** defaults anymore. Only use them after the 2 s / 5 mm checkout is stable.
