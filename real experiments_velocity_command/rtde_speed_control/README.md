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

Run only after the ROS driver and controller are already up:

```bash
source ~/catkin_ws/devel/setup.bash
export UR3E_ENABLE_ROS_DRIVER_EXPERIMENT_I_ACCEPT_RISK=1
python3 run_clean_repetitive_tracking_rtde_speed.py --live-backend ros_real_ur3e
```

Full 14 cm, 20 s experiments are **not** defaults anymore. Only use them after the 2 s / 5 mm checkout is stable.
