# ROS UR driver position-trajectory checkout for UR3e

This folder name is historical. The current code does **not** use bare `ur_rtde`, `speedJ`, `servoJ`, raw sockets, or port `30002`.

Current command path:

```text
ZNN computes qdot
-> runtime guards in rtde_realtime_adapter.py
-> one-step trajectory_msgs/JointTrajectory
-> /scaled_pos_joint_traj_controller/command
-> Universal Robots official ROS driver + External Control URCap
-> UR3e
```

This is not a full pre-generated trajectory. It sends short online trajectory points through the UR ROS driver controller. The position target is integrated from the controller output as `q_next = q_target + qdot * tau`, matching the live-simulation servo-like command convention.

The scripts are locked by default. They refuse to publish nonzero commands unless:

```bash
export UR3E_ENABLE_ROS_DRIVER_EXPERIMENT_I_ACCEPT_RISK=1
```

Do not set that variable until:

- UR ROS driver is running.
- External Control is active on the teach pendant.
- `/scaled_pos_joint_traj_controller/command` has exactly one expected subscriber.
- Robot speed slider is reduced.
- Emergency stop has been tested.
- The workspace is clear.

Safety defaults:

```text
duration = 2 s
tau = 0.005 s
circle radius = 5 mm
```

The ROS transport layer does not add an extra velocity clamp, slew-rate limiter, or position-step clamp. It forwards the controller's `qdot` through `q_next = q_target + qdot * tau` after only these runtime guards:

- command output must be explicitly enabled;
- `/joint_states` must be fresh;
- the command topic must have a controller subscriber;
- the command vector must be finite and 6-dimensional.

Trajectory and joint-velocity limits are therefore the same as the live-simulation controller path: the script-level TVQP/ZNN limits such as `theta_dot_limit`, dynamic joint bounds, and the UR ROS controller itself.

The script does not reset or home the real robot. It reads the current `/joint_states` sample and uses that as the live run's initial joint reference. Move the robot to the desired initial posture before launching the experiment.

After each live run, the adapter diagnostic report records:

```text
raw_velocity_max_abs  raw ZNN velocity peak before ROS publication
```

Run only after the ROS driver and controller are already up:

```bash
source ~/catkin_ws/devel/setup.bash
export UR3E_ENABLE_ROS_DRIVER_EXPERIMENT_I_ACCEPT_RISK=1
python3 run_clean_repetitive_tracking_rtde_position.py --live-backend real_ur3e
```

Full 14 cm, 20 s experiments are **not** defaults anymore. Only use them after the 2 s / 5 mm checkout is stable.
