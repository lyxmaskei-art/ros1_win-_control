# ROS real UR3e experiments

This folder contains the ROS1 real-robot versions of the five UR3e experiment
programs.

## Files

- `_ros_real_ur3e.py`: shared ROS1 joint-state / joint-position interface.
- `01_clean_repetitive_tracking/run_clean_repetitive_tracking.py`
- `02_drift_free_ablation/run_with_drift_free.py`
- `02_drift_free_ablation/run_without_drift_free.py`
- `03_real_time_feasibility/benchmark_step_time.py`
- `04_mild_robustness/run_mild_disturbance.py`

## ROS setup

```bash
source /home/lyx/文档/catkin_ws/devel_isolated/setup.bash
export ROS_IP=192.168.126.11
export ROS_HOSTNAME=
export ROS_MASTER_URI=http://192.168.126.11:11311
```

## Current real-machine test command

Use DLCCZNN, not PDNN. Current TCP and low-shake parameters:

```bash
python3 01_clean_repetitive_tracking/run_clean_repetitive_tracking.py \
  --experiment single \
  --method method2_dlccznn \
  --trajectory-name circle \
  --tcp-offset 0,0,0.145 \
  --task-gain 240 \
  --solver-gamma 30 \
  --drift-gain 2 \
  --theta-dot-limit 0.4 \
  --tau 0.005
```

If the real robot still shakes, enable the optional command-acceleration limiter:

```bash
  --max-command-accel 12
```

The circle trajectory is configured as a 14 cm diameter circle. The default run
duration is 20 s and the reference period is 10 s, so a formal run draws two
cycles.
