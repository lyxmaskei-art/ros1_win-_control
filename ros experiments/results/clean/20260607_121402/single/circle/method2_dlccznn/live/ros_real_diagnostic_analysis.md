# ROS real diagnostics analysis

Source: `results/clean/20260607_121402/single/circle/method2_dlccznn/live/ros_real_cycle_diagnostics.csv`

## Key metrics

- Samples: 1000
- tau: 0.005 s
- deadline miss ratio: 0.000%
- period overrun ratio: 0.000%
- measured TCP pose used ratio: 0.000%
- cycle_period_s: median=0.004999856000267755, p95=0.005031850200202826, p99=0.005112019999796757, max=0.0052492059999167395
- cycle_work_s: median=0.0013350640001590364, p95=0.0015696419502774005, p99=0.00183446659008041, max=0.002400401000159036
- state_age_s: median=0.0010978725001677958, p95=0.0022538834000442876, p99=0.0023000107799589385, max=0.002512883000235888
- controller_step_s: median=0.0009397445001013693, p95=0.0011258930501981013, p99=0.0013912135301188755, max=0.0020464240001274447
- publish_s: median=4.441400005816831e-05, p95=5.532355028208257e-05, p99=6.718991989600908e-05, max=0.0001193449998027063
- command_step_max_abs_rad: median=0.0005999999999999998, p95=0.0006000000000000033, p99=0.0006000000000000033, max=0.0006000000000000033
- command_accel_norm_rad_s2: median=0.16365103503983142, p95=4.242640790863755, p99=5.3369238266558, max=6.086988463469232
- feedback_velocity_norm_rad_s: median=0.030626771363378776, p95=0.03625333178862392, p99=0.03723280926782466, max=0.03867592832896235
- tcp_pose_age_s: median=None, p95=None, p99=None, max=None
- tcp_fk_error_norm_m: median=None, p95=None, p99=None, max=None

## Findings

1. [LOW] No obvious timing or command-smoothness fault in diagnostics
   Evidence: All checked p95/p99 values are within the default thresholds.
   Action: If the robot still visibly shakes, suspect the position-command interface or TCP/model mismatch; test velocity or time-stamped trajectory control.
