# ROS real diagnostics analysis

Source: `results/clean/20260607_121715/single/circle/method2_dlccznn/live/ros_real_cycle_diagnostics.csv`

## Key metrics

- Samples: 1000
- tau: 0.005 s
- deadline miss ratio: 0.000%
- period overrun ratio: 0.000%
- measured TCP pose used ratio: 0.000%
- cycle_period_s: median=0.004999322000003303, p95=0.005060590099856199, p99=0.0051244202401358055, max=0.005250783999599662
- cycle_work_s: median=0.001457107500073107, p95=0.001932361500053048, p99=0.0020723048599438697, max=0.00323131100003593
- state_age_s: median=0.001080039000044053, p95=0.0019732035001879923, p99=0.0020430475200146248, max=0.0032895800000005693
- controller_step_s: median=0.0010599514998830273, p95=0.0014709322497537868, p99=0.0015312587401058407, max=0.0026600880000842153
- publish_s: median=5.111099994792312e-05, p95=6.511804999718151e-05, p99=0.00011774493977554805, max=0.00029019500016147504
- command_step_max_abs_rad: median=0.0020000000000000018, p95=0.0020000000000000018, p99=0.0020000000000000018, max=0.0020000000000000018
- command_accel_norm_rad_s2: median=3.3282865538077884, p95=20.78630878986155, p99=24.032396310040316, max=26.832890361214925
- feedback_velocity_norm_rad_s: median=0.0825660256706718, p95=0.11452770406446987, p99=0.12005788653030294, max=0.12327273522259365
- tcp_pose_age_s: median=None, p95=None, p99=None, max=None
- tcp_fk_error_norm_m: median=None, p95=None, p99=None, max=None

## Findings

1. [MEDIUM] Command acceleration is high
   Evidence: command_accel_norm_rad_s2 p95=20.7863, max=26.8329
   Action: Keep --max-command-accel enabled and sweep 6, 8, 12 rad/s^2 before changing the main gains.
