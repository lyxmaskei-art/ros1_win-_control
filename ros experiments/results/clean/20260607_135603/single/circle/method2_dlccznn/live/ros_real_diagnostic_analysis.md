# ROS real diagnostics analysis

Source: `results/clean/20260607_135603/single/circle/method2_dlccznn/live/ros_real_cycle_diagnostics.csv`

## Key metrics

- Samples: 2000
- tau: 0.005 s
- deadline miss ratio: 0.000%
- period overrun ratio: 0.050%
- measured TCP pose used ratio: 0.000%
- cycle_period_s: median=0.004999632999897585, p95=0.005088961099318112, p99=0.005168142000038642, max=0.006533040999784134
- cycle_work_s: median=0.0014497209995170124, p95=0.0018730146000962122, p99=0.002039243389517651, max=0.0025939479983208003
- state_age_s: median=0.001020260000586859, p95=0.0022324362494146044, p99=0.0023301355006515225, max=0.004489095999815618
- controller_step_s: median=0.0009690835004221299, p95=0.0013356041002225538, p99=0.0014469120308058336, max=0.0017241259993170388
- publish_s: median=0.00010087399914482376, p95=0.00014569195000149193, p99=0.00021177861000978737, max=0.0012310300007811747
- command_step_max_abs_rad: median=0.0020000000000000018, p95=0.0020000000000000018, p99=0.0020000000000000018, max=0.0020000000000000018
- command_accel_norm_rad_s2: median=0.7179596880285251, p95=20.78461194149661, p99=24.000021502678507, max=27.145974362727518
- feedback_velocity_norm_rad_s: median=0.07513333117522858, p95=0.08254129443551786, p99=0.08357365614060855, max=0.08511213509932877
- tcp_pose_age_s: median=None, p95=None, p99=None, max=None
- tcp_fk_error_norm_m: median=None, p95=None, p99=None, max=None

## Findings

1. [MEDIUM] Command acceleration is high
   Evidence: command_accel_norm_rad_s2 p95=20.7846, max=27.146
   Action: Keep --max-command-accel enabled and sweep 6, 8, 12 rad/s^2 before changing the main gains.
