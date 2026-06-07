# ROS real diagnostics analysis

Source: `results/clean/20260607_123417/single/circle/method2_dlccznn/live/ros_real_cycle_diagnostics.csv`

## Key metrics

- Samples: 2000
- tau: 0.005 s
- deadline miss ratio: 0.000%
- period overrun ratio: 0.000%
- measured TCP pose used ratio: 0.000%
- cycle_period_s: median=0.004999867999686103, p95=0.0050683219995335095, p99=0.005122230399902037, max=0.006084186999942176
- cycle_work_s: median=0.0013876750003873894, p95=0.0018622889993821445, p99=0.002000997030027065, max=0.002465511000082188
- state_age_s: median=0.0010374045000389742, p95=0.002069762500013894, p99=0.0021546268192651043, max=0.002889633000449976
- controller_step_s: median=0.0010087860000567161, p95=0.0013956855501874085, p99=0.0015034833401568902, max=0.002075050999337691
- publish_s: median=4.595599966705777e-05, p95=6.42565499219927e-05, p99=7.752691050882276e-05, max=0.00028746800035150954
- command_step_max_abs_rad: median=0.0020000000000000018, p95=0.0020000000000000018, p99=0.0020000000000000018, max=0.0020000000000000018
- command_accel_norm_rad_s2: median=5.217103136962999, p95=17.891614538443882, p99=19.595917942264386, max=19.595917942268013
- feedback_velocity_norm_rad_s: median=0.08987969875637924, p95=0.11905231302354566, p99=0.12213585732505859, max=0.1270149733190499
- tcp_pose_age_s: median=None, p95=None, p99=None, max=None
- tcp_fk_error_norm_m: median=None, p95=None, p99=None, max=None

## Findings

1. [MEDIUM] Command acceleration is high
   Evidence: command_accel_norm_rad_s2 p95=17.8916, max=19.5959
   Action: Keep --max-command-accel enabled and sweep 6, 8, 12 rad/s^2 before changing the main gains.
