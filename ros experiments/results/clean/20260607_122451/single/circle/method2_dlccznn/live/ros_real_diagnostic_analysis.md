# ROS real diagnostics analysis

Source: `results/clean/20260607_122451/single/circle/method2_dlccznn/live/ros_real_cycle_diagnostics.csv`

## Key metrics

- Samples: 1000
- tau: 0.005 s
- deadline miss ratio: 0.000%
- period overrun ratio: 0.000%
- measured TCP pose used ratio: 0.000%
- cycle_period_s: median=0.005000484000447614, p95=0.005057269099415862, p99=0.005120639980013948, max=0.005183676999877207
- cycle_work_s: median=0.0014396560000022873, p95=0.0016967700999884977, p99=0.00205113439957131, max=0.0029295959993760334
- state_age_s: median=0.0009931494996635593, p95=0.002205332549601735, p99=0.002272796340221248, max=0.0038910270004635095
- controller_step_s: median=0.0010424464999232441, p95=0.0012447572496057542, p99=0.0014530447199831542, max=0.002433777999613085
- publish_s: median=4.8565999804850435e-05, p95=7.742405059616431e-05, p99=0.00010369891989284951, max=0.000149268999848573
- command_step_max_abs_rad: median=0.0020000000000000018, p95=0.0020000000000000018, p99=0.0020000000000000018, max=0.0020000000000000018
- command_accel_norm_rad_s2: median=3.0095345947474827, p95=17.320608678768373, p99=20.18364551456883, max=22.379171819046814
- feedback_velocity_norm_rad_s: median=0.08162647951676365, p95=0.11582435834214919, p99=0.1227634370541844, max=0.1271430552580461
- tcp_pose_age_s: median=None, p95=None, p99=None, max=None
- tcp_fk_error_norm_m: median=None, p95=None, p99=None, max=None

## Findings

1. [MEDIUM] Command acceleration is high
   Evidence: command_accel_norm_rad_s2 p95=17.3206, max=22.3792
   Action: Keep --max-command-accel enabled and sweep 6, 8, 12 rad/s^2 before changing the main gains.
