# ROS real diagnostics analysis

Source: `results/clean/20260607_123302/single/circle/method2_dlccznn/live/ros_real_cycle_diagnostics.csv`

## Key metrics

- Samples: 2000
- tau: 0.005 s
- deadline miss ratio: 0.000%
- period overrun ratio: 0.050%
- measured TCP pose used ratio: 0.000%
- cycle_period_s: median=0.004999810000299476, p95=0.005084144200282026, p99=0.0051309844006391355, max=0.0067031959997621016
- cycle_work_s: median=0.001386323000133416, p95=0.0017908975504724368, p99=0.0019467044494922447, max=0.0027481190008984413
- state_age_s: median=0.001038868000250659, p95=0.0016503751503023523, p99=0.0023024060996885963, max=0.003290453000772686
- controller_step_s: median=0.0010011675003624987, p95=0.0013456145003146957, p99=0.0014842795394633867, max=0.002109585000653169
- publish_s: median=4.59110001429508e-05, p95=6.130799970378575e-05, p99=7.589171004838136e-05, max=0.00028511399978015106
- command_step_max_abs_rad: median=0.0020000000000000018, p95=0.0020000000000000018, p99=0.0020000000000000018, max=0.0020000000000000018
- command_accel_norm_rad_s2: median=6.3076426131864975, p95=24.000049997793937, p99=29.393876913396547, max=29.39387691340017
- feedback_velocity_norm_rad_s: median=0.09178353261871348, p95=0.11861098314080941, p99=0.12188322612902548, max=0.12712167123796242
- tcp_pose_age_s: median=None, p95=None, p99=None, max=None
- tcp_fk_error_norm_m: median=None, p95=None, p99=None, max=None

## Findings

1. [MEDIUM] Command acceleration is high
   Evidence: command_accel_norm_rad_s2 p95=24, max=29.3939
   Action: Keep --max-command-accel enabled and sweep 6, 8, 12 rad/s^2 before changing the main gains.
