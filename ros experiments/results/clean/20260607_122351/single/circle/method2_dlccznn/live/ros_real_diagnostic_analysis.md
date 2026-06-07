# ROS real diagnostics analysis

Source: `results/clean/20260607_122351/single/circle/method2_dlccznn/live/ros_real_cycle_diagnostics.csv`

## Key metrics

- Samples: 1000
- tau: 0.005 s
- deadline miss ratio: 0.000%
- period overrun ratio: 0.000%
- measured TCP pose used ratio: 0.000%
- cycle_period_s: median=0.00499979500000336, p95=0.005063027599499037, p99=0.005142700279502605, max=0.0052342939998197835
- cycle_work_s: median=0.001460625500385504, p95=0.0018454057006692891, p99=0.0019412267994994182, max=0.002756351999778417
- state_age_s: median=0.0009512064998489222, p95=0.0015964015999088588, p99=0.0016502252997088363, max=0.002541419999943173
- controller_step_s: median=0.0010505430000193883, p95=0.0013858840004104423, p99=0.0014829667101548692, max=0.002141711000149371
- publish_s: median=4.743899989989586e-05, p95=6.524005038954782e-05, p99=8.852541068335994e-05, max=0.00021592500070255483
- command_step_max_abs_rad: median=0.0020000000000000018, p95=0.0020000000000000018, p99=0.0020000000000000018, max=0.0020000000000000018
- command_accel_norm_rad_s2: median=3.2144626904694342, p95=13.856622809458358, p99=16.000622898747167, max=17.889056770145107
- feedback_velocity_norm_rad_s: median=0.08170690289239682, p95=0.11420769090206935, p99=0.11897227085144707, max=0.12442995579056036
- tcp_pose_age_s: median=None, p95=None, p99=None, max=None
- tcp_fk_error_norm_m: median=None, p95=None, p99=None, max=None

## Findings

1. [MEDIUM] Command acceleration is high
   Evidence: command_accel_norm_rad_s2 p95=13.8566, max=17.8891
   Action: Keep --max-command-accel enabled and sweep 6, 8, 12 rad/s^2 before changing the main gains.
