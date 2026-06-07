# ROS real diagnostics analysis

Source: `results/clean/20260607_120901/single/circle/method2_dlccznn/live/ros_real_cycle_diagnostics.csv`

## Key metrics

- Samples: 1000
- tau: 0.005 s
- deadline miss ratio: 0.000%
- period overrun ratio: 0.000%
- measured TCP pose used ratio: 0.000%
- cycle_period_s: median=0.0049992909998763935, p95=0.005090673299719128, p99=0.005147153199877721, max=0.005249491000085982
- cycle_work_s: median=0.0013187100000777718, p95=0.001765249100071742, p99=0.001952408289998857, max=0.0022663710001324944
- state_age_s: median=0.00103722950007068, p95=0.0022626195499697134, p99=0.00234695951014146, max=0.003968758999690181
- controller_step_s: median=0.0009126050001668773, p95=0.0013285979999636763, p99=0.001432207910188481, max=0.0016167859998859058
- publish_s: median=4.435849996298202e-05, p95=6.137750006018904e-05, p99=7.633831036400807e-05, max=0.00019536799982233788
- command_step_max_abs_rad: median=0.0010000000000000009, p95=0.0010000000000000009, p99=0.0010000000000000009, max=0.0010000000000000009
- command_accel_norm_rad_s2: median=0.35587395514297915, p95=10.3923202549914, p99=12.000021260049653, max=13.416541137210205
- feedback_velocity_norm_rad_s: median=0.048544006427414174, p95=0.05921331334892099, p99=0.061353353380404325, max=0.06353748119978954
- tcp_pose_age_s: median=None, p95=None, p99=None, max=None
- tcp_fk_error_norm_m: median=None, p95=None, p99=None, max=None

## Findings

1. [LOW] No obvious timing or command-smoothness fault in diagnostics
   Evidence: All checked p95/p99 values are within the default thresholds.
   Action: If the robot still visibly shakes, suspect the position-command interface or TCP/model mismatch; test velocity or time-stamped trajectory control.
