# ROS real diagnostics analysis

Source: `results/clean/20260607_135745/single/circle/method2_dlccznn/live/ros_real_cycle_diagnostics.csv`

## Key metrics

- Samples: 2000
- tau: 0.005 s
- deadline miss ratio: 0.000%
- period overrun ratio: 0.000%
- measured TCP pose used ratio: 0.000%
- cycle_period_s: median=0.004999806000341778, p95=0.005034146999423683, p99=0.005112408618624613, max=0.0053118899995752145
- cycle_work_s: median=0.0013368189993343549, p95=0.001584734998868953, p99=0.0018767231300626007, max=0.0027672629985318054
- state_age_s: median=0.0009732594999150024, p95=0.0019793270000263872, p99=0.0020179477491365106, max=0.003735283000423806
- controller_step_s: median=0.0009175899995170766, p95=0.0011791863004873447, p99=0.0014197933091782034, max=0.00220149500091793
- publish_s: median=4.309549967729254e-05, p95=5.441555067591252e-05, p99=6.424400913601858e-05, max=0.00021255799947539344
- command_step_max_abs_rad: median=0.0020000000000000018, p95=0.0020000000000000018, p99=0.0020000000000000018, max=0.0020000000000000018
- command_accel_norm_rad_s2: median=24.01885777503891, p95=26.85270246272719, p99=27.552570178979376, max=29.393876913400423
- feedback_velocity_norm_rad_s: median=0.6296164061964716, p95=0.9301973746187578, p99=1.0503833986750297, max=1.1971352967596638
- tcp_pose_age_s: median=None, p95=None, p99=None, max=None
- tcp_fk_error_norm_m: median=None, p95=None, p99=None, max=None

## Findings

1. [HIGH] Command acceleration spikes
   Evidence: command_accel_norm_rad_s2 p95=26.8527, max=29.3939
   Action: Enable or lower --max-command-accel, reduce --theta-dot-limit, and reduce solver/task gains before increasing trajectory speed.
