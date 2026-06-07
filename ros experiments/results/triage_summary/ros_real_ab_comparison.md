# ROS real A/B comparison

| Rank | Command mode | Score | High | Medium | Deadline miss | Period overrun | TCP used | State age p95 (s) | Command accel p95 | TCP-FK p95 (m) | Mean error (m) | Final drift (rad) | Top finding |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | position_array | 4.85989 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.00226262 | 10.3923 | 0 | 0.02198932481523703 | 0.15990996381567327 | No obvious timing or command-smoothness fault in diagnostics |
| 2 | position_array | 4.86022 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.00225388 | 4.24264 | 0 | 0.032754119727989556 | 0.11380077482201534 | No obvious timing or command-smoothness fault in diagnostics |
| 3 | position_array | 103.484 | 0 | 1 | 0.000% | 0.000% | 0.000% | 0.00206976 | 17.8916 | 0 | 0.016228171688615508 | 0.005129245055569422 | Command acceleration is high |
| 4 | position_array | 104.175 | 0 | 1 | 0.000% | 0.000% | 0.000% | 0.0015964 | 13.8566 | 0 | 0.006285380563934208 | 0.21451527670191223 | Command acceleration is high |
| 5 | position_array | 104.25 | 0 | 1 | 0.000% | 0.050% | 0.000% | 0.00165038 | 24 | 0 | 0.01515605759911701 | 0.016748217273353343 | Command acceleration is high |
| 6 | position_array | 104.492 | 0 | 1 | 0.000% | 0.000% | 0.000% | 0.00220533 | 17.3206 | 0 | 0.006208493486896206 | 0.21167361068485796 | Command acceleration is high |
| 7 | position_array | 104.854 | 0 | 1 | 0.000% | 0.000% | 0.000% | 0.0019732 | 20.7863 | 0 | 0.006301423346648842 | 0.2125419493051158 | Command acceleration is high |
| 8 | joint_trajectory | 105.434 | 0 | 1 | 0.000% | 0.050% | 0.000% | 0.00223244 | 20.7846 | 0 | 0.029554453026228174 | 0.022737842692345005 | Command acceleration is high |
| 9 | velocity_array | 1005.04 | 1 | 0 | 0.000% | 0.000% | 0.000% | 0.00197933 | 26.8527 | 0 | 0.015645938254986922 | 0.07671379552700447 | Command acceleration spikes |

## Recommendation

Best diagnostic score: `position_array` from `results/clean/20260607_120901/single/circle/method2_dlccznn/live`.
No high-severity finding in the best run. If visual shake also improved, keep this command mode for the next tuning pass.
