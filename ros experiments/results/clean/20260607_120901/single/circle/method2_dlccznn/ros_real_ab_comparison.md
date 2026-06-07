# ROS real A/B comparison

| Rank | Command mode | Score | High | Medium | Deadline miss | Period overrun | TCP used | State age p95 (s) | Command accel p95 | TCP-FK p95 (m) | Mean error (m) | Final drift (rad) | Top finding |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | position_array | 4.85989 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.00226262 | 10.3923 | 0 | 0.02198932481523703 | 0.15990996381567327 | No obvious timing or command-smoothness fault in diagnostics |
| 2 | position_array | 4.86022 | 0 | 0 | 0.000% | 0.000% | 0.000% | 0.00225388 | 4.24264 | 0 | 0.032754119727989556 | 0.11380077482201534 | No obvious timing or command-smoothness fault in diagnostics |
| 3 | position_array | 104.854 | 0 | 1 | 0.000% | 0.000% | 0.000% | 0.0019732 | 20.7863 | 0 | 0.006301423346648842 | 0.2125419493051158 | Command acceleration is high |

## Recommendation

Best diagnostic score: `position_array` from `results/clean/20260607_120901/single/circle/method2_dlccznn/live`.
No high-severity finding in the best run. If visual shake also improved, keep this command mode for the next tuning pass.
