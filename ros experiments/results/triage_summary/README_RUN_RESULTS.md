# ROS Real UR3e README Procedure Results

## Executed

- joint_trajectory preflight: `fail`; command topic `/scaled_pos_joint_traj_controller/command` had no subscriber.
- position_array preflight: `pass`; `/joint_states` median rate `459.214 Hz`.
- position_array real runs: 7 runs collected, all analyzed with `analyze_ros_real_diagnostics.py`.
- A/B comparison generated with `compare_ros_real_runs.py`.
- Full artifact zip generated with `collect_ros_real_artifacts.py --include-heavy`.

## Not Executed On Hardware

- joint_trajectory motion run: skipped because preflight failed; no subscriber/controller for `/scaled_pos_joint_traj_controller/command`.
- velocity_array motion run: not executed from this automation because current preflight evidence showed `joint_group_vel_controller` was `initialized`, not `running`; running it requires explicitly switching controllers on the robot side.

## Run Summary

| Run | Duration inferred | Mean error (m) | Final drift (rad) | Accel p95 | Deadline | Overrun | Finding |
|---|---:|---:|---:|---:|---:|---:|---|
| `20260607_120901` | 5s | 0.0219893 | 0.15991 | 10.3923 | 0.000% | 0.000% | No obvious timing or command-smoothness fault in diagnostics |
| `20260607_121402` | 5s | 0.0327541 | 0.113801 | 4.24264 | 0.000% | 0.000% | No obvious timing or command-smoothness fault in diagnostics |
| `20260607_123417` | 10s | 0.0162282 | 0.00512925 | 17.8916 | 0.000% | 0.000% | Command acceleration is high |
| `20260607_122351` | 5s | 0.00628538 | 0.214515 | 13.8566 | 0.000% | 0.000% | Command acceleration is high |
| `20260607_123302` | 10s | 0.0151561 | 0.0167482 | 24 | 0.000% | 0.050% | Command acceleration is high |
| `20260607_122451` | 5s | 0.00620849 | 0.211674 | 17.3206 | 0.000% | 0.000% | Command acceleration is high |
| `20260607_121715` | 5s | 0.00630142 | 0.212542 | 20.7863 | 0.000% | 0.000% | Command acceleration is high |

## Geometry Check From History

For the full 10 s circle run `20260607_123417`, the desired circle is in the YZ plane. Local-FK actual range was compressed mainly in Y:

- desired Y range: 0.1400 m; actual Y range: 0.1212 m
- desired Z range: 0.1400 m; actual Z range: 0.1422 m
- max error occurred around t=5.95 s, mostly in Y: about 0.0627 m
- final drift after a full period was small: 0.00513 rad

## Files

- Full artifact zip: `results/triage_summary/ros_real_triage_full.zip`
- Comparison CSV: `results/triage_summary/ros_real_ab_comparison.csv`
- Comparison MD: `results/triage_summary/ros_real_ab_comparison.md`
- This summary: `results/triage_summary/README_RUN_RESULTS.md`

