# Live 14 cm Circle Source Package

This package preserves the source code and run artifacts for the two live 14 cm circle experiment groups run on 2026-06-08.

## Contents

- `source/servo_like_live/`: copied source scripts used for the runs. The Python files are preserved as-is from `real experiments_velocity_command/servo_like_live`.
- `results/live_diameter14cm_old_params_20260608/`: 14 cm `circle_periodic` live results.
- `results/live_diameter14cm_circle_old_params_20260608/`: 14 cm ordinary `circle` live results.

## Shared Settings

- `tau = 0.005`
- `duration = 20`
- `trajectory_period = 10`
- `heart_scale = 0.0175`
- circle radius = `4 * heart_scale = 0.07 m`
- circle diameter = `0.14 m`
- `task_gain = 340`
- `drift_gain = 50` for drift-free cases
- `solver_gamma = 4352`
- `activation_power = 0.8`
- `activation_exp_clip = 4.0`
- `dlccznn_inner_steps = 60`

## Circle Periodic Results

| Experiment | Disturbance | Mean error (m) | Max error (m) | Final drift (rad) | Max qdot (rad/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 01 clean | none | 5.152e-4 | 7.796e-4 | 1.805e-3 | 3.468e-1 |
| 02 with drift-free | none | 5.152e-4 | 7.796e-4 | 1.805e-3 | 3.468e-1 |
| 02 without drift-free | none | 5.045e-4 | 5.625e-4 | 8.361e-1 | 2.443e-1 |
| 04 mild robustness | triangle | 5.117e-4 | 7.398e-4 | 1.797e-3 | 3.599e-1 |

## Ordinary Circle Results

| Experiment | Disturbance | Mean error (m) | Max error (m) | Final drift (rad) | Max qdot (rad/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 01 clean | none | 4.987e-4 | 8.909e-4 | 3.382e-4 | 5.138e-1 |
| 02 with drift-free | none | 4.987e-4 | 8.909e-4 | 3.382e-4 | 5.138e-1 |
| 02 without drift-free | none | 4.783e-4 | 5.321e-4 | 7.697e-1 | 3.445e-1 |
| 04 mild robustness | triangle | 4.970e-4 | 9.049e-4 | 3.347e-4 | 5.301e-1 |

