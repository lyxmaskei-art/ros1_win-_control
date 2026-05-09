# Experiment Status

## Method 2 DLCCZNN

| Trajectory | Mean position error (m) | Final position error (m) | Final drift norm (rad) |
| --- | ---: | ---: | ---: |
| Heart | `1.993e-4` | `1.253e-6` | `3.982e-6` |
| Circle | `4.857e-5` | `1.743e-7` | `9.687e-7` |
| Line | `1.871e-5` | `2.468e-7` | `1.173e-6` |

## Method 3a

| Trajectory | Mean position error (m) | Final position error (m) | Final drift norm (rad) |
| --- | ---: | ---: | ---: |
| Heart | `2.043e-4` | `6.080e-7` | `4.667e-7` |
| Circle | `5.027e-5` | `3.942e-11` | `2.969e-11` |
| Line | `2.000e-5` | `3.707e-8` | `4.590e-8` |

## Method 2 DLCCZNN Live

| Trajectory | Mean position error (m) | Final position error (m) | Final drift norm (rad) |
| --- | ---: | ---: | ---: |
| Heart | `3.357e-4` | `1.426e-4` | `4.548e-3` |
| Circle | `1.633e-4` | `1.451e-4` | `4.409e-3` |
| Line | `1.401e-4` | `1.447e-4` | `4.438e-3` |

## Method 3a Live

| Trajectory | Mean position error (m) | Final position error (m) | Final drift norm (rad) |
| --- | ---: | ---: | ---: |
| Heart | `1.308e-4` | `5.885e-5` | `4.192e-4` |
| Circle | `6.514e-5` | `5.843e-5` | `4.182e-4` |
| Line | `5.624e-5` | `5.844e-5` | `4.183e-4` |

## Interpretation

1. The first simple-shape version was misleading because its terminal velocity was not zero. After converting `circle` and `line` to rest-to-rest trajectories, the offline final drift returns to the same tiny scale as the heart trajectory.
2. `Line` is now the cleanest simple shape for both methods. It gives the lowest mean position error in both offline and live experiments.
3. For `Method 2 DLCCZNN`, the live final drift stays in the same `4.4e-3 rad` band on `heart`, `circle`, and `line`, while the mean position error improves substantially on the simple shapes.
4. For `Method 3a`, the live final drift is almost unchanged across all three trajectories, but the mean position error still drops by about one half on the simple shapes.
