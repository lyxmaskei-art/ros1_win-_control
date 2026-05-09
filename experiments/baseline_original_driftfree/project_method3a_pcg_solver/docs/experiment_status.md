# Experiment Status

## Offline sweep on the heart trajectory

| `max_iters` | Mean position error (m) | Final position error (m) | Final drift norm (rad) | Mean residual norm |
| --- | ---: | ---: | ---: | ---: |
| 1 | `2.043e-4` | `5.447e-6` | `1.808e+0` | `5.308e-1` |
| 2 | `2.474e-3` | `4.462e-5` | `9.633e+0` | `3.298e+0` |
| 3 | `8.562e-3` | `4.854e-2` | `6.989e+0` | `5.795e-1` |
| 4 | `4.571e-2` | `1.844e-2` | `4.089e+0` | `7.633e-1` |
| 5 | `2.043e-4` | `6.079e-7` | `2.284e-5` | `4.153e-6` |
| 6 | `2.043e-4` | `6.081e-7` | `9.104e-6` | `5.492e-6` |

## Retained configuration

The retained setting is `max_iters = 5` because it already matches the direct-solve position accuracy while using fewer iterations than the full `6` step cap.

Its finalized offline result is:

| Solver | Mean position error (m) | Final position error (m) | Final drift norm (rad) | Mean inner iterations |
| --- | ---: | ---: | ---: | ---: |
| Method 3a direct solve | `2.043e-4` | `6.080e-7` | `4.667e-7` | `1.0` |
| Method 3a warm-started PCG (`max_iters = 5`) | `2.043e-4` | `6.079e-7` | `2.284e-5` | `4.444` |

## Live result of the retained PCG setting

| Solver | Mean position error (m) | Final position error (m) | Final drift norm (rad) | Mean inner iterations |
| --- | ---: | ---: | ---: | ---: |
| Method 3a warm-started PCG (`max_iters = 5`) | `1.302e-4` | `5.815e-5` | `4.071e-4` | `4.526` |

## Interpretation

1. `PCG(5)` is the first replacement in this direction that keeps the offline position accuracy essentially unchanged.
2. The drift-free recovery is still weaker than the exact direct solve, but it remains in the `1e-5 rad` level, which is much better than the previous DLCCZNN replacement line.
3. The live experiment also converges stably. Its final drift is in the `1e-4 rad` level, so the replacement still preserves a strong drift-free effect in simulation.
4. For `Method 3a`, warm-started PCG is currently the best alternative solver found in this workspace. It is far better than the previous DLCCZNN replacement line, and it is the first alternative that remains convincing in both offline and live tests.
