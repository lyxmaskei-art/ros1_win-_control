# Current Experiment Status

The DLCCZNN replacement project has now been executed for three methods:

- `Method 1`: DLCCZNN replaces the original PDNN style KKT solver
- `Method 2`: DLCCZNN replaces the original Scheme-B PDNN style KKT solver
- `Method 3c`: DLCCZNN replaces the direct solve inside the Method 3a outer objective

## Offline comparison

| Method | Mean position error (m) | Final position error (m) | Final drift norm (rad) | Baseline final drift norm (rad) |
| --- | ---: | ---: | ---: | ---: |
| Method 1 DLCCZNN | `2.104e-4` | `2.954e-5` | `1.035e-2` | `1.889e-2` |
| Method 2 DLCCZNN | `1.931e-4` | `1.525e-6` | `4.174e-6` | `1.894e-5` |
| Method 3c DLCCZNN | `2.044e-4` | `1.596e-6` | `4.472e-1` | `1.867e-6` |

## Live comparison

| Method | Mean position error (m) | Final position error (m) | Final drift norm (rad) | Baseline final drift norm (rad) |
| --- | ---: | ---: | ---: | ---: |
| Method 1 DLCCZNN | `2.562e-3` | `8.249e-4` | `2.673e-1` | `2.677e-1` |
| Method 2 DLCCZNN tuned | `3.359e-4` | `1.423e-4` | `4.550e-3` | `4.541e-3` |
| Method 3c DLCCZNN | `1.277e-4` | `4.534e-5` | `6.273e-1` | `4.137e-4` |

## Interpretation

1. `Method 1` benefits from the DLCCZNN replacement. In both offline and live runs it is slightly better than the original Method 1, especially in final position error.
2. `Method 2` is the strongest replacement line. With the current retained solver setting `solver_gamma_gain = 2560` and `solver_substeps = 60`, its live drift is essentially unchanged relative to the original Method 2 and its final live position error is slightly better. Its live mean position error is still about two times the original Method 2, so it is close but not yet a full replacement.
3. `Method 3c` is the most important negative result. The strict DLCCZNN replacement preserves strong position accuracy, but it does not preserve the drift-free recovery property of the original direct solve Method 3a. In other words, for this problem the low-complexity neural solver is not accurate enough to stand in for the exact linear solve if drift-free is the main target.
