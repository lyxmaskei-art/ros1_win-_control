# Method 3a PCG Solver Project

This project evaluates whether a warm started preconditioned conjugate gradient solver can replace the direct linear solve inside Method 3a while preserving high drift free accuracy.

## Entry Files

- Offline experiment: `scripts/run_offline.py`
- Live CoppeliaSim experiment: `scripts/run_live.py`

Both files are complete standalone experiment scripts. They do not import a legacy integrated runner.

## Current Retained Setting

- `solver_max_iters = 5`
- `solver_tol = 1e-12`
- `use_jacobi_preconditioner = True`

## Current Conclusion

1. Method 3a solves a `6 x 6` symmetric positive definite linear system at each sample, so warm started PCG is structurally suitable.
2. In offline heart tracking, `max_iters = 5` reproduces the direct solve position accuracy almost exactly.
3. The live retained `PCG(5)` setting gives mean position error `1.302e-4 m`, final position error `5.815e-5 m`, and final joint drift norm `4.071e-4 rad`.
