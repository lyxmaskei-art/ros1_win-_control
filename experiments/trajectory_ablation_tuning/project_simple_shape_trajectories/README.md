# Simple Shape Trajectory Project

This project applies simple trajectories to two existing methods:

- Method 2 DLCCZNN
- Method 3a direct solver

## Entry Files

- Method 2 offline: `python scripts/run_method2_offline.py --trajectory-name circle`
- Method 2 live: `python scripts/run_method2_live.py --trajectory-name circle`
- Method 3a offline: `python scripts/run_method3a_offline.py --trajectory-name circle`
- Method 3a live: `python scripts/run_method3a_live.py --trajectory-name circle`

The available trajectory names include `heart`, `circle`, and `line`. Each offline or live entry is a complete standalone script.

## Current Conclusion

1. `circle` and `line` significantly reduce the mean position error relative to the original heart trajectory in both offline and live experiments.
2. Rest to rest boundary conditions keep the offline final joint drift at the same tiny level as the heart trajectory.
3. In live experiments, Method 2 DLCCZNN improves from `3.357e-4 m` mean position error on `heart` to `1.633e-4 m` on `circle` and `1.401e-4 m` on `line`.
4. In live experiments, Method 3a improves from `1.308e-4 m` mean position error on `heart` to `6.514e-5 m` on `circle` and `5.624e-5 m` on `line`.
