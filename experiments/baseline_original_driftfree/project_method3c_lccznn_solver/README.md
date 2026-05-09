# DLCCZNN Replacement Project

This folder contains separated offline and live scripts for the early DLCCZNN replacement experiments.

## Entry Files

- Method 1 offline: `python scripts/run_method1_offline.py`
- Method 1 live: `python scripts/run_method1_live.py`
- Method 2 offline: `python scripts/run_method2_offline.py`
- Method 2 live: `python scripts/run_method2_live.py`
- Method 3c offline: `python scripts/run_method3c_offline.py`
- Method 3c live: `python scripts/run_method3c_live.py`

Each script is complete and can be run directly in VSCode.

## Current Practical Conclusion

- Method 1 DLCCZNN is slightly better than the original Method 1 in both offline and live runs.
- Method 2 DLCCZNN is the strongest early replacement result. The retained solver setting is `solver_gamma_gain = 2560` and `solver_substeps = 60`.
- Method 3c DLCCZNN preserves strong position accuracy but does not preserve the original Method 3a drift free effect.
