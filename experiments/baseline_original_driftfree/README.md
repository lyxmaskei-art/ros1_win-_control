# Baseline Original Drift Free

This folder keeps the baseline drift free experiments and the early solver replacement experiments.

| Project | Offline entry | Live entry | Description |
| --- | --- | --- | --- |
| `project_method3a_pcg_solver` | `scripts/run_offline.py` | `scripts/run_live.py` | Method 3a position and drift objective with warm started PCG. |
| `project_method3c_lccznn_solver` | `scripts/run_method1_offline.py`, `scripts/run_method2_offline.py`, `scripts/run_method3c_offline.py` | `scripts/run_method1_live.py`, `scripts/run_method2_live.py`, `scripts/run_method3c_live.py` | Early Method 1 and Method 2 DLCCZNN replacement plus Method 3c LCCZNN. |

Each entry file is a complete runnable script for direct VSCode execution. The offline and live files are intentionally separated.
