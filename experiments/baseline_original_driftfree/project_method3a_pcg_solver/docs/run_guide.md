# Run Guide

## Offline

```powershell
cd C:\Users\lyx\Desktop\sci\experiments\baseline_original_driftfree\project_method3a_pcg_solver
python scripts\run_offline.py
```

## Offline Sweep Examples

```powershell
python scripts\run_offline.py --solver-max-iters 1
python scripts\run_offline.py --solver-max-iters 5
python scripts\run_offline.py --solver-max-iters 6
```

## Live

```powershell
python scripts\run_live.py
```

If CoppeliaSim is not open or the remote API is not reachable on `19997/19998`, the live script will stop with a connection error.
