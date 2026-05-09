# Run Guide

## Method 3c

```powershell
cd C:\Users\lyx\Desktop\sci\experiments\baseline_original_driftfree\project_method3c_lccznn_solver
python scripts\run_method3c_offline.py
python scripts\run_method3c_live.py
```

## Method 1

```powershell
python scripts\run_method1_offline.py
python scripts\run_method1_live.py
```

## Method 2

```powershell
python scripts\run_method2_offline.py
python scripts\run_method2_live.py --solver-gamma-gain 2560 --solver-substeps 60
```
