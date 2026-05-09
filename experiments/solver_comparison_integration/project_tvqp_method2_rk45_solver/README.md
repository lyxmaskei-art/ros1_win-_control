# Method2 TVQP RK45 Solver

This project compares the Method 2 solver layer using RK45 integration against the previous Euler style update.

## Entry Files

- Offline experiment: `scripts/run_offline.py`
- Live CoppeliaSim experiment: `scripts/run_live.py`

Both files are complete standalone scripts and do not import the old integrated runner.

## Example Commands

```powershell
cd C:\Users\lyx\Desktop\sci\experiments\solver_comparison_integration\project_tvqp_method2_rk45_solver
python .\scripts\run_offline.py --solver-gamma-gain 4608 --rtol 1e-5 --atol 1e-8
python .\scripts\run_live.py --solver-gamma-gain 4608 --rtol 1e-5 --atol 1e-8
```
