# Euler Ns40 Trajectory-Wise Live Optimization

This project contains a self-contained Method2 + ELNCP + LCCZNN live experiment runner for Euler inner discretization with `Ns=40`.

## Scope

- Clean live trajectory-wise tuning over `small_circle`, `line`, `circle`, `ellipse`, and `figure8`.
- Internal neural-dynamics noise replay using `n(t)=0.5t`.
- CoppeliaSim live summaries, joint drift tables, raw JSON audit hashes, and final CSV reports.

The noise is added to the internal solver dynamics `y_dot`, not to robot measurement or CoppeliaSim execution.

## Main Scripts

- `scripts/run_ns40_euler_live_case.py`: self-contained single live case runner.
- `scripts/run_ns40_live_from_candidates.py`: runs an explicit candidate CSV.
- `scripts/run_ns40_clean_live_sweep.py`: resumable grid sweep runner.
- `scripts/run_ns40_noise_live_from_best.py`: runs noise live cases from clean best rows.
- `scripts/summarize_ns40_live_results.py`: produces trajectory-wise best rows and clean-vs-noise summaries.

## Final Results

Final report:

`results/final_ns40_live_trajectory_opt_report/strict_chinese_ns40_live_report.md`

Key CSV files:

- `results/final_ns40_live_trajectory_opt_report/clean_best_by_trajectory.csv`
- `results/final_ns40_live_trajectory_opt_report/clean_vs_noise_paired.csv`
- `results/final_ns40_live_trajectory_opt_report/dataset_summary.csv`
- `results/final_ns40_live_trajectory_opt_report/best_live_joint_drift_components.csv`
- `results/final_ns40_live_trajectory_opt_report/raw_json_audit.csv`

## Reproduce

Run an explicit clean candidate set:

```powershell
python scripts/run_ns40_live_from_candidates.py `
  --candidates-csv results/round1_candidates.csv `
  --output-root results/round1_clean_live_10s `
  --duration 10.0 `
  --tau 0.005 `
  --noise-profile clean
```

Run noise replay from the clean best rows:

```powershell
python scripts/run_ns40_noise_live_from_best.py `
  --best-csv results/round123_summary/clean_best_by_trajectory.csv `
  --output-root results/noise_n2_live_from_round123_best_10s `
  --duration 10.0 `
  --tau 0.005 `
  --noise-profile paper_n2_linear
```

Summarize:

```powershell
python scripts/summarize_ns40_live_results.py `
  --clean-manifest results/combined_round1_round2_round3_clean_manifest.csv `
  --noise-manifest results/noise_n2_live_from_round123_best_10s/sweep_manifest.csv `
  --output-dir results/final_ns40_live_trajectory_opt_report
```
