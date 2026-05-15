import argparse
import csv
import json
from pathlib import Path

from run_ns40_clean_live_sweep import run_case


def load_best_rows(best_csv):
    rows = list(csv.DictReader(Path(best_csv).open("r", newline="", encoding="utf-8-sig")))
    if not rows:
        raise ValueError(f"No rows in {best_csv}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Run n(t)=0.5t noise live cases from trajectory-wise clean best rows.")
    parser.add_argument("--best-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--activation-exp-clip", type=float, default=4.0)
    parser.add_argument("--noise-profile", default="paper_n2_linear")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(
        json.dumps({"best_csv": str(args.best_csv), "arguments": vars(args)}, indent=2),
        encoding="utf-8",
    )
    for i, row in enumerate(load_best_rows(args.best_csv), start=1):
        result = run_case(
            args,
            row["trajectory"],
            float(row["solver_gamma"]),
            float(row["drift_gain"]),
            float(row["task_gain"]),
            float(row["activation_power"]),
            args.noise_profile,
        )
        print(
            f"[noise {i}] {result['status']} {result['trajectory']} "
            f"g={float(result['solver_gamma']):g} d={float(result['drift_gain']):g} "
            f"t={float(result['task_gain']):g} p={float(result['activation_power']):g} "
            f"pos={result.get('mean_position_error_m', '')} "
            f"drift={result.get('final_joint_drift_norm_rad', '')} "
            f"res={result.get('final_solver_residual', '')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
