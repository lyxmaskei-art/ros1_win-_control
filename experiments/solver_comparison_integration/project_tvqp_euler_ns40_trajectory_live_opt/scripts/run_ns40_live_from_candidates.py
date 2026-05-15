import argparse
import csv
import json
from pathlib import Path

from run_ns40_clean_live_sweep import run_case


def parse_args():
    parser = argparse.ArgumentParser(description="Run explicit Method2 Euler Ns40 live candidate CSV.")
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--activation-exp-clip", type=float, default=4.0)
    parser.add_argument("--noise-profile", default="clean")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def load_candidates(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(
        json.dumps({"candidates_csv": str(args.candidates_csv), "arguments": vars(args)}, indent=2),
        encoding="utf-8",
    )
    for index, row in enumerate(load_candidates(args.candidates_csv), start=1):
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
            f"[{index}] {result['status']} {result['trajectory']} "
            f"g={float(result['solver_gamma']):g} d={float(result['drift_gain']):g} "
            f"t={float(result['task_gain']):g} p={float(result['activation_power']):g} "
            f"pos={result.get('mean_position_error_m', '')} "
            f"drift={result.get('final_joint_drift_norm_rad', '')} "
            f"res={result.get('final_solver_residual', '')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
