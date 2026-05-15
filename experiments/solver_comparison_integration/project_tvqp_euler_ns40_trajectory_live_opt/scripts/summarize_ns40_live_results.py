import argparse
import csv
import json
import math
from pathlib import Path


TRAJECTORIES = ["small_circle", "line", "circle", "ellipse", "figure8"]


def read_rows(path):
    rows = []
    if not Path(path).exists():
        return rows
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def as_float(row, key, default=math.inf):
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def score(row):
    pos = as_float(row, "mean_position_error_m")
    drift = as_float(row, "final_joint_drift_norm_rad")
    residual = as_float(row, "final_solver_residual")
    if not all(math.isfinite(v) for v in (pos, drift, residual)):
        return math.inf
    return math.log10(max(pos, 1e-300)) + 0.25 * math.log10(max(drift, 1e-300)) + 0.10 * math.log10(max(residual, 1e-300))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sci(value):
    try:
        return f"{float(value):.3e}"
    except Exception:
        return ""


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def best_by_trajectory(rows):
    best = []
    for trajectory in TRAJECTORIES:
        candidates = [
            row
            for row in rows
            if row.get("trajectory") == trajectory and row.get("status") in ("fully_completed", "skipped_completed")
        ]
        if not candidates:
            continue
        chosen = sorted(candidates, key=score)[0]
        chosen = dict(chosen)
        chosen["composite_score"] = score(chosen)
        best.append(chosen)
    return best


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize strict Ns40 live sweep and noise replay results.")
    parser.add_argument("--clean-manifest", required=True)
    parser.add_argument("--noise-manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_rows = read_rows(args.clean_manifest)
    clean_best = best_by_trajectory(clean_rows)
    write_csv(output_dir / "clean_best_by_trajectory.csv", clean_best)

    noise_rows = read_rows(args.noise_manifest) if args.noise_manifest else []
    write_csv(output_dir / "noise_rows.csv", noise_rows)

    paired = []
    for clean in clean_best:
        noise_matches = [
            row
            for row in noise_rows
            if row.get("trajectory") == clean.get("trajectory")
            and row.get("solver_gamma") == clean.get("solver_gamma")
            and row.get("drift_gain") == clean.get("drift_gain")
            and row.get("task_gain") == clean.get("task_gain")
            and row.get("activation_power") == clean.get("activation_power")
        ]
        if not noise_matches:
            continue
        noise = noise_matches[-1]
        paired.append(
            {
                "trajectory": clean["trajectory"],
                "solver_gamma": clean["solver_gamma"],
                "drift_gain": clean["drift_gain"],
                "task_gain": clean["task_gain"],
                "activation_power": clean["activation_power"],
                "clean_mean_position_error_m": clean.get("mean_position_error_m", ""),
                "noise_mean_position_error_m": noise.get("mean_position_error_m", ""),
                "position_ratio": as_float(noise, "mean_position_error_m") / as_float(clean, "mean_position_error_m"),
                "clean_final_joint_drift_norm_rad": clean.get("final_joint_drift_norm_rad", ""),
                "noise_final_joint_drift_norm_rad": noise.get("final_joint_drift_norm_rad", ""),
                "drift_ratio": as_float(noise, "final_joint_drift_norm_rad") / as_float(clean, "final_joint_drift_norm_rad"),
                "clean_final_solver_residual": clean.get("final_solver_residual", ""),
                "noise_final_solver_residual": noise.get("final_solver_residual", ""),
                "residual_ratio": as_float(noise, "final_solver_residual") / as_float(clean, "final_solver_residual"),
            }
        )
    write_csv(output_dir / "clean_vs_noise_paired.csv", paired)

    report = [
        "# Euler Ns40 trajectory-wise live optimization report",
        "",
        "## Clean best by trajectory",
        markdown_table(
            ["trajectory", "gamma", "drift", "task", "power", "mean pos", "drift norm", "residual", "score"],
            [
                [
                    row.get("trajectory", ""),
                    sci(row.get("solver_gamma", "")),
                    sci(row.get("drift_gain", "")),
                    sci(row.get("task_gain", "")),
                    sci(row.get("activation_power", "")),
                    sci(row.get("mean_position_error_m", "")),
                    sci(row.get("final_joint_drift_norm_rad", "")),
                    sci(row.get("final_solver_residual", "")),
                    sci(row.get("composite_score", "")),
                ]
                for row in clean_best
            ],
        ),
    ]
    if paired:
        report.extend(
            [
                "",
                "## Clean vs n(t)=0.5t noise live",
                markdown_table(
                    [
                        "trajectory",
                        "clean pos",
                        "noise pos",
                        "pos ratio",
                        "clean drift",
                        "noise drift",
                        "drift ratio",
                        "clean residual",
                        "noise residual",
                        "res ratio",
                    ],
                    [
                        [
                            row["trajectory"],
                            sci(row["clean_mean_position_error_m"]),
                            sci(row["noise_mean_position_error_m"]),
                            sci(row["position_ratio"]),
                            sci(row["clean_final_joint_drift_norm_rad"]),
                            sci(row["noise_final_joint_drift_norm_rad"]),
                            sci(row["drift_ratio"]),
                            sci(row["clean_final_solver_residual"]),
                            sci(row["noise_final_solver_residual"]),
                            sci(row["residual_ratio"]),
                        ]
                        for row in paired
                    ],
                ),
            ]
        )
    report.extend(
        [
            "",
            "## Supervision checks",
            f"- Clean manifest rows: {len(clean_rows)}.",
            f"- Clean completed/skipped rows: {sum(1 for row in clean_rows if row.get('status') in ('fully_completed', 'skipped_completed'))}.",
            f"- Trajectory-wise clean best rows: {len(clean_best)}.",
            f"- Noise rows: {len(noise_rows)}.",
        ]
    )
    (output_dir / "strict_chinese_ns40_live_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "clean_best": len(clean_best), "paired": len(paired)}, indent=2))


if __name__ == "__main__":
    main()
