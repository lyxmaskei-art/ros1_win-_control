import argparse
import csv
import json
from itertools import product
from pathlib import Path

import run_method2_offline as method2


TUNING_STAMP = "20260421"
TUNING_ROOT = method2.RESULTS_DIR / f"tuning_{TUNING_STAMP}" / "method2"
ANALYSIS_DIR = TUNING_ROOT / "analysis"
BROAD_DIR = TUNING_ROOT / "offline_broad_runs"
REFINE_DIR = TUNING_ROOT / "offline_refined_runs"
LIVE_DIR = TUNING_ROOT / "live_selected_runs"

BROAD_GRID = {
    "task_gain": [120.0, 160.0, 200.0],
    "mu_gain": [6.0, 10.0, 14.0],
    "solver_substeps": [20, 40, 60],
    "solver_gamma_gain": [1280.0, 2560.0, 3840.0],
}


def make_case_id(params):
    return (
        f"tg{int(round(params['task_gain']))}_"
        f"mu{params['mu_gain']:.1f}_"
        f"ap{params['activation_power']:.2f}_"
        f"ns{int(params['solver_substeps'])}_"
        f"gg{int(round(params['solver_gamma_gain']))}"
    ).replace(".", "p")


def summarize_case(case_id, params, summary):
    row = {
        "case_id": case_id,
        **params,
        "status": summary.get("status", "unknown"),
        "mean_position_error_m": summary.get("mean_position_error_m", float("inf")),
        "final_position_error_m": summary.get("final_position_error_m", float("inf")),
        "final_joint_drift_norm_rad": summary.get("final_joint_drift_norm_rad", float("inf")),
        "mean_inner_residual_norm": summary.get("mean_inner_residual_norm", float("inf")),
        "wall_clock_run_s": summary.get("wall_clock_run_s", float("inf")),
        "result_dir": summary.get("result_dir", ""),
    }
    return row


def add_composite_scores(rows):
    valid_rows = [row for row in rows if row["status"] == "ok"]
    if not valid_rows:
        return rows

    metric_keys = [
        "mean_position_error_m",
        "final_position_error_m",
        "final_joint_drift_norm_rad",
        "mean_inner_residual_norm",
    ]
    mins = {}
    for key in metric_keys:
        mins[key] = min(max(float(row[key]), 1e-18) for row in valid_rows)

    for row in rows:
        if row["status"] != "ok":
            row["composite_score"] = float("inf")
            continue
        norm_mean = float(row["mean_position_error_m"]) / mins["mean_position_error_m"]
        norm_final = float(row["final_position_error_m"]) / mins["final_position_error_m"]
        norm_drift = float(row["final_joint_drift_norm_rad"]) / mins["final_joint_drift_norm_rad"]
        norm_residual = float(row["mean_inner_residual_norm"]) / mins["mean_inner_residual_norm"]
        row["composite_score"] = (
            0.35 * norm_mean
            + 0.20 * norm_final
            + 0.35 * norm_drift
            + 0.10 * norm_residual
        )

    rows.sort(key=lambda item: (item["composite_score"], item["mean_position_error_m"], item["final_joint_drift_norm_rad"]))
    return rows


def write_ranked_outputs(rows, json_path, csv_path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_offline_case(params, output_root, offline_duration):
    cfg = method2.Method2DLCCZNNConfig(
        offline_duration=offline_duration,
        task_gain=params["task_gain"],
        mu_gain=params["mu_gain"],
        activation_power=params["activation_power"],
        activation_exp_clip=4.0,
        solver_substeps=params["solver_substeps"],
        solver_gamma_gain=params["solver_gamma_gain"],
        solver_power=0.9,
        solver_exp_clip=4.0,
        solver_lambda=1e-8,
    )
    robot = method2.UR3eKinematics()
    return method2.run_offline_experiment(cfg, robot, output_root / "offline", save_artifacts=False)


def run_live_case(params, output_root, live_duration):
    cfg = method2.Method2DLCCZNNConfig(
        duration=live_duration,
        task_gain=params["task_gain"],
        mu_gain=params["mu_gain"],
        activation_power=params["activation_power"],
        activation_exp_clip=4.0,
        solver_substeps=params["solver_substeps"],
        solver_gamma_gain=params["solver_gamma_gain"],
        solver_power=0.9,
        solver_exp_clip=4.0,
        solver_lambda=1e-8,
    )
    robot = method2.UR3eKinematics()
    return method2.run_live_experiment(cfg, robot, output_root / "live", save_artifacts=False)


def run_broad_sweep(offline_duration):
    rows = []
    for task_gain, mu_gain, solver_substeps, solver_gamma_gain in product(
        BROAD_GRID["task_gain"],
        BROAD_GRID["mu_gain"],
        BROAD_GRID["solver_substeps"],
        BROAD_GRID["solver_gamma_gain"],
    ):
        params = {
            "task_gain": float(task_gain),
            "mu_gain": float(mu_gain),
            "activation_power": 0.90,
            "solver_substeps": int(solver_substeps),
            "solver_gamma_gain": float(solver_gamma_gain),
        }
        case_id = make_case_id(params)
        summary = run_offline_case(params, BROAD_DIR / case_id, offline_duration)
        rows.append(summarize_case(case_id, params, summary))
        print(f"[offline-broad] {case_id}: mean={summary['mean_position_error_m']:.6e}, drift={summary['final_joint_drift_norm_rad']:.6e}")

    rows = add_composite_scores(rows)
    write_ranked_outputs(
        rows,
        ANALYSIS_DIR / "offline_broad_ranked.json",
        ANALYSIS_DIR / "offline_broad_ranked.csv",
    )
    return rows


def build_refined_grid(base_params):
    task_gain_base = float(base_params["task_gain"])
    mu_gain_base = float(base_params["mu_gain"])
    activation_base = float(base_params["activation_power"])
    substeps_base = int(base_params["solver_substeps"])
    gamma_base = float(base_params["solver_gamma_gain"])

    task_gain_values = sorted({max(40.0, task_gain_base - 20.0), task_gain_base, task_gain_base + 20.0})
    mu_gain_values = sorted({max(2.0, mu_gain_base - 2.0), mu_gain_base, mu_gain_base + 2.0})
    activation_values = sorted({max(0.70, round(activation_base - 0.05, 2)), round(activation_base, 2), min(1.00, round(activation_base + 0.05, 2))})
    substeps_values = sorted({max(10, substeps_base - 20), substeps_base, substeps_base + 20})
    gamma_values = sorted({max(256.0, gamma_base - 768.0), gamma_base, gamma_base + 768.0})

    grid = []
    for task_gain, mu_gain, activation_power, solver_substeps, solver_gamma_gain in product(
        task_gain_values,
        mu_gain_values,
        activation_values,
        substeps_values,
        gamma_values,
    ):
        grid.append(
            {
                "task_gain": float(task_gain),
                "mu_gain": float(mu_gain),
                "activation_power": float(activation_power),
                "solver_substeps": int(solver_substeps),
                "solver_gamma_gain": float(solver_gamma_gain),
            }
        )
    return grid


def load_ranked_rows(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_refined_sweep(offline_duration):
    broad_rows = load_ranked_rows(ANALYSIS_DIR / "offline_broad_ranked.json")
    base_params = broad_rows[0]
    grid = build_refined_grid(base_params)
    rows = []
    for params in grid:
        case_id = make_case_id(params)
        summary = run_offline_case(params, REFINE_DIR / case_id, offline_duration)
        rows.append(summarize_case(case_id, params, summary))
        print(f"[offline-refine] {case_id}: mean={summary['mean_position_error_m']:.6e}, drift={summary['final_joint_drift_norm_rad']:.6e}")

    rows = add_composite_scores(rows)
    write_ranked_outputs(
        rows,
        ANALYSIS_DIR / "offline_refined_ranked.json",
        ANALYSIS_DIR / "offline_refined_ranked.csv",
    )
    return rows


def pick_live_candidates(top_n):
    combined = {}
    for path in (ANALYSIS_DIR / "offline_broad_ranked.json", ANALYSIS_DIR / "offline_refined_ranked.json"):
        for row in load_ranked_rows(path):
            combined[row["case_id"]] = row

    ranked = add_composite_scores(list(combined.values()))
    selected = []
    used = set()

    for row in ranked:
        if row["case_id"] in used:
            continue
        selected.append(row)
        used.add(row["case_id"])
        if len(selected) >= top_n:
            break

    drift_best = min(ranked, key=lambda item: item["final_joint_drift_norm_rad"])
    mean_best = min(ranked, key=lambda item: item["mean_position_error_m"])
    for row in (drift_best, mean_best):
        if row["case_id"] not in used:
            selected.append(row)
            used.add(row["case_id"])

    return selected


def run_live_selection(live_duration, top_n):
    candidates = pick_live_candidates(top_n)
    rows = []
    for row in candidates:
        params = {
            "task_gain": float(row["task_gain"]),
            "mu_gain": float(row["mu_gain"]),
            "activation_power": float(row["activation_power"]),
            "solver_substeps": int(row["solver_substeps"]),
            "solver_gamma_gain": float(row["solver_gamma_gain"]),
        }
        case_id = make_case_id(params)
        summary = run_live_case(params, LIVE_DIR / case_id, live_duration)
        live_row = summarize_case(case_id, params, summary)
        rows.append(live_row)
        print(f"[live] {case_id}: mean={summary['mean_position_error_m']:.6e}, drift={summary['final_joint_drift_norm_rad']:.6e}")

    rows = add_composite_scores(rows)
    write_ranked_outputs(
        rows,
        ANALYSIS_DIR / "live_ranked.json",
        ANALYSIS_DIR / "live_ranked.csv",
    )

    best = rows[0]
    (ANALYSIS_DIR / "best_live_method2.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Method 2 DLCCZNN parameter tuning workflow.")
    parser.add_argument("--stage", choices=["offline-broad", "offline-refine", "live", "all"], default="all")
    parser.add_argument("--offline-duration", type=float, default=20.0)
    parser.add_argument("--live-duration", type=float, default=10.0)
    parser.add_argument("--top-live", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in ("offline-broad", "all"):
        run_broad_sweep(args.offline_duration)

    if args.stage in ("offline-refine", "all"):
        run_refined_sweep(args.offline_duration)

    if args.stage in ("live", "all"):
        run_live_selection(args.live_duration, args.top_live)


if __name__ == "__main__":
    main()
