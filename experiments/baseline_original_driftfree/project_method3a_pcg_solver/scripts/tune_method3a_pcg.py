import argparse
import csv
import json
from itertools import product

import run_offline as method3a


TUNING_STAMP = "20260421"
TUNING_ROOT = method3a.RESULTS_DIR / f"tuning_{TUNING_STAMP}" / "method3a_pcg"
ANALYSIS_DIR = TUNING_ROOT / "analysis"
BROAD_DIR = TUNING_ROOT / "offline_broad_runs"
REFINE_DIR = TUNING_ROOT / "offline_refined_runs"
LIVE_DIR = TUNING_ROOT / "live_selected_runs"

BROAD_GRID = {
    "task_gain": [180.0, 198.0, 216.0],
    "position_weight": [8000.0, 10000.0, 15000.0],
    "drift_weight": [5e-4, 1e-3, 2e-3],
    "regularization_gain": [1e-10, 1e-9, 1e-8],
    "solver_max_iters": [4, 5, 6],
}


def token(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3g}".replace("-", "m").replace(".", "p").replace("+", "")


def make_case_id(params):
    return (
        f"tg{token(params['task_gain'])}_"
        f"pw{token(params['position_weight'])}_"
        f"dw{token(params['drift_weight'])}_"
        f"rg{token(params['regularization_gain'])}_"
        f"it{int(params['solver_max_iters'])}_"
        f"jac{token(params['use_jacobi_preconditioner'])}"
    )


def summarize_case(case_id, params, summary):
    row = {
        "case_id": case_id,
        **params,
        "status": summary.get("status", "unknown"),
        "mean_position_error_m": summary.get("mean_position_error_m", float("inf")),
        "final_position_error_m": summary.get("final_position_error_m", float("inf")),
        "final_joint_drift_norm_rad": summary.get("final_joint_drift_norm_rad", float("inf")),
        "mean_inner_residual_norm": summary.get("mean_inner_residual_norm", float("inf")),
        "mean_inner_iterations": summary.get("mean_inner_iterations", float("inf")),
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
        "mean_inner_iterations",
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
        norm_iters = float(row["mean_inner_iterations"]) / mins["mean_inner_iterations"]
        row["composite_score"] = (
            0.30 * norm_mean
            + 0.20 * norm_final
            + 0.30 * norm_drift
            + 0.10 * norm_residual
            + 0.10 * norm_iters
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
    cfg = method3a.Method3aPCGConfig(
        offline_duration=offline_duration,
        task_gain=params["task_gain"],
        position_weight=params["position_weight"],
        drift_weight=params["drift_weight"],
        regularization_gain=params["regularization_gain"],
        solver_max_iters=params["solver_max_iters"],
        solver_tol=1e-12,
        solver_reg=1e-12,
        use_jacobi_preconditioner=params["use_jacobi_preconditioner"],
    )
    robot = method3a.UR3eKinematics()
    return method3a.run_offline_experiment(cfg, robot, output_root / "offline", save_artifacts=False)


def run_live_case(params, output_root, live_duration):
    cfg = method3a.Method3aPCGConfig(
        duration=live_duration,
        task_gain=params["task_gain"],
        position_weight=params["position_weight"],
        drift_weight=params["drift_weight"],
        regularization_gain=params["regularization_gain"],
        solver_max_iters=params["solver_max_iters"],
        solver_tol=1e-12,
        solver_reg=1e-12,
        use_jacobi_preconditioner=params["use_jacobi_preconditioner"],
    )
    robot = method3a.UR3eKinematics()
    return method3a.run_live_experiment(cfg, robot, output_root / "live", save_artifacts=False)


def run_broad_sweep(offline_duration):
    rows = []
    for task_gain, position_weight, drift_weight, regularization_gain, solver_max_iters in product(
        BROAD_GRID["task_gain"],
        BROAD_GRID["position_weight"],
        BROAD_GRID["drift_weight"],
        BROAD_GRID["regularization_gain"],
        BROAD_GRID["solver_max_iters"],
    ):
        params = {
            "task_gain": float(task_gain),
            "position_weight": float(position_weight),
            "drift_weight": float(drift_weight),
            "regularization_gain": float(regularization_gain),
            "solver_max_iters": int(solver_max_iters),
            "use_jacobi_preconditioner": True,
        }
        case_id = make_case_id(params)
        summary = run_offline_case(params, BROAD_DIR / case_id, offline_duration)
        rows.append(summarize_case(case_id, params, summary))
        print(
            f"[offline-broad] {case_id}: "
            f"mean={summary['mean_position_error_m']:.6e}, "
            f"drift={summary['final_joint_drift_norm_rad']:.6e}, "
            f"iters={summary['mean_inner_iterations']:.3f}"
        )

    rows = add_composite_scores(rows)
    write_ranked_outputs(
        rows,
        ANALYSIS_DIR / "offline_broad_ranked.json",
        ANALYSIS_DIR / "offline_broad_ranked.csv",
    )
    return rows


def build_refined_grid(base_params):
    task_gain_base = float(base_params["task_gain"])
    position_weight_base = float(base_params["position_weight"])
    drift_weight_base = float(base_params["drift_weight"])
    regularization_base = float(base_params["regularization_gain"])
    solver_iters_base = int(base_params["solver_max_iters"])

    task_gain_values = sorted({max(120.0, task_gain_base - 18.0), task_gain_base, task_gain_base + 18.0})
    position_weight_values = sorted({max(4000.0, position_weight_base - 2000.0), position_weight_base, position_weight_base + 2000.0})
    drift_weight_values = sorted({max(1e-5, drift_weight_base / 2.0), drift_weight_base, drift_weight_base * 2.0})
    regularization_values = sorted({max(1e-12, regularization_base / 10.0), regularization_base, regularization_base * 10.0})
    solver_iters_values = sorted({max(2, solver_iters_base - 1), solver_iters_base, solver_iters_base + 1})

    grid = []
    for task_gain, position_weight, drift_weight, regularization_gain, solver_max_iters in product(
        task_gain_values,
        position_weight_values,
        drift_weight_values,
        regularization_values,
        solver_iters_values,
    ):
        grid.append(
            {
                "task_gain": float(task_gain),
                "position_weight": float(position_weight),
                "drift_weight": float(drift_weight),
                "regularization_gain": float(regularization_gain),
                "solver_max_iters": int(solver_max_iters),
                "use_jacobi_preconditioner": True,
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
        print(
            f"[offline-refine] {case_id}: "
            f"mean={summary['mean_position_error_m']:.6e}, "
            f"drift={summary['final_joint_drift_norm_rad']:.6e}, "
            f"iters={summary['mean_inner_iterations']:.3f}"
        )

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
            "position_weight": float(row["position_weight"]),
            "drift_weight": float(row["drift_weight"]),
            "regularization_gain": float(row["regularization_gain"]),
            "solver_max_iters": int(row["solver_max_iters"]),
            "use_jacobi_preconditioner": bool(row["use_jacobi_preconditioner"]),
        }
        case_id = make_case_id(params)
        summary = run_live_case(params, LIVE_DIR / case_id, live_duration)
        live_row = summarize_case(case_id, params, summary)
        rows.append(live_row)
        print(
            f"[live] {case_id}: "
            f"mean={summary['mean_position_error_m']:.6e}, "
            f"drift={summary['final_joint_drift_norm_rad']:.6e}, "
            f"iters={summary['mean_inner_iterations']:.3f}"
        )

    rows = add_composite_scores(rows)
    write_ranked_outputs(
        rows,
        ANALYSIS_DIR / "live_ranked.json",
        ANALYSIS_DIR / "live_ranked.csv",
    )

    best = rows[0]
    (ANALYSIS_DIR / "best_live_method3a_pcg.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Method 3a PCG parameter tuning workflow.")
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
