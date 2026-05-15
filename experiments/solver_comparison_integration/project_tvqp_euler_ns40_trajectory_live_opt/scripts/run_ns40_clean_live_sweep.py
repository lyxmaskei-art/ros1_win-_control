import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
RUN_CASE = CURRENT_DIR / "run_ns40_euler_live_case.py"


def find_workspace_root(start_dir):
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "experiments").is_dir() and (
            (candidate / "sim").is_dir() or (candidate / "code" / "sim").is_dir()
        ):
            return candidate
    raise RuntimeError(f"Cannot locate workspace root from {start_dir}")


WORKSPACE_ROOT = find_workspace_root(CURRENT_DIR)


def parse_csv_values(text, cast):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return values


def safe_value(value):
    text = str(value)
    for old, new in (("-", "m"), (".", "p"), ("+", "p"), (" ", ""), ("/", "_"), ("\\", "_"), (":", "_")):
        text = text.replace(old, new)
    return "".join(ch for ch in text if ch.isalnum() or ch == "_")


def case_name(trajectory, gamma, drift_gain, task_gain, power, noise_profile):
    return (
        f"{trajectory}__euler__ns40"
        f"__g{safe_value(gamma)}"
        f"__d{safe_value(drift_gain)}"
        f"__t{safe_value(task_gain)}"
        f"__p{safe_value(power)}"
        f"__noise{safe_value(noise_profile)}"
    )


def summary_path(output_root, trajectory):
    return (
        output_root
        / "single"
        / trajectory
        / "method2_dlccznn"
        / "live"
        / "method2_dlccznn_live_with_fb_summary.json"
    )


def build_command(args, output_root, trajectory, gamma, drift_gain, task_gain, power, noise_profile):
    return [
        sys.executable,
        str(RUN_CASE),
        "--experiment",
        "single",
        "--method",
        "method2_dlccznn",
        "--trajectory-name",
        str(trajectory),
        "--duration",
        str(args.duration),
        "--tau",
        str(args.tau),
        "--task-gain",
        str(task_gain),
        "--drift-gain",
        str(drift_gain),
        "--solver-gamma",
        str(gamma),
        "--activation-power",
        str(power),
        "--activation-exp-clip",
        str(args.activation_exp_clip),
        "--drift-feedback-mode",
        "nonlinear",
        "--dlccznn-inner-steps",
        "40",
        "--dlccznn-integrator",
        "euler",
        "--noise-profile",
        str(noise_profile),
        "--output-root",
        str(output_root),
        "--skip-plots",
    ]


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def row_from_summary(data, extra):
    row = dict(extra)
    row.update(
        {
            "mean_position_error_m": data.get("mean_position_error_m", ""),
            "final_position_error_m": data.get("final_position_error_m", ""),
            "max_position_error_m": data.get("max_position_error_m", ""),
            "final_joint_drift_norm_rad": data.get("final_joint_drift_norm_rad", ""),
            "max_joint_drift_rad": data.get("max_joint_drift_rad", ""),
            "mean_task_residual": data.get("mean_task_residual", ""),
            "final_task_residual": data.get("final_task_residual", ""),
            "mean_solver_residual": data.get("mean_solver_residual", ""),
            "final_solver_residual": data.get("final_solver_residual", ""),
            "min_boundary_slack": data.get("min_boundary_slack", ""),
            "runtime_s": data.get("runtime_s", ""),
            "avg_step_time_s": data.get("avg_step_time_s", ""),
            "summary_path": extra.get("summary_path", ""),
            "joint_drift_table": json.dumps(data.get("joint_drift_table", []), ensure_ascii=False),
        }
    )
    return row


def run_case(args, trajectory, gamma, drift_gain, task_gain, power, noise_profile):
    output_root = Path(args.output_root) / case_name(trajectory, gamma, drift_gain, task_gain, power, noise_profile)
    summary = summary_path(output_root, trajectory)
    log_path = output_root / "case.log"
    meta_path = output_root / "case_meta.json"
    manifest = Path(args.output_root) / "sweep_manifest.csv"
    command = build_command(args, output_root, trajectory, gamma, drift_gain, task_gain, power, noise_profile)

    extra = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "",
        "trajectory": trajectory,
        "integrator": "euler",
        "ns": 40,
        "solver_gamma": gamma,
        "drift_gain": drift_gain,
        "task_gain": task_gain,
        "activation_power": power,
        "noise_profile": noise_profile,
        "duration": args.duration,
        "tau": args.tau,
        "exit_code": "",
        "runtime_wall_s": "",
        "summary_path": str(summary),
        "log_path": str(log_path),
    }
    if summary.exists() and not args.rerun_completed:
        data = json.loads(summary.read_text(encoding="utf-8"))
        extra["status"] = "skipped_completed"
        extra["exit_code"] = 0
        row = row_from_summary(data, extra)
        append_csv(manifest, row)
        return row

    output_root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n\n")
        log.flush()
        proc = subprocess.run(command, cwd=WORKSPACE_ROOT, stdout=log, stderr=subprocess.STDOUT)
    runtime_wall_s = time.perf_counter() - t0
    extra["exit_code"] = proc.returncode
    extra["runtime_wall_s"] = f"{runtime_wall_s:.3f}"
    if summary.exists() and proc.returncode == 0:
        data = json.loads(summary.read_text(encoding="utf-8"))
        extra["status"] = "fully_completed"
        row = row_from_summary(data, extra)
    else:
        extra["status"] = "unfinished"
        row = row_from_summary({}, extra)
    meta_path.write_text(json.dumps({"row": row, "command": command}, indent=2, ensure_ascii=False), encoding="utf-8")
    append_csv(manifest, row)
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Strict resumable clean live sweep for Method2 Euler Ns40.")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results" / "clean_live_ns40_sweep"))
    parser.add_argument("--trajectories", default="small_circle,line,circle,ellipse,figure8")
    parser.add_argument("--solver-gammas", default="3072,3584,4096,4608")
    parser.add_argument("--drift-gains", default="40,60,80")
    parser.add_argument("--task-gains", default="300,340,380")
    parser.add_argument("--activation-powers", default="0.72,0.75,0.78")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--activation-exp-clip", type=float, default=4.0)
    parser.add_argument("--noise-profile", default="clean")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    (Path(args.output_root) / "run_config.json").write_text(
        json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"), "arguments": vars(args)}, indent=2),
        encoding="utf-8",
    )
    trajectories = parse_csv_values(args.trajectories, str)
    gammas = parse_csv_values(args.solver_gammas, float)
    drift_gains = parse_csv_values(args.drift_gains, float)
    task_gains = parse_csv_values(args.task_gains, float)
    powers = parse_csv_values(args.activation_powers, float)
    total = 0
    for trajectory in trajectories:
        for gamma in gammas:
            for drift_gain in drift_gains:
                for task_gain in task_gains:
                    for power in powers:
                        if args.max_cases is not None and total >= args.max_cases:
                            return
                        total += 1
                        row = run_case(args, trajectory, gamma, drift_gain, task_gain, power, args.noise_profile)
                        print(
                            f"[{total}] {row['status']} {trajectory} ns=40 "
                            f"g={gamma:g} d={drift_gain:g} t={task_gain:g} p={power:g} "
                            f"pos={row.get('mean_position_error_m', '')} "
                            f"drift={row.get('final_joint_drift_norm_rad', '')} "
                            f"res={row.get('final_solver_residual', '')}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
