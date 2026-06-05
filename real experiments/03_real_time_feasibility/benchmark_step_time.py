from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(r"C:\Users\lyx\Desktop\sci\new experiments\project_4_4_circle_solver_family_comparison")
OUT = Path(r"C:\Users\lyx\Desktop\sci\paper_artifacts\section4_4\step_time_benchmark")
OUT.mkdir(parents=True, exist_ok=True)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args_namespace(**overrides):
    defaults = dict(
        duration=10.0,
        offline_duration=20.0,
        trajectory_period=10.0,
        tau=0.005,
        trajectory_name="circle",
        task_gain=None,
        drift_gain=None,
        solver_gamma=None,
        activation_power=None,
        activation_exp_clip=None,
        drift_feedback_mode=None,
        solver_regularization=None,
        theta_dot_limit=None,
        eta=None,
        dlccznn_inner_steps=1,
        pdnn_gain=None,
        pdnn_inner_steps=1,
        pdnn_max_gradient_norm=1e4,
        internal_disturbance="none",
        disturbance_scale=1.0,
        skip_plots=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def args_for_case(base_args, case: str):
    values = vars(base_args).copy()
    values["internal_disturbance"] = "linear" if case == "linear_0p5t" else "none"
    return SimpleNamespace(**values)


def benchmark_method(mod, method: str, case: str, args):
    robot = mod.UR3eKinematics()
    run_args = args_for_case(args, case)
    settings = mod.make_settings(run_args)
    builder = mod.METHOD_BUILDERS[method]
    controller = builder(robot, settings)
    mod.apply_config_overrides(controller, run_args)

    theta_reference = settings.theta_initial_command.copy()
    theta_current = theta_reference.copy()
    controller.reset(theta_reference)

    initial_pos = robot.forward_kinematics(theta_reference)[:3, 3]
    trajectory, trajectory_tag = mod.build_trajectory(
        settings.trajectory_name,
        settings.trajectory_period,
        settings.heart_scale,
        initial_pos,
    )

    total_steps = int(round(settings.offline_duration / settings.tau))
    times = np.empty(total_steps, dtype=float)
    position_errors = np.empty(total_steps, dtype=float)
    drift_norms = np.empty(total_steps, dtype=float)

    for step in range(total_steps):
        tk = step * settings.tau
        desired_pos, desired_vel = trajectory.get_pose(tk)
        t0 = time.perf_counter()
        result = controller.step(theta_current, desired_pos, desired_vel, use_feedback=True, t_current=tk)
        times[step] = time.perf_counter() - t0
        position_errors[step] = float(np.linalg.norm(result["current_pos"] - desired_pos))
        drift_norms[step] = float(np.linalg.norm(theta_current - theta_reference))
        theta_current = result["theta_next"].copy()

    return {
        "time_s": np.arange(total_steps, dtype=float) * settings.tau,
        "step_time_s": times,
        "position_errors": position_errors,
        "drift_norms": drift_norms,
        "method": method,
        "solver_type": getattr(controller.cfg, "solver_type", ""),
        "case": case,
        "trajectory": trajectory_tag,
        "settings": {
            "tau": settings.tau,
            "offline_duration": settings.offline_duration,
            "trajectory_name": settings.trajectory_name,
            "trajectory_period": settings.trajectory_period,
            "task_gain": getattr(controller.cfg, "task_gain", ""),
            "drift_gain": getattr(controller.cfg, "drift_gain", ""),
            "drift_feedback_mode": getattr(controller.cfg, "drift_feedback_mode", ""),
            "solver_gamma": getattr(controller.cfg, "solver_gamma", ""),
            "activation_power": getattr(controller.cfg, "activation_power", ""),
            "activation_exp_clip": getattr(controller.cfg, "activation_exp_clip", ""),
            "dlccznn_inner_steps": getattr(controller.cfg, "dlccznn_inner_steps", ""),
            "pdnn_gain": getattr(controller.cfg, "pdnn_gain", ""),
            "pdnn_inner_steps": getattr(controller.cfg, "pdnn_inner_steps", ""),
            "internal_disturbance": getattr(controller.cfg, "internal_disturbance", ""),
            "disturbance_scale": getattr(controller.cfg, "disturbance_scale", ""),
        },
    }


def save_result(result, label: str):
    stem = f"{result['case']}_{label}"
    np.savez(
        OUT / f"{stem}_step_time.npz",
        time_s=result["time_s"],
        step_time_s=result["step_time_s"],
        position_errors=result["position_errors"],
        drift_norms=result["drift_norms"],
    )
    (OUT / f"{stem}_step_time_meta.json").write_text(
        json.dumps(
            {
                "case": result["case"],
                "method": label,
                "internal_method": result["method"],
                "solver_type": result["solver_type"],
                "trajectory": result["trajectory"],
                "settings": result["settings"],
                "timing_scope": "time.perf_counter around controller.step(theta_current, desired_pos, desired_vel, use_feedback=True, t_current=tk)",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "case": result["case"],
        "method": label,
        "internal_method": result["method"],
        "solver_type": result["solver_type"],
        "mean_step_time_s": float(np.mean(result["step_time_s"])),
        "median_step_time_s": float(np.median(result["step_time_s"])),
        "p95_step_time_s": float(np.percentile(result["step_time_s"], 95)),
        "max_step_time_s": float(np.max(result["step_time_s"])),
        "steps": int(result["step_time_s"].size),
        "timing_scope": "controller.step",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["clean", "linear_0p5t", "all"], default="all")
    ns = parser.parse_args()

    corrected = load_module(
        ROOT / "02_corrected_znn_type_comparison" / "scripts" / "run_offline.py",
        "corrected_run_offline",
    )
    pdnn = load_module(
        ROOT / "01_pdnn_lvi_slvi_dlccznn_comparison" / "scripts" / "run_offline.py",
        "pdnn_run_offline",
    )

    jobs = [
        (corrected, "method2_dlccznn", "JDF-DLCCZNN", args_namespace(dlccznn_inner_steps=30, solver_gamma=3072.0, drift_gain=5.0, task_gain=340.0, activation_power=0.9, activation_exp_clip=4.0, pdnn_gain=20.0)),
        (corrected, "method2_cznn", "CZNN", args_namespace(dlccznn_inner_steps=30, solver_gamma=3072.0, drift_gain=5.0, task_gain=340.0, activation_power=0.9, activation_exp_clip=4.0, pdnn_gain=20.0)),
        (corrected, "method2_ftznn", "FTZNN", args_namespace(dlccznn_inner_steps=30, solver_gamma=3072.0, drift_gain=5.0, task_gain=340.0, activation_power=0.9, activation_exp_clip=4.0, pdnn_gain=20.0)),
        (corrected, "method2_fxtznn", "FXTZNN", args_namespace(dlccznn_inner_steps=30, solver_gamma=3072.0, drift_gain=5.0, task_gain=340.0, activation_power=0.9, activation_exp_clip=4.0, pdnn_gain=20.0)),
        (pdnn, "method2_tvqp_lvi_pdnn", "TVQP-LVI PDNN", args_namespace(pdnn_inner_steps=30, pdnn_gain=3072.0, drift_gain=5.0, task_gain=340.0, activation_power=0.9, activation_exp_clip=4.0)),
        (pdnn, "method2_tvqp_slvi_pdnn", "TVQP-S-LVI PDNN", args_namespace(pdnn_inner_steps=30, pdnn_gain=3072.0, drift_gain=5.0, task_gain=340.0, activation_power=0.9, activation_exp_clip=4.0)),
    ]
    cases = ["clean", "linear_0p5t"] if ns.case == "all" else [ns.case]

    rows = []
    failures = []
    for case in cases:
        for mod, method, label, args in jobs:
            print(f"Benchmarking {case} {label}...")
            try:
                result = benchmark_method(mod, method, case, args)
                rows.append(save_result(result, label))
            except Exception as exc:
                print(f"  FAILED: {case} {label}: {exc}")
                failures.append({
                    "case": case,
                    "method": label,
                    "internal_method": method,
                    "error": repr(exc),
                })

    if rows:
        with (OUT / "step_time_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if failures:
        with (OUT / "step_time_failures.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(failures[0].keys()))
            writer.writeheader()
            writer.writerows(failures)
    print(f"Wrote step time benchmark to {OUT}")


if __name__ == "__main__":
    main()
