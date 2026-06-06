from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _ros_real_ur3e import (
    DEFAULT_JOINT_NAMES,
    ROSUR3eInterface,
    add_ros_real_arguments,
    ensure_ros_node,
    normalize_ros_real_args,
    save_preflight_report,
)


DEFAULT_THETA_INITIAL = np.array([0.0, -np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0], dtype=float)
DEFAULT_THETA_LIMIT = 2.0 * np.pi * np.ones(6, dtype=float)


def load_experiment_module():
    path = PROJECT_ROOT / "01_clean_repetitive_tracking" / "run_clean_repetitive_tracking.py"
    spec = importlib.util.spec_from_file_location("clean_repetitive_tracking_ros", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load experiment module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values, pct):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, pct))


def build_summary(step_times, tau, missed_deadlines, runtime_s, steps):
    step_times = np.asarray(step_times, dtype=float)
    return {
        "steps": int(steps),
        "tau_s": float(tau),
        "deadline_s": float(tau),
        "runtime_s": float(runtime_s),
        "mean_step_time_s": float(np.mean(step_times)),
        "median_step_time_s": float(np.median(step_times)),
        "p95_step_time_s": percentile(step_times, 95),
        "p99_step_time_s": percentile(step_times, 99),
        "max_step_time_s": float(np.max(step_times)),
        "deadline_miss_count": int(missed_deadlines),
        "deadline_miss_ratio": float(missed_deadlines) / float(max(steps, 1)),
        "timing_scope": "time.perf_counter around controller.step only",
    }


def run_ros_benchmark(args):
    mod = load_experiment_module()
    ensure_ros_node("real_ur3e_step_time_benchmark")
    output_dir = Path(args.output_root) if args.output_root else PROJECT_ROOT / "results" / "realtime" / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    interface = ROSUR3eInterface(args.joint_state_topic, args.command_topic, args.joint_names)
    theta_lower = np.asarray(args.effective_theta_lower, dtype=float)
    theta_upper = np.asarray(args.effective_theta_upper, dtype=float)
    theta_initial = np.asarray(args.effective_theta_initial, dtype=float)
    preflight = interface.build_preflight_report(args, theta_lower, theta_upper)
    preflight["theta_initial_command_rad"] = theta_initial.tolist()
    preflight_path = save_preflight_report(output_dir, preflight)
    print(f"Saved ROS preflight report to {preflight_path}")
    print(json.dumps(preflight, indent=2, ensure_ascii=False))

    if args.check_only:
        return
    if args.dry_run:
        print("Dry-run requested. No initialization move or benchmark command was published.")
        return
    if not preflight.get("command_subscriber_ready", False):
        raise RuntimeError(f"No subscriber connected to {args.command_topic}.")

    robot = mod.UR3eKinematics()
    if hasattr(robot, "tool_offset"):
        robot.tool_offset = np.asarray(args.effective_tcp_offset, dtype=float).copy()
    settings = mod.make_settings(args)
    settings.theta_initial_command = theta_initial.copy()
    controller = mod.METHOD_BUILDERS[args.method](robot, settings)
    mod.apply_config_overrides(controller, args)
    controller.update_joint_limits(theta_lower, theta_upper)

    theta_reference = interface.move_to_joint_positions(
        theta_goal=theta_initial,
        steps=args.settle_steps,
        rate_hz=args.settle_rate,
        max_step=args.max_command_step,
    )
    controller.reset(theta_reference)

    initial_pos = robot.forward_kinematics(theta_reference)[:3, 3]
    trajectory, trajectory_tag = mod.build_trajectory(
        settings.trajectory_name,
        settings.trajectory_period,
        settings.heart_scale,
        initial_pos,
    )

    rospy = ensure_ros_node("real_ur3e_step_time_benchmark")
    total_steps = int(round(float(settings.duration) / float(settings.tau)))
    rate = rospy.Rate(1.0 / float(settings.tau))
    step_times = np.empty(total_steps, dtype=float)
    position_errors = np.empty(total_steps, dtype=float)
    drift_norms = np.empty(total_steps, dtype=float)
    missed_deadlines = 0
    t_start = time.perf_counter()

    for step in range(total_steps):
        if rospy.is_shutdown():
            raise RuntimeError("ROS shut down during benchmark.")
        tk = step * float(settings.tau)
        theta_current = interface.get_joint_positions(max_age=args.state_timeout)
        desired_pos, desired_vel = trajectory.get_pose(tk)
        t0 = time.perf_counter()
        result = controller.step(theta_current, desired_pos, desired_vel, use_feedback=True)
        elapsed = time.perf_counter() - t0
        step_times[step] = elapsed
        if elapsed > float(settings.tau):
            missed_deadlines += 1
        theta_next = np.asarray(result["theta_next"], dtype=float)
        if args.max_command_step is not None and float(args.max_command_step) > 0.0:
            max_step = float(args.max_command_step)
            theta_next = np.clip(theta_next, theta_current - max_step, theta_current + max_step)
        theta_next = np.clip(theta_next, theta_lower, theta_upper)
        position_errors[step] = float(np.linalg.norm(np.asarray(result["current_pos"], dtype=float) - desired_pos))
        drift_norms[step] = float(np.linalg.norm(theta_current - theta_reference))
        interface.publish_joint_positions(theta_next)
        rate.sleep()

    runtime_s = time.perf_counter() - t_start
    summary = build_summary(step_times, settings.tau, missed_deadlines, runtime_s, total_steps)
    summary.update(
        {
            "method": args.method,
            "trajectory_name": trajectory_tag,
            "joint_state_topic": args.joint_state_topic,
            "command_topic": args.command_topic,
            "joint_names": list(args.joint_names),
            "tcp_offset_m_tool_frame": np.asarray(args.effective_tcp_offset, dtype=float).tolist(),
        }
    )

    np.savez_compressed(
        output_dir / "ros_real_step_time_history.npz",
        time_s=np.arange(total_steps, dtype=float) * float(settings.tau),
        step_time_s=step_times,
        position_errors=position_errors,
        drift_norms=drift_norms,
    )
    (output_dir / "ros_real_step_time_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "ros_real_step_time_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote real-time feasibility benchmark to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="ROS real UR3e controller step-time benchmark.")
    parser.add_argument("--method", choices=["method2_dlccznn", "method2_pdnn"], default="method2_dlccznn")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--offline-duration", type=float, default=20.0)
    parser.add_argument("--trajectory-period", type=float, default=10.0)
    parser.add_argument("--trajectory-name", default="circle")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--task-gain", type=float, default=None)
    parser.add_argument("--drift-gain", type=float, default=None)
    parser.add_argument("--solver-gamma", type=float, default=None)
    parser.add_argument("--activation-power", type=float, default=None)
    parser.add_argument("--activation-exp-clip", type=float, default=None)
    parser.add_argument("--drift-feedback-mode", choices=["linear", "nonlinear"], default=None)
    parser.add_argument("--solver-regularization", type=float, default=None)
    parser.add_argument("--theta-dot-limit", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--dlccznn-inner-steps", type=int, default=1)
    parser.add_argument("--pdnn-gain", type=float, default=20.0)
    parser.add_argument("--pdnn-inner-steps", type=int, default=1)
    parser.add_argument("--pdnn-max-gradient-norm", type=float, default=1e4)
    parser.add_argument("--skip-plots", action="store_true")
    add_ros_real_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    args = normalize_ros_real_args(
        args,
        default_theta_initial=DEFAULT_THETA_INITIAL.copy(),
        default_theta_lower=-DEFAULT_THETA_LIMIT.copy(),
        default_theta_upper=DEFAULT_THETA_LIMIT.copy(),
    )
    if len(args.joint_names) != len(DEFAULT_JOINT_NAMES):
        raise ValueError("--joint-names must contain 6 joint names.")
    run_ros_benchmark(args)


if __name__ == "__main__":
    main()
