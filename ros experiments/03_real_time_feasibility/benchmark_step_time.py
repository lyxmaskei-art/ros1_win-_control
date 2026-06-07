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
    StopMotionGuard,
    _limit_vector_norm,
    add_ros_real_arguments,
    ensure_ros_node,
    format_preflight_failure,
    normalize_ros_real_args,
    save_ros_cycle_diagnostics,
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

    interface = ROSUR3eInterface(
        args.joint_state_topic,
        args.command_topic,
        args.joint_names,
        command_mode=args.command_mode,
        trajectory_command_topic=args.trajectory_command_topic,
        velocity_command_topic=args.velocity_command_topic,
        trajectory_command_duration=args.trajectory_command_duration,
        trajectory_window_duration=args.trajectory_window_duration,
        trajectory_window_points=args.trajectory_window_points,
        tcp_pose_topic=args.tcp_pose_topic,
    )
    stop_guard = StopMotionGuard(interface)
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
    preflight_verdict = preflight.get("preflight_verdict", {})
    if preflight_verdict.get("status") == "fail":
        raise RuntimeError(format_preflight_failure(preflight_verdict))
    if not preflight.get("command_subscriber_ready", False):
        raise RuntimeError(f"No subscriber connected to {interface.command_topic}.")

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
    last_commanded_velocity = None
    last_commanded_accel = None
    last_limited_velocity = None
    last_limited_accel = None
    last_cycle_start = None
    cycle_diagnostics = []

    for step in range(total_steps):
        if rospy.is_shutdown():
            raise RuntimeError("ROS shut down during benchmark.")
        cycle_start = time.perf_counter()
        tk = step * float(settings.tau)
        state_read_start = time.perf_counter()
        theta_current = interface.get_joint_positions(max_age=args.state_timeout)
        state_read_s = time.perf_counter() - state_read_start
        state_age_s = interface.get_latest_state_age()
        desired_pos, desired_vel = trajectory.get_pose(tk)
        t0 = time.perf_counter()
        result = controller.step(theta_current, desired_pos, desired_vel, use_feedback=True)
        elapsed = time.perf_counter() - t0
        step_times[step] = elapsed
        if elapsed > float(settings.tau):
            missed_deadlines += 1
        theta_next = np.asarray(result["theta_next"], dtype=float)
        limited_velocity = np.asarray(result.get("theta_dot_next", np.zeros_like(theta_current)), dtype=float)
        raw_limited_velocity = limited_velocity.copy()
        if args.max_command_accel is not None and float(args.max_command_accel) > 0.0:
            if last_limited_velocity is None:
                last_limited_velocity = limited_velocity.copy()
            accel = (limited_velocity - last_limited_velocity) / float(settings.tau)
            accel = _limit_vector_norm(accel, float(args.max_command_accel))
            if args.max_command_jerk is not None and float(args.max_command_jerk) > 0.0:
                if last_limited_accel is None:
                    last_limited_accel = accel.copy()
                jerk = (accel - last_limited_accel) / float(settings.tau)
                jerk = _limit_vector_norm(jerk, float(args.max_command_jerk))
                accel = last_limited_accel + float(settings.tau) * jerk
                accel = _limit_vector_norm(accel, float(args.max_command_accel))
                last_limited_accel = accel.copy()
            limited_velocity = last_limited_velocity + float(settings.tau) * accel
            theta_next = theta_current + float(settings.tau) * limited_velocity
            last_limited_velocity = limited_velocity.copy()
        if args.max_command_step is not None and float(args.max_command_step) > 0.0:
            max_step = float(args.max_command_step)
            theta_next = np.clip(theta_next, theta_current - max_step, theta_current + max_step)
        theta_next = np.clip(theta_next, theta_lower, theta_upper)
        target_delta = theta_next - theta_current
        commanded_velocity = target_delta / float(settings.tau)
        if args.max_command_accel is not None and float(args.max_command_accel) > 0.0:
            last_limited_velocity = commanded_velocity.copy()
        position_errors[step] = float(np.linalg.norm(np.asarray(result["current_pos"], dtype=float) - desired_pos))
        drift_norms[step] = float(np.linalg.norm(theta_current - theta_reference))
        if last_commanded_velocity is None:
            commanded_accel = np.zeros_like(commanded_velocity)
        else:
            commanded_accel = (commanded_velocity - last_commanded_velocity) / float(settings.tau)
        if last_commanded_accel is None:
            commanded_jerk = np.zeros_like(commanded_accel)
        else:
            commanded_jerk = (commanded_accel - last_commanded_accel) / float(settings.tau)
        last_commanded_velocity = commanded_velocity.copy()
        last_commanded_accel = commanded_accel.copy()
        feedback_velocity = interface.get_latest_velocity()
        feedback_velocity_norm = (
            float("nan")
            if feedback_velocity is None
            else float(np.linalg.norm(np.asarray(feedback_velocity, dtype=float)))
        )
        publish_start = time.perf_counter()
        if args.command_mode == "velocity_array":
            interface.publish_joint_velocities(commanded_velocity)
        else:
            interface.publish_joint_positions(
                theta_next,
                duration_s=settings.tau,
                joint_velocities=commanded_velocity,
            )
        publish_s = time.perf_counter() - publish_start
        work_end = time.perf_counter()
        rate.sleep()
        sleep_end = time.perf_counter()
        cycle_work_s = work_end - cycle_start
        sleep_s = sleep_end - work_end
        cycle_period_s = float("nan") if last_cycle_start is None else cycle_start - last_cycle_start
        last_cycle_start = cycle_start
        cycle_diagnostics.append(
            {
                "step": int(step),
                "time_s": float(tk),
                "cycle_period_s": float(cycle_period_s),
                "cycle_work_s": float(cycle_work_s),
                "state_read_s": float(state_read_s),
                "state_age_s": float(state_age_s),
                "controller_step_s": float(elapsed),
                "publish_s": float(publish_s),
                "sleep_s": float(sleep_s),
                "command_step_norm_rad": float(np.linalg.norm(target_delta)),
                "command_step_max_abs_rad": float(np.max(np.abs(target_delta))),
                "command_velocity_norm_rad_s": float(np.linalg.norm(commanded_velocity)),
                "command_accel_norm_rad_s2": float(np.linalg.norm(commanded_accel)),
                "command_jerk_norm_rad_s3": float(np.linalg.norm(commanded_jerk)),
                "raw_command_velocity_norm_rad_s": float(np.linalg.norm(raw_limited_velocity)),
                "feedback_velocity_norm_rad_s": feedback_velocity_norm,
                "deadline_miss": bool(cycle_work_s > float(settings.tau)),
                "period_overrun": bool(np.isfinite(cycle_period_s) and cycle_period_s > 1.25 * float(settings.tau)),
            }
        )

    stop_guard.stop()
    runtime_s = time.perf_counter() - t_start
    summary = build_summary(step_times, settings.tau, missed_deadlines, runtime_s, total_steps)
    cycle_summary = save_ros_cycle_diagnostics(output_dir, cycle_diagnostics, settings.tau)
    summary.update(
        {
            "method": args.method,
            "trajectory_name": trajectory_tag,
            "joint_state_topic": args.joint_state_topic,
            "command_topic": interface.command_topic,
            "position_command_topic": args.command_topic,
            "trajectory_command_topic": args.trajectory_command_topic,
            "velocity_command_topic": args.velocity_command_topic,
            "command_mode": args.command_mode,
            "trajectory_command_duration_s": interface.get_trajectory_command_duration(settings.tau),
            "joint_names": list(args.joint_names),
            "tcp_offset_m_tool_frame": np.asarray(args.effective_tcp_offset, dtype=float).tolist(),
            "ros_real_cycle_diagnostics": cycle_summary,
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
