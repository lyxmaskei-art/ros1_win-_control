from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _ros_real_ur3e import (
    DEFAULT_JOINT_NAMES,
    ROSUR3eInterface,
    add_ros_real_arguments,
    ensure_ros_node,
    evaluate_preflight_report,
    import_ros_deps,
    normalize_ros_real_args,
    save_preflight_report,
)


DEFAULT_THETA_INITIAL = np.array([0.0, -np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0], dtype=float)
DEFAULT_THETA_LIMIT = 2.0 * np.pi * np.ones(6, dtype=float)


def _stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def collect_joint_state_rate(joint_state_topic, joint_names, sample_duration_s):
    rospy, JointState, *_ = import_ros_deps()
    intervals = []
    stamp_ages = []
    samples = 0
    accepted_samples = 0
    missing_joint_samples = 0
    last_wall_time = None

    def callback(msg):
        nonlocal samples, accepted_samples, missing_joint_samples, last_wall_time
        samples += 1
        names = set(msg.name or [])
        if any(name not in names for name in joint_names):
            missing_joint_samples += 1
            return
        accepted_samples += 1
        now_wall = time.monotonic()
        if last_wall_time is not None:
            intervals.append(now_wall - last_wall_time)
        last_wall_time = now_wall
        try:
            if msg.header.stamp != rospy.Time():
                stamp_ages.append((rospy.Time.now() - msg.header.stamp).to_sec())
        except Exception:
            pass

    subscriber = rospy.Subscriber(joint_state_topic, JointState, callback, queue_size=10)
    deadline = time.monotonic() + float(sample_duration_s)
    rate = rospy.Rate(100)
    try:
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            rate.sleep()
    finally:
        subscriber.unregister()

    interval_stats = _stats(intervals)
    rate_hz = None
    if interval_stats["median"] not in (None, 0.0):
        rate_hz = 1.0 / float(interval_stats["median"])
    return {
        "topic": str(joint_state_topic),
        "sample_duration_s": float(sample_duration_s),
        "samples": int(samples),
        "accepted_samples": int(accepted_samples),
        "missing_joint_samples": int(missing_joint_samples),
        "median_rate_hz": rate_hz,
        "arrival_interval_s": interval_stats,
        "ros_stamp_age_s": _stats(stamp_ages),
    }


def main():
    parser = argparse.ArgumentParser(description="Check ROS real UR3e topics and controllers before moving hardware.")
    add_ros_real_arguments(parser)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-duration", type=float, default=2.0)
    parser.add_argument("--node-name", default="ros_real_ur3e_preflight_check")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit with code 2 unless the preflight verdict is pass.",
    )
    args = parser.parse_args()
    args = normalize_ros_real_args(args, DEFAULT_THETA_INITIAL, -DEFAULT_THETA_LIMIT, DEFAULT_THETA_LIMIT)

    ensure_ros_node(args.node_name)
    interface = ROSUR3eInterface(
        args.joint_state_topic,
        args.command_topic,
        args.joint_names,
        command_mode=args.command_mode,
        trajectory_command_topic=args.trajectory_command_topic,
        velocity_command_topic=args.velocity_command_topic,
        trajectory_command_duration=args.trajectory_command_duration,
        tcp_pose_topic=args.tcp_pose_topic,
    )
    preflight = interface.build_preflight_report(
        args,
        np.asarray(args.effective_theta_lower, dtype=float),
        np.asarray(args.effective_theta_upper, dtype=float),
    )
    preflight["joint_state_rate_probe"] = collect_joint_state_rate(
        args.joint_state_topic,
        args.joint_names,
        args.sample_duration,
    )
    preflight["preflight_verdict"] = evaluate_preflight_report(preflight, tau=args.tau)
    out_dir = Path(args.output_dir) if args.output_dir else Path("results") / "preflight" / time.strftime("%Y%m%d_%H%M%S")
    path = save_preflight_report(out_dir, preflight)
    print(json.dumps({"preflight_report": str(path), "report": preflight}, indent=2, ensure_ascii=False))
    if args.require_pass and preflight["preflight_verdict"]["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
