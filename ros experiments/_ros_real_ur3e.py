from __future__ import annotations

import atexit
import csv
import json
import threading
import time
from pathlib import Path

import numpy as np


DEFAULT_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

REAL_SAFE_PRESET = {
    "tcp_offset": "0,0,0.145",
    "task_gain": 240.0,
    "solver_gamma": 30.0,
    "drift_gain": 2.0,
    "theta_dot_limit": 0.4,
    "max_command_accel": 12.0,
}

COMMAND_MODE_TOPIC_TYPES = {
    "position_array": "std_msgs/Float64MultiArray",
    "joint_trajectory": "trajectory_msgs/JointTrajectory",
    "velocity_array": "std_msgs/Float64MultiArray",
}

COMMAND_MODE_CONTROLLER_HINTS = {
    "position_array": "pos_joint_group_controller",
    "joint_trajectory": "scaled_pos_joint_traj_controller",
    "velocity_array": "joint_group_vel_controller",
}


def import_ros_deps():
    try:
        import rospy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    except ImportError as exc:
        raise RuntimeError(
            "ROS live mode requires rospy, geometry_msgs, sensor_msgs, std_msgs, and trajectory_msgs. "
            "Run this script from a sourced ROS1 workspace, for example: "
            "source /opt/ros/noetic/setup.bash && source ~/文档/catkin_ws/devel/setup.bash"
        ) from exc
    return rospy, JointState, Float64MultiArray, JointTrajectory, JointTrajectoryPoint, PoseStamped


def ensure_ros_node(node_name: str):
    rospy, *_ = import_ros_deps()
    if not rospy.core.is_initialized():
        rospy.init_node(node_name, anonymous=False)
    return rospy


def parse_vector_arg(value, size: int, name: str):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        values = [float(v) for v in value]
    else:
        values = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(values) != int(size):
        raise ValueError(f"{name} must contain exactly {size} comma-separated values; got {len(values)}.")
    return np.asarray(values, dtype=float)


def parse_joint_names(value):
    if value is None:
        return DEFAULT_JOINT_NAMES
    names = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if len(names) != 6:
        raise ValueError(f"--joint-names must contain exactly 6 names, got {len(names)}.")
    return names


def save_preflight_report(output_dir, report):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ros_real_preflight_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _finite_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _summarize_ros_cycle_diagnostics(rows, tau):
    if not rows:
        return {}
    keys = (
        "cycle_period_s",
        "cycle_work_s",
        "state_read_s",
        "state_age_s",
        "controller_step_s",
        "publish_s",
        "sleep_s",
        "command_step_norm_rad",
        "command_step_max_abs_rad",
        "command_velocity_norm_rad_s",
        "command_accel_norm_rad_s2",
        "feedback_velocity_norm_rad_s",
        "tcp_pose_age_s",
        "tcp_fk_error_norm_m",
    )
    summary = {key: _finite_stats([row.get(key, float("nan")) for row in rows]) for key in keys}
    deadline_miss_count = sum(1 for row in rows if row.get("deadline_miss", False))
    period_overrun_count = sum(1 for row in rows if row.get("period_overrun", False))
    tcp_pose_used_count = sum(1 for row in rows if row.get("tcp_pose_used", False))
    summary.update(
        {
            "samples": len(rows),
            "tau_s": float(tau),
            "deadline_miss_count": int(deadline_miss_count),
            "deadline_miss_ratio": float(deadline_miss_count) / float(max(len(rows), 1)),
            "period_overrun_count": int(period_overrun_count),
            "period_overrun_ratio": float(period_overrun_count) / float(max(len(rows), 1)),
            "tcp_pose_used_count": int(tcp_pose_used_count),
            "tcp_pose_used_ratio": float(tcp_pose_used_count) / float(max(len(rows), 1)),
            "timing_scope": "whole ROS loop: state read, controller step, publish, and rate sleep",
        }
    )
    return summary


def save_ros_cycle_diagnostics(output_dir, rows, tau):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summarize_ros_cycle_diagnostics(rows, tau)
    if not rows:
        return summary

    fieldnames = list(rows[0].keys())
    with (output_dir / "ros_real_cycle_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "ros_real_cycle_diagnostics.npz",
        **{key: np.asarray([row.get(key, float("nan")) for row in rows], dtype=float) for key in fieldnames if key != "deadline_miss" and key != "period_overrun" and key != "tcp_pose_used"},
        deadline_miss=np.asarray([bool(row.get("deadline_miss", False)) for row in rows], dtype=bool),
        period_overrun=np.asarray([bool(row.get("period_overrun", False)) for row in rows], dtype=bool),
        tcp_pose_used=np.asarray([bool(row.get("tcp_pose_used", False)) for row in rows], dtype=bool),
    )
    (output_dir / "ros_real_cycle_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def inspect_published_topics(rospy):
    try:
        return [
            {"topic": str(topic), "type": str(topic_type)}
            for topic, topic_type in rospy.get_published_topics(namespace="/")
        ]
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def inspect_ros_master_state(rospy):
    report = {
        "available": False,
        "publishers": [],
        "subscribers": [],
        "services": [],
        "error": "",
    }
    try:
        code, message, state = rospy.get_master().getSystemState()
        if code != 1:
            report["error"] = str(message)
            return report
        publishers, subscribers, services = state
        report["available"] = True
        report["publishers"] = [{"topic": str(topic), "nodes": [str(node) for node in nodes]} for topic, nodes in publishers]
        report["subscribers"] = [{"topic": str(topic), "nodes": [str(node) for node in nodes]} for topic, nodes in subscribers]
        report["services"] = [{"service": str(service), "nodes": [str(node) for node in nodes]} for service, nodes in services]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def inspect_ros_controllers(rospy, service_name="/controller_manager/list_controllers", timeout_s=0.5):
    report = {
        "service_name": service_name,
        "available": False,
        "controllers": [],
        "error": "",
    }
    try:
        rospy.wait_for_service(service_name, timeout=float(timeout_s))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    try:
        from controller_manager_msgs.srv import ListControllers

        proxy = rospy.ServiceProxy(service_name, ListControllers)
        response = proxy()
        report["available"] = True
        report["controllers"] = [
            {
                "name": str(controller.name),
                "state": str(controller.state),
                "type": str(controller.type),
                "claimed_resources": [
                    {
                        "hardware_interface": str(resource.hardware_interface),
                        "resources": [str(item) for item in resource.resources],
                    }
                    for resource in getattr(controller, "claimed_resources", [])
                ],
            }
            for controller in response.controller
        ]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _topic_type_lookup(published_topics):
    if not isinstance(published_topics, list):
        return {}
    return {item.get("topic"): item.get("type") for item in published_topics if isinstance(item, dict)}


def _controller_lookup(controller_report):
    controllers = {}
    for item in controller_report.get("controllers", []) if isinstance(controller_report, dict) else []:
        name = item.get("name")
        if name:
            controllers[name] = item
    return controllers


def _subscriber_nodes(master_state, topic):
    if not isinstance(master_state, dict):
        return []
    for item in master_state.get("subscribers", []):
        if item.get("topic") == topic:
            return list(item.get("nodes", []))
    return []


def build_command_mode_recommendations(
    published_topics,
    controller_report,
    *,
    position_command_topic,
    trajectory_command_topic,
    velocity_command_topic,
    master_state=None,
):
    topic_by_mode = {
        "position_array": str(position_command_topic),
        "joint_trajectory": str(trajectory_command_topic),
        "velocity_array": str(velocity_command_topic),
    }
    topic_types = _topic_type_lookup(published_topics)
    controllers = _controller_lookup(controller_report)
    recommendations = []
    for mode, topic in topic_by_mode.items():
        expected_type = COMMAND_MODE_TOPIC_TYPES[mode]
        observed_type = topic_types.get(topic)
        controller_name = COMMAND_MODE_CONTROLLER_HINTS[mode]
        controller = controllers.get(controller_name, {})
        controller_state = controller.get("state", "unknown")
        topic_ok = observed_type in (None, expected_type)
        subscriber_nodes = _subscriber_nodes(master_state, topic)
        running = controller_state == "running"
        subscriber_present = bool(subscriber_nodes)
        available = bool(subscriber_present and topic_ok)
        if mode == "position_array":
            priority = 3
            note = "Fallback mode; dense position streaming can still excite real-controller interpolation."
        elif mode == "joint_trajectory":
            priority = 1
            note = "Prefer this when the trajectory controller is running; commands carry time_from_start."
        else:
            priority = 2
            note = "Useful A/B test when a velocity controller is running; start with short trials."
        if observed_type is not None and observed_type != expected_type:
            note = f"Topic type mismatch: expected {expected_type}, observed {observed_type}."
        elif not subscriber_present:
            note = "No subscriber is registered for this command topic."
        elif controller_report.get("available") and controller_state != "unknown" and not running:
            note = f"Controller {controller_name} is not running."
        recommendations.append(
            {
                "mode": mode,
                "priority": priority,
                "command_topic": topic,
                "expected_topic_type": expected_type,
                "observed_topic_type": observed_type,
                "topic_type_matches": bool(topic_ok),
                "subscriber_nodes": subscriber_nodes,
                "subscriber_present": bool(subscriber_present),
                "controller_name_hint": controller_name,
                "controller_state": controller_state,
                "controller_name_hint_running": bool(running),
                "available": bool(available),
                "note": note,
            }
        )
    recommendations.sort(key=lambda item: (not item["available"], item["priority"]))
    return recommendations


def _add_preflight_issue(issues, severity, title, evidence, action):
    issues.append(
        {
            "severity": severity,
            "title": title,
            "evidence": evidence,
            "action": action,
        }
    )


def _finite_float_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def evaluate_preflight_report(report, tau=None):
    tau = _finite_float_or_none(tau)
    issues = []
    selected_mode = report.get("command_mode", "")
    selected_topic = report.get("command_topic", "")

    if not report.get("command_subscriber_ready", False):
        _add_preflight_issue(
            issues,
            "fail",
            "Selected command topic has no subscriber",
            f"command_topic={selected_topic}",
            "Start the matching UR controller or select the command mode/topic marked available in command_mode_recommendations.",
        )

    selected_recommendation = None
    for item in report.get("command_mode_recommendations", []):
        if item.get("mode") == selected_mode:
            selected_recommendation = item
            break
    if selected_recommendation is not None:
        if not selected_recommendation.get("topic_type_matches", True):
            _add_preflight_issue(
                issues,
                "fail",
                "Selected command topic type does not match command mode",
                (
                    f"mode={selected_mode}, topic={selected_topic}, "
                    f"expected={selected_recommendation.get('expected_topic_type')}, "
                    f"observed={selected_recommendation.get('observed_topic_type')}"
                ),
                "Use the topic whose ROS message type matches the selected command mode.",
            )
        controller_state = selected_recommendation.get("controller_state")
        controller_available = bool(report.get("controller_manager", {}).get("available", False))
        if controller_available and controller_state not in ("running", "unknown"):
            _add_preflight_issue(
                issues,
                "warn",
                "Controller-name hint is not running",
                f"mode={selected_mode}, controller={selected_recommendation.get('controller_name_hint')}, state={controller_state}",
                "If command_subscriber_ready is true, this may be a custom controller name; otherwise switch or start the matching controller.",
            )

    latest_age = _finite_float_or_none(report.get("latest_state_age_s"))
    if latest_age is not None and tau is not None and latest_age > 2.0 * tau:
        _add_preflight_issue(
            issues,
            "warn",
            "Latest joint state is already older than two control periods",
            f"latest_state_age_s={latest_age:.6g}, tau_s={tau:.6g}",
            "Run the non-moving preflight rate probe and reduce the outer-loop rate if feedback is stale.",
        )

    if report.get("tcp_pose_enabled", False):
        topic_types = _topic_type_lookup(report.get("published_topics"))
        observed_tcp_type = topic_types.get(report.get("tcp_pose_topic"))
        if observed_tcp_type is not None and observed_tcp_type != "geometry_msgs/PoseStamped":
            _add_preflight_issue(
                issues,
                "warn",
                "TCP pose topic has an unexpected message type",
                f"tcp_pose_topic={report.get('tcp_pose_topic')}, observed_type={observed_tcp_type}",
                "Use a geometry_msgs/PoseStamped TCP topic or leave --tcp-pose-topic empty to use local FK only.",
            )
        tcp_age = _finite_float_or_none(report.get("latest_tcp_pose_age_s"))
        if report.get("latest_tcp_position_m") is None:
            _add_preflight_issue(
                issues,
                "warn",
                "TCP pose topic is enabled but no pose has been received",
                f"tcp_pose_topic={report.get('tcp_pose_topic')}",
                "Check the topic type is geometry_msgs/PoseStamped or leave --tcp-pose-topic empty to use local FK only.",
            )
        elif tau is not None and tcp_age is not None and tcp_age > 2.0 * tau:
            _add_preflight_issue(
                issues,
                "warn",
                "TCP pose feedback is stale",
                f"latest_tcp_pose_age_s={tcp_age:.6g}, tau_s={tau:.6g}",
                "Use the TCP pose only as offline evidence unless its rate is close to the control loop.",
            )

    rate_probe = report.get("joint_state_rate_probe")
    if isinstance(rate_probe, dict) and tau is not None and tau > 0.0:
        required_rate = 1.0 / tau
        median_rate = _finite_float_or_none(rate_probe.get("median_rate_hz"))
        if median_rate is None:
            _add_preflight_issue(
                issues,
                "fail",
                "Cannot estimate joint-state feedback rate",
                f"accepted_samples={rate_probe.get('accepted_samples')}, sample_duration_s={rate_probe.get('sample_duration_s')}",
                "Fix /joint_states publication and joint-name ordering before moving hardware.",
            )
        elif median_rate < 0.8 * required_rate:
            _add_preflight_issue(
                issues,
                "fail",
                "Joint-state feedback rate is too low for the requested control period",
                f"median_rate_hz={median_rate:.6g}, required_rate_hz={required_rate:.6g}",
                "Increase joint-state publish rate or test a slower outer loop, for example --tau 0.01.",
            )
        elif median_rate < required_rate:
            _add_preflight_issue(
                issues,
                "warn",
                "Joint-state feedback rate is below the requested control rate",
                f"median_rate_hz={median_rate:.6g}, requested_rate_hz={required_rate:.6g}",
                "Expect feedback age and period jitter; compare with --tau 0.01 before tuning gains.",
            )

        interval_p95 = _finite_float_or_none(rate_probe.get("arrival_interval_s", {}).get("p95"))
        if interval_p95 is not None and interval_p95 > 1.5 * tau:
            _add_preflight_issue(
                issues,
                "warn",
                "Joint-state arrival jitter is high",
                f"arrival_interval_s p95={interval_p95:.6g}, tau_s={tau:.6g}",
                "Reduce other ROS load and avoid 200 Hz control until feedback timing is stable.",
            )

        stamp_age_p95 = _finite_float_or_none(rate_probe.get("ros_stamp_age_s", {}).get("p95"))
        if stamp_age_p95 is not None and stamp_age_p95 > 2.0 * tau:
            _add_preflight_issue(
                issues,
                "warn",
                "Joint-state ROS stamps are stale",
                f"ros_stamp_age_s p95={stamp_age_p95:.6g}, tau_s={tau:.6g}",
                "Check ROS networking, time synchronization, and driver load before increasing gains.",
            )

    severity_order = {"fail": 0, "warn": 1, "info": 2}
    issues.sort(key=lambda item: severity_order.get(item["severity"], 99))
    status = "pass"
    if any(item["severity"] == "fail" for item in issues):
        status = "fail"
    elif any(item["severity"] == "warn" for item in issues):
        status = "warn"
    return {
        "status": status,
        "issues": issues,
        "selected_command_mode": selected_mode,
        "selected_command_topic": selected_topic,
        "tau_s": tau,
    }


def format_preflight_failure(verdict):
    issues = verdict.get("issues", []) if isinstance(verdict, dict) else []
    if not issues:
        return "ROS real preflight failed without a detailed issue list."
    lines = ["ROS real preflight failed:"]
    for item in issues:
        if item.get("severity") == "fail":
            lines.append(f"- {item.get('title')}: {item.get('evidence')}. Action: {item.get('action')}")
    return "\n".join(lines)


def add_ros_real_arguments(parser):
    parser.add_argument("--joint-state-topic", type=str, default="/joint_states")
    parser.add_argument("--command-topic", type=str, default="/pos_joint_group_controller/command")
    parser.add_argument("--command-mode", choices=["position_array", "joint_trajectory", "velocity_array"], default="position_array")
    parser.add_argument("--trajectory-command-topic", type=str, default="/scaled_pos_joint_traj_controller/command")
    parser.add_argument("--velocity-command-topic", type=str, default="/joint_group_vel_controller/command")
    parser.add_argument(
        "--tcp-pose-topic",
        type=str,
        default="",
        help="Optional geometry_msgs/PoseStamped topic for measured TCP pose. Leave empty to use local FK only.",
    )
    parser.add_argument("--tcp-pose-timeout", type=float, default=0.05)
    parser.add_argument(
        "--trajectory-command-duration",
        type=float,
        default=None,
        help="time_from_start for one-point JointTrajectory commands. Default: max(2*tau, 0.02).",
    )
    parser.add_argument("--joint-names", type=str, default=",".join(DEFAULT_JOINT_NAMES))
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--state-timeout", type=float, default=0.5)
    parser.add_argument("--command-wait-timeout", type=float, default=3.0)
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--settle-rate", type=float, default=50.0)
    parser.add_argument("--max-command-step", type=float, default=0.01)
    parser.add_argument("--max-command-accel", type=float, default=None)
    parser.add_argument("--theta-initial-command", type=str, default=None)
    parser.add_argument("--theta-lower", type=str, default=None)
    parser.add_argument("--theta-upper", type=str, default=None)
    parser.add_argument(
        "--tcp-offset",
        type=str,
        default=None,
        help="Comma-separated TCP offset in the local tool frame, in meters. Example: 0,0,0.12",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--real-safe-preset",
        action="store_true",
        help=(
            "Fill unset real-machine low-shake defaults: TCP offset, task/solver/drift gains, "
            "theta-dot limit, and command acceleration limit. Explicit CLI values still win."
        ),
    )
    return parser


def _set_if_missing(args, name, value):
    if hasattr(args, name) and getattr(args, name) is None:
        setattr(args, name, value)


def apply_real_safe_preset(args):
    if not getattr(args, "real_safe_preset", False):
        return args
    for name, value in REAL_SAFE_PRESET.items():
        _set_if_missing(args, name, value)
    return args


def normalize_ros_real_args(args, default_theta_initial, default_theta_lower, default_theta_upper):
    args = apply_real_safe_preset(args)
    args.joint_names = parse_joint_names(args.joint_names)
    args.theta_initial_command_vector = parse_vector_arg(
        args.theta_initial_command,
        6,
        "--theta-initial-command",
    )
    args.theta_lower_vector = parse_vector_arg(args.theta_lower, 6, "--theta-lower")
    args.theta_upper_vector = parse_vector_arg(args.theta_upper, 6, "--theta-upper")
    args.tcp_offset_vector = parse_vector_arg(args.tcp_offset, 3, "--tcp-offset")
    args.effective_theta_initial = (
        np.asarray(args.theta_initial_command_vector, dtype=float)
        if args.theta_initial_command_vector is not None
        else np.asarray(default_theta_initial, dtype=float)
    )
    args.effective_theta_lower = (
        np.asarray(args.theta_lower_vector, dtype=float)
        if args.theta_lower_vector is not None
        else np.asarray(default_theta_lower, dtype=float)
    )
    args.effective_theta_upper = (
        np.asarray(args.theta_upper_vector, dtype=float)
        if args.theta_upper_vector is not None
        else np.asarray(default_theta_upper, dtype=float)
    )
    args.effective_tcp_offset = (
        np.asarray(args.tcp_offset_vector, dtype=float)
        if args.tcp_offset_vector is not None
        else np.zeros(3, dtype=float)
    )
    return args


def _call_controller_step(controller, theta_current, desired_pos, desired_vel, use_feedback, tk, step_accepts_time):
    if step_accepts_time:
        return controller.step(theta_current, desired_pos, desired_vel, use_feedback=use_feedback, t_current=tk)
    return controller.step(theta_current, desired_pos, desired_vel, use_feedback=use_feedback)


class StopMotionGuard:
    def __init__(self, interface):
        self.interface = interface
        self.armed = True

    def stop(self):
        if self.armed and self.interface is not None:
            self.interface.stop_motion()
        self.armed = False

    def __del__(self):
        if self.armed and self.interface is not None:
            try:
                self.interface.stop_motion()
            except Exception:
                pass


def run_ros_position_live(
    *,
    node_name,
    method_name,
    controller_builder,
    robot,
    settings,
    output_dir,
    use_feedback,
    args,
    method_display,
    build_trajectory,
    apply_config_overrides,
    create_history,
    record_history,
    save_all_figures,
    save_history_data,
    save_summary,
    step_accepts_time=False,
    experiment_label=None,
):
    rospy = ensure_ros_node(node_name)
    output_dir = Path(output_dir)
    display_name = method_display.get(method_name, method_name)
    mode_label = "with_fb" if use_feedback else "nofb"
    label = str(experiment_label or node_name).replace("run_", "").replace("_ros_real", "")
    stem = label
    title_prefix = f"{label.replace('_', ' ').title()} - {display_name} ROS Real ({mode_label})"
    print(f"  [{display_name}] Running ROS real UR3e ({mode_label})...")

    interface = ROSUR3eInterface(
        joint_state_topic=args.joint_state_topic,
        command_topic=args.command_topic,
        joint_names=args.joint_names,
        command_mode=args.command_mode,
        trajectory_command_topic=args.trajectory_command_topic,
        velocity_command_topic=args.velocity_command_topic,
        trajectory_command_duration=args.trajectory_command_duration,
        tcp_pose_topic=args.tcp_pose_topic,
    )
    stop_guard = StopMotionGuard(interface)
    controller = controller_builder(robot, settings)
    apply_config_overrides(controller, args)
    if hasattr(robot, "tool_offset"):
        robot.tool_offset = np.asarray(args.effective_tcp_offset, dtype=float).copy()

    theta_lower = np.asarray(args.effective_theta_lower, dtype=float)
    theta_upper = np.asarray(args.effective_theta_upper, dtype=float)
    theta_initial_command = np.asarray(args.effective_theta_initial, dtype=float)
    controller.update_joint_limits(theta_lower, theta_upper)

    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_report = interface.build_preflight_report(args, theta_lower, theta_upper)
    preflight_report["real_safe_preset"] = bool(getattr(args, "real_safe_preset", False))
    preflight_report["real_safe_preset_defaults"] = dict(REAL_SAFE_PRESET)
    preflight_report["theta_initial_command_rad"] = theta_initial_command.tolist()
    preflight_report["tcp_offset_m_tool_frame"] = np.asarray(args.effective_tcp_offset, dtype=float).tolist()
    preflight_path = save_preflight_report(output_dir, preflight_report)
    print(f"    Saved ROS preflight report to {preflight_path}")
    print(json.dumps(preflight_report, indent=2, ensure_ascii=False))

    if getattr(args, "check_only", False):
        return None, None
    if getattr(args, "dry_run", False):
        print("    Dry-run requested. No initialization move or trajectory command was published.")
        return None, None
    preflight_verdict = preflight_report.get("preflight_verdict", {})
    if preflight_verdict.get("status") == "fail":
        raise RuntimeError(format_preflight_failure(preflight_verdict))
    if not preflight_report.get("command_subscriber_ready", False):
        raise RuntimeError(
            f"No subscriber connected to {interface.command_topic}. "
            "Start the UR driver/controller first, or pass the correct --command-topic."
        )

    theta_reference = interface.move_to_joint_positions(
        theta_goal=theta_initial_command,
        steps=args.settle_steps,
        rate_hz=args.settle_rate,
        max_step=args.max_command_step,
    )
    controller.reset(theta_reference)
    initial_pos = robot.forward_kinematics(theta_reference)[:3, 3]
    trajectory, trajectory_tag = build_trajectory(
        settings.trajectory_name,
        settings.trajectory_period,
        settings.heart_scale,
        initial_pos,
    )

    history = create_history()
    theta_current = theta_reference.copy()
    total_steps = int(round(float(settings.duration) / float(settings.tau)))
    rate = rospy.Rate(1.0 / float(settings.tau))
    t_start = time.perf_counter()
    last_limited_velocity = None
    last_commanded_velocity = None
    last_cycle_start = None
    cycle_diagnostics = []

    for step in range(total_steps):
        if rospy.is_shutdown():
            raise RuntimeError("ROS shut down during real UR3e experiment.")
        cycle_start = time.perf_counter()
        tk = step * float(settings.tau)
        state_read_start = time.perf_counter()
        theta_current = interface.get_joint_positions(max_age=args.state_timeout)
        state_read_s = time.perf_counter() - state_read_start
        state_age_s = interface.get_latest_state_age()
        desired_pos, desired_vel = trajectory.get_pose(tk)
        controller_start = time.perf_counter()
        result = _call_controller_step(
            controller,
            theta_current,
            desired_pos,
            desired_vel,
            use_feedback,
            tk,
            step_accepts_time,
        )
        controller_step_s = time.perf_counter() - controller_start
        theta_next = np.asarray(result["theta_next"], dtype=float)
        limited_velocity = np.asarray(result.get("theta_dot_next", np.zeros_like(theta_current)), dtype=float)
        if args.max_command_accel is not None and float(args.max_command_accel) > 0.0:
            if last_limited_velocity is None:
                last_limited_velocity = limited_velocity.copy()
            accel_step = float(args.max_command_accel) * float(settings.tau)
            limited_velocity = np.clip(
                limited_velocity,
                last_limited_velocity - accel_step,
                last_limited_velocity + accel_step,
            )
            theta_next = theta_current + float(settings.tau) * limited_velocity
            last_limited_velocity = limited_velocity.copy()
        if args.max_command_step is not None and float(args.max_command_step) > 0.0:
            max_step = float(args.max_command_step)
            theta_next = np.clip(theta_next, theta_current - max_step, theta_current + max_step)
        theta_next = np.clip(theta_next, theta_lower, theta_upper)

        fk_pos = np.asarray(result.get("current_pos", robot.forward_kinematics(theta_current)[:3, 3]), dtype=float)
        measured_tcp_pos = interface.get_latest_tcp_position(max_age=args.tcp_pose_timeout)
        tcp_pose_used = measured_tcp_pos is not None
        tcp_pose_age_s = interface.get_latest_tcp_pose_age()
        actual_pos = fk_pos if not tcp_pose_used else np.asarray(measured_tcp_pos, dtype=float)
        tcp_fk_error_norm_m = (
            float("nan")
            if not tcp_pose_used
            else float(np.linalg.norm(actual_pos - fk_pos))
        )
        feedback_velocity = interface.get_latest_velocity()
        target_delta = theta_next - theta_current
        commanded_velocity = target_delta / float(settings.tau)
        if last_commanded_velocity is None:
            commanded_accel = np.zeros_like(commanded_velocity)
        else:
            commanded_accel = (commanded_velocity - last_commanded_velocity) / float(settings.tau)
        last_commanded_velocity = commanded_velocity.copy()
        record_history(
            history,
            tk,
            theta_current,
            actual_pos,
            fk_pos,
            actual_pos,
            desired_pos,
            theta_reference,
            task_residual=result.get("task_residual"),
            solver_residual=result.get("solver_residual_norm"),
            solver_energy=result.get("solver_energy"),
            boundary_slack=result.get("boundary_slack"),
            joint_velocity=commanded_velocity,
            joint_velocity_raw=result.get("theta_dot_raw"),
            joint_position_target=theta_next,
            joint_velocity_feedback=feedback_velocity,
        )

        publish_start = time.perf_counter()
        if args.command_mode == "velocity_array":
            interface.publish_joint_velocities(commanded_velocity)
        else:
            interface.publish_joint_positions(theta_next, duration_s=settings.tau)
        publish_s = time.perf_counter() - publish_start
        work_end = time.perf_counter()
        rate.sleep()
        sleep_end = time.perf_counter()
        cycle_work_s = work_end - cycle_start
        sleep_s = sleep_end - work_end
        cycle_period_s = float("nan") if last_cycle_start is None else cycle_start - last_cycle_start
        last_cycle_start = cycle_start
        feedback_velocity_norm = (
            float("nan")
            if feedback_velocity is None
            else float(np.linalg.norm(np.asarray(feedback_velocity, dtype=float)))
        )
        cycle_diagnostics.append(
            {
                "step": int(step),
                "time_s": float(tk),
                "cycle_period_s": float(cycle_period_s),
                "cycle_work_s": float(cycle_work_s),
                "state_read_s": float(state_read_s),
                "state_age_s": float(state_age_s),
                "controller_step_s": float(controller_step_s),
                "publish_s": float(publish_s),
                "sleep_s": float(sleep_s),
                "command_step_norm_rad": float(np.linalg.norm(target_delta)),
                "command_step_max_abs_rad": float(np.max(np.abs(target_delta))),
                "command_velocity_norm_rad_s": float(np.linalg.norm(commanded_velocity)),
                "command_accel_norm_rad_s2": float(np.linalg.norm(commanded_accel)),
                "feedback_velocity_norm_rad_s": feedback_velocity_norm,
                "tcp_pose_used": bool(tcp_pose_used),
                "tcp_pose_age_s": float(tcp_pose_age_s),
                "tcp_fk_error_norm_m": tcp_fk_error_norm_m,
                "deadline_miss": bool(cycle_work_s > float(settings.tau)),
                "period_overrun": bool(np.isfinite(cycle_period_s) and cycle_period_s > 1.25 * float(settings.tau)),
            }
        )

    stop_guard.stop()
    runtime_s = time.perf_counter() - t_start
    theta_current = interface.get_joint_positions(max_age=max(float(args.state_timeout), 1.0))
    terminal_delta = theta_current - theta_reference
    terminal_feedback_report = {
        "theta_feedback_rad": theta_current.tolist(),
        "joint_drift_rad": terminal_delta.tolist(),
        "joint_drift_norm_rad": float(np.linalg.norm(terminal_delta)),
    }
    print(f"    Runtime: {runtime_s:.1f}s, steps: {total_steps}")

    if not getattr(args, "skip_plots", False):
        save_all_figures(history, output_dir, stem, title_prefix)
    else:
        save_history_data(output_dir, stem, history)

    summary = save_summary(output_dir, stem, theta_reference, theta_current, history)
    tcp_pose_enabled = bool(args.tcp_pose_topic)
    summary["live_position_error_source"] = (
        f"Measured TCP pose from ROS PoseStamped topic {args.tcp_pose_topic}"
        if tcp_pose_enabled
        else "Local UR3e forward kinematics from ROS /joint_states feedback"
    )
    summary["live_fk_position_error_source"] = "Local UR3e forward kinematics from ROS /joint_states feedback"
    summary["live_visual_tip_error_source"] = (
        f"Measured TCP pose from ROS PoseStamped topic {args.tcp_pose_topic}"
        if tcp_pose_enabled
        else "Not used in ROS real mode; duplicated local FK position for plotting compatibility"
    )
    summary["joint_drift_source"] = "ROS joint-position feedback relative to measured post-initialization joint state"
    summary["theta_final_semantics"] = "terminal ROS joint feedback after the last published joint-position command"
    summary["runtime_s"] = runtime_s
    summary["avg_step_time_s"] = runtime_s / max(total_steps, 1)
    summary["ros_real_cycle_diagnostics"] = save_ros_cycle_diagnostics(output_dir, cycle_diagnostics, settings.tau)
    summary["physical_reset_before_run"] = preflight_report
    summary["physical_reset_after_run"] = {"reset_mode": "not_applicable_ros_real"}
    summary["terminal_feedback_after_last_command"] = terminal_feedback_report
    summary["trajectory_name"] = trajectory_tag
    summary["solver_type"] = getattr(controller.cfg, "solver_type", "")
    summary["task_gain"] = getattr(controller.cfg, "task_gain", "")
    summary["drift_gain"] = getattr(controller.cfg, "drift_gain", "")
    summary["drift_free_term"] = getattr(controller.cfg, "drift_free_term", "active")
    summary["drift_feedback_mode"] = getattr(controller.cfg, "drift_feedback_mode", "")
    summary["solver_gamma"] = getattr(controller.cfg, "solver_gamma", "")
    summary["activation_power"] = getattr(controller.cfg, "activation_power", "")
    summary["activation_exp_clip"] = getattr(controller.cfg, "activation_exp_clip", "")
    summary["dlccznn_inner_steps"] = getattr(controller.cfg, "dlccznn_inner_steps", "")
    summary["pdnn_gain"] = getattr(controller.cfg, "pdnn_gain", "")
    summary["pdnn_inner_steps"] = getattr(controller.cfg, "pdnn_inner_steps", "")
    summary["ros_real_interface"] = {
        "joint_state_topic": args.joint_state_topic,
        "tcp_pose_topic": args.tcp_pose_topic,
        "tcp_pose_enabled": tcp_pose_enabled,
        "tcp_pose_timeout_s": float(args.tcp_pose_timeout),
        "command_topic": interface.command_topic,
        "position_command_topic": args.command_topic,
        "trajectory_command_topic": args.trajectory_command_topic,
        "velocity_command_topic": args.velocity_command_topic,
        "command_mode": args.command_mode,
        "trajectory_command_duration_s": interface.get_trajectory_command_duration(settings.tau),
        "joint_names": list(args.joint_names),
        "tcp_offset_m_tool_frame": np.asarray(args.effective_tcp_offset, dtype=float).tolist(),
        "real_safe_preset": bool(getattr(args, "real_safe_preset", False)),
        "real_safe_preset_defaults": dict(REAL_SAFE_PRESET),
        "max_command_step_rad": None if args.max_command_step is None else float(args.max_command_step),
        "max_command_accel_rad_s2": None if args.max_command_accel is None else float(args.max_command_accel),
    }
    if hasattr(controller.cfg, "internal_disturbance"):
        summary["internal_disturbance"] = getattr(controller.cfg, "internal_disturbance", "none")
        summary["disturbance_scale"] = getattr(controller.cfg, "disturbance_scale", 1.0)
    (output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return history, summary


class ROSUR3eInterface:
    def __init__(
        self,
        joint_state_topic,
        command_topic,
        joint_names,
        command_mode="position_array",
        trajectory_command_topic=None,
        velocity_command_topic=None,
        trajectory_command_duration=None,
        tcp_pose_topic="",
    ):
        self.rospy, JointState, Float64MultiArray, JointTrajectory, JointTrajectoryPoint, PoseStamped = import_ros_deps()
        self.Float64MultiArray = Float64MultiArray
        self.JointTrajectory = JointTrajectory
        self.JointTrajectoryPoint = JointTrajectoryPoint
        self.PoseStamped = PoseStamped
        self.joint_state_topic = str(joint_state_topic)
        self.tcp_pose_topic = str(tcp_pose_topic or "")
        self.command_mode = str(command_mode)
        if self.command_mode not in {"position_array", "joint_trajectory", "velocity_array"}:
            raise ValueError(f"Unsupported command mode: {self.command_mode}")
        self.position_command_topic = str(command_topic)
        self.trajectory_command_topic = str(trajectory_command_topic or command_topic)
        self.velocity_command_topic = str(velocity_command_topic or command_topic)
        if self.command_mode == "position_array":
            self.command_topic = self.position_command_topic
        elif self.command_mode == "joint_trajectory":
            self.command_topic = self.trajectory_command_topic
        else:
            self.command_topic = self.velocity_command_topic
        self.trajectory_command_duration = (
            None if trajectory_command_duration is None else float(trajectory_command_duration)
        )
        self.joint_names = tuple(joint_names)
        self._lock = threading.Lock()
        self._latest_positions = None
        self._latest_velocity = None
        self._latest_names = None
        self._latest_ros_stamp = None
        self._latest_wall_time = None
        self._latest_tcp_position = None
        self._latest_tcp_ros_stamp = None
        self._latest_tcp_wall_time = None
        self._stop_registered = False
        command_msg_type = JointTrajectory if self.command_mode == "joint_trajectory" else Float64MultiArray
        self._publisher = self.rospy.Publisher(self.command_topic, command_msg_type, queue_size=10)
        if self.command_mode == "velocity_array":
            self.rospy.on_shutdown(self.stop_motion)
            atexit.register(self.stop_motion)
            self._stop_registered = True
        self._subscriber = self.rospy.Subscriber(
            self.joint_state_topic,
            JointState,
            self._joint_state_callback,
            queue_size=1,
        )
        self._tcp_pose_subscriber = None
        if self.tcp_pose_topic:
            self._tcp_pose_subscriber = self.rospy.Subscriber(
                self.tcp_pose_topic,
                PoseStamped,
                self._tcp_pose_callback,
                queue_size=1,
            )

    def _joint_state_callback(self, msg):
        if not msg.name or not msg.position:
            return
        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        missing = [name for name in self.joint_names if name not in name_to_index]
        if missing:
            return
        ordered_positions = np.asarray([msg.position[name_to_index[name]] for name in self.joint_names], dtype=float)
        ordered_velocity = None
        if msg.velocity and len(msg.velocity) >= len(msg.name):
            ordered_velocity = np.asarray([msg.velocity[name_to_index[name]] for name in self.joint_names], dtype=float)
        with self._lock:
            self._latest_positions = ordered_positions
            self._latest_velocity = ordered_velocity
            self._latest_names = tuple(msg.name)
            self._latest_ros_stamp = msg.header.stamp if msg.header.stamp != self.rospy.Time() else None
            self._latest_wall_time = time.monotonic()

    def _tcp_pose_callback(self, msg):
        position = np.asarray(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=float,
        )
        with self._lock:
            self._latest_tcp_position = position
            self._latest_tcp_ros_stamp = msg.header.stamp if msg.header.stamp != self.rospy.Time() else None
            self._latest_tcp_wall_time = time.monotonic()

    def wait_for_joint_state(self, timeout):
        deadline = time.monotonic() + float(timeout)
        rate = self.rospy.Rate(100)
        while not self.rospy.is_shutdown():
            with self._lock:
                if self._latest_positions is not None:
                    return self._latest_positions.copy()
            if time.monotonic() >= deadline:
                break
            rate.sleep()
        if self.rospy.is_shutdown():
            raise RuntimeError("ROS shut down before joint states became available.")
        raise TimeoutError(f"Timed out waiting for joint states on {self.joint_state_topic}.")

    def get_joint_positions(self, max_age):
        with self._lock:
            positions = None if self._latest_positions is None else self._latest_positions.copy()
            wall_time = self._latest_wall_time
        if positions is None:
            raise RuntimeError(f"No joint state has been received on {self.joint_state_topic}.")
        if wall_time is None or (time.monotonic() - wall_time) > float(max_age):
            raise RuntimeError(f"Latest joint state is stale on {self.joint_state_topic}.")
        return positions

    def get_latest_velocity(self):
        with self._lock:
            return None if self._latest_velocity is None else self._latest_velocity.copy()

    def get_latest_state_age(self):
        with self._lock:
            wall_time = self._latest_wall_time
        return float("nan") if wall_time is None else float(time.monotonic() - wall_time)

    def get_latest_tcp_position(self, max_age=None):
        with self._lock:
            position = None if self._latest_tcp_position is None else self._latest_tcp_position.copy()
            wall_time = self._latest_tcp_wall_time
        if position is None:
            return None
        if max_age is not None and (wall_time is None or (time.monotonic() - wall_time) > float(max_age)):
            return None
        return position

    def get_latest_tcp_pose_age(self):
        with self._lock:
            wall_time = self._latest_tcp_wall_time
        return float("nan") if wall_time is None else float(time.monotonic() - wall_time)

    def get_trajectory_command_duration(self, tau):
        if self.trajectory_command_duration is not None and self.trajectory_command_duration > 0.0:
            return float(self.trajectory_command_duration)
        return max(2.0 * float(tau), 0.02)

    def publish_joint_positions(self, joint_targets, duration_s=None):
        targets = np.asarray(joint_targets, dtype=float).tolist()
        if self.command_mode == "velocity_array":
            raise RuntimeError("publish_joint_positions cannot be used when command_mode=velocity_array.")
        if self.command_mode == "position_array":
            msg = self.Float64MultiArray()
            msg.data = targets
            self._publisher.publish(msg)
            return

        duration = self.get_trajectory_command_duration(0.0 if duration_s is None else duration_s)
        msg = self.JointTrajectory()
        msg.header.stamp = self.rospy.Time.now()
        msg.joint_names = list(self.joint_names)
        point = self.JointTrajectoryPoint()
        point.positions = targets
        point.time_from_start = self.rospy.Duration.from_sec(duration)
        msg.points = [point]
        self._publisher.publish(msg)

    def publish_joint_velocities(self, joint_velocities):
        if self.command_mode != "velocity_array":
            raise RuntimeError("publish_joint_velocities requires command_mode=velocity_array.")
        msg = self.Float64MultiArray()
        msg.data = np.asarray(joint_velocities, dtype=float).tolist()
        self._publisher.publish(msg)

    def stop_motion(self):
        if self.command_mode == "velocity_array":
            try:
                self.publish_joint_velocities(np.zeros(len(self.joint_names), dtype=float))
            except Exception:
                pass

    def wait_for_command_subscriber(self, timeout):
        deadline = time.monotonic() + float(timeout)
        rate = self.rospy.Rate(50)
        while not self.rospy.is_shutdown():
            if self._publisher.get_num_connections() > 0:
                return True
            if time.monotonic() >= deadline:
                break
            rate.sleep()
        return self._publisher.get_num_connections() > 0

    def move_to_joint_positions(self, theta_goal, steps, rate_hz, max_step):
        if self.command_mode == "velocity_array":
            return self._move_to_joint_positions_velocity(theta_goal, steps, rate_hz, max_step)

        theta_goal = np.asarray(theta_goal, dtype=float)
        theta_start = self.get_joint_positions(max_age=1.0)
        theta_prev_cmd = theta_start.copy()
        total_steps = max(int(steps), 1)
        rate = self.rospy.Rate(float(rate_hz))
        for step in range(total_steps):
            if self.rospy.is_shutdown():
                raise RuntimeError("ROS shut down during initial joint move.")
            alpha = float(step + 1) / float(total_steps)
            theta_cmd = theta_start + alpha * (theta_goal - theta_start)
            if max_step is not None and float(max_step) > 0.0:
                theta_cmd = np.clip(theta_cmd, theta_prev_cmd - float(max_step), theta_prev_cmd + float(max_step))
            self.publish_joint_positions(theta_cmd, duration_s=1.0 / float(rate_hz))
            theta_prev_cmd = theta_cmd
            rate.sleep()
        self.publish_joint_positions(theta_goal, duration_s=1.0 / float(rate_hz))
        self.rospy.sleep(max(1.0 / float(rate_hz), 0.02))
        return self.get_joint_positions(max_age=1.0)

    def _move_to_joint_positions_velocity(self, theta_goal, steps, rate_hz, max_step):
        theta_goal = np.asarray(theta_goal, dtype=float)
        rate_hz = float(rate_hz)
        dt = 1.0 / rate_hz
        theta_start = self.get_joint_positions(max_age=1.0)
        max_step = 0.005 if max_step is None or float(max_step) <= 0.0 else float(max_step)
        max_velocity = max_step / dt
        total_steps = max(int(steps), int(np.ceil(np.max(np.abs(theta_goal - theta_start)) / max_step)) + 10, 1)
        rate = self.rospy.Rate(rate_hz)
        last_theta = theta_start.copy()
        stale_count = 0
        for _ in range(total_steps):
            if self.rospy.is_shutdown():
                raise RuntimeError("ROS shut down during velocity-mode initial joint move.")
            theta_current = self.get_joint_positions(max_age=1.0)
            if np.linalg.norm(theta_current - last_theta, ord=np.inf) < 1e-5:
                stale_count += 1
            else:
                stale_count = 0
            last_theta = theta_current.copy()
            error = theta_goal - theta_current
            if np.linalg.norm(error, ord=np.inf) <= max(2e-3, 0.25 * max_step):
                break
            velocity_cmd = np.clip(error / dt, -max_velocity, max_velocity)
            self.publish_joint_velocities(velocity_cmd)
            rate.sleep()
            if stale_count > max(int(rate_hz), 10):
                raise RuntimeError("Joint feedback did not change during velocity-mode initialization.")
        self.publish_joint_velocities(np.zeros_like(theta_goal))
        self.rospy.sleep(max(dt, 0.02))
        return self.get_joint_positions(max_age=1.0)

    def build_preflight_report(self, args, theta_lower, theta_upper):
        latest = self.wait_for_joint_state(timeout=args.startup_timeout)
        subscriber_ready = self.wait_for_command_subscriber(timeout=args.command_wait_timeout)
        velocity = self.get_latest_velocity()
        tcp_position = self.get_latest_tcp_position(max_age=None)
        tcp_pose_age = self.get_latest_tcp_pose_age()
        published_topics = inspect_published_topics(self.rospy)
        master_state = inspect_ros_master_state(self.rospy)
        controller_report = inspect_ros_controllers(self.rospy)
        command_mode_recommendations = build_command_mode_recommendations(
            published_topics,
            controller_report,
            position_command_topic=self.position_command_topic,
            trajectory_command_topic=self.trajectory_command_topic,
            velocity_command_topic=self.velocity_command_topic,
            master_state=master_state,
        )
        report = {
            "interface": "ros1_joint_position_command",
            "joint_state_topic": self.joint_state_topic,
            "tcp_pose_topic": self.tcp_pose_topic,
            "tcp_pose_enabled": bool(self.tcp_pose_topic),
            "latest_tcp_position_m": None if tcp_position is None else tcp_position.tolist(),
            "latest_tcp_pose_age_s": None if tcp_position is None else float(tcp_pose_age),
            "command_topic": self.command_topic,
            "position_command_topic": self.position_command_topic,
            "trajectory_command_topic": self.trajectory_command_topic,
            "velocity_command_topic": self.velocity_command_topic,
            "command_mode": self.command_mode,
            "command_message_type": (
                "std_msgs/Float64MultiArray"
                if self.command_mode == "position_array"
                else "trajectory_msgs/JointTrajectory"
                if self.command_mode == "joint_trajectory"
                else "std_msgs/Float64MultiArray"
            ),
            "trajectory_command_duration_s": self.get_trajectory_command_duration(getattr(args, "tau", 0.0)),
            "joint_names": list(self.joint_names),
            "latest_joint_position_rad": latest.tolist(),
            "latest_joint_velocity_rad_s": None if velocity is None else velocity.tolist(),
            "latest_state_age_s": 0.0 if self._latest_wall_time is None else float(time.monotonic() - self._latest_wall_time),
            "command_subscriber_ready": bool(subscriber_ready),
            "published_topics": published_topics,
            "ros_master_state": master_state,
            "controller_manager": controller_report,
            "command_mode_recommendations": command_mode_recommendations,
            "theta_lower_rad": np.asarray(theta_lower, dtype=float).tolist(),
            "theta_upper_rad": np.asarray(theta_upper, dtype=float).tolist(),
            "real_safe_preset": bool(getattr(args, "real_safe_preset", False)),
            "real_safe_preset_defaults": dict(REAL_SAFE_PRESET),
            "max_command_step_rad": None if args.max_command_step is None else float(args.max_command_step),
            "max_command_accel_rad_s2": None if args.max_command_accel is None else float(args.max_command_accel),
            "settle_steps": int(args.settle_steps),
            "settle_rate_hz": float(args.settle_rate),
            "state_timeout_s": float(args.state_timeout),
        }
        report["preflight_verdict"] = evaluate_preflight_report(report, tau=getattr(args, "tau", None))
        return report
