from __future__ import annotations

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


def import_ros_deps():
    try:
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
    except ImportError as exc:
        raise RuntimeError(
            "ROS live mode requires rospy, sensor_msgs, and std_msgs. "
            "Run this script from a sourced ROS1 workspace, for example: "
            "source /opt/ros/noetic/setup.bash && source ~/文档/catkin_ws/devel/setup.bash"
        ) from exc
    return rospy, JointState, Float64MultiArray


def ensure_ros_node(node_name: str):
    rospy, _, _ = import_ros_deps()
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


def add_ros_real_arguments(parser):
    parser.add_argument("--joint-state-topic", type=str, default="/joint_states")
    parser.add_argument("--command-topic", type=str, default="/pos_joint_group_controller/command")
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
    return parser


def normalize_ros_real_args(args, default_theta_initial, default_theta_lower, default_theta_upper):
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
    )
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
    if not preflight_report.get("command_subscriber_ready", False):
        raise RuntimeError(
            f"No subscriber connected to {args.command_topic}. "
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

    for step in range(total_steps):
        if rospy.is_shutdown():
            raise RuntimeError("ROS shut down during real UR3e experiment.")
        tk = step * float(settings.tau)
        theta_current = interface.get_joint_positions(max_age=args.state_timeout)
        desired_pos, desired_vel = trajectory.get_pose(tk)
        result = _call_controller_step(
            controller,
            theta_current,
            desired_pos,
            desired_vel,
            use_feedback,
            tk,
            step_accepts_time,
        )
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
        feedback_velocity = interface.get_latest_velocity()
        record_history(
            history,
            tk,
            theta_current,
            fk_pos,
            fk_pos,
            fk_pos,
            desired_pos,
            theta_reference,
            task_residual=result.get("task_residual"),
            solver_residual=result.get("solver_residual_norm"),
            solver_energy=result.get("solver_energy"),
            boundary_slack=result.get("boundary_slack"),
            joint_velocity=limited_velocity,
            joint_velocity_raw=result.get("theta_dot_raw"),
            joint_position_target=theta_next,
            joint_velocity_feedback=feedback_velocity,
        )

        interface.publish_joint_positions(theta_next)
        rate.sleep()

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
    summary["live_position_error_source"] = "Local UR3e forward kinematics from ROS /joint_states feedback"
    summary["live_fk_position_error_source"] = "Local UR3e forward kinematics from ROS /joint_states feedback"
    summary["live_visual_tip_error_source"] = "Not used in ROS real mode; duplicated local FK position for plotting compatibility"
    summary["joint_drift_source"] = "ROS joint-position feedback relative to measured post-initialization joint state"
    summary["theta_final_semantics"] = "terminal ROS joint feedback after the last published joint-position command"
    summary["runtime_s"] = runtime_s
    summary["avg_step_time_s"] = runtime_s / max(total_steps, 1)
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
        "command_topic": args.command_topic,
        "joint_names": list(args.joint_names),
        "tcp_offset_m_tool_frame": np.asarray(args.effective_tcp_offset, dtype=float).tolist(),
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
    def __init__(self, joint_state_topic, command_topic, joint_names):
        self.rospy, JointState, Float64MultiArray = import_ros_deps()
        self.Float64MultiArray = Float64MultiArray
        self.joint_state_topic = str(joint_state_topic)
        self.command_topic = str(command_topic)
        self.joint_names = tuple(joint_names)
        self._lock = threading.Lock()
        self._latest_positions = None
        self._latest_velocity = None
        self._latest_names = None
        self._latest_ros_stamp = None
        self._latest_wall_time = None
        self._publisher = self.rospy.Publisher(self.command_topic, Float64MultiArray, queue_size=10)
        self._subscriber = self.rospy.Subscriber(
            self.joint_state_topic,
            JointState,
            self._joint_state_callback,
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

    def publish_joint_positions(self, joint_targets):
        msg = self.Float64MultiArray()
        msg.data = np.asarray(joint_targets, dtype=float).tolist()
        self._publisher.publish(msg)

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
            self.publish_joint_positions(theta_cmd)
            theta_prev_cmd = theta_cmd
            rate.sleep()
        self.publish_joint_positions(theta_goal)
        self.rospy.sleep(max(1.0 / float(rate_hz), 0.02))
        return self.get_joint_positions(max_age=1.0)

    def build_preflight_report(self, args, theta_lower, theta_upper):
        latest = self.wait_for_joint_state(timeout=args.startup_timeout)
        subscriber_ready = self.wait_for_command_subscriber(timeout=args.command_wait_timeout)
        velocity = self.get_latest_velocity()
        return {
            "interface": "ros1_joint_position_command",
            "joint_state_topic": self.joint_state_topic,
            "command_topic": self.command_topic,
            "command_message_type": "std_msgs/Float64MultiArray",
            "joint_names": list(self.joint_names),
            "latest_joint_position_rad": latest.tolist(),
            "latest_joint_velocity_rad_s": None if velocity is None else velocity.tolist(),
            "latest_state_age_s": 0.0 if self._latest_wall_time is None else float(time.monotonic() - self._latest_wall_time),
            "command_subscriber_ready": bool(subscriber_ready),
            "theta_lower_rad": np.asarray(theta_lower, dtype=float).tolist(),
            "theta_upper_rad": np.asarray(theta_upper, dtype=float).tolist(),
            "max_command_step_rad": None if args.max_command_step is None else float(args.max_command_step),
            "max_command_accel_rad_s2": None if args.max_command_accel is None else float(args.max_command_accel),
            "settle_steps": int(args.settle_steps),
            "settle_rate_hz": float(args.settle_rate),
            "state_timeout_s": float(args.state_timeout),
        }
