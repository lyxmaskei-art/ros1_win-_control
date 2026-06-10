import os
import time

import numpy as np

try:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
except ImportError as exc:
    rospy = None
    JointState = None
    Float64MultiArray = None
    JointTrajectory = None
    JointTrajectoryPoint = None
    ROS_IMPORT_ERROR = exc
else:
    ROS_IMPORT_ERROR = None


def make_rtde_adapter_class(default_control_mode):
    """Legacy factory name kept so the experiment scripts need no broad edits.

    Despite the filename, this adapter does not use ur_rtde. It talks to the
    official UR ROS driver / ros_control command topics and is locked by
    default for hardware safety.
    """

    default_control_mode = str(default_control_mode).strip().lower()
    if default_control_mode not in {"position", "speed"}:
        raise ValueError("default_control_mode must be 'position' or 'speed'.")

    class ROSURDriverAdapter:
        simx_return_ok = 0
        simx_return_initialize_error_flag = 1
        simx_opmode_blocking = 0
        simx_opmode_streaming = 1
        simx_opmode_buffer = 2
        simx_opmode_oneshot = 3
        sim_jointfloatparam_upper_limit = "jointfloatparam_upper_limit"

        def __init__(self):
            self.control_mode = default_control_mode
            self.command_backend = (
                "ros_position_trajectory" if self.control_mode == "position" else "ros_velocity"
            )
            self.stepping_dt = float(os.environ.get("UR3E_TAU", "0.005"))
            self.joint_state_topic = os.environ.get("UR3E_JOINT_STATE_TOPIC", "/joint_states")
            self.velocity_command_topic = os.environ.get(
                "UR3E_VELOCITY_COMMAND_TOPIC",
                "/joint_group_vel_controller/command",
            )
            self.position_command_topic = os.environ.get(
                "UR3E_POSITION_TRAJECTORY_TOPIC",
                "/scaled_pos_joint_traj_controller/command",
            )
            names = os.environ.get(
                "UR3E_JOINT_NAMES",
                "shoulder_pan_joint,shoulder_lift_joint,elbow_joint,wrist_1_joint,wrist_2_joint,wrist_3_joint",
            )
            self.joint_names = tuple(name.strip() for name in names.split(",") if name.strip())
            self.enable_commands = os.environ.get("UR3E_ENABLE_ROS_DRIVER_EXPERIMENT_I_ACCEPT_RISK", "0") == "1"
            self.max_abs_qdot = float(os.environ.get("UR3E_SAFE_MAX_ABS_QDOT", "0.03"))
            self.max_qdot_delta = float(os.environ.get("UR3E_SAFE_MAX_QDOT_DELTA", "0.01"))
            self._max_position_step_override = os.environ.get("UR3E_SAFE_MAX_POSITION_STEP")
            self.max_position_step = self._compute_max_position_step()
            self.feedback_timeout = float(os.environ.get("UR3E_FEEDBACK_TIMEOUT", "0.2"))
            self.require_command_subscriber = os.environ.get("UR3E_REQUIRE_COMMAND_SUBSCRIBER", "1") != "0"
            self._latest_q = None
            self._latest_stamp = None
            self._latest_wall_time = None
            self._feedback_seq = 0
            self._velocity_pub = None
            self._trajectory_pub = None
            self._joint_sub = None
            self._paused = False
            self._pending_positions = None
            self._pending_velocities = None
            self._position_target = None
            self._last_velocity = np.zeros(6, dtype=float)
            self._last_target_or_velocity = None
            self._last_error = ""
            self._command_count = 0
            self._blocked_command_count = 0
            self._abs_clip_count = 0
            self._delta_clip_count = 0
            self._raw_velocity_max_abs = 0.0
            self._target_step_samples = []

        def configure_real_transport(
            self,
            backend=None,
            robot_ip=None,
            script_port=None,
            servoj_t=None,
            servoj_lookahead_time=None,
            servoj_gain=None,
            speedj_accel=None,
            speedj_t=None,
            wait_fresh_feedback=None,
            feedback_wait_timeout=None,
            use_actual_dt=None,
        ):
            if speedj_t is not None and self.control_mode == "speed":
                self.stepping_dt = float(speedj_t)
            if servoj_t is not None and self.control_mode == "position":
                self.stepping_dt = float(servoj_t)
            self.max_position_step = self._compute_max_position_step()
            requested = "" if backend is None else str(backend).strip().lower()
            allowed_position = {"", "position", "topic_position", "servoj", "rtde_servoj", "ros_position", "ros_position_trajectory"}
            allowed_speed = {"", "speed", "speedj", "rtde_speedj", "ros_velocity"}
            if self.control_mode == "position":
                if requested not in allowed_position:
                    raise RuntimeError(f"This folder is ROS position-trajectory control only; got backend={requested!r}.")
                self.command_backend = "ros_position_trajectory"
            else:
                if requested not in allowed_speed:
                    raise RuntimeError(f"This folder is ROS velocity control only; got backend={requested!r}.")
                self.command_backend = "ros_velocity"

        def _compute_max_position_step(self):
            if self._max_position_step_override is not None:
                return float(self._max_position_step_override)
            return float(self.max_abs_qdot) * float(self.stepping_dt)

        def _require_ros(self):
            if ROS_IMPORT_ERROR is not None:
                raise RuntimeError("ROS Python packages are required. Source the UR ROS workspace first.") from ROS_IMPORT_ERROR
            if len(self.joint_names) != 6:
                raise RuntimeError("UR3E_JOINT_NAMES must contain exactly 6 joint names.")

        def _init_ros(self):
            self._require_ros()
            if not rospy.core.is_initialized():
                rospy.init_node("znn_ur3e_ros_driver_adapter", anonymous=True, disable_signals=True)
            if self._joint_sub is None:
                self._joint_sub = rospy.Subscriber(
                    self.joint_state_topic,
                    JointState,
                    self._joint_state_cb,
                    queue_size=1,
                    tcp_nodelay=True,
                )
            if self.control_mode == "speed" and self._velocity_pub is None:
                self._velocity_pub = rospy.Publisher(
                    self.velocity_command_topic,
                    Float64MultiArray,
                    queue_size=1,
                    tcp_nodelay=True,
                )
            if self.control_mode == "position" and self._trajectory_pub is None:
                self._trajectory_pub = rospy.Publisher(
                    self.position_command_topic,
                    JointTrajectory,
                    queue_size=1,
                    tcp_nodelay=True,
                )
            self.wait_for_joint_state(float(os.environ.get("UR3E_JOINT_STATE_TIMEOUT", "10.0")))
            if self.require_command_subscriber:
                self._wait_for_command_subscriber(float(os.environ.get("UR3E_COMMAND_SUBSCRIBER_TIMEOUT", "10.0")))

        def _joint_state_cb(self, msg):
            if not msg.name or not msg.position:
                return
            index = {name: i for i, name in enumerate(msg.name)}
            if any(name not in index for name in self.joint_names):
                return
            q = np.asarray([msg.position[index[name]] for name in self.joint_names], dtype=float)
            stamp = None
            try:
                stamp = msg.header.stamp.to_sec()
            except Exception:
                stamp = None
            self._latest_q = q
            self._latest_stamp = stamp
            self._latest_wall_time = time.monotonic()
            self._feedback_seq += 1

        def _wait_for_command_subscriber(self, timeout):
            pub = self._velocity_pub if self.control_mode == "speed" else self._trajectory_pub
            deadline = time.monotonic() + float(timeout)
            rate = rospy.Rate(20)
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                if pub is not None and pub.get_num_connections() > 0:
                    return True
                rate.sleep()
            raise TimeoutError("No ROS controller subscriber is connected to the command topic.")

        def wait_for_joint_state(self, timeout):
            deadline = time.monotonic() + float(timeout)
            rate = rospy.Rate(100)
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                if self._latest_q is not None:
                    return self._latest_q.copy()
                rate.sleep()
            raise TimeoutError(f"Timed out waiting for joint states on {self.joint_state_topic}.")

        def _actual_q(self):
            if self._latest_q is None:
                return self.wait_for_joint_state(self.feedback_timeout)
            if self._latest_wall_time is None or time.monotonic() - self._latest_wall_time > self.feedback_timeout:
                raise RuntimeError("UR3e joint feedback is stale; refusing to command.")
            return self._latest_q.copy()

        def _require_enabled_for_motion(self):
            if not self.enable_commands:
                self._blocked_command_count += 1
                raise RuntimeError(
                    "ROS real-arm command output is disabled by default. Set "
                    "UR3E_ENABLE_ROS_DRIVER_EXPERIMENT_I_ACCEPT_RISK=1 only after "
                    "controller/topic verification, reduced robot speed, and an emergency stop test."
                )

        def _safe_velocity(self, velocity):
            v = np.asarray(velocity, dtype=float)
            if v.shape != (6,) or not np.all(np.isfinite(v)):
                raise RuntimeError("Invalid velocity command.")
            self._raw_velocity_max_abs = max(self._raw_velocity_max_abs, float(np.max(np.abs(v))))
            if np.any(np.abs(v) > self.max_abs_qdot):
                self._abs_clip_count += 1
            v = np.clip(v, -self.max_abs_qdot, self.max_abs_qdot)
            raw_dv = v - self._last_velocity
            if np.any(np.abs(raw_dv) > self.max_qdot_delta):
                self._delta_clip_count += 1
            dv = np.clip(raw_dv, -self.max_qdot_delta, self.max_qdot_delta)
            v = self._last_velocity + dv
            self._last_velocity = v.copy()
            return v

        def _account_command(self, values):
            values = np.asarray(values, dtype=float)
            if self._last_target_or_velocity is not None and values.shape == self._last_target_or_velocity.shape:
                self._target_step_samples.append(float(np.max(np.abs(values - self._last_target_or_velocity))))
                self._target_step_samples = self._target_step_samples[-10000:]
            self._last_target_or_velocity = values.copy()
            self._command_count += 1

        def _publish_zero(self):
            if self.control_mode == "speed" and self._velocity_pub is not None:
                msg = Float64MultiArray()
                msg.data = [0.0] * 6
                self._velocity_pub.publish(msg)

        def _send_velocity(self, velocity):
            self._require_enabled_for_motion()
            velocity = self._safe_velocity(velocity)
            if self._velocity_pub is None:
                raise RuntimeError("Velocity publisher is not initialized.")
            msg = Float64MultiArray()
            msg.data = velocity.tolist()
            self._velocity_pub.publish(msg)
            self._account_command(velocity)

        def _send_position_step(self, velocity):
            self._require_enabled_for_motion()
            velocity = self._safe_velocity(velocity)
            q = self._actual_q() if self._position_target is None else self._position_target.copy()
            step = np.clip(velocity * float(self.stepping_dt), -self.max_position_step, self.max_position_step)
            q_next = q + step
            self._position_target = q_next.copy()
            msg = JointTrajectory()
            msg.joint_names = list(self.joint_names)
            point = JointTrajectoryPoint()
            point.positions = q_next.tolist()
            point.velocities = velocity.tolist()
            point.time_from_start = rospy.Duration.from_sec(max(float(self.stepping_dt) * 2.0, 0.02))
            msg.points = [point]
            self._trajectory_pub.publish(msg)
            self._account_command(q_next)

        def _flush_pending_commands(self):
            velocity = np.zeros(6, dtype=float)
            has_velocity = False
            if self._pending_velocities is not None:
                mask = np.isfinite(self._pending_velocities)
                velocity[mask] = self._pending_velocities[mask]
                has_velocity = bool(np.any(mask))
            self._pending_positions = None
            self._pending_velocities = None
            if not has_velocity:
                return
            if self.control_mode == "speed":
                self._send_velocity(velocity)
            else:
                self._send_position_step(velocity)

        def simxStart(self, host, port, wait_until_connected, do_not_reconnect, timeout_ms, comm_thread_cycle_ms):
            self._init_ros()
            self.client = 0
            return 0

        def simxFinish(self, client_id):
            try:
                self._publish_zero()
            except Exception:
                pass
            return self.simx_return_ok

        def simxSynchronous(self, client_id, enable):
            return self.simx_return_ok

        def simxSynchronousTrigger(self, client_id):
            if rospy is not None:
                rospy.sleep(max(float(self.stepping_dt), 0.0))
            else:
                time.sleep(max(float(self.stepping_dt), 0.0))
            return self.simx_return_ok

        def simxGetPingTime(self, client_id):
            return self.simx_return_ok

        def simxPauseCommunication(self, client_id, pause):
            if pause:
                self._paused = True
                self._pending_positions = np.full(6, np.nan, dtype=float)
                self._pending_velocities = np.full(6, np.nan, dtype=float)
            else:
                self._flush_pending_commands()
                self._paused = False
            return self.simx_return_ok

        def simxStartSimulation(self, client_id, opmode):
            self._position_target = self._actual_q()
            return self.simx_return_ok

        def simxStopSimulation(self, client_id, opmode):
            self._publish_zero()
            return self.simx_return_ok

        def simxGetObjectHandle(self, client_id, object_name, opmode):
            name = str(object_name).strip("/")
            if name.startswith("UR3e_joint"):
                try:
                    return self.simx_return_ok, int(name.replace("UR3e_joint", ""))
                except ValueError:
                    return self.simx_return_initialize_error_flag, 0
            if name in ("Tip",):
                return self.simx_return_ok, 100
            return self.simx_return_initialize_error_flag, 0

        def simxGetObjectFloatParameter(self, client_id, handle, parameter, opmode):
            return self.simx_return_ok, 2.0 * np.pi

        def simxGetJointPosition(self, client_id, handle, opmode):
            try:
                index = int(handle) - 1
                return self.simx_return_ok, float(self._actual_q()[index])
            except Exception as exc:
                self._last_error = str(exc)
                return self.simx_return_initialize_error_flag, 0.0

        def simxSetJointPosition(self, client_id, handle, position, opmode):
            self._last_error = "Direct joint-position commands are disabled in ROS driver mode."
            return self.simx_return_initialize_error_flag

        def simxGetObjectPosition(self, client_id, handle, relative_to_handle, opmode):
            try:
                q = self._actual_q()
                try:
                    from __main__ import UR3eKinematics
                    tcp = UR3eKinematics().forward_kinematics(q)[:3, 3]
                except Exception as exc:
                    raise RuntimeError(f"Could not compute UR3e FK from joint feedback: {exc}") from exc
                return self.simx_return_ok, np.asarray(tcp, dtype=float).tolist()
            except Exception as exc:
                self._last_error = str(exc)
                return self.simx_return_initialize_error_flag, [0.0, 0.0, 0.0]

        def simxSetJointTargetVelocity(self, client_id, handle, velocity, opmode):
            try:
                index = int(handle) - 1
                if index < 0 or index >= 6:
                    raise RuntimeError(f"Invalid UR3e joint handle: {handle}")
                if self._paused:
                    if self._pending_velocities is None:
                        self._pending_velocities = np.full(6, np.nan, dtype=float)
                    self._pending_velocities[index] = float(velocity)
                else:
                    self._pending_velocities = np.zeros(6, dtype=float)
                    self._pending_velocities[index] = float(velocity)
                    self._flush_pending_commands()
                return self.simx_return_ok
            except Exception as exc:
                self._last_error = str(exc)
                return self.simx_return_initialize_error_flag

        def diagnostic_report(self):
            def stats(values):
                if not values:
                    return {"mean": None, "max": None, "min": None}
                arr = np.asarray(values, dtype=float)
                return {"mean": float(np.mean(arr)), "max": float(np.max(arr)), "min": float(np.min(arr))}

            return {
                "transport": "ros_ur_driver",
                "command_backend": self.command_backend,
                "control_mode": self.control_mode,
                "joint_state_topic": self.joint_state_topic,
                "velocity_command_topic": self.velocity_command_topic,
                "position_command_topic": self.position_command_topic,
                "effective_control_period_s": self.stepping_dt,
                "enable_commands": self.enable_commands,
                "max_abs_qdot": self.max_abs_qdot,
                "max_qdot_delta": self.max_qdot_delta,
                "max_position_step": self.max_position_step,
                "command_count": int(self._command_count),
                "blocked_command_count": int(self._blocked_command_count),
                "raw_velocity_max_abs": float(self._raw_velocity_max_abs),
                "abs_clip_count": int(self._abs_clip_count),
                "delta_clip_count": int(self._delta_clip_count),
                "target_or_velocity_step_norm": stats(self._target_step_samples),
                "last_error": self._last_error,
            }

    return ROSURDriverAdapter
