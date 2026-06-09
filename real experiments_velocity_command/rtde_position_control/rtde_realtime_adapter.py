import os
import time

import numpy as np

try:
    import rtde_control
    import rtde_receive
except ImportError as exc:
    rtde_control = None
    rtde_receive = None
    RTDE_IMPORT_ERROR = exc
else:
    RTDE_IMPORT_ERROR = None


def make_rtde_adapter_class(default_control_mode):
    default_control_mode = str(default_control_mode).strip().lower()
    if default_control_mode not in {"position", "speed"}:
        raise ValueError("default_control_mode must be 'position' or 'speed'.")

    class RTDERealUR3eAdapter:
        simx_return_ok = 0
        simx_return_initialize_error_flag = 1
        simx_opmode_blocking = 0
        simx_opmode_streaming = 1
        simx_opmode_buffer = 2
        simx_opmode_oneshot = 3
        sim_jointfloatparam_upper_limit = "jointfloatparam_upper_limit"

        def __init__(self):
            self.stepping_dt = float(os.environ.get("UR3E_TAU", "0.005"))
            self.control_mode = default_control_mode
            self.command_backend = (
                "rtde_servoj" if self.control_mode == "position" else "rtde_speedj"
            )
            self.robot_ip = os.environ.get("UR3E_ROBOT_IP", os.environ.get("ROBOT_IP", "")).strip()
            self.servoj_t = float(os.environ.get("UR3E_SERVOJ_T", str(self.stepping_dt)))
            self.servoj_lookahead_time = float(os.environ.get("UR3E_SERVOJ_LOOKAHEAD_TIME", "0.05"))
            self.servoj_gain = float(os.environ.get("UR3E_SERVOJ_GAIN", "500"))
            self.speedj_accel = float(os.environ.get("UR3E_SPEEDJ_ACCEL", "0.5"))
            self.speedj_t = float(os.environ.get("UR3E_SPEEDJ_T", str(self.stepping_dt)))
            self.use_actual_dt_for_position_integration = os.environ.get("UR3E_USE_ACTUAL_DT", "0") == "1"
            self.client = None
            self._rtde_c = None
            self._rtde_r = None
            self._paused = False
            self._pending_positions = None
            self._pending_velocities = None
            self._position_target = None
            self._last_command_wall_time = None
            self._last_period_start = None
            self._last_target_or_velocity = None
            self._last_error = ""
            self._command_count = 0
            self._period_wait_count = 0
            self._command_dt_samples = []
            self._target_step_samples = []
            self._feedback_age_samples = []
            self._speed_scaling_samples = []

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
            if robot_ip is not None:
                self.robot_ip = str(robot_ip).strip()
            if servoj_t is not None:
                self.servoj_t = float(servoj_t)
                self.stepping_dt = float(servoj_t)
            if servoj_lookahead_time is not None:
                self.servoj_lookahead_time = float(servoj_lookahead_time)
            if servoj_gain is not None:
                self.servoj_gain = float(servoj_gain)
            if speedj_accel is not None:
                self.speedj_accel = float(speedj_accel)
            if speedj_t is not None:
                self.speedj_t = float(speedj_t)
                if self.control_mode == "speed":
                    self.stepping_dt = float(speedj_t)
            if use_actual_dt is not None:
                self.use_actual_dt_for_position_integration = bool(use_actual_dt)
            # Keep the folder's command semantics fixed. The CLI argument is accepted
            # for compatibility with the old scripts but cannot silently switch modes.
            requested = "" if backend is None else str(backend).strip().lower()
            allowed_position = {"", "servoj", "rtde_servoj", "position", "topic_position"}
            allowed_speed = {"", "speedj", "rtde_speedj", "speed"}
            if self.control_mode == "position":
                if requested not in allowed_position:
                    raise RuntimeError(
                        f"This folder is RTDE servoJ position-control only; got backend={requested!r}."
                    )
                self.command_backend = "rtde_servoj"
            else:
                if requested not in allowed_speed:
                    raise RuntimeError(
                        f"This folder is RTDE speedJ velocity-control only; got backend={requested!r}."
                    )
                self.command_backend = "rtde_speedj"

        def _require_rtde(self):
            if RTDE_IMPORT_ERROR is not None:
                raise RuntimeError(
                    "Python package ur_rtde is required. Install it on Ubuntu with: "
                    "python3 -m pip install ur_rtde"
                ) from RTDE_IMPORT_ERROR
            if not self.robot_ip:
                raise RuntimeError("Set UR3E_ROBOT_IP or ROBOT_IP before running a real UR3e experiment.")

        def _connect_rtde(self):
            self._require_rtde()
            if self._rtde_c is None:
                self._rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            if self._rtde_r is None:
                self._rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            if self._position_target is None:
                self._position_target = np.asarray(self._rtde_r.getActualQ(), dtype=float)

        def _read_actual_q(self):
            self._connect_rtde()
            return np.asarray(self._rtde_r.getActualQ(), dtype=float)

        def _read_actual_tcp_position(self):
            self._connect_rtde()
            pose = np.asarray(self._rtde_r.getActualTCPPose(), dtype=float)
            if pose.size < 3:
                raise RuntimeError("RTDE getActualTCPPose returned fewer than 3 values.")
            return pose[:3]

        def _read_speed_scaling(self):
            if self._rtde_r is None or not hasattr(self._rtde_r, "getSpeedScaling"):
                return None
            try:
                return float(self._rtde_r.getSpeedScaling())
            except Exception:
                return None

        def _command_dt(self):
            now = time.monotonic()
            if self._last_command_wall_time is None:
                dt = float(self.stepping_dt)
            else:
                dt = max(now - self._last_command_wall_time, 1e-6)
            self._last_command_wall_time = now
            self._command_dt_samples.append(float(dt))
            if len(self._command_dt_samples) > 20000:
                self._command_dt_samples = self._command_dt_samples[-10000:]
            return dt

        def _begin_period(self):
            if self._rtde_c is not None and hasattr(self._rtde_c, "initPeriod"):
                self._last_period_start = self._rtde_c.initPeriod()
            else:
                self._last_period_start = None

        def _wait_period(self):
            if self._rtde_c is not None and self._last_period_start is not None and hasattr(self._rtde_c, "waitPeriod"):
                self._rtde_c.waitPeriod(self._last_period_start)
                self._period_wait_count += 1
                self._last_period_start = None
            else:
                time.sleep(max(float(self.stepping_dt), 0.0))

        def _account_command(self, values):
            values = np.asarray(values, dtype=float)
            if self._last_target_or_velocity is not None and values.shape == self._last_target_or_velocity.shape:
                self._target_step_samples.append(float(np.max(np.abs(values - self._last_target_or_velocity))))
                if len(self._target_step_samples) > 20000:
                    self._target_step_samples = self._target_step_samples[-10000:]
            self._last_target_or_velocity = values.copy()
            self._command_count += 1
            speed_scaling = self._read_speed_scaling()
            if speed_scaling is not None:
                self._speed_scaling_samples.append(speed_scaling)
                if len(self._speed_scaling_samples) > 20000:
                    self._speed_scaling_samples = self._speed_scaling_samples[-10000:]

        def _send_servoj_target(self, target):
            self._connect_rtde()
            target = np.asarray(target, dtype=float)
            if target.shape != (6,):
                raise RuntimeError(f"servoJ target must contain 6 joints, got shape={target.shape}.")
            self._begin_period()
            self._rtde_c.servoJ(
                target.tolist(),
                0.0,
                0.0,
                float(self.stepping_dt),
                float(self.servoj_lookahead_time),
                float(self.servoj_gain),
            )
            self._account_command(target)

        def _send_speedj_velocity(self, velocity):
            self._connect_rtde()
            velocity = np.asarray(velocity, dtype=float)
            if velocity.shape != (6,):
                raise RuntimeError(f"speedJ velocity must contain 6 joints, got shape={velocity.shape}.")
            self._begin_period()
            self._rtde_c.speedJ(velocity.tolist(), float(self.speedj_accel), float(self.stepping_dt))
            self._account_command(velocity)

        def _flush_pending_commands(self):
            self._connect_rtde()
            current_q = self._read_actual_q()
            target = current_q.copy() if self._position_target is None else self._position_target.copy()
            velocity = np.zeros(6, dtype=float)
            has_position = False
            has_velocity = False
            if self._pending_positions is not None:
                mask = np.isfinite(self._pending_positions)
                target[mask] = self._pending_positions[mask]
                has_position = bool(np.any(mask))
            if self._pending_velocities is not None:
                mask = np.isfinite(self._pending_velocities)
                velocity[mask] = self._pending_velocities[mask]
                has_velocity = bool(np.any(mask))
            self._pending_positions = None
            self._pending_velocities = None
            if not has_position and not has_velocity:
                return
            measured_dt = self._command_dt()
            if self.control_mode == "position":
                integration_dt = measured_dt if self.use_actual_dt_for_position_integration else float(self.stepping_dt)
                if has_velocity:
                    target = target + velocity * float(integration_dt)
                self._position_target = target.copy()
                self._send_servoj_target(target)
            else:
                if has_position and not has_velocity:
                    velocity = np.clip((target - current_q) / max(float(self.stepping_dt), 1e-6), -2.0, 2.0)
                self._send_speedj_velocity(velocity)

        def simxStart(self, host, port, wait_until_connected, do_not_reconnect, timeout_ms, comm_thread_cycle_ms):
            self._connect_rtde()
            self.client = 0
            return 0

        def simxFinish(self, client_id):
            try:
                if self._rtde_c is not None:
                    if self.control_mode == "speed" and hasattr(self._rtde_c, "speedStop"):
                        self._rtde_c.speedStop(float(self.speedj_accel))
                    if self.control_mode == "position" and hasattr(self._rtde_c, "servoStop"):
                        self._rtde_c.servoStop()
                    if hasattr(self._rtde_c, "stopScript"):
                        self._rtde_c.stopScript()
            finally:
                self._rtde_c = None
                self._rtde_r = None
            return self.simx_return_ok

        def simxSynchronous(self, client_id, enable):
            return self.simx_return_ok

        def simxSynchronousTrigger(self, client_id):
            self._wait_period()
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
            self._connect_rtde()
            self._position_target = self._read_actual_q()
            return self.simx_return_ok

        def simxStopSimulation(self, client_id, opmode):
            try:
                if self._rtde_c is not None:
                    if self.control_mode == "speed" and hasattr(self._rtde_c, "speedStop"):
                        self._rtde_c.speedStop(float(self.speedj_accel))
                    elif self.control_mode == "position" and hasattr(self._rtde_c, "servoStop"):
                        self._rtde_c.servoStop()
            except Exception as exc:
                self._last_error = str(exc)
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
                return self.simx_return_ok, float(self._read_actual_q()[index])
            except Exception as exc:
                self._last_error = str(exc)
                return self.simx_return_initialize_error_flag, 0.0

        def simxSetJointPosition(self, client_id, handle, position, opmode):
            try:
                index = int(handle) - 1
                if index < 0 or index >= 6:
                    raise RuntimeError(f"Invalid UR3e joint handle: {handle}")
                if self._paused:
                    if self._pending_positions is None:
                        self._pending_positions = np.full(6, np.nan, dtype=float)
                    self._pending_positions[index] = float(position)
                else:
                    target = self._read_actual_q() if self._position_target is None else self._position_target.copy()
                    target[index] = float(position)
                    self._position_target = target.copy()
                    if self.control_mode == "position":
                        self._send_servoj_target(target)
                    else:
                        current = self._read_actual_q()
                        velocity = np.clip((target - current) / max(float(self.stepping_dt), 1e-6), -2.0, 2.0)
                        self._send_speedj_velocity(velocity)
                return self.simx_return_ok
            except Exception as exc:
                self._last_error = str(exc)
                return self.simx_return_initialize_error_flag

        def simxGetObjectPosition(self, client_id, handle, relative_to_handle, opmode):
            try:
                return self.simx_return_ok, self._read_actual_tcp_position().tolist()
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
                return {
                    "mean": float(np.mean(arr)),
                    "max": float(np.max(arr)),
                    "min": float(np.min(arr)),
                }

            actual_qd = None
            try:
                if self._rtde_r is not None and hasattr(self._rtde_r, "getActualQd"):
                    actual_qd = np.asarray(self._rtde_r.getActualQd(), dtype=float).tolist()
            except Exception:
                actual_qd = None
            return {
                "command_backend": self.command_backend,
                "control_mode": self.control_mode,
                "transport": "ur_rtde",
                "robot_ip": self.robot_ip,
                "servoj_t": self.servoj_t,
                "effective_control_period_s": self.stepping_dt,
                "servoj_lookahead_time": self.servoj_lookahead_time,
                "servoj_gain": self.servoj_gain,
                "speedj_accel": self.speedj_accel,
                "speedj_t": self.speedj_t,
                "use_actual_dt_for_position_integration": self.use_actual_dt_for_position_integration,
                "command_count": int(self._command_count),
                "period_wait_count": int(self._period_wait_count),
                "command_dt_s": stats(self._command_dt_samples),
                "target_or_velocity_step_norm": stats(self._target_step_samples),
                "speed_scaling": stats(self._speed_scaling_samples),
                "actual_qd_last": actual_qd,
                "last_error": self._last_error,
            }

    return RTDERealUR3eAdapter
