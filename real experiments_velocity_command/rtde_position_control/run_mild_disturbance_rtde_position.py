# Complete live experiment entry for direct VSCode execution.
# This file is intentionally self-contained for direct VSCode execution.

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogFormatterMathtext, LogLocator, MaxNLocator, ScalarFormatter
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


def find_workspace_code_root(start_dir):
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "core").is_dir() and (candidate / "sim").is_dir():
            return candidate
        nested = candidate / "code"
        if (nested / "core").is_dir() and (nested / "sim").is_dir():
            return nested
    return start_dir


CODE_ROOT = find_workspace_code_root(CURRENT_DIR)
SIM_DIR = CODE_ROOT / 'sim'
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError as exc:
    RemoteAPIClient = None
    ZMQ_IMPORT_ERROR = exc
else:
    ZMQ_IMPORT_ERROR = None

try:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
except ImportError as exc:
    rospy = None
    JointState = None
    Float64MultiArray = None
    ROS_IMPORT_ERROR = exc
else:
    ROS_IMPORT_ERROR = None


DEFAULT_ZMQ_REMOTE_API_PORT = 23000
DEFAULT_REMOTE_API_PORTS = (DEFAULT_ZMQ_REMOTE_API_PORT,)
DEFAULT_THETA_INITIAL = np.array(
    [0.0, -np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0],
    dtype=float,
)
DEFAULT_THETA_LIMIT = 2.0 * np.pi * np.ones(6, dtype=float)
CONTROL_POINT_OBJECT_NAME = 'Tip'
VISUAL_TIP_OBJECT_NAME = 'Tip'
STOP_SIMULATION_ON_EXIT = True


class ZMQCoppeliaSimAdapter:
    simx_return_ok = 0
    simx_return_initialize_error_flag = 1
    simx_opmode_blocking = 0
    simx_opmode_streaming = 1
    simx_opmode_buffer = 2
    simx_opmode_oneshot = 3
    sim_jointfloatparam_upper_limit = "jointfloatparam_upper_limit"

    def __init__(self):
        self.client = None
        self.sim = None
        self.stepping_dt = 0.005

    def _require_sim(self):
        if self.sim is None:
            raise RuntimeError("CoppeliaSim ZMQ remote API is not connected.")
        return self.sim

    def _world_handle(self):
        sim_api = self._require_sim()
        return getattr(sim_api, "handle_world", -1)

    def _resolve_param(self, param):
        sim_api = self._require_sim()
        if param == self.sim_jointfloatparam_upper_limit:
            return getattr(sim_api, "jointfloatparam_upper_limit")
        return param

    def simxFinish(self, client_id):
        if client_id == -1:
            self.client = None
            self.sim = None

    def simxStart(self, host, port, wait_until_connected, do_not_reconnect, timeout_ms, comm_thread_cycle_ms):
        if RemoteAPIClient is None:
            raise RuntimeError(
                "ZMQ remote API requires coppeliasim_zmqremoteapi_client. "
                "Install it in this Python environment and start CoppeliaSim's ZMQ remote API server."
            ) from ZMQ_IMPORT_ERROR
        try:
            self.client = RemoteAPIClient(host=host, port=int(port))
            try:
                self.sim = self.client.require("sim")
            except AttributeError:
                self.sim = self.client.getObject("sim")
            return 0
        except Exception:
            self.client = None
            self.sim = None
            return -1

    def simxSynchronous(self, client_id, enable):
        if self.client is not None:
            self.client.setStepping(bool(enable))
        return self.simx_return_ok

    def simxSynchronousTrigger(self, client_id):
        if self.client is not None:
            self.client.step()
        return self.simx_return_ok

    def simxGetPingTime(self, client_id):
        return self.simx_return_ok

    def simxPauseCommunication(self, client_id, pause):
        return self.simx_return_ok

    def simxStartSimulation(self, client_id, opmode):
        self._require_sim().startSimulation()
        return self.simx_return_ok

    def simxStopSimulation(self, client_id, opmode):
        self._require_sim().stopSimulation()
        return self.simx_return_ok

    def simxGetObjectHandle(self, client_id, object_name, opmode):
        sim_api = self._require_sim()
        candidates = [str(object_name)]
        if not str(object_name).startswith("/"):
            candidates.append("/" + str(object_name))
        for candidate in candidates:
            try:
                return self.simx_return_ok, sim_api.getObject(candidate)
            except Exception:
                continue
        return self.simx_return_initialize_error_flag, 0

    def simxGetObjectFloatParameter(self, client_id, handle, parameter, opmode):
        try:
            return self.simx_return_ok, self._require_sim().getObjectFloatParam(handle, self._resolve_param(parameter))
        except Exception:
            return self.simx_return_initialize_error_flag, 0.0

    def simxGetJointPosition(self, client_id, handle, opmode):
        try:
            return self.simx_return_ok, self._require_sim().getJointPosition(handle)
        except Exception:
            return self.simx_return_initialize_error_flag, 0.0

    def simxSetJointPosition(self, client_id, handle, position, opmode):
        try:
            self._require_sim().setJointPosition(handle, float(position))
            return self.simx_return_ok
        except Exception:
            return self.simx_return_initialize_error_flag

    def simxGetObjectPosition(self, client_id, handle, relative_to_handle, opmode):
        try:
            relative = self._world_handle() if int(relative_to_handle) == -1 else relative_to_handle
            return self.simx_return_ok, self._require_sim().getObjectPosition(handle, relative)
        except Exception:
            return self.simx_return_initialize_error_flag, [0.0, 0.0, 0.0]

    def simxSetJointTargetVelocity(self, client_id, handle, velocity, opmode):
        sim_api = self._require_sim()
        velocity = float(velocity)
        try:
            current = sim_api.getJointPosition(handle)
            sim_api.setJointTargetPosition(handle, current + velocity * float(self.stepping_dt))
        except Exception:
            return self.simx_return_initialize_error_flag
        # Servo-like servoj-equivalent mode: the controller still emits a
        # single-step joint velocity, but CoppeliaSim executes a target-position
        # command through its own joint servo instead of direct setJointPosition.
        return self.simx_return_ok


from rtde_realtime_adapter import make_rtde_adapter_class
ROSRealUR3eAdapter = make_rtde_adapter_class('position')
sim = ROSRealUR3eAdapter()


RUN_TAG_KEYS = (
    "experiment",
    "method",
    "trajectory_name",
    "no_feedback",
    "include_live",
    "duration",
    "offline_duration",
    "tau",
    "task_gain",
    "drift_gain",
    "solver_gamma",
    "activation_power",
    "solver_regularization",
    "theta_dot_limit",
    "eta",
    "dlccznn_inner_steps",
    "pdnn_gain",
    "pdnn_inner_steps",
    "internal_disturbance",
    "disturbance_scale",
    "solver_gamma_gain",
    "rk45_mode",
    "rtol",
    "atol",
    "max_step_factor",
)


def _safe_tag_value(value):
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        value = f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        value = "_".join(_safe_tag_value(v) for v in value)
    text = str(value)
    for old, new in (("-", "m"), (".", "p"), ("+", "p"), (" ", ""), ("/", "_"), ("\\", "_"), (":", "_")):
        text = text.replace(old, new)
    return "".join(ch for ch in text if ch.isalnum() or ch in "_")


def build_run_output_root(base_root, args, mode):
    parts = [mode]
    for key in RUN_TAG_KEYS:
        if hasattr(args, key):
            parts.append(f"{key}_{_safe_tag_value(getattr(args, key))}")
    tag = "__".join(parts)[:170].rstrip("_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(base_root) / "runs" / f"{tag}__{stamp}"


def save_run_config(output_root, args, mode):
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "script": str(Path(__file__).resolve()),
        "output_root": str(output_root),
        "arguments": vars(args),
    }
    (output_root / "run_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

METHOD_DISPLAY = {
    'method1': 'Method 1: vector-error continuous TVQP + ELNCP + DLCCZNN',
    'method2': 'Method 2: scalar-energy continuous TVQP + ELNCP + DLCCZNN',
    'method2_dlccznn': 'Method 2: scalar-energy continuous TVQP + ELNCP + DLCCZNN',
    'method2_pdnn': 'Method 2: continuous TVQP + ELNCP + PDNN',
}

JOINT_COLORS = ('#1f77b4', '#d95f02', '#e6ab02', '#7b3294', '#c49a00', '#56b4e9')
JOINT_LINESTYLES = ('-', '-', '--', '--', ':', ':')
PAPER_LINEWIDTH = 1.6
PAPER_DPI = 280
SMALL_LINEWIDTH = 1.2
PLOT_LABEL_FONTSIZE = 9.5
PLOT_TICK_FONTSIZE = 8
PLOT_LEGEND_FONTSIZE = 7.8


def sign_bi_power(z, power):
    z = np.asarray(z, dtype=float)
    abs_z = np.abs(z)
    power_term = np.where(abs_z == 0.0, 0.0, np.power(abs_z, power))
    return np.sign(z) * power_term


def sig_exp_activation(z, power, exp_clip):
    z = np.asarray(z, dtype=float)
    abs_z = np.abs(z)
    exp_term = np.exp(np.minimum(abs_z, exp_clip))
    return sign_bi_power(z, power=power) * exp_term


def positive_exp_activation(value, power, exp_clip):
    scalar = max(float(value), 0.0)
    scalar = min(scalar, float(exp_clip))
    return (scalar ** power) * np.exp(scalar)


class UR3eKinematics:
    def __init__(self, tcp_offset=None):
        self.d = np.array([0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921], dtype=float)
        self.a = np.array([0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0], dtype=float)
        self.alpha = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=float)
        self.num_joints = 6
        self.tcp_offset = np.array([0.0, 0.0, 0.01], dtype=float) if tcp_offset is None else np.asarray(tcp_offset, dtype=float)

    def transformation_matrix(self, theta_i, a_i, d_i, alpha_i):
        c_theta, s_theta = np.cos(theta_i), np.sin(theta_i)
        c_alpha, s_alpha = np.cos(alpha_i), np.sin(alpha_i)
        return np.array(
            [
                [c_theta, -s_theta * c_alpha, s_theta * s_alpha, a_i * c_theta],
                [s_theta, c_theta * c_alpha, -c_theta * s_alpha, a_i * s_theta],
                [0.0, s_alpha, c_alpha, d_i],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def forward_kinematics(self, theta, return_intermediate=False):
        theta = np.asarray(theta, dtype=float)
        transforms = []
        current = np.eye(4, dtype=float)
        for i in range(self.num_joints):
            current = current @ self.transformation_matrix(theta[i], self.a[i], self.d[i], self.alpha[i])
            transforms.append(current.copy())
        if return_intermediate:
            return transforms
        tcp_transform = current.copy()
        tcp_transform[:3, 3] = current[:3, 3] + current[:3, :3] @ self.tcp_offset
        return tcp_transform

    def jacobian(self, theta):
        transforms = self.forward_kinematics(theta, return_intermediate=True)
        tool_transform = transforms[-1]
        o_n = tool_transform[:3, 3] + tool_transform[:3, :3] @ self.tcp_offset
        jacobian = np.zeros((6, self.num_joints), dtype=float)

        z0 = np.array([0.0, 0.0, 1.0], dtype=float)
        jacobian[:3, 0] = np.cross(z0, o_n)
        jacobian[3:, 0] = z0

        for i in range(1, self.num_joints):
            t_prev = transforms[i - 1]
            o_prev = t_prev[:3, 3]
            z_prev = t_prev[:3, 2]
            jacobian[:3, i] = np.cross(z_prev, o_n - o_prev)
            jacobian[3:, i] = z_prev
        return jacobian


def make_offset_from_initial_position(initial_position, scale):
    heart_start_local = np.array([0.0, 0.0, 5.0], dtype=float)
    return np.asarray(initial_position, dtype=float) - scale * heart_start_local


class HeartTrajectory:
    def __init__(self, duration, scale=0.008, offset=np.array([0.3, 0.0, 0.4], dtype=float)):
        self.duration = float(duration)
        self.scale = float(scale)
        self.offset = np.asarray(offset, dtype=float)
        self.dt = 1e-5

    def _calculate_position(self, t):
        phase = (float(t) / self.duration) * 2.0 * np.pi
        y_heart = 16.0 * (np.sin(phase) ** 3)
        z_heart = 13.0 * np.cos(phase) - 5.0 * np.cos(2.0 * phase) - 2.0 * np.cos(3.0 * phase) - np.cos(4.0 * phase)
        return self.scale * np.array([0.0, y_heart, z_heart], dtype=float) + self.offset

    def get_pose(self, t):
        pos_current = self._calculate_position(t)
        pos_future = self._calculate_position(t + self.dt)
        pos_past = self._calculate_position(t - self.dt)
        velocity = (pos_future - pos_past) / (2.0 * self.dt)
        return pos_current, velocity


class CircleTrajectory:
    def __init__(self, duration, radius=0.032, offset=np.array([0.3, 0.0, 0.4], dtype=float)):
        self.duration = float(duration)
        self.radius = float(radius)
        self.offset = np.asarray(offset, dtype=float)

    def get_pose(self, t):
        local_t = float(t) % self.duration
        u = np.clip(local_t / self.duration, 0.0, 1.0)
        sigma = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        sigma_dot = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / self.duration
        phase = 2.0 * np.pi * sigma
        phase_dot = 2.0 * np.pi * sigma_dot
        position = self.offset + np.array(
            [0.0, self.radius * np.sin(phase), self.radius * (1.0 - np.cos(phase))],
            dtype=float,
        )
        velocity = np.array(
            [0.0, self.radius * phase_dot * np.cos(phase), self.radius * phase_dot * np.sin(phase)],
            dtype=float,
        )
        return position, velocity


class PeriodicCircleTrajectory:
    def __init__(self, duration, radius=0.032, offset=np.array([0.3, 0.0, 0.4], dtype=float)):
        self.duration = float(duration)
        self.radius = float(radius)
        self.offset = np.asarray(offset, dtype=float)

    def get_pose(self, t):
        phase = 2.0 * np.pi * (float(t) / self.duration)
        phase_dot = 2.0 * np.pi / self.duration
        position = self.offset + np.array(
            [0.0, self.radius * np.sin(phase), self.radius * (1.0 - np.cos(phase))],
            dtype=float,
        )
        velocity = np.array(
            [0.0, self.radius * phase_dot * np.cos(phase), self.radius * phase_dot * np.sin(phase)],
            dtype=float,
        )
        return position, velocity


class SmallCircleTrajectory(CircleTrajectory):
    def __init__(self, duration, radius=0.016, offset=np.array([0.3, 0.0, 0.4], dtype=float)):
        super().__init__(duration=duration, radius=radius, offset=offset)


class LineTrajectory:
    def __init__(self, duration, amplitude=0.04, offset=np.array([0.3, 0.0, 0.4], dtype=float)):
        self.duration = float(duration)
        self.amplitude = float(amplitude)
        self.offset = np.asarray(offset, dtype=float)

    def get_pose(self, t):
        local_t = float(t) % self.duration
        u = np.clip(local_t / self.duration, 0.0, 1.0)
        s = 0.5 * (1.0 - np.cos(2.0 * np.pi * u))
        s_dot = (np.pi / self.duration) * np.sin(2.0 * np.pi * u)
        position = self.offset + np.array([0.0, self.amplitude * s, 0.0], dtype=float)
        velocity = np.array([0.0, self.amplitude * s_dot, 0.0], dtype=float)
        return position, velocity


class EllipseTrajectory:
    def __init__(self, duration, radius_y=0.04, radius_z=0.018, offset=np.array([0.3, 0.0, 0.4], dtype=float)):
        self.duration = float(duration)
        self.radius_y = float(radius_y)
        self.radius_z = float(radius_z)
        self.offset = np.asarray(offset, dtype=float)

    def get_pose(self, t):
        local_t = float(t) % self.duration
        u = np.clip(local_t / self.duration, 0.0, 1.0)
        sigma = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        sigma_dot = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / self.duration
        phase = 2.0 * np.pi * sigma
        phase_dot = 2.0 * np.pi * sigma_dot
        position = self.offset + np.array(
            [0.0, self.radius_y * np.sin(phase), self.radius_z * (1.0 - np.cos(phase))],
            dtype=float,
        )
        velocity = np.array(
            [0.0, self.radius_y * phase_dot * np.cos(phase), self.radius_z * phase_dot * np.sin(phase)],
            dtype=float,
        )
        return position, velocity


class FigureEightTrajectory:
    def __init__(self, duration, amplitude_y=0.035, amplitude_z=0.02, offset=np.array([0.3, 0.0, 0.4], dtype=float)):
        self.duration = float(duration)
        self.amplitude_y = float(amplitude_y)
        self.amplitude_z = float(amplitude_z)
        self.offset = np.asarray(offset, dtype=float)

    def get_pose(self, t):
        local_t = float(t) % self.duration
        u = np.clip(local_t / self.duration, 0.0, 1.0)
        sigma = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        sigma_dot = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / self.duration
        phase = 2.0 * np.pi * sigma
        phase_dot = 2.0 * np.pi * sigma_dot
        position = self.offset + np.array(
            [0.0, self.amplitude_y * np.sin(phase), self.amplitude_z * np.sin(phase) * np.cos(phase)],
            dtype=float,
        )
        velocity = np.array(
            [
                0.0,
                self.amplitude_y * phase_dot * np.cos(phase),
                self.amplitude_z * phase_dot * np.cos(2.0 * phase),
            ],
            dtype=float,
        )
        return position, velocity


def build_trajectory(name, duration, scale, initial_position):
    name = str(name).lower()
    initial_position = np.asarray(initial_position, dtype=float)
    if name == 'heart':
        return HeartTrajectory(
            duration=duration,
            scale=scale,
            offset=make_offset_from_initial_position(initial_position, scale),
        ), 'heart'
    if name == 'circle':
        return CircleTrajectory(duration=duration, radius=4.0 * scale, offset=initial_position), 'circle'
    if name == 'circle_periodic':
        return PeriodicCircleTrajectory(duration=duration, radius=4.0 * scale, offset=initial_position), 'circle_periodic'
    if name == 'small_circle':
        return SmallCircleTrajectory(duration=duration, radius=2.0 * scale, offset=initial_position), 'small_circle'
    if name == 'line':
        return LineTrajectory(duration=duration, amplitude=5.0 * scale, offset=initial_position), 'line'
    if name == 'ellipse':
        return EllipseTrajectory(duration=duration, radius_y=5.0 * scale, radius_z=2.5 * scale, offset=initial_position), 'ellipse'
    if name == 'figure8':
        return FigureEightTrajectory(duration=duration, amplitude_y=4.0 * scale, amplitude_z=2.5 * scale, offset=initial_position), 'figure8'
    raise ValueError(f'Unsupported trajectory name: {name}')


def compute_dynamic_velocity_bounds(theta_current, theta_lower, theta_upper, theta_dot_limit, eta, tau):
    theta_current = np.asarray(theta_current, dtype=float)
    theta_dot_lower = -float(theta_dot_limit) * np.ones_like(theta_current)
    theta_dot_upper = float(theta_dot_limit) * np.ones_like(theta_current)
    eta_rate = float(eta) / float(tau)
    xi_minus = np.maximum(theta_dot_lower, eta_rate * (theta_lower - theta_current))
    xi_plus = np.minimum(theta_dot_upper, eta_rate * (theta_upper - theta_current))
    return xi_minus, xi_plus


def connect_to_coppeliasim():
    sim.simxFinish(-1)
    for port in DEFAULT_REMOTE_API_PORTS:
        client_id = sim.simxStart('127.0.0.1', port, True, True, 5000, 5)
        if client_id != -1:
            return client_id, port
    return -1, None


def get_joint_handles(client_id):
    joint_handles = []
    for i in range(1, 7):
        err, handle = sim.simxGetObjectHandle(client_id, f'UR3e_joint{i}', sim.simx_opmode_blocking)
        if err != sim.simx_return_ok:
            raise RuntimeError(f'Could not get UR3e_joint{i}, error={err}')
        joint_handles.append(handle)
    return joint_handles


def get_required_object_handle(client_id, object_name):
    err = sim.simx_return_initialize_error_flag
    handle = 0
    for _ in range(12):
        err, handle = sim.simxGetObjectHandle(client_id, str(object_name), sim.simx_opmode_blocking)
        if err == sim.simx_return_ok:
            break
        time.sleep(0.01)
    if err != sim.simx_return_ok:
        raise RuntimeError(f"Failed to get CoppeliaSim object '{object_name}', error code={err}")
    return handle


def query_joint_limits_from_sim(client_id, joint_handles):
    theta_upper = DEFAULT_THETA_LIMIT.copy()
    theta_lower = -DEFAULT_THETA_LIMIT.copy()
    for i, handle in enumerate(joint_handles):
        err, upper_limit = sim.simxGetObjectFloatParameter(
            client_id,
            handle,
            sim.sim_jointfloatparam_upper_limit,
            sim.simx_opmode_blocking,
        )
        if err == sim.simx_return_ok and upper_limit > 0.0:
            theta_upper[i] = upper_limit
            theta_lower[i] = -upper_limit
    return theta_lower, theta_upper


def configure_servo_like_velocity_control(client_id, joint_handles, max_force=1000.0, max_velocity=2.0):
    sim_api = getattr(sim, 'sim', None)
    report = {
        'command_transport': 'servo_like_servoj_equivalent',
        'manual_position_integration': False,
        'direct_joint_position_write': False,
        'configured': False,
        'max_force': float(max_force),
        'max_velocity': float(max_velocity),
        'joints': [],
    }
    if sim_api is None:
        report['error'] = 'CoppeliaSim API is not connected.'
        return report

    for handle in joint_handles:
        handle = int(handle)
        item = {'handle': handle}
        for key, param_name in (
            ('motor_enabled_before', 'jointintparam_motor_enabled'),
            ('dynctrlmode_before', 'jointintparam_dynctrlmode'),
            ('ctrl_enabled_before', 'jointintparam_ctrl_enabled'),
        ):
            try:
                item[key] = int(sim_api.getObjectInt32Param(handle, getattr(sim_api, param_name)))
            except Exception as exc:
                item[f'{key}_error'] = str(exc)

        try:
            sim_api.setJointMode(handle, sim_api.jointmode_dynamic, 0)
            sim_api.setObjectInt32Param(handle, sim_api.jointintparam_motor_enabled, 1)
            sim_api.setObjectInt32Param(handle, sim_api.jointintparam_dynctrlmode, sim_api.jointdynctrl_position)
            try:
                sim_api.setObjectInt32Param(handle, sim_api.jointintparam_ctrl_enabled, 1)
            except Exception as exc:
                item['ctrl_enable_error'] = str(exc)
            sim_api.setJointTargetForce(handle, float(max_force))
            try:
                sim_api.setObjectFloatParam(handle, sim_api.jointfloatparam_maxvel, float(max_velocity))
            except Exception as exc:
                item['max_velocity_error'] = str(exc)
            sim_api.setJointTargetVelocity(handle, 0.0)
            sim_api.setJointTargetPosition(handle, sim_api.getJointPosition(handle))
            item['configured'] = True
        except Exception as exc:
            item['configured'] = False
            item['configure_error'] = str(exc)

        for key, param_name in (
            ('motor_enabled_after', 'jointintparam_motor_enabled'),
            ('dynctrlmode_after', 'jointintparam_dynctrlmode'),
            ('ctrl_enabled_after', 'jointintparam_ctrl_enabled'),
        ):
            try:
                item[key] = int(sim_api.getObjectInt32Param(handle, getattr(sim_api, param_name)))
            except Exception as exc:
                item[f'{key}_error'] = str(exc)
        try:
            item['target_velocity_after'] = float(sim_api.getJointTargetVelocity(handle))
        except Exception as exc:
            item['target_velocity_after_error'] = str(exc)
        report['joints'].append(item)

    report['configured'] = all(item.get('configured', False) for item in report['joints'])
    return report


def read_joint_positions(client_id, joint_handles, opmode):
    positions = []
    for i, handle in enumerate(joint_handles, start=1):
        err, joint_pos = sim.simxGetJointPosition(client_id, handle, opmode)
        if err != sim.simx_return_ok:
            raise RuntimeError(f'Could not read UR3e_joint{i}, error={err}')
        positions.append(joint_pos)
    return np.asarray(positions, dtype=float)


def begin_joint_position_stream(client_id, joint_handles):
    for handle in joint_handles:
        sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_streaming)


def begin_object_position_stream(client_id, object_handles):
    for handle in object_handles:
        sim.simxGetObjectPosition(client_id, handle, -1, sim.simx_opmode_streaming)


def read_joint_positions_fast(client_id, joint_handles):
    positions = []
    for i, handle in enumerate(joint_handles, start=1):
        joint_pos = None
        err = sim.simx_return_initialize_error_flag
        for _ in range(12):
            err, joint_pos = sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_buffer)
            if err == sim.simx_return_ok:
                break
            err, joint_pos = sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_blocking)
            if err == sim.simx_return_ok:
                break
            time.sleep(0.002)
        if err != sim.simx_return_ok:
            raise RuntimeError(f'Could not read UR3e_joint{i}, error={err}')
        positions.append(joint_pos)
    return np.asarray(positions, dtype=float)


def read_object_position_fast(client_id, object_handle, object_name):
    position = None
    err = sim.simx_return_initialize_error_flag
    for _ in range(12):
        err, position = sim.simxGetObjectPosition(client_id, object_handle, -1, sim.simx_opmode_buffer)
        if err == sim.simx_return_ok:
            break
        err, position = sim.simxGetObjectPosition(client_id, object_handle, -1, sim.simx_opmode_blocking)
        if err == sim.simx_return_ok:
            break
        time.sleep(0.002)
    if err != sim.simx_return_ok:
        raise RuntimeError(f"Failed to read CoppeliaSim object '{object_name}' position, error code={err}")
    return np.asarray(position, dtype=float)


def send_joint_velocity_commands(client_id, joint_handles, joint_velocities):
    sim.simxPauseCommunication(client_id, True)
    try:
        for handle, velocity in zip(joint_handles, joint_velocities):
            sim.simxSetJointTargetVelocity(client_id, handle, float(velocity), sim.simx_opmode_oneshot)
    finally:
        sim.simxPauseCommunication(client_id, False)


def set_joint_positions_direct(client_id, joint_handles, joint_positions):
    sim.simxPauseCommunication(client_id, True)
    try:
        for handle, position in zip(joint_handles, joint_positions):
            err = sim.simxSetJointPosition(client_id, handle, float(position), sim.simx_opmode_oneshot)
            if err != sim.simx_return_ok:
                raise RuntimeError(f'Could not set joint position, error={err}')
    finally:
        sim.simxPauseCommunication(client_id, False)


def reset_simulation_with_toolbar_equivalent(
    client_id,
    joint_handles,
    tau,
    expected_theta=None,
    stop_wait_s=1.0,
    warmup_steps=8,
):
    before_stop = read_joint_positions_fast(client_id, joint_handles)
    sim.stepping_dt = float(tau)
    backend = getattr(sim, 'command_backend', 'topic_position')
    real_ur_backend = backend in {'servoj', 'speedj', 'rtde_servoj', 'rtde_speedj'}
    sim.simxStopSimulation(client_id, sim.simx_opmode_oneshot)
    sim.simxGetPingTime(client_id)
    time.sleep(max(float(stop_wait_s), 0.0))

    if expected_theta is not None and not real_ur_backend:
        expected = np.asarray(expected_theta, dtype=float)
        set_joint_positions_direct(client_id, joint_handles, expected)
        sim.simxGetPingTime(client_id)

    sim.simxSynchronous(client_id, True)
    sim.simxStartSimulation(client_id, sim.simx_opmode_oneshot)
    sim.simxGetPingTime(client_id)

    if expected_theta is not None and not real_ur_backend:
        set_joint_positions_direct(client_id, joint_handles, expected)
        sim.simxGetPingTime(client_id)

    for _ in range(max(int(warmup_steps), 1)):
        if not real_ur_backend:
            sim.simxSynchronousTrigger(client_id)
            sim.simxGetPingTime(client_id)
            time.sleep(min(float(tau), 0.005))

    after_start = read_joint_positions_fast(client_id, joint_handles)
    reset_settle_steps = 0
    if expected_theta is not None:
        expected = np.asarray(expected_theta, dtype=float)
        reset_max_qdot = float(os.environ.get('UR3E_RESET_MAX_QDOT', '0.5' if real_ur_backend else '2.0'))
        reset_tolerance = float(os.environ.get('UR3E_RESET_TOLERANCE_RAD', '0.001' if real_ur_backend else '0.000001'))
        max_reset_steps = int(os.environ.get('UR3E_RESET_MAX_STEPS', '1200' if real_ur_backend else '600'))
        for _ in range(max_reset_steps):
            error = expected - after_start
            if np.max(np.abs(error)) <= reset_tolerance:
                break
            settle_velocity = np.clip(error / max(float(tau), 1e-6), -reset_max_qdot, reset_max_qdot)
            send_joint_velocity_commands(client_id, joint_handles, settle_velocity)
            sim.simxSynchronousTrigger(client_id)
            sim.simxGetPingTime(client_id)
            time.sleep(min(float(tau), 0.005))
            after_start = read_joint_positions_fast(client_id, joint_handles)
            reset_settle_steps += 1
        send_joint_velocity_commands(client_id, joint_handles, np.zeros_like(expected))
    expected_error = None
    max_abs_expected_error = None
    if expected_theta is not None:
        expected_theta = np.asarray(expected_theta, dtype=float)
        expected_error = after_start - expected_theta
        max_abs_expected_error = float(np.max(np.abs(expected_error)))

    report = {
        'reset_mode': 'real_arm_velocity_settle' if real_ur_backend else 'stop_start_toolbar_equivalent',
        'real_command_backend': backend,
        'direct_joint_position_write_used': bool(expected_theta is not None and not real_ur_backend),
        'reset_settle_steps': int(reset_settle_steps),
        'before_stop_rad': np.asarray(before_stop, dtype=float).tolist(),
        'after_start_rad': np.asarray(after_start, dtype=float).tolist(),
        'expected_theta_rad': None if expected_theta is None else expected_theta.tolist(),
        'expected_error_rad': None if expected_error is None else np.asarray(expected_error, dtype=float).tolist(),
        'max_abs_expected_error_rad': max_abs_expected_error,
        'stop_wait_s': float(stop_wait_s),
        'warmup_steps': int(warmup_steps),
    }
    return np.asarray(after_start, dtype=float), report

def startup_handshake_and_settle(
    client_id,
    joint_handles,
    theta_goal,
    settle_steps,
    tau,
    tolerance_rad=1e-3,
    max_extra_steps=400,
):
    theta_goal = np.asarray(theta_goal, dtype=float)
    tolerance_rad = float(tolerance_rad)
    total_steps = int(settle_steps) + int(max_extra_steps)
    theta_measured = read_joint_positions_fast(client_id, joint_handles)

    for step_index in range(total_steps):
        settle_velocity = np.clip(
            (theta_goal - theta_measured) / max(float(tau), 1e-6),
            -2.0,
            2.0,
        )
        send_joint_velocity_commands(client_id, joint_handles, settle_velocity)
        sim.simxSynchronousTrigger(client_id)
        sim.simxGetPingTime(client_id)
        time.sleep(min(float(tau), 0.005))
        theta_measured = read_joint_positions_fast(client_id, joint_handles)
        if step_index + 1 >= int(settle_steps):
            if np.max(np.abs(theta_measured - theta_goal)) <= tolerance_rad:
                break
    send_joint_velocity_commands(client_id, joint_handles, np.zeros_like(theta_goal))
    return theta_goal, theta_measured


def _clip_joint_step(theta_current, tau, qdot_cmd, theta_lower, theta_upper):
    return np.clip(
        np.asarray(theta_current, dtype=float) + float(tau) * np.asarray(qdot_cmd, dtype=float),
        np.asarray(theta_lower, dtype=float),
        np.asarray(theta_upper, dtype=float),
    )


def create_history():
    return {
        'time_s': [],
        'actual_positions': [],
        'fk_positions': [],
        'tip_positions': [],
        'desired_positions': [],
        'position_errors': [],
        'fk_position_errors': [],
        'tip_position_errors': [],
        'joint_positions': [],
        'joint_velocity_commands': [],
        'joint_velocity_raw': [],
        'joint_drift': [],
        'task_residuals': [],
        'solver_residuals': [],
        'solver_energies': [],
        'boundary_slacks': [],
        'drift_norms': [],
    }


def record_history(history, tk, theta_current, sim_control_pos, fk_pos, sim_tip_pos, desired_pos, theta_reference, task_residual=None,
                   solver_residual=None, solver_energy=None, boundary_slack=None, joint_velocity=None,
                   joint_velocity_raw=None):
    drift = np.asarray(theta_current, dtype=float) - np.asarray(theta_reference, dtype=float)
    sim_control_pos = np.asarray(sim_control_pos, dtype=float)
    fk_pos = np.asarray(fk_pos, dtype=float)
    sim_tip_pos = np.asarray(sim_tip_pos, dtype=float)
    desired_pos = np.asarray(desired_pos, dtype=float)
    if joint_velocity is None:
        joint_velocity = np.zeros_like(theta_current, dtype=float)
    if joint_velocity_raw is None:
        joint_velocity_raw = np.asarray(joint_velocity, dtype=float)
    history['time_s'].append(float(tk))
    history['actual_positions'].append(sim_control_pos.tolist())
    history['fk_positions'].append(fk_pos.tolist())
    history['tip_positions'].append(sim_tip_pos.tolist())
    history['desired_positions'].append(desired_pos.tolist())
    history['position_errors'].append(float(np.linalg.norm(sim_control_pos - desired_pos)))
    history['fk_position_errors'].append(float(np.linalg.norm(fk_pos - desired_pos)))
    history['tip_position_errors'].append(float(np.linalg.norm(sim_tip_pos - desired_pos)))
    history['joint_positions'].append(np.asarray(theta_current, dtype=float).tolist())
    history['joint_velocity_commands'].append(np.asarray(joint_velocity, dtype=float).tolist())
    history['joint_velocity_raw'].append(np.asarray(joint_velocity_raw, dtype=float).tolist())
    history['joint_drift'].append(drift.tolist())
    history['drift_norms'].append(float(np.linalg.norm(drift)))
    if task_residual is not None:
        history['task_residuals'].append(float(np.linalg.norm(task_residual)))
    if solver_residual is not None:
        history['solver_residuals'].append(float(solver_residual))
    if solver_energy is not None:
        history['solver_energies'].append(float(solver_energy))
    if boundary_slack is not None:
        history['boundary_slacks'].append(float(boundary_slack))


def configure_3d_axes(ax, desired, actual):
    combined = np.vstack((desired, actual))
    ranges = np.ptp(combined, axis=0)
    safe_ranges = np.maximum(ranges, 1e-6)
    center = np.mean(combined, axis=0)
    half_span = 0.5 * float(np.max(safe_ranges)) * 1.18
    half_span = max(half_span, 0.01)
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect((1.0, 1.0, 1.0))
    if hasattr(ax, 'set_proj_type'):
        ax.set_proj_type('ortho')
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.tick_params(direction='in', top=True, right=True, pad=1, labelsize=8)
    ax.view_init(elev=22, azim=-52)


def configure_time_axes(ax, time_vec):
    ax.set_xlim(float(time_vec[0]), float(time_vec[-1]))
    ax.margins(x=0.02)
    ax.tick_params(direction='in', top=True, right=True, pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def configure_log_axes(ax, values, min_decade=None, max_decade=None, pad_decades=0.5):
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return
    lo = np.floor(np.log10(np.min(positive))) - pad_decades
    hi = np.ceil(np.log10(np.max(positive))) + pad_decades
    if min_decade is not None:
        lo = max(lo, min_decade)
    if max_decade is not None:
        hi = min(hi, max_decade)
    if hi <= lo:
        hi = lo + 1.0
    ax.set_yscale('log')
    ax.set_ylim(10 ** lo, 10 ** hi)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))


def configure_component_scientific_axis(ax, components, force_limit=None, force_power=None):
    components = np.asarray(components, dtype=float)
    finite_abs = np.abs(components[np.isfinite(components)])
    finite_abs = finite_abs[finite_abs > 0.0]
    if finite_abs.size == 0:
        scale_exp = -6 if force_power is None else int(force_power)
        upper = 1e-6
    else:
        component_norm = np.linalg.norm(components, axis=1)
        positive_norm = component_norm[np.isfinite(component_norm) & (component_norm > 0.0)]
        reference = float(np.mean(positive_norm)) if positive_norm.size else float(np.max(finite_abs))
        scale_exp = int(np.floor(np.log10(max(reference, 1e-15)))) if force_power is None else int(force_power)
        upper = float(np.max(finite_abs)) * 1.18
    if force_limit is not None:
        upper = max(upper, float(force_limit))
    if upper <= 0.0:
        upper = 10.0 ** scale_exp
    ax.set_ylim(-upper, upper)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((scale_exp, scale_exp))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(PLOT_TICK_FONTSIZE)


def joint_curve_style(index):
    return {
        'color': JOINT_COLORS[index % len(JOINT_COLORS)],
        'linestyle': JOINT_LINESTYLES[index % len(JOINT_LINESTYLES)],
        'linewidth': PAPER_LINEWIDTH,
    }


def plot_segmented_3d(ax, points, color, pattern, linewidth, alpha=1.0, zorder=1):
    n = len(points)
    if n < 3:
        return
    unit = max(2, n // 520)
    idx = 0
    pattern_idx = 0
    while idx < n - 1:
        span = max(2, int(pattern[pattern_idx % len(pattern)]) * unit)
        end = min(n, idx + span)
        if pattern_idx % 2 == 0 and end - idx >= 2:
            segment = points[idx:end]
            ax.plot(
                segment[:, 0], segment[:, 1], segment[:, 2],
                color=color, linestyle='-', linewidth=linewidth, alpha=alpha, zorder=zorder,
            )
        idx = end
        pattern_idx += 1


def add_trajectory_legend(ax):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='blue', linestyle='--', linewidth=2.8, label='Actual trajectory'),
        Line2D([0], [0], color='red', linestyle='-.', linewidth=2.2, label='Desired trajectory'),
    ]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 0.99), ncol=1, frameon=True,
              fancybox=False, edgecolor='black', fontsize=9, borderpad=0.25,
              handlelength=2.2, handletextpad=0.45)


def plot_interleaved_trajectory(ax, actual, desired, color_actual='#1f77b4', color_desired='#d62728'):
    n = len(desired)
    if n < 3:
        ax.plot(desired[:, 0], desired[:, 1], desired[:, 2], color=color_desired, linestyle='-', linewidth=1.7, alpha=0.95, label='Desired trajectory')
        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], color=color_actual, linestyle='-', linewidth=1.6, alpha=0.95, label='Actual trajectory')
        return
    segment_edges = np.linspace(0, n - 1, 24, dtype=int)
    for idx in range(len(segment_edges) - 1):
        s = segment_edges[idx]
        e = segment_edges[idx + 1] + 1
        if e - s < 2:
            continue
        if idx % 2 == 0:
            ax.plot(
                desired[s:e, 0], desired[s:e, 1], desired[s:e, 2],
                color=color_desired, linestyle='-', linewidth=2.0, alpha=0.98,
                label='Desired trajectory' if idx == 0 else None,
            )
        else:
            ax.plot(
                actual[s:e, 0], actual[s:e, 1], actual[s:e, 2],
                color=color_actual, linestyle='-', linewidth=1.9, alpha=0.95,
                label='Actual trajectory' if idx == 1 else None,
            )


def plot_trajectory(history, output_path, title_prefix):
    desired = np.asarray(history['desired_positions'], dtype=float)
    actual = np.asarray(history['actual_positions'], dtype=float)
    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(
        desired[:, 0], desired[:, 1], desired[:, 2],
        color='red', linestyle='-.', linewidth=2.45, alpha=1.0, label='Desired trajectory',
    )
    plot_segmented_3d(ax, actual, color='blue', pattern=(1, 4), linewidth=2.15, alpha=1.0, zorder=3)
    configure_3d_axes(ax, desired, actual)
    ax.set_xlabel('X(t) (m)', labelpad=8, fontsize=10)
    ax.set_ylabel('Y(t) (m)', labelpad=8, fontsize=10)
    ax.set_zlabel('', labelpad=2)
    ax.text2D(1.08, 0.55, 'Z(t) (m)', transform=ax.transAxes, rotation=90,
              ha='center', va='center', fontsize=10)
    ax.tick_params(labelsize=8, pad=1)
    ax.grid(True, alpha=0.22)
    add_trajectory_legend(ax)
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.02, top=0.98)
    fig.savefig(output_path, dpi=PAPER_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_joint_angles(history, output_path, title_prefix):
    joints = np.asarray(history['joint_positions'], dtype=float)
    time_vec = np.asarray(history['time_s'], dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for i in range(joints.shape[1]):
        ax.plot(time_vec, joints[:, i], label=rf'$\theta^{{{i+1}}}$', **joint_curve_style(i))
    configure_time_axes(ax, time_vec)
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$\theta$ (rad)')
    ax.grid(True, alpha=0.16)
    ax.legend(ncol=3, fontsize=PLOT_LEGEND_FONTSIZE, frameon=True, fancybox=False, edgecolor='black',
              handlelength=1.4, columnspacing=0.65, handletextpad=0.3, borderpad=0.22,
              loc='upper center', bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path, dpi=PAPER_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_joint_velocities(history, output_path, title_prefix):
    velocities = np.asarray(history['joint_velocity_commands'], dtype=float)
    time_vec = np.asarray(history['time_s'], dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for i in range(velocities.shape[1]):
        ax.plot(time_vec, velocities[:, i], label=rf'$\dot{{\theta}}^{{{i+1}}}$', **joint_curve_style(i))
    configure_time_axes(ax, time_vec)
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    ax.grid(True, alpha=0.16)
    ax.legend(ncol=3, fontsize=PLOT_LEGEND_FONTSIZE, frameon=True, fancybox=False, edgecolor='black',
              handlelength=1.4, columnspacing=0.65, handletextpad=0.3, borderpad=0.22,
              loc='upper center', bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path, dpi=PAPER_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_position_error(history, output_path, title_prefix):
    errors = np.asarray(history['position_errors'], dtype=float)
    time_vec = np.asarray(history['time_s'], dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    plot_errors = np.where(errors > 0, errors, np.nan)
    ax.plot(time_vec, plot_errors, color='#1f4ed8', linewidth=1.6)
    configure_time_axes(ax, time_vec)
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$||x(q)-x_d||_2$ (m)')
    positive = errors[np.isfinite(errors) & (errors > 0)]
    if positive.size > 0:
        # Ignore near-zero initial alignment points when choosing the live display scale.
        main_floor = max(float(np.quantile(positive, 0.01)), float(np.mean(positive)) * 0.2, 1e-12)
        lower = np.floor(np.log10(main_floor)) - 0.35
        upper = np.ceil(np.log10(np.max(positive))) + 0.25
        configure_log_axes(ax, errors, min_decade=lower, max_decade=upper, pad_decades=0.15)
    ax.grid(True, which='major', alpha=0.16)
    ax.grid(True, which='minor', alpha=0.08)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PAPER_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_position_error_components(history, output_path, title_prefix, y_limits=None, y_power=None):
    desired = np.asarray(history['desired_positions'], dtype=float)
    actual = np.asarray(history['actual_positions'], dtype=float)
    error_components = actual - desired
    time_vec = np.asarray(history['time_s'], dtype=float)
    fig, ax = plt.subplots(figsize=(4.9, 3.7))
    ax.plot(time_vec, error_components[:, 0], color='#0072BD', linestyle='-', linewidth=1.35, label=r'$\epsilon_x(t)$')
    ax.plot(time_vec, error_components[:, 1], color='#D95319', linestyle='--', linewidth=1.35, label=r'$\epsilon_y(t)$')
    ax.plot(time_vec, error_components[:, 2], color='#EDB120', linestyle='-.', linewidth=1.35, label=r'$\epsilon_z(t)$')
    configure_time_axes(ax, time_vec)
    ax.set_xlabel(r'Time $t$ (s)', fontsize=12)
    ax.set_ylabel(r'$\epsilon$ (m)', fontsize=12)
    if y_limits is not None:
        ax.set_ylim(y_limits)
        formatter = ScalarFormatter(useMathText=True)
        limit_exp = int(np.floor(np.log10(max(abs(y_limits[0]), abs(y_limits[1]), 1e-15)))) if y_power is None else int(y_power)
        formatter.set_powerlimits((limit_exp, limit_exp))
        ax.yaxis.set_major_formatter(formatter)
    else:
        configure_component_scientific_axis(ax, error_components, force_power=y_power)
    ax.grid(True, alpha=0.18)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=True,
              fancybox=False, edgecolor='black', fontsize=9, borderpad=0.25,
              handlelength=2.0, handletextpad=0.45)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path, dpi=PAPER_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_joint_drift(history, output_path, title_prefix):
    drift = np.asarray(history['joint_drift'], dtype=float)
    time_vec = np.asarray(history['time_s'], dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for i in range(drift.shape[1]):
        ax.plot(time_vec, drift[:, i], label=rf'$\Delta\theta^{{{i+1}}}$', **joint_curve_style(i))
    configure_time_axes(ax, time_vec)
    ax.axhline(0.0, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$\Delta\theta$ (rad)')
    ax.grid(True, alpha=0.16)
    ax.legend(ncol=3, fontsize=PLOT_LEGEND_FONTSIZE, frameon=True, fancybox=False, edgecolor='black',
              handlelength=1.4, columnspacing=0.65, handletextpad=0.3, borderpad=0.22,
              loc='upper center', bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path, dpi=PAPER_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_task_residual(history, output_path, title_prefix):
    residuals = np.asarray(history.get('task_residuals', []), dtype=float)
    if len(residuals) == 0:
        return
    time_vec = np.asarray(history['time_s'], dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    plot_residuals = np.where(residuals > 0, residuals, np.nan)
    ax.plot(time_vec, plot_residuals, color='#2ca02c', linewidth=2.0)
    configure_time_axes(ax, time_vec)
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$||J\dot{q} - b||_2$')
    positive = residuals[np.isfinite(residuals) & (residuals > 0)]
    if positive.size > 0:
        lower = np.floor(np.log10(np.min(positive))) - 1.0
        upper = np.ceil(np.log10(np.max(positive))) + 0.5
        configure_log_axes(ax, residuals, min_decade=lower, max_decade=upper, pad_decades=0.15)
    ax.grid(True, which='major', alpha=0.16)
    ax.grid(True, which='minor', alpha=0.08)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PAPER_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_solver_residual(history, output_path, title_prefix):
    residuals = np.asarray(history.get('solver_residuals', []), dtype=float)
    if len(residuals) == 0:
        return
    time_vec = np.asarray(history['time_s'], dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    plot_residuals = np.where(residuals > 0, residuals, np.nan)
    ax.plot(time_vec, plot_residuals, color='#8e44ad', linewidth=2.0)
    configure_time_axes(ax, time_vec)
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$||F(y,t)||_2$')
    positive = residuals[np.isfinite(residuals) & (residuals > 0)]
    if positive.size > 0:
        lower = np.floor(np.log10(np.min(positive))) - 1.0
        upper = np.ceil(np.log10(np.max(positive))) + 0.5
        configure_log_axes(ax, residuals, min_decade=lower, max_decade=upper, pad_decades=0.15)
    ax.grid(True, which='major', alpha=0.16)
    ax.grid(True, which='minor', alpha=0.08)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PAPER_DPI, bbox_inches='tight')
    plt.close(fig)


def save_history_data(output_dir, stem, history):
    arrays = {}
    for key, value in history.items():
        if len(value) == 0:
            continue
        arrays[key] = np.asarray(value, dtype=float)
    np.savez_compressed(output_dir / f'{stem}_history.npz', **arrays)


def save_all_figures(history, output_dir, stem, title_prefix):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory(history, output_dir / f'{stem}_trajectory.png', title_prefix)
    plot_joint_angles(history, output_dir / f'{stem}_joint_angles.png', title_prefix)
    plot_joint_velocities(history, output_dir / f'{stem}_joint_velocities.png', title_prefix)
    plot_position_error(history, output_dir / f'{stem}_position_error.png', title_prefix)
    plot_position_error_components(history, output_dir / f'{stem}_position_error_components.png', title_prefix, y_limits=(-1e-3, 1e-3))
    plot_joint_drift(history, output_dir / f'{stem}_joint_drift.png', title_prefix)
    plot_task_residual(history, output_dir / f'{stem}_task_residual.png', title_prefix)
    plot_solver_residual(history, output_dir / f'{stem}_solver_residual.png', title_prefix)
    save_history_data(output_dir, stem, history)


def save_summary(output_dir, stem, theta_initial, theta_final, history):
    output_dir.mkdir(parents=True, exist_ok=True)
    theta_i = np.asarray(theta_initial, dtype=float)
    theta_f = np.asarray(theta_final, dtype=float)
    delta = theta_f - theta_i

    rows = []
    for i in range(len(theta_i)):
        rows.append({
            'joint': f'theta_{i + 1}',
            'initial_rad': float(theta_i[i]),
            'final_rad': float(theta_f[i]),
            'delta_rad': float(delta[i]),
        })

    errors = np.asarray(history['position_errors'], dtype=float)
    fk_errors = np.asarray(history['fk_position_errors'], dtype=float)
    tip_errors = np.asarray(history['tip_position_errors'], dtype=float)
    drift_norms = np.asarray(history.get('drift_norms', []), dtype=float)
    task_residuals = np.asarray(history.get('task_residuals', []), dtype=float)
    solver_residuals = np.asarray(history.get('solver_residuals', []), dtype=float)
    solver_energies = np.asarray(history.get('solver_energies', []), dtype=float)
    boundary_slacks = np.asarray(history.get('boundary_slacks', []), dtype=float)

    summary = {
        'stem': stem,
        'live_position_error_source': f"CoppeliaSim object '{CONTROL_POINT_OBJECT_NAME}' in world frame",
        'live_fk_position_error_source': 'Local forward kinematics from CoppeliaSim joint feedback',
        'live_visual_tip_error_source': f"CoppeliaSim object '{VISUAL_TIP_OBJECT_NAME}' in world frame",
        'joint_drift_source': 'CoppeliaSim joint-position feedback relative to post-reset initial feedback',
        'theta_final_semantics': 'terminal CoppeliaSim joint feedback after the last command has been executed',
        'theta_initial': theta_i.tolist(),
        'theta_final': theta_f.tolist(),
        'joint_drift_table': rows,
        'max_joint_drift_rad': float(np.max(np.abs(delta))),
        'final_joint_drift_norm_rad': float(np.linalg.norm(delta)),
        'mean_position_error_m': float(np.mean(errors)),
        'final_position_error_m': float(errors[-1]),
        'max_position_error_m': float(np.max(errors)),
        'mean_fk_position_error_m': float(np.mean(fk_errors)),
        'final_fk_position_error_m': float(fk_errors[-1]),
        'max_fk_position_error_m': float(np.max(fk_errors)),
        'mean_tip_position_error_m': float(np.mean(tip_errors)),
        'final_tip_position_error_m': float(tip_errors[-1]),
        'max_tip_position_error_m': float(np.max(tip_errors)),
    }
    if len(drift_norms) > 0:
        summary['max_drift_norm'] = float(np.max(drift_norms))
        summary['final_drift_norm'] = float(drift_norms[-1])
    if len(task_residuals) > 0:
        summary['mean_task_residual'] = float(np.mean(task_residuals))
        summary['final_task_residual'] = float(task_residuals[-1])
    if len(solver_residuals) > 0:
        summary['mean_solver_residual'] = float(np.mean(solver_residuals))
        summary['final_solver_residual'] = float(solver_residuals[-1])
    if len(solver_energies) > 0:
        summary['mean_solver_energy'] = float(np.mean(solver_energies))
        summary['final_solver_energy'] = float(solver_energies[-1])
    if len(boundary_slacks) > 0:
        summary['min_boundary_slack'] = float(np.min(boundary_slacks))
        summary['final_boundary_slack'] = float(boundary_slacks[-1])

    json_path = output_dir / f'{stem}_summary.json'
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    csv_path = output_dir / f'{stem}_joint_drift_table.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['joint', 'initial_rad', 'final_rad', 'delta_rad'])
        writer.writeheader()
        writer.writerows(rows)

    return summary


@dataclass
class ExperimentSettings:
    duration: float = 10.0
    offline_duration: float = 20.0
    trajectory_period: float = 10.0
    tau: float = 0.005
    heart_scale: float = 0.008
    trajectory_name: str = 'heart'
    theta_dot_limit: float = 2.0
    eta: float = 0.9
    dlccznn_inner_steps: int = 1
    pdnn_gain: float = 20.0
    pdnn_inner_steps: int = 1
    pdnn_max_gradient_norm: float = 1e4
    internal_disturbance: str = 'none'
    disturbance_scale: float = 1.0
    task_feedback_limit: float = 0.0
    task_command_filter_alpha: float = 1.0
    command_filter_alpha: float = 1.0
    command_delta_limit: float = 0.0
    command_accel_limit: float = 0.0
    command_jerk_limit: float = 0.0
    theta_initial_command: np.ndarray = field(default_factory=lambda: DEFAULT_THETA_INITIAL.copy())


@dataclass
class MethodConfig:
    tau: float
    task_gain: float
    drift_gain: float
    activation_power: float
    activation_exp_clip: float
    solver_gamma: float
    solver_regularization: float = 1e-8
    ncp_epsilon: float = 1e-10
    theta_dot_limit: float = 2.0
    eta: float = 0.9
    theta_lower: np.ndarray = field(default_factory=lambda: -DEFAULT_THETA_LIMIT.copy())
    theta_upper: np.ndarray = field(default_factory=lambda: DEFAULT_THETA_LIMIT.copy())
    energy_shape_gain: float = 0.0
    energy_shape_clip: float = 4.0
    use_energy_shape: bool = False
    error_mode: str = 'scalar'
    solver_type: str = 'dlccznn'
    drift_feedback_mode: str = 'linear'
    dlccznn_inner_steps: int = 1
    pdnn_gain: float = 20.0
    pdnn_inner_steps: int = 1
    pdnn_max_gradient_norm: float = 1e4
    internal_disturbance: str = 'none'
    disturbance_scale: float = 1.0
    task_feedback_limit: float = 0.0
    task_command_filter_alpha: float = 1.0


class ContinuousTVQPELNCPController:
    def __init__(self, robot, config, method_name):
        self.robot = robot
        self.cfg = config
        self.method_name = method_name
        self.n_joints = robot.num_joints
        self.task_dim = 3
        self.weight_matrix = np.eye(self.n_joints, dtype=float)
        self.theta_initial = None
        self.state = np.zeros(self.n_joints + self.task_dim + self.n_joints, dtype=float)
        self.prev_signals = None
        self.filtered_task_command = None

    def _internal_disturbance_profile(self, t_current):
        """Paper noise profiles used before mapping into the neural dynamics."""
        mode = getattr(self.cfg, 'internal_disturbance', 'none')
        if mode in (None, '', 'none'):
            return 0.0
        t_value = float(t_current)
        if mode == 'constant':
            profile = 5.0
        elif mode == 'linear':
            profile = 0.5 * t_value
        elif mode == 'triangle':
            profile = max(0.0, 0.5 * min(t_value, 20.0 - t_value))
        elif mode == 'quadratic':
            profile = 0.5 * (t_value - 5.0) ** 2
        else:
            raise ValueError(f'Unsupported internal disturbance profile: {mode}')
        return float(getattr(self.cfg, 'disturbance_scale', 1.0)) * profile

    def _internal_disturbance_vector(self, t_current, size):
        """Paper-style additive state noise n(t) in the neural dynamics.

        In Eq. (27), n(t) is added to dot(y); the scalar g.T @ F @ n(t)
        is the induced term in the residual-energy dynamics.
        """
        profile = self._internal_disturbance_profile(t_current)
        if profile == 0.0:
            return np.zeros(size, dtype=float)
        return float(profile) * np.ones(size, dtype=float)

    def reset(self, theta_initial):
        self.theta_initial = np.asarray(theta_initial, dtype=float).copy()
        self.state = np.zeros(self.n_joints + self.task_dim + self.n_joints, dtype=float)
        self.prev_signals = None
        self.filtered_task_command = None

    def update_joint_limits(self, theta_lower, theta_upper):
        self.cfg.theta_lower = np.asarray(theta_lower, dtype=float).copy()
        self.cfg.theta_upper = np.asarray(theta_upper, dtype=float).copy()

    def _compute_velocity_bounds(self, theta_current):
        return compute_dynamic_velocity_bounds(
            theta_current=np.asarray(theta_current, dtype=float),
            theta_lower=self.cfg.theta_lower,
            theta_upper=self.cfg.theta_upper,
            theta_dot_limit=self.cfg.theta_dot_limit,
            eta=self.cfg.eta,
            tau=self.cfg.tau,
        )

    def _drift_feedback(self, drift_delta):
        drift_delta = np.asarray(drift_delta, dtype=float)
        if getattr(self.cfg, 'drift_feedback_mode', 'linear') == 'nonlinear':
            return self.cfg.drift_gain * sig_exp_activation(
                drift_delta,
                power=self.cfg.activation_power,
                exp_clip=self.cfg.activation_exp_clip,
            )
        return self.cfg.drift_gain * drift_delta

    def _positive_scalar_activation(self, value):
        return positive_exp_activation(
            value=value,
            power=self.cfg.activation_power,
            exp_clip=self.cfg.activation_exp_clip,
        )

    def _vector_activation(self, value):
        return sig_exp_activation(
            value,
            power=self.cfg.activation_power,
            exp_clip=self.cfg.activation_exp_clip,
        )

    def _energy_drive(self, energy):
        return self.cfg.solver_gamma * self._positive_scalar_activation(energy)

    def _compute_dlccznn_y_dot(self, y_state, residual_pack, residual_jacobian, residual_time_part):
        residual = residual_pack['residual']
        solver_energy = 0.5 * float(residual @ residual)
        if self.cfg.error_mode == 'vector':
            target = -self.cfg.solver_gamma * self._vector_activation(residual) - residual_time_part
            damping = max(float(self.cfg.solver_regularization), 1e-10)
            normal_matrix = residual_jacobian @ residual_jacobian.T + damping * np.eye(len(residual), dtype=float)
            y_dot = residual_jacobian.T @ np.linalg.solve(normal_matrix, target)
            scalar_drive = float(np.linalg.norm(target))
        else:
            direction = residual_jacobian.T @ residual
            scalar_drive = self._energy_drive(solver_energy) + float(residual @ residual_time_part)
            denom = float(direction @ direction) + self.cfg.solver_regularization
            if denom <= 0.0:
                denom = 1e-12
            y_dot = -(direction / denom) * scalar_drive
        return y_dot, solver_energy, scalar_drive

    def _build_residual(self, y_state, jacobian_task, task_command, drift_feedback, lower, upper):
        u = y_state[: self.n_joints]
        lam = y_state[self.n_joints : self.n_joints + self.task_dim]
        omega = np.maximum(y_state[self.n_joints + self.task_dim :], 0.0)

        lower_slack = u - lower
        upper_slack = upper - u
        use_lower_branch = lower_slack <= upper_slack
        s_value = np.where(use_lower_branch, lower_slack, upper_slack)
        ds_du = np.where(use_lower_branch, 1.0, -1.0)
        sigma = -ds_du

        sqrt_term = np.sqrt(s_value * s_value + omega * omega + self.cfg.ncp_epsilon)
        pfb_residual = s_value + omega - sqrt_term
        alpha = 1.0 - (s_value / sqrt_term)
        beta = 1.0 - (omega / sqrt_term)

        stationarity = self.weight_matrix @ u + drift_feedback - jacobian_task.T @ lam + sigma * omega
        equality = jacobian_task @ u - task_command
        residual = np.concatenate((stationarity, equality, pfb_residual))

        return {
            'residual': residual,
            'stationarity': stationarity,
            'equality': equality,
            'pfb_residual': pfb_residual,
            'u': u,
            'lambda': lam,
            'omega': omega,
            'sigma': sigma,
            's_value': s_value,
            'ds_du': ds_du,
            'alpha': alpha,
            'beta': beta,
            'use_lower_branch': use_lower_branch,
            'min_boundary_slack': float(np.min(s_value)),
        }

    def _build_residual_jacobian(self, residual_pack, jacobian_task):
        n = self.n_joints
        m = self.task_dim
        total = n + m + n
        fy = np.zeros((total, total), dtype=float)

        sigma = residual_pack['sigma']
        alpha = residual_pack['alpha']
        beta = residual_pack['beta']
        ds_du = residual_pack['ds_du']

        u_slice = slice(0, n)
        lambda_slice = slice(n, n + m)
        omega_slice = slice(n + m, n + m + n)
        r1_slice = slice(0, n)
        r2_slice = slice(n, n + m)
        r3_slice = slice(n + m, n + m + n)

        fy[r1_slice, u_slice] = self.weight_matrix
        fy[r1_slice, lambda_slice] = -jacobian_task.T
        fy[r1_slice, omega_slice] = np.diag(sigma)
        fy[r2_slice, u_slice] = jacobian_task
        fy[r3_slice, u_slice] = np.diag(alpha * ds_du)
        fy[r3_slice, omega_slice] = np.diag(beta)
        return fy

    def _estimate_time_derivatives(self, jacobian_task, task_command, drift_feedback, lower, upper):
        if self.prev_signals is None:
            zero_matrix = np.zeros_like(jacobian_task)
            zero_vector_j = np.zeros_like(task_command)
            zero_vector_u = np.zeros_like(lower)
            return {
                'jacobian_dot': zero_matrix,
                'task_command_dot': zero_vector_j,
                'drift_feedback_dot': zero_vector_u,
                'lower_dot': zero_vector_u,
                'upper_dot': zero_vector_u,
            }

        inv_tau = 1.0 / self.cfg.tau
        return {
            'jacobian_dot': (jacobian_task - self.prev_signals['jacobian_task']) * inv_tau,
            'task_command_dot': (task_command - self.prev_signals['task_command']) * inv_tau,
            'drift_feedback_dot': (drift_feedback - self.prev_signals['drift_feedback']) * inv_tau,
            'lower_dot': (lower - self.prev_signals['lower']) * inv_tau,
            'upper_dot': (upper - self.prev_signals['upper']) * inv_tau,
        }

    def step(self, theta_current, desired_position, desired_velocity, use_feedback=True, t_current=0.0):
        if self.theta_initial is None:
            self.reset(theta_current)

        theta_current = np.asarray(theta_current, dtype=float)
        current_pos = self.robot.forward_kinematics(theta_current)[:3, 3]
        jacobian_task = self.robot.jacobian(theta_current)[: self.task_dim, :]

        position_error = current_pos - desired_position
        if use_feedback:
            task_feedback = -self.cfg.task_gain * position_error
            feedback_limit = float(getattr(self.cfg, 'task_feedback_limit', 0.0))
            feedback_norm = float(np.linalg.norm(task_feedback))
            if feedback_limit > 0.0 and feedback_norm > feedback_limit:
                task_feedback = task_feedback * (feedback_limit / max(feedback_norm, 1e-12))
            task_command = desired_velocity + task_feedback
        else:
            task_command = desired_velocity.copy()
        task_alpha = float(np.clip(getattr(self.cfg, 'task_command_filter_alpha', 1.0), 0.0, 1.0))
        if task_alpha < 1.0:
            if self.filtered_task_command is None:
                self.filtered_task_command = task_command.copy()
            else:
                self.filtered_task_command = (
                    task_alpha * task_command + (1.0 - task_alpha) * self.filtered_task_command
                )
            task_command = self.filtered_task_command.copy()

        drift_delta = theta_current - self.theta_initial
        drift_feedback = self._drift_feedback(drift_delta)
        lower, upper = self._compute_velocity_bounds(theta_current)

        y_state = self.state.copy()
        residual_pack = self._build_residual(
            y_state=y_state,
            jacobian_task=jacobian_task,
            task_command=task_command,
            drift_feedback=drift_feedback,
            lower=lower,
            upper=upper,
        )
        derivatives = self._estimate_time_derivatives(
            jacobian_task=jacobian_task,
            task_command=task_command,
            drift_feedback=drift_feedback,
            lower=lower,
            upper=upper,
        )

        inner_steps = max(int(getattr(self.cfg, 'dlccznn_inner_steps', 1)), 1)
        inner_tau = self.cfg.tau / float(inner_steps)
        y_next = y_state.copy()
        u_next_raw = y_next[: self.n_joints].copy()
        solver_energy = 0.5 * float(residual_pack['residual'] @ residual_pack['residual'])
        scalar_drive = 0.0

        for inner_index in range(inner_steps):
            residual_pack = self._build_residual(
                y_state=y_next,
                jacobian_task=jacobian_task,
                task_command=task_command,
                drift_feedback=drift_feedback,
                lower=lower,
                upper=upper,
            )
            u = residual_pack['u']
            lam = residual_pack['lambda']
            alpha = residual_pack['alpha']
            use_lower_branch = residual_pack['use_lower_branch']

            r1_t = derivatives['drift_feedback_dot'] - derivatives['jacobian_dot'].T @ lam
            r2_t = derivatives['jacobian_dot'] @ u - derivatives['task_command_dot']
            s_partial_dot = np.where(use_lower_branch, -derivatives['lower_dot'], derivatives['upper_dot'])
            r3_t = alpha * s_partial_dot
            residual_time_part = np.concatenate((r1_t, r2_t, r3_t))

            residual_jacobian = self._build_residual_jacobian(residual_pack, jacobian_task)
            y_dot, solver_energy, scalar_drive = self._compute_dlccznn_y_dot(
                y_next,
                residual_pack,
                residual_jacobian,
                residual_time_part,
            )
            y_dot = y_dot + self._internal_disturbance_vector(
                float(t_current) + inner_index * inner_tau,
                y_dot.size,
            )
            y_next = y_next + inner_tau * y_dot
            u_next_raw = y_next[: self.n_joints].copy()
            u_projected = np.clip(u_next_raw, lower, upper)
            lambda_projected = y_next[self.n_joints : self.n_joints + self.task_dim].copy()
            omega_projected = np.maximum(y_next[self.n_joints + self.task_dim :], 0.0)
            y_next = np.concatenate((u_projected, lambda_projected, omega_projected))
            if not np.all(np.isfinite(y_next)):
                raise FloatingPointError('DLCCZNN inner loop diverged; reduce solver gain or inner steps.')

        residual_pack = self._build_residual(
            y_state=y_next,
            jacobian_task=jacobian_task,
            task_command=task_command,
            drift_feedback=drift_feedback,
            lower=lower,
            upper=upper,
        )

        final_residual = residual_pack['residual']
        solver_energy = 0.5 * float(final_residual @ final_residual)
        u_next = np.clip(u_next_raw, lower, upper)
        lambda_next = y_next[self.n_joints : self.n_joints + self.task_dim].copy()
        omega_next = np.maximum(y_next[self.n_joints + self.task_dim :], 0.0)

        self.state = np.concatenate((u_next, lambda_next, omega_next))
        self.prev_signals = {
            'jacobian_task': jacobian_task.copy(),
            'task_command': task_command.copy(),
            'drift_feedback': drift_feedback.copy(),
            'lower': lower.copy(),
            'upper': upper.copy(),
        }

        theta_next = _clip_joint_step(
            theta_current=theta_current,
            tau=self.cfg.tau,
            qdot_cmd=u_next,
            theta_lower=self.cfg.theta_lower,
            theta_upper=self.cfg.theta_upper,
        )

        return {
            'theta_next': theta_next,
            'theta_dot_next': u_next,
            'theta_dot_raw': u_next_raw,
            'dual_next': lambda_next,
            'omega_next': omega_next,
            'current_pos': current_pos,
            'desired_pos': desired_position,
            'desired_vel': desired_velocity,
            'tracking_error': position_error,
            'task_command': task_command,
            'task_residual': jacobian_task @ u_next - task_command,
            'drift_delta': drift_delta,
            'drift_feedback': drift_feedback,
            'solver_residual_norm': float(np.linalg.norm(final_residual)),
            'solver_energy': solver_energy,
            'scalar_drive': scalar_drive,
            'boundary_slack': residual_pack['min_boundary_slack'],
            'xi_minus': lower,
            'xi_plus': upper,
        }


class ContinuousTVQPELNCPPDNNController(ContinuousTVQPELNCPController):
    def step(self, theta_current, desired_position, desired_velocity, use_feedback=True, t_current=0.0):
        if self.theta_initial is None:
            self.reset(theta_current)

        theta_current = np.asarray(theta_current, dtype=float)
        current_pos = self.robot.forward_kinematics(theta_current)[:3, 3]
        jacobian_task = self.robot.jacobian(theta_current)[: self.task_dim, :]

        position_error = current_pos - desired_position
        if use_feedback:
            task_feedback = -self.cfg.task_gain * position_error
            feedback_limit = float(getattr(self.cfg, 'task_feedback_limit', 0.0))
            feedback_norm = float(np.linalg.norm(task_feedback))
            if feedback_limit > 0.0 and feedback_norm > feedback_limit:
                task_feedback = task_feedback * (feedback_limit / max(feedback_norm, 1e-12))
            task_command = desired_velocity + task_feedback
        else:
            task_command = desired_velocity.copy()
        task_alpha = float(np.clip(getattr(self.cfg, 'task_command_filter_alpha', 1.0), 0.0, 1.0))
        if task_alpha < 1.0:
            if self.filtered_task_command is None:
                self.filtered_task_command = task_command.copy()
            else:
                self.filtered_task_command = (
                    task_alpha * task_command + (1.0 - task_alpha) * self.filtered_task_command
                )
            task_command = self.filtered_task_command.copy()

        drift_delta = theta_current - self.theta_initial
        drift_feedback = self._drift_feedback(drift_delta)
        lower, upper = self._compute_velocity_bounds(theta_current)

        y_state = self.state.copy()
        inner_steps = max(int(self.cfg.pdnn_inner_steps), 1)
        inner_tau = self.cfg.tau / float(inner_steps)
        last_raw = y_state[: self.n_joints].copy()
        last_pack = None
        last_jacobian = None

        for inner_index in range(inner_steps):
            residual_pack = self._build_residual(
                y_state=y_state,
                jacobian_task=jacobian_task,
                task_command=task_command,
                drift_feedback=drift_feedback,
                lower=lower,
                upper=upper,
            )
            residual = residual_pack['residual']
            residual_jacobian = self._build_residual_jacobian(residual_pack, jacobian_task)
            gradient = residual_jacobian.T @ residual
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm > self.cfg.pdnn_max_gradient_norm:
                gradient = gradient * (self.cfg.pdnn_max_gradient_norm / max(gradient_norm, 1e-12))
            disturbance = self._internal_disturbance_vector(
                float(t_current) + inner_index * inner_tau,
                y_state.size,
            )
            y_next = y_state - inner_tau * float(self.cfg.pdnn_gain) * gradient
            y_next = y_next + inner_tau * disturbance
            last_raw = y_next[: self.n_joints].copy()
            u_next = np.clip(last_raw, lower, upper)
            lambda_next = y_next[self.n_joints : self.n_joints + self.task_dim].copy()
            omega_next = np.maximum(y_next[self.n_joints + self.task_dim :], 0.0)
            y_state = np.concatenate((u_next, lambda_next, omega_next))
            last_pack = residual_pack
            last_jacobian = residual_jacobian
            if not np.all(np.isfinite(y_state)):
                raise FloatingPointError('PDNN solver diverged; reduce pdnn_gain or increase pdnn_inner_steps.')

        final_pack = self._build_residual(
            y_state=y_state,
            jacobian_task=jacobian_task,
            task_command=task_command,
            drift_feedback=drift_feedback,
            lower=lower,
            upper=upper,
        )
        if last_pack is None:
            last_pack = final_pack
        if last_jacobian is None:
            last_jacobian = self._build_residual_jacobian(final_pack, jacobian_task)

        residual = final_pack['residual']
        solver_energy = 0.5 * float(residual @ residual)
        self.state = y_state.copy()
        self.prev_signals = {
            'jacobian_task': jacobian_task.copy(),
            'task_command': task_command.copy(),
            'drift_feedback': drift_feedback.copy(),
            'lower': lower.copy(),
            'upper': upper.copy(),
        }

        u_next = y_state[: self.n_joints].copy()
        lambda_next = y_state[self.n_joints : self.n_joints + self.task_dim].copy()
        omega_next = y_state[self.n_joints + self.task_dim :].copy()
        theta_next = _clip_joint_step(
            theta_current=theta_current,
            tau=self.cfg.tau,
            qdot_cmd=u_next,
            theta_lower=self.cfg.theta_lower,
            theta_upper=self.cfg.theta_upper,
        )

        return {
            'theta_next': theta_next,
            'theta_dot_next': u_next,
            'theta_dot_raw': last_raw,
            'dual_next': lambda_next,
            'omega_next': omega_next,
            'current_pos': current_pos,
            'desired_pos': desired_position,
            'desired_vel': desired_velocity,
            'tracking_error': position_error,
            'task_command': task_command,
            'task_residual': jacobian_task @ u_next - task_command,
            'drift_delta': drift_delta,
            'drift_feedback': drift_feedback,
            'solver_residual_norm': float(np.linalg.norm(residual)),
            'solver_energy': solver_energy,
            'scalar_drive': float(np.linalg.norm(last_jacobian.T @ residual)),
            'boundary_slack': final_pack['min_boundary_slack'],
            'xi_minus': lower,
            'xi_plus': upper,
        }


def build_method1(robot, settings):
    cfg = MethodConfig(
        tau=settings.tau,
        task_gain=120.0,
        drift_gain=0.1,
        activation_power=0.9,
        activation_exp_clip=2.0,
        solver_gamma=120.0,
        theta_dot_limit=settings.theta_dot_limit,
        eta=settings.eta,
        error_mode='vector',
    )
    return ContinuousTVQPELNCPController(robot, cfg, 'method1')


def build_method2(robot, settings):
    cfg = MethodConfig(
        tau=settings.tau,
        task_gain=160.0,
        drift_gain=10.0,
        activation_power=0.9,
        activation_exp_clip=4.0,
        solver_gamma=80.0,
        theta_dot_limit=settings.theta_dot_limit,
        eta=settings.eta,
        dlccznn_inner_steps=getattr(settings, 'dlccznn_inner_steps', 1),
        internal_disturbance=getattr(settings, 'internal_disturbance', 'none'),
        disturbance_scale=getattr(settings, 'disturbance_scale', 1.0),
        task_feedback_limit=getattr(settings, 'task_feedback_limit', 0.0),
        task_command_filter_alpha=getattr(settings, 'task_command_filter_alpha', 1.0),
        error_mode='scalar',
    )
    return ContinuousTVQPELNCPController(robot, cfg, 'method2')


def build_method2_dlccznn(robot, settings):
    controller = build_method2(robot, settings)
    controller.method_name = 'method2_dlccznn'
    controller.cfg.solver_type = 'dlccznn'
    return controller


def build_method2_pdnn(robot, settings):
    cfg = MethodConfig(
        tau=settings.tau,
        task_gain=160.0,
        drift_gain=10.0,
        activation_power=0.9,
        activation_exp_clip=4.0,
        solver_gamma=80.0,
        theta_dot_limit=settings.theta_dot_limit,
        eta=settings.eta,
        error_mode='scalar',
        solver_type='pdnn',
        pdnn_gain=getattr(settings, 'pdnn_gain', 20.0),
        pdnn_inner_steps=getattr(settings, 'pdnn_inner_steps', 1),
        pdnn_max_gradient_norm=getattr(settings, 'pdnn_max_gradient_norm', 1e4),
        internal_disturbance=getattr(settings, 'internal_disturbance', 'none'),
        disturbance_scale=getattr(settings, 'disturbance_scale', 1.0),
        task_feedback_limit=getattr(settings, 'task_feedback_limit', 0.0),
        task_command_filter_alpha=getattr(settings, 'task_command_filter_alpha', 1.0),
    )
    return ContinuousTVQPELNCPPDNNController(robot, cfg, 'method2_pdnn')


METHOD_BUILDERS = {
    'method1': build_method1,
    'method2': build_method2,
    'method2_dlccznn': build_method2_dlccznn,
    'method2_pdnn': build_method2_pdnn,
}


def apply_config_overrides(controller, args):
    if args.task_gain is not None:
        controller.cfg.task_gain = float(args.task_gain)
    if args.drift_gain is not None:
        controller.cfg.drift_gain = float(args.drift_gain)
    if args.solver_gamma is not None:
        controller.cfg.solver_gamma = float(args.solver_gamma)
    if args.activation_power is not None:
        controller.cfg.activation_power = float(args.activation_power)
    if args.activation_exp_clip is not None:
        controller.cfg.activation_exp_clip = float(args.activation_exp_clip)
    if args.drift_feedback_mode is not None:
        controller.cfg.drift_feedback_mode = str(args.drift_feedback_mode)
    if args.solver_regularization is not None:
        controller.cfg.solver_regularization = float(args.solver_regularization)
    if args.theta_dot_limit is not None:
        controller.cfg.theta_dot_limit = float(args.theta_dot_limit)
    if args.eta is not None:
        controller.cfg.eta = float(args.eta)
    return controller


def run_live(method_name, controller_builder, robot, settings, output_dir, use_feedback=True, args=None):
    display_name = METHOD_DISPLAY.get(method_name, method_name)
    mode_label = 'with_fb' if use_feedback else 'nofb'
    stem = f'{method_name}_live_{mode_label}'
    title_prefix = f'{display_name} Live ({mode_label})'

    print(f'  [{display_name}] Running live ({mode_label})...')

    client_id, port = connect_to_coppeliasim()
    if client_id == -1:
        print('    CoppeliaSim not available, skipping live.')
        return None, None

    print(f'    Connected to CoppeliaSim on port {port}')
    sim.simxSynchronous(client_id, True)
    sim.simxStartSimulation(client_id, sim.simx_opmode_oneshot)

    controller = controller_builder(robot, settings)
    if args is not None:
        apply_config_overrides(controller, args)
    history = create_history()
    theta_current = None
    runtime_s = 0.0
    total_steps = 0
    pre_reset_report = {}
    post_reset_report = {}
    terminal_feedback_report = {}
    servo_like_config_report = {}

    try:
        joint_handles = get_joint_handles(client_id)
        control_point_handle = get_required_object_handle(client_id, CONTROL_POINT_OBJECT_NAME)
        visual_tip_handle = get_required_object_handle(client_id, VISUAL_TIP_OBJECT_NAME)
        begin_joint_position_stream(client_id, joint_handles)
        begin_object_position_stream(client_id, (control_point_handle, visual_tip_handle))

        theta_lower, theta_upper = query_joint_limits_from_sim(client_id, joint_handles)
        controller.update_joint_limits(theta_lower, theta_upper)
        servo_like_config_report = configure_servo_like_velocity_control(client_id, joint_handles)
        print(
            '    Servo-like velocity control: '
            f"configured={servo_like_config_report.get('configured')}, "
            'manual_position_integration=False',
            flush=True,
        )

        theta_reference, pre_reset_report = reset_simulation_with_toolbar_equivalent(
            client_id,
            joint_handles,
            tau=settings.tau,
            expected_theta=settings.theta_initial_command,
            stop_wait_s=1.0,
            warmup_steps=8,
        )
        servo_like_config_report['after_reset_reconfigure'] = configure_servo_like_velocity_control(
            client_id,
            joint_handles,
        )
        controller.reset(theta_reference)
        print(
            '    Reset before run: '
            f"mode={pre_reset_report['reset_mode']}, "
            f"expected_error={pre_reset_report['max_abs_expected_error_rad']:.3e} rad",
            flush=True,
        )

        initial_pos = read_object_position_fast(client_id, control_point_handle, CONTROL_POINT_OBJECT_NAME)
        trajectory, trajectory_tag = build_trajectory(
            settings.trajectory_name,
            settings.trajectory_period,
            settings.heart_scale,
            initial_pos,
        )

        total_steps = int(round(settings.duration / settings.tau))
        t_start = time.perf_counter()
        previous_command = np.zeros(len(joint_handles), dtype=float)
        previous_acceleration = np.zeros(len(joint_handles), dtype=float)

        for step in range(total_steps):
            tk = step * settings.tau
            theta_current = read_joint_positions_fast(client_id, joint_handles)
            sim_control_pos = read_object_position_fast(client_id, control_point_handle, CONTROL_POINT_OBJECT_NAME)
            sim_tip_pos = read_object_position_fast(client_id, visual_tip_handle, VISUAL_TIP_OBJECT_NAME)
            desired_pos, desired_vel = trajectory.get_pose(tk)
            result = controller.step(theta_current, desired_pos, desired_vel, use_feedback=use_feedback, t_current=tk)
            command_to_send = np.asarray(result['theta_dot_next'], dtype=float)
            if settings.command_delta_limit > 0.0:
                max_delta = float(settings.command_delta_limit) * float(settings.tau)
                command_to_send = previous_command + np.clip(command_to_send - previous_command, -max_delta, max_delta)
            if settings.command_filter_alpha < 1.0:
                alpha = float(np.clip(settings.command_filter_alpha, 0.0, 1.0))
                command_to_send = alpha * command_to_send + (1.0 - alpha) * previous_command
            if settings.command_accel_limit > 0.0 or settings.command_jerk_limit > 0.0:
                desired_acceleration = (command_to_send - previous_command) / float(settings.tau)
                if settings.command_jerk_limit > 0.0:
                    max_accel_delta = float(settings.command_jerk_limit) * float(settings.tau)
                    desired_acceleration = previous_acceleration + np.clip(
                        desired_acceleration - previous_acceleration,
                        -max_accel_delta,
                        max_accel_delta,
                    )
                if settings.command_accel_limit > 0.0:
                    desired_acceleration = np.clip(
                        desired_acceleration,
                        -float(settings.command_accel_limit),
                        float(settings.command_accel_limit),
                    )
                command_to_send = previous_command + float(settings.tau) * desired_acceleration
                previous_acceleration = desired_acceleration.copy()
            else:
                previous_acceleration = (command_to_send - previous_command) / float(settings.tau)
            previous_command = command_to_send.copy()
            if hasattr(controller, 'state') and hasattr(controller, 'n_joints'):
                controller.state[: controller.n_joints] = command_to_send

            fk_pos = result['current_pos']
            record_history(
                history,
                tk,
                theta_current,
                sim_control_pos,
                fk_pos,
                sim_tip_pos,
                desired_pos,
                theta_reference,
                task_residual=result.get('task_residual'),
                solver_residual=result.get('solver_residual_norm'),
                solver_energy=result.get('solver_energy'),
                boundary_slack=result.get('boundary_slack'),
                joint_velocity=command_to_send,
                joint_velocity_raw=result.get('theta_dot_raw'),
            )

            send_joint_velocity_commands(client_id, joint_handles, command_to_send)
            sim.simxSynchronousTrigger(client_id)
            sim.simxGetPingTime(client_id)

        theta_current = read_joint_positions_fast(client_id, joint_handles)
        terminal_delta = np.asarray(theta_current, dtype=float) - np.asarray(theta_reference, dtype=float)
        terminal_feedback_report = {
            'theta_feedback_rad': np.asarray(theta_current, dtype=float).tolist(),
            'joint_drift_rad': terminal_delta.tolist(),
            'joint_drift_norm_rad': float(np.linalg.norm(terminal_delta)),
        }

        runtime_s = time.perf_counter() - t_start
        print(f'    Runtime: {runtime_s:.1f}s, steps: {total_steps}')
        _, post_reset_report = reset_simulation_with_toolbar_equivalent(
            client_id,
            joint_handles,
            tau=settings.tau,
            expected_theta=theta_reference,
            stop_wait_s=1.0,
            warmup_steps=8,
        )
        print(
            '    Reset after run: '
            f"mode={post_reset_report['reset_mode']}, "
            f"expected_error={post_reset_report['max_abs_expected_error_rad']:.3e} rad",
            flush=True,
        )

    finally:
        if STOP_SIMULATION_ON_EXIT:
            sim.simxStopSimulation(client_id, sim.simx_opmode_oneshot)
        sim.simxFinish(client_id)

    if args is None or not getattr(args, 'skip_plots', False):
        save_all_figures(history, output_dir, stem, title_prefix)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_history_data(output_dir, stem, history)
    summary = save_summary(output_dir, stem, theta_reference, theta_current, history)
    summary['runtime_s'] = runtime_s
    summary['avg_step_time_s'] = runtime_s / max(total_steps, 1)
    summary['physical_reset_before_run'] = pre_reset_report
    summary['physical_reset_after_run'] = post_reset_report
    summary['terminal_feedback_after_last_command'] = terminal_feedback_report
    summary['servo_like_velocity_control'] = servo_like_config_report
    summary['command_transport'] = 'real_ur3e_configurable_transport'
    if hasattr(sim, 'diagnostic_report'):
        summary['real_transport_diagnostics'] = sim.diagnostic_report()
    summary['manual_position_integration'] = False
    summary['direct_joint_position_write'] = False
    summary['trajectory_name'] = trajectory_tag
    summary['solver_type'] = getattr(controller.cfg, 'solver_type', '')
    summary['task_gain'] = getattr(controller.cfg, 'task_gain', '')
    summary['drift_gain'] = getattr(controller.cfg, 'drift_gain', '')
    summary['drift_feedback_mode'] = getattr(controller.cfg, 'drift_feedback_mode', '')
    summary['solver_gamma'] = getattr(controller.cfg, 'solver_gamma', '')
    summary['activation_power'] = getattr(controller.cfg, 'activation_power', '')
    summary['activation_exp_clip'] = getattr(controller.cfg, 'activation_exp_clip', '')
    summary['dlccznn_inner_steps'] = getattr(controller.cfg, 'dlccznn_inner_steps', '')
    summary['pdnn_gain'] = getattr(controller.cfg, 'pdnn_gain', '')
    summary['pdnn_inner_steps'] = getattr(controller.cfg, 'pdnn_inner_steps', '')
    summary['internal_disturbance'] = getattr(controller.cfg, 'internal_disturbance', 'none')
    summary['disturbance_scale'] = getattr(controller.cfg, 'disturbance_scale', 1.0)
    summary['internal_disturbance_injection'] = 'paper_eq27_state_noise_vector'
    summary['task_feedback_limit'] = settings.task_feedback_limit
    summary['task_command_filter_alpha'] = settings.task_command_filter_alpha
    summary['command_filter_alpha'] = settings.command_filter_alpha
    summary['command_delta_limit'] = settings.command_delta_limit
    summary['command_accel_limit'] = settings.command_accel_limit
    summary['command_jerk_limit'] = settings.command_jerk_limit
    (output_dir / f'{stem}_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return history, summary


def write_comparison_tables(output_root, merged_summaries):
    rows = []
    for key, summary in sorted(merged_summaries.items()):
        rows.append({
            'experiment': key,
            'mean_position_error_m': summary.get('mean_position_error_m', ''),
            'final_position_error_m': summary.get('final_position_error_m', ''),
            'final_joint_drift_norm_rad': summary.get('final_joint_drift_norm_rad', ''),
            'mean_task_residual': summary.get('mean_task_residual', ''),
            'final_solver_residual': summary.get('final_solver_residual', ''),
            'min_boundary_slack': summary.get('min_boundary_slack', ''),
            'runtime_s': summary.get('runtime_s', ''),
        })

    csv_path = output_root / 'comparison_summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'experiment',
                'mean_position_error_m',
                'final_position_error_m',
                'final_joint_drift_norm_rad',
                'mean_task_residual',
                'final_solver_residual',
                'min_boundary_slack',
                'runtime_s',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def append_summary_row(rows, label, summary):
    if not summary:
        return
    rows.append({
        'experiment': label,
        'trajectory_name': summary.get('trajectory_name', ''),
        'solver_type': summary.get('solver_type', ''),
        'dlccznn_inner_steps': summary.get('dlccznn_inner_steps', ''),
        'pdnn_gain': summary.get('pdnn_gain', ''),
        'pdnn_inner_steps': summary.get('pdnn_inner_steps', ''),
        'mean_position_error_m': summary.get('mean_position_error_m', ''),
        'final_position_error_m': summary.get('final_position_error_m', ''),
        'max_position_error_m': summary.get('max_position_error_m', ''),
        'final_joint_drift_norm_rad': summary.get('final_joint_drift_norm_rad', ''),
        'max_joint_drift_rad': summary.get('max_joint_drift_rad', ''),
        'mean_task_residual': summary.get('mean_task_residual', ''),
        'final_task_residual': summary.get('final_task_residual', ''),
        'mean_solver_residual': summary.get('mean_solver_residual', ''),
        'final_solver_residual': summary.get('final_solver_residual', ''),
        'min_boundary_slack': summary.get('min_boundary_slack', ''),
        'runtime_s': summary.get('runtime_s', ''),
        'avg_step_time_s': summary.get('avg_step_time_s', ''),
    })


def write_summary_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'experiment',
        'trajectory_name',
        'solver_type',
        'dlccznn_inner_steps',
        'pdnn_gain',
        'pdnn_inner_steps',
        'mean_position_error_m',
        'final_position_error_m',
        'max_position_error_m',
        'final_joint_drift_norm_rad',
        'max_joint_drift_rad',
        'mean_task_residual',
        'final_task_residual',
        'mean_solver_residual',
        'final_solver_residual',
        'min_boundary_slack',
        'runtime_s',
        'avg_step_time_s',
    ]
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_settings(args, trajectory_name=None, tau=None, pdnn_inner_steps=None, dlccznn_inner_steps=None):
    settings = ExperimentSettings(
        duration=args.duration,
        offline_duration=args.offline_duration,
        trajectory_period=float(args.trajectory_period),
        tau=float(tau if tau is not None else args.tau),
        trajectory_name=trajectory_name if trajectory_name is not None else args.trajectory_name,
        heart_scale=float(args.heart_scale),
        dlccznn_inner_steps=int(
            dlccznn_inner_steps if dlccznn_inner_steps is not None else args.dlccznn_inner_steps
        ),
        pdnn_gain=float(args.pdnn_gain),
        pdnn_inner_steps=int(pdnn_inner_steps if pdnn_inner_steps is not None else args.pdnn_inner_steps),
        pdnn_max_gradient_norm=float(args.pdnn_max_gradient_norm),
        internal_disturbance=str(args.internal_disturbance),
        disturbance_scale=float(args.disturbance_scale),
        task_feedback_limit=float(args.task_feedback_limit),
        task_command_filter_alpha=float(args.task_command_filter_alpha),
        command_filter_alpha=float(args.command_filter_alpha),
        command_delta_limit=float(args.command_delta_limit),
        command_accel_limit=float(args.command_accel_limit),
        command_jerk_limit=float(args.command_jerk_limit),
    )
    if args.theta_dot_limit is not None:
        settings.theta_dot_limit = float(args.theta_dot_limit)
    if args.eta is not None:
        settings.eta = float(args.eta)
    return settings


def run_one_case(method_name, mode, robot, settings, output_dir, use_feedback=True, args=None):
    builder = METHOD_BUILDERS[method_name]
    _, summary = run_live(
        method_name,
        builder,
        robot,
        settings,
        output_dir,
        use_feedback=use_feedback,
        args=args,
    )
    return summary
    raise ValueError(f'Unsupported run mode: {mode}')


def run_pdnn_accuracy(args, robot, output_root, use_feedback=True):
    rows = []
    for inner_steps in args.pdnn_accuracy_steps:
        settings = make_settings(args, trajectory_name='heart', pdnn_inner_steps=inner_steps)
        out_dir = output_root / 'pdnn_continuous_accuracy' / f'heart_pdnn_inner{inner_steps}'
        summary = run_one_case(
            'method2_pdnn',
            'offline',
            robot,
            settings,
            out_dir,
            use_feedback=use_feedback,
            args=args,
        )
        append_summary_row(rows, f'pdnn_accuracy_heart_inner{inner_steps}', summary)
    write_summary_csv(output_root / 'pdnn_continuous_accuracy' / 'pdnn_accuracy_summary.csv', rows)
    return rows


def run_shape_screen(args, robot, output_root, use_feedback=True):
    rows = []
    for shape_name in args.shape_candidates:
        settings = make_settings(args, trajectory_name=shape_name)
        out_dir = output_root / 'shape_screening' / shape_name / 'dlccznn_offline'
        summary = run_one_case(
            'method2_dlccznn',
            'offline',
            robot,
            settings,
            out_dir,
            use_feedback=use_feedback,
            args=args,
        )
        append_summary_row(rows, f'shape_screen_{shape_name}_dlccznn', summary)
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            float(row['mean_position_error_m']),
            float(row['final_joint_drift_norm_rad']),
        ),
    )
    write_summary_csv(output_root / 'shape_screening' / 'shape_screening_summary.csv', rows_sorted)
    selected = [row['trajectory_name'] for row in rows_sorted[:3]]
    (output_root / 'shape_screening' / 'top3_shapes.json').write_text(
        json.dumps({'top3_shapes': selected, 'rows': rows_sorted}, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return rows_sorted, selected


def run_comparison_for_shape(args, robot, output_root, shape_name, use_feedback=True, include_live=False):
    rows = []

    base_dir = output_root / f'comparison_{shape_name}'

    # Group 1: same real-time budget.
    settings = make_settings(args, trajectory_name=shape_name, pdnn_inner_steps=1)
    summary = run_one_case(
        'method2_dlccznn',
        'offline',
        robot,
        settings,
        base_dir / 'group1_same_realtime_budget' / 'dlccznn_offline',
        use_feedback=use_feedback,
        args=args,
    )
    append_summary_row(rows, f'{shape_name}_g1_dlccznn_offline', summary)
    summary = run_one_case(
        'method2_pdnn',
        'offline',
        robot,
        settings,
        base_dir / 'group1_same_realtime_budget' / 'pdnn_inner1_offline',
        use_feedback=use_feedback,
        args=args,
    )
    append_summary_row(rows, f'{shape_name}_g1_pdnn_inner1_offline', summary)

    # Group 2: PDNN inner steps needed to approach DLCCZNN accuracy.
    for inner_steps in args.pdnn_cost_steps:
        settings = make_settings(args, trajectory_name=shape_name, pdnn_inner_steps=inner_steps)
        summary = run_one_case(
            'method2_pdnn',
            'offline',
            robot,
            settings,
            base_dir / 'group2_same_accuracy_cost' / f'pdnn_inner{inner_steps}_offline',
            use_feedback=use_feedback,
            args=args,
        )
        append_summary_row(rows, f'{shape_name}_g2_pdnn_inner{inner_steps}_offline', summary)

    # Group 3: sampling-period robustness.
    for tau_value in args.robustness_taus:
        settings = make_settings(args, trajectory_name=shape_name, tau=tau_value, pdnn_inner_steps=1)
        summary = run_one_case(
            'method2_dlccznn',
            'offline',
            robot,
            settings,
            base_dir / 'group3_sampling_period' / f'dlccznn_tau{tau_value:g}_offline',
            use_feedback=use_feedback,
            args=args,
        )
        append_summary_row(rows, f'{shape_name}_g3_dlccznn_tau{tau_value:g}_offline', summary)
        summary = run_one_case(
            'method2_pdnn',
            'offline',
            robot,
            settings,
            base_dir / 'group3_sampling_period' / f'pdnn_inner1_tau{tau_value:g}_offline',
            use_feedback=use_feedback,
            args=args,
        )
        append_summary_row(rows, f'{shape_name}_g3_pdnn_inner1_tau{tau_value:g}_offline', summary)

    # Group 4: runtime and inner-step scaling.
    for inner_steps in args.pdnn_runtime_steps:
        settings = make_settings(args, trajectory_name=shape_name, pdnn_inner_steps=inner_steps)
        summary = run_one_case(
            'method2_pdnn',
            'offline',
            robot,
            settings,
            base_dir / 'group4_runtime_scaling' / f'pdnn_inner{inner_steps}_offline',
            use_feedback=use_feedback,
            args=args,
        )
        append_summary_row(rows, f'{shape_name}_g4_pdnn_inner{inner_steps}_offline', summary)

    if include_live:
        live_settings = make_settings(args, trajectory_name=shape_name, pdnn_inner_steps=1)
        summary = run_one_case(
            'method2_dlccznn',
            'live',
            robot,
            live_settings,
            base_dir / 'live_representative' / 'dlccznn_live',
            use_feedback=use_feedback,
            args=args,
        )
        append_summary_row(rows, f'{shape_name}_live_dlccznn', summary)
        summary = run_one_case(
            'method2_pdnn',
            'live',
            robot,
            live_settings,
            base_dir / 'live_representative' / 'pdnn_inner1_live',
            use_feedback=use_feedback,
            args=args,
        )
        append_summary_row(rows, f'{shape_name}_live_pdnn_inner1', summary)

    write_summary_csv(base_dir / f'{shape_name}_comparison_summary.csv', rows)
    return rows


def write_experiment_report(output_root, all_rows, selected_shapes):
    docs_dir = PROJECT_ROOT / 'docs'
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / 'pdnn_vs_dlccznn_experiment_report.md'
    summary_csv = output_root / 'pdnn_vs_dlccznn_all_summaries.csv'
    selected_text = ', '.join(selected_shapes) if selected_shapes else 'not selected'

    best_rows = sorted(
        [row for row in all_rows if row.get('mean_position_error_m') not in ('', None)],
        key=lambda row: float(row['mean_position_error_m']),
    )[:8]
    table_lines = [
        '| Experiment | Trajectory | Solver | PDNN steps | Mean position error (m) | Final drift (rad) | Runtime (s) |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in best_rows:
        table_lines.append(
            f"| {row['experiment']} | {row['trajectory_name']} | {row['solver_type']} | "
            f"{row['pdnn_inner_steps']} | {float(row['mean_position_error_m']):.6e} | "
            f"{float(row['final_joint_drift_norm_rad']):.6e} | {float(row['runtime_s']):.3f} |"
        )

    report = f"""---
title: PDNN 与 DLCCZNN 在 TVQP + ELNCP 重复运动学中的对照实验
created: 2026-05-07
project: project_tvqp_elncp_pdnn_comparison
---

# PDNN 与 DLCCZNN 在 TVQP + ELNCP 重复运动学中的对照实验

## 实验目的

本实验固定上层 Method 2 连续时变二次规划与 ELNCP 不等式约束处理方式，只替换底层神经动力学求解器。对照对象为已有的 TVQP + ELNCP + DLCCZNN 方法，新建基线为 TVQP + ELNCP + PDNN 方法。

## 统一残差

设求解变量为

$$
y(t)=
\\begin{{bmatrix}}
u(t)\\\\
\\lambda(t)\\\\
\\omega(t)
\\end{{bmatrix}},
$$

其中 $u(t)$ 为关节速度，$\\lambda(t)$ 为任务等式约束乘子，$\\omega(t)$ 为 ELNCP 引入的非负互补变量。统一残差写为

$$
F(y,t)=
\\begin{{bmatrix}}
Wu+\\mu(q-q_0)-J(q)^T\\lambda+\\sigma\\omega\\\\
J(q)u-b(t,q)\\\\
\\psi_\\varepsilon(s(u),\\omega)
\\end{{bmatrix}}.
$$

ELNCP 的边界松弛量为

$$
s_i(u_i)=\\min(u_i-\\xi_i^-,\\xi_i^+-u_i),
$$

扰动 Fischer Burmeister 函数为

$$
\\psi_\\varepsilon(s_i,\\omega_i)
=s_i+\\omega_i-\\sqrt{{s_i^2+\\omega_i^2+\\varepsilon}}.
$$

## PDNN 求解动态

本项目采用残差能量型 PDNN：

$$
V(y,t)=\\frac12 F(y,t)^T F(y,t),
$$

于是

$$
\\dot y=-\\rho \\nabla_y V(y,t)
=-\\rho F_y(y,t)^T F(y,t).
$$

当需要离散执行时，使用与控制更新一致的显式欧拉格式：

$$
y_{{k+1}}=y_k-\\tau\\rho F_y(y_k,t_k)^TF(y_k,t_k).
$$

若使用 $N_p$ 个 PDNN 内部小步，则

$$
h_p=\\frac{{\\tau}}{{N_p}},
\\qquad
y^{{r+1}}=y^r-h_p\\rho F_y(y^r,t_k)^TF(y^r,t_k).
$$

这组实验中，$N_p=1$ 表示与控制周期同频的一步 PDNN，$N_p>1$ 用于测试 PDNN 连续模型在更细欧拉积分下的极限精度。

## 优选图形

简单图形筛选得到的前三个图形为：

{selected_text}

## 结果总表

完整 CSV 位于：

`{summary_csv}`

下面列出按平均位置误差排序的前若干项。

{chr(10).join(table_lines)}

## 初步结论

1. 若 PDNN 在 $N_p=1$ 时明显弱于 DLCCZNN，但增加 $N_p$ 后逐步接近，则说明 PDNN 对连续积分步长更敏感，而 DLCCZNN 更适合直接离散控制周期。
2. 若 PDNN 需要大量内部欧拉小步才能达到相近误差，则其运行成本会随 $N_p$ 线性增长，这可以作为凸显 DLCCZNN 离散结构优势的对照证据。
3. 简单图形的误差通常受轨迹曲率、起止速度连续性和冗余关节漂移分配共同影响，不能只按几何形状复杂度判断。
"""
    report_path.write_text(report, encoding='utf-8')
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(description='TVQP + ELNCP + PDNN versus DLCCZNN comparison runner')
    parser.add_argument(
        '--live-backend',
        choices=['real_ur3e', 'coppelia_zmq'],
        default='real_ur3e',
        help='Execution backend for live runs.',
    )
    parser.add_argument(
        '--experiment',
        choices=['single', 'pdnn_accuracy', 'shape_screen', 'comparison', 'full_pipeline'],
        default='single',
    )
    parser.add_argument(
        '--method',
        choices=['method2_pdnn', 'method2_dlccznn'],
        default='method2_dlccznn',
        help='Only used when --experiment single.',
    )
    parser.add_argument('--trajectory-name', default='circle')
    parser.add_argument('--no-feedback', action='store_true', help='Remove position feedback from task equality')
    parser.add_argument('--duration', type=float, default=20.0, help='Live experiment duration (s)')
    parser.add_argument('--offline-duration', type=float, default=20.0, help='Offline experiment duration (s)')
    parser.add_argument('--trajectory-period', type=float, default=10.0, help='Reference trajectory period (s)')
    parser.add_argument('--heart-scale', type=float, default=0.0175, help='Base trajectory scale; circle radius is 4 * heart_scale.')
    parser.add_argument('--tau', type=float, default=0.005, help='Control time step (s)')
    parser.add_argument('--output-root', default=None, help='Output root directory')
    parser.add_argument('--task-gain', type=float, default=160.0, help='Override task feedback gain')
    parser.add_argument('--drift-gain', type=float, default=10.0, help='Override drift gain')
    parser.add_argument('--solver-gamma', type=float, default=4352.0, help='Override DLCCZNN solver gain')
    parser.add_argument('--activation-power', type=float, default=0.8, help='Override solver activation power')
    parser.add_argument('--activation-exp-clip', type=float, default=4.0, help='Override solver activation exponential clip')
    parser.add_argument('--drift-feedback-mode', choices=['linear', 'nonlinear'], default='nonlinear')
    parser.add_argument('--solver-regularization', type=float, default=None, help='Override solver regularization')
    parser.add_argument('--theta-dot-limit', type=float, default=None, help='Override joint velocity limit')
    parser.add_argument('--real-command-backend', choices=['rtde_servoj', 'rtde_speedj', 'servoj', 'speedj', 'topic_position'], default='rtde_servoj', help='Real UR3e transport backend.')
    parser.add_argument('--ur3e-robot-ip', default=None, help='UR controller IP for servoj/speedj backends.')
    parser.add_argument('--ur3e-script-port', type=int, default=None, help='Ignored by RTDE folders; kept only for CLI compatibility.')
    parser.add_argument('--ur3e-servoj-t', type=float, default=0.005, help='RTDE servoJ t/control-period parameter.')
    parser.add_argument('--ur3e-servoj-lookahead-time', type=float, default=0.05, help='URScript servoj lookahead_time parameter.')
    parser.add_argument('--ur3e-servoj-gain', type=float, default=500.0, help='URScript servoj gain parameter.')
    parser.add_argument('--ur3e-speedj-accel', type=float, default=0.5, help='RTDE speedJ acceleration parameter.')
    parser.add_argument('--ur3e-speedj-t', type=float, default=0.005, help='RTDE speedJ t/control-period parameter.')
    parser.add_argument('--no-wait-fresh-feedback', action='store_true', help='Do not wait for a new /joint_states sample after each command.')
    parser.add_argument('--feedback-wait-timeout', type=float, default=0.03, help='Max seconds to wait for fresh feedback after each command.')
    parser.add_argument('--use-actual-dt', action='store_true', help='Use measured command dt for qdot-to-q integration in position backends.')
    parser.add_argument('--eta', type=float, default=None, help='Override dynamic bound safety coefficient')
    parser.add_argument('--dlccznn-inner-steps', type=int, default=60)
    parser.add_argument('--pdnn-gain', type=float, default=20.0)
    parser.add_argument('--pdnn-inner-steps', type=int, default=1)
    parser.add_argument('--pdnn-max-gradient-norm', type=float, default=1e4)
    parser.add_argument(
        '--internal-disturbance',
        choices=['none', 'constant', 'linear', 'triangle', 'quadratic'],
        default='linear',
        help='Additive internal neural-dynamics disturbance: constant n1=5, linear n2=0.5t, triangle ramps 0.5t to 10s then back to 0 at 20s, quadratic n3=0.5(t-5)^2.',
    )
    parser.add_argument('--disturbance-scale', type=float, default=1.0)
    parser.add_argument(
        '--task-feedback-limit',
        type=float,
        default=0.0,
        help='Optional task-space feedback magnitude limit in m/s before QP; 0 disables limiting.',
    )
    parser.add_argument(
        '--task-command-filter-alpha',
        type=float,
        default=1.0,
        help='Optional first-order smoothing alpha for the task-space command before QP; 1 disables filtering.',
    )
    parser.add_argument(
        '--command-filter-alpha',
        type=float,
        default=1.0,
        help='Optional first-order smoothing alpha for sent joint velocity commands; 1 disables filtering.',
    )
    parser.add_argument(
        '--command-delta-limit',
        type=float,
        default=0.0,
        help='Optional per-joint command slew-rate limit in rad/s^2; 0 disables limiting.',
    )
    parser.add_argument(
        '--command-accel-limit',
        type=float,
        default=0.0,
        help='Optional per-joint acceleration limit for sent velocity commands in rad/s^2; 0 disables limiting.',
    )
    parser.add_argument(
        '--command-jerk-limit',
        type=float,
        default=0.0,
        help='Optional per-joint jerk limit for sent velocity commands in rad/s^3; 0 disables limiting.',
    )
    parser.add_argument('--pdnn-accuracy-steps', type=int, nargs='+', default=[1, 5, 20, 50])
    parser.add_argument('--pdnn-cost-steps', type=int, nargs='+', default=[1, 5, 10, 20, 50])
    parser.add_argument('--pdnn-runtime-steps', type=int, nargs='+', default=[1, 2, 5, 10, 20, 50])
    parser.add_argument('--robustness-taus', type=float, nargs='+', default=[0.002, 0.005, 0.01, 0.02])
    parser.add_argument(
        '--shape-candidates',
        nargs='+',
        default=['circle', 'circle_periodic', 'line', 'ellipse', 'figure8', 'small_circle'],
    )
    parser.add_argument('--top-shape-count', type=int, default=3)
    parser.add_argument('--include-live', action='store_true')
    parser.add_argument('--skip-plots', action='store_true', help='Skip figure generation during parameter sweeps')
    return parser.parse_args()


def main():
    global sim
    args = parse_args()
    if args.live_backend == 'coppelia_zmq':
        sim = ZMQCoppeliaSimAdapter()
    else:
        sim = ROSRealUR3eAdapter()
    if hasattr(sim, 'configure_real_transport'):
        sim.configure_real_transport(
            backend=args.real_command_backend,
            robot_ip=args.ur3e_robot_ip,
            script_port=args.ur3e_script_port,
            servoj_t=args.ur3e_servoj_t,
            servoj_lookahead_time=args.ur3e_servoj_lookahead_time,
            servoj_gain=args.ur3e_servoj_gain,
            speedj_accel=args.ur3e_speedj_accel,
            speedj_t=args.ur3e_speedj_t,
            wait_fresh_feedback=not args.no_wait_fresh_feedback,
            feedback_wait_timeout=args.feedback_wait_timeout,
            use_actual_dt=args.use_actual_dt,
        )
    use_feedback = not args.no_feedback
    fb_label = 'with_feedback' if use_feedback else 'without_feedback'
    output_root = Path(args.output_root) if args.output_root else build_run_output_root(PROJECT_ROOT / 'results' / fb_label, args, "live")
    output_root.mkdir(parents=True, exist_ok=True)
    save_run_config(output_root, args, "live")

    print('=== TVQP + ELNCP + PDNN versus DLCCZNN comparison ===')
    print(f'Output: {output_root}')
    print(f'Feedback: {fb_label}')
    print(f'Experiment: {args.experiment}')
    print()

    robot = UR3eKinematics()
    all_rows = []
    selected_shapes = []

    if args.experiment == 'single':
        settings = make_settings(args)
        case_root = output_root / 'single' / args.trajectory_name / args.method
        summary = run_one_case(args.method, 'live', robot, settings, case_root / 'live', use_feedback, args)
        append_summary_row(all_rows, f'single_{args.trajectory_name}_{args.method}_live', summary)

    elif args.experiment == 'pdnn_accuracy':
        all_rows.extend(run_pdnn_accuracy(args, robot, output_root, use_feedback))

    elif args.experiment == 'shape_screen':
        rows, selected_shapes = run_shape_screen(args, robot, output_root, use_feedback)
        selected_shapes = selected_shapes[: max(int(args.top_shape_count), 1)]
        all_rows.extend(rows)

    elif args.experiment == 'comparison':
        target_shapes = [args.trajectory_name]
        for shape_name in target_shapes:
            all_rows.extend(
                run_comparison_for_shape(
                    args,
                    robot,
                    output_root,
                    shape_name,
                    use_feedback=use_feedback,
                    include_live=True,
                )
            )

    elif args.experiment == 'full_pipeline':
        all_rows.extend(run_pdnn_accuracy(args, robot, output_root, use_feedback))
        screen_rows, selected_shapes = run_shape_screen(args, robot, output_root, use_feedback)
        selected_shapes = selected_shapes[: max(int(args.top_shape_count), 1)]
        all_rows.extend(screen_rows)
        all_rows.extend(
            run_comparison_for_shape(
                args,
                robot,
                output_root,
                'heart',
                use_feedback=use_feedback,
                include_live=True,
            )
        )
        for shape_name in selected_shapes:
            all_rows.extend(
                run_comparison_for_shape(
                    args,
                    robot,
                    output_root,
                    shape_name,
                    use_feedback=use_feedback,
                    include_live=True,
                )
            )

    write_summary_csv(output_root / 'pdnn_vs_dlccznn_all_summaries.csv', all_rows)
    report_path = write_experiment_report(output_root, all_rows, selected_shapes)
    print(f'All summaries saved to {output_root / "pdnn_vs_dlccznn_all_summaries.csv"}')
    print(f'Report saved to {report_path}')

    if all_rows:
        print('\n=== Summary Table ===')
        print(f'{"Experiment":<52} {"MeanPosErr(m)":>15} {"FinalDrift(rad)":>16} {"Runtime(s)":>12}')
        print('-' * 100)
        for row in all_rows:
            if row.get('mean_position_error_m') in ('', None):
                continue
            print(
                f'{row["experiment"]:<52} '
                f'{float(row["mean_position_error_m"]):>15.3e} '
                f'{float(row["final_joint_drift_norm_rad"]):>16.3e} '
                f'{float(row["runtime_s"]):>12.3f}'
            )

    print('\nDone.')


if __name__ == '__main__':
    main()
