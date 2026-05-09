# Complete live experiment entry for direct VSCode execution.
# This file is intentionally self-contained for direct VSCode execution.

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent


def find_workspace_code_root(start_dir):
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "core").is_dir() and (candidate / "sim").is_dir():
            return candidate
        nested = candidate / "code"
        if (nested / "core").is_dir() and (nested / "sim").is_dir():
            return nested
    raise RuntimeError(f"Cannot locate workspace code root from {start_dir}")


CODE_ROOT = find_workspace_code_root(CURRENT_DIR)
SIM_DIR = CODE_ROOT / 'sim'
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

import sim


DEFAULT_REMOTE_API_PORTS = (19997, 19998)
DEFAULT_THETA_INITIAL = np.array(
    [0.0, -np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0],
    dtype=float,
)
DEFAULT_THETA_LIMIT = 2.0 * np.pi * np.ones(6, dtype=float)


RUN_TAG_KEYS = (
    "method",
    "no_feedback",
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
)


def _safe_tag_value(value):
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        value = f"{value:.6g}"
    text = str(value)
    for old, new in (("-", "m"), (".", "p"), ("+", "p"), (" ", ""), ("/", "_"), ("\\", "_"), (":", "_")):
        text = text.replace(old, new)
    return "".join(ch for ch in text if ch.isalnum() or ch in "_")


def build_run_output_root(base_root, args, mode):
    parts = [mode]
    for key in RUN_TAG_KEYS:
        if hasattr(args, key):
            parts.append(f"{key}_{_safe_tag_value(getattr(args, key))}")
    tag = "__".join(parts)[:150].rstrip("_")
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
}


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
    def __init__(self):
        self.d = np.array([0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921], dtype=float)
        self.a = np.array([0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0], dtype=float)
        self.alpha = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=float)
        self.num_joints = 6

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
        return transforms if return_intermediate else transforms[-1]

    def jacobian(self, theta):
        transforms = self.forward_kinematics(theta, return_intermediate=True)
        o_n = transforms[-1][:3, 3]
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


def read_joint_positions_fast(client_id, joint_handles):
    positions = []
    for i, handle in enumerate(joint_handles, start=1):
        err, joint_pos = sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_buffer)
        if err == sim.simx_return_ok:
            positions.append(joint_pos)
            continue
        err, joint_pos = sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_blocking)
        if err != sim.simx_return_ok:
            raise RuntimeError(f'Could not read UR3e_joint{i}, error={err}')
        positions.append(joint_pos)
    return np.asarray(positions, dtype=float)


def send_joint_targets(client_id, joint_handles, joint_targets):
    sim.simxPauseCommunication(client_id, True)
    try:
        for handle, target in zip(joint_handles, joint_targets):
            sim.simxSetJointTargetPosition(client_id, handle, float(target), sim.simx_opmode_oneshot)
    finally:
        sim.simxPauseCommunication(client_id, False)


def startup_handshake_and_settle(client_id, joint_handles, theta_goal, settle_steps, tau):
    theta_goal = np.asarray(theta_goal, dtype=float)
    for handle, target in zip(joint_handles, theta_goal):
        sim.simxSetJointTargetPosition(client_id, handle, float(target), sim.simx_opmode_oneshot)
    for _ in range(int(settle_steps)):
        sim.simxSynchronousTrigger(client_id)
        sim.simxGetPingTime(client_id)
        time.sleep(min(float(tau), 0.005))
    theta_measured = read_joint_positions_fast(client_id, joint_handles)
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
        'desired_positions': [],
        'position_errors': [],
        'joint_positions': [],
        'joint_drift': [],
        'task_residuals': [],
        'solver_residuals': [],
        'solver_energies': [],
        'boundary_slacks': [],
        'drift_norms': [],
    }


def record_history(history, tk, theta_current, current_pos, desired_pos, theta_reference, task_residual=None,
                   solver_residual=None, solver_energy=None, boundary_slack=None):
    drift = np.asarray(theta_current, dtype=float) - np.asarray(theta_reference, dtype=float)
    history['time_s'].append(float(tk))
    history['actual_positions'].append(np.asarray(current_pos, dtype=float).tolist())
    history['desired_positions'].append(np.asarray(desired_pos, dtype=float).tolist())
    history['position_errors'].append(float(np.linalg.norm(current_pos - desired_pos)))
    history['joint_positions'].append(np.asarray(theta_current, dtype=float).tolist())
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
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect(tuple(safe_ranges))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.view_init(elev=28, azim=-58)


def plot_trajectory(history, output_path, title_prefix):
    desired = np.asarray(history['desired_positions'], dtype=float)
    actual = np.asarray(history['actual_positions'], dtype=float)
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(desired[:, 0], desired[:, 1], desired[:, 2], color='#c0392b', linewidth=2.2, label='Desired')
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], color='#1f77b4', linestyle='--', linewidth=2.0, label='Actual')
    configure_3d_axes(ax, desired, actual)
    ax.set_title(f'{title_prefix} Trajectory')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def plot_joint_angles(history, output_path, title_prefix):
    joints = np.asarray(history['joint_positions'], dtype=float)
    time_vec = np.asarray(history['time_s'], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i in range(joints.shape[1]):
        ax.plot(time_vec, joints[:, i], linewidth=1.6, label=rf'$\theta_{{{i+1}}}$')
    ax.set_title(f'{title_prefix} Joint Angles')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('rad')
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def plot_position_error(history, output_path, title_prefix):
    errors = np.asarray(history['position_errors'], dtype=float)
    time_vec = np.asarray(history['time_s'], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(time_vec, np.clip(errors, 1e-12, None), color='#ff7f0e', linewidth=2.2)
    ax.set_title(f'{title_prefix} Position Error')
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$||x(q)-x_d||_2$ (m)')
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def plot_joint_drift(history, output_path, title_prefix):
    drift = np.asarray(history['joint_drift'], dtype=float)
    time_vec = np.asarray(history['time_s'], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i in range(drift.shape[1]):
        ax.plot(time_vec, drift[:, i], linewidth=1.6, label=rf'$\Delta\theta_{{{i+1}}}(t)$')
    ax.axhline(0.0, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
    ax.set_title(f'{title_prefix} Joint Drift Error')
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$\theta_i(t)-\theta_i(0)$ (rad)')
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def plot_task_residual(history, output_path, title_prefix):
    residuals = np.asarray(history.get('task_residuals', []), dtype=float)
    if len(residuals) == 0:
        return
    time_vec = np.asarray(history['time_s'], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(time_vec, np.clip(residuals, 1e-12, None), color='#2ca02c', linewidth=2.0)
    ax.set_title(f'{title_prefix} Task Residual')
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$||J\dot{q} - b||_2$')
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def plot_solver_residual(history, output_path, title_prefix):
    residuals = np.asarray(history.get('solver_residuals', []), dtype=float)
    if len(residuals) == 0:
        return
    time_vec = np.asarray(history['time_s'], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(time_vec, np.clip(residuals, 1e-12, None), color='#8e44ad', linewidth=2.0)
    ax.set_title(f'{title_prefix} Solver Residual')
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$||F(y,t)||_2$')
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def save_all_figures(history, output_dir, stem, title_prefix):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory(history, output_dir / f'{stem}_trajectory.png', title_prefix)
    plot_joint_angles(history, output_dir / f'{stem}_joint_angles.png', title_prefix)
    plot_position_error(history, output_dir / f'{stem}_position_error.png', title_prefix)
    plot_joint_drift(history, output_dir / f'{stem}_joint_drift.png', title_prefix)
    plot_task_residual(history, output_dir / f'{stem}_task_residual.png', title_prefix)
    plot_solver_residual(history, output_dir / f'{stem}_solver_residual.png', title_prefix)


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
    drift_norms = np.asarray(history.get('drift_norms', []), dtype=float)
    task_residuals = np.asarray(history.get('task_residuals', []), dtype=float)
    solver_residuals = np.asarray(history.get('solver_residuals', []), dtype=float)
    solver_energies = np.asarray(history.get('solver_energies', []), dtype=float)
    boundary_slacks = np.asarray(history.get('boundary_slacks', []), dtype=float)

    summary = {
        'stem': stem,
        'theta_initial': theta_i.tolist(),
        'theta_final': theta_f.tolist(),
        'joint_drift_table': rows,
        'max_joint_drift_rad': float(np.max(np.abs(delta))),
        'final_joint_drift_norm_rad': float(np.linalg.norm(delta)),
        'mean_position_error_m': float(np.mean(errors)),
        'final_position_error_m': float(errors[-1]),
        'max_position_error_m': float(np.max(errors)),
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
    tau: float = 0.005
    heart_scale: float = 0.008
    theta_dot_limit: float = 2.0
    eta: float = 0.9
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

    def reset(self, theta_initial):
        self.theta_initial = np.asarray(theta_initial, dtype=float).copy()
        self.state = np.zeros(self.n_joints + self.task_dim + self.n_joints, dtype=float)
        self.prev_signals = None

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
        return self.cfg.drift_gain * np.asarray(drift_delta, dtype=float)

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

    def step(self, theta_current, desired_position, desired_velocity, use_feedback=True):
        if self.theta_initial is None:
            self.reset(theta_current)

        theta_current = np.asarray(theta_current, dtype=float)
        current_pos = self.robot.forward_kinematics(theta_current)[:3, 3]
        jacobian_task = self.robot.jacobian(theta_current)[: self.task_dim, :]

        position_error = current_pos - desired_position
        if use_feedback:
            task_command = desired_velocity - self.cfg.task_gain * position_error
        else:
            task_command = desired_velocity.copy()

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

        residual = residual_pack['residual']
        u = residual_pack['u']
        lam = residual_pack['lambda']
        omega = residual_pack['omega']
        sigma = residual_pack['sigma']
        alpha = residual_pack['alpha']
        beta = residual_pack['beta']
        ds_du = residual_pack['ds_du']
        use_lower_branch = residual_pack['use_lower_branch']

        stationarity = residual_pack['stationarity']
        equality = residual_pack['equality']
        pfb_residual = residual_pack['pfb_residual']

        r1_t = derivatives['drift_feedback_dot'] - derivatives['jacobian_dot'].T @ lam
        r2_t = derivatives['jacobian_dot'] @ u - derivatives['task_command_dot']
        s_partial_dot = np.where(use_lower_branch, -derivatives['lower_dot'], derivatives['upper_dot'])
        r3_t = alpha * s_partial_dot
        residual_time_part = np.concatenate((r1_t, r2_t, r3_t))

        residual_jacobian = self._build_residual_jacobian(residual_pack, jacobian_task)
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
        y_next = y_state + self.cfg.tau * y_dot

        u_next_raw = y_next[: self.n_joints].copy()
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
            'solver_residual_norm': float(np.linalg.norm(residual)),
            'solver_energy': solver_energy,
            'scalar_drive': scalar_drive,
            'boundary_slack': residual_pack['min_boundary_slack'],
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
        error_mode='scalar',
    )
    return ContinuousTVQPELNCPController(robot, cfg, 'method2')


METHOD_BUILDERS = {
    'method1': build_method1,
    'method2': build_method2,
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

    try:
        joint_handles = get_joint_handles(client_id)
        begin_joint_position_stream(client_id, joint_handles)

        theta_lower, theta_upper = query_joint_limits_from_sim(client_id, joint_handles)
        controller.update_joint_limits(theta_lower, theta_upper)

        _, theta_reference = startup_handshake_and_settle(
            client_id,
            joint_handles,
            settings.theta_initial_command,
            settle_steps=20,
            tau=settings.tau,
        )
        controller.reset(theta_reference)

        initial_pos = robot.forward_kinematics(theta_reference)[:3, 3]
        trajectory_offset = make_offset_from_initial_position(initial_pos, settings.heart_scale)
        trajectory = HeartTrajectory(
            duration=settings.duration,
            scale=settings.heart_scale,
            offset=trajectory_offset,
        )

        total_steps = int(round(settings.duration / settings.tau))
        t_start = time.perf_counter()

        for step in range(total_steps):
            tk = step * settings.tau
            theta_current = read_joint_positions_fast(client_id, joint_handles)
            desired_pos, desired_vel = trajectory.get_pose(tk)
            result = controller.step(theta_current, desired_pos, desired_vel, use_feedback=use_feedback)

            current_pos = result['current_pos']
            record_history(
                history,
                tk,
                theta_current,
                current_pos,
                desired_pos,
                theta_reference,
                task_residual=result.get('task_residual'),
                solver_residual=result.get('solver_residual_norm'),
                solver_energy=result.get('solver_energy'),
                boundary_slack=result.get('boundary_slack'),
            )

            send_joint_targets(client_id, joint_handles, result['theta_next'])
            sim.simxSynchronousTrigger(client_id)
            sim.simxGetPingTime(client_id)

        runtime_s = time.perf_counter() - t_start
        print(f'    Runtime: {runtime_s:.1f}s, steps: {total_steps}')

    finally:
        sim.simxStopSimulation(client_id, sim.simx_opmode_oneshot)
        sim.simxFinish(client_id)

    save_all_figures(history, output_dir, stem, title_prefix)
    summary = save_summary(output_dir, stem, theta_reference, theta_current, history)
    summary['runtime_s'] = runtime_s
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


def main():
    parser = argparse.ArgumentParser(description='Continuous TVQP + ELNCP + DLCCZNN experiments')
    parser.add_argument('--method', default='all', help='Method to run: method1, method2, or all')
    parser.add_argument('--no-feedback', action='store_true', help='Remove position feedback from task equality')
    parser.add_argument('--duration', type=float, default=10.0, help='Live experiment duration (s)')
    parser.add_argument('--offline-duration', type=float, default=20.0, help='Offline experiment duration (s)')
    parser.add_argument('--tau', type=float, default=0.005, help='Control time step (s)')
    parser.add_argument('--output-root', default=None, help='Output root directory')
    parser.add_argument('--task-gain', type=float, default=None, help='Override task feedback gain')
    parser.add_argument('--drift-gain', type=float, default=None, help='Override drift gain')
    parser.add_argument('--solver-gamma', type=float, default=None, help='Override DLCCZNN solver gain')
    parser.add_argument('--activation-power', type=float, default=None, help='Override solver activation power')
    parser.add_argument('--activation-exp-clip', type=float, default=None, help='Override solver activation exponential clip')
    parser.add_argument('--solver-regularization', type=float, default=None, help='Override solver regularization')
    parser.add_argument('--theta-dot-limit', type=float, default=None, help='Override joint velocity limit')
    parser.add_argument('--eta', type=float, default=None, help='Override dynamic bound safety coefficient')
    args = parser.parse_args()

    use_feedback = not args.no_feedback
    fb_label = 'with_feedback' if use_feedback else 'without_feedback'
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = build_run_output_root(CURRENT_DIR / f'results_{fb_label}', args, "live")

    output_root.mkdir(parents=True, exist_ok=True)
    save_run_config(output_root, args, "live")

    print(f'=== Continuous TVQP + ELNCP + DLCCZNN: {fb_label} ===')
    print(f'Output: {output_root}')
    print()

    robot = UR3eKinematics()
    settings = ExperimentSettings(
        duration=args.duration,
        offline_duration=args.offline_duration,
        tau=args.tau,
    )
    if args.theta_dot_limit is not None:
        settings.theta_dot_limit = float(args.theta_dot_limit)
    if args.eta is not None:
        settings.eta = float(args.eta)

    if args.method == 'all':
        methods_to_run = ['method1', 'method2']
    else:
        methods_to_run = [args.method]

    all_summaries = {}

    for method_name in methods_to_run:
        method_output_dir = output_root / method_name
        builder = METHOD_BUILDERS[method_name]

        print(f'--- {METHOD_DISPLAY[method_name]}: Live ---')
        _, summary = run_live(
            method_name,
            builder,
            robot,
            settings,
            method_output_dir / 'live',
            use_feedback=use_feedback,
            args=args,
        )
        if summary:
            all_summaries[f'{method_name}_live'] = summary
            print(f'  Mean pos err: {summary["mean_position_error_m"]:.3e} m')
            print(f'  Final drift norm: {summary["final_joint_drift_norm_rad"]:.3e} rad')
            print(f'  Final solver residual: {summary.get("final_solver_residual", 0.0):.3e}')

        print()

    if all_summaries:
        summary_path = output_root / 'all_summaries.json'
        summary_path.write_text(
            json.dumps(all_summaries, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        write_comparison_tables(output_root, all_summaries)
        print(f'All summaries saved to {summary_path}')

        print('\n=== Summary Table ===')
        print(f'{"Experiment":<28} {"MeanPosErr(m)":>15} {"FinalDrift(rad)":>16} {"FinalSolverRes":>16}')
        print('-' * 80)
        for key, summary in sorted(all_summaries.items()):
            print(
                f'{key:<28} '
                f'{summary["mean_position_error_m"]:>15.3e} '
                f'{summary["final_joint_drift_norm_rad"]:>16.3e} '
                f'{summary.get("final_solver_residual", 0.0):>16.3e}'
            )

    print('\nDone.')


if __name__ == '__main__':
    main()

