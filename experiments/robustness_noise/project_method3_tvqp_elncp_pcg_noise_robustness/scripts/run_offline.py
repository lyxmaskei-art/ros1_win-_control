# Complete offline experiment entry for direct VSCode execution.
# This file is intentionally self-contained for direct VSCode execution.

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

RUN_TAG_KEYS = (
    "duration",
    "tau",
    "trials",
    "seed",
    "channels",
    "profiles",
    "live_trials",
    "task_gain",
    "position_weight",
    "drift_weight",
    "pcg_max_iters",
    "pcg_tol",
    "normal_reg",
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


def find_workspace_code_root(start_dir):
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "core").is_dir() and (candidate / "sim").is_dir():
            return candidate
        nested = candidate / "code"
        if (nested / "core").is_dir() and (nested / "sim").is_dir():
            return nested
    raise RuntimeError(f"Cannot locate workspace code root from {start_dir}")

CODE_ROOT = find_workspace_code_root(SCRIPT_DIR)
for candidate in (CODE_ROOT / "sim", CODE_ROOT / "core", CODE_ROOT / "methods"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np


METHOD_NAME = "Method 3: continuous TVQP + ELNCP + PCG"
OUTPUT_STEM = "method3_tvqp_elncp_pcg"
DEFAULT_REMOTE_API_PORTS = (19997, 19998)
DEFAULT_THETA_LIMIT = 2.0 * np.pi * np.ones(6, dtype=float)


def compute_dynamic_velocity_bounds(theta_current, theta_lower, theta_upper, theta_dot_limit, eta, tau):
    theta_current = np.asarray(theta_current, dtype=float)
    theta_dot_lower = -float(theta_dot_limit) * np.ones_like(theta_current)
    theta_dot_upper = float(theta_dot_limit) * np.ones_like(theta_current)
    eta_rate = float(eta) / float(tau)
    xi_minus = np.maximum(theta_dot_lower, eta_rate * (theta_lower - theta_current))
    xi_plus = np.minimum(theta_dot_upper, eta_rate * (theta_upper - theta_current))
    return xi_minus, xi_plus


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


def tuned_method3_task_gain(tau):
    tau = float(tau)
    if tau <= 0.002:
        return 160.0
    if tau <= 0.005:
        return 198.0
    if tau <= 0.01:
        return 120.0
    if tau <= 0.02:
        return 80.0
    return 20.0


@dataclass
class Method3TVQPELNCPPCGConfig:
    duration: float = 10.0
    tau: float = 0.005
    heart_scale: float = 0.008
    task_gain: float | None = None
    position_weight: float = 15000.0
    drift_weight: float = 5e-4
    regularization_gain: float = 1e-10
    eta: float = 0.9
    theta_dot_limit: float = 2.0
    offline_duration: float = 20.0
    ncp_epsilon: float = 1e-10
    pcg_max_iters: int = 8
    pcg_tol: float = 1e-10
    normal_reg: float = 1e-10
    max_active_set_iters: int = 16
    active_set_tol: float = 1e-9
    use_jacobi_preconditioner: bool = True
    theta_initial_command: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0], dtype=float)
    )
    theta_lower: np.ndarray = field(default_factory=lambda: -DEFAULT_THETA_LIMIT.copy())
    theta_upper: np.ndarray = field(default_factory=lambda: DEFAULT_THETA_LIMIT.copy())

    def __post_init__(self):
        if self.task_gain is None:
            self.task_gain = tuned_method3_task_gain(self.tau)

    @property
    def steps(self):
        return int(round(self.duration / self.tau))


class Method3TVQPELNCPPCGController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_joints = 6
        self.identity = np.eye(self.n_joints, dtype=float)
        self.theta_initial = None
        self.u_state = np.zeros(self.n_joints, dtype=float)
        self.lower_multiplier = np.zeros(self.n_joints, dtype=float)
        self.upper_multiplier = np.zeros(self.n_joints, dtype=float)
        self.prev_signals = None

    def reset(self, theta_initial):
        self.theta_initial = np.asarray(theta_initial, dtype=float).copy()
        self.u_state = np.zeros(self.n_joints, dtype=float)
        self.lower_multiplier = np.zeros(self.n_joints, dtype=float)
        self.upper_multiplier = np.zeros(self.n_joints, dtype=float)
        self.prev_signals = None

    def update_joint_limits(self, theta_lower, theta_upper):
        self.cfg.theta_lower = np.asarray(theta_lower, dtype=float).copy()
        self.cfg.theta_upper = np.asarray(theta_upper, dtype=float).copy()

    def _build_objective(self, theta_current, current_pos, desired_pos, desired_vel, jacobian_task):
        position_error = np.asarray(current_pos, dtype=float) - np.asarray(desired_pos, dtype=float)
        drift_delta = np.asarray(theta_current, dtype=float) - self.theta_initial
        position_drive = ((1.0 / self.cfg.tau) + self.cfg.task_gain) * position_error - desired_vel
        q_matrix = (
            self.cfg.position_weight * (jacobian_task.T @ jacobian_task)
            + self.cfg.drift_weight * self.identity
            + self.cfg.regularization_gain * self.identity
        )
        q_vector = (
            self.cfg.position_weight * (jacobian_task.T @ position_drive)
            + self.cfg.drift_weight * (drift_delta / self.cfg.tau)
        )
        return q_matrix, q_vector, position_error, drift_delta, position_drive

    def _project_to_bounds(self, values, lower, upper):
        return np.minimum(np.maximum(np.asarray(values, dtype=float), lower), upper)

    def _pcg_solve_reduced(self, matrix, rhs, x0):
        size = int(rhs.shape[0])
        if size == 0:
            return np.zeros(0, dtype=float), 0, 0.0
        matrix = np.asarray(matrix, dtype=float)
        matrix = 0.5 * (matrix + matrix.T) + self.cfg.normal_reg * np.eye(size, dtype=float)
        rhs = np.asarray(rhs, dtype=float).reshape(-1)
        x = np.asarray(x0, dtype=float).reshape(-1).copy()

        residual = rhs - matrix @ x
        residual_norm = float(np.linalg.norm(residual))
        iterations = 0
        if residual_norm > self.cfg.pcg_tol:
            if self.cfg.use_jacobi_preconditioner:
                diag = np.maximum(np.diag(matrix), 1e-14)
                z = residual / diag
            else:
                z = residual.copy()
            direction = z.copy()
            rz_old = float(residual @ z)
            for iterations in range(1, max(int(self.cfg.pcg_max_iters), 1) + 1):
                mat_dir = matrix @ direction
                denom = float(direction @ mat_dir)
                if denom <= 1e-24:
                    break
                alpha = rz_old / denom
                x = x + alpha * direction
                residual = residual - alpha * mat_dir
                residual_norm = float(np.linalg.norm(residual))
                if residual_norm <= self.cfg.pcg_tol:
                    break
                if self.cfg.use_jacobi_preconditioner:
                    z = residual / diag
                else:
                    z = residual.copy()
                rz_new = float(residual @ z)
                beta = rz_new / max(rz_old, 1e-30)
                direction = z + beta * direction
                rz_old = rz_new

        return x, int(iterations), float(np.linalg.norm(matrix @ x - rhs))

    def _build_residual(self, u, lower_multiplier, upper_multiplier, q_matrix, q_vector, lower, upper):
        n = self.n_joints
        u = np.asarray(u, dtype=float).reshape(n)
        lower_multiplier = np.maximum(np.asarray(lower_multiplier, dtype=float).reshape(n), 0.0)
        upper_multiplier = np.maximum(np.asarray(upper_multiplier, dtype=float).reshape(n), 0.0)
        lower_slack = u - lower
        upper_slack = upper - u

        lower_sqrt = np.sqrt(lower_slack * lower_slack + lower_multiplier * lower_multiplier + self.cfg.ncp_epsilon)
        upper_sqrt = np.sqrt(upper_slack * upper_slack + upper_multiplier * upper_multiplier + self.cfg.ncp_epsilon)
        lower_pfb = lower_slack + lower_multiplier - lower_sqrt
        upper_pfb = upper_slack + upper_multiplier - upper_sqrt

        stationarity = q_matrix @ u + q_vector - lower_multiplier + upper_multiplier
        residual = np.concatenate((stationarity, lower_pfb, upper_pfb))

        return {
            "residual": residual,
            "stationarity": stationarity,
            "pfb_residual": np.concatenate((lower_pfb, upper_pfb)),
            "u": u,
            "lower_multiplier": lower_multiplier,
            "upper_multiplier": upper_multiplier,
            "min_boundary_slack": float(np.min(np.minimum(lower_slack, upper_slack))),
        }

    def _solve_box_qp_by_elncp_active_pcg(self, q_matrix, q_vector, lower, upper):
        n = self.n_joints
        lower = np.asarray(lower, dtype=float).reshape(n)
        upper = np.asarray(upper, dtype=float).reshape(n)
        u = self._project_to_bounds(self.u_state, lower, upper)
        total_pcg_iters = 0
        last_linear_residual = 0.0
        active_iters = 0

        for active_iters in range(1, max(int(self.cfg.max_active_set_iters), 1) + 1):
            gradient = q_matrix @ u + q_vector
            at_lower = u <= lower + self.cfg.active_set_tol
            at_upper = u >= upper - self.cfg.active_set_tol
            active_lower = at_lower & (gradient >= -self.cfg.active_set_tol)
            active_upper = at_upper & (gradient <= self.cfg.active_set_tol)
            free = ~(active_lower | active_upper)

            u_next = u.copy()
            if np.any(free):
                free_idx = np.flatnonzero(free)
                active_idx = np.flatnonzero(~free)
                q_ff = q_matrix[np.ix_(free_idx, free_idx)]
                rhs = -q_vector[free_idx]
                if active_idx.size > 0:
                    rhs = rhs - q_matrix[np.ix_(free_idx, active_idx)] @ u[active_idx]
                x0 = u[free_idx]
                solved, pcg_iters, linear_residual = self._pcg_solve_reduced(q_ff, rhs, x0)
                total_pcg_iters += pcg_iters
                last_linear_residual = linear_residual
                u_next[free_idx] = solved

            below = u_next < lower
            above = u_next > upper
            if np.any(below):
                u_next[below] = lower[below]
            if np.any(above):
                u_next[above] = upper[above]

            gradient_next = q_matrix @ u_next + q_vector
            at_lower_next = u_next <= lower + self.cfg.active_set_tol
            at_upper_next = u_next >= upper - self.cfg.active_set_tol
            free_next = ~(at_lower_next | at_upper_next)
            lower_ok = np.all(gradient_next[at_lower_next] >= -1e-7)
            upper_ok = np.all(gradient_next[at_upper_next] <= 1e-7)
            free_ok = np.all(np.abs(gradient_next[free_next]) <= max(1e-7, 10.0 * self.cfg.pcg_tol))
            u = u_next
            if lower_ok and upper_ok and free_ok:
                break

        gradient = q_matrix @ u + q_vector
        lower_multiplier = np.zeros(n, dtype=float)
        upper_multiplier = np.zeros(n, dtype=float)
        at_lower = u <= lower + self.cfg.active_set_tol
        at_upper = u >= upper - self.cfg.active_set_tol
        lower_multiplier[at_lower] = np.maximum(gradient[at_lower], 0.0)
        upper_multiplier[at_upper] = np.maximum(-gradient[at_upper], 0.0)

        residual_pack = self._build_residual(u, lower_multiplier, upper_multiplier, q_matrix, q_vector, lower, upper)
        return {
            "u": u,
            "lower_multiplier": lower_multiplier,
            "upper_multiplier": upper_multiplier,
            "residual_pack": residual_pack,
            "pcg_iterations": total_pcg_iters,
            "active_iterations": active_iters,
            "linear_residual_norm": last_linear_residual,
        }

    def step(self, theta_current, current_pos, desired_pos, desired_vel, jacobian_task, lower, upper):
        if self.theta_initial is None:
            self.reset(theta_current)

        q_matrix, q_vector, position_error, drift_delta, _ = self._build_objective(
            theta_current=theta_current,
            current_pos=current_pos,
            desired_pos=desired_pos,
            desired_vel=desired_vel,
            jacobian_task=jacobian_task,
        )

        solver_output = self._solve_box_qp_by_elncp_active_pcg(q_matrix, q_vector, lower, upper)
        u_next = solver_output["u"]
        lower_multiplier = solver_output["lower_multiplier"]
        upper_multiplier = solver_output["upper_multiplier"]
        final_pack = solver_output["residual_pack"]

        if not np.all(np.isfinite(u_next)):
            raise FloatingPointError("Method3 TVQP ELNCP PCG velocity solve diverged.")

        self.u_state = u_next.copy()
        self.lower_multiplier = lower_multiplier.copy()
        self.upper_multiplier = upper_multiplier.copy()
        self.prev_signals = {
            "q_matrix": q_matrix.copy(),
            "q_vector": q_vector.copy(),
            "lower": np.asarray(lower, dtype=float).copy(),
            "upper": np.asarray(upper, dtype=float).copy(),
        }

        theta_dot = u_next.copy()
        return {
            "theta_dot": theta_dot,
            "theta_dot_raw": u_next,
            "lower_multiplier": lower_multiplier,
            "upper_multiplier": upper_multiplier,
            "position_error": position_error,
            "task_residual": desired_vel - jacobian_task @ theta_dot,
            "drift_delta": drift_delta,
            "q_matrix": q_matrix,
            "q_vector": q_vector,
            "inner_residual_norm": float(np.linalg.norm(final_pack["residual"])),
            "stationarity_norm": float(np.linalg.norm(final_pack["stationarity"])),
            "ncp_residual_norm": float(np.linalg.norm(final_pack["pfb_residual"])),
            "inner_energy": 0.5 * float(final_pack["residual"] @ final_pack["residual"]),
            "inner_iterations": solver_output["pcg_iterations"],
            "active_iterations": solver_output["active_iterations"],
            "linear_residual_norm": solver_output["linear_residual_norm"],
            "boundary_slack": final_pack["min_boundary_slack"],
        }


def create_history():
    return {
        "time_s": [],
        "actual_positions": [],
        "desired_positions": [],
        "position_errors": [],
        "joint_positions": [],
        "joint_drift": [],
        "task_residuals": [],
        "inner_residual_norm": [],
        "stationarity_norm": [],
        "ncp_residual_norm": [],
        "linear_residual_norm": [],
        "inner_iterations": [],
        "active_iterations": [],
        "boundary_slacks": [],
    }


def record_history(history, tk, theta_current, current_pos, desired_pos, theta_reference, step_result):
    theta_current = np.asarray(theta_current, dtype=float)
    current_pos = np.asarray(current_pos, dtype=float)
    desired_pos = np.asarray(desired_pos, dtype=float)
    theta_reference = np.asarray(theta_reference, dtype=float)
    drift = theta_current - theta_reference

    history["time_s"].append(float(tk))
    history["actual_positions"].append(current_pos.tolist())
    history["desired_positions"].append(desired_pos.tolist())
    history["position_errors"].append(float(np.linalg.norm(current_pos - desired_pos)))
    history["joint_positions"].append(theta_current.tolist())
    history["joint_drift"].append(drift.tolist())
    history["task_residuals"].append(float(np.linalg.norm(step_result["task_residual"])))
    history["inner_residual_norm"].append(float(step_result["inner_residual_norm"]))
    history["stationarity_norm"].append(float(step_result["stationarity_norm"]))
    history["ncp_residual_norm"].append(float(step_result["ncp_residual_norm"]))
    history["linear_residual_norm"].append(float(step_result["linear_residual_norm"]))
    history["inner_iterations"].append(int(step_result["inner_iterations"]))
    history["active_iterations"].append(int(step_result["active_iterations"]))
    history["boundary_slacks"].append(float(step_result["boundary_slack"]))


def configure_trajectory_axes(ax, desired, actual):
    combined = np.vstack((desired, actual))
    ranges = np.ptp(combined, axis=0)
    safe_ranges = np.maximum(ranges, 1e-6)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(tuple(safe_ranges))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.view_init(elev=28, azim=-58)


def plot_trajectory_figure(history, output_path, title_prefix):
    desired = np.asarray(history["desired_positions"], dtype=float)
    actual = np.asarray(history["actual_positions"], dtype=float)
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(desired[:, 0], desired[:, 1], desired[:, 2], color="#c0392b", linewidth=2.2, label="Desired")
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], color="#1f77b4", linestyle="--", linewidth=2.0, label="Actual")
    configure_trajectory_axes(ax, desired, actual)
    ax.set_title(f"{title_prefix} Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_joint_angles_figure(history, output_path, title_prefix):
    joints = np.asarray(history["joint_positions"], dtype=float)
    time_vec = np.asarray(history["time_s"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i in range(joints.shape[1]):
        ax.plot(time_vec, joints[:, i], linewidth=1.6, label=rf"$\theta_{{{i + 1}}}$")
    ax.set_title(f"{title_prefix} Joint Angles")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("rad")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_position_error_figure(history, output_path, title_prefix):
    errors = np.asarray(history["position_errors"], dtype=float)
    time_vec = np.asarray(history["time_s"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(time_vec, np.clip(errors, 1e-12, None), color="#ff7f0e", linewidth=2.2)
    ax.set_title(f"{title_prefix} Position Error")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"$||x(q)-x_d||_2$ (m)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_joint_drift_figure(history, output_path, title_prefix):
    drift = np.asarray(history["joint_drift"], dtype=float)
    time_vec = np.asarray(history["time_s"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i in range(drift.shape[1]):
        ax.plot(time_vec, drift[:, i], linewidth=1.6, label=rf"$\Delta \theta_{{{i + 1}}}(t)$")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_title(f"{title_prefix} Joint Drift Error")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"$\theta_i(t)-\theta_i(0)$ (rad)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_scalar_history(history, key, output_path, title, ylabel, log_scale=True):
    values = np.asarray(history[key], dtype=float)
    time_vec = np.asarray(history["time_s"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if log_scale:
        ax.semilogy(time_vec, np.clip(np.abs(values), 1e-14, None), linewidth=2.0)
    else:
        ax.plot(time_vec, values, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("t (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_joint_recovery_table(theta_initial, theta_final, output_dir, table_label, title_prefix):
    theta_initial = np.asarray(theta_initial, dtype=float)
    theta_final = np.asarray(theta_final, dtype=float)
    delta = theta_final - theta_initial
    csv_path = output_dir / f"{OUTPUT_STEM}_{table_label}_joint_table.csv"

    rows = []
    for i in range(theta_initial.shape[0]):
        rows.append(
            [
                f"theta_{i + 1}",
                f"{theta_initial[i]:.9f}",
                f"{theta_final[i]:.9f}",
                f"{delta[i]:.9e}",
            ]
        )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Joint", "Initial angle (rad)", "Final angle (rad)", "Delta (rad)"])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Joint", "Initial angle (rad)", "Final angle (rad)", "Delta (rad)"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)
    ax.set_title(f"{title_prefix} Joint Recovery Table ({table_label})")
    fig.tight_layout()
    fig.savefig(output_dir / f"{OUTPUT_STEM}_{table_label}_joint_table.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_figures(history, output_dir, title_prefix, theta_initial, theta_final, table_label):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory_figure(history, output_dir / f"{OUTPUT_STEM}_{table_label}_trajectory.png", title_prefix)
    plot_joint_angles_figure(history, output_dir / f"{OUTPUT_STEM}_{table_label}_joint_angles.png", title_prefix)
    plot_position_error_figure(history, output_dir / f"{OUTPUT_STEM}_{table_label}_position_error.png", title_prefix)
    plot_joint_drift_figure(history, output_dir / f"{OUTPUT_STEM}_{table_label}_joint_drift.png", title_prefix)
    plot_scalar_history(
        history,
        "inner_residual_norm",
        output_dir / f"{OUTPUT_STEM}_{table_label}_elncp_residual.png",
        f"{title_prefix} ELNCP Residual",
        r"$||F(y,t)||_2$",
    )
    plot_scalar_history(
        history,
        "inner_iterations",
        output_dir / f"{OUTPUT_STEM}_{table_label}_pcg_iterations.png",
        f"{title_prefix} PCG Iterations",
        "iterations",
        log_scale=False,
    )
    plot_scalar_history(
        history,
        "boundary_slacks",
        output_dir / f"{OUTPUT_STEM}_{table_label}_boundary_slack.png",
        f"{title_prefix} ELNCP Boundary Slack",
        r"$\min_i s_i$",
        log_scale=False,
    )
    save_joint_recovery_table(theta_initial, theta_final, output_dir, table_label, title_prefix)


def summarize_run(label, history, theta_initial, theta_final, result_dir, cfg, port=None):
    position_errors = np.asarray(history["position_errors"], dtype=float)
    drift = np.asarray(history["joint_drift"], dtype=float)
    summary = {
        "label": label,
        "status": "ok",
        "method": METHOD_NAME,
        "tau": float(cfg.tau),
        "task_gain": float(cfg.task_gain),
        "position_weight": float(cfg.position_weight),
        "drift_weight": float(cfg.drift_weight),
        "regularization_gain": float(cfg.regularization_gain),
        "ncp_epsilon": float(cfg.ncp_epsilon),
        "pcg_max_iters": int(cfg.pcg_max_iters),
        "mean_position_error_m": float(np.mean(position_errors)),
        "max_position_error_m": float(np.max(position_errors)),
        "final_position_error_m": float(position_errors[-1]),
        "max_abs_joint_drift_rad": float(np.max(np.abs(drift))),
        "final_joint_drift_norm_rad": float(np.linalg.norm(drift[-1])),
        "mean_elncp_residual_norm": float(np.mean(np.asarray(history["inner_residual_norm"], dtype=float))),
        "max_elncp_residual_norm": float(np.max(np.asarray(history["inner_residual_norm"], dtype=float))),
        "mean_stationarity_norm": float(np.mean(np.asarray(history["stationarity_norm"], dtype=float))),
        "mean_ncp_residual_norm": float(np.mean(np.asarray(history["ncp_residual_norm"], dtype=float))),
        "mean_pcg_iterations": float(np.mean(np.asarray(history["inner_iterations"], dtype=float))),
        "max_pcg_iterations": int(np.max(np.asarray(history["inner_iterations"], dtype=float))),
        "mean_active_iterations": float(np.mean(np.asarray(history["active_iterations"], dtype=float))),
        "max_active_iterations": int(np.max(np.asarray(history["active_iterations"], dtype=float))),
        "min_boundary_slack": float(np.min(np.asarray(history["boundary_slacks"], dtype=float))),
        "theta_initial_rad": np.asarray(theta_initial, dtype=float).tolist(),
        "theta_final_rad": np.asarray(theta_final, dtype=float).tolist(),
        "theta_delta_rad": (np.asarray(theta_final, dtype=float) - np.asarray(theta_initial, dtype=float)).tolist(),
        "num_samples": len(history["time_s"]),
        "result_dir": str(result_dir),
    }
    if port is not None:
        summary["coppeliasim_port"] = int(port)

    (result_dir / f"{OUTPUT_STEM}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (result_dir / f"{OUTPUT_STEM}_summary.txt").write_text("\n".join([f"{k}: {v}" for k, v in summary.items()]) + "\n", encoding="utf-8")
    return summary


def run_offline_experiment(cfg, robot, output_dir, save_artifacts=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    controller = Method3TVQPELNCPPCGController(cfg)
    history = create_history()
    theta_reference = cfg.theta_initial_command.copy()
    theta_current = theta_reference.copy()
    controller.reset(theta_reference)

    initial_pos = robot.forward_kinematics(theta_reference)[:3, 3]
    trajectory_offset = make_offset_from_initial_position(initial_pos, cfg.heart_scale)
    trajectory = HeartTrajectory(duration=cfg.offline_duration, scale=cfg.heart_scale, offset=trajectory_offset)

    steps = int(round(cfg.offline_duration / cfg.tau))
    t_start = time.perf_counter()
    for step in range(steps):
        tk = step * cfg.tau
        desired_pos, desired_vel = trajectory.get_pose(tk)
        current_pos = robot.forward_kinematics(theta_current)[:3, 3]
        jacobian_task = robot.jacobian(theta_current)[:3, :]
        lower, upper = compute_dynamic_velocity_bounds(
            theta_current=theta_current,
            theta_lower=cfg.theta_lower,
            theta_upper=cfg.theta_upper,
            theta_dot_limit=cfg.theta_dot_limit,
            eta=cfg.eta,
            tau=cfg.tau,
        )
        step_result = controller.step(theta_current, current_pos, desired_pos, desired_vel, jacobian_task, lower, upper)
        theta_current = np.clip(theta_current + cfg.tau * step_result["theta_dot"], cfg.theta_lower, cfg.theta_upper)
        new_pos = robot.forward_kinematics(theta_current)[:3, 3]
        record_history(history, tk, theta_current, new_pos, desired_pos, theta_reference, step_result)
    wall_clock = time.perf_counter() - t_start

    if save_artifacts:
        save_figures(history, output_dir, METHOD_NAME + " Offline", theta_reference, theta_current, "offline")
    summary = summarize_run("Method 3 TVQP ELNCP PCG offline", history, theta_reference, theta_current, output_dir, cfg)
    summary["wall_clock_run_s"] = wall_clock
    (output_dir / f"{OUTPUT_STEM}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def noise_envelope(profile, t, duration):
    """Normalized counterparts of n1, n2, n3 in the reference paper."""
    progress = 0.0 if duration <= 0.0 else float(np.clip(t / duration, 0.0, 1.0))
    if profile == "constant":
        return 1.0
    if profile == "linear":
        return progress
    if profile == "quadratic":
        return 4.0 * (progress - 0.5) ** 2
    raise ValueError(f"Unknown noise profile: {profile}")


def unit_noise_direction(rng, size=6):
    direction = rng.standard_normal(size)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.ones(size, dtype=float)
        norm = float(np.linalg.norm(direction))
    return direction / norm


def make_cfg(duration, tau, args):
    return Method3TVQPELNCPPCGConfig(
        duration=duration,
        offline_duration=duration,
        tau=tau,
        task_gain=args.task_gain,
        position_weight=args.position_weight,
        drift_weight=args.drift_weight,
        regularization_gain=args.regularization_gain,
        ncp_epsilon=args.ncp_epsilon,
        pcg_max_iters=args.pcg_max_iters,
        pcg_tol=args.pcg_tol,
        normal_reg=args.normal_reg,
        max_active_set_iters=args.max_active_set_iters,
        active_set_tol=args.active_set_tol,
        use_jacobi_preconditioner=not args.no_jacobi_preconditioner,
    )


def create_noise_history():
    history = create_history()
    history["noise_norm"] = []
    history["noise_envelope"] = []
    return history


def record_noise_history(history, tk, theta_current, current_pos, desired_pos, theta_reference, step_result, noise_vec, envelope):
    record_history(history, tk, theta_current, current_pos, desired_pos, theta_reference, step_result)
    history["noise_norm"].append(float(np.linalg.norm(noise_vec)))
    history["noise_envelope"].append(float(envelope))


def summarize_noise_run(
    label,
    history,
    theta_initial,
    theta_final,
    result_dir,
    cfg,
    noise_channel,
    noise_profile,
    noise_level,
    trial,
    seed,
    wall_clock,
):
    position_errors = np.asarray(history["position_errors"], dtype=float)
    drift = np.asarray(history["joint_drift"], dtype=float)
    inner_residual = np.asarray(history["inner_residual_norm"], dtype=float)
    stationarity = np.asarray(history["stationarity_norm"], dtype=float)
    ncp_residual = np.asarray(history["ncp_residual_norm"], dtype=float)
    boundary_slacks = np.asarray(history["boundary_slacks"], dtype=float)
    pcg_iters = np.asarray(history["inner_iterations"], dtype=float)
    active_iters = np.asarray(history["active_iterations"], dtype=float)
    noise_norm = np.asarray(history["noise_norm"], dtype=float)

    summary = {
        "label": label,
        "status": "ok",
        "method": METHOD_NAME,
        "noise_channel": noise_channel,
        "noise_profile": noise_profile,
        "noise_level": float(noise_level),
        "trial": int(trial),
        "seed": int(seed),
        "tau": float(cfg.tau),
        "duration_s": float(cfg.offline_duration),
        "task_gain": float(cfg.task_gain),
        "position_weight": float(cfg.position_weight),
        "drift_weight": float(cfg.drift_weight),
        "pcg_max_iters": int(cfg.pcg_max_iters),
        "mean_position_error_m": float(np.mean(position_errors)),
        "max_position_error_m": float(np.max(position_errors)),
        "final_position_error_m": float(position_errors[-1]),
        "mean_joint_drift_norm_rad": float(np.mean(np.linalg.norm(drift, axis=1))),
        "max_abs_joint_drift_rad": float(np.max(np.abs(drift))),
        "final_joint_drift_norm_rad": float(np.linalg.norm(drift[-1])),
        "mean_elncp_residual_norm": float(np.mean(inner_residual)),
        "max_elncp_residual_norm": float(np.max(inner_residual)),
        "mean_stationarity_norm": float(np.mean(stationarity)),
        "mean_ncp_residual_norm": float(np.mean(ncp_residual)),
        "mean_pcg_iterations": float(np.mean(pcg_iters)),
        "max_pcg_iterations": int(np.max(pcg_iters)),
        "mean_active_iterations": float(np.mean(active_iters)),
        "max_active_iterations": int(np.max(active_iters)),
        "min_boundary_slack": float(np.min(boundary_slacks)),
        "mean_noise_norm": float(np.mean(noise_norm)),
        "max_noise_norm": float(np.max(noise_norm)),
        "theta_initial_rad": np.asarray(theta_initial, dtype=float).tolist(),
        "theta_final_rad": np.asarray(theta_final, dtype=float).tolist(),
        "theta_delta_rad": (np.asarray(theta_final, dtype=float) - np.asarray(theta_initial, dtype=float)).tolist(),
        "num_samples": len(history["time_s"]),
        "wall_clock_run_s": float(wall_clock),
        "result_dir": str(result_dir),
    }
    return summary


def run_offline_noise_trial(cfg, robot, noise_channel, noise_profile, noise_level, trial, seed, output_dir, save_artifacts):
    rng = np.random.RandomState(seed)
    sensor_direction = unit_noise_direction(rng, 6)
    actuator_direction = unit_noise_direction(rng, 6)
    controller = Method3TVQPELNCPPCGController(cfg)
    history = create_noise_history()

    theta_reference = cfg.theta_initial_command.copy()
    theta_true = theta_reference.copy()
    controller.reset(theta_reference)

    initial_pos = robot.forward_kinematics(theta_reference)[:3, 3]
    trajectory_offset = make_offset_from_initial_position(initial_pos, cfg.heart_scale)
    trajectory = HeartTrajectory(duration=cfg.offline_duration, scale=cfg.heart_scale, offset=trajectory_offset)

    steps = int(round(cfg.offline_duration / cfg.tau))
    t_start = time.perf_counter()
    for step in range(steps):
        tk = step * cfg.tau
        envelope = noise_envelope(noise_profile, tk, cfg.offline_duration)

        sensor_noise = np.zeros(6, dtype=float)
        actuator_noise = np.zeros(6, dtype=float)
        if noise_channel in ("sensor", "both"):
            sensor_noise = float(noise_level) * envelope * sensor_direction
        if noise_channel in ("actuator", "both"):
            actuator_noise = float(noise_level) * envelope * actuator_direction

        theta_sensed = theta_true + sensor_noise
        desired_pos, desired_vel = trajectory.get_pose(tk)
        current_pos_sensed = robot.forward_kinematics(theta_sensed)[:3, 3]
        jacobian_sensed = robot.jacobian(theta_sensed)[:3, :]

        lower, upper = compute_dynamic_velocity_bounds(
            theta_current=theta_sensed,
            theta_lower=cfg.theta_lower,
            theta_upper=cfg.theta_upper,
            theta_dot_limit=cfg.theta_dot_limit,
            eta=cfg.eta,
            tau=cfg.tau,
        )
        step_result = controller.step(theta_sensed, current_pos_sensed, desired_pos, desired_vel, jacobian_sensed, lower, upper)

        theta_dot_actual = step_result["theta_dot"] + actuator_noise
        theta_true = np.clip(theta_true + cfg.tau * theta_dot_actual, cfg.theta_lower, cfg.theta_upper)
        true_pos_after = robot.forward_kinematics(theta_true)[:3, 3]

        injected_noise = sensor_noise if noise_channel == "sensor" else actuator_noise
        if noise_channel == "both":
            injected_noise = np.concatenate((sensor_noise, actuator_noise))
        record_noise_history(
            history,
            tk,
            theta_true,
            true_pos_after,
            desired_pos,
            theta_reference,
            step_result,
            injected_noise,
            envelope,
        )

    wall_clock = time.perf_counter() - t_start
    summary = summarize_noise_run(
        label=f"{METHOD_LABEL} offline {noise_channel} {noise_profile} level={noise_level:g} trial={trial}",
        history=history,
        theta_initial=theta_reference,
        theta_final=theta_true,
        result_dir=output_dir,
        cfg=cfg,
        noise_channel=noise_channel,
        noise_profile=noise_profile,
        noise_level=noise_level,
        trial=trial,
        seed=seed,
        wall_clock=wall_clock,
    )

    if save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_figures(
            history,
            output_dir,
            f"{METHOD_LABEL} {noise_channel} {noise_profile} {noise_level:g}",
            theta_reference,
            theta_true,
            f"{noise_channel}_{noise_profile}_{noise_level:g}_trial{trial}",
        )
        (output_dir / f"{OUTPUT_STEM}_history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / f"{OUTPUT_STEM}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return summary


def aggregate_summaries(summaries):
    if not summaries:
        return {}
    metric_keys = [
        "mean_position_error_m",
        "max_position_error_m",
        "final_position_error_m",
        "mean_joint_drift_norm_rad",
        "max_abs_joint_drift_rad",
        "final_joint_drift_norm_rad",
        "mean_elncp_residual_norm",
        "max_elncp_residual_norm",
        "mean_stationarity_norm",
        "mean_ncp_residual_norm",
        "mean_pcg_iterations",
        "max_pcg_iterations",
        "mean_active_iterations",
        "max_active_iterations",
        "min_boundary_slack",
        "mean_noise_norm",
        "max_noise_norm",
        "wall_clock_run_s",
    ]
    row = {
        "method": summaries[0]["method"],
        "noise_channel": summaries[0]["noise_channel"],
        "noise_profile": summaries[0]["noise_profile"],
        "noise_level": summaries[0]["noise_level"],
        "trials": len(summaries),
        "duration_s": summaries[0]["duration_s"],
        "tau": summaries[0]["tau"],
    }
    for key in metric_keys:
        values = np.asarray([float(s[key]) for s in summaries], dtype=float)
        row[f"{key}_mean"] = float(np.mean(values))
        row[f"{key}_std"] = float(np.std(values))
        row[f"{key}_min"] = float(np.min(values))
        row[f"{key}_max"] = float(np.max(values))
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_summary(rows, output_dir, metric_key, title, ylabel):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax, channel in zip(axes, ("sensor", "actuator")):
        for profile in NOISE_PROFILES:
            selected = [
                r for r in rows
                if r["noise_channel"] == channel and r["noise_profile"] == profile and r["noise_level"] > 0.0
            ]
            selected.sort(key=lambda r: float(r["noise_level"]))
            if not selected:
                continue
            levels = np.asarray([float(r["noise_level"]) for r in selected], dtype=float)
            means = np.asarray([float(r[f"{metric_key}_mean"]) for r in selected], dtype=float)
            stds = np.asarray([float(r[f"{metric_key}_std"]) for r in selected], dtype=float)
            ax.errorbar(levels, means, yerr=stds, marker="o", linewidth=2.0, capsize=3, label=profile)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{channel} noise")
        ax.set_xlabel("peak noise level")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / f"{OUTPUT_STEM}_{metric_key}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_all_summaries(rows, output_dir):
    metrics = [
        ("mean_position_error_m", "Mean Position Error", "m"),
        ("final_position_error_m", "Final Position Error", "m"),
        ("final_joint_drift_norm_rad", "Final Joint Drift Norm", "rad"),
        ("mean_elncp_residual_norm", "Mean ELNCP Residual", "residual norm"),
        ("mean_pcg_iterations", "Mean PCG Iterations", "iterations"),
        ("min_boundary_slack", "Minimum Boundary Slack", "slack"),
    ]
    for key, title, ylabel in metrics:
        plot_metric_summary(rows, output_dir, key, title, ylabel)


def build_report(rows, baseline_rows, output_path, result_root=None, live_rows=None):
    result_root = Path(result_root) if result_root is not None else RESULTS_DIR
    live_rows = live_rows or []

    def best_row(channel, metric):
        candidates = [r for r in rows if r["noise_channel"] == channel and float(r["noise_level"]) > 0.0]
        if not candidates:
            return None
        return min(candidates, key=lambda r: float(r[f"{metric}_mean"]))

    def worst_row(channel, metric):
        candidates = [r for r in rows if r["noise_channel"] == channel and float(r["noise_level"]) > 0.0]
        if not candidates:
            return None
        return max(candidates, key=lambda r: float(r[f"{metric}_mean"]))

    def baseline_for(channel):
        candidates = [r for r in baseline_rows if r["noise_channel"] == channel]
        return candidates[0] if candidates else None

    def percent_change(value, base_value):
        base_value = float(base_value)
        if abs(base_value) <= 1e-30:
            return 0.0
        return 100.0 * (float(value) - base_value) / base_value

    lines = [
        "# Method3 TVQP ELNCP PCG 噪声鲁棒性验证",
        "",
        "本实验借鉴参考论文的鲁棒性验证方式，将噪声按照三类时间轮廓注入系统：",
        "",
        "$$",
        "n_1(t)=\\sigma,\\qquad n_2(t)=\\sigma\\frac{t}{T},\\qquad n_3(t)=4\\sigma\\left(\\frac{t}{T}-\\frac{1}{2}\\right)^2.",
        "$$",
        "",
        "其中 $\\sigma$ 是本机器人实验中的峰值噪声幅值。为了避免直接使用论文中对机器人过大的绝对噪声幅值，本文采用归一化时间轮廓，并分别作用于传感通道和执行通道。",
        "",
        "## 实验设置",
        "",
        "- 方法：Method3 TVQP ELNCP PCG。",
        "- 轨迹：UR3e heart trajectory。",
        "- 模式：offline 20 s。",
        "- 采样周期：$\\tau=0.005$ s。",
        "- 传感噪声：$q_{\\rm meas}=q+\\sigma h(t)d_s$。",
        "- 执行噪声：$\\dot q_{\\rm real}=\\dot q_{\\rm cmd}+\\sigma h(t)d_a$。",
        "- 每个噪声水平重复多次，方向 $d_s,d_a$ 由固定随机种子生成并单位化。",
        "",
        "## 基线结果",
        "",
        "| 通道 | 平均位置误差 | 最终位置误差 | 最终关节漂移 | 平均 ELNCP 残差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in baseline_rows:
        lines.append(
            f"| {row['noise_channel']} baseline | "
            f"{float(row['mean_position_error_m_mean']):.6e} | "
            f"{float(row['final_position_error_m_mean']):.6e} | "
            f"{float(row['final_joint_drift_norm_rad_mean']):.6e} | "
            f"{float(row['mean_elncp_residual_norm_mean']):.6e} |"
        )

    lines.extend([
        "",
        "## 鲁棒性趋势摘要",
        "",
        "| 噪声通道 | 最小平均位置误差组合 | 最小平均位置误差 | 最大退化组合 | 最大平均位置误差 |",
        "|---|---|---:|---|---:|",
    ])
    for channel in ("sensor", "actuator"):
        best = best_row(channel, "mean_position_error_m")
        worst = worst_row(channel, "mean_position_error_m")
        if best and worst:
            lines.append(
                f"| {channel} | "
                f"{best['noise_profile']}, $\\sigma={float(best['noise_level']):.1e}$ | "
                f"{float(best['mean_position_error_m_mean']):.6e} | "
                f"{worst['noise_profile']}, $\\sigma={float(worst['noise_level']):.1e}$ | "
                f"{float(worst['mean_position_error_m_mean']):.6e} |"
            )

    lines.extend([
        "",
        "| 噪声通道 | 最小最终漂移组合 | 最小最终漂移 | 最大漂移组合 | 最大最终漂移 |",
        "|---|---|---:|---|---:|",
    ])
    for channel in ("sensor", "actuator"):
        best = best_row(channel, "final_joint_drift_norm_rad")
        worst = worst_row(channel, "final_joint_drift_norm_rad")
        if best and worst:
            lines.append(
                f"| {channel} | "
                f"{best['noise_profile']}, $\\sigma={float(best['noise_level']):.1e}$ | "
                f"{float(best['final_joint_drift_norm_rad_mean']):.6e} | "
                f"{worst['noise_profile']}, $\\sigma={float(worst['noise_level']):.1e}$ | "
                f"{float(worst['final_joint_drift_norm_rad_mean']):.6e} |"
            )

    lines.extend([
        "",
        "## 数据分析",
        "",
    ])
    for channel in ("sensor", "actuator"):
        baseline = baseline_for(channel)
        worst_position = worst_row(channel, "mean_position_error_m")
        worst_drift = worst_row(channel, "final_joint_drift_norm_rad")
        worst_residual = worst_row(channel, "mean_elncp_residual_norm")
        if baseline and worst_position and worst_drift and worst_residual:
            position_change = percent_change(
                worst_position["mean_position_error_m_mean"],
                baseline["mean_position_error_m_mean"],
            )
            drift_ratio = float(worst_drift["final_joint_drift_norm_rad_mean"]) / max(
                float(baseline["final_joint_drift_norm_rad_mean"]), 1e-30
            )
            residual_change = percent_change(
                worst_residual["mean_elncp_residual_norm_mean"],
                baseline["mean_elncp_residual_norm_mean"],
            )
            lines.extend([
                f"- `{channel}` 通道最差平均位置误差出现在 `{worst_position['noise_profile']}` 噪声、"
                f"$\\sigma={float(worst_position['noise_level']):.1e}$，"
                f"平均位置误差为 ${float(worst_position['mean_position_error_m_mean']):.6e}$ m，"
                f"相对无噪声基线变化约 {position_change:.2f}%。",
                f"- `{channel}` 通道最差最终关节漂移出现在 `{worst_drift['noise_profile']}` 噪声、"
                f"$\\sigma={float(worst_drift['noise_level']):.1e}$，"
                f"最终关节漂移为 ${float(worst_drift['final_joint_drift_norm_rad_mean']):.6e}$ rad，"
                f"约为无噪声基线的 {drift_ratio:.2e} 倍。",
                f"- `{channel}` 通道最差平均 ELNCP 残差为 "
                f"${float(worst_residual['mean_elncp_residual_norm_mean']):.6e}$，"
                f"相对基线变化约 {residual_change:.2f}%。",
            ])
    lines.extend([
        "",
        "从上述结果看，求解器内部残差没有出现量级失控；性能退化主要来自噪声污染后的控制状态和执行状态，而不是 PCG 或 ELNCP 子问题本身发散。传感通道比执行通道更敏感，这符合该方法每个采样周期都依赖当前关节测量值构造雅可比矩阵、位置误差和边界约束的机制。",
    ])

    if live_rows:
        short_live = any(float(row.get("duration_s", 0.0)) < 10.0 for row in live_rows)
        live_title = "## Live 线性时变噪声冒烟测试（不纳入鲁棒性结论）" if short_live else "## Live 线性时变噪声验证"
        live_note = (
            "当前 live 数据包含短时运行。由于 heart trajectory 会在设置的 `duration` 内走完整个心形，"
            "当 `duration` 取 1 s 时，轨迹速度相当于 20 s 标准实验的约 20 倍，容易触发速度边界、PCG 活跃集饱和和大残差。"
            "因此这类短时 live 只能说明噪声注入和仿真链路能跑通，不能作为鲁棒性结论。"
            if short_live
            else "该 live 部分用于检查真实同步控制循环下的噪声传播趋势，离线全扫仍作为主要统计结论。"
        )
        lines.extend([
            "",
            live_title,
            "",
            "参考论文在机器人轨迹跟踪部分重点采用线性时变噪声，因此这里额外选取",
            "",
            "$$",
            "n_2(t)=\\sigma\\frac{t}{T}",
            "$$",
            "",
            live_note,
            "",
            "| 通道 | 轮廓 | 时长 | 噪声幅值 | 平均位置误差 | 最终位置误差 | 最终关节漂移 | 平均 ELNCP 残差 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in live_rows:
            lines.append(
                f"| {row['noise_channel']} | {row['noise_profile']} | "
                f"{float(row['duration_s']):.1f} s | "
                f"{float(row['noise_level']):.1e} | "
                f"{float(row['mean_position_error_m_mean']):.6e} | "
                f"{float(row['final_position_error_m_mean']):.6e} | "
                f"{float(row['final_joint_drift_norm_rad_mean']):.6e} | "
                f"{float(row['mean_elncp_residual_norm_mean']):.6e} |"
            )

    if live_rows:
        if any(float(row.get("duration_s", 0.0)) < 10.0 for row in live_rows):
            live_conclusion = (
                "3. 当前 live 数据属于短时链路冒烟测试，不应作为鲁棒性证据。"
                "原因是 1 s 心形轨迹会显著放大期望速度，导致控制器饱和和残差异常。"
                "后续需要按标准 20 s 或至少 10 s 轨迹周期重新跑 live 噪声实验。"
            )
        else:
            live_conclusion = (
                "3. 已按参考论文机器人验证中的线性时变噪声思路补充 CoppeliaSim live 验证。"
                "该 live 部分不替代离线全噪声轮廓统计，而是用于检查真实同步控制循环下的噪声传播是否与离线趋势一致。"
            )
    else:
        live_conclusion = (
            "3. 后续若要完全对标参考论文的机器人实验，应再选择线性时变噪声 $n_2(t)$ 做 CoppeliaSim live 验证，"
            "并报告轨迹误差曲线。"
        )

    lines.extend([
        "",
        "## 结论",
        "",
        "1. 该实验是对新 Method3 TVQP ELNCP PCG 的直接噪声鲁棒性验证，不再借用早期 Method2 或 Method3a 的噪声数据。",
        "",
        "2. 评价指标包括位置误差、关节漂移、ELNCP 残差、PCG 迭代次数和边界 slack，因此既能评价控制效果，也能评价求解器内部稳定性。",
        "",
        live_conclusion,
        "",
        "## 原始数据",
        "",
        f"- 离线聚合表：`{result_root / 'offline_noise_summary.csv'}`",
        f"- 离线单次 trial 表：`{result_root / 'offline_noise_trials.csv'}`",
        f"- 离线图像目录：`{result_root / 'plots'}`",
    ])
    if live_rows:
        live_root = result_root / "live_linear_reference"
        lines.extend([
            f"- Live 聚合表：`{live_root / 'live_noise_summary.csv'}`",
            f"- Live 单次 trial 表：`{live_root / 'live_noise_trials.csv'}`",
            f"- Live 图像目录：`{live_root / 'plots'}`",
        ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def run_offline_sweep(args):
    robot = UR3eKinematics()
    cfg = make_cfg(args.duration, args.tau, args)
    result_root = Path(args.output_root) if args.output_root else build_run_output_root(RESULTS_DIR, args, "offline")
    save_run_config(result_root, args, "offline")
    result_root.mkdir(parents=True, exist_ok=True)

    all_trial_rows = []
    aggregate_rows = []
    baseline_rows = []

    for channel in args.channels:
        print(f"[baseline] channel={channel}")
        baseline_summaries = []
        for trial in range(args.trials):
            seed = args.seed + 100000 + trial
            summary = run_offline_noise_trial(
                cfg=cfg,
                robot=robot,
                noise_channel=channel,
                noise_profile="constant",
                noise_level=0.0,
                trial=trial,
                seed=seed,
                output_dir=result_root / "representative" / f"{channel}_baseline_trial{trial}",
                save_artifacts=(trial == 0 and not args.skip_representative_figures),
            )
            baseline_summaries.append(summary)
            all_trial_rows.append(summary)
        baseline_row = aggregate_summaries(baseline_summaries)
        baseline_rows.append(baseline_row)
        aggregate_rows.append(baseline_row)

        for profile in args.profiles:
            for level in NOISE_LEVELS[channel]:
                print(f"[offline] channel={channel} profile={profile} level={level:.1e}", flush=True)
                trial_summaries = []
                for trial in range(args.trials):
                    seed = args.seed + abs(hash((channel, profile, level, trial))) % 1_000_000
                    is_representative = (
                        trial == 0
                        and level in (NOISE_LEVELS[channel][0], NOISE_LEVELS[channel][-1])
                        and not args.skip_representative_figures
                    )
                    output_dir = result_root / "representative" / f"{channel}_{profile}_{level:g}_trial{trial}"
                    summary = run_offline_noise_trial(
                        cfg=cfg,
                        robot=robot,
                        noise_channel=channel,
                        noise_profile=profile,
                        noise_level=level,
                        trial=trial,
                        seed=seed,
                        output_dir=output_dir,
                        save_artifacts=is_representative,
                    )
                    trial_summaries.append(summary)
                    all_trial_rows.append(summary)
                aggregate_rows.append(aggregate_summaries(trial_summaries))

    write_csv(result_root / "offline_noise_trials.csv", all_trial_rows)
    write_csv(result_root / "offline_noise_summary.csv", aggregate_rows)
    (result_root / "offline_noise_summary.json").write_text(
        json.dumps(aggregate_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_all_summaries(aggregate_rows, result_root / "plots")
    build_report(
        aggregate_rows,
        baseline_rows,
        DOCS_DIR / "method3_tvqp_elncp_pcg_noise_robustness_report.md",
        result_root=result_root,
    )
    return aggregate_rows


def load_aggregate_rows(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def rebuild_report_from_results(result_root):
    result_root = Path(result_root)
    offline_rows = load_aggregate_rows(result_root / "offline_noise_summary.csv")
    if not offline_rows:
        raise FileNotFoundError(f"Missing offline summary: {result_root / 'offline_noise_summary.csv'}")
    baseline_rows = [r for r in offline_rows if float(r["noise_level"]) == 0.0]
    live_rows = load_aggregate_rows(result_root / "live_linear_reference" / "live_noise_summary.csv")
    build_report(
        offline_rows,
        baseline_rows,
        DOCS_DIR / "method3_tvqp_elncp_pcg_noise_robustness_report.md",
        result_root=result_root,
        live_rows=live_rows,
    )
    return {"offline_rows": len(offline_rows), "live_rows": len(live_rows)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Noise robustness validation for Method3 TVQP ELNCP PCG.")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--channels", nargs="+", choices=["sensor", "actuator"], default=["sensor", "actuator"])
    parser.add_argument("--profiles", nargs="+", choices=list(NOISE_PROFILES), default=list(NOISE_PROFILES))
    parser.add_argument("--live-trials", type=int, default=1)
    parser.add_argument("--live-sensor-levels", nargs="+", type=float, default=[1e-4, 1e-3])
    parser.add_argument("--live-actuator-levels", nargs="+", type=float, default=[1e-3, 1e-2])
    parser.add_argument("--task-gain", type=float, default=None)
    parser.add_argument("--position-weight", type=float, default=15000.0)
    parser.add_argument("--drift-weight", type=float, default=5e-4)
    parser.add_argument("--regularization-gain", type=float, default=1e-10)
    parser.add_argument("--ncp-epsilon", type=float, default=1e-10)
    parser.add_argument("--pcg-max-iters", type=int, default=8)
    parser.add_argument("--pcg-tol", type=float, default=1e-10)
    parser.add_argument("--normal-reg", type=float, default=1e-10)
    parser.add_argument("--max-active-set-iters", type=int, default=16)
    parser.add_argument("--active-set-tol", type=float, default=1e-9)
    parser.add_argument("--no-jacobi-preconditioner", action="store_true")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--skip-representative-figures", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.output_root is None:
        args.output_root = str(build_run_output_root(RESULTS_DIR, args, "offline"))
    rows = run_offline_sweep(args)
    result_root = Path(args.output_root)
    print(json.dumps({"status": "ok", "rows": len(rows), "result_root": str(result_root)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

