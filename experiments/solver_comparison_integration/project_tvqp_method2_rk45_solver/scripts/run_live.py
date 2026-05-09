# Complete live experiment entry for direct VSCode execution.
# This file is intentionally self-contained for direct VSCode execution.

"""
Method 2 continuous TVQP experiment with RK45 integration in the LCCZNN layer.

This script is intentionally kept as an independent project entry. It reuses
the existing UR3e model, trajectory, plotting, and CoppeliaSim I/O utilities,
but replaces the solver state update

    y_{k+1} = y_k + tau f(y_k)

with a Dormand-Prince RK45 integration of

    dy/ds = f(y),  s in [0, tau].

The outer robot command remains the same as the original experiment:

    theta_{k+1} = theta_k + tau qdot_k.

Therefore the experiment isolates the effect of the LCCZNN solver integrator
without changing the sampled robot-control interface.
"""

from __future__ import annotations

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
from scipy.integrate import solve_ivp


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

THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parents[1]


def find_workspace_code_root(start_dir):
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "core").is_dir() and (candidate / "sim").is_dir():
            return candidate
        nested = candidate / "code"
        if (nested / "core").is_dir() and (nested / "sim").is_dir():
            return nested
    raise RuntimeError(f"Cannot locate workspace code root from {start_dir}")


def find_feedback_ablation_root(start_dir):
    for candidate in (start_dir, *start_dir.parents):
        direct = candidate / "experiments_feedback_ablation"
        if direct.is_dir():
            return direct
        nested = candidate / "experiments" / "trajectory_ablation_tuning" / "experiments_feedback_ablation"
        if nested.is_dir():
            return nested
        legacy = candidate / "code" / "experiments_feedback_ablation"
        if legacy.is_dir():
            return legacy
    return None


CODE_DIR = find_workspace_code_root(THIS_FILE.parent)
CORE_DIR = CODE_DIR / "core"
SIM_DIR = CODE_DIR / "sim"
FEEDBACK_ABLATION_ROOT = find_feedback_ablation_root(THIS_FILE.parent)

for path in (CORE_DIR, SIM_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sim
import discrete_rmp_qp_common as common
from discrete_rmp_qp_common import (
    UR3eKinematics,
    HeartTrajectory,
    begin_joint_position_stream,
    connect_to_coppeliasim,
    make_offset_from_initial_position,
    query_joint_limits_from_sim,
    read_joint_positions_fast,
    send_joint_targets,
    startup_handshake_and_settle,
)

DEFAULT_THETA_INITIAL = np.array(
    [0.0, -np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0], dtype=float
)

METHOD_DISPLAY = {
    "method2": "Method 2: energy shaping + RK45 LCCZNN",
}


def tuned_method2_task_gain(tau):
    tau = float(tau)
    if tau <= 0.005:
        return 220.0
    if tau <= 0.02:
        return 80.0
    return 20.0


@dataclass
class ExperimentSettings:
    duration: float = 10.0
    offline_duration: float = 20.0
    tau: float = 0.005
    heart_scale: float = 0.008
    theta_dot_limit: float = 2.0
    eta: float = 0.9
    theta_initial_command: np.ndarray = field(
        default_factory=lambda: DEFAULT_THETA_INITIAL.copy()
    )

    @property
    def steps(self):
        return int(round(self.duration / self.tau))

    @property
    def offline_steps(self):
        return int(round(self.offline_duration / self.tau))


def create_history():
    return {
        "time_s": [],
        "actual_positions": [],
        "desired_positions": [],
        "position_errors": [],
        "joint_positions": [],
        "joint_drift": [],
        "task_residuals": [],
        "drift_norms": [],
    }


def record_history(history, tk, theta_current, current_pos, desired_pos, theta_reference, task_residual=None):
    drift = np.asarray(theta_current, dtype=float) - np.asarray(theta_reference, dtype=float)
    history["time_s"].append(float(tk))
    history["actual_positions"].append(np.asarray(current_pos, dtype=float).tolist())
    history["desired_positions"].append(np.asarray(desired_pos, dtype=float).tolist())
    history["position_errors"].append(float(np.linalg.norm(current_pos - desired_pos)))
    history["joint_positions"].append(np.asarray(theta_current, dtype=float).tolist())
    history["joint_drift"].append(drift.tolist())
    if task_residual is not None:
        history["task_residuals"].append(float(np.linalg.norm(task_residual)))
    history["drift_norms"].append(float(np.linalg.norm(drift)))


def configure_3d_axes(ax, desired, actual):
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


def plot_trajectory(history, output_path, title_prefix):
    desired = np.asarray(history["desired_positions"], dtype=float)
    actual = np.asarray(history["actual_positions"], dtype=float)
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(desired[:, 0], desired[:, 1], desired[:, 2], color="#c0392b", linewidth=2.2, label="Desired")
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], color="#1f77b4", linestyle="--", linewidth=2.0, label="Actual")
    configure_3d_axes(ax, desired, actual)
    ax.set_title(f"{title_prefix} Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_joint_angles(history, output_path, title_prefix):
    joints = np.asarray(history["joint_positions"], dtype=float)
    time_vec = np.asarray(history["time_s"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i in range(joints.shape[1]):
        ax.plot(time_vec, joints[:, i], linewidth=1.6, label=rf"$\theta_{{{i+1}}}$")
    ax.set_title(f"{title_prefix} Joint Angles")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("rad")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_position_error(history, output_path, title_prefix):
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


def plot_joint_drift(history, output_path, title_prefix):
    drift = np.asarray(history["joint_drift"], dtype=float)
    time_vec = np.asarray(history["time_s"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i in range(drift.shape[1]):
        ax.plot(time_vec, drift[:, i], linewidth=1.6, label=rf"$\Delta\theta_{{{i+1}}}(t)$")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_title(f"{title_prefix} Joint Drift Error")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"$\theta_i(t)-\theta_i(0)$ (rad)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_task_residual(history, output_path, title_prefix):
    residuals = np.asarray(history.get("task_residuals", []), dtype=float)
    if len(residuals) == 0:
        return
    time_vec = np.asarray(history["time_s"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(time_vec, np.clip(residuals, 1e-12, None), color="#2ca02c", linewidth=2.0)
    ax.set_title(f"{title_prefix} Task Residual")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"$||J\dot{q} - b||_2$")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_all_figures(history, output_dir, stem, title_prefix):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory(history, output_dir / f"{stem}_trajectory.png", title_prefix)
    plot_joint_angles(history, output_dir / f"{stem}_joint_angles.png", title_prefix)
    plot_position_error(history, output_dir / f"{stem}_position_error.png", title_prefix)
    plot_joint_drift(history, output_dir / f"{stem}_joint_drift.png", title_prefix)
    plot_task_residual(history, output_dir / f"{stem}_task_residual.png", title_prefix)


def save_summary(output_dir, stem, theta_initial, theta_final, history):
    output_dir.mkdir(parents=True, exist_ok=True)
    theta_i = np.asarray(theta_initial, dtype=float)
    theta_f = np.asarray(theta_final, dtype=float)
    delta = theta_f - theta_i
    rows = []
    for i in range(len(theta_i)):
        rows.append({
            "joint": f"theta_{i+1}",
            "initial_rad": float(theta_i[i]),
            "final_rad": float(theta_f[i]),
            "delta_rad": float(delta[i]),
        })
    errors = np.asarray(history["position_errors"], dtype=float)
    drift_norms = np.asarray(history.get("drift_norms", []), dtype=float)
    summary = {
        "stem": stem,
        "theta_initial": theta_i.tolist(),
        "theta_final": theta_f.tolist(),
        "joint_drift_table": rows,
        "max_joint_drift_rad": float(np.max(np.abs(delta))),
        "final_joint_drift_norm_rad": float(np.linalg.norm(delta)),
        "mean_position_error_m": float(np.mean(errors)),
        "final_position_error_m": float(errors[-1]),
        "max_position_error_m": float(np.max(errors)),
    }
    if len(drift_norms) > 0:
        summary["max_drift_norm"] = float(np.max(drift_norms))
        summary["final_drift_norm"] = float(drift_norms[-1])
    (output_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output_dir / f"{stem}_joint_drift_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["joint", "initial_rad", "final_rad", "delta_rad"])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def scalar_lccznn_activation(value: float, power: float, exp_clip: float) -> float:
    energy = max(float(value), 0.0)
    return (energy ** float(power)) * np.exp(min(energy, float(exp_clip)))


@dataclass
class RK45LCCZNNSolverConfig:
    tau: float = 0.005
    gamma: float = 4608.0
    activation_power: float = 0.85
    activation_exp_clip: float = 4.0
    lambda_reg: float = 1e-8
    residual_tol: float = 1e-10
    rtol: float = 1e-6
    atol: float = 1e-9
    max_step_factor: float = 1.0
    mode: str = "adaptive"


class RK45LCCZNNSolver:
    """Integrate the continuous LCCZNN correction ODE by RK45 for one sample."""

    def __init__(self, size: int, config: RK45LCCZNNSolverConfig):
        self.size = int(size)
        self.cfg = config
        self.state = np.zeros(self.size, dtype=float)

    def reset(self):
        self.state = np.zeros(self.size, dtype=float)

    def set_state(self, y0):
        self.state = np.asarray(y0, dtype=float).copy()

    def _field(self, y, residual_fn, direction_fn):
        y = np.asarray(y, dtype=float)
        residual = np.asarray(residual_fn(y), dtype=float)
        energy = 0.5 * float(residual @ residual)
        if not np.isfinite(energy):
            raise FloatingPointError("Non-finite RK45 LCCZNN residual energy.")

        direction = np.asarray(direction_fn(y, residual), dtype=float)
        denom = float(direction @ direction) + float(self.cfg.lambda_reg)
        if denom <= 0.0 or not np.isfinite(denom):
            denom = 1e-12

        gain = self.cfg.gamma * scalar_lccznn_activation(
            energy,
            power=self.cfg.activation_power,
            exp_clip=self.cfg.activation_exp_clip,
        )
        return -gain * direction / denom

    def _dormand_prince_step(self, z0, residual_fn, direction_fn):
        """One Dormand-Prince RK45 step over the control interval."""
        h = float(self.cfg.tau)
        f = lambda y: self._field(y, residual_fn, direction_fn)

        k1 = f(z0)
        k2 = f(z0 + h * (1.0 / 5.0) * k1)
        k3 = f(z0 + h * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2))
        k4 = f(
            z0
            + h
            * (
                (44.0 / 45.0) * k1
                - (56.0 / 15.0) * k2
                + (32.0 / 9.0) * k3
            )
        )
        k5 = f(
            z0
            + h
            * (
                (19372.0 / 6561.0) * k1
                - (25360.0 / 2187.0) * k2
                + (64448.0 / 6561.0) * k3
                - (212.0 / 729.0) * k4
            )
        )
        k6 = f(
            z0
            + h
            * (
                (9017.0 / 3168.0) * k1
                - (355.0 / 33.0) * k2
                + (46732.0 / 5247.0) * k3
                + (49.0 / 176.0) * k4
                - (5103.0 / 18656.0) * k5
            )
        )

        z5 = z0 + h * (
            (35.0 / 384.0) * k1
            + (500.0 / 1113.0) * k3
            + (125.0 / 192.0) * k4
            - (2187.0 / 6784.0) * k5
            + (11.0 / 84.0) * k6
        )

        # The embedded fourth-order estimate is recorded as a diagnostic only.
        # We deliberately use one RK45 step per control sample to keep the
        # comparison isolated from extra adaptive inner iterations.
        k7 = f(z5)
        z4 = z0 + h * (
            (5179.0 / 57600.0) * k1
            + (7571.0 / 16695.0) * k3
            + (393.0 / 640.0) * k4
            - (92097.0 / 339200.0) * k5
            + (187.0 / 2100.0) * k6
            + (1.0 / 40.0) * k7
        )
        return z5, float(np.linalg.norm(z5 - z4)), 7

    def _adaptive_rk45_step(self, z0, residual_fn, direction_fn):
        rhs = lambda _, y: self._field(y, residual_fn, direction_fn)
        max_step = max(float(self.cfg.tau) * float(self.cfg.max_step_factor), 1e-12)
        sol = solve_ivp(
            rhs,
            (0.0, float(self.cfg.tau)),
            z0,
            method="RK45",
            t_eval=[float(self.cfg.tau)],
            rtol=float(self.cfg.rtol),
            atol=float(self.cfg.atol),
            max_step=max_step,
        )
        if not sol.success:
            raise FloatingPointError(f"Adaptive RK45 LCCZNN integration failed: {sol.message}")
        return np.asarray(sol.y[:, -1], dtype=float), 0.0, int(sol.nfev), str(sol.message)

    def step(self, residual_fn, direction_fn, project_fn=None):
        z0 = self.state.copy()
        residual0 = np.asarray(residual_fn(z0), dtype=float)
        residual_norm0 = float(np.linalg.norm(residual0))
        z_raw = z0.copy()
        nfev = 0
        error_estimate_norm = 0.0
        solver_status = "skipped_tol"
        solver_message = "Initial residual below tolerance."

        if residual_norm0 > self.cfg.residual_tol:
            if self.cfg.mode == "fixed":
                z_raw, error_estimate_norm, nfev = self._dormand_prince_step(
                    z0, residual_fn, direction_fn
                )
                solver_message = "One fixed Dormand-Prince RK45 step completed."
            elif self.cfg.mode == "adaptive":
                z_raw, error_estimate_norm, nfev, solver_message = self._adaptive_rk45_step(
                    z0, residual_fn, direction_fn
                )
            else:
                raise ValueError(f"Unsupported RK45 mode: {self.cfg.mode}")
            solver_status = "ok"

        z = z_raw.copy() if project_fn is None else project_fn(z_raw)
        if not np.all(np.isfinite(z)):
            raise FloatingPointError("RK45 LCCZNN solver diverged; reduce gain or tolerance.")

        self.state = z.copy()
        final_residual = np.asarray(residual_fn(z), dtype=float)
        return {
            "state": z.copy(),
            "state_raw": z_raw.copy(),
            "residual": final_residual.copy(),
            "residual_norm": float(np.linalg.norm(final_residual)),
            "energy": 0.5 * float(final_residual @ final_residual),
            "nfev": nfev,
            "error_estimate_norm": error_estimate_norm,
            "solver_status": solver_status,
            "solver_message": solver_message,
        }


def clip_joint_step(theta_current, tau, qdot_cmd, theta_lower, theta_upper):
    return np.clip(
        np.asarray(theta_current, dtype=float) + float(tau) * np.asarray(qdot_cmd, dtype=float),
        np.asarray(theta_lower, dtype=float),
        np.asarray(theta_upper, dtype=float),
    )


@dataclass
class Method2RK45Config:
    tau: float = 0.005
    task_gain: float | None = None
    mu_gain: float = 20.0
    activation_power: float = 0.85
    activation_exp_clip: float = 4.0
    eta: float = 0.9
    theta_dot_limit: float = 2.0
    theta_lower: np.ndarray = field(
        default_factory=lambda: -2.0 * np.pi * np.ones(6, dtype=float)
    )
    theta_upper: np.ndarray = field(
        default_factory=lambda: 2.0 * np.pi * np.ones(6, dtype=float)
    )
    solver_gamma_gain: float = 4608.0
    solver_power: float = 0.85
    solver_exp_clip: float = 4.0
    solver_lambda: float = 1e-8
    solver_tol: float = 1e-10
    rk45_rtol: float = 1e-6
    rk45_atol: float = 1e-9
    rk45_max_step_factor: float = 1.0
    rk45_mode: str = "adaptive"

    def __post_init__(self):
        if self.task_gain is None:
            self.task_gain = tuned_method2_task_gain(self.tau)


class Method2RK45Controller:
    """
    Method 2 TVQP controller with RK45-integrated LCCZNN solver dynamics.

    The QP shell is the same as the original with-feedback Method 2:

        min 0.5 qdot^T qdot + c_d^T qdot
        s.t. J qdot = rdot_d - k_p e_p.

    The KKT residual is R(y)=H y+p. Only the continuous LCCZNN flow used to
    reduce this residual is integrated by RK45.
    """

    def __init__(self, robot, config: Method2RK45Config):
        self.robot = robot
        self.cfg = config
        self.n_joints = robot.num_joints
        self.task_dim = 3
        self.identity = np.eye(self.n_joints, dtype=float)
        self.theta_initial = None
        self.solver = RK45LCCZNNSolver(
            size=self.n_joints + self.task_dim,
            config=RK45LCCZNNSolverConfig(
                tau=config.tau,
                gamma=config.solver_gamma_gain,
                activation_power=config.solver_power,
                activation_exp_clip=config.solver_exp_clip,
                lambda_reg=config.solver_lambda,
                residual_tol=config.solver_tol,
                rtol=config.rk45_rtol,
                atol=config.rk45_atol,
                max_step_factor=config.rk45_max_step_factor,
                mode=config.rk45_mode,
            ),
        )

    def reset(self, theta_initial):
        self.theta_initial = np.asarray(theta_initial, dtype=float).copy()
        self.solver.reset()

    def update_joint_limits(self, theta_lower, theta_upper):
        self.cfg.theta_lower = np.asarray(theta_lower, dtype=float).copy()
        self.cfg.theta_upper = np.asarray(theta_upper, dtype=float).copy()

    def _compute_velocity_bounds(self, theta_current):
        theta_dot_lower = -self.cfg.theta_dot_limit * np.ones(self.n_joints, dtype=float)
        theta_dot_upper = self.cfg.theta_dot_limit * np.ones(self.n_joints, dtype=float)
        return common.compute_dynamic_velocity_bounds(
            theta_current=np.asarray(theta_current, dtype=float),
            theta_lower=self.cfg.theta_lower,
            theta_upper=self.cfg.theta_upper,
            theta_dot_lower=theta_dot_lower,
            theta_dot_upper=theta_dot_upper,
            eta=self.cfg.eta,
            tau=self.cfg.tau,
        )

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
        drift_feedback = self.cfg.mu_gain * common.sig_exp_activation(
            drift_delta,
            power=self.cfg.activation_power,
            exp_clip=self.cfg.activation_exp_clip,
        )

        lower, upper = self._compute_velocity_bounds(theta_current)
        h_matrix = np.block(
            [
                [self.identity, -jacobian_task.T],
                [jacobian_task, np.zeros((self.task_dim, self.task_dim), dtype=float)],
            ]
        )
        p_vector = np.concatenate((drift_feedback, -task_command))

        def residual_fn(y):
            return h_matrix @ y + p_vector

        def direction_fn(_, residual):
            return h_matrix.T @ residual

        def project_fn(y_raw):
            y_next = np.asarray(y_raw, dtype=float).copy()
            y_next[: self.n_joints] = np.clip(y_next[: self.n_joints], lower, upper)
            return y_next

        solver_output = self.solver.step(residual_fn, direction_fn, project_fn)
        y_final = solver_output["state"]
        y_raw = solver_output["state_raw"]

        qdot_cmd = y_final[: self.n_joints].copy()
        qdot_raw = y_raw[: self.n_joints].copy()
        dual_next = y_final[self.n_joints :].copy()
        theta_next = clip_joint_step(
            theta_current,
            self.cfg.tau,
            qdot_cmd,
            self.cfg.theta_lower,
            self.cfg.theta_upper,
        )

        return {
            "theta_next": theta_next,
            "theta_dot_next": qdot_cmd,
            "theta_dot_raw": qdot_raw,
            "dual_next": dual_next,
            "current_pos": current_pos,
            "desired_pos": desired_position,
            "desired_vel": desired_velocity,
            "tracking_error": position_error,
            "task_command": task_command,
            "task_residual": task_command - jacobian_task @ qdot_cmd,
            "drift_delta": drift_delta,
            "drift_feedback": drift_feedback,
            "inner_residual_norm": solver_output["residual_norm"],
            "inner_energy": solver_output["energy"],
            "rk45_nfev": solver_output["nfev"],
            "xi_minus": lower,
            "xi_plus": upper,
        }


def build_method2_rk45(robot, settings=None):
    cfg = Method2RK45Config()
    if settings is not None:
        cfg.tau = float(getattr(settings, "tau", cfg.tau))
        cfg.task_gain = tuned_method2_task_gain(cfg.tau)
        cfg.theta_dot_limit = float(getattr(settings, "theta_dot_limit", cfg.theta_dot_limit))
        cfg.eta = float(getattr(settings, "eta", cfg.eta))
        cfg.solver_gamma_gain = float(getattr(settings, "solver_gamma_gain", cfg.solver_gamma_gain))
        cfg.rk45_rtol = float(getattr(settings, "rk45_rtol", cfg.rk45_rtol))
        cfg.rk45_atol = float(getattr(settings, "rk45_atol", cfg.rk45_atol))
        cfg.rk45_max_step_factor = float(
            getattr(settings, "rk45_max_step_factor", cfg.rk45_max_step_factor)
        )
        cfg.rk45_mode = str(getattr(settings, "rk45_mode", cfg.rk45_mode))
        if hasattr(settings, "theta_lower"):
            cfg.theta_lower = np.asarray(settings.theta_lower, dtype=float).copy()
        if hasattr(settings, "theta_upper"):
            cfg.theta_upper = np.asarray(settings.theta_upper, dtype=float).copy()
    else:
        cfg.task_gain = tuned_method2_task_gain(cfg.tau)
    return Method2RK45Controller(robot, cfg)


def collect_solver_diagnostics(output_dir: Path, stem: str, history, summary: dict, settings):
    """Add RK45 metadata while keeping the original summary schema intact."""
    summary.update(
        {
            "solver_integrator": "RK45",
            "solver_gamma_gain": float(settings.solver_gamma_gain),
            "rk45_rtol": float(settings.rk45_rtol),
            "rk45_atol": float(settings.rk45_atol),
            "rk45_max_step_factor": float(settings.rk45_max_step_factor),
            "rk45_mode": str(settings.rk45_mode),
        }
    )
    json_path = output_dir / f"{stem}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_live(method_name, controller_builder, robot, settings, output_dir, use_feedback=True):
    display_name = METHOD_DISPLAY.get(method_name, method_name)
    mode_label = "with_fb" if use_feedback else "nofb"
    stem = f"{method_name}_live_{mode_label}"
    title_prefix = f"{display_name} Live ({mode_label})"
    print(f"  [{display_name}] Running live ({mode_label})...")

    client_id, port = connect_to_coppeliasim()
    if client_id == -1:
        print("    CoppeliaSim not available, skipping live.")
        return None, None

    print(f"    Connected to CoppeliaSim on port {port}")
    sim.simxSynchronous(client_id, True)
    sim.simxStartSimulation(client_id, sim.simx_opmode_oneshot)

    controller = controller_builder(robot, settings)
    history = create_history()
    theta_current = None
    theta_reference = settings.theta_initial_command.copy()

    try:
        joint_handles = []
        for i in range(1, 7):
            err, handle = sim.simxGetObjectHandle(
                client_id, f"UR3e_joint{i}", sim.simx_opmode_blocking
            )
            if err != sim.simx_return_ok:
                raise RuntimeError(f"Joint UR3e_joint{i} not found, error={err}")
            joint_handles.append(handle)
        begin_joint_position_stream(client_id, joint_handles)

        theta_lower, theta_upper = query_joint_limits_from_sim(client_id, joint_handles)
        settings.theta_lower = theta_lower
        settings.theta_upper = theta_upper

        _, theta_reference = startup_handshake_and_settle(
            client_id,
            joint_handles,
            settings.theta_initial_command,
            settle_steps=20,
            tau=settings.tau,
            actuation_mode="target",
        )
        controller.reset(theta_reference)

        if hasattr(controller, "update_joint_limits"):
            controller.update_joint_limits(theta_lower, theta_upper)

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
            current_pos = result["current_pos"]
            record_history(history, tk, theta_current, current_pos, desired_pos, theta_reference, result.get("task_residual"))
            send_joint_targets(client_id, joint_handles, result["theta_next"], actuation_mode="target")
            sim.simxSynchronousTrigger(client_id)
            sim.simxGetPingTime(client_id)

        runtime_s = time.perf_counter() - t_start
        print(f"    Runtime: {runtime_s:.1f}s, steps: {total_steps}")
    finally:
        sim.simxStopSimulation(client_id, sim.simx_opmode_oneshot)
        sim.simxFinish(client_id)

    if theta_current is None:
        theta_current = theta_reference
    save_all_figures(history, output_dir, stem, title_prefix)
    summary = save_summary(output_dir, stem, theta_reference, theta_current, history)
    summary["runtime_s"] = runtime_s
    return history, summary


def write_comparison(output_root: Path):
    baseline_root = FEEDBACK_ABLATION_ROOT
    baseline_offline = (baseline_root / (
        "experiments_tvqp_dlccznn_paper_with_feedback/method2/offline/"
        "method2_offline_with_fb_summary.json"
    )) if baseline_root is not None else Path("__missing_baseline_offline__")
    baseline_live = (baseline_root / (
        "experiments_tvqp_dlccznn_paper_with_feedback/method2/live/"
        "method2_live_with_fb_summary.json"
    )) if baseline_root is not None else Path("__missing_baseline_live__")
    rk45_offline = output_root / "method2/offline/method2_offline_with_fb_summary.json"
    rk45_live = output_root / "method2/live/method2_live_with_fb_summary.json"

    rows = []
    for label, path in (
        ("euler_offline", baseline_offline),
        ("rk45_offline", rk45_offline),
        ("euler_live", baseline_live),
        ("rk45_live", rk45_live),
    ):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "case": label,
                "mean_position_error_m": data.get("mean_position_error_m"),
                "final_position_error_m": data.get("final_position_error_m"),
                "max_position_error_m": data.get("max_position_error_m"),
                "final_joint_drift_norm_rad": data.get("final_joint_drift_norm_rad"),
                "max_joint_drift_rad": data.get("max_joint_drift_rad"),
                "path": str(path),
            }
        )

    if rows:
        path = output_root / "comparison_euler_vs_rk45.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def run_experiment(args):
    output_root = Path(args.output_root) if args.output_root else build_run_output_root(PROJECT_DIR / "results", args, "live")
    output_root.mkdir(parents=True, exist_ok=True)
    save_run_config(output_root, args, "live")

    settings = ExperimentSettings(
        duration=float(args.duration),
        offline_duration=float(args.offline_duration),
        tau=float(args.tau),
        heart_scale=float(args.heart_scale),
        theta_dot_limit=float(args.theta_dot_limit),
        eta=float(args.eta),
        theta_initial_command=DEFAULT_THETA_INITIAL.copy(),
    )
    settings.solver_gamma_gain = float(args.solver_gamma_gain)
    settings.rk45_rtol = float(args.rtol)
    settings.rk45_atol = float(args.atol)
    settings.rk45_max_step_factor = float(args.max_step_factor)
    settings.rk45_mode = str(args.rk45_mode)

    robot = UR3eKinematics()
    all_summaries = {}

    print("=== Method 2 TVQP with feedback, RK45-integrated LCCZNN solver ===")
    print(f"Output: {output_root}")
    print(
        "Settings: "
        f"tau={settings.tau}, duration={settings.duration}, "
        f"offline_duration={settings.offline_duration}, "
        f"gamma={settings.solver_gamma_gain}, rtol={settings.rk45_rtol}, "
        f"atol={settings.rk45_atol}, max_step_factor={settings.rk45_max_step_factor}, "
        f"rk45_mode={settings.rk45_mode}"
    )

    _, summary = run_live(
        "method2",
        build_method2_rk45,
        robot,
        settings,
        output_root / "method2" / "live",
        use_feedback=True,
    )
    if summary:
        collect_solver_diagnostics(
            output_root / "method2" / "live",
            "method2_live_with_fb",
            None,
            summary,
            settings,
        )
        all_summaries["method2_live"] = summary

    if all_summaries:
        (output_root / "all_summaries.json").write_text(
            json.dumps(all_summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_comparison(output_root)

        print("\n=== Summary ===")
        for key, item in all_summaries.items():
            print(
                f"{key}: mean={item['mean_position_error_m']:.6e} m, "
                f"final={item['final_position_error_m']:.6e} m, "
                f"drift={item['final_joint_drift_norm_rad']:.6e} rad"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Run Method 2 TVQP with RK45-integrated LCCZNN solver."
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--offline-duration", type=float, default=20.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--heart-scale", type=float, default=0.008)
    parser.add_argument("--theta-dot-limit", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--solver-gamma-gain", type=float, default=4608.0)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--max-step-factor", type=float, default=1.0)
    parser.add_argument("--rk45-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

