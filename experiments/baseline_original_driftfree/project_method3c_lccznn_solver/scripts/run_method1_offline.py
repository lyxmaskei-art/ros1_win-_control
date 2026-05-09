# Complete offline experiment entry for direct VSCode execution.
# This file is intentionally self-contained for direct VSCode execution.

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

from _bootstrap import RESULTS_DIR


METHOD_NAME = "Method 1: sig-type drift QP with DLCCZNN solver"
OUTPUT_STEM = "method1_sig_rmp_qp_dlccznn"
DEFAULT_REMOTE_API_PORTS = (19997, 19998)
DEFAULT_THETA_LIMIT = 2.0 * np.pi * np.ones(6, dtype=float)
RUN_TAG_KEYS = (
    "duration",
    "offline_duration",
    "tau",
    "task_gain",
    "mu_gain",
    "solver_substeps",
    "solver_gamma_gain",
    "solver_lambda",
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
    tag = "__".join(parts)[:140].rstrip("_")
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


def sign_bi_power(z, power):
    z = np.asarray(z, dtype=float)
    abs_z = np.abs(z)
    powered = np.where(abs_z == 0.0, 0.0, np.power(abs_z, power))
    return np.sign(z) * powered


def sig_exp_activation(z, power, exp_clip):
    z = np.asarray(z, dtype=float)
    abs_z = np.minimum(np.abs(z), exp_clip)
    return sign_bi_power(z, power) * np.exp(abs_z)


def dlccznn_scalar_activation(value, power, exp_clip):
    scalar = max(float(value), 0.0)
    return (scalar ** float(power)) * np.exp(min(scalar, float(exp_clip)))


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


@dataclass
class Method1DLCCZNNConfig:
    duration: float = 10.0
    tau: float = 0.005
    heart_scale: float = 0.008
    task_gain: float = 20.0
    mu_gain: float = 0.1
    activation_power: float = 0.9
    activation_exp_clip: float = 2.0
    eta: float = 0.9
    theta_dot_limit: float = 2.0
    offline_duration: float = 20.0
    solver_substeps: int = 20
    solver_gamma_gain: float = 80.0
    solver_power: float = 0.9
    solver_exp_clip: float = 4.0
    solver_lambda: float = 1e-8
    solver_tol: float = 1e-10
    theta_initial_command: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0], dtype=float)
    )
    theta_lower: np.ndarray = field(default_factory=lambda: -DEFAULT_THETA_LIMIT.copy())
    theta_upper: np.ndarray = field(default_factory=lambda: DEFAULT_THETA_LIMIT.copy())

    @property
    def steps(self):
        return int(round(self.duration / self.tau))


class Method1DLCCZNNSolver:
    def __init__(self, cfg, num_primal=6, num_dual=3):
        self.cfg = cfg
        self.num_primal = int(num_primal)
        self.num_dual = int(num_dual)
        self.state = np.zeros(self.num_primal + self.num_dual, dtype=float)

    def reset(self):
        self.state = np.zeros(self.num_primal + self.num_dual, dtype=float)

    def step(self, h_matrix, p_vector, lower, upper):
        h_matrix = np.asarray(h_matrix, dtype=float)
        p_vector = np.asarray(p_vector, dtype=float).reshape(-1)
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)

        inner_tau = self.cfg.tau / float(max(int(self.cfg.solver_substeps), 1))
        y = self.state.copy()
        y_raw = y.copy()
        residual = h_matrix @ y + p_vector

        for _ in range(max(int(self.cfg.solver_substeps), 1)):
            residual = h_matrix @ y + p_vector
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm <= self.cfg.solver_tol:
                break

            energy = 0.5 * float(residual @ residual)
            direction = h_matrix.T @ residual
            denom = float(direction @ direction) + self.cfg.solver_lambda
            if denom <= 0.0:
                denom = 1e-12

            correction = self.cfg.solver_gamma_gain * dlccznn_scalar_activation(
                energy,
                power=self.cfg.solver_power,
                exp_clip=self.cfg.solver_exp_clip,
            )
            y_raw = y - inner_tau * correction * (direction / denom)
            y = y_raw.copy()
            y[: self.num_primal] = np.clip(y[: self.num_primal], lower, upper)
            if not np.all(np.isfinite(y)):
                raise FloatingPointError("Method 1 DLCCZNN solver diverged; reduce solver gain.")

        self.state = y.copy()
        final_residual = h_matrix @ y + p_vector
        return {
            "theta_dot": y[: self.num_primal].copy(),
            "theta_dot_raw": y_raw[: self.num_primal].copy(),
            "dual_next": y[self.num_primal :].copy(),
            "residual_norm": float(np.linalg.norm(final_residual)),
            "energy": 0.5 * float(final_residual @ final_residual),
        }


class Method1Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.identity = np.eye(6, dtype=float)
        self.theta_initial = None
        self.solver = Method1DLCCZNNSolver(cfg)

    def reset(self, theta_initial):
        self.theta_initial = np.asarray(theta_initial, dtype=float).copy()
        self.solver.reset()

    def step(self, theta_current, current_pos, desired_pos, desired_vel, jacobian_task, lower, upper):
        if self.theta_initial is None:
            self.reset(theta_current)

        position_error = current_pos - desired_pos
        task_feedback = desired_vel - self.cfg.task_gain * position_error
        drift_delta = np.asarray(theta_current, dtype=float) - self.theta_initial
        drift_feedback = self.cfg.mu_gain * sig_exp_activation(
            drift_delta,
            power=self.cfg.activation_power,
            exp_clip=self.cfg.activation_exp_clip,
        )

        h_matrix = np.block(
            [
                [self.identity, -jacobian_task.T],
                [jacobian_task, np.zeros((3, 3), dtype=float)],
            ]
        )
        p_vector = np.concatenate((drift_feedback, -task_feedback))
        solver_output = self.solver.step(h_matrix, p_vector, lower, upper)
        theta_dot = solver_output["theta_dot"]

        return {
            "theta_dot": theta_dot,
            "theta_dot_raw": solver_output["theta_dot_raw"],
            "dual_next": solver_output["dual_next"],
            "position_error": position_error,
            "task_feedback": task_feedback,
            "task_residual": task_feedback - jacobian_task @ theta_dot,
            "drift_delta": drift_delta,
            "drift_feedback": drift_feedback,
            "inner_residual_norm": solver_output["residual_norm"],
            "inner_energy": solver_output["energy"],
        }


def create_history():
    return {
        "time_s": [],
        "actual_positions": [],
        "desired_positions": [],
        "position_errors": [],
        "joint_positions": [],
        "joint_drift": [],
        "inner_residual_norm": [],
    }


def record_history(history, tk, theta_current, current_pos, desired_pos, theta_reference, inner_residual_norm):
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
    history["inner_residual_norm"].append(float(inner_residual_norm))


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


def plot_inner_residual_figure(history, output_path, title_prefix):
    values = np.asarray(history["inner_residual_norm"], dtype=float)
    time_vec = np.asarray(history["time_s"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(time_vec, np.clip(values, 1e-14, None), color="#2e8b57", linewidth=2.0)
    ax.set_title(f"{title_prefix} KKT Residual Norm")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"$||H y + p||_2$")
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
    plot_inner_residual_figure(history, output_dir / f"{OUTPUT_STEM}_{table_label}_inner_residual.png", title_prefix)
    save_joint_recovery_table(theta_initial, theta_final, output_dir, table_label, title_prefix)


def summarize_run(label, history, theta_initial, theta_final, result_dir, port=None):
    position_errors = np.asarray(history["position_errors"], dtype=float)
    drift = np.asarray(history["joint_drift"], dtype=float)
    summary = {
        "label": label,
        "status": "ok",
        "mean_position_error_m": float(np.mean(position_errors)),
        "max_position_error_m": float(np.max(position_errors)),
        "final_position_error_m": float(position_errors[-1]),
        "max_abs_joint_drift_rad": float(np.max(np.abs(drift))),
        "final_joint_drift_norm_rad": float(np.linalg.norm(drift[-1])),
        "mean_inner_residual_norm": float(np.mean(np.asarray(history["inner_residual_norm"], dtype=float))),
        "max_inner_residual_norm": float(np.max(np.asarray(history["inner_residual_norm"], dtype=float))),
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


def run_offline_experiment(cfg, robot, output_dir):
    controller = Method1Controller(cfg)
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
        record_history(
            history,
            tk,
            theta_current,
            new_pos,
            desired_pos,
            theta_reference,
            step_result["inner_residual_norm"],
        )
    wall_clock = time.perf_counter() - t_start

    save_figures(history, output_dir, METHOD_NAME + " Offline", theta_reference, theta_current, "offline")
    summary = summarize_run("Method 1 DLCCZNN offline", history, theta_reference, theta_current, output_dir)
    summary["wall_clock_run_s"] = wall_clock
    (output_dir / f"{OUTPUT_STEM}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Integrated Method 1 DLCCZNN project runner.")
    parser.add_argument("--duration", type=float, default=10.0, help="Live duration in seconds.")
    parser.add_argument("--offline-duration", type=float, default=20.0, help="Offline duration in seconds.")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--task-gain", type=float, default=20.0)
    parser.add_argument("--mu-gain", type=float, default=0.1)
    parser.add_argument("--activation-power", type=float, default=0.9)
    parser.add_argument("--activation-exp-clip", type=float, default=2.0)
    parser.add_argument("--solver-substeps", type=int, default=20)
    parser.add_argument("--solver-gamma-gain", type=float, default=80.0)
    parser.add_argument("--solver-power", type=float, default=0.9)
    parser.add_argument("--solver-exp-clip", type=float, default=4.0)
    parser.add_argument("--solver-lambda", type=float, default=1e-8)
    parser.add_argument("--output-root", type=str, default=None, help="Optional custom result root directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Method1DLCCZNNConfig(
        duration=args.duration,
        offline_duration=args.offline_duration,
        tau=args.tau,
        task_gain=args.task_gain,
        mu_gain=args.mu_gain,
        activation_power=args.activation_power,
        activation_exp_clip=args.activation_exp_clip,
        solver_substeps=args.solver_substeps,
        solver_gamma_gain=args.solver_gamma_gain,
        solver_power=args.solver_power,
        solver_exp_clip=args.solver_exp_clip,
        solver_lambda=args.solver_lambda,
    )
    robot = UR3eKinematics()
    output_root = Path(args.output_root) if args.output_root else build_run_output_root(RESULTS_DIR / OUTPUT_STEM, args, "offline")
    output_root.mkdir(parents=True, exist_ok=True)
    save_run_config(output_root, args, "offline")

    print("Running offline experiment ...")
    offline_summary = run_offline_experiment(cfg, robot, output_root / "offline")
    print(json.dumps(offline_summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

