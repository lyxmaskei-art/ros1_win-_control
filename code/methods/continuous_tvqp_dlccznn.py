"""
Paper-aligned sampled TVQP controllers with DLCCZNN-style discrete updates.

This module keeps the sampled TVQP controller interface used by the standalone
offline and live experiment entries, while replacing the previous generic Euler
residual iteration with discrete update laws extracted from the user's DLCCZNN
paper and the later drift-free reformulation paper.

Current public methods:
- Method 1: classical drift-free shell + DLCCZNN solver
- Method 2: same shell + DLCCZNN solver with tuned outer settings
- Method 3a: sampled position/drift objective + DLCCZNN solver on Q z + c = 0

The implementation is intentionally explicit and keeps the state variables in
their physical primal-dual / joint-velocity coordinates so that the discrete
update law remains transparent.
"""

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = CURRENT_DIR.parent
CORE_DIR = CODE_ROOT / "core"
for candidate in (CURRENT_DIR, CORE_DIR):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import discrete_rmp_qp_common as common


class VariableTransformer:
    """
    Compatibility placeholder for older imports.

    The current paper-aligned DLCCZNN implementation works directly in the
    physical primal / dual coordinates instead of the previous alpha-space
    reparameterization. The helper is retained so existing imports do not fail.
    """

    @staticmethod
    def compute_params(lower, upper):
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        mid = 0.5 * (upper + lower)
        half_range = np.maximum(0.5 * (upper - lower), 1e-12)
        return mid, half_range

    @staticmethod
    def to_alpha(qdot, mid, half_range):
        qdot = np.asarray(qdot, dtype=float)
        normalized = (qdot - mid) / half_range
        normalized = np.clip(normalized, -1.0 + 1e-12, 1.0 - 1e-12)
        return np.arcsin(normalized)

    @staticmethod
    def to_qdot(alpha, mid, half_range):
        alpha = np.asarray(alpha, dtype=float)
        return mid + half_range * np.sin(alpha)

    @staticmethod
    def jacobian_diag(alpha, half_range):
        return half_range * np.cos(alpha)


@dataclass
class DLCCZNNSolverConfig:
    tau: float = 0.005
    gamma: float = 80.0
    activation_power: float = 0.9
    activation_exp_clip: float = 4.0
    # Retained only for backward compatibility with existing configs.
    # The strict paper-aligned discrete solver now uses one update per sample.
    substeps: int = 20
    lambda_reg: float = 1e-8
    residual_tol: float = 1e-10


def _scalar_dlccznn_activation(value, power, exp_clip):
    scalar = max(float(value), 0.0)
    return (scalar ** float(power)) * np.exp(min(scalar, float(exp_clip)))


class DLCCZNNSolver:
    """
    Generic discrete DLCCZNN helper for residual equations of the form R(z)=0.

    The update follows the paper-aligned low-complexity discrete structure:

        z_{k+1} = z_k
                  - tau * gamma * phi(v_k) * d_k / (||d_k||^2 + lambda)

    where v_k = 0.5 ||R(z_k)||^2 and d_k is a low-complexity correction
    direction, typically J_R(z_k)^T R(z_k).

    One control sample performs one discrete DLCCZNN update. The previous
    inner-substep loop was only a numerical stabilization device and does not
    belong to the strict paper-aligned sampled update law.
    """

    def __init__(self, size, config):
        self.size = int(size)
        self.cfg = config
        self.state = np.zeros(self.size, dtype=float)

    def reset(self):
        self.state = np.zeros(self.size, dtype=float)

    def set_state(self, y0):
        self.state = np.asarray(y0, dtype=float).copy()

    def step(self, residual_fn, direction_fn, project_fn=None):
        z = self.state.copy()
        z_raw = z.copy()
        residual = residual_fn(z)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > self.cfg.residual_tol:
            energy = 0.5 * float(residual @ residual)
            direction = np.asarray(direction_fn(z, residual), dtype=float)
            denom = float(direction @ direction) + float(self.cfg.lambda_reg)
            if denom <= 0.0:
                denom = 1e-12

            correction = self.cfg.gamma * _scalar_dlccznn_activation(
                energy,
                power=self.cfg.activation_power,
                exp_clip=self.cfg.activation_exp_clip,
            )
            z_raw = z - self.cfg.tau * correction * (direction / denom)
            z = z_raw.copy() if project_fn is None else project_fn(z_raw)

            if not np.all(np.isfinite(z)):
                raise FloatingPointError("DLCCZNN solver diverged; reduce gain or step size.")

        self.state = z.copy()
        final_residual = residual_fn(z)
        return {
            'state': z.copy(),
            'state_raw': z_raw.copy(),
            'residual': final_residual.copy(),
            'residual_norm': float(np.linalg.norm(final_residual)),
            'energy': 0.5 * float(final_residual @ final_residual),
        }


def _clip_joint_step(theta_current, tau, qdot_cmd, theta_lower, theta_upper):
    return np.clip(
        np.asarray(theta_current, dtype=float) + float(tau) * np.asarray(qdot_cmd, dtype=float),
        np.asarray(theta_lower, dtype=float),
        np.asarray(theta_upper, dtype=float),
    )


def tuned_method2_task_gain(tau):
    tau = float(tau)
    if tau <= 0.005:
        return 220.0
    if tau <= 0.02:
        return 80.0
    return 20.0


def tuned_method3a_task_gain(tau):
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
class Method1Config:
    tau: float = 0.005
    task_gain: float = 20.0
    mu_gain: float = 0.1
    activation_power: float = 0.9
    activation_exp_clip: float = 2.0
    eta: float = 0.9
    theta_dot_limit: float = 2.0
    theta_lower: np.ndarray = field(
        default_factory=lambda: -2.0 * np.pi * np.ones(6, dtype=float)
    )
    theta_upper: np.ndarray = field(
        default_factory=lambda: 2.0 * np.pi * np.ones(6, dtype=float)
    )
    solver_substeps: int = 20
    solver_gamma_gain: float = 80.0
    solver_power: float = 0.9
    solver_exp_clip: float = 4.0
    solver_lambda: float = 1e-8
    solver_tol: float = 1e-10


class Method1Controller:
    """
    Method 1: classical drift-free shell solved by a paper-aligned DLCCZNN law.

    The outer shell remains:
        min 0.5 qdot^T I qdot + c_hat^T qdot
        s.t. J qdot = b_fb,
             xi^- <= qdot <= xi^+

    The inner solver now follows a discrete DLCCZNN-style residual update on
    the KKT root H y + p = 0 instead of the previous generic Euler iteration.
    """

    def __init__(self, robot, config):
        self.robot = robot
        self.cfg = config
        self.n_joints = robot.num_joints
        self.task_dim = 3
        self.identity = np.eye(self.n_joints, dtype=float)
        self.theta_initial = None
        self.solver = DLCCZNNSolver(
            size=self.n_joints + self.task_dim,
            config=DLCCZNNSolverConfig(
                tau=config.tau,
                gamma=config.solver_gamma_gain,
                activation_power=config.solver_power,
                activation_exp_clip=config.solver_exp_clip,
                substeps=config.solver_substeps,
                lambda_reg=config.solver_lambda,
                residual_tol=config.solver_tol,
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
        jacobian_task = self.robot.jacobian(theta_current)[:self.task_dim, :]

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
        y_final = solver_output['state']
        y_raw = solver_output['state_raw']

        qdot_cmd = y_final[: self.n_joints].copy()
        qdot_raw = y_raw[: self.n_joints].copy()
        dual_next = y_final[self.n_joints :].copy()
        theta_next = _clip_joint_step(
            theta_current,
            self.cfg.tau,
            qdot_cmd,
            self.cfg.theta_lower,
            self.cfg.theta_upper,
        )

        return {
            'theta_next': theta_next,
            'theta_dot_next': qdot_cmd,
            'theta_dot_raw': qdot_raw,
            'dual_next': dual_next,
            'current_pos': current_pos,
            'desired_pos': desired_position,
            'desired_vel': desired_velocity,
            'tracking_error': position_error,
            'task_command': task_command,
            'task_residual': task_command - jacobian_task @ qdot_cmd,
            'drift_delta': drift_delta,
            'drift_feedback': drift_feedback,
            'inner_residual_norm': solver_output['residual_norm'],
            'inner_energy': solver_output['energy'],
            'xi_minus': lower,
            'xi_plus': upper,
        }


@dataclass
class Method2Config:
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
    solver_substeps: int = 80
    solver_gamma_gain: float = 4608.0
    solver_power: float = 0.85
    solver_exp_clip: float = 4.0
    solver_lambda: float = 1e-8
    solver_tol: float = 1e-10

    def __post_init__(self):
        if self.task_gain is None:
            self.task_gain = tuned_method2_task_gain(self.tau)


class Method2Controller:
    """
    Method 2: same outer drift-free shell, but run under a tuned DLCCZNN solver.

    In the original paper Method 2 differs from Method 1 mainly through the
    solver layer. After replacing PDNN with DLCCZNN, the structural difference
    is retained through its tuned outer parameters and DLCCZNN hyperparameters.
    """

    def __init__(self, robot, config):
        self.robot = robot
        self.cfg = config
        self.n_joints = robot.num_joints
        self.task_dim = 3
        self.identity = np.eye(self.n_joints, dtype=float)
        self.theta_initial = None
        self.solver = DLCCZNNSolver(
            size=self.n_joints + self.task_dim,
            config=DLCCZNNSolverConfig(
                tau=config.tau,
                gamma=config.solver_gamma_gain,
                activation_power=config.solver_power,
                activation_exp_clip=config.solver_exp_clip,
                substeps=config.solver_substeps,
                lambda_reg=config.solver_lambda,
                residual_tol=config.solver_tol,
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
        jacobian_task = self.robot.jacobian(theta_current)[:self.task_dim, :]

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
        y_final = solver_output['state']
        y_raw = solver_output['state_raw']

        qdot_cmd = y_final[: self.n_joints].copy()
        qdot_raw = y_raw[: self.n_joints].copy()
        dual_next = y_final[self.n_joints :].copy()
        theta_next = _clip_joint_step(
            theta_current,
            self.cfg.tau,
            qdot_cmd,
            self.cfg.theta_lower,
            self.cfg.theta_upper,
        )

        return {
            'theta_next': theta_next,
            'theta_dot_next': qdot_cmd,
            'theta_dot_raw': qdot_raw,
            'dual_next': dual_next,
            'current_pos': current_pos,
            'desired_pos': desired_position,
            'desired_vel': desired_velocity,
            'tracking_error': position_error,
            'task_command': task_command,
            'task_residual': task_command - jacobian_task @ qdot_cmd,
            'drift_delta': drift_delta,
            'drift_feedback': drift_feedback,
            'inner_residual_norm': solver_output['residual_norm'],
            'inner_energy': solver_output['energy'],
            'xi_minus': lower,
            'xi_plus': upper,
        }


@dataclass
class Method3aConfig:
    tau: float = 0.005
    task_gain: float | None = None
    position_weight: float = 10000.0
    drift_weight: float = 0.001
    regularization_gain: float = 1e-9
    eta: float = 0.9
    theta_dot_limit: float = 2.0
    theta_lower: np.ndarray = field(
        default_factory=lambda: -2.0 * np.pi * np.ones(6, dtype=float)
    )
    theta_upper: np.ndarray = field(
        default_factory=lambda: 2.0 * np.pi * np.ones(6, dtype=float)
    )
    solver_substeps: int = 20
    solver_gamma_gain: float = 80.0
    solver_power: float = 0.9
    solver_exp_clip: float = 4.0
    solver_lambda: float = 1e-8
    solver_tol: float = 1e-10

    def __post_init__(self):
        if self.task_gain is None:
            self.task_gain = tuned_method3a_task_gain(self.tau)


class Method3aController:
    """
    Method 3a: sampled position/drift objective solved by a paper-aligned
    DLCCZNN discrete root solver on Q_k z + c_k = 0.
    """

    def __init__(self, robot, config):
        self.robot = robot
        self.cfg = config
        self.n_joints = robot.num_joints
        self.identity = np.eye(self.n_joints, dtype=float)
        self.theta_initial = None
        self.solver = DLCCZNNSolver(
            size=self.n_joints,
            config=DLCCZNNSolverConfig(
                tau=config.tau,
                gamma=config.solver_gamma_gain,
                activation_power=config.solver_power,
                activation_exp_clip=config.solver_exp_clip,
                substeps=config.solver_substeps,
                lambda_reg=config.solver_lambda,
                residual_tol=config.solver_tol,
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
        jacobian_task = self.robot.jacobian(theta_current)[:3, :]

        position_error = current_pos - desired_position
        drift_delta = theta_current - self.theta_initial

        if use_feedback:
            task_command = desired_velocity - self.cfg.task_gain * position_error
            position_drive = ((1.0 / self.cfg.tau) + self.cfg.task_gain) * position_error - desired_velocity
        else:
            task_command = desired_velocity.copy()
            position_drive = (1.0 / self.cfg.tau) * position_error - desired_velocity

        q_matrix = (
            self.cfg.position_weight * (jacobian_task.T @ jacobian_task)
            + self.cfg.drift_weight * self.identity
            + self.cfg.regularization_gain * self.identity
        )
        q_vector = (
            self.cfg.position_weight * (jacobian_task.T @ position_drive)
            + self.cfg.drift_weight * (drift_delta / self.cfg.tau)
        )

        lower, upper = self._compute_velocity_bounds(theta_current)

        def residual_fn(z):
            return q_matrix @ z + q_vector

        def direction_fn(_, residual):
            return q_matrix.T @ residual

        def project_fn(z_raw):
            return np.clip(np.asarray(z_raw, dtype=float), lower, upper)

        solver_output = self.solver.step(residual_fn, direction_fn, project_fn)
        qdot_cmd = solver_output['state'].copy()
        qdot_raw = solver_output['state_raw'].copy()
        theta_next = _clip_joint_step(
            theta_current,
            self.cfg.tau,
            qdot_cmd,
            self.cfg.theta_lower,
            self.cfg.theta_upper,
        )

        return {
            'theta_next': theta_next,
            'theta_dot_next': qdot_cmd,
            'theta_dot_raw': qdot_raw,
            'dual_next': np.zeros(3, dtype=float),
            'current_pos': current_pos,
            'desired_pos': desired_position,
            'desired_vel': desired_velocity,
            'tracking_error': position_error,
            'task_command': task_command,
            'task_residual': task_command - jacobian_task @ qdot_cmd,
            'drift_delta': drift_delta,
            'q_matrix': q_matrix,
            'q_vector': q_vector,
            'inner_residual_norm': solver_output['residual_norm'],
            'inner_energy': solver_output['energy'],
            'xi_minus': lower,
            'xi_plus': upper,
        }


@dataclass
class Method3bConfig:
    tau: float = 0.005
    task_gain: float | None = None
    position_weight: float = 10000.0
    drift_weight: float = 0.001
    regularization_gain: float = 1e-9
    eta: float = 0.9
    theta_dot_limit: float = 2.0
    theta_lower: np.ndarray = field(
        default_factory=lambda: -2.0 * np.pi * np.ones(6, dtype=float)
    )
    theta_upper: np.ndarray = field(
        default_factory=lambda: 2.0 * np.pi * np.ones(6, dtype=float)
    )
    solver_substeps: int = 20
    solver_gamma_gain: float = 80.0
    solver_power: float = 0.9
    solver_exp_clip: float = 4.0
    solver_lambda: float = 1e-8
    solver_tol: float = 1e-10

    def __post_init__(self):
        if self.task_gain is None:
            self.task_gain = tuned_method3a_task_gain(self.tau)


class Method3bController(Method3aController):
    """
    Compatibility alias.

    The old exploratory Method 3b is retired in favor of the paper-aligned
    DLCCZNN treatment of the Method 3a sampled objective. Keeping this class
    avoids breaking historical imports.
    """

    def __init__(self, robot, config):
        lifted = Method3aConfig(
            tau=config.tau,
            task_gain=config.task_gain,
            position_weight=config.position_weight,
            drift_weight=config.drift_weight,
            regularization_gain=config.regularization_gain,
            eta=config.eta,
            theta_dot_limit=config.theta_dot_limit,
            theta_lower=np.asarray(config.theta_lower, dtype=float).copy(),
            theta_upper=np.asarray(config.theta_upper, dtype=float).copy(),
            solver_substeps=config.solver_substeps,
            solver_gamma_gain=config.solver_gamma_gain,
            solver_power=config.solver_power,
            solver_exp_clip=config.solver_exp_clip,
            solver_lambda=config.solver_lambda,
            solver_tol=config.solver_tol,
        )
        super().__init__(robot, lifted)


def build_method1(robot, settings=None):
    cfg = Method1Config()
    if settings is not None:
        cfg.tau = float(getattr(settings, 'tau', cfg.tau))
        cfg.task_gain = float(getattr(settings, 'task_gain', cfg.task_gain))
        if hasattr(settings, 'theta_lower'):
            cfg.theta_lower = np.asarray(settings.theta_lower, dtype=float).copy()
        if hasattr(settings, 'theta_upper'):
            cfg.theta_upper = np.asarray(settings.theta_upper, dtype=float).copy()
    return Method1Controller(robot, cfg)


def build_method2(robot, settings=None):
    cfg = Method2Config()
    if settings is not None:
        cfg.tau = float(getattr(settings, 'tau', cfg.tau))
        cfg.task_gain = tuned_method2_task_gain(cfg.tau)
        if hasattr(settings, 'theta_lower'):
            cfg.theta_lower = np.asarray(settings.theta_lower, dtype=float).copy()
        if hasattr(settings, 'theta_upper'):
            cfg.theta_upper = np.asarray(settings.theta_upper, dtype=float).copy()
    else:
        cfg.task_gain = tuned_method2_task_gain(cfg.tau)
    return Method2Controller(robot, cfg)


def build_method3a(robot, settings=None):
    cfg = Method3aConfig()
    if settings is not None:
        cfg.tau = float(getattr(settings, 'tau', cfg.tau))
        cfg.task_gain = tuned_method3a_task_gain(cfg.tau)
        if hasattr(settings, 'theta_lower'):
            cfg.theta_lower = np.asarray(settings.theta_lower, dtype=float).copy()
        if hasattr(settings, 'theta_upper'):
            cfg.theta_upper = np.asarray(settings.theta_upper, dtype=float).copy()
    else:
        cfg.task_gain = tuned_method3a_task_gain(cfg.tau)
    return Method3aController(robot, cfg)


def build_method3b(robot, settings=None):
    cfg = Method3bConfig()
    if settings is not None:
        cfg.tau = float(getattr(settings, 'tau', cfg.tau))
        cfg.task_gain = tuned_method3a_task_gain(cfg.tau)
        if hasattr(settings, 'theta_lower'):
            cfg.theta_lower = np.asarray(settings.theta_lower, dtype=float).copy()
        if hasattr(settings, 'theta_upper'):
            cfg.theta_upper = np.asarray(settings.theta_upper, dtype=float).copy()
    else:
        cfg.task_gain = tuned_method3a_task_gain(cfg.tau)
    return Method3bController(robot, cfg)


METHOD_BUILDERS = {
    'method1': build_method1,
    'method2': build_method2,
    'method3a': build_method3a,
    'method3b': build_method3b,
}
