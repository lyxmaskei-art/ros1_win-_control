"""
离散 RMP-QP / drift-free 集成实现。

本文件的角色不是只给一个“能跑”的控制器，而是把三类思路放在同一套
机器人模型、同一套误差统计、同一套约束处理框架下做可复现实验：

1. Method 1
   沿用书中 / WAY1 主线，把 drift-free 项作为 QP 的线性项。
2. Method 2
   保留 S-LVI-PDNN 风格，通过残差能量 shaping 改变求解器动态。
3. Method 3a / 3b
   按用户要求，把“位置误差 / 漂移误差统一成标量二范数”后的离散目标
   直接落成关节速度层优化问题。

注意：
- 这里的 Method 3a / 3b 已经不是早期版本那种“把线性 drift 项硬塞回去”
  的写法，而是显式按照离散标量二范数目标构造。
- 对 Method 3a / 3b，直接解析 / KKT 求解比继续套 S-LVI-PDNN 更稳定，
  原因是它们的 Hessian 和约束结构已经足够简单，继续走神经动力学只会
  引入额外离散化误差。
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


METHOD_LABELS = {
    'method1_sig_rmp_qp': 'Method 1: sig-type drift QP',
    'method2_scheme_b_positive': 'Method 2: Scheme-B positive dynamics',
    'method3a_position_and_drift': 'Method 3a: soft task + drift objective',
    'method3b_drift_free_hard_constraint': 'Method 3b: drift objective + pure hard task constraint',
}

METHOD_GROUPS = {
    'method1': ['method1_sig_rmp_qp'],
    'method2': ['method2_scheme_b_positive'],
    'method3': ['method3a_position_and_drift', 'method3b_drift_free_hard_constraint'],
}


@dataclass
class IntegratedMethodConfig:
    """
    集成控制器参数。

    这里把“任务层参数”“drift-free 参数”“QP / PDNN 数值参数”统一放在一起，
    方便做 sweep 和统一记录。
    """
    tau: float = 0.005
    gamma_gain: float = 80.0
    eta: float = 0.9
    task_gain: float = 20.0
    weight_gain: float = 1.0
    mu_gain: float = 0.1
    regularization_gain: float = 1e-8
    activation_power: float = 0.9
    activation_exp_clip: float = 2.0
    position_weight: float = 1.0
    drift_weight: float = 1.0
    solver_substeps: int = 1
    scheme_b_beta_gain: float = 0.04
    scheme_b_lambda_reg: float = 1e-5
    tracking_energy_clip: float = 5.0
    system_energy_clip: float = 5.0
    startup_speed_floor: float = 1e-4
    startup_velocity_floor: float = 1e-4
    startup_hold_steps: int = 60
    startup_gain: float = 0.6
    startup_lambda: float = 1e-9
    theta_lower: np.ndarray = field(default_factory=lambda: -common.DEFAULT_THETA_LIMIT.copy())
    theta_upper: np.ndarray = field(default_factory=lambda: common.DEFAULT_THETA_LIMIT.copy())
    theta_dot_lower: np.ndarray = field(default_factory=lambda: -common.DEFAULT_THETA_DOT_LIMIT.copy())
    theta_dot_upper: np.ndarray = field(default_factory=lambda: common.DEFAULT_THETA_DOT_LIMIT.copy())


class IntegratedControllerBase:
    """所有离散方法共享的机器人层基类。"""

    def __init__(self, robot, config, method_name, task_dim=3):
        self.robot = robot
        self.cfg = config
        self.method_name = str(method_name)
        self.task_dim = int(task_dim)
        self.identity = np.eye(self.robot.num_joints, dtype=float)
        self.weight_matrix = self.cfg.weight_gain * self.identity
        self.theta_initial = None

    def reset(self, theta_initial):
        self.theta_initial = np.asarray(theta_initial, dtype=float).copy()

    def update_joint_limits(self, theta_lower, theta_upper):
        self.cfg.theta_lower = np.asarray(theta_lower, dtype=float).copy()
        self.cfg.theta_upper = np.asarray(theta_upper, dtype=float).copy()

    def compute_velocity_bounds(self, theta_current):
        return common.compute_dynamic_velocity_bounds(
            theta_current=theta_current,
            theta_lower=self.cfg.theta_lower,
            theta_upper=self.cfg.theta_upper,
            theta_dot_lower=self.cfg.theta_dot_lower,
            theta_dot_upper=self.cfg.theta_dot_upper,
            eta=self.cfg.eta,
            tau=self.cfg.tau,
        )

    def sig_drift_feedback(self, drift_delta):
        return self.cfg.mu_gain * common.sig_exp_activation(
            drift_delta,
            power=self.cfg.activation_power,
            exp_clip=self.cfg.activation_exp_clip,
        )

    def positive_activation(self, scalar_value):
        return common.positive_exp_activation(
            scalar_value,
            power=self.cfg.activation_power,
            exp_clip=self.cfg.activation_exp_clip,
        )

    def build_step_data(self, theta_current, desired_position, desired_velocity):
        """
        构造一步控制里各方法都要用到的公共量。

        变量说明：
        - position_error = x(q_k) - x_d(k)
          这里统一采用“当前减期望”的符号，后续所有公式都围绕这一定义写。
        - task_feedback = rdot_d - k_p * position_error
          这是传统任务反馈形式，对 Method 1 / 2 直接作为等式右端使用。
        - task_pure = rdot_d
          这是不带任何误差补偿的硬任务速度，Method 3b 按用户要求必须用它。
        - drift_delta = q_k - q_0
          周期末漂移误差的基础量，也是所有 drift-free 项的源头。
        """
        theta_current = np.asarray(theta_current, dtype=float)
        desired_position = np.asarray(desired_position, dtype=float)
        desired_velocity = np.asarray(desired_velocity, dtype=float)

        current_pos = self.robot.forward_kinematics(theta_current)[:3, 3]
        jacobian_task = self.robot.jacobian(theta_current)[: self.task_dim, :]
        position_error = current_pos - desired_position
        task_feedback = desired_velocity - self.cfg.task_gain * position_error
        task_pure = desired_velocity.copy()
        drift_delta = theta_current - self.theta_initial

        return {
            'theta_current': theta_current,
            'desired_position': desired_position,
            'desired_velocity': desired_velocity,
            'current_pos': current_pos,
            'jacobian_task': jacobian_task,
            'position_error': position_error,
            'task_feedback': task_feedback,
            'task_pure': task_pure,
            'drift_delta': drift_delta,
            'drift_energy': 0.5 * float(drift_delta @ drift_delta),
        }


class QPControllerBase(IntegratedControllerBase):
    """
    需要“先建 QP / LVI，再求解”的方法基类。

    Method 1 继续走简化 S-LVI-PDNN；
    Method 3a / 3b 虽然也复用这个外壳，但内部会覆写 solve_problem，
    直接用解析解 / KKT 解，避免额外的神经网络离散误差。
    """

    def __init__(self, robot, config, method_name, task_dim=3):
        super().__init__(robot, config, method_name, task_dim=task_dim)
        solver_substeps = max(int(self.cfg.solver_substeps), 1)
        self.solver = common.SimplifiedLVIPDNNSolver(
            num_vars=self.robot.num_joints,
            tau=self.cfg.tau / float(solver_substeps),
            gamma_gain=self.cfg.gamma_gain,
        )

    def reset(self, theta_initial):
        super().reset(theta_initial)
        self.solver.reset()

    def build_problem(self, step_data, lower, upper):
        raise NotImplementedError

    def solve_problem(self, problem):
        solver_output = None
        for _ in range(max(int(self.cfg.solver_substeps), 1)):
            solver_output = self.solver.step(problem)
        return solver_output

    def step(self, theta_current, desired_position, desired_velocity):
        theta_current = np.asarray(theta_current, dtype=float)
        if self.theta_initial is None:
            self.reset(theta_current)

        step_data = self.build_step_data(theta_current, desired_position, desired_velocity)
        lower, upper = self.compute_velocity_bounds(theta_current)
        problem, context = self.build_problem(step_data, lower, upper)
        solver_output = self.solve_problem(problem)

        theta_dot_next = np.asarray(solver_output['solution'], dtype=float)
        theta_next = np.clip(
            theta_current + self.cfg.tau * theta_dot_next,
            self.cfg.theta_lower,
            self.cfg.theta_upper,
        )

        residual_target = np.asarray(
            context.get('residual_target', step_data['task_feedback']),
            dtype=float,
        )
        task_residual = residual_target - step_data['jacobian_task'] @ theta_dot_next

        if not (np.all(np.isfinite(theta_next)) and np.all(np.isfinite(theta_dot_next))):
            raise FloatingPointError(f'{self.method_name} diverged; reduce gains or activation clip.')

        return {
            'theta_next': theta_next,
            'theta_dot_next': theta_dot_next,
            'theta_dot_raw': np.asarray(solver_output['primal_raw'], dtype=float),
            'dual_next': np.asarray(solver_output['dual_next'], dtype=float),
            'current_pos': step_data['current_pos'],
            'desired_pos': step_data['desired_position'],
            'desired_vel': step_data['desired_velocity'],
            'tracking_error': step_data['position_error'],
            'task_command': np.asarray(context.get('task_command', residual_target), dtype=float),
            'task_residual': task_residual,
            'drift_delta': step_data['drift_delta'],
            'drift_energy': step_data['drift_energy'],
            'c_hat': np.asarray(
                context.get('c_hat', np.zeros(self.robot.num_joints, dtype=float)),
                dtype=float,
            ),
            'q_matrix': np.asarray(problem.q_matrix, dtype=float),
            'q_vector': np.asarray(problem.q_vector, dtype=float),
            'xi_minus': lower,
            'xi_plus': upper,
            'lvi_input': np.asarray(solver_output['lvi_input'], dtype=float),
            'projected_input': np.asarray(solver_output['projected'], dtype=float),
            'v_dot': np.asarray(solver_output['v_dot'], dtype=float),
        }


class Method1SigRMPQPController(QPControllerBase):
    def __init__(self, robot, config):
        super().__init__(robot, config, method_name='method1_sig_rmp_qp')

    def build_problem(self, step_data, lower, upper):
        # Method 1 的漂移抑制完全体现在 QP 线性项 c_hat 中。
        drift_feedback = self.sig_drift_feedback(step_data['drift_delta'])
        problem = common.QPProblem(
            q_matrix=self.weight_matrix,
            q_vector=drift_feedback,
            aeq_matrix=step_data['jacobian_task'],
            beq_vector=step_data['task_feedback'],
            lower=lower,
            upper=upper,
        )
        return problem, {
            'task_command': step_data['task_feedback'],
            'residual_target': step_data['task_feedback'],
            'c_hat': drift_feedback,
        }


class Method2SchemeBController(IntegratedControllerBase):
    """
    Method 2: Scheme-B / 能量 shaping 版本。

    与 Method 1 的根本差异不在 QP 目标本身，而在求解器动态：
    - Method 1: 只把 drift-free 当作 QP 的线性偏置项。
    - Method 2: 直接对 LVI 残差能量做非线性增益 shaping。
    """

    def __init__(self, robot, config):
        super().__init__(robot, config, method_name='method2_scheme_b_positive')
        self.state = np.zeros(self.robot.num_joints + self.task_dim, dtype=float)

    def reset(self, theta_initial):
        super().reset(theta_initial)
        self.state.fill(0.0)

    def step(self, theta_current, desired_position, desired_velocity):
        theta_current = np.asarray(theta_current, dtype=float)
        if self.theta_initial is None:
            self.reset(theta_current)

        step_data = self.build_step_data(theta_current, desired_position, desired_velocity)
        lower, upper = self.compute_velocity_bounds(theta_current)
        jacobian_task = step_data['jacobian_task']
        drift_feedback = self.sig_drift_feedback(step_data['drift_delta'])

        # 先把等式约束 QP 写成 KKT / LVI 形式：
        # H y + p = 0, 其中 y = [q_dot; lambda]
        upper_block = np.hstack((self.weight_matrix, -jacobian_task.T))
        lower_block = np.hstack((jacobian_task, np.zeros((self.task_dim, self.task_dim), dtype=float)))
        h_matrix = np.vstack((upper_block, lower_block))
        p_vector = np.concatenate((drift_feedback, -step_data['task_feedback']))

        substeps = max(int(self.cfg.solver_substeps), 1)
        solver_dt = self.cfg.tau / float(substeps)
        primal_raw = self.state[: self.robot.num_joints].copy()
        theta_dot_raw = primal_raw.copy()
        lvi_input = self.state.copy()
        projected = self.state.copy()
        residual = np.zeros_like(self.state)
        nonlinear_gain = 0.0

        for _ in range(substeps):
            lvi_input = self.state - (h_matrix @ self.state + p_vector)
            projected = lvi_input.copy()
            projected[: self.robot.num_joints] = np.clip(projected[: self.robot.num_joints], lower, upper)
            residual = projected - self.state
            # Scheme-B 的关键点：不是只改 q_dot，而是对“求解器残差能量”
            # v_q = 0.5 ||P(y - (Hy+p)) - y||^2 做非线性增益 shaping。
            residual_energy = min(0.5 * float(residual.T @ residual), self.cfg.system_energy_clip)
            nonlinear_gain = 1.0 + self.cfg.mu_gain * self.positive_activation(residual_energy)

            # Scheme-B 的核心不是改 QP，而是根据求解器残差能量自适应放大更新速度。
            state_dot = self.cfg.gamma_gain * nonlinear_gain * residual
            theta_dot_raw = self.state[: self.robot.num_joints] + solver_dt * state_dot[: self.robot.num_joints]
            primal_raw = theta_dot_raw.copy()
            primal_next = np.clip(primal_raw, lower, upper)
            dual_next = self.state[self.robot.num_joints :] + solver_dt * state_dot[self.robot.num_joints :]
            self.state = np.concatenate((primal_next, dual_next))

        theta_dot_cmd = self.state[: self.robot.num_joints].copy()
        dual_next = self.state[self.robot.num_joints :].copy()
        theta_next = np.clip(
            theta_current + self.cfg.tau * theta_dot_cmd,
            self.cfg.theta_lower,
            self.cfg.theta_upper,
        )

        if not (
            np.all(np.isfinite(theta_next))
            and np.all(np.isfinite(theta_dot_cmd))
            and np.all(np.isfinite(dual_next))
        ):
            raise FloatingPointError(f'{self.method_name} diverged; reduce gains or activation clip.')

        return {
            'theta_next': theta_next,
            'theta_dot_next': theta_dot_cmd,
            'theta_dot_raw': theta_dot_raw,
            'dual_next': dual_next,
            'current_pos': step_data['current_pos'],
            'desired_pos': step_data['desired_position'],
            'desired_vel': step_data['desired_velocity'],
            'tracking_error': step_data['position_error'],
            'task_command': step_data['task_feedback'],
            'task_residual': step_data['task_feedback'] - jacobian_task @ theta_dot_cmd,
            'drift_delta': step_data['drift_delta'],
            'drift_energy': step_data['drift_energy'],
            'c_hat': drift_feedback,
            'q_matrix': h_matrix,
            'q_vector': p_vector,
            'xi_minus': lower,
            'xi_plus': upper,
            'lvi_input': lvi_input,
            'projected_input': projected,
            'v_dot': self.cfg.gamma_gain * nonlinear_gain * residual,
        }


class Method3aPositionDriftController(QPControllerBase):
    """
    Method 3a: 位置误差 + 漂移误差同时进入离散标量二范数目标。

    这里已经按离散时间写法构造目标，而不是连续时间形式照搬到离散控制。
    """

    def __init__(self, robot, config):
        super().__init__(robot, config, method_name='method3a_position_and_drift')

    def solve_problem(self, problem):
        # Method 3a 的目标没有硬等式约束，因此直接解 Q qdot + c = 0 最稳。
        q_matrix = 0.5 * (np.asarray(problem.q_matrix, dtype=float) + np.asarray(problem.q_matrix, dtype=float).T)
        q_vector = np.asarray(problem.q_vector, dtype=float).reshape(-1)
        lower = np.asarray(problem.lower, dtype=float).reshape(-1)
        upper = np.asarray(problem.upper, dtype=float).reshape(-1)

        primal_raw = -np.linalg.solve(q_matrix, q_vector)
        primal_next = np.clip(primal_raw, lower, upper)
        return {
            'solution': primal_next,
            'primal_raw': primal_raw,
            'dual_next': np.zeros(0, dtype=float),
            'lvi_input': primal_raw.copy(),
            'projected': primal_next.copy(),
            'v_dot': np.zeros_like(primal_next),
        }

    def build_problem(self, step_data, lower, upper):
        tau = self.cfg.tau
        # Discrete scalarized objective:
        # min ||e_p(k) + tau * (J qdot - rdot + k_p e_p(k))||_2^2 + ||e_d(k) + tau * qdot||_2^2
        # Up to a positive scalar factor, this yields the QP below.
        #
        # 其中：
        # - e_p(k) = x(q_k) - x_d(k)
        # - e_d(k) = q_k - q_0
        # - tau * (J qdot - rdot + k_p e_p) 是一步离散后的任务误差传播项
        # - tau * qdot 是一步离散后的关节漂移传播项
        #
        # 展开后得到标准二次型：
        #   min 0.5 qdot^T Q qdot + c^T qdot
        # 这里省略不影响最优解的正比例因子，直接保留最稳定的 Q / c 形式。
        position_drive = ((1.0 / tau) + self.cfg.task_gain) * step_data['position_error'] - step_data['desired_velocity']
        # Q 来自两个二范数目标展开后的二次项，c 来自对应的一次项。
        q_matrix = (
            self.cfg.position_weight * (step_data['jacobian_task'].T @ step_data['jacobian_task'])
            + self.cfg.drift_weight * self.weight_matrix
            + self.cfg.regularization_gain * self.identity
        )
        q_vector = (
            self.cfg.position_weight * (step_data['jacobian_task'].T @ position_drive)
            + self.cfg.drift_weight * (step_data['drift_delta'] / tau)
        )
        problem = common.QPProblem(
            q_matrix=q_matrix,
            q_vector=q_vector,
            aeq_matrix=np.zeros((0, self.robot.num_joints), dtype=float),
            beq_vector=np.zeros(0, dtype=float),
            lower=lower,
            upper=upper,
        )
        return problem, {
            'task_command': step_data['desired_velocity'],
            'residual_target': step_data['desired_velocity'],
            'c_hat': step_data['drift_delta'] / tau,
        }


class Method3bHardConstraintController(QPControllerBase):
    """
    Method 3b: 只最小化漂移误差，任务速度作为硬约束。

    按用户要求：
    - 约束端必须是 J qdot = rdot_d
    - 不能添加任何 +lambda * e 形式的反馈补偿
    """

    def __init__(self, robot, config):
        super().__init__(robot, config, method_name='method3b_drift_free_hard_constraint')

    def solve_problem(self, problem):
        # Method 3b 保留原始硬约束 J qdot = rdot_d，因此直接解 KKT 方程组。
        q_matrix = 0.5 * (np.asarray(problem.q_matrix, dtype=float) + np.asarray(problem.q_matrix, dtype=float).T)
        q_vector = np.asarray(problem.q_vector, dtype=float).reshape(-1)
        aeq_matrix = np.asarray(problem.aeq_matrix, dtype=float)
        beq_vector = np.asarray(problem.beq_vector, dtype=float).reshape(-1)
        lower = np.asarray(problem.lower, dtype=float).reshape(-1)
        upper = np.asarray(problem.upper, dtype=float).reshape(-1)

        if aeq_matrix.ndim == 1:
            aeq_matrix = aeq_matrix.reshape(1, -1)
        kkt_matrix = np.block([
            [q_matrix, aeq_matrix.T],
            [aeq_matrix, np.zeros((aeq_matrix.shape[0], aeq_matrix.shape[0]), dtype=float)],
        ])
        kkt_rhs = np.concatenate((-q_vector, beq_vector))
        kkt_solution = np.linalg.solve(kkt_matrix, kkt_rhs)
        primal_raw = kkt_solution[: self.robot.num_joints]
        dual_next = kkt_solution[self.robot.num_joints :]
        primal_next = np.clip(primal_raw, lower, upper)
        return {
            'solution': primal_next,
            'primal_raw': primal_raw,
            'dual_next': dual_next,
            'lvi_input': kkt_solution.copy(),
            'projected': np.concatenate((primal_next, dual_next)),
            'v_dot': np.zeros_like(np.concatenate((primal_next, dual_next))),
        }

    def build_problem(self, step_data, lower, upper):
        tau = self.cfg.tau
        problem = common.QPProblem(
            q_matrix=self.cfg.drift_weight * self.weight_matrix + self.cfg.regularization_gain * self.identity,
            q_vector=self.cfg.drift_weight * (step_data['drift_delta'] / tau),
            aeq_matrix=step_data['jacobian_task'],
            beq_vector=step_data['task_pure'],
            lower=lower,
            upper=upper,
        )
        return problem, {
            'task_command': step_data['task_pure'],
            'residual_target': step_data['task_pure'],
            'c_hat': step_data['drift_delta'] / tau,
        }


def velocity_limits(settings, robot):
    """把统一的关节速度上限扩展成每个关节一份的上下界向量。"""
    bound = settings.theta_dot_limit * np.ones(robot.num_joints, dtype=float)
    return -bound, bound


def tuned_method2_task_gain(tau):
    """
    按步长经验选取 Method 2 的任务反馈增益。

    原因：
    - tau 越小，离散化越精细，允许更大的 task_gain。
    - tau 越大，同样的增益更容易把闭环推到振荡边界。
    """
    tau = float(tau)
    if tau <= 0.005:
        return 160.0
    if tau <= 0.02:
        return 40.0
    return 20.0


def tuned_method3a_task_gain(tau):
    """
    按步长经验选取 Method 3a 的稳定 task_gain。

    Method 3a 对 task_gain 极其敏感：
    - tau=0.005 时，198 左右是甜点区；
    - 再往上到 202 左右就会突然跳出稳定域。
    因此这里采用分段经验值，而不是简单线性缩放。
    """
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


def make_config(settings, robot, **overrides):
    """把实验层参数映射成控制器内部配置，并允许方法级覆写。"""
    theta_dot_lower, theta_dot_upper = velocity_limits(settings, robot)
    base = dict(
        tau=settings.tau,
        gamma_gain=80.0,
        eta=settings.eta,
        task_gain=settings.task_gain,
        weight_gain=1.0,
        mu_gain=0.1,
        regularization_gain=1e-8,
        activation_power=0.9,
        activation_exp_clip=2.0,
        position_weight=1.0,
        drift_weight=1.0,
        solver_substeps=1,
        scheme_b_beta_gain=0.04,
        scheme_b_lambda_reg=1e-5,
        tracking_energy_clip=5.0,
        system_energy_clip=5.0,
        startup_speed_floor=1e-4,
        startup_velocity_floor=1e-4,
        startup_hold_steps=60,
        startup_gain=0.6,
        startup_lambda=1e-9,
        theta_lower=settings.theta_lower.copy(),
        theta_upper=settings.theta_upper.copy(),
        theta_dot_lower=theta_dot_lower,
        theta_dot_upper=theta_dot_upper,
    )
    base.update(overrides)
    return IntegratedMethodConfig(**base)


def build_method1(robot, settings):
    """构造 Method 1：sig 漂移反馈 + 简化 S-LVI-PDNN。"""
    return Method1SigRMPQPController(
        robot,
        make_config(
            settings,
            robot,
            regularization_gain=0.0,
            solver_substeps=1,
        ),
    )


def build_method2(robot, settings):
    """构造 Method 2：Scheme-B 正函数能量整形求解器。"""
    return Method2SchemeBController(
        robot,
        make_config(
            settings,
            robot,
            task_gain=tuned_method2_task_gain(settings.tau),
            mu_gain=10.0,
            gamma_gain=1280.0,
            activation_power=0.9,
            activation_exp_clip=4.0,
            solver_substeps=20,
        ),
    )


def build_method3a(robot, settings):
    """构造 Method 3a：位置误差与漂移误差共同标量化的离散优化器。"""
    return Method3aPositionDriftController(
        robot,
        make_config(
            settings,
            robot,
            task_gain=tuned_method3a_task_gain(settings.tau),
            position_weight=10000.0,
            drift_weight=0.001,
            regularization_gain=1e-9,
            solver_substeps=1,
        ),
    )


def build_method3b(robot, settings):
    """构造 Method 3b：只最小化漂移，任务速度保持纯硬约束。"""
    return Method3bHardConstraintController(
        robot,
        make_config(
            settings,
            robot,
            drift_weight=1.0,
            regularization_gain=1e-9,
            solver_substeps=1,
        ),
    )


def build_method_factories():
    """统一暴露方法名到构造器的映射，供脚本层按名字调度。"""
    return {
        'method1_sig_rmp_qp': build_method1,
        'method2_scheme_b_positive': build_method2,
        'method3a_position_and_drift': build_method3a,
        'method3b_drift_free_hard_constraint': build_method3b,
    }
