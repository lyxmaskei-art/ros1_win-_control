"""
离散 RMP-QP / DLCCZNN 风格实验的公共模块。

这里集中放置：
1. UR3e 运动学与参考轨迹；
2. 统一实验配置与 QP 数据结构；
3. 离线验证与 CoppeliaSim 真实闭环验证逻辑；
4. 结果统计、保存和绘图工具。
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SIM_DIR = PROJECT_ROOT / 'sim'
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

import sim  # CoppeliaSim remote API


UR3E_D = np.array([0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921], dtype=float)
UR3E_A = np.array([0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0], dtype=float)
UR3E_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=float)
DEFAULT_THETA_LIMIT = 2.0 * np.pi * np.ones(6, dtype=float)
DEFAULT_THETA_DOT_LIMIT = 2.0 * np.ones(6, dtype=float)
DEFAULT_REMOTE_API_PORTS = (19997, 19998)


def env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value is not None else default


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value is not None else default


def sign_bi_power(z, power):
    """逐元素计算 sign(z) * |z|^power，是 DLCCZNN 类激活函数的基础部件。"""
    z = np.asarray(z, dtype=float)
    abs_z = np.abs(z)
    power_term = np.where(abs_z == 0.0, 0.0, np.power(abs_z, power))
    return np.sign(z) * power_term


def sig_exp_activation(z, power, exp_clip):
    """
    sig 型激活函数。

    作用：
    1. 小误差区仍保留非线性灵敏度；
    2. 大误差区通过 exp 项增强回拉趋势；
    3. 通过 exp_clip 防止数值爆炸。
    """
    z = np.asarray(z, dtype=float)
    abs_z = np.minimum(np.abs(z), exp_clip)
    return sign_bi_power(z, power=power) * np.exp(abs_z)


def positive_exp_activation(z, power, exp_clip):
    """Method 2 使用的正函数激活，只对非负标量能量进行放大。"""
    scalar = np.maximum(float(z), 0.0)
    scalar = min(scalar, exp_clip)
    return (scalar ** power) * np.exp(scalar)


def positive_power_activation(z, power, epsilon=1e-12):
    """纯正函数幂次激活，主要用于做对照或保守增益整形。"""
    scalar = max(float(z), 0.0)
    return (scalar + epsilon) ** power - (epsilon ** power)


def order_of_magnitude(value):
    """把误差量级转成 10^k 里的 k，便于做书中那种数量级对比表。"""
    scalar = float(value)
    return None if scalar <= 0.0 else int(np.floor(np.log10(scalar)))


class UR3eKinematics:
    def __init__(self):
        self.d = UR3E_D.copy()
        self.a = UR3E_A.copy()
        self.alpha = UR3E_ALPHA.copy()
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
    """根据初始末端位置反推心形轨迹的平移偏置，保证轨迹从当前姿态附近起跑。"""
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
        """返回时刻 t 的期望末端位置与速度，供外层控制器离散跟踪。"""
        pos_current = self._calculate_position(t)
        pos_future = self._calculate_position(t + self.dt)
        pos_past = self._calculate_position(t - self.dt)
        velocity = (pos_future - pos_past) / (2.0 * self.dt)
        return pos_current, velocity


@dataclass
class ExperimentSettings:
    """
    统一实验设置。

    说明：
    - tau 是外层控制离散步长，不一定等于 CoppeliaSim 内部积分步长。
    - cycles_offline / cycles_live 表示重复周期数，周期长度由 duration 决定。
    - task_gain 是“任务空间速度反馈增益”，不同方法会按自己的 builder 再做调优。
    """
    duration: float = 10.0
    cycles_offline: int = 5
    cycles_live: int = 1
    tau: float = 0.005
    heart_scale: float = 0.008
    task_gain: float = 20.0
    theta_dot_limit: float = 2.0
    eta: float = 0.9
    theta_initial_command: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0], dtype=float)
    )
    theta_lower: np.ndarray = field(default_factory=lambda: -DEFAULT_THETA_LIMIT.copy())
    theta_upper: np.ndarray = field(default_factory=lambda: DEFAULT_THETA_LIMIT.copy())

    @property
    def steps_per_cycle(self):
        return int(self.duration / self.tau)


@dataclass
class QPProblem:
    """统一的 QP 描述结构。"""
    q_matrix: np.ndarray
    q_vector: np.ndarray
    aeq_matrix: np.ndarray
    beq_vector: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


class SimplifiedLVIPDNNSolver:
    """
    简化版 S-LVI-PDNN。

    这里只保留书中最核心的离散形式：
    y_{k+1} = y_k + tau * gamma * ( P(y_k - (Hy_k + p)) - y_k )

    注意：
    - 这是“数值实现友好版”，不是严格逐式复刻书中的所有连续分析细节。
    - 当 QP 结构已经足够简单时，直接解线性系统通常会比继续离散这个动态更稳。
    """

    def __init__(self, num_vars, tau, gamma_gain):
        self.num_vars = int(num_vars)
        self.tau = float(tau)
        self.gamma_gain = float(gamma_gain)
        self.state = np.zeros(self.num_vars, dtype=float)
        self.dual_size = 0

    def reset(self):
        self.state = np.zeros(self.num_vars + self.dual_size, dtype=float)

    def _ensure_state_size(self, dual_size):
        dual_size = int(dual_size)
        if self.state.shape[0] != self.num_vars + dual_size:
            self.dual_size = dual_size
            self.state = np.zeros(self.num_vars + dual_size, dtype=float)
        else:
            self.dual_size = dual_size

    def step(self, problem):
        q_matrix = 0.5 * (np.asarray(problem.q_matrix, dtype=float) + np.asarray(problem.q_matrix, dtype=float).T)
        q_vector = np.asarray(problem.q_vector, dtype=float).reshape(-1)
        aeq_matrix = np.asarray(problem.aeq_matrix, dtype=float)
        beq_vector = np.asarray(problem.beq_vector, dtype=float).reshape(-1)
        lower = np.asarray(problem.lower, dtype=float).reshape(-1)
        upper = np.asarray(problem.upper, dtype=float).reshape(-1)

        if aeq_matrix.ndim == 1:
            aeq_matrix = aeq_matrix.reshape(1, -1)
        if aeq_matrix.size == 0:
            aeq_matrix = np.zeros((0, self.num_vars), dtype=float)
        dual_size = aeq_matrix.shape[0]
        self._ensure_state_size(dual_size)

        if dual_size > 0:
            upper_block = np.hstack((q_matrix, -aeq_matrix.T))
            lower_block = np.hstack((aeq_matrix, np.zeros((dual_size, dual_size), dtype=float)))
            h_matrix = np.vstack((upper_block, lower_block))
            p_vector = np.concatenate((q_vector, -beq_vector))
        else:
            h_matrix = q_matrix
            p_vector = q_vector

        lvi_input = self.state - (h_matrix @ self.state + p_vector)
        projected = lvi_input.copy()
        projected[: self.num_vars] = np.clip(projected[: self.num_vars], lower, upper)
        v_dot = self.gamma_gain * (projected - self.state)
        v_next_raw = self.state + self.tau * v_dot
        primal_raw = v_next_raw[: self.num_vars]
        primal_next = np.clip(primal_raw, lower, upper)

        if dual_size > 0:
            dual_next = v_next_raw[self.num_vars :]
            self.state = np.concatenate((primal_next, dual_next))
        else:
            dual_next = np.zeros(0, dtype=float)
            self.state = primal_next

        return {
            'solution': primal_next,
            'primal_raw': primal_raw,
            'dual_next': dual_next,
            'lvi_input': lvi_input,
            'projected': projected,
            'v_dot': v_dot,
        }


def compute_dynamic_velocity_bounds(theta_current, theta_lower, theta_upper, theta_dot_lower, theta_dot_upper, eta, tau):
    """
    把关节角位置约束实时转换成关节速度约束。

    逻辑与 Zhang 2009 一致：
    - 如果当前关节已经靠近上/下界，则收紧当前步允许的 qdot；
    - eta/tau 决定“为了不越界，本步最多还能走多快”。
    """
    theta_current = np.asarray(theta_current, dtype=float)
    eta_rate = float(eta) / float(tau)
    xi_minus = np.maximum(theta_dot_lower, eta_rate * (theta_lower - theta_current))
    xi_plus = np.minimum(theta_dot_upper, eta_rate * (theta_upper - theta_current))
    return xi_minus, xi_plus


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
        if err == sim.simx_return_ok and upper_limit > 0:
            theta_upper[i] = upper_limit
            theta_lower[i] = -upper_limit

    return theta_lower, theta_upper


def read_joint_positions(client_id, joint_handles):
    joint_positions = []
    for i, handle in enumerate(joint_handles, start=1):
        err, joint_pos = sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_blocking)
        if err != sim.simx_return_ok:
            raise RuntimeError(f'Could not read joint position UR3e_joint{i}, error code={err}')
        joint_positions.append(joint_pos)
    return np.asarray(joint_positions, dtype=float)


def begin_joint_position_stream(client_id, joint_handles):
    for handle in joint_handles:
        sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_streaming)


def read_joint_positions_fast(client_id, joint_handles):
    joint_positions = []
    for i, handle in enumerate(joint_handles, start=1):
        err, joint_pos = sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_buffer)
        if err == sim.simx_return_ok:
            joint_positions.append(joint_pos)
            continue

        err, joint_pos = sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_blocking)
        if err != sim.simx_return_ok:
            raise RuntimeError(f'Could not read joint position UR3e_joint{i}, error code={err}')
        joint_positions.append(joint_pos)

    return np.asarray(joint_positions, dtype=float)


def send_joint_targets(client_id, joint_handles, joint_targets, actuation_mode='target'):
    sim.simxPauseCommunication(client_id, True)
    try:
        for handle, target in zip(joint_handles, joint_targets):
            if actuation_mode == 'target':
                sim.simxSetJointTargetPosition(client_id, handle, float(target), sim.simx_opmode_oneshot)
            else:
                sim.simxSetJointPosition(client_id, handle, float(target), sim.simx_opmode_oneshot)
    finally:
        sim.simxPauseCommunication(client_id, False)


def startup_handshake_and_settle(client_id, joint_handles, theta_goal, settle_steps, tau, actuation_mode='target'):
    theta_goal = np.asarray(theta_goal, dtype=float)
    theta_feedback = read_joint_positions(client_id, joint_handles)
    total_steps = max(int(settle_steps), 1)

    for step in range(total_steps):
        if step == 0 or total_steps == 1:
            theta_cmd = theta_feedback
        else:
            alpha = step / float(total_steps - 1)
            theta_cmd = theta_feedback + alpha * (theta_goal - theta_feedback)

        send_joint_targets(client_id, joint_handles, theta_cmd, actuation_mode=actuation_mode)
        sim.simxSynchronousTrigger(client_id)
        sim.simxGetPingTime(client_id)
        time.sleep(tau)

    theta_settled = read_joint_positions(client_id, joint_handles)
    return theta_feedback, theta_settled


def connect_to_coppeliasim():
    candidate_ports = []
    env_port = os.environ.get('COPPELIASIM_PORT')
    if env_port is not None:
        candidate_ports.append(int(env_port))
    for port in DEFAULT_REMOTE_API_PORTS:
        if port not in candidate_ports:
            candidate_ports.append(port)

    sim.simxFinish(-1)
    for port in candidate_ports:
        client_id = sim.simxStart('127.0.0.1', port, True, True, 5000, 5)
        if client_id != -1:
            return client_id, port
    return -1, None


def build_heart_trajectory(robot, settings, theta_reference):
    """以给定初始关节角为基准，构造与之对齐的心形任务轨迹。"""
    initial_pos = robot.forward_kinematics(theta_reference)[:3, 3]
    trajectory_offset = make_offset_from_initial_position(initial_pos, scale=settings.heart_scale)
    trajectory = HeartTrajectory(duration=settings.duration, scale=settings.heart_scale, offset=trajectory_offset)
    return trajectory, initial_pos, trajectory_offset


def default_cycle_summary(method_name, cycle_index, theta_final, theta_reference, desired_positions, actual_positions, task_residuals):
    """
    计算单个周期结束时的关键精度指标。

    同时保留两种口径：
    - max_angular_error = max_i |theta_i(T) - theta_i(0)|
    - terminal_drift_norm = ||theta(T) - theta(0)||_2

    其中第二个量对 Method 2 / 3 这种“先标量化二范数再建模”的方法更关键。
    """
    desired = np.asarray(desired_positions, dtype=float)
    actual = np.asarray(actual_positions, dtype=float)
    task_residuals = np.asarray(task_residuals, dtype=float)
    terminal_joint_drift = theta_final - theta_reference
    max_angular_error = float(np.max(np.abs(terminal_joint_drift)))
    terminal_drift_norm = float(np.linalg.norm(terminal_joint_drift))
    position_errors = np.linalg.norm(actual - desired, axis=1)

    return {
        'method': method_name,
        'cycle': int(cycle_index),
        'terminal_joint_drift': terminal_joint_drift.tolist(),
        'max_angular_error': max_angular_error,
        'max_angular_error_order_of_magnitude': order_of_magnitude(max_angular_error),
        'terminal_drift_norm': terminal_drift_norm,
        'terminal_drift_norm_order_of_magnitude': order_of_magnitude(terminal_drift_norm),
        'terminal_position_error_mm': float(position_errors[-1] * 1000.0),
        'mean_position_error_mm': float(np.mean(position_errors) * 1000.0),
        'max_position_error_mm': float(np.max(position_errors) * 1000.0),
        'mean_task_residual_norm': float(np.mean(task_residuals)),
        'max_task_residual_norm': float(np.max(task_residuals)),
    }


def build_method_summary(method_name, per_cycle_metrics, runtime_s):
    """把逐周期指标汇总成最终报表字段。"""
    one_cycle = per_cycle_metrics[0]
    final_cycle = per_cycle_metrics[-1]
    final_max_angular = float(final_cycle['max_angular_error'])
    final_drift_norm = float(final_cycle['terminal_drift_norm'])
    return {
        'method': method_name,
        'runtime_s': float(runtime_s),
        'cycles': len(per_cycle_metrics),
        'one_cycle_max_angular_error': float(one_cycle['max_angular_error']),
        'one_cycle_max_angular_error_order_of_magnitude': one_cycle['max_angular_error_order_of_magnitude'],
        'one_cycle_terminal_drift_norm': float(one_cycle['terminal_drift_norm']),
        'one_cycle_terminal_drift_norm_order_of_magnitude': one_cycle['terminal_drift_norm_order_of_magnitude'],
        'multi_cycle_final_max_angular_error': final_max_angular,
        'multi_cycle_final_max_angular_error_order_of_magnitude': order_of_magnitude(final_max_angular),
        'multi_cycle_final_terminal_drift_norm': final_drift_norm,
        'multi_cycle_final_terminal_drift_norm_order_of_magnitude': order_of_magnitude(final_drift_norm),
        'multi_cycle_max_position_error_mm': float(np.max([item['max_position_error_mm'] for item in per_cycle_metrics])),
        'multi_cycle_mean_position_error_mm': float(np.mean([item['mean_position_error_mm'] for item in per_cycle_metrics])),
        'multi_cycle_max_task_residual_norm': float(np.max([item['max_task_residual_norm'] for item in per_cycle_metrics])),
        'final_cycle_terminal_joint_drift': final_cycle['terminal_joint_drift'],
    }


def run_offline_method(method_name, controller, robot, settings, collect_history=False):
    """
    纯数据离线验证。

    这里没有任何仿真反馈，所有状态都按控制器输出和机器人运动学模型直接推进，
    因而它更适合看“数学层 / 数值层”本身的误差上限。
    """
    theta = settings.theta_initial_command.copy()
    controller.reset(theta)
    trajectory, _, _ = build_heart_trajectory(robot, settings, theta)

    cycle_metrics = []
    history = {
        'time_s': [],
        'actual_positions': [],
        'desired_positions': [],
        'task_residuals': [],
        'drift_norms': [],
        'position_error_mm': [],
        'joint_positions': [],
    }
    total_steps = settings.cycles_offline * settings.steps_per_cycle

    t_start = time.perf_counter()
    for step in range(total_steps):
        local_step = step % settings.steps_per_cycle
        desired_position, desired_velocity = trajectory.get_pose(local_step * settings.tau)
        step_data = controller.step(theta, desired_position, desired_velocity)

        # offline 不读仿真反馈，直接把控制器给出的下一步关节角作为系统状态推进，
        # 因而这里更接近“算法自身的理论数值表现”。
        theta = step_data['theta_next'].copy()
        actual_pos = robot.forward_kinematics(theta)[:3, 3]
        drift_norm = float(np.linalg.norm(theta - controller.theta_initial))
        position_error_mm = float(np.linalg.norm(actual_pos - desired_position) * 1000.0)

        history['time_s'].append(float(step * settings.tau))
        history['actual_positions'].append(actual_pos.tolist())
        history['desired_positions'].append(np.asarray(desired_position, dtype=float).tolist())
        history['task_residuals'].append(float(np.linalg.norm(step_data['task_residual'])))
        history['drift_norms'].append(drift_norm)
        history['position_error_mm'].append(position_error_mm)
        history['joint_positions'].append(theta.tolist())

        if (step + 1) % settings.steps_per_cycle == 0:
            cycle_index = (step + 1) // settings.steps_per_cycle
            cycle_start = step + 1 - settings.steps_per_cycle

            # 每跑完一个完整周期就单独记一次摘要，后面才能同时拿到
            # “单周期结果”和“多周期累计结果”。
            cycle_metrics.append(
                default_cycle_summary(
                    method_name=method_name,
                    cycle_index=cycle_index,
                    theta_final=theta,
                    theta_reference=controller.theta_initial,
                    desired_positions=history['desired_positions'][cycle_start: step + 1],
                    actual_positions=history['actual_positions'][cycle_start: step + 1],
                    task_residuals=history['task_residuals'][cycle_start: step + 1],
                )
            )
    runtime_s = time.perf_counter() - t_start

    payload = {
        'cycle_metrics': cycle_metrics,
        'summary': build_method_summary(method_name, cycle_metrics, runtime_s),
        'theta_reference': np.asarray(controller.theta_initial, dtype=float).tolist(),
        'theta_final': np.asarray(theta, dtype=float).tolist(),
    }
    if collect_history:
        payload['history'] = history
    return payload


def run_live_method(method_name, controller, robot, settings, actuation_mode='target', collect_history=False):
    """
    闭环 live 验证。

    这里会真正把关节目标发给 CoppeliaSim，再读取反馈角度，因此会额外受到：
    - 位置目标接口误差
    - 反馈延迟
    - 外层 tau 与仿真内部步长不一致
    的影响。
    """
    client_id, port = connect_to_coppeliasim()
    if client_id == -1:
        raise RuntimeError('Failed to connect to CoppeliaSim.')

    cycle_metrics = []
    history = {
        'time_s': [],
        'actual_positions': [],
        'desired_positions': [],
        'task_residuals': [],
        'drift_norms': [],
        'position_error_mm': [],
        'joint_positions': [],
    }

    sim.simxSynchronous(client_id, True)
    sim.simxStartSimulation(client_id, sim.simx_opmode_oneshot)
    try:
        joint_handles = []
        for i in range(1, 7):
            err, handle = sim.simxGetObjectHandle(client_id, f'UR3e_joint{i}', sim.simx_opmode_blocking)
            if err != sim.simx_return_ok:
                raise RuntimeError(f'Could not get joint handle UR3e_joint{i}, error code={err}')
            joint_handles.append(handle)
        begin_joint_position_stream(client_id, joint_handles)

        theta_lower, theta_upper = query_joint_limits_from_sim(client_id, joint_handles)
        settings.theta_lower = theta_lower
        settings.theta_upper = theta_upper
        controller.update_joint_limits(theta_lower, theta_upper)

        _, theta_reference = startup_handshake_and_settle(
            client_id,
            joint_handles,
            settings.theta_initial_command,
            settle_steps=20,
            tau=settings.tau,
            actuation_mode=actuation_mode,
        )
        controller.reset(theta_reference)
        trajectory, _, _ = build_heart_trajectory(robot, settings, theta_reference)
        theta_current = theta_reference.copy()

        total_steps = settings.cycles_live * settings.steps_per_cycle
        t_start = time.perf_counter()
        for step in range(total_steps):
            local_step = step % settings.steps_per_cycle
            desired_position, desired_velocity = trajectory.get_pose(local_step * settings.tau)
            step_data = controller.step(theta_current, desired_position, desired_velocity)

            # live 模式真正把目标关节角发给 CoppeliaSim，再等一拍读回反馈，
            # 因此这里的误差同时包含控制误差、执行误差和同步误差。
            send_joint_targets(client_id, joint_handles, step_data['theta_next'], actuation_mode=actuation_mode)
            sim.simxSynchronousTrigger(client_id)
            sim.simxGetPingTime(client_id)

            theta_current = read_joint_positions_fast(client_id, joint_handles)
            actual_pos = robot.forward_kinematics(theta_current)[:3, 3]
            drift_norm = float(np.linalg.norm(theta_current - controller.theta_initial))
            position_error_mm = float(np.linalg.norm(actual_pos - desired_position) * 1000.0)

            history['time_s'].append(float(step * settings.tau))
            history['actual_positions'].append(actual_pos.tolist())
            history['desired_positions'].append(np.asarray(desired_position, dtype=float).tolist())
            history['task_residuals'].append(float(np.linalg.norm(step_data['task_residual'])))
            history['drift_norms'].append(drift_norm)
            history['position_error_mm'].append(position_error_mm)
            history['joint_positions'].append(theta_current.tolist())

            if (step + 1) % settings.steps_per_cycle == 0:
                cycle_index = (step + 1) // settings.steps_per_cycle
                cycle_start = step + 1 - settings.steps_per_cycle
                cycle_metrics.append(
                    default_cycle_summary(
                        method_name=method_name,
                        cycle_index=cycle_index,
                        theta_final=theta_current,
                        theta_reference=controller.theta_initial,
                        desired_positions=history['desired_positions'][cycle_start: step + 1],
                        actual_positions=history['actual_positions'][cycle_start: step + 1],
                        task_residuals=history['task_residuals'][cycle_start: step + 1],
                    )
                )
        runtime_s = time.perf_counter() - t_start

    finally:
        sim.simxStopSimulation(client_id, sim.simx_opmode_oneshot)
        sim.simxFinish(client_id)

    payload = {
        'port': port,
        'cycle_metrics': cycle_metrics,
        'summary': build_method_summary(method_name, cycle_metrics, runtime_s),
        'theta_reference': np.asarray(controller.theta_initial, dtype=float).tolist(),
        'theta_final': np.asarray(theta_current, dtype=float).tolist(),
    }
    if collect_history:
        payload['history'] = history
    return payload


def save_summary_json(output_path, payload):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def plot_comparison(output_path, offline_results, live_results):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method_names = list(offline_results.keys())
    offline_values = [offline_results[name]['summary']['multi_cycle_final_max_angular_error'] for name in method_names]
    live_values = [live_results[name]['summary']['one_cycle_max_angular_error'] for name in method_names]

    x = np.arange(len(method_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, offline_values, width=width, label='Offline multicycle')
    ax.bar(x + width / 2, live_values, width=width, label='Live one cycle')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15)
    ax.set_ylabel('Maximum angular error (rad)')
    ax.set_title('Discrete RMP-QP angular drift comparison')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
