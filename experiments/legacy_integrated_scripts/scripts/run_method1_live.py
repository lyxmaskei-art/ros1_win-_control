# Complete live experiment entry for direct VSCode execution.
# This file is intentionally self-contained for direct VSCode execution.

import csv
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

from _bootstrap import RESULTS_DIR
import sim


METHOD_NAME = 'Method 1: sig-type drift QP'
OUTPUT_STEM = 'method1_sig_rmp_qp'
DEFAULT_REMOTE_API_PORTS = (19997, 19998)
SIGN_BI_POWER = 0.9
DEFAULT_THETA_LIMIT = 2.0 * np.pi * np.ones(6, dtype=float)


def sign_bi_power(z, power=SIGN_BI_POWER):
    """实现 sign(z) * |z|^power，这是论文里 sig^r(z) 的标准写法。"""
    z = np.asarray(z, dtype=float)
    abs_z = np.abs(z)
    powered = np.where(abs_z == 0.0, 0.0, np.power(abs_z, power))
    return np.sign(z) * powered


def sig_exp_activation(z, power=SIGN_BI_POWER, exp_clip=2.0):
    """Method 1 使用的非线性漂移反馈：sig^r(z) * exp(|z|)。"""
    z = np.asarray(z, dtype=float)
    abs_z = np.minimum(np.abs(z), exp_clip)
    return sign_bi_power(z, power=power) * np.exp(abs_z)


def compute_dynamic_velocity_bounds(theta_current, theta_lower, theta_upper, theta_dot_limit, eta, tau):
    """
    把关节位置限制实时转成当前步的速度限制。
    这样做的目的，是保证欧拉积分后 q_{k+1} 仍然不越界。
    """
    theta_current = np.asarray(theta_current, dtype=float)
    theta_dot_lower = -float(theta_dot_limit) * np.ones_like(theta_current)
    theta_dot_upper = float(theta_dot_limit) * np.ones_like(theta_current)
    eta_rate = float(eta) / float(tau)
    xi_minus = np.maximum(theta_dot_lower, eta_rate * (theta_lower - theta_current))
    xi_plus = np.minimum(theta_dot_upper, eta_rate * (theta_upper - theta_current))
    return xi_minus, xi_plus


class UR3eKinematics:
    """UR3e 的 DH 运动学与 6x6 雅可比矩阵。"""

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
        if theta.shape[0] != self.num_joints:
            raise ValueError(f'需要 {self.num_joints} 个关节角，实际得到 {theta.shape[0]} 个。')

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
    """让心形轨迹从当前末端位置附近起步，减少一开始的巨大突变误差。"""
    heart_start_local = np.array([0.0, 0.0, 5.0], dtype=float)
    return np.asarray(initial_position, dtype=float) - scale * heart_start_local


class HeartTrajectory:
    """YZ 平面心形轨迹，与 all lccznn.py 保持同一类任务形式。"""

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


class SimplifiedLVIPDNNSolver:
    """
    简化版 S-LVI-PDNN。
    状态 y = [qdot; lambda]，每步执行：
        y_{k+1} = y_k + tau * gamma * ( P(y_k - (H y_k + p)) - y_k )
    其中 P(.) 只对原始变量 qdot 做区间投影。
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

    def step(self, q_matrix, q_vector, aeq_matrix, beq_vector, lower, upper):
        q_matrix = 0.5 * (q_matrix + q_matrix.T)
        aeq_matrix = np.asarray(aeq_matrix, dtype=float)
        beq_vector = np.asarray(beq_vector, dtype=float).reshape(-1)
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)

        if aeq_matrix.ndim == 1:
            aeq_matrix = aeq_matrix.reshape(1, -1)
        if aeq_matrix.size == 0:
            aeq_matrix = np.zeros((0, self.num_vars), dtype=float)

        dual_size = aeq_matrix.shape[0]
        self._ensure_state_size(dual_size)

        upper_block = np.hstack((q_matrix, -aeq_matrix.T))
        lower_block = np.hstack((aeq_matrix, np.zeros((dual_size, dual_size), dtype=float)))
        h_matrix = np.vstack((upper_block, lower_block))
        p_vector = np.concatenate((q_vector, -beq_vector))

        lvi_input = self.state - (h_matrix @ self.state + p_vector)
        projected = lvi_input.copy()
        projected[: self.num_vars] = np.clip(projected[: self.num_vars], lower, upper)
        v_dot = self.gamma_gain * (projected - self.state)
        v_next_raw = self.state + self.tau * v_dot
        primal_raw = v_next_raw[: self.num_vars]
        primal_next = np.clip(primal_raw, lower, upper)
        dual_next = v_next_raw[self.num_vars :]
        self.state = np.concatenate((primal_next, dual_next))
        return primal_next, primal_raw, dual_next


@dataclass
class Method1Config:
    duration: float = 10.0
    tau: float = 0.005
    heart_scale: float = 0.008
    task_gain: float = 20.0
    mu_gain: float = 0.1
    activation_power: float = 0.9
    activation_exp_clip: float = 2.0
    gamma_gain: float = 80.0
    eta: float = 0.9
    theta_dot_limit: float = 2.0
    offline_duration: float = 20.0
    theta_initial_command: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, -np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0],
            dtype=float,
        )
    )
    theta_lower: np.ndarray = field(default_factory=lambda: -DEFAULT_THETA_LIMIT.copy())
    theta_upper: np.ndarray = field(default_factory=lambda: DEFAULT_THETA_LIMIT.copy())

    @property
    def steps(self):
        return int(self.duration / self.tau)


class Method1Controller:
    """
    Method 1 的控制律：
    1. 任务约束使用带反馈的任务速度
           J(q_k) qdot_k = rdot_d(k) - k_p (x(q_k) - x_d(k))
    2. 漂移抑制进入 QP 的线性项
           c_hat(q_k) = mu * sig^r(q_k - q_0) * exp(|q_k - q_0|)
    3. 每步求解
           min 1/2 qdot^T I qdot + c_hat^T qdot
           s.t. J qdot = task_feedback,  xi^- <= qdot <= xi^+
    """

    def __init__(self, config):
        self.cfg = config
        self.identity = np.eye(6, dtype=float)
        self.solver = SimplifiedLVIPDNNSolver(num_vars=6, tau=self.cfg.tau, gamma_gain=self.cfg.gamma_gain)
        self.theta_initial = None

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

        theta_dot, theta_dot_raw, dual_next = self.solver.step(
            q_matrix=self.identity,
            q_vector=drift_feedback,
            aeq_matrix=jacobian_task,
            beq_vector=task_feedback,
            lower=lower,
            upper=upper,
        )

        return {
            'theta_dot': theta_dot,
            'theta_dot_raw': theta_dot_raw,
            'dual_next': dual_next,
            'position_error': position_error,
            'task_feedback': task_feedback,
            'task_residual': task_feedback - jacobian_task @ theta_dot,
            'drift_delta': drift_delta,
            'drift_feedback': drift_feedback,
        }


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
            raise RuntimeError(f'获取 UR3e_joint{i} 失败，错误码 {err}')
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
            raise RuntimeError(f'读取 UR3e_joint{i} 角度失败，错误码 {err}')
        positions.append(joint_pos)
    return np.asarray(positions, dtype=float)


def begin_joint_position_stream(client_id, joint_handles):
    for handle in joint_handles:
        sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_streaming)


def read_joint_positions_fast(client_id, joint_handles):
    positions = []
    for i, handle in enumerate(joint_handles, start=1):
        err, joint_pos = sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_buffer)
        if err != sim.simx_return_ok:
            err, joint_pos = sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_blocking)
            if err != sim.simx_return_ok:
                raise RuntimeError(f'读取 UR3e_joint{i} 角度失败，错误码 {err}')
        positions.append(joint_pos)
    return np.asarray(positions, dtype=float)


def send_joint_targets(client_id, joint_handles, joint_targets):
    sim.simxPauseCommunication(client_id, True)
    try:
        for handle, target in zip(joint_handles, joint_targets):
            sim.simxSetJointTargetPosition(client_id, handle, float(target), sim.simx_opmode_oneshot)
    finally:
        sim.simxPauseCommunication(client_id, False)


def startup_handshake_and_settle(client_id, joint_handles, theta_goal, tau):
    theta_goal = np.asarray(theta_goal, dtype=float)
    theta_feedback = read_joint_positions(client_id, joint_handles, sim.simx_opmode_blocking)

    for step in range(20):
        if step == 0:
            theta_cmd = theta_feedback
        else:
            alpha = step / 19.0
            theta_cmd = theta_feedback + alpha * (theta_goal - theta_feedback)
        send_joint_targets(client_id, joint_handles, theta_cmd)
        sim.simxSynchronousTrigger(client_id)
        sim.simxGetPingTime(client_id)
        time.sleep(tau)

    theta_settled = read_joint_positions(client_id, joint_handles, sim.simx_opmode_blocking)
    return theta_feedback, theta_settled



def make_controller(cfg):
    return Method1Controller(cfg)


def create_history():
    return {
        'time_s': [],
        'actual_positions': [],
        'desired_positions': [],
        'position_errors': [],
        'joint_positions': [],
        'joint_drift': [],
    }


def record_history(history, tk, theta_current, current_pos, desired_pos, theta_reference):
    theta_current = np.asarray(theta_current, dtype=float)
    current_pos = np.asarray(current_pos, dtype=float)
    desired_pos = np.asarray(desired_pos, dtype=float)
    theta_reference = np.asarray(theta_reference, dtype=float)
    drift = theta_current - theta_reference

    history['time_s'].append(float(tk))
    history['actual_positions'].append(current_pos.tolist())
    history['desired_positions'].append(desired_pos.tolist())
    history['position_errors'].append(float(np.linalg.norm(current_pos - desired_pos)))
    history['joint_positions'].append(theta_current.tolist())
    history['joint_drift'].append(drift.tolist())


def configure_trajectory_axes(ax, desired, actual):
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


def plot_trajectory_figure(history, output_path, title_prefix):
    desired = np.asarray(history['desired_positions'], dtype=float)
    actual = np.asarray(history['actual_positions'], dtype=float)

    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(desired[:, 0], desired[:, 1], desired[:, 2], color='#c0392b', linewidth=2.2, label='Desired')
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], color='#1f77b4', linestyle='--', linewidth=2.0, label='Actual')
    configure_trajectory_axes(ax, desired, actual)
    ax.set_title(f'{title_prefix} Trajectory')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def plot_joint_angles_figure(history, output_path, title_prefix):
    joints = np.asarray(history['joint_positions'], dtype=float)
    time_vec = np.asarray(history['time_s'], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i in range(joints.shape[1]):
        ax.plot(time_vec, joints[:, i], linewidth=1.6, label=rf'$\theta_{{{i + 1}}}$')
    ax.set_title(f'{title_prefix} Joint Angles')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('rad')
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def plot_position_error_figure(history, output_path, title_prefix):
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


def plot_joint_drift_figure(history, output_path, title_prefix):
    drift = np.asarray(history['joint_drift'], dtype=float)
    time_vec = np.asarray(history['time_s'], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i in range(drift.shape[1]):
        ax.plot(time_vec, drift[:, i], linewidth=1.6, label=rf'$\Delta \theta_{{{i + 1}}}(t)$')
    ax.axhline(0.0, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
    ax.set_title(f'{title_prefix} Joint Drift Error')
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$\theta_i(t)-\theta_i(0)$ (rad)')
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def save_joint_drift_table(theta_initial, theta_final, output_dir, table_label):
    theta_initial = np.asarray(theta_initial, dtype=float)
    theta_final = np.asarray(theta_final, dtype=float)
    delta = theta_final - theta_initial

    csv_path = output_dir / f'{OUTPUT_STEM}_{table_label}_joint_table.csv'
    png_path = output_dir / f'{OUTPUT_STEM}_{table_label}_joint_table.png'

    rows = []
    for i in range(theta_initial.shape[0]):
        rows.append([
            f'$\\theta_{{{i + 1}}}$',
            f'{theta_initial[i]:.6f}',
            f'{theta_final[i]:.6f}',
            f'{delta[i]:.6e}',
        ])

    with csv_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Joint', 'Initial angle (rad)', 'Final angle (rad)', 'Delta (rad)'])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=['Joint', 'Initial angle (rad)', 'Final angle (rad)', 'Delta (rad)'],
        loc='center',
        cellLoc='center',
        colLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#eaf2f8')
            cell.set_text_props(weight='bold')
        if col == 3 and row > 0:
            cell.set_facecolor('#fef5e7')
    ax.set_title(f'{METHOD_NAME} Offline Joint Drift Table ({table_label})')
    fig.tight_layout()
    fig.savefig(png_path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def save_live_figures(history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory_figure(history, output_dir / f'{OUTPUT_STEM}_live_trajectory.png', METHOD_NAME)
    plot_joint_angles_figure(history, output_dir / f'{OUTPUT_STEM}_live_joint_angles.png', METHOD_NAME)
    plot_position_error_figure(history, output_dir / f'{OUTPUT_STEM}_live_position_error.png', METHOD_NAME)
    plot_joint_drift_figure(history, output_dir / f'{OUTPUT_STEM}_live_joint_drift.png', METHOD_NAME)


def save_offline_figures(history, theta_initial, theta_final, output_dir, offline_duration):
    output_dir.mkdir(parents=True, exist_ok=True)
    title_prefix = f'{METHOD_NAME} Offline'
    plot_position_error_figure(history, output_dir / f'{OUTPUT_STEM}_offline_position_error.png', title_prefix)
    plot_joint_drift_figure(history, output_dir / f'{OUTPUT_STEM}_offline_joint_drift.png', title_prefix)
    save_joint_drift_table(theta_initial, theta_final, output_dir, f'offline_{offline_duration:g}s')


def run_live_experiment(cfg, robot, output_dir):
    controller = make_controller(cfg)
    print(f'???? {METHOD_NAME} ??????? ...')
    client_id, port = connect_to_coppeliasim()
    if client_id == -1:
        print('????? CoppeliaSim???????????')
        return None
    print(f'???? CoppeliaSim??? {port}?')

    sim.simxSynchronous(client_id, True)
    sim.simxStartSimulation(client_id, sim.simx_opmode_oneshot)

    history = create_history()
    theta_current = None

    try:
        joint_handles = get_joint_handles(client_id)
        begin_joint_position_stream(client_id, joint_handles)
        theta_lower, theta_upper = query_joint_limits_from_sim(client_id, joint_handles)
        cfg.theta_lower = theta_lower
        cfg.theta_upper = theta_upper

        _, theta_reference = startup_handshake_and_settle(client_id, joint_handles, cfg.theta_initial_command, cfg.tau)
        controller.reset(theta_reference)

        initial_pos = robot.forward_kinematics(theta_reference)[:3, 3]
        trajectory_offset = make_offset_from_initial_position(initial_pos, cfg.heart_scale)
        trajectory = HeartTrajectory(duration=cfg.duration, scale=cfg.heart_scale, offset=trajectory_offset)

        for step in range(cfg.steps):
            tk = step * cfg.tau
            theta_current = read_joint_positions_fast(client_id, joint_handles)
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
            record_history(history, tk, theta_current, current_pos, desired_pos, theta_reference)
            step_result = controller.step(
                theta_current=theta_current,
                current_pos=current_pos,
                desired_pos=desired_pos,
                desired_vel=desired_vel,
                jacobian_task=jacobian_task,
                lower=lower,
                upper=upper,
            )
            theta_next = np.clip(theta_current + cfg.tau * step_result['theta_dot'], cfg.theta_lower, cfg.theta_upper)
            send_joint_targets(client_id, joint_handles, theta_next)
            sim.simxSynchronousTrigger(client_id)
            sim.simxGetPingTime(client_id)

        print(f'{METHOD_NAME} ?????????')
    finally:
        sim.simxStopSimulation(client_id, sim.simx_opmode_oneshot)
        sim.simxFinish(client_id)

    save_live_figures(history, output_dir)
    return {
        'history': history,
        'theta_final': theta_current,
    }


def main():
    output_root = RESULTS_DIR / 'single_file_integrated_runs' / OUTPUT_STEM
    robot = UR3eKinematics()
    cfg = Method1Config()
    print(f'Running {METHOD_NAME} live experiment ...')
    result = run_live_experiment(cfg, robot, output_root / 'live')
    if result is not None:
        print(f'{METHOD_NAME} live result saved to {output_root / "live"}')
    return result


if __name__ == '__main__':
    main()

