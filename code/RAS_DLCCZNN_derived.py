import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import sim  # CoppeliaSim ???????


DEFAULT_ACTIVATION_POWER = 1.0
UR3E_D = np.array([0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921], dtype=float)
UR3E_A = np.array([0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0], dtype=float)
UR3E_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=float)
DEFAULT_THETA_LIMIT = 2.0 * np.pi * np.ones(6, dtype=float)
DEFAULT_THETA_DOT_LIMIT = 1.5 * np.ones(6, dtype=float)
DEFAULT_REMOTE_API_PORT = int(os.environ.get('COPPELIASIM_PORT', '19998'))


def env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value is not None else default


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value is not None else default


def sign_bi_power(z, power=DEFAULT_ACTIVATION_POWER):
    """????? sig^p(z)=sign(z)*|z|^p?"""
    z = np.asarray(z, dtype=float)
    abs_z = np.abs(z)
    power_term = np.where(abs_z == 0.0, 0.0, np.power(abs_z, power))
    return np.sign(z) * power_term


def nonlinear_activation(z, power=DEFAULT_ACTIVATION_POWER, exp_clip=20.0):
    """?????? ?(z)=sig^p(z)*exp(|z|)?"""
    z = np.asarray(z, dtype=float)
    abs_z = np.minimum(np.abs(z), exp_clip)
    return sign_bi_power(z, power=power) * np.exp(abs_z)


class UR3eKinematics:
    """UR3e ??????? 6x6 ??????"""

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
        if theta.shape[0] != self.num_joints:
            raise ValueError(f'Expected {self.num_joints} joints, got {theta.shape[0]}.')

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
    """?????? t=0 ?????????????"""
    heart_start_local = np.array([0.0, 0.0, 5.0], dtype=float)
    return np.asarray(initial_position, dtype=float) - scale * heart_start_local


class HeartTrajectory:
    """YZ ?????????"""

    def __init__(self, duration, scale=0.008, offset=np.array([0.3, 0.0, 0.4], dtype=float)):
        self.duration = float(duration)
        self.scale = float(scale)
        self.offset = np.asarray(offset, dtype=float)
        self.dt = 1e-5

    def _calculate_position(self, t):
        a = (t / self.duration) * 2.0 * np.pi
        y_heart = 16.0 * (np.sin(a) ** 3)
        z_heart = 13.0 * np.cos(a) - 5.0 * np.cos(2.0 * a) - 2.0 * np.cos(3.0 * a) - np.cos(4.0 * a)
        return self.scale * np.array([0.0, y_heart, z_heart], dtype=float) + self.offset

    def get_pose(self, t):
        pos_current = self._calculate_position(t)
        pos_future = self._calculate_position(t + self.dt)
        pos_past = self._calculate_position(t - self.dt)
        velocity = (pos_future - pos_past) / (2.0 * self.dt)
        return pos_current, velocity


@dataclass
class RASDLCCZNNConfig:
    tau: float = 0.005
    mu_gain: float = 50.0
    gamma_gain: float = 50.0
    lambda_reg: float = 1e-5
    w_gain: float = 1.0
    beta_gain: float = 0.04
    eta: float = 0.9
    activation_power: float = DEFAULT_ACTIVATION_POWER
    activation_exp_clip: float = 8.0
    tracking_v_clip: float = 5.0
    vsys_clip: float = 5.0
    startup_speed_floor: float = 1e-4
    startup_velocity_floor: float = 1e-4
    startup_hold_steps: int = 60
    startup_gain: float = 0.6
    startup_lambda: float = 1e-9
    theta_lower: np.ndarray = None
    theta_upper: np.ndarray = None
    theta_dot_lower: np.ndarray = None
    theta_dot_upper: np.ndarray = None

    def __post_init__(self):
        if self.theta_lower is None:
            self.theta_lower = -DEFAULT_THETA_LIMIT.copy()
        if self.theta_upper is None:
            self.theta_upper = DEFAULT_THETA_LIMIT.copy()
        if self.theta_dot_lower is None:
            self.theta_dot_lower = -DEFAULT_THETA_DOT_LIMIT.copy()
        if self.theta_dot_upper is None:
            self.theta_dot_upper = DEFAULT_THETA_DOT_LIMIT.copy()


class RASDLCCZNNController:
    """?? B: ???????? x=[theta_dot; alpha]?"""

    def __init__(self, robot, config):
        self.robot = robot
        self.cfg = config
        self.W = self.cfg.w_gain * np.eye(robot.num_joints, dtype=float)
        self.theta_initial = None
        self.x_state = np.zeros(robot.num_joints + 1, dtype=float)
        self.prev_a = None
        self.prev_b = None
        self.startup_counter = 0

    def build_system_matrix(self, j_psi, psi_error):
        top_right = (j_psi.T @ psi_error).reshape(-1, 1)
        bottom_left = (psi_error.T @ j_psi).reshape(1, -1)
        top = np.hstack((self.W, top_right))
        bottom = np.hstack((bottom_left, np.zeros((1, 1), dtype=float)))
        return np.vstack((top, bottom))

    def build_z_vector(self, theta_current):
        if self.theta_initial is None:
            self.theta_initial = np.asarray(theta_current, dtype=float).copy()
        return self.cfg.beta_gain * (np.asarray(theta_current, dtype=float) - self.theta_initial)

    def build_system_vector(self, theta_current, psi_error, p_dot_d, track_energy):
        z_vector = self.build_z_vector(theta_current)
        track_energy_for_activation = min(track_energy, self.cfg.tracking_v_clip)
        phi_track = float(
            nonlinear_activation(
                track_energy_for_activation,
                power=self.cfg.activation_power,
                exp_clip=self.cfg.activation_exp_clip,
            )
        )
        lower = float(psi_error.T @ p_dot_d - self.cfg.mu_gain * phi_track)
        return np.concatenate((-z_vector, np.array([lower], dtype=float)))

    def estimate_derivative(self, current_value, previous_value):
        if previous_value is None:
            return np.zeros_like(current_value)
        return (current_value - previous_value) / self.cfg.tau

    def clip_theta_dot(self, theta_current, theta_dot_raw):
        eta_rate = self.cfg.eta / self.cfg.tau
        xi_minus = np.maximum(self.cfg.theta_dot_lower, eta_rate * (self.cfg.theta_lower - theta_current))
        xi_plus = np.minimum(self.cfg.theta_dot_upper, eta_rate * (self.cfg.theta_upper - theta_current))
        return np.clip(theta_dot_raw, xi_minus, xi_plus)

    def bootstrap_theta_dot(self, theta_current, p_dot_d):
        """??? B ???????????????? J^T ????"""
        j_psi = self.robot.jacobian(theta_current)[:3, :]
        seed_direction = j_psi.T @ np.asarray(p_dot_d, dtype=float)
        denominator = np.linalg.norm(seed_direction) + self.cfg.startup_lambda
        seed = self.cfg.startup_gain * seed_direction / denominator
        return self.clip_theta_dot(theta_current, seed)

    def step(self, theta_current, p_d, p_dot_d):
        theta_current = np.asarray(theta_current, dtype=float)
        current_pos = self.robot.forward_kinematics(theta_current)[:3, 3]
        psi_error = current_pos - p_d
        j_psi = self.robot.jacobian(theta_current)[:3, :]

        startup_active = (
            np.linalg.norm(p_dot_d) > self.cfg.startup_velocity_floor
            and self.startup_counter < self.cfg.startup_hold_steps
            and (
                np.linalg.norm(self.x_state[: self.robot.num_joints]) < self.cfg.startup_speed_floor
                or self.startup_counter > 0
            )
        )
        startup_theta_dot = None
        if startup_active:
            # ?????????????? J^T ?????????????????????
            startup_theta_dot = self.bootstrap_theta_dot(theta_current, p_dot_d)
            self.startup_counter += 1

        track_energy = 0.5 * float(psi_error.T @ psi_error)
        system_matrix = self.build_system_matrix(j_psi, psi_error)
        system_vector = self.build_system_vector(theta_current, psi_error, p_dot_d, track_energy)
        a_dot = self.estimate_derivative(system_matrix, self.prev_a)
        b_dot = self.estimate_derivative(system_vector, self.prev_b)

        psi_sys = system_matrix @ self.x_state - system_vector
        v_sys = 0.5 * float(psi_sys.T @ psi_sys)
        v_sys_for_activation = min(v_sys, self.cfg.vsys_clip)
        phi_v = float(
            nonlinear_activation(
                v_sys_for_activation,
                power=self.cfg.activation_power,
                exp_clip=self.cfg.activation_exp_clip,
            )
        )

        gradient = system_matrix.T @ psi_sys
        denominator = float(gradient.T @ gradient) + self.cfg.lambda_reg
        drift_term = float(psi_sys.T @ (a_dot @ self.x_state - b_dot))
        correction_scalar = drift_term + self.cfg.gamma_gain * phi_v

        x_raw = self.x_state - self.cfg.tau * gradient * (correction_scalar / denominator)
        theta_dot_raw = x_raw[: self.robot.num_joints]
        alpha_raw = float(x_raw[-1])

        theta_dot_cmd = self.clip_theta_dot(theta_current, theta_dot_raw)
        if startup_active:
            theta_dot_cmd = startup_theta_dot
        x_cmd = np.concatenate((theta_dot_cmd, np.array([alpha_raw], dtype=float)))
        theta_next = np.clip(
            theta_current + self.cfg.tau * theta_dot_cmd,
            self.cfg.theta_lower,
            self.cfg.theta_upper,
        )

        psi_cmd = system_matrix @ x_cmd - system_vector
        self.x_state = x_cmd
        self.prev_a = system_matrix.copy()
        self.prev_b = system_vector.copy()

        if not (
            np.all(np.isfinite(theta_next))
            and np.all(np.isfinite(theta_dot_cmd))
            and np.isfinite(alpha_raw)
            and np.isfinite(v_sys)
        ):
            raise FloatingPointError('??????????????? gamma_gain ????????')

        return {
            'theta_next': theta_next,
            'theta_dot_next': theta_dot_cmd,
            'theta_dot_raw': theta_dot_raw,
            'alpha_next': alpha_raw,
            'current_pos': current_pos,
            'psi_error': psi_error,
            'psi_sys': psi_sys,
            'psi_cmd': psi_cmd,
            'v_sys': v_sys,
            'track_energy': track_energy,
            'drift_term': drift_term,
            'correction_scalar': correction_scalar,
        }


def plot_results(history_actual_pos, history_desired_pos, history_error_norm, history_sys_norm):
    actual = np.asarray(history_actual_pos)
    desired = np.asarray(history_desired_pos)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(actual[:, 0], actual[:, 1], actual[:, 2], label='Actual', linewidth=1.5)
    ax1.plot(desired[:, 0], desired[:, 1], desired[:, 2], label='Desired', linewidth=1.0)
    ax1.set_title('Tracking Trajectory')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    ax2 = fig.add_subplot(132)
    ax2.plot(history_error_norm, linewidth=1.2)
    ax2.set_title('Position Error Norm')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('||psi||')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(133)
    ax3.plot(history_sys_norm, linewidth=1.2)
    ax3.set_title('System Residual Norm')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('||Psi_sys||')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = CURRENT_DIR / 'RAS_DLCCZNN_result.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'Result figure saved to: {output_path}')
    plt.close(fig)


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
    """? CoppeliaSim ??????????????????????????"""
    joint_positions = []
    for i, handle in enumerate(joint_handles, start=1):
        err, joint_pos = sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_blocking)
        if err != sim.simx_return_ok:
            raise RuntimeError(f'Could not read joint position UR3e_joint{i}, error code={err}')
        joint_positions.append(joint_pos)
    return np.asarray(joint_positions, dtype=float)


def send_joint_targets(client_id, joint_handles, joint_targets):
    sim.simxPauseCommunication(client_id, True)
    try:
        for handle, target in zip(joint_handles, joint_targets):
            sim.simxSetJointTargetPosition(client_id, handle, float(target), sim.simx_opmode_oneshot)
    finally:
        sim.simxPauseCommunication(client_id, False)


def main():
    print('Connecting to CoppeliaSim...')
    sim.simxFinish(-1)
    remote_api_port = DEFAULT_REMOTE_API_PORT
    client_id = sim.simxStart('127.0.0.1', remote_api_port, True, True, 5000, 5)
    if client_id == -1:
        print('Failed to connect to CoppeliaSim.')
        return

    history_actual_pos = []
    history_desired_pos = []
    history_error_norm = []
    history_sys_norm = []

    print(f'Connected on port {remote_api_port}.')
    sim.simxSynchronous(client_id, True)
    sim.simxStartSimulation(client_id, sim.simx_opmode_oneshot)

    try:
        joint_handles = []
        for i in range(1, 7):
            err, handle = sim.simxGetObjectHandle(client_id, f'UR3e_joint{i}', sim.simx_opmode_blocking)
            if err != sim.simx_return_ok:
                raise RuntimeError(f'Could not get joint handle UR3e_joint{i}, error code={err}')
            joint_handles.append(handle)

        robot = UR3eKinematics()
        theta_lower, theta_upper = query_joint_limits_from_sim(client_id, joint_handles)

        # ??????????????????
        theta_dot_limit = env_float('RAS_THETA_DOT_LIMIT', 1.2)  # ????????????????????
        mu_gain = env_float('RAS_MU_GAIN', 50.0)  # ???????????????
        gamma_gain = env_float('RAS_GAMMA_GAIN', mu_gain)  # ?? B ??????? ??
        lambda_reg = env_float('RAS_LAMBDA', 1e-5)  # ????? ???? A^T Psi_sys ????????
        activation_power = env_float('RAS_P', env_float('RAS_ACTIVATION_POWER', 1.0))  # ???? p ???

        config = RASDLCCZNNConfig(
            theta_lower=theta_lower,
            theta_upper=theta_upper,
            tau=env_float('RAS_TAU', 0.005),  # ??????????? DLCCZNN ???????
            mu_gain=mu_gain,
            gamma_gain=gamma_gain,
            lambda_reg=lambda_reg,
            w_gain=env_float('RAS_W_GAIN', 1.0),  # W=wgain*I?
            beta_gain=env_float('RAS_BETA_GAIN', 0.04),  # ?????????
            eta=env_float('RAS_ETA', 0.9),  # ???????
            activation_power=activation_power,
            activation_exp_clip=env_float('RAS_EXP_CLIP', 8.0),  # ?? exp ????
            tracking_v_clip=env_float('RAS_TRACK_V_CLIP', 5.0),  # ???????????????
            vsys_clip=env_float('RAS_VSYS_CLIP', 5.0),  # ?? Vsys ?????????
            startup_speed_floor=env_float('RAS_START_SPEED_FLOOR', 1e-4),  # ???????????????????
            startup_velocity_floor=env_float('RAS_START_VEL_FLOOR', 1e-4),  # ????????????????
            startup_hold_steps=env_int('RAS_START_HOLD_STEPS', 60),  # ????????????
            startup_gain=env_float('RAS_START_GAIN', 0.6),  # J^T ?????????????
            startup_lambda=env_float('RAS_START_LAMBDA', 1e-9),  # ?????????????
            theta_dot_lower=-theta_dot_limit * np.ones(6, dtype=float),
            theta_dot_upper=theta_dot_limit * np.ones(6, dtype=float),
        )
        controller = RASDLCCZNNController(robot, config)

        t_start = 0.0
        t_end = env_float('RAS_T_END', 10.0)
        tau = config.tau
        num_steps = int((t_end - t_start) / tau)
        t_discrete = np.linspace(t_start, t_end, num_steps + 1)

        # ?????????????????????
        theta_current = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0.0, 0.0], dtype=float)
        initial_pos = robot.forward_kinematics(theta_current)[:3, 3]
        heart_scale = env_float('RAS_HEART_SCALE', 0.008)
        trajectory_offset = make_offset_from_initial_position(initial_pos, scale=heart_scale)
        trajectory = HeartTrajectory(duration=t_end, scale=heart_scale, offset=trajectory_offset)

        print('Joint lower limits:', np.round(theta_lower, 4))
        print('Joint upper limits:', np.round(theta_upper, 4))
        print('theta_dot limits:', np.round(config.theta_dot_lower, 4), np.round(config.theta_dot_upper, 4))
        print('Initial end-effector position:', np.round(initial_pos, 6))
        print('Auto trajectory offset:', np.round(trajectory_offset, 6))

        print('Moving robot to initial posture...')
        send_joint_targets(client_id, joint_handles, theta_current)
        for _ in range(20):
            sim.simxSynchronousTrigger(client_id)
            sim.simxGetPingTime(client_id)
            time.sleep(tau)
        theta_current = read_joint_positions(client_id, joint_handles)
        print('Measured joint position after settling:', np.round(theta_current, 6))

        print('Running pure RAS controller with Route B...')
        for k in range(num_steps):
            if sim.simxGetConnectionId(client_id) == -1:
                print(f'Connection lost at step {k}.')
                break

            theta_current = read_joint_positions(client_id, joint_handles)
            tk = t_discrete[k]
            p_d, p_dot_d = trajectory.get_pose(tk)
            try:
                step_data = controller.step(theta_current, p_d, p_dot_d)
            except FloatingPointError as exc:
                print(f'Numerical divergence at step {k}: {exc}')
                break

            theta_target = step_data['theta_next']
            send_joint_targets(client_id, joint_handles, theta_target)
            sim.simxSynchronousTrigger(client_id)
            sim.simxGetPingTime(client_id)
            theta_current = read_joint_positions(client_id, joint_handles)
            pos_after_cmd = robot.forward_kinematics(theta_current)[:3, 3]

            history_actual_pos.append(pos_after_cmd)
            history_desired_pos.append(p_d)
            history_error_norm.append(np.linalg.norm(pos_after_cmd - p_d))
            history_sys_norm.append(np.linalg.norm(step_data['psi_sys']))

            if k % 50 == 0:
                print(
                    f"step={k:4d}, pos_err={history_error_norm[-1]:.6f}, "
                    f"sys_err={history_sys_norm[-1]:.6e}, cmd_err={np.linalg.norm(step_data['psi_cmd']):.6e}, "
                    f"alpha={step_data['alpha_next']:.6e}, Vsys={step_data['v_sys']:.6e}"
                )


        print('Simulation finished.')
        if history_error_norm:
            print(f'Final position error norm: {history_error_norm[-1]:.6f}')
            print(f'Mean position error norm: {np.mean(history_error_norm):.6f}')
            print(f'Max position error norm: {np.max(history_error_norm):.6f}')

    finally:
        if client_id != -1:
            sim.simxStopSimulation(client_id, sim.simx_opmode_oneshot)
            sim.simxFinish(client_id)

    if history_actual_pos:
        plot_results(
            history_actual_pos=history_actual_pos,
            history_desired_pos=history_desired_pos,
            history_error_norm=history_error_norm,
            history_sys_norm=history_sys_norm,
        )


if __name__ == '__main__':
    main()
