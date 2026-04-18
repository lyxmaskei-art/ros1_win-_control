import time

import matplotlib.pyplot as plt
import numpy as np
import sim  # CoppeliaSim 远程 API 客户端


SIGN_BI_POWER = 0.9  # sig^r(z) 中的 r


def sign_bi_power(z, power=SIGN_BI_POWER):
    """计算 sig^r(z) = sign(z) * |z|^r。"""
    z = np.asarray(z, dtype=float)
    abs_z = np.abs(z)
    power_term = np.where(abs_z == 0.0, 0.0, np.power(abs_z, power))
    return np.sign(z) * power_term



def nonlinear_activation(z, power=SIGN_BI_POWER, max_abs_exp=None):
    """计算 phi(z) = sig^r(z) * exp(|z|)。"""
    sig_r = sign_bi_power(z, power)
    abs_z = np.abs(np.asarray(z, dtype=float))
    if max_abs_exp is not None:
        abs_z = np.minimum(abs_z, max_abs_exp)
    return sig_r * np.exp(abs_z)


class UR3eKinematics:
    """UR3e 的正向运动学与雅可比矩阵。"""

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
            raise ValueError(f'Expected {self.num_joints} joint angles, got {theta.shape[0]}.')

        T_matrices = []
        T_current = np.eye(4, dtype=float)
        for i in range(self.num_joints):
            T_i = self.transformation_matrix(theta[i], self.a[i], self.d[i], self.alpha[i])
            T_current = T_current @ T_i
            T_matrices.append(T_current.copy())
        return T_matrices if return_intermediate else T_matrices[-1]

    def jacobian(self, theta):
        """返回完整 6x6 雅可比矩阵，其中前 3 行为平移部分。"""
        T_matrices = self.forward_kinematics(theta, return_intermediate=True)
        O_n = T_matrices[-1][:3, 3]
        J = np.zeros((6, self.num_joints), dtype=float)

        z0 = np.array([0.0, 0.0, 1.0], dtype=float)
        J[:3, 0] = np.cross(z0, O_n)
        J[3:, 0] = z0

        for i in range(1, self.num_joints):
            T_prev = T_matrices[i - 1]
            O_prev = T_prev[:3, 3]
            Z_prev = T_prev[:3, 2]
            J[:3, i] = np.cross(Z_prev, O_n - O_prev)
            J[3:, i] = Z_prev

        return J



def make_offset_from_initial_position(initial_position, scale):
    """使爱心轨迹在 t=0 时从当前末端执行器位置精确起步。"""
    heart_start_local = np.array([0.0, 0.0, 5.0], dtype=float)
    return np.asarray(initial_position, dtype=float) - scale * heart_start_local


class HeartTrajectory:
    """YZ 平面上的爱心轨迹。"""

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



def main():
    print('Program started, connecting to CoppeliaSim...')
    sim.simxFinish(-1)
    clientID = sim.simxStart('127.0.0.1', 19998, True, True, 5000, 5)
    if clientID == -1:
        print('Connection failed.')
        return
    print('Connection successful.')

    sim.simxSynchronous(clientID, True)
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)

    try:
        joint_handles = [sim.simxGetObjectHandle(clientID, f'UR3e_joint{i}', sim.simx_opmode_blocking)[1] for i in range(1, 7)]
        print('Successfully got all joint handles.')
    except Exception as e:
        print(f'Failed to get joint handles: {e}')
        sim.simxFinish(clientID)
        return

    robot = UR3eKinematics()
    u = 100.0
    lambda_reg = 1e-5
    tau = 0.005

    t_start = 0.0
    t_end = 10.0
    N = int((t_end - t_start) / tau)
    t_discrete = np.linspace(t_start, t_end, N + 1)

    # 在此修改 theta_current。爱心轨迹起点会自动跟随该初始姿态。
    #theta_current = np.array([2 * np.pi / 9, -389 * np.pi / 900, -2177 * np.pi / 3600, 53 * np.pi / 1440, 5 * np.pi / 18, 0], dtype=float)
    theta_current = np.array([np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0, 0], dtype=float)

    initial_pos = robot.forward_kinematics(theta_current)[:3, 3]
    trajectory_scale = 0.008
    trajectory_offset = make_offset_from_initial_position(initial_pos, scale=trajectory_scale)
    trajectory = HeartTrajectory(duration=t_end, scale=trajectory_scale, offset=trajectory_offset)

    print('Initial end-effector position:', np.round(initial_pos, 6))
    print('Auto-computed trajectory offset:', np.round(trajectory_offset, 6))
    print('Moving robot to initial posture...')
    for i in range(robot.num_joints):
        sim.simxSetJointTargetPosition(clientID, joint_handles[i], theta_current[i], sim.simx_opmode_oneshot)

    for _ in range(10):
        sim.simxSynchronousTrigger(clientID)
        time.sleep(tau)

    history_actual_pos = []
    history_desired_pos = []
    history_error_norm = []

    print('Starting trajectory tracking...')
    for k in range(N):
        if sim.simxGetConnectionId(clientID) == -1:
            print(f'Connection lost at step {k}.')
            break

        tk = t_discrete[k]
        r_d, r_dot_d = trajectory.get_pose(tk)
        pos_current = robot.forward_kinematics(theta_current)[:3, 3]
        psi_error = pos_current - r_d

        J_k = robot.jacobian(theta_current)[:3, :]

        v_k = 0.5 * np.sum(psi_error ** 2)
        phi_vk = nonlinear_activation(v_k)
        J_T_f = J_k.T @ psi_error
        denominator = np.sum(J_T_f ** 2) + lambda_reg
        fT_dfdt = -psi_error.T @ r_dot_d
        theta_dot_k = -(J_T_f / denominator) * (fT_dfdt + u * phi_vk)

        theta_current = theta_current + tau * theta_dot_k

        history_actual_pos.append(pos_current)
        history_desired_pos.append(r_d)
        history_error_norm.append(np.linalg.norm(psi_error))

        for i in range(robot.num_joints):
            sim.simxSetJointTargetPosition(clientID, joint_handles[i], theta_current[i], sim.simx_opmode_oneshot)

        sim.simxSynchronousTrigger(clientID)

    sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
    sim.simxFinish(clientID)
    print('Simulation finished, connection closed.')


if __name__ == '__main__':
    main()
