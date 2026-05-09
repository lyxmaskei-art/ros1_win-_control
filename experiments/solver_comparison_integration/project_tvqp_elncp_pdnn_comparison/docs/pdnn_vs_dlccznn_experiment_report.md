---
title: PDNN 与 DLCCZNN 在 TVQP + ELNCP 重复运动学中的对照实验
created: 2026-05-07
project: project_tvqp_elncp_pdnn_comparison
---

# PDNN 与 DLCCZNN 在 TVQP + ELNCP 重复运动学中的对照实验

## 实验目的

本实验固定上层 Method 2 连续时变二次规划与 ELNCP 不等式约束处理方式，只替换底层神经动力学求解器。对照对象为已有的 TVQP + ELNCP + DLCCZNN 方法，新建基线为 TVQP + ELNCP + PDNN 方法。

## 统一残差

设求解变量为

$$
y(t)=
\begin{bmatrix}
u(t)\\
\lambda(t)\\
\omega(t)
\end{bmatrix},
$$

其中 $u(t)$ 为关节速度，$\lambda(t)$ 为任务等式约束乘子，$\omega(t)$ 为 ELNCP 引入的非负互补变量。统一残差写为

$$
F(y,t)=
\begin{bmatrix}
Wu+\mu(q-q_0)-J(q)^T\lambda+\sigma\omega\\
J(q)u-b(t,q)\\
\psi_\varepsilon(s(u),\omega)
\end{bmatrix}.
$$

ELNCP 的边界松弛量为

$$
s_i(u_i)=\min(u_i-\xi_i^-,\xi_i^+-u_i),
$$

扰动 Fischer Burmeister 函数为

$$
\psi_\varepsilon(s_i,\omega_i)
=s_i+\omega_i-\sqrt{s_i^2+\omega_i^2+\varepsilon}.
$$

## PDNN 求解动态

本项目采用残差能量型 PDNN：

$$
V(y,t)=\frac12 F(y,t)^T F(y,t),
$$

于是

$$
\dot y=-\rho \nabla_y V(y,t)
=-\rho F_y(y,t)^T F(y,t).
$$

当需要离散执行时，使用与控制更新一致的显式欧拉格式：

$$
y_{k+1}=y_k-\tau\rho F_y(y_k,t_k)^TF(y_k,t_k).
$$

若使用 $N_p$ 个 PDNN 内部小步，则

$$
h_p=\frac{\tau}{N_p},
\qquad
y^{r+1}=y^r-h_p\rho F_y(y^r,t_k)^TF(y^r,t_k).
$$

这组实验中，$N_p=1$ 表示与控制周期同频的一步 PDNN，$N_p>1$ 用于测试 PDNN 连续模型在更细欧拉积分下的极限精度。

## 优选图形

简单图形筛选得到的前三个图形为：

not selected

## 结果总表

完整 CSV 位于：

`C:\Users\lyx\Desktop\sci\experiments\solver_comparison_integration\project_tvqp_elncp_pdnn_comparison\results\formal_offline\pdnn_vs_dlccznn_all_summaries.csv`

下面列出按平均位置误差排序的前若干项。

| Experiment | Trajectory | Solver | PDNN steps | Mean position error (m) | Final drift (rad) | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| single_heart_method2_pdnn_offline | heart | pdnn | 1 | 2.641467e-02 | 3.011737e-02 | 0.240 |

## 初步结论

1. 若 PDNN 在 $N_p=1$ 时明显弱于 DLCCZNN，但增加 $N_p$ 后逐步接近，则说明 PDNN 对连续积分步长更敏感，而 DLCCZNN 更适合直接离散控制周期。
2. 若 PDNN 需要大量内部欧拉小步才能达到相近误差，则其运行成本会随 $N_p$ 线性增长，这可以作为凸显 DLCCZNN 离散结构优势的对照证据。
3. 简单图形的误差通常受轨迹曲率、起止速度连续性和冗余关节漂移分配共同影响，不能只按几何形状复杂度判断。
