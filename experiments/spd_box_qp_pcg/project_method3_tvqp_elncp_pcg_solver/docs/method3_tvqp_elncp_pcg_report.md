---
title: Method3 连续时变 ELNCP PCG 实验记录
created: 2026-05-08
project: project_method3_tvqp_elncp_pcg_solver
links:
  - "[[Method3a PCG Solver]]"
  - "[[TVQP ELNCP PDNN DLCCZNN Comparison]]"
  - "[[连续时变QP与重复运动学]]"
---

# Method3 连续时变 ELNCP PCG 实验记录

## 1. 目的

本次实验把原来的 Method3a 位置与漂移二次目标改写为连续时变 box QP 形式，并把速度不等式约束由求解后裁剪改成 ELNCP 风格的 KKT 互补边界处理。每个采样时刻的自由变量子问题用 PCG 求解，实验流程保持与原 Method3a PCG 项目一致，包含 offline 和 live 两类验证。

## 2. 从 Method3 目标到时变 box QP

设关节角为

$$
q(t)\in\mathbb{R}^{6},
$$

末端位置为

$$
x(q(t))\in\mathbb{R}^{3},
$$

期望轨迹为

$$
x_d(t),\quad \dot{x}_d(t).
$$

定义位置误差和漂移误差为

$$
e_p(t)=x(q(t))-x_d(t),
$$

$$
e_d(t)=q(t)-q(0).
$$

在采样周期

$$
\tau=0.005\ {\rm s}
$$

下，Method3 的核心思想是同时压低下一步位置误差和下一步关节漂移。用一阶近似

$$
x(q+\tau u)\approx x(q)+\tau J(q)u
$$

得到下一步位置误差近似为

$$
e_p(t+\tau)\approx e_p(t)+\tau\left(J(q(t))u(t)-\dot{x}_d(t)\right).
$$

若加入任务反馈项

$$
k_p e_p(t),
$$

则被压低的位置通道为

$$
e_p(t)+\tau\left(J(q(t))u(t)-\dot{x}_d(t)+k_p e_p(t)\right).
$$

漂移通道为

$$
e_d(t)+\tau u(t).
$$

于是 Method3 的连续时变二次规划可以写成

$$
\min_{u(t)}
\frac{w_p}{2}
\left\|
e_p(t)+\tau\left(J(q(t))u(t)-\dot{x}_d(t)+k_p e_p(t)\right)
\right\|_2^2
+
\frac{w_d}{2}
\left\|
e_d(t)+\tau u(t)
\right\|_2^2
+
\frac{\epsilon_Q}{2}\|u(t)\|_2^2 .
$$

去掉与

$$
u(t)
$$

无关的常数项，并整体除去共同的

$$
\tau^2
$$

因子，可得标准二次型

$$
\min_{u(t)}
\frac{1}{2}u(t)^TQ(t)u(t)+c(t)^Tu(t),
$$

其中

$$
Q(t)=w_pJ(q(t))^TJ(q(t))+w_dI+\epsilon_QI,
$$

$$
c(t)=w_pJ(q(t))^T
\left[
\left(\frac{1}{\tau}+k_p\right)e_p(t)-\dot{x}_d(t)
\right]
+
w_d\frac{e_d(t)}{\tau}.
$$

由于

$$
w_d>0,\quad \epsilon_Q>0,
$$

所以

$$
Q(t)\succ0.
$$

因此每个采样时刻对应一个严格凸 box QP。

## 3. 速度边界与 ELNCP 互补条件

动态速度边界仍沿用原实验设置：

$$
\xi_i^-(t)=\max\left(-\dot q_{\max},\frac{\eta}{\tau}(q_i^- - q_i(t))\right),
$$

$$
\xi_i^+(t)=\min\left(\dot q_{\max},\frac{\eta}{\tau}(q_i^+ - q_i(t))\right).
$$

新方法不再把无约束解直接裁剪，而是求解 box QP：

$$
\min_{u(t)}
\frac{1}{2}u(t)^TQ(t)u(t)+c(t)^Tu(t)
$$

满足

$$
\xi^-(t)\le u(t)\le \xi^+(t).
$$

引入下界乘子和上界乘子：

$$
\lambda^-(t)\ge0,\quad \lambda^+(t)\ge0.
$$

KKT 驻点条件为

$$
Q(t)u(t)+c(t)-\lambda^-(t)+\lambda^+(t)=0.
$$

边界互补条件为

$$
0\le u_i(t)-\xi_i^-(t)\perp \lambda_i^-(t)\ge0,
$$

$$
0\le \xi_i^+(t)-u_i(t)\perp \lambda_i^+(t)\ge0.
$$

使用扰动 Fischer Burmeister 函数表示互补关系：

$$
\psi_\varepsilon(a,b)=a+b-\sqrt{a^2+b^2+\varepsilon}.
$$

则 ELNCP 残差写成

$$
F(y,t)=
\begin{bmatrix}
Q(t)u(t)+c(t)-\lambda^-(t)+\lambda^+(t)\\
\psi_\varepsilon(u(t)-\xi^-(t),\lambda^-(t))\\
\psi_\varepsilon(\xi^+(t)-u(t),\lambda^+(t))
\end{bmatrix},
$$

其中

$$
y(t)=
\begin{bmatrix}
u(t)\\
\lambda^-(t)\\
\lambda^+(t)
\end{bmatrix}.
$$

## 4. PCG 离散求解方式

如果直接对

$$
F(y,t)=0
$$

做显式 Euler 残差跟踪，短周期下会出现速度变量暂时离开可行域的问题。第一次 1 s 冒烟测试已经验证该写法不稳，会导致关节快速触边。

因此最终代码采用更稳的 KKT active set 等价实现：

1. 根据当前梯度

$$
g(t)=Q(t)u(t)+c(t)
$$

识别下界活跃集、上界活跃集和自由集。

2. 对活跃边界固定

$$
u_i=\xi_i^-
$$

或

$$
u_i=\xi_i^+.
$$

3. 对自由变量集合

$$
\mathcal{F}
$$

求解 reduced SPD 线性系统

$$
Q_{\mathcal{F}\mathcal{F}}u_{\mathcal{F}}
=
-c_{\mathcal{F}}
-Q_{\mathcal{F}\mathcal{A}}u_{\mathcal{A}}.
$$

4. 上式使用 warm started PCG 求解。

5. 根据最终梯度恢复互补乘子：

$$
\lambda_i^-=\max(g_i,0),\quad i\in\mathcal{A}^-,
$$

$$
\lambda_i^+=\max(-g_i,0),\quad i\in\mathcal{A}^+.
$$

这一步仍然是 ELNCP/KKT 边界处理，不是原 Method3a 的“无约束 PCG 解完再 clip”。区别在于原方法没有显式维护互补乘子，也没有在 reduced KKT 子问题上重新求自由变量。

## 5. 参数

为了不人为改变 Method3 的原始行为，本次默认参数沿用原 Method3a PCG 最佳 live 邻域：

$$
k_p=198,
$$

$$
w_p=15000,
$$

$$
w_d=5\times10^{-4},
$$

$$
\epsilon_Q=10^{-10}.
$$

新增求解器参数为

$$
k_{\rm pcg}^{\max}=8,
$$

$$
k_{\rm act}^{\max}=16,
$$

$$
\varepsilon_{\rm ncp}=10^{-10}.
$$

说明：标准实验的 JSON 汇总里保留了早期不稳定显式残差跟踪原型中的 `residual_gamma` 字段。最终代码已经删除该参数，实际求解采用 active set KKT 与 reduced PCG，不使用该字段。

## 6. Offline 20 s 结果

结果目录：

[offline 20s 标准结果](C:/Users/lyx/Desktop/sci/code/project_method3_tvqp_elncp_pcg_solver/results/standard_heart20_default/offline)

核心指标如下：

| 指标 | 数值 |
|---|---:|
| 平均位置误差 | 2.0433668773123088e-04 m |
| 最大位置误差 | 3.074013271522352e-04 m |
| 最终位置误差 | 6.079975503728618e-07 m |
| 最大单关节绝对漂移 | 4.125530800102486e-01 rad |
| 最终关节漂移范数 | 4.6671803940884594e-07 rad |
| 平均 ELNCP 残差 | 1.9095206686859357e-08 |
| 最大 ELNCP 残差 | 1.4811182048296623e-07 |
| 平均 PCG 迭代数 | 9.51525 |
| 最大 PCG 迭代数 | 13 |
| 平均 active set 迭代数 | 1.5305 |
| 最大 active set 迭代数 | 2 |

与原 Method3a PCG offline 对比：

| 方法 | 平均位置误差 | 最终位置误差 | 最终关节漂移范数 | 平均求解迭代 |
|---|---:|---:|---:|---:|
| 原 Method3a PCG | 2.0433671160163258e-04 m | 6.079450556495063e-07 m | 2.2841365565166473e-05 rad | 4.44375 |
| Method3 TVQP ELNCP PCG | 2.0433668773123088e-04 m | 6.079975503728618e-07 m | 4.6671803940884594e-07 rad | 9.51525 |

结论是：offline 下新方法保持了原 Method3a PCG 的位置精度，同时最终漂移恢复更强，但求解迭代次数约增加到原来的两倍左右。

## 7. Live 10 s 结果

结果目录：

[live 10s 标准结果](C:/Users/lyx/Desktop/sci/code/project_method3_tvqp_elncp_pcg_solver/results/standard_heart10_default/live)

连接端口：

$$
19997.
$$

核心指标如下：

| 指标 | 数值 |
|---|---:|
| 平均位置误差 | 1.302305246026711e-04 m |
| 最大位置误差 | 2.3121506529864e-04 m |
| 最终位置误差 | 5.8159545931849617e-05 m |
| 最大单关节绝对漂移 | 4.1294002532958984e-01 rad |
| 最终关节漂移范数 | 4.170262650664861e-04 rad |
| 平均 ELNCP 残差 | 6.023431456961759e-09 |
| 最大 ELNCP 残差 | 1.2939632773619373e-07 |
| 平均 PCG 迭代数 | 10.7005 |
| 最大 PCG 迭代数 | 13 |
| 平均 active set 迭代数 | 1.887 |
| 最大 active set 迭代数 | 2 |

与原 Method3a PCG best live 对比：

| 方法 | 平均位置误差 | 最终位置误差 | 最终关节漂移范数 | 平均求解迭代 |
|---|---:|---:|---:|---:|
| 原 Method3a PCG best live | 1.3023023638833518e-04 m | 5.814786417501239e-05 m | 3.197148817648393e-04 rad | 4.6655 |
| Method3 TVQP ELNCP PCG live | 1.302305246026711e-04 m | 5.8159545931849617e-05 m | 4.170262650664861e-04 rad | 10.7005 |

结论是：live 下新方法的位置精度与原 best live 基本等价，但最终关节漂移略差，且计算迭代开销更高。

## 8. 复杂度

设关节数为

$$
n=6,
$$

任务空间维数为

$$
m=3,
$$

PCG 平均迭代数为

$$
k_{\rm pcg},
$$

active set 平均外层次数为

$$
k_{\rm act}.
$$

构造

$$
J^TJ
$$

的代价约为

$$
\mathcal{O}(mn^2).
$$

每次 reduced PCG 的矩阵向量乘代价最多为

$$
\mathcal{O}(n^2).
$$

因此单周期主复杂度为

$$
\mathcal{O}(mn^2+k_{\rm act}k_{\rm pcg}n^2).
$$

当前 offline 的平均统计为

$$
k_{\rm pcg}\approx9.51525,\quad k_{\rm act}\approx1.5305.
$$

当前 live 的平均统计为

$$
k_{\rm pcg}\approx10.7005,\quad k_{\rm act}\approx1.887.
$$

相比原 Method3a PCG 的

$$
\mathcal{O}(mn^2+kn^2),
$$

新方法多了 active set/ELNCP 边界判别层。因此它的优势不在低复杂度，而在更严格地解释边界约束和 KKT 互补结构。

## 9. 判断

本次转换是可行的，代码和实验都已经跑通。

从结果看：

1. offline 指标很好，位置误差与原 Method3a PCG 几乎完全一致，最终关节漂移更小。
2. live 指标没有明显优于原 Method3a PCG，位置误差几乎一致，但最终漂移略差。
3. 新方法的 ELNCP 残差非常小，说明约束残差本身处理得干净。
4. 新方法的计算代价高于原 Method3a PCG，不适合作为低复杂度卖点。
5. 如果论文要强调 Method3，这个版本更适合作为“约束处理严格化对照”，而不是主方法。

## 10. 文件索引

代码文件：

`C:\Users\lyx\Desktop\sci\experiments\spd_box_qp_pcg\project_method3_tvqp_elncp_pcg_solver\scripts\run_offline.py` ? `run_live.py`

对比表：

[comparison_with_method3a_pcg.csv](C:/Users/lyx/Desktop/sci/code/project_method3_tvqp_elncp_pcg_solver/results/comparison_with_method3a_pcg.csv)

offline 图：

[offline 图片目录](C:/Users/lyx/Desktop/sci/code/project_method3_tvqp_elncp_pcg_solver/results/standard_heart20_default/offline)

live 图：

[live 图片目录](C:/Users/lyx/Desktop/sci/code/project_method3_tvqp_elncp_pcg_solver/results/standard_heart10_default/live)
