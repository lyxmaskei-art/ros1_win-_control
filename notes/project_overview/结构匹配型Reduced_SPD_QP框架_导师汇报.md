---
title: 结构匹配型 Reduced SPD QP 机器人控制框架阶段汇报
date: 2026-05-08
type: advisor-report
tags:
  - robotics
  - quadratic-programming
  - drift-free
  - PCG
  - reduced-SPD-QP
---

# 结构匹配型 Reduced SPD QP 机器人控制框架阶段汇报

> [!abstract]
> 本阶段工作的核心发现是：对于一类速度级机器人控制 QP，尤其是以任务跟踪误差、关节漂移抑制和关节速度边界为核心的冗余机械臂控制问题，可以不再采用“先写出通用 QP，再选择通用求解器”的传统路线，而是从一开始就面向求解结构重构 QP，使问题落入严格凸 reduced SPD 子问题，并进一步用 warm started PCG 高效求解。该思路目前已在 UR3e drift free repetitive motion 问题中完成 offline 和 CoppeliaSim live 验证，表现出较好的轨迹跟踪精度、角度漂移恢复能力和计算结构清晰性。

## 1. 研究背景与问题来源

冗余机械臂的重复运动规划中，一个长期存在的问题是 **joint angle drift**。即使末端轨迹能够周期性闭合，关节角也可能在多个周期后逐渐偏离初始构型，从而带来关节限位风险、能耗增加和重复任务稳定性下降。

经典处理思路通常把该问题写成速度级逆运动学或 QP 问题，再通过 PDNN、RNN、ZNN、active set、OSQP 等方法求解。该路线的重点通常在于：

1. 如何建立 drift free 约束或漂移抑制指标。
2. 如何处理关节速度和关节位置边界。
3. 如何选择一个能够实时求解 QP 的求解器。

我们在实验过程中发现，仅仅“更换求解器”并不能稳定带来更好的结果。比如 DLCCZNN 在一些 Method2 结构中具有理论优势，但在当前 UR3e live 仿真里，其实际精度并没有超过 Method3 PCG。进一步分析后发现，关键不只是求解器本身，而是 **QP 问题结构是否与求解器天然匹配**。

因此，本阶段提出的核心思路是：

> 与其在一个通用 QP 上被动选择求解器，不如主动重构机器人控制 QP，使其形成适合 PCG 求解的 reduced SPD 结构。

这一思路可以概括为：

$$
\text{Robotic task}
\rightarrow
\text{structure matched QP reformulation}
\rightarrow
\text{ELNCP or active set reduction}
\rightarrow
\text{reduced SPD system}
\rightarrow
\text{warm started PCG}.
$$

## 2. 目前提出的核心范式

以当前 drift free repetitive motion 为样板问题，设关节角为

$$
q(t)\in\mathbb{R}^6,
$$

末端位置为

$$
x(q(t))\in\mathbb{R}^3,
$$

期望轨迹为

$$
x_d(t),\quad \dot{x}_d(t).
$$

定义位置误差和关节漂移为

$$
e_p(t)=x(q(t))-x_d(t),
$$

$$
e_d(t)=q(t)-q_0.
$$

在采样周期

$$
\tau=0.005\ {\rm s}
$$

下，使用一阶近似

$$
x(q+\tau\dot q)\approx x(q)+\tau J(q)\dot q.
$$

因此下一步位置误差可近似写成

$$
e_p(t+\tau)
\approx
e_p(t)+\tau\left(J(q(t))\dot q(t)-\dot{x}_d(t)\right).
$$

为了增强任务误差收敛，引入反馈项

$$
k_pe_p(t).
$$

于是被压低的位置通道写成

$$
e_p(t)+\tau\left(J(q(t))\dot q(t)-\dot{x}_d(t)+k_pe_p(t)\right).
$$

关节漂移通道写成

$$
q(t)-q_0+\tau\dot q(t).
$$

由此得到本阶段提出的核心优化形式：

$$
\min_{\dot q}
\frac{w_p}{2}
\left\|
e_p+\tau\left(J\dot q-\dot x_d+k_pe_p\right)
\right\|^2
+
\frac{w_d}{2}
\left\|
q-q_0+\tau\dot q
\right\|^2
+
\frac{\epsilon}{2}\|\dot q\|^2 .
$$

对应动态速度边界为

$$
\xi^-(t)\le \dot q(t)\le \xi^+(t).
$$

该形式的意义在于：它并不是单独最小化当前速度，也不是单独压制漂移，而是同时压低 **下一步任务位置误差** 和 **下一步关节角漂移**。这使 drift free 目标与离散控制周期直接对应，更符合实际机器人控制中的采样更新形式。

## 3. 从原问题到 SPD box QP

去掉与 $\dot q$ 无关的常数项，并按共同尺度整理后，上式可写为标准二次型：

$$
\min_{\dot q}
\frac{1}{2}\dot q^TQ\dot q+c^T\dot q,
$$

其中

$$
Q=w_pJ^TJ+w_dI+\epsilon_QI,
$$

$$
c=
w_pJ^T
\left[
\left(\frac{1}{\tau}+k_p\right)e_p-\dot x_d
\right]
+
w_d\frac{q-q_0}{\tau}.
$$

由于

$$
w_d>0,\quad \epsilon_Q>0,
$$

所以有

$$
Q\succ0.
$$

因此，每个采样时刻对应一个严格凸 SPD box QP：

$$
\min_{\dot q}
\frac{1}{2}\dot q^TQ\dot q+c^T\dot q
$$

subject to

$$
\xi^-(t)\le \dot q(t)\le \xi^+(t).
$$

这一步是目前工作的关键：我们不是把一般 QP 强行交给 PCG，而是通过目标函数结构设计，使该问题自然形成 PCG 擅长处理的 SPD 结构。

## 4. 约束处理与 reduced SPD PCG

PCG 的本质适用对象是对称正定线性系统：

$$
Ax=b,\quad A=A^T,\quad A\succ0.
$$

对于无约束 SPD QP：

$$
\min_x\frac{1}{2}x^TQx+c^Tx,
$$

其最优性条件为

$$
Qx=-c.
$$

因此可以直接使用 PCG。

对于带 box 约束的问题，边界活跃后可以将变量分为活跃集 $\mathcal A$ 和自由集 $\mathcal F$。活跃变量固定在上下界，自由变量满足 reduced SPD 系统：

$$
Q_{\mathcal F\mathcal F}\dot q_{\mathcal F}
=
-c_{\mathcal F}
-Q_{\mathcal F\mathcal A}\dot q_{\mathcal A}.
$$

因为

$$
Q\succ0,
$$

所以任意主子矩阵满足

$$
Q_{\mathcal F\mathcal F}\succ0.
$$

因此 reduced 子问题仍然可以由 PCG 求解。

对于更一般的线性等式或活跃不等式约束，可以进一步写成

$$
Cx=d.
$$

通过零空间降维

$$
x=x_p+Nz,\quad CN=0,
$$

代回目标函数得到

$$
\min_z
\frac{1}{2}z^T(N^THN)z+
\left[N^T(Hx_p+c)\right]^Tz.
$$

只要

$$
H\succ0
$$

且 $N$ 满列秩，就有

$$
N^THN\succ0.
$$

这说明该思路可以从单纯 box QP 扩展为 **ELNCP or active set based reduced SPD QP**。但需要强调的是，该框架并不声称所有 QP 都能无损转化为 SPD box QP，而是针对一类机器人速度级凸 QP，利用结构重构和约束降维得到 PCG 可解的 reduced SPD 子问题。

## 5. 当前实验结果

本阶段使用 UR3e 心形重复轨迹对该方法进行了 offline 和 CoppeliaSim live 验证。主要参数为：

$$
\tau=0.005,\quad
k_p=198,\quad
w_p=15000,\quad
w_d=5\times10^{-4}.
$$

### 5.1 Offline 20 s 结果

| 指标 | 数值 |
|---|---:|
| 平均位置误差 | \(2.0434\times10^{-4}\ {\rm m}\) |
| 最大位置误差 | \(3.0740\times10^{-4}\ {\rm m}\) |
| 最终位置误差 | \(6.0800\times10^{-7}\ {\rm m}\) |
| 最大单关节绝对漂移 | \(4.1255\times10^{-1}\ {\rm rad}\) |
| 最终关节漂移范数 | \(4.6672\times10^{-7}\ {\rm rad}\) |
| 平均 ELNCP 残差 | \(1.9095\times10^{-8}\) |
| 平均 PCG 迭代次数 | \(9.5153\) |
| 平均 active set 迭代次数 | \(1.5305\) |

offline 结果说明，该方法在完整 20 s 周期后能够将最终位置误差压到亚微米量级，并将最终关节漂移范数压到 $10^{-7}$ rad 量级。

### 5.2 CoppeliaSim live 10 s 结果

| 指标 | 数值 |
|---|---:|
| 平均位置误差 | \(1.3023\times10^{-4}\ {\rm m}\) |
| 最大位置误差 | \(2.3122\times10^{-4}\ {\rm m}\) |
| 最终位置误差 | \(5.8153\times10^{-5}\ {\rm m}\) |
| 最大单关节绝对漂移 | \(4.1318\times10^{-1}\ {\rm rad}\) |
| 最终关节漂移范数 | \(4.1728\times10^{-4}\ {\rm rad}\) |
| 平均 ELNCP 残差 | \(5.8797\times10^{-9}\) |
| 平均 PCG 迭代次数 | \(10.6775\) |
| 平均 active set 迭代次数 | \(1.8815\) |

live 结果说明，在同步仿真闭环中，该方法仍然能够保持较低的位置误差和较好的 drift free 恢复效果。与此前 Method2 DLCCZNN live 结果相比，当前 Method3 PCG 在平均位置误差和最终关节漂移方面均更优。

### 5.3 与现有实验线的关系

目前已有实验显示：

1. Method2 DLCCZNN 保留了较强的神经动力学求解器意义，但在当前 UR3e live 心形轨迹中，精度不如 Method3 PCG。
2. 原 Method3a PCG 已经表现出较好的位置和漂移性能，但边界处理更接近求解后裁剪。
3. 当前 Method3 TVQP ELNCP PCG 在保留 Method3 高精度结构的同时，引入了更严格的边界互补解释和 reduced PCG 求解。

因此，目前更合理的论文主线不是“DLCCZNN 一定优于 PCG”，而是：

> 对于 drift free repetitive motion 这类结构明确的机器人 QP，主动构造 reduced SPD 结构并使用 warm started PCG，可以获得更直接、更高效、更稳定的实时求解效果。

### 5.4 计算复杂度分析

为了更清楚地说明该范式的计算优势，设关节维数为

$$
n,
$$

任务空间维数为

$$
m,
$$

自由变量维数为

$$
n_f\le n,
$$

PCG 平均迭代次数为

$$
k_{\rm pcg},
$$

active set 或 ELNCP 边界识别平均迭代次数为

$$
k_{\rm a}.
$$

若采用通用 QP 求解器直接处理 KKT 线性系统，通常需要求解

$$
\begin{bmatrix}
H & A^T\\
A & 0
\end{bmatrix}
\begin{bmatrix}
x\\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
-c\\
b
\end{bmatrix}.
$$

若该 KKT 系统总维数记为

$$
d=n+p+s,
$$

其中 \(p\) 为等式约束数量，\(s\) 为显式不等式或互补变量数量，则直接分解的复杂度一般为

$$
\mathcal O(d^3).
$$

PDNN 方法通常不显式分解 KKT 系统，而是通过连续动力学或其欧拉离散形式迭代逼近最优解。若每次内部更新需要计算 \(Hx\)、\(Ax\) 和约束残差，则单次更新复杂度可写为

$$
\mathcal O\left(n^2+(p+s)n\right).
$$

若一个控制周期内使用

$$
N_{\rm pdnn}
$$

次内部更新，则每周期复杂度为

$$
\mathcal O\left(
N_{\rm pdnn}\left[n^2+(p+s)n\right]
\right).
$$

DLCCZNN 的一次离散更新可理解为对 KKT 残差

$$
r_k=M_k y_k-b_k
$$

进行显式离散反馈。若 \(y_k\in\mathbb R^d\)，且不进行矩阵分解，则单次更新复杂度为

$$
\mathcal O(d^2).
$$

若一个控制周期内使用

$$
N_{\rm d}
$$

次 DLCCZNN 内部更新，则复杂度为

$$
\mathcal O(N_{\rm d}d^2).
$$

当 \(N_{\rm d}=1\) 时，DLCCZNN 具有

$$
\mathcal O(d^2)
$$

的低复杂度优势；但如果为了提高 live 精度而显著增大 \(N_{\rm d}\)，实际计算代价会随内部更新次数线性增加。

对于原 Method3a PCG，核心子问题可写为

$$
Q\dot q=-c,
$$

其中

$$
Q=w_pJ^TJ+w_dI+\epsilon_QI\succ0.
$$

构造 \(Q\) 的复杂度为

$$
\mathcal O(mn^2).
$$

若采用稠密矩阵形式做 PCG，每次矩阵向量乘法复杂度为

$$
\mathcal O(n^2),
$$

因此每个控制周期复杂度为

$$
\mathcal O\left(mn^2+k_{\rm pcg}n^2\right).
$$

若采用矩阵自由形式

$$
Qv=w_pJ^T(Jv)+w_dv+\epsilon_Qv,
$$

则每次 PCG 迭代可写为

$$
\mathcal O(mn),
$$

对应复杂度为

$$
\mathcal O\left(mn^2+k_{\rm pcg}mn\right).
$$

对于当前提出的 Method3 TVQP ELNCP PCG，边界变量被识别后，PCG 只作用在自由变量子空间：

$$
Q_{\mathcal F\mathcal F}\dot q_{\mathcal F}
=
-c_{\mathcal F}
-Q_{\mathcal F\mathcal A}\dot q_{\mathcal A}.
$$

因此，稠密实现下每周期复杂度可写为

$$
\mathcal O\left(
mn^2+
k_{\rm a}\left[n+k_{\rm pcg}n_f^2\right]
\right).
$$

矩阵自由实现下可写为

$$
\mathcal O\left(
mn^2+
k_{\rm a}\left[n+k_{\rm pcg}mn_f\right]
\right).
$$

由于

$$
n_f\le n,
$$

且本实验中平均 active set 迭代次数约为

$$
k_{\rm a}\approx1.5\sim1.9,
$$

平均 PCG 迭代次数约为

$$
k_{\rm pcg}\approx9.5\sim10.7,
$$

所以该方法的主要代价集中在少量低维 SPD 系统的矩阵向量乘法上，而不是通用 KKT 系统的三次复杂度分解。

在当前 UR3e 实验中，

$$
n=6,\quad m=3.
$$

因此小维度下不同求解器的绝对运行时间差距不会像高维系统中那样显著。本文更应强调的是结构优势：当自由度增加、约束数量增加、QP 在高频控制中连续出现且可以 warm start 时，reduced SPD PCG 能避免通用 QP 分解的

$$
\mathcal O(d^3)
$$

代价，并把每周期核心求解压缩为

$$
\mathcal O(k_{\rm a}k_{\rm pcg}n_f^2)
$$

或矩阵自由形式下的

$$
\mathcal O(k_{\rm a}k_{\rm pcg}mn_f).
$$

因此，本阶段方法的复杂度优势不是来自“PCG 永远比所有方法快”，而是来自 **机器人 QP 结构重构以后，求解对象从一般 KKT 系统变成了低维 reduced SPD 系统**。

## 6. 与传统方法的区别

传统 QP 机器人控制通常采用如下路线：

$$
\text{task formulation}
\rightarrow
\text{generic QP}
\rightarrow
\text{generic solver}.
$$

本阶段提出的路线是：

$$
\text{task formulation}
\rightarrow
\text{solver aware QP reformulation}
\rightarrow
\text{reduced SPD structure}
\rightarrow
\text{warm started PCG}.
$$

两者区别在于：传统方法更关注“选择什么求解器”，而本方法更关注“如何把机器人任务重构为适合求解器的结构”。

这也是目前认为该方向具有进一步发展潜力的主要原因。

## 7. 初步贡献总结

目前可以归纳出三个主要贡献：

### 7.1 提出 one step position drift coupled QP

将下一步位置误差和下一步关节漂移同时放入同一个二次目标中，使任务跟踪与 drift free 恢复在离散控制周期内直接耦合。

### 7.2 构造严格凸 SPD box QP

通过目标函数设计使 Hessian 满足

$$
Q=w_pJ^TJ+w_dI+\epsilon_QI\succ0,
$$

从而为 PCG 求解提供结构基础。

### 7.3 形成 reduced SPD PCG 求解框架

通过 active set 或 ELNCP 思路识别边界约束，并对自由变量或零空间变量形成 reduced SPD 子问题，使 PCG 不再只是一个普通线性代数工具，而是成为机器人 QP 结构匹配求解框架的核心。

## 8. 适用范围与边界

该框架适合以下问题：

1. 速度级逆运动学。
2. drift free repetitive motion。
3. 关节速度和位置边界主导的机器人控制。
4. 任务误差可以软化为二次目标的问题。
5. 姿态恢复、关节漂移抑制和 null space regulation。
6. 高频实时控制中连续出现、可 warm start 的 QP 序列。

该框架不应过度推广到以下问题：

1. 非凸 QP。
2. 强接触互补约束。
3. 摩擦锥和复杂接触动力学。
4. 严格层级优先级不可软化的 HQP。
5. 需要全局非线性避障保证的问题。

因此，更严谨的表述是：

> 一类机器人速度级凸 QP，尤其是任务软目标和边界主导约束问题，可以通过 structure matched reformulation 与 ELNCP/active set reduction 转化为 reduced SPD 子问题，并由 warm started PCG 高效求解。

而不是：

> 所有 QP 都可以转化为 SPD box QP 并由 PCG 求解。

## 9. 后续计划

为了将该方向进一步发展为可投稿的完整工作，后续建议补充以下内容：

1. **真实 UR3e 实验**：验证真实通信周期、执行误差和传感噪声下的 drift free 效果。
2. **强 baseline 对比**：加入 PDNN、DLCCZNN、OSQP、qpOASES、传统伪逆法、原 Method3a clip 版本。
3. **鲁棒性实验**：加入关节测量噪声、速度执行噪声、轨迹扰动和采样周期变化。
4. **多轨迹实验**：在心形轨迹以外，加入圆、直线、Lissajous、空间曲线等轨迹。
5. **理论补充**：给出 reduced SPD 条件、active set 约束切换稳定性分析、复杂度分析和失败边界。
6. **框架化总结**：将不同机器人 QP 分为原生 SPD box、降维后 SPD、不可直接转化三类，形成 problem taxonomy。

## 10. 目前判断

从当前结果看，该方法已经不是单纯“换一个 PCG 求解器”，而是形成了一个较清晰的研究框架：

> 面向 reduced SPD 可解结构的机器人 QP 重构方法。

该框架在 UR3e drift free repetitive motion 中已经表现出较好的精度和实时求解潜力。现阶段最需要补强的是：真实机器人实验、系统鲁棒性对照和理论边界证明。如果这些部分能够完成，该工作有望从单个方法扩展为一个较完整的机器人 QP 求解范式。

从投稿角度看，若只保留当前 CoppeliaSim 和单任务实验，更适合 Robotics and Autonomous Systems、Control Engineering Practice 或 Mechatronics。若补充真实 UR3e、完整对照和鲁棒性实验，可考虑 IEEE RA-L 或 IEEE/ASME Transactions on Mechatronics。若进一步扩展成多任务、多机器人和一般 reduced SPD QP 框架，则具备向更高层级机器人期刊冲击的基础。
