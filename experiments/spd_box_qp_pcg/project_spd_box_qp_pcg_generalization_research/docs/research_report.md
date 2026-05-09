# SPD Box QP + PCG 泛化性调研报告

## 1. 核心问题

这次调研检验的问题是：

> 是否可以把以后所有 QP 问题都转化为类似 Method3 的严格凸 SPD box QP，然后统一用 PCG 求解，并获得高精度、高鲁棒性和强泛化能力？

当前结论是：

**不能把“所有 QP”都等价转成 SPD box QP。**

但可以把结论收窄成一个更可靠、也更适合写论文的版本：

**当机器人控制 QP 的硬约束主要是关节速度上下界，任务约束可以接受软化为二次误差项，并且 Hessian 可构造成严格正定矩阵时，该问题天然适合转化为 SPD box QP，并可用 warm started active set PCG 高效求解。**

也就是说，Method3 的强点不是“PCG 可以通吃所有 QP”，而是：

**你的 Method3 把 drift free repetitive motion resolution 重构成了一个与 PCG 天然匹配的 SPD box QP。**

## 2. 一般 QP 与 SPD box QP 的关系

一般 QP 可写成：

$$
\min_x \frac{1}{2}x^THx+c^Tx
$$

subject to

$$
A_{\rm eq}x=b_{\rm eq},
$$

$$
A_{\rm in}x\le b_{\rm in},
$$

$$
\ell\le x\le u.
$$

Method3 类型的 SPD box QP 是：

$$
\min_x \frac{1}{2}x^TQx+g^Tx
$$

subject to

$$
\ell\le x\le u,
$$

其中

$$
Q=Q^T\succ0.
$$

这说明 SPD box QP 是一般凸 QP 的一个重要子类，而不是一般 QP 的等价全集。

## 3. 为什么不能“所有 QP 都转”

### 3.1 非凸 QP 不能无损变成 SPD QP

如果

$$
H\nsucceq0,
$$

原问题就是非凸 QP。强行加正则项

$$
H+\rho I\succ0
$$

会改变目标函数的曲率，也通常改变最优解。

因此，非凸 QP 不能通过简单正则化无损变成 SPD QP。

### 3.2 硬等式约束不能无代价地消掉

对于

$$
A_{\rm eq}x=b_{\rm eq},
$$

可以做零空间消元：

$$
x=x_p+Nz,
$$

然后把问题降到变量 \(z\) 上。

但这只对等式约束有效。若还存在一般不等式

$$
A_{\rm in}x\le b_{\rm in},
$$

代入后得到

$$
A_{\rm in}Nz\le b_{\rm in}-A_{\rm in}x_p,
$$

这仍然是耦合线性不等式，不会自动变成 box 约束。

另一种做法是罚函数：

$$
\frac{\rho}{2}\|A_{\rm eq}x-b_{\rm eq}\|^2.
$$

这会得到 SPD 或更接近 SPD 的无约束问题，但只是近似。有限 \(\rho\) 下等式约束通常有残差；\(\rho\) 很大时条件数变差，PCG 收敛反而会变慢。

### 3.3 一般线性不等式不是 box

box 约束的可行域是：

$$
\ell_i\le x_i\le u_i.
$$

它的每个约束只作用于一个变量。

但一般线性不等式是：

$$
a_i^Tx\le b_i.
$$

它会耦合多个变量。例如：

$$
x_1+x_2\le1.
$$

这个约束不能被简单替换成

$$
0\le x_1\le1,\quad 0\le x_2\le1.
$$

因为后者允许

$$
x_1=x_2=1,
$$

但这违反

$$
x_1+x_2\le1.
$$

## 4. 什么时候可以安全转成 SPD box QP

可以安全转化的典型条件是：

1. 目标函数可以写成加权最小二乘。
2. Hessian 能通过正则项保证严格正定。
3. 硬约束主要是变量上下界。
4. 任务约束允许作为软目标，而不是必须精确满足的硬等式。
5. 不存在接触力锥、碰撞约束、动力学一致性等强耦合硬不等式。

例如 Method3：

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
\frac{\epsilon}{2}\|\dot q\|^2
$$

subject to

$$
\xi^-(t)\le \dot q\le \xi^+(t).
$$

展开后：

$$
Q=w_pJ^TJ+w_dI+\epsilon I,
$$

$$
c=w_pJ^T\left[\left(\frac{1}{\tau}+k_p\right)e_p-\dot x_d\right]
+w_d\frac{q-q_0}{\tau}.
$$

由于

$$
w_d>0,\quad \epsilon>0,
$$

所以

$$
Q\succ0.
$$

这就是 PCG 很适合它的根本原因。

## 5. PCG 的正确定位

PCG 的本质是求解：

$$
Ax=b,\quad A=A^T,\quad A\succ0.
$$

在无约束 SPD QP 中：

$$
\min_x\frac{1}{2}x^TQx+c^Tx,
$$

最优性条件是：

$$
Qx=-c.
$$

所以 PCG 可直接用于求解。

在 box QP 中，active set 固定活跃边界后，自由变量满足：

$$
Q_{\mathcal F\mathcal F}x_{\mathcal F}
=
-c_{\mathcal F}
-Q_{\mathcal F\mathcal A}x_{\mathcal A}.
$$

只要

$$
Q\succ0,
$$

就有

$$
Q_{\mathcal F\mathcal F}\succ0.
$$

因此 reduced problem 仍然可以用 PCG。

## 5.1 文献调研中的共识

从现有数值优化和机器人控制文献看，PCG 通常不是作为“通用 QP 求解器”出现，而是作为一个线性代数内核出现：

1. 大规模稀疏 SPD 线性系统中，PCG 是经典工具。
2. interior point、SQP、trust region、Newton CG 等方法中，PCG 常用于求解 Newton 方向或 KKT 系统中的正定子系统。
3. 机器人 MPC 中，PCG 可被用于快速求解线性化或凝聚后的结构化子问题。
4. SLAM 和 bundle adjustment 中，PCG 常用于 Schur complement 或 normal equation。
5. 但一般 QP 的主流求解器仍包括 active set、interior point、operator splitting、ADMM、SQP 和层级 QP，而不是单独的 PCG。

这和本文判断一致：

**PCG 很适合解 SPD 子系统；要让 PCG 成为主求解器，必须先让问题结构变成 SPD box QP 或能被可靠分解成 SPD reduced systems。**

## 6. 机器人控制中的适用边界

### 6.1 适合

适合 SPD box QP + PCG 的问题包括：

1. 速度级逆运动学。
2. 关节速度上下界约束。
3. 关节位置安全边界被转化为动态速度上下界。
4. 任务误差可作为二次软目标。
5. drift free 或 posture regulation 可作为二次软目标。
6. 小到中等规模的冗余机械臂控制。
7. 高频实时控制中需要 warm start 的连续 QP 序列。

### 6.2 不适合

不适合直接改成 SPD box QP 的问题包括：

1. 严格任务等式必须无误差满足。
2. 碰撞距离约束作为硬线性化不等式。
3. 摩擦锥、接触力、单边接触约束。
4. 全身控制中的动力学一致性等式。
5. 层级 QP 中的严格任务优先级。
6. 非凸 QP 或 QCQP。
7. 一般 MPC 中的大量耦合状态控制约束。

这些问题可以用 penalty、slack、ADMM、active set、interior point 或 SQP 处理，但不应声称它们被等价转成了纯 SPD box QP。

## 7. 实验结论摘要

脚本：

`scripts/synthetic_qp_generalization_tests.py`

输出：

`results/synthetic_qp_generalization_summary.csv`

实验包括四类：

1. Method3-like SPD box QP。
2. 硬等式约束 QP。
3. 耦合线性不等式 QP。
4. 非凸 indefinite QP。

实验结论：

1. 原生 SPD box QP 可以被 active set PCG 高精度求解。
2. 硬等式约束用罚函数替代后只能近似，且惩罚越大条件数越差。
3. 耦合线性不等式不能直接替换成 box，否则会产生不可接受的约束违反。
4. 非凸 QP 正则化成 SPD 会改变原问题最优解。

### 7.1 合成实验关键数值

原生 SPD box QP 测试中，active set 固定后用 PCG 求解 reduced SPD 子问题，得到：

$$
\|x_{\rm pcg}-x_{\rm exact}\|=3.61\times10^{-16},
$$

目标函数差值为：

$$
-2.58\times10^{-12}.
$$

这说明对于真正的 SPD box QP，PCG 可以达到数值精确。

硬等式罚函数测试中，当惩罚因子增大到：

$$
\rho=10^4
$$

时，等式残差仍为：

$$
6.00\times10^{-5},
$$

而条件数升高到：

$$
2.00\times10^4.
$$

这说明“罚函数近似”并不是“等价转化”。

耦合线性不等式测试中，把

$$
x_1+x_2\le1
$$

替换成 box 后，得到的 box 解违反原约束：

$$
x_1+x_2-1=1.
$$

这说明一般耦合不等式不能被直接 box 化。

非凸 QP 测试中，强行 SPD 化后最优点改变，距离原全局最优集合为：

$$
1.0.
$$

这说明非凸 QP 不能无损正定化。

### 7.2 Method3 机器人验证

为了验证“可行子类”不是纯理论，本项目复跑了 Method3 TVQP ELNCP PCG 的 20 s offline 心形轨迹：

| 指标 | 数值 |
|---|---:|
| 平均位置误差 | \(2.0433668773123088\times10^{-4}\ {\rm m}\) |
| 最大位置误差 | \(3.074013271522352\times10^{-4}\ {\rm m}\) |
| 最终位置误差 | \(6.079975503728618\times10^{-7}\ {\rm m}\) |
| 最大单关节绝对漂移 | \(4.125530800102486\times10^{-1}\ {\rm rad}\) |
| 最终关节漂移范数 | \(4.6671803940884594\times10^{-7}\ {\rm rad}\) |
| 平均 PCG 迭代次数 | \(9.51525\) |
| 平均 active set 迭代次数 | \(1.5305\) |

这说明在 Method3 这种位置误差与关节漂移软耦合、硬约束主要是动态速度 box 的结构下，SPD box QP 加 PCG 是有效的。

本项目还重新进行了标准 10 s live 验证。由于当前 CoppeliaSim 同步推进较慢，该 10 s live 实验实际 wall clock 约为：

$$
743.21\ {\rm s}.
$$

但数据本身正常，并且与此前标准 10 s live 结果一致：

| 指标 | Method3 TVQP ELNCP PCG 标准 10 s live |
|---|---:|
| 平均位置误差 | \(1.3023099307802444\times10^{-4}\ {\rm m}\) |
| 最大位置误差 | \(2.3121553853496093\times10^{-4}\ {\rm m}\) |
| 最终位置误差 | \(5.815280268085359\times10^{-5}\ {\rm m}\) |
| 最大单关节绝对漂移 | \(4.131762981414795\times10^{-1}\ {\rm rad}\) |
| 最终关节漂移范数 | \(4.1728008429060056\times10^{-4}\ {\rm rad}\) |
| 平均 ELNCP 残差 | \(5.879662313701967\times10^{-9}\) |
| 平均 PCG 迭代次数 | \(10.6775\) |
| 平均 active set 迭代次数 | \(1.8815\) |

该 live 结果说明，在真实同步仿真闭环里，Method3 的 SPD box QP + reduced PCG 结构仍然能维持稳定的 drift free 重复运动效果。

## 8. 对你当前研究的判断

你的大想法需要收窄，但收窄后是很有价值的。

不建议写：

> All QPs can be transformed into SPD box QPs and solved by PCG.

建议写：

> A broad class of velocity level robotic QPs with soft task objectives and bound dominated constraints can be reformulated as strictly convex SPD box QPs. This structure enables warm started reduced PCG to provide fast and accurate drift free repetitive motion resolution.

中文对应为：

**一大类以关节边界为主要硬约束、以任务误差和姿态漂移为软目标的速度级机器人 QP，可以被重构为严格凸 SPD box QP，并由 warm started reduced PCG 高效求解。**

这比“所有 QP 都能转”更严谨，也更容易通过审稿。

## 8.1 论文中可使用的强表述

可以写：

> For velocity level repetitive motion resolution, once the tracking objective and joint drift suppression are formulated as one step quadratic penalties and the physical bounds are expressed as dynamic velocity limits, the resulting problem becomes a strictly convex SPD box QP. This structure is particularly suitable for warm started reduced PCG because each active set subproblem preserves positive definiteness.

不建议写：

> Any QP can be transformed into an SPD box QP and solved by PCG.

这句话太大，并且已被等式约束、耦合不等式和非凸 QP 反例推翻。

## 9. 下一步建议

1. 不要把 PCG 当作通用 QP 求解器来讲。
2. 把 Method3 PCG 定位为结构匹配求解器。
3. 加一个 problem taxonomy 表，把 QP 分成可转、近似可转、不可转三类。
4. 用 Method3 的 live 数据证明“在 drift free repetitive IK 这种结构下确实有效”。
5. 用一个反例实验主动说明方法边界，避免审稿人抓住“泛化过度”攻击。

## 10. 参考来源

1. Stellato, Banjac, Goulart, Bemporad, Boyd, **OSQP: An Operator Splitting Solver for Quadratic Programs**, Mathematical Programming Computation, 2020.  
   该文说明通用凸 QP 求解通常需要处理 quasi definite 线性系统、warm start、factorization caching 和不可行性检测，而不是简单地统一转为 SPD box QP。  
   https://stanford.edu/~boyd/papers/osqp.html

2. Adabag, Atal, Gerard, Plancher, **MPCGPU: Real Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU**, ICRA 2024.  
   该文把 PCG 用作 NMPC 内部稀疏线性系统求解内核，说明 PCG 在机器人实时优化中很有价值，但其角色仍是结构化线性求解器。  
   https://arxiv.org/abs/2309.08079

3. Ceres Solver documentation, **Solving Non Linear Least Squares**.  
   Ceres 的 iterative Schur 使用 conjugate gradients 求解 Schur complement，说明 CG 类方法常用于大规模最小二乘线性化子问题。  
   https://ceres-solver.org/nnls_solving.html

4. Netlib Templates, **The Preconditioned Conjugate Gradient Method**.  
   该资料给出 PCG 的经典定位：用于求解对称正定线性系统。  
   https://www.netlib.org/utk/papers/etemplates/node10.html
