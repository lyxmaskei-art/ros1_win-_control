# Method 3a 公式总览

> [!abstract] 文档定位
> 这份文档只做一件事：
> 把 Method 3a 的核心公式，用最短路径整理成一个适合在 Obsidian 中快速复习的版本。

配套阅读：

- [[三种方法公式与逻辑总览]]
- [[三种方法详细公式推导]]
- [[三种drift-free方法代码级完整推导]]

对应代码：

- `core/discrete_rmp_qp_integrated_methods.py`
  里面的 `Method3aPositionDriftController`

## 1. 变量定义

| 变量 | 含义 |
| --- | --- |
| $q_k$ | 第 $k$ 步当前关节角，维数 $n$ |
| $q_0$ | 初始关节角，维数 $n$ |
| $\dot q_k$ | 第 $k$ 步要求解的关节角速度，维数 $n$ |
| $x(q_k)$ | 第 $k$ 步末端当前位置，维数 $m$ |
| $x_d(k)$ | 第 $k$ 步期望末端位置，维数 $m$ |
| $\dot r_d(k)$ | 第 $k$ 步期望任务空间速度，维数 $m$ |
| $J_k$ | 第 $k$ 步雅可比矩阵，$J_k=J(q_k)$，维数 $m\times n$ |
| $\tau$ | 离散步长 |
| $e_p(k)$ | 位置误差 |
| $e_d(k)$ | 漂移误差 |
| $k_p$ | 任务反馈增益，对应 `task_gain` |
| $\alpha$ | 位置项权重，对应 `position_weight` |
| $\beta$ | 漂移项权重，对应 `drift_weight` |

其中

$$
e_p(k)=x(q_k)-x_d(k),
$$

$$
e_d(k)=q_k-q_0.
$$

## 2. Method 3a 的核心思想

Method 3a 不是把连续时间公式硬塞到 QP 里，而是先写“一步离散后误差会怎么传播”，再把这个一步传播误差的二范数平方作为优化目标。

也就是说，它直接优化下一步误差，而不是只优化当前瞬时速度。

## 3. 位置误差的一步离散传播

一阶近似下：

$$
x(q_{k+1}) \approx x(q_k)+\tau J_k \dot q_k.
$$

于是下一步位置误差近似为：

$$
e_p(k+1)\approx e_p(k)+\tau\big(J_k\dot q_k-\dot r_d(k)+k_p e_p(k)\big).
$$

因此位置项目标写成：

$$
\mathcal J_p
=
\left\|
e_p(k)+\tau\big(J_k\dot q_k-\dot r_d(k)+k_p e_p(k)\big)
\right\|_2^2.
$$

## 4. 漂移误差的一步离散传播

关节离散更新为：

$$
q_{k+1}=q_k+\tau \dot q_k.
$$

所以漂移误差传播为：

$$
e_d(k+1)=e_d(k)+\tau \dot q_k.
$$

因此漂移项目标写成：

$$
\mathcal J_d
=
\left\|
e_d(k)+\tau \dot q_k
\right\|_2^2.
$$

## 5. 总目标函数

Method 3a 的总目标函数为：

$$
\begin{aligned}
\min_{\dot q_k}\quad &
\alpha
\left\|
e_p(k)+\tau\big(J_k\dot q_k-\dot r_d(k)+k_p e_p(k)\big)
\right\|_2^2 \\
&\quad +
\beta
\left\|
e_d(k)+\tau \dot q_k
\right\|_2^2.
\end{aligned}
$$

这就是“位置误差 + 漂移误差”的离散标量化二范数方案。

## 6. 化成标准二次型

先定义一个中间量：

$$
a_k=\left(\frac{1}{\tau}+k_p\right)e_p(k)-\dot r_d(k).
$$

则目标可写成标准 QP 形式：

$$
\min_{\dot q_k}
\quad
\frac{1}{2}\dot q_k^\top Q_k \dot q_k+c_k^\top \dot q_k.
$$

其中：

$$
Q_k=\alpha J_k^\top J_k+\beta I,
$$

$$
c_k=\alpha J_k^\top a_k+\beta \frac{e_d(k)}{\tau}.
$$

为了数值稳定，代码里还会加一个很小的正则：

$$
Q_k \leftarrow Q_k+\varepsilon I.
$$

所以最终代码形式就是：

$$
Q_k=\alpha J_k^\top J_k+\beta I+\varepsilon I,
$$

$$
c_k=
\alpha J_k^\top
\left[
\left(\frac{1}{\tau}+k_p\right)e_p(k)-\dot r_d(k)
\right]
+\beta \frac{e_d(k)}{\tau}.
$$

## 7. 代码里的对应实现

核心实现就在：

- `core/discrete_rmp_qp_integrated_methods.py`

对应语句是：

$$
\texttt{position\_drive}
=
\left(\frac{1}{\tau}+\texttt{task\_gain}\right)\texttt{position\_error}
-\texttt{desired\_velocity},
$$

$$
\texttt{q\_matrix}
=
\texttt{position\_weight}(J^\top J)
+\texttt{drift\_weight}\,I
+\texttt{regularization\_gain}\,I,
$$

$$
\texttt{q\_vector}
=
\texttt{position\_weight}(J^\top \texttt{position\_drive})
+\texttt{drift\_weight}\frac{\texttt{drift\_delta}}{\tau}.
$$

## 8. 为什么 Method 3a 这里不用 S-LVI-PDNN

因为 Method 3a 当前这个问题已经是一个很干净的无等式约束二次型：

- Hessian 是正定的；
- 变量只有 $\dot q$；
- 不需要再把问题包装成额外神经动力学状态。

所以直接解

$$
\dot q_\star=-Q_k^{-1}c_k
$$

再做速度边界裁剪

$$
\dot q=
\operatorname{clip}\!\big(\dot q_\star,\xi^{-},\xi^{+}\big)
$$

会比继续用 PDNN 离散迭代更稳，也更符合当前追求高精度漂移抑制的目标。

## 9. Method 3a 相对 Method 3b 的本质差异

Method 3a：

- 位置误差和漂移误差都放进目标函数；
- 任务允许“软收敛”；
- 可以通过调节 $\alpha$、$\beta$、$k_p$ 在跟踪与闭环漂移之间平衡。

Method 3b：

- 只最小化漂移项；
- 任务速度 $J\dot q=\dot r_d$ 作为硬约束；
- 严格不允许加任何 $+\lambda e$ 反馈补偿。

因此在当前问题里，Method 3a 更适合冲高精度，Method 3b 更适合做结构对照。

## 10. 当前实验上最关键的经验结论

1. Method 3a 的核心不是“多加一个漂移偏置项”，而是“直接把下一步位置误差和下一步漂移误差都离散成二范数目标”。
2. $\tau$ 对结果影响极大。$\tau$ 变小后，Method 3a 的漂移误差会持续下降。
3. `task_gain` 在 Method 3a 中非常敏感，存在明显的甜点区，超过这个区间误差会突然恶化。
