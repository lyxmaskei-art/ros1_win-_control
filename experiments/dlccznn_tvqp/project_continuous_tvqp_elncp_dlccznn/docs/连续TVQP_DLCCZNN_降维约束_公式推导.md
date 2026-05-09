---
tags:
  - continuous-tvqp
  - dlccznn
  - drift-free
  - formula-derivation
  - inequality-handling
aliases:
  - 连续TVQP公式推导
  - TVQP与DLCCZNN推导说明
  - 降维约束公式推导
---

# 连续 TVQP 与 DLCCZNN 的公式推导和代码对应说明

> [!abstract]
> 这份文档对应代码文件 [continuous_tvqp_dlccznn.py](C:\Users\lyx\Desktop\sci\code\core\continuous_tvqp_dlccznn.py)。
> 目标不是把理论说得“更满”，而是把当前实现中真正做了什么、哪些部分是严格公式、哪些部分是工程近似，全部讲清楚。

**关联阅读**

- [[三种方法公式与逻辑总览]]
- [[三种方法详细公式推导]]
- [[三种drift-free方法代码级完整推导]]
- [[实验结果分组差异说明]]
- [[当前方法知识脉络与决策地图_20260427]]

---

## 1. 问题背景与统一符号

设机器人关节角为

$$
q_k \in \mathbb{R}^n,
\qquad
q_0 \in \mathbb{R}^n
$$

其中 \(q_0\) 为初始关节角。

末端任务空间位置记为

$$
x(q_k) \in \mathbb{R}^m,
\qquad
x_d(k) \in \mathbb{R}^m,
\qquad
\dot r_d(k) \in \mathbb{R}^m
$$

这里本文代码只使用位置控制，因此

$$
m = 3.
$$

对应雅可比矩阵为

$$
J_k = J(q_k) \in \mathbb{R}^{m \times n}.
$$

离散控制步长为

$$
\tau > 0.
$$

统一定义两类误差：

位置误差

$$
e_p(k) = x(q_k) - x_d(k),
$$

漂移误差

$$
e_d(k) = q_k - q_0.
$$

在带任务反馈时，代码中采用的任务速度命令为

$$
b_{\mathrm{fb}}(k) = \dot r_d(k) - k_p e_p(k),
$$

其中 \(k_p\) 对应代码参数 `task_gain`。

在去反馈模式下，

$$
b_{\mathrm{pure}}(k) = \dot r_d(k).
$$

因此三种方法在 `use_feedback=True` 时，任务层真正追踪的是 \(b_{\mathrm{fb}}(k)\)，不是纯任务速度 \(\dot r_d(k)\)。

---

## 2. 关节速度不等式约束的处理

### 2.1 动态速度边界

代码并没有直接从一个标准连续时变不等式系统出发，而是每个离散时刻先构造一组“冻结”的动态速度上下界：

$$
\xi_i^{-}(k) =
\max\left(
\dot \theta_{\min,i},
\frac{\eta}{\tau}\bigl(\theta_{\min,i} - q_{k,i}\bigr)
\right),
$$

$$
\xi_i^{+}(k) =
\min\left(
\dot \theta_{\max,i},
\frac{\eta}{\tau}\bigl(\theta_{\max,i} - q_{k,i}\bigr)
\right),
\qquad i=1,\dots,n.
$$

这里

- \(\theta_{\min}, \theta_{\max}\) 是关节角位置上下界
- \(\dot \theta_{\min}, \dot \theta_{\max}\) 是关节速度硬件上下界
- \(\eta \in (0,1]\) 是安全系数

这一步对应代码中的 `compute_dynamic_velocity_bounds(...)`。

### 2.2 用变量变换隐式满足速度 box 约束

定义每个关节的中点和半宽：

$$
m_i(k) = \frac{\xi_i^{+}(k) + \xi_i^{-}(k)}{2},
\qquad
R_i(k) = \frac{\xi_i^{+}(k) - \xi_i^{-}(k)}{2}.
$$

再引入无约束变量 \(\alpha_i \in \mathbb{R}\)，令

$$
\dot q_i = m_i(k) + R_i(k)\sin \alpha_i.
$$

写成向量形式：

$$
\boxed{
\dot q(\alpha)
=
m(k) + R(k)\odot \sin(\alpha)
}
$$

其中 \(R(k)\) 按元素作用，\(\odot\) 表示 Hadamard 乘积。

由于

$$
\sin \alpha_i \in [-1,1],
$$

必有

$$
\dot q_i \in [\xi_i^{-}(k), \xi_i^{+}(k)].
$$

所以速度不等式约束不再需要显式对 \(\dot q\) 做 clip。

### 2.3 正反变换

反解到 \(\alpha\) 空间时使用

$$
\alpha_i
=
\arcsin\left(
\frac{\dot q_i - m_i(k)}{R_i(k)}
\right).
$$

对应代码中的 `to_alpha(...)`。

从 \(\alpha\) 还原回关节速度时使用

$$
\dot q_i
=
m_i(k) + R_i(k)\sin \alpha_i,
$$

对应 `to_qdot(...)`。

### 2.4 一个必须说明的实现事实

虽然 \(\dot q\) 的速度约束已经通过 \(\sin\) 变换隐式满足，但代码最后更新关节角时仍然做了

$$
q_{k+1}
=
\operatorname{clip}
\bigl(q_k + \tau \dot q_{\mathrm{cmd}},\ \theta_{\min},\ \theta_{\max}\bigr).
$$

也就是说：

- 速度约束层面，主要由变量变换处理
- 关节位置最终落点层面，代码仍保留了一层保险性的 clip

这是一种工程防护，不应在理论表述中写成“完全没有任何 clip”

---

## 3. DLCCZNN 的离散求解骨架

### 3.1 连续时间原型

对一个时变误差系统

$$
e(t) = H(t,y(t)),
$$

DLCCZNN 的基本连续时间形式可以写成

$$
\dot y(t) = -\gamma \Phi(e(t)).
$$

其中 \(\Phi(\cdot)\) 是非线性激活函数，\(\gamma > 0\) 是收敛增益。

### 3.2 代码里的欧拉离散

代码没有实现高阶离散格式，而是使用显式欧拉：

$$
y_{\ell+1}
=
y_{\ell}
- h \gamma \Phi(e_\ell),
\qquad
h = \frac{\tau}{N_s},
$$

其中 \(N_s\) 对应 `substeps`。

因此每个控制周期 \(k\) 内部，又做了 \(N_s\) 次子步推进。

### 3.3 激活函数

代码使用的是 sig exp 型激活，可抽象写成

$$
\Phi(z)
=
\operatorname{sign}(z)\,|z|^r\,
\exp\bigl(\min(|z|,c_{\exp})\bigr),
$$

其中

- \(r\) 对应 `activation_power`
- \(c_{\exp}\) 对应 `activation_exp_clip`

---

## 4. Method 1 的推导

### 4.1 原始速度层 QP

Method 1 在速度空间可理解为

$$
\begin{aligned}
\min_{\dot q_k}\quad
&
\frac{1}{2}\dot q_k^\top W \dot q_k
+
\hat c(k)^\top \dot q_k
\\
\text{s.t.}\quad
&
J_k \dot q_k = b(k),
\\
&
\xi^{-}(k) \le \dot q_k \le \xi^{+}(k),
\end{aligned}
$$

其中

$$
W = w I,
$$

\(w\) 对应 `weight_gain`。

任务右端项 \(b(k)\) 取值为

$$
b(k)=
\begin{cases}
b_{\mathrm{fb}}(k), & \text{带反馈模式} \\
b_{\mathrm{pure}}(k), & \text{去反馈模式}
\end{cases}
$$

漂移偏置项为

$$
\hat c(k)
=
\mu\,
\Phi_{\mathrm{sig}}\bigl(e_d(k)\bigr),
$$

这里 \(\mu\) 对应 `mu_gain`，\(\Phi_{\mathrm{sig}}\) 对应代码中的 `sig_exp_activation(...)`。

### 4.2 KKT 残差系统

引入拉格朗日乘子 \(\lambda \in \mathbb{R}^m\)，再将

$$
\dot q = m(k) + R(k)\odot \sin \alpha
$$

代入 KKT 条件，可得误差系统

$$
y =
\begin{bmatrix}
\alpha \\
\lambda
\end{bmatrix}
\in \mathbb{R}^{n+m},
$$

$$
\boxed{
e(y,t_k)
=
\begin{bmatrix}
W\dot q(\alpha) + \hat c(k) - J_k^\top \lambda \\
J_k\dot q(\alpha) - b(k)
\end{bmatrix}
}
$$

当 \(e(y,t_k)=0\) 时，对应当前冻结时刻 \(t_k\) 下的 KKT 平衡点。

### 4.3 DLCCZNN 更新

Method 1 的求解器直接使用

$$
y_{\ell+1}
=
y_\ell
- h\gamma \Phi\bigl(e(y_\ell,t_k)\bigr).
$$

最终取前 \(n\) 维 \(\alpha\) 并恢复

$$
\dot q_{\mathrm{cmd}}
=
m(k)+R(k)\odot\sin\alpha^\star.
$$

### 4.4 代码层面的含义

Method 1 的本质是：

1. 先保留经典 drift free 等式约束结构
2. 再把速度约束塞进变量变换
3. 最后用 DLCCZNN 欧拉离散去追 KKT 残差零点

---

## 5. Method 2 的推导

### 5.1 与 Method 1 的共同点

Method 2 与 Method 1 使用同一个 KKT 残差：

$$
e(y,t_k)
=
\begin{bmatrix}
W\dot q(\alpha) + \hat c(k) - J_k^\top \lambda \\
J_k\dot q(\alpha) - b(k)
\end{bmatrix}.
$$

因此，Method 2 并不是换了目标函数，而是换了残差推进机制。

### 5.2 残差能量

定义当前子步的残差能量为

$$
E_{\mathrm{res}}(\ell)
=
\frac{1}{2}\|e(y_\ell,t_k)\|_2^2.
$$

代码中为避免数值爆炸，还做了截断：

$$
E_{\mathrm{clip}}(\ell)
=
\min\bigl(E_{\mathrm{res}}(\ell),E_{\max}\bigr),
$$

其中 \(E_{\max}\) 对应 `system_energy_clip`。

### 5.3 自适应增益

Method 2 引入正半轴激活

$$
\Phi_+(s)
=
\max(s,0)^r\,
\exp\bigl(\min(\max(s,0),c_{\exp})\bigr),
$$

再构造增益

$$
\boxed{
g(\ell)
=
1+\mu\,\Phi_+\bigl(E_{\mathrm{clip}}(\ell)\bigr)
}
$$

这里的 \(\mu\) 仍然对应 `mu_gain`，但它的角色已经从 Method 1 的“漂移偏置系数”变成了“残差能量放大系数”。

### 5.4 更新律

于是 Method 2 的内层更新变为

$$
\boxed{
y_{\ell+1}
=
y_\ell
- h\gamma g(\ell)\Phi\bigl(e(y_\ell,t_k)\bigr)
}
$$

当残差大时，\(g(\ell)\) 放大更新幅值；当残差收缩后，\(g(\ell)\) 自动回落。

### 5.5 为什么 Method 2 往往比 Method 1 稳

因为它仍然保持了“经典等式约束壳层”

$$
J_k \dot q_k = b(k)
$$

的结构，只是在数值求解层引入了更激进但又可回退的残差驱动机制。

所以 Method 2 的改动是“求解器增强”，不是“问题结构重写”。

---

## 6. Method 3a 的推导

Method 3a 与前两者最不同的地方，在于它不再把问题写成“等式约束 QP 的 KKT 系统”，而是直接写成一步预测误差最小化。

### 6.1 一步位置误差传播

由一阶欧拉近似

$$
x(q_{k+1})
\approx
x(q_k)+\tau J_k \dot q_k
$$

可得

$$
e_p(k+1)
\approx
e_p(k)+\tau\bigl(J_k\dot q_k-\dot r_d(k)\bigr).
$$

若保留位置反馈项 \(k_p e_p(k)\)，代码将其吸收入驱动项，等价写成

$$
e_p(k+1)
\approx
e_p(k)+\tau\bigl(J_k\dot q_k-\dot r_d(k)+k_p e_p(k)\bigr).
$$

### 6.2 一步漂移误差传播

由

$$
q_{k+1}=q_k+\tau\dot q_k
$$

得

$$
e_d(k+1)=e_d(k)+\tau\dot q_k.
$$

### 6.3 二次目标函数

于是 Method 3a 构造

$$
\min_{\dot q_k}
\;
\alpha_p
\left\|
e_p(k)+\tau\bigl(J_k\dot q_k-\dot r_d(k)+\delta_{\mathrm{fb}}\bigr)
\right\|_2^2
+
\alpha_d
\left\|
e_d(k)+\tau\dot q_k
\right\|_2^2,
$$

其中

$$
\delta_{\mathrm{fb}}
=
\begin{cases}
k_p e_p(k), & \text{带反馈模式} \\
0, & \text{去反馈模式}
\end{cases}
$$

\(\alpha_p\) 对应 `position_weight`，\(\alpha_d\) 对应 `drift_weight`。

### 6.4 标准二次型

定义

$$
a_k
=
\begin{cases}
\left(\frac{1}{\tau}+k_p\right)e_p(k)-\dot r_d(k), & \text{带反馈模式} \\
\frac{1}{\tau}e_p(k)-\dot r_d(k), & \text{去反馈模式}
\end{cases}
$$

则目标函数可改写为

$$
\min_{\dot q_k}
\;
\frac{1}{2}\dot q_k^\top Q_k \dot q_k + c_k^\top \dot q_k,
$$

其中

$$
\boxed{
Q_k
=
\alpha_p J_k^\top J_k
+
\alpha_d I
+
\varepsilon I
}
$$

$$
\boxed{
c_k
=
\alpha_p J_k^\top a_k
+
\alpha_d \frac{e_d(k)}{\tau}
}
$$

这里 \(\varepsilon\) 对应 `regularization_gain`。

### 6.5 代码实际做法

代码并没有让 DLCCZNN 直接去求解完整的 \(Q_k \dot q + c_k = 0\) 体系，而是分成两步：

第一步，先解析求最优速度

$$
\boxed{
\dot q_\star(k) = -Q_k^{-1}c_k
}
$$

第二步，再将其映射到 \(\alpha\) 空间

$$
\alpha_\star(k)
=
\arcsin
\left(
\frac{\dot q_\star(k)-m(k)}{R(k)}
\right)
$$

第三步，DLCCZNN 只负责跟踪 \(\alpha_\star(k)\)

$$
e_\alpha(\ell)=\alpha_\ell-\alpha_\star(k),
$$

$$
\boxed{
\alpha_{\ell+1}
=
\alpha_\ell
- h\gamma \Phi\bigl(e_\alpha(\ell)\bigr)
}
$$

最后恢复

$$
\dot q_{\mathrm{cmd}}
=
m(k)+R(k)\odot\sin\alpha.
$$

### 6.6 这一点必须诚实说明

所以，当前代码里的 Method 3a 更准确的说法是：

> 解析 QP 最优解加上 \(\alpha\) 空间的 DLCCZNN 跟踪

而不是：

> 用 DLCCZNN 直接求解完整连续 TVQP 的 KKT 或梯度残差系统

这也是它与 Method 1、Method 2 在理论层面最本质的区别。

---

## 7. 三种方法的代码级对照

### 7.1 状态变量维度

Method 1 和 Method 2 的内部状态都是

$$
y=
\begin{bmatrix}
\alpha \\
\lambda
\end{bmatrix}
\in \mathbb{R}^{n+m}.
$$

对 UR3e 来说，

$$
n=6,\quad m=3,\quad n+m=9.
$$

Method 3a 的内部状态只有

$$
\alpha \in \mathbb{R}^6.
$$

### 7.2 残差定义

Method 1 和 Method 2：

$$
e=
\begin{bmatrix}
W\dot q+\hat c-J^\top\lambda \\
J\dot q-b
\end{bmatrix}
$$

Method 3a：

$$
e_\alpha=\alpha-\alpha_\star.
$$

### 7.3 残差统计口径

这次检查中已把 `method3a` 的任务残差统计修正为

$$
r_{\mathrm{task}} = b(k) - J_k\dot q_{\mathrm{cmd}},
$$

从而与 Method 1 和 Method 2 保持一致。也就是说，若 `use_feedback=True`，三者现在都以

$$
b(k)=b_{\mathrm{fb}}(k)
$$

为统计基准。

---

## 8. 理论模型与实际实现的边界

这一部分是最容易写虚的地方，这里单独列出来。

### 8.1 当前实现已经做到的

1. 任务层与漂移层都写成了时变函数形式
2. 速度 box 约束通过变量变换进入了无约束 \(\alpha\) 空间
3. Method 1 和 Method 2 使用 DLCCZNN 欧拉离散推进 KKT 残差
4. Method 3a 使用解析最优解加 DLCCZNN 跟踪

### 8.2 当前实现仍然带有的工程近似

1. 动态速度边界 \(\xi^\pm(k)\) 仍是每一步冻结构造出来的，而不是从一套完全连续光滑的不等式动态系统中直接推导
2. 最后更新 \(q_{k+1}\) 时，仍保留了对关节位置上下界的 clip
3. Method 3a 不是“DLCCZNN 直接求解完整 TVQP”，而是“先解析求解，再在 \(\alpha\) 空间做跟踪”

### 8.3 这意味着什么

如果后续论文要主打“连续时变 TVQP 加 DLCCZNN”，那么最扎实的理论主角更适合放在：

- Method 1 的 KKT 残差框架
- Method 2 的残差能量 shaping 框架

而 Method 3a 更适合被表述成：

- 离散误差目标驱动的高精度参考方法
- 或解析最优解与 DLCCZNN 跟踪耦合的方法

---

## 9. 这次检查后对 Claude 版本的修正结论

### 9.1 已确认并修正

1. `method3a` 与 `method3b` 的任务残差统计口径原先没有跟随反馈项，现已统一到 \(b(k)-J_k\dot q_{\mathrm{cmd}}\)
2. 噪声鲁棒性实验里，传感器噪声场景原先用带噪测量位姿计算位置误差，现已改为用真实状态位姿计算真实跟踪误差

### 9.2 已确认但暂不改写为更强理论

1. 动态速度边界仍沿用冻结式构造
2. 关节角更新后仍保留位置 clip
3. Method 3a 仍是解析解加跟踪，而非纯残差求解

这些都应该在论文或笔记中诚实表述，不能再写成“完全连续无裁剪纯 DLCCZNN 统一求解”。

---

## 10. 后续最值得继续优化的方向

> [!tip]
> 这一部分只给方向，不在本次检查中直接改代码。

1. 把 \(\xi^\pm(k)\) 的构造从“冻结式速度边界”进一步改写为更连续、更平滑的时变不等式系统
2. 让 Method 3a 从“解析解加跟踪”升级为“直接对连续时变残差做 DLCCZNN 求解”
3. 若要做更严格的理论证明，应围绕离散 Lyapunov 或误差递推界来写，而不是只写直观收敛叙述
4. 将 live 结果重新并回 `experiments_tvqp_with_feedback` 与 `experiments_tvqp_without_feedback` 两个总目录，避免结果分散在 `experiments_parameter_tuning` 下

---

## 11. 一句话总结

当前这套代码最准确的数学画像是：

> 先把关节速度不等式约束通过 \(\sin\) 变量变换转入无约束 \(\alpha\) 空间，再在冻结的离散时刻上，对时变任务误差构造 DLCCZNN 欧拉更新；其中 Method 1 和 Method 2 直接推进 KKT 残差，Method 3a 则推进解析最优解在 \(\alpha\) 空间的跟踪误差。
