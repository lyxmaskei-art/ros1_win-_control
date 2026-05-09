# Method 2 TVQP with RK45 Integrated LCCZNN

## 实验目标

本实验针对已有的 **TVQP with feedback Method 2** 分支，只替换一处数值积分方式：

原实验中 LCCZNN 求解器状态采用显式 Euler 更新，

$$
y_{k+1}=y_k+\tau f(y_k,t_k).
$$

本实验改成在每个外层采样周期内对连续 LCCZNN 流使用 RK45 积分，

$$
y_{k+1}=y_k+\int_{0}^{\tau} f(y(s),t_k)\,ds.
$$

外层机器人关节更新仍保持原来的 sampled control 形式，

$$
\theta_{k+1}=\theta_k+\tau \dot q_k.
$$

因此本实验只评价 **求解器内部 LCCZNN 积分方式** 的影响，不同时改变机器人控制接口。

## Method 2 问题形式

令

$$
y=
\begin{bmatrix}
\dot q\\
\lambda
\end{bmatrix}.
$$

Method 2 保留经典 drift free QP 壳层：

$$
\min_{\dot q}\ \frac{1}{2}\dot q^\top \dot q+c_d^\top \dot q
$$

subject to

$$
J(q)\dot q=\dot r_d-k_p\bigl(x(q)-x_d\bigr).
$$

其中漂移反馈项为

$$
c_d
=\mu\,\operatorname{sig}_{p}\bigl(q-q_0\bigr).
$$

KKT 残差写成

$$
R(y,t)=H(t)y+p(t),
$$

其中

$$
H(t)=
\begin{bmatrix}
I & -J(t)^\top\\
J(t) & 0
\end{bmatrix},
$$

$$
p(t)=
\begin{bmatrix}
c_d(t)\\
-\dot r_d(t)+k_p\bigl(x(q)-x_d\bigr)
\end{bmatrix}.
$$

## RK45 替换方式

原 Euler 版本：

$$
y_{k+1}
=y_k-\tau\gamma\phi\left(E_k\right)
\frac{d_k}{d_k^\top d_k+\lambda_r},
$$

其中

$$
E_k=\frac{1}{2}R(y_k,t_k)^\top R(y_k,t_k),
$$

$$
d_k=H(t_k)^\top R(y_k,t_k).
$$

RK45 版本把上式视为连续 LCCZNN 微分方程：

$$
\frac{dy}{ds}
=-\gamma\phi\left(E(y,t_k)\right)
\frac{d(y,t_k)}{d(y,t_k)^\top d(y,t_k)+\lambda_r}.
$$

在一个控制周期内，冻结 $t_k$ 对应的 QP 系数，对 $s\in[0,\tau]$ 积分：

$$
y_{k+1}
=\operatorname{RK45}\left(y_k,f,\tau\right).
$$

速度边界仍沿用原工程投影：

$$
\dot q_{k+1}\leftarrow
\operatorname{clip}\bigl(\dot q_{k+1},\xi^-,\xi^+\bigr).
$$

## 实验设置

| 项目 | 数值 |
|---|---:|
| 方法 | TVQP with feedback Method 2 |
| 轨迹 | Heart trajectory |
| 外层采样周期 | 0.005 s |
| Offline 时长 | 20 s |
| Live 时长 | 10 s |
| LCCZNN gain | 4608 |
| RK45 rtol | 1e-5 |
| RK45 atol | 1e-8 |
| RK45 mode | adaptive |

## 结果对比

| 分支 | 场景 | 平均位置误差 | 最终位置误差 | 最大位置误差 | 最终关节漂移范数 |
|---|---|---:|---:|---:|---:|
| Euler | Offline | 1.187e-5 m | 8.240e-7 m | 1.440e-4 m | 2.144e-6 rad |
| RK45 | Offline | 8.910e-7 m | 4.234e-7 m | 1.276e-6 m | 1.268e-6 rad |
| Euler | Live | 2.427e-4 m | 1.044e-4 m | 4.781e-4 m | 1.416e-3 rad |
| RK45 | Live | 2.357e-4 m | 1.042e-4 m | 4.180e-4 m | 1.430e-3 rad |

## 结果分析

RK45 在 offline 中改善非常明显。平均位置误差从

$$
1.187\times 10^{-5}\ \mathrm{m}
$$

降低到

$$
8.910\times 10^{-7}\ \mathrm{m}.
$$

最终关节漂移范数也从

$$
2.144\times 10^{-6}\ \mathrm{rad}
$$

降低到

$$
1.268\times 10^{-6}\ \mathrm{rad}.
$$

这说明 Euler 离散在 offline 条件下确实引入了可观察的求解器积分误差，而 RK45 能更准确地沿着连续 LCCZNN 残差下降流推进。

Live 中改善较小。平均位置误差从

$$
2.427\times 10^{-4}\ \mathrm{m}
$$

降低到

$$
2.357\times 10^{-4}\ \mathrm{m}.
$$

最终位置误差几乎不变，最终漂移范数略微增大：

$$
1.416\times 10^{-3}\ \mathrm{rad}
\rightarrow
1.430\times 10^{-3}\ \mathrm{rad}.
$$

这说明 live 精度瓶颈不完全在 LCCZNN 数值积分层，而更多来自实时通信、CoppeliaSim target position 执行延迟、关节读数离散化、外层轨迹闭环误差以及仿真驱动接口。

## 诊断结论

固定单步 Dormand Prince RK45 也做过 1 s smoke test，但在原高增益

$$
\gamma=4608
$$

下表现不稳定，1 s offline 已出现明显误差放大。因此正式实验采用常规自适应 RK45，而不是固定单步 RK45。

最终结论：

1. RK45 对 offline TVQP Method 2 有显著提升。
2. RK45 对 live TVQP Method 2 只有轻微位置误差提升，不能根本改变 live 排名。
3. 当前 live 的主要瓶颈仍然不是 Euler 积分误差，而是仿真执行闭环和采样控制接口。
4. 如果论文中讨论 TVQP Method 2，RK45 可以作为“数值积分增强版”展示，但它目前还没有超过原 frozen QP Method 2 DLCCZNN 主线。

## 文件位置

代码：

```text
C:\Users\lyx\Desktop\sci\experiments\solver_comparison_integration\project_tvqp_method2_rk45_solver\scripts\run_offline.py / run_live.py
```

结果：

```text
C:\Users\lyx\Desktop\sci\code\project_tvqp_method2_rk45_solver\results\final_gamma4608_rtol1e-5
```

对比表：

```text
C:\Users\lyx\Desktop\sci\code\project_tvqp_method2_rk45_solver\results\final_gamma4608_rtol1e-5\comparison_euler_vs_rk45.csv
```
