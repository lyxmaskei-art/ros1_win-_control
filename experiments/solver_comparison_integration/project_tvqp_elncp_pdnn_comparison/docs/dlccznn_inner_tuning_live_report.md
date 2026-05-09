---
title: 新轨迹 DLCCZNN 高内部迭代 Live 完整实验报告
created: 2026-05-07
project: project_tvqp_elncp_pdnn_comparison
tags:
  - DLCCZNN
  - live
  - UR3e
  - CoppeliaSim
  - trajectory
---

# 新轨迹 DLCCZNN 高内部迭代 Live 完整实验报告

## 1. 实验目的

本次实验针对新添加的五条简单轨迹，统一提升 Method 2 TVQP + ELNCP + DLCCZNN 在每个控制周期内的求解器内部迭代次数，获得每条轨迹在 live 仿真下的高精度数据。

五条轨迹为：

1. small circle
2. line
3. figure8
4. circle
5. ellipse

固定参数：

$$
\tau=0.005\ \mathrm{s},
\qquad
\gamma=500,
\qquad
T=10\ \mathrm{s}.
$$

对每条轨迹都完成以下三组 live 实验：

$$
N_s\in\{20,40,60\}.
$$

其中 $N_s$ 表示每个外层控制周期内部的 DLCCZNN 更新次数。

## 2. 文件位置

集成代码：

`C:\Users\lyx\Desktop\sci\experiments\solver_comparison_integration\project_tvqp_elncp_pdnn_comparison\scripts\run_offline.py` ? `run_live.py`

完整 live 汇总表：

`C:\Users\lyx\Desktop\sci\code\project_tvqp_elncp_pdnn_comparison\results\dlccznn_inner_tuning_live\dlccznn_inner_tuning_live_summary_all_ns.csv`

每条轨迹最高精度表：

`C:\Users\lyx\Desktop\sci\code\project_tvqp_elncp_pdnn_comparison\results\dlccznn_inner_tuning_live\dlccznn_inner_tuning_live_best_accuracy.csv`

复杂度精度折中表：

`C:\Users\lyx\Desktop\sci\code\project_tvqp_elncp_pdnn_comparison\results\dlccznn_inner_tuning_live\dlccznn_inner_tuning_live_tradeoff.csv`

## 3. 内部迭代实现方式

在每个控制周期 $t_k$，先固定当前的机器人状态、雅可比矩阵、任务反馈速度、漂移反馈和 ELNCP 边界，然后对求解变量

$$
y=
\begin{bmatrix}
u\\
\lambda\\
\omega
\end{bmatrix}
$$

进行 $N_s$ 次内部 DLCCZNN 更新。内部步长为

$$
h_s=\frac{\tau}{N_s}.
$$

统一残差为

$$
F(y,t_k)=
\begin{bmatrix}
Wu+r_q(q_k)-J(q_k)^T\lambda+\sigma\omega\\
J(q_k)u-b(t_k,q_k)\\
\psi_\varepsilon(s(u),\omega)
\end{bmatrix}.
$$

残差能量为

$$
V(y,t_k)=\frac12F(y,t_k)^TF(y,t_k).
$$

DLCCZNN 内部更新方向为

$$
d(y,t_k)=F_y(y,t_k)^TF(y,t_k).
$$

标量激励函数为

$$
\Gamma(V)=\gamma V^\alpha\exp(\min(V,c)).
$$

内部更新可写为

$$
y^{r+1}
=y^r
-h_s
\frac{d(y^r,t_k)}
{d(y^r,t_k)^Td(y^r,t_k)+\varepsilon_r}
\left[\Gamma(V)+F(y^r,t_k)^TF_t(y^r,t_k)\right].
$$

完成 $N_s$ 次内部更新后，外层关节仍只更新一次：

$$
q_{k+1}=q_k+\tau u_k.
$$

因此，$N_s$ 提高的是每个控制周期内的求解精度，同时也线性提高求解器内部计算量。

## 4. Live 完整结果

| 轨迹 | $\gamma$ | $N_s$ | 平均位置误差 m | 最终位置误差 m | 最终关节漂移 rad |
|---|---:|---:|---:|---:|---:|
| circle | 500 | 20 | 1.945791e-04 | 1.448294e-04 | 7.883440e-03 |
| circle | 500 | 40 | 1.783209e-04 | 1.448515e-04 | 7.883216e-03 |
| circle | 500 | 60 | 1.733597e-04 | 1.448484e-04 | 7.883186e-03 |
| ellipse | 500 | 20 | 2.024528e-04 | 1.448525e-04 | 7.883950e-03 |
| ellipse | 500 | 40 | 1.835837e-04 | 1.448369e-04 | 7.883414e-03 |
| ellipse | 500 | 60 | 1.779765e-04 | 1.448521e-04 | 7.883502e-03 |
| figure8 | 500 | 20 | 1.844970e-04 | 1.448682e-04 | 7.882475e-03 |
| figure8 | 500 | 40 | 1.680516e-04 | 1.448266e-04 | 7.882491e-03 |
| figure8 | 500 | 60 | 1.643505e-04 | 1.448199e-04 | 7.882491e-03 |
| line | 500 | 20 | 1.499101e-04 | 1.447733e-04 | 7.879492e-03 |
| line | 500 | 40 | 1.439075e-04 | 1.448151e-04 | 7.880136e-03 |
| line | 500 | 60 | 1.422730e-04 | 1.447688e-04 | 7.880200e-03 |
| small circle | 500 | 20 | 1.600941e-04 | 1.448386e-04 | 7.883083e-03 |
| small circle | 500 | 40 | 1.534221e-04 | 1.449095e-04 | 7.887785e-03 |
| small circle | 500 | 60 | 1.532821e-04 | 1.448334e-04 | 7.882685e-03 |

## 5. 每条轨迹的最高精度配置

| 轨迹 | 最优 $N_s$ | 平均位置误差 m | 最终位置误差 m | 最终关节漂移 rad |
|---|---:|---:|---:|---:|
| line | 60 | 1.422730e-04 | 1.447688e-04 | 7.880200e-03 |
| small circle | 60 | 1.532821e-04 | 1.448334e-04 | 7.882685e-03 |
| figure8 | 60 | 1.643505e-04 | 1.448199e-04 | 7.882491e-03 |
| circle | 60 | 1.733597e-04 | 1.448484e-04 | 7.883186e-03 |
| ellipse | 60 | 1.779765e-04 | 1.448521e-04 | 7.883502e-03 |

按平均位置误差排序，最好的 live 轨迹是 line。

## 6. 复杂度精度折中

虽然 $N_s=60$ 在每条轨迹上都取得了最低平均位置误差，但 $N_s=60$ 的内部计算量是 $N_s=20$ 的 3 倍。对多数轨迹来说，从 $N_s=20$ 增加到 $N_s=60$ 的误差下降幅度有限。

以 line 为例：

$$
N_s=20:\quad \bar e_p=1.499101\times10^{-4}\ \mathrm{m}.
$$

$$
N_s=60:\quad \bar e_p=1.422730\times10^{-4}\ \mathrm{m}.
$$

相对提升为

$$
\frac{1.499101-1.422730}{1.499101}\approx 5.1\%.
$$

但内部复杂度提高为 3 倍。因此，如果目标是“复杂度和精度都取优”，主推配置应为

$$
\boxed{\text{line},\quad \gamma=500,\quad N_s=20.}
$$

该配置的 live 指标为：

$$
\bar e_p=1.499101\times10^{-4}\ \mathrm{m},
$$

$$
e_p(T)=1.447733\times10^{-4}\ \mathrm{m},
$$

$$
\lVert q(T)-q(0)\rVert_2=7.879492\times10^{-3}\ \mathrm{rad}.
$$

如果目标是展示最高精度，则采用

$$
\boxed{\text{line},\quad \gamma=500,\quad N_s=60.}
$$

对应平均位置误差为

$$
1.422730\times10^{-4}\ \mathrm{m}.
$$

## 7. 与单步 live 的对比

上一次单步 DLCCZNN live 的代表性结果约为：

| 轨迹 | 单步平均位置误差 m |
|---|---:|
| small circle | 6.053594e-04 |
| line | 6.828574e-04 |
| figure8 | 1.141999e-03 |

本次高内部迭代后，$N_s=20$ 已经显著提升：

| 轨迹 | $N_s$ | 平均位置误差 m |
|---|---:|---:|
| small circle | 20 | 1.600941e-04 |
| line | 20 | 1.499101e-04 |
| figure8 | 20 | 1.844970e-04 |

这说明提高内部 DLCCZNN 更新次数对 live 精度非常有效。特别是 figure8，平均位置误差从约 $1.14\times10^{-3}$ m 降到 $1.84\times10^{-4}$ m。

## 8. 结论

1. 已对所有新增轨迹完成 $N_s=20,40,60$ 的 live 实验。
2. 每条轨迹的最高精度都出现在 $N_s=60$。
3. 全部结果中平均位置误差最低的是 line，$\gamma=500,N_s=60$，平均误差为 `1.422730e-04 m`。
4. 综合复杂度和精度，推荐主推 line，$\gamma=500,N_s=20$，平均误差为 `1.499101e-04 m`。
5. live 最终关节漂移在不同轨迹和不同 $N_s$ 下都稳定在约 $7.88\times10^{-3}$ rad，说明该指标更多受仿真执行链路和位置伺服影响，而不是单纯由内部求解器迭代数决定。

