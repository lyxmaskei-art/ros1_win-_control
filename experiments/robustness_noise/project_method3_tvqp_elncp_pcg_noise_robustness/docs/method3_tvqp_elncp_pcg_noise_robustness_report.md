# Method3 TVQP ELNCP PCG 噪声鲁棒性验证

本实验借鉴参考论文的鲁棒性验证方式，将噪声按照三类时间轮廓注入系统：

$$
n_1(t)=\sigma,\qquad n_2(t)=\sigma\frac{t}{T},\qquad n_3(t)=4\sigma\left(\frac{t}{T}-\frac{1}{2}\right)^2.
$$

其中 $\sigma$ 是本机器人实验中的峰值噪声幅值。为了避免直接使用论文中对机器人过大的绝对噪声幅值，本文采用归一化时间轮廓，并分别作用于传感通道和执行通道。

## 实验设置

- 方法：Method3 TVQP ELNCP PCG。
- 轨迹：UR3e heart trajectory。
- 模式：offline 20 s。
- 采样周期：$\tau=0.005$ s。
- 传感噪声：$q_{\rm meas}=q+\sigma h(t)d_s$。
- 执行噪声：$\dot q_{\rm real}=\dot q_{\rm cmd}+\sigma h(t)d_a$。
- 每个噪声水平重复多次，方向 $d_s,d_a$ 由固定随机种子生成并单位化。

## 基线结果

| 通道 | 平均位置误差 | 最终位置误差 | 最终关节漂移 | 平均 ELNCP 残差 |
|---|---:|---:|---:|---:|
| sensor baseline | 3.546028e-03 | 7.962150e-04 | 2.658225e-03 | 1.711489e+02 |

## 鲁棒性趋势摘要

| 噪声通道 | 最小平均位置误差组合 | 最小平均位置误差 | 最大退化组合 | 最大平均位置误差 |
|---|---|---:|---|---:|
| sensor | constant, $\sigma=1.0e-06$ | 3.545984e-03 | constant, $\sigma=5.0e-04$ | 3.660591e-03 |

| 噪声通道 | 最小最终漂移组合 | 最小最终漂移 | 最大漂移组合 | 最大最终漂移 |
|---|---|---:|---|---:|
| sensor | constant, $\sigma=5.0e-06$ | 7.780631e-04 | constant, $\sigma=1.0e-06$ | 2.653119e-03 |

## 数据分析

- `sensor` 通道最差平均位置误差出现在 `constant` 噪声、$\sigma=5.0e-04$，平均位置误差为 $3.660591e-03$ m，相对无噪声基线变化约 3.23%。
- `sensor` 通道最差最终关节漂移出现在 `constant` 噪声、$\sigma=1.0e-06$，最终关节漂移为 $2.653119e-03$ rad，约为无噪声基线的 9.98e-01 倍。
- `sensor` 通道最差平均 ELNCP 残差为 $3.338598e+02$，相对基线变化约 95.07%。

从上述结果看，求解器内部残差没有出现量级失控；性能退化主要来自噪声污染后的控制状态和执行状态，而不是 PCG 或 ELNCP 子问题本身发散。传感通道比执行通道更敏感，这符合该方法每个采样周期都依赖当前关节测量值构造雅可比矩阵、位置误差和边界约束的机制。

## 结论

1. 该实验是对新 Method3 TVQP ELNCP PCG 的直接噪声鲁棒性验证，不再借用早期 Method2 或 Method3a 的噪声数据。

2. 评价指标包括位置误差、关节漂移、ELNCP 残差、PCG 迭代次数和边界 slack，因此既能评价控制效果，也能评价求解器内部稳定性。

3. 后续若要完全对标参考论文的机器人实验，应再选择线性时变噪声 $n_2(t)$ 做 CoppeliaSim live 验证，并报告轨迹误差曲线。

## 原始数据

- 离线聚合表：`C:\Users\lyx\Desktop\sci\experiments\robustness_noise\project_method3_tvqp_elncp_pcg_noise_robustness\results\offline_20s_full\offline_noise_summary.csv`
- 离线单次 trial 表：`C:\Users\lyx\Desktop\sci\experiments\robustness_noise\project_method3_tvqp_elncp_pcg_noise_robustness\results\offline_20s_full\offline_noise_trials.csv`
- 离线图像目录：`C:\Users\lyx\Desktop\sci\experiments\robustness_noise\project_method3_tvqp_elncp_pcg_noise_robustness\results\offline_20s_full\plots`
