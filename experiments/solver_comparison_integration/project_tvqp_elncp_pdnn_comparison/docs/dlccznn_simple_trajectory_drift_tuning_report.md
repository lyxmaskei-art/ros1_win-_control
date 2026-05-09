# Method 2 DLCCZNN 简单轨迹 live 角度漂移调参报告

## 1. 问题定位

用户给出的 heart 基线结果为：

| 轨迹 | 场景 | 平均位置误差 | 最终位置误差 | 最终关节漂移范数 |
|---|---:|---:|---:|---:|
| heart | offline | 1.187e-5 m | 8.240e-7 m | 2.144e-6 rad |
| heart | live | 2.427e-4 m | 1.044e-4 m | 1.416e-3 rad |

前一轮简单轨迹 live 调参虽然提高了 `dlccznn_inner_steps`，但最终关节漂移仍约为 `7.88e-3 rad`，没有达到 heart live 的 `1.416e-3 rad`。检查后发现主要原因不是简单轨迹本身不适合，而是新项目 `project_tvqp_elncp_pdnn_comparison` 中 Method 2 的漂移反馈默认采用线性形式：

$$
c_{\mathrm{drift}}(t)=\mu\left(q(t)-q(0)\right),
$$

而旧 heart 低漂移基线采用的是非线性双幂指数激活漂移反馈：

$$
c_{\mathrm{drift}}(t)
=\mu\,\operatorname{sgn}(\Delta q)
|\Delta q|^{p}
\text{指数增强项},
\qquad
\Delta q=q(t)-q(0).
$$

对应代码中实际恢复为：

$$
c_{\mathrm{drift}}(t)
=\mu\,\operatorname{sgn}(\Delta q)
|\Delta q|^{p}
\exp\left(\min\{|\Delta q|,c_{\exp}\}\right)
$$

的逐元素非线性激活形式。这个非线性漂移反馈在关节偏离初始位姿时提供更强的回零驱动，因此对最终角度漂移更敏感。

## 2. 本次代码修正

在集成脚本中新增了可选参数：

```text
--drift-feedback-mode linear
--drift-feedback-mode nonlinear
```

默认仍保持 `linear`，避免破坏已有结果；本次调参显式使用 `nonlinear`，用于复现旧 heart 低漂移配置中的漂移反馈结构。

修改文件：

```text
C:\Users\lyx\Desktop\sci\experiments\solver_comparison_integration\project_tvqp_elncp_pdnn_comparison\scripts\run_offline.py / run_live.py
```

## 3. 最优参数组合

本次对新添加的简单轨迹统一采用如下参数：

| 参数 | 数值 |
|---|---:|
| `task_gain` | 260 |
| `drift_gain` | 20 |
| `solver_gamma` | 3072 |
| `activation_power` | 0.80 |
| `activation_exp_clip` | 4.0 |
| `dlccznn_inner_steps` | 60 |
| `drift_feedback_mode` | nonlinear |
| `tau` | 0.005 s |
| `duration` | 10 s |

该组合本质上是把旧 heart 低漂移配置迁移到简单轨迹实验，并保留提高后的控制周期内部 DLCCZNN 更新次数。

## 4. Live 实验结果

结果保存位置：

```text
C:\Users\lyx\Desktop\sci\code\project_tvqp_elncp_pdnn_comparison\results\drift_feedback_tuning_live
```

汇总表：

```text
C:\Users\lyx\Desktop\sci\code\project_tvqp_elncp_pdnn_comparison\results\drift_feedback_tuning_live\tuned_live_summary.csv
```

| 轨迹 | 平均位置误差 | 最终位置误差 | 最终关节漂移范数 |
|---|---:|---:|---:|
| small circle | 9.522e-5 m | 8.862e-5 m | 9.371e-4 rad |
| ellipse | 1.130e-4 m | 8.860e-5 m | 9.371e-4 rad |
| circle | 1.088e-4 m | 8.864e-5 m | 9.371e-4 rad |
| figure eight | 1.044e-4 m | 8.984e-5 m | 9.372e-4 rad |
| line | 8.803e-5 m | 9.010e-5 m | 9.384e-4 rad |

对比 heart live 基线：

| 对比项 | 最终关节漂移范数 |
|---|---:|
| heart live baseline | 1.416e-3 rad |
| 本次 tuned simple trajectory best | 9.371e-4 rad |

因此，本次简单轨迹 tuned live 已经优于 heart live 的角度漂移结果。

对比图：

```text
C:\Users\lyx\Desktop\sci\code\project_tvqp_elncp_pdnn_comparison\results\drift_feedback_tuning_live\tuned_live_vs_heart_drift.png
```

## 5. Offline 检查结果

离线验证表保存位置：

```text
C:\Users\lyx\Desktop\sci\code\project_tvqp_elncp_pdnn_comparison\results\drift_feedback_tuning_offline\tuned_offline_summary.csv
```

代表性结果如下：

| 轨迹 | 漂移反馈 | 平均位置误差 | 最终位置误差 | 最终关节漂移范数 |
|---|---|---:|---:|---:|
| small circle | nonlinear | 7.383e-6 m | 4.934e-7 m | 1.695e-6 rad |
| line | linear | 6.473e-6 m | 5.143e-7 m | 7.170e-6 rad |
| line | nonlinear | 7.247e-6 m | 1.642e-6 m | 8.203e-6 rad |

offline 结果说明：在理想离散仿真中，当前参数已经能把最终角度漂移压到微弧度量级；live 中剩余的 `9e-4 rad` 主要来自 CoppeliaSim 关节伺服、同步通信、目标位置执行误差和闭环采样误差。

## 6. 结论

1. 用户指出“参数还有很大改进空间”是正确的。
2. 前一轮没有达到 heart live 的主要原因不是简单轨迹导致精度差，而是 Method 2 新项目中使用了线性漂移反馈，没有恢复旧 heart 基线中的非线性漂移反馈结构。
3. 只提高 `dlccznn_inner_steps` 主要降低位置跟踪误差，但对最终角度回零不够；最终角度漂移更依赖 `drift_gain`、漂移反馈非线性形式、`task_gain` 与 `solver_gamma` 的配合。
4. 采用 `task_gain=260, drift_gain=20, solver_gamma=3072, activation_power=0.80, dlccznn_inner_steps=60, nonlinear drift feedback` 后，五条简单轨迹 live 的最终关节漂移均约为 `9.37e-4 rad`，已经优于 heart live 的 `1.416e-3 rad`。

## 7. 后续可继续优化方向

如果还要进一步压低 live 角度漂移，建议按以下优先级继续：

1. 在 `drift_gain=20` 附近扫描 `15, 20, 30, 40`，观察是否能把最终漂移压到 `5e-4 rad` 附近。
2. 在 `task_gain=220, 260, 300` 中寻找位置误差和回零误差的平衡点。
3. 保持 `activation_power=0.80`，再测试 `0.75` 与 `0.85`，避免非线性反馈过强导致 live 伺服抖动。
4. `solver_gamma=3072` 已经有效，不建议首先继续盲目加大；若要加大，应同时观察 solver residual 和 live 运行时间。
5. `dlccznn_inner_steps=60` 当前能给出稳定结果。若考虑复杂度，可以补测 `Ns=30, 40, 50`，寻找精度和计算量的折中点。
