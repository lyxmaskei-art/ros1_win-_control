# Method 2 DLCCZNN 多参数耦合搜索报告

## 1. 搜索目标

本轮搜索目标不是单独追求 offline 最小误差，而是寻找可以在 live 仿真中稳定降低最终关节角度漂移的参数组合。评价指标优先级为：

1. 最终关节漂移范数尽可能小。
2. 平均位置误差不能明显退化。
3. 参数在多条轨迹上保持稳定，而不是只对单条轨迹过拟合。

基线为上一轮 tuned live 参数：

| 参数 | 数值 |
|---|---:|
| task gain | 260 |
| drift gain | 20 |
| solver gamma | 3072 |
| activation power | 0.80 |
| DLCCZNN inner steps | 60 |
| drift feedback mode | nonlinear |

上一轮五条轨迹 live 的最终角度漂移约为 `9.37e-4 rad`。

## 2. 第一阶段：task gain 与 drift gain 耦合

搜索范围：

| 参数 | 搜索值 |
|---|---|
| task gain | 180, 220, 260, 300, 340 |
| drift gain | 10, 15, 20, 30, 40, 60 |
| solver gamma | 固定 3072 |
| activation power | 固定 0.80 |
| DLCCZNN inner steps | 固定 60 |

代表轨迹为 `small_circle` 和 `line`，使用 offline 快速筛选。

主要发现：

1. `drift_gain=10` 在 offline 中最容易得到极低最终漂移。
2. `task_gain=300` 或 `340` 同时有利于降低平均位置误差。
3. `drift_gain` 过大在 offline 中会破坏回零效果，尤其 `40` 和 `60` 明显退化。

这一阶段的最佳 offline 区域为：

| 参数 | 推荐区间 |
|---|---|
| task gain | 300 到 340 |
| drift gain | 10 |

## 3. 第二阶段：solver gamma, activation power 与 Ns 耦合

搜索范围：

| 参数 | 搜索值 |
|---|---|
| task gain | 300, 340 |
| drift gain | 10 |
| solver gamma | 2048, 3072, 4096, 4608 |
| activation power | 0.75, 0.80, 0.85 |
| DLCCZNN inner steps | 40, 60 |

代表轨迹仍为 `small_circle` 和 `line`。

offline 跨轨迹最优候选为：

| task gain | drift gain | gamma | power | Ns | 平均位置误差 | 最大最终漂移 |
|---:|---:|---:|---:|---:|---:|---:|
| 340 | 10 | 2048 | 0.85 | 60 | 5.532e-6 m | 5.363e-7 rad |
| 300 | 10 | 2048 | 0.85 | 60 | 5.959e-6 m | 5.662e-7 rad |
| 340 | 10 | 3072 | 0.85 | 60 | 4.853e-6 m | 1.057e-6 rad |
| 340 | 10 | 4096 | 0.85 | 60 | 4.958e-6 m | 1.123e-6 rad |

主要发现：

1. `activation_power=0.85` 在 offline 中明显优于 `0.75` 和 `0.80`。
2. `Ns=60` 明显优于 `Ns=40`。
3. `gamma` 不是越大越好，`2048` 和 `3072` 最稳。

## 4. 第三阶段：五轨迹 offline 验证

将候选扩展到五条轨迹：`small_circle`, `line`, `circle`, `ellipse`, `figure8`。

offline 最优候选为：

| 参数 | 数值 |
|---|---:|
| task gain | 340 |
| drift gain | 10 |
| solver gamma | 3072 |
| activation power | 0.85 |
| DLCCZNN inner steps | 60 |

五轨迹 offline 平均位置误差约为 `4.05e-6 m`，最大最终关节漂移约为 `3.30e-7 rad`。

但是该候选在 live 中并不最优。

## 5. 第四阶段：live 检验与修正

将 offline 最优候选 `task=340, drift=10, gamma=3072, power=0.85, Ns=60` 放到 live 中验证，结果最终关节漂移升高到约 `3.16e-3 rad`。

这说明 offline 最优不等于 live 最优。原因是 live 中存在 CoppeliaSim 关节伺服误差、同步通信误差和实际目标位置执行滞后。`drift_gain=10` 在理想离线模型中足够，但在 live 执行扰动下回零驱动不足。

因此，live 搜索重新围绕 `drift_gain` 做局部扫描。

## 6. 第五阶段：live line 局部扫描

代表轨迹选 `line`，因为之前的 live 结果中各简单轨迹最终漂移非常接近，`line` 可以作为快速代表。

局部扫描结果：

| task gain | drift gain | gamma | power | Ns | 平均位置误差 | 最终位置误差 | 最终关节漂移 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 30 | 3072 | 0.80 | 60 | 7.808e-5 m | 7.646e-5 m | 5.813e-4 rad |
| 260 | 30 | 3072 | 0.80 | 60 | 9.015e-5 m | 8.795e-5 m | 5.914e-4 rad |
| 260 | 25 | 3072 | 0.80 | 60 | 8.894e-5 m | 9.004e-5 m | 7.249e-4 rad |
| 300 | 20 | 3072 | 0.80 | 60 | 7.643e-5 m | 7.657e-5 m | 9.284e-4 rad |
| 220 | 20 | 3072 | 0.80 | 60 | 1.042e-4 m | 1.044e-4 m | 9.457e-4 rad |
| 260 | 15 | 3072 | 0.80 | 60 | 8.751e-5 m | 9.001e-5 m | 1.321e-3 rad |
| 260 | 20 | 3072 | 0.85 | 60 | 8.835e-5 m | 8.948e-5 m | 1.418e-3 rad |

live 中最优候选变为：

| 参数 | 数值 |
|---|---:|
| task gain | 300 |
| drift gain | 30 |
| solver gamma | 3072 |
| activation power | 0.80 |
| activation exp clip | 4.0 |
| DLCCZNN inner steps | 60 |
| drift feedback mode | nonlinear |

## 7. 最终五轨迹 live 结果

最终参数 `task=300, drift=30, gamma=3072, power=0.80, Ns=60` 在五条轨迹上的 live 结果如下：

| 轨迹 | 平均位置误差 | 最终位置误差 | 最终关节漂移范数 |
|---|---:|---:|---:|
| figure8 | 9.837e-5 m | 7.705e-5 m | 5.800e-4 rad |
| line | 7.808e-5 m | 7.646e-5 m | 5.813e-4 rad |
| circle | 1.025e-4 m | 7.680e-5 m | 5.814e-4 rad |
| small circle | 8.497e-5 m | 7.795e-5 m | 5.818e-4 rad |
| ellipse | 1.089e-4 m | 7.796e-5 m | 5.822e-4 rad |

相比上一轮 tuned live 的 `9.37e-4 rad`，最终漂移下降约 `38%`。

相比 heart live 基线的 `1.416e-3 rad`，最终漂移下降约 `59%`。

结果图：

```text
C:\Users\lyx\Desktop\sci\code\project_tvqp_elncp_pdnn_comparison\results\coupled_param_search\stage6_live_final_t300_d30\live_drift_comparison_old_vs_coupled_best.png
```

## 8. 参数关系总结

1. `task_gain` 控制轨迹反馈强度。live 中 `300` 比 `260` 更有利于位置误差，同时不会破坏最终回零。
2. `drift_gain` 对 live 关节回零最敏感。offline 中 `10` 最优，但 live 中 `30` 更优，说明仿真执行扰动需要更强的回零反馈。
3. `activation_power` 在 offline 中 `0.85` 更强，但 live 中会使最终漂移反而变差；当前 live 最优仍是 `0.80`。
4. `solver_gamma=3072` 是目前最稳的折中。`2048` 在 offline 稳，但 live 回零不足；更高的 `4096/4608` 未显示出必要优势。
5. `Ns=60` 仍是当前精度优先设置。若后续考虑复杂度，可以在最终参数附近补测 `Ns=40, 50`。

## 9. 当前推荐参数

用于追求 live 最小关节角度漂移：

```text
task_gain = 300
drift_gain = 30
solver_gamma = 3072
activation_power = 0.80
activation_exp_clip = 4.0
dlccznn_inner_steps = 60
drift_feedback_mode = nonlinear
```

如果后续要继续优化，建议只围绕 live 做小范围搜索：

| 参数 | 建议补测 |
|---|---|
| task gain | 280, 300, 320 |
| drift gain | 28, 30, 32, 35 |
| activation power | 0.78, 0.80, 0.82 |
| Ns | 50, 60 |

不建议继续用 offline 最优直接替代 live 最优，因为本轮已经证明两者目标存在明显偏差。
