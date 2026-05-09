# Experiments Index

本目录保存所有完整实验项目。每个项目尽量保持自己的 `scripts`、`results`、`docs` 和 `README.md`，避免实验结果散落到 `code/core`。

## 分类

| 分类目录 | 研究线 | 内容 |
| --- | --- | --- |
| `baseline_original_driftfree` | 原始 drift free 与早期基线 | Method 3a PCG、Method 3c LCCZNN、早期 Method 1/2/3 对照。 |
| `dlccznn_tvqp` | DLCCZNN 与连续 TVQP | 连续 TVQP 加 ELNCP 后用 DLCCZNN 离散求解，Method 2 DLCCZNN ELNCP 边界处理。 |
| `spd_box_qp_pcg` | Reduced SPD box QP 与 PCG | Method 3 TVQP ELNCP PCG 主实验，以及 SPD box QP 泛化测试。 |
| `solver_comparison_integration` | 求解器与积分方式对比 | PDNN 对照实验，RK45 与 Euler 对比。 |
| `trajectory_ablation_tuning` | 轨迹、反馈消融和调参 | 简单轨迹、反馈有无、参数调优和最终 live 结果。 |
| `robustness_noise` | 噪声鲁棒性 | Method 2/3a 噪声测试和 Method 3 PCG 噪声鲁棒性。 |
| `legacy_integrated_scripts` | 历史集成脚本 | 从 `code/scripts` 迁出的早期全集成入口。 |

## 迁移验证标准

每个项目迁移后至少需要通过以下检查：

1. Python 编译检查：确认脚本语法和导入路径没有直接断裂。
2. 入口检查：运行 `--help` 或等价入口，确认参数解析可用。
3. Offline 检查：生成轨迹误差、角度漂移误差、关节角图和 summary 文件。
4. Live 检查：在 CoppeliaSim 已打开时生成 live 图和 summary；若无法连接，记录为环境连接失败。

正式论文数据仍以各项目 `results` 中标准时长结果为准。迁移烟测结果只用于证明目录转换没有破坏可运行性。
