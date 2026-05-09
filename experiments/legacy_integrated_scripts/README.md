# Legacy Baseline Scripts

这里保存早期三种 drift free baseline 的历史代码和结果。当前代码已经按实验入口拆分为独立 offline 与 live 文件，不再保留早期全集成脚本。

这些脚本仍可直接运行，但属于历史 baseline。后续新实验统一采用每个项目下两个明确入口：

| 入口 | 作用 |
| --- | --- |
| `run_offline.py` | 只跑 offline 仿真、绘图、表格和 summary。 |
| `run_live.py` | 只跑 CoppeliaSim live、绘图、表格和 summary。 |

共享参数名和输出字段应保持一致，方便横向比较和论文制表。

## 结果目录

`results` 保存早期三种 baseline 的历史结果，对应 `scripts` 下的独立入口：

| 目录 | 含义 |
| --- | --- |
| `results/method1_sig_rmp_qp` | Method 1 原始 drift free QP baseline 的 offline 和 live 输出。 |
| `results/method2_scheme_b_positive` | Method 2 原始 drift free QP baseline 的 offline 和 live 输出。 |
| `results/method3a_position_and_drift` | Method 3a 原始 position and drift QP baseline 的 offline、live 和 20 s live 输出。 |
| `results/_paper_metrics` | 早期论文制表用指标快照。 |

这些结果属于原始三方法 baseline 的历史数据；新的 DLCCZNN、TVQP、PCG、鲁棒性和轨迹消融结果统一放在 `experiments` 下对应分类目录。
