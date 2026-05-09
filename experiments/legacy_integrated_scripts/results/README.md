# Legacy Baseline Results

本目录保存早期三种 drift free baseline 的历史结果。当前代码入口已经拆分为独立 offline 与 live 文件，不再使用早期单文件全集成脚本。

对应入口位于 `../scripts`：

| 脚本 | 结果目录 |
| --- | --- |
| `run_method1_offline.py` 与 `run_method1_live.py` | `method1_sig_rmp_qp` |
| `run_method2_offline.py` 与 `run_method2_live.py` | `method2_scheme_b_positive` |
| `run_method3a_offline.py` 与 `run_method3a_live.py` | `method3a_position_and_drift` |

`_paper_metrics` 是早期论文制表指标快照。临时 `_tmp_*` 验证结果已经删除，不作为正式论文精度结论。
