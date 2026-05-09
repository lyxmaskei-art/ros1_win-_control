# SCI Drift Free Redundant Manipulator Control Workspace

本仓库保存 UR3e 冗余机械臂 drift free 轨迹跟踪控制研究的代码、实验结果、论文草稿、参考文献和阶段性研究笔记。当前工作重点包括离散 drift free QP、DLCCZNN、连续 TVQP、ELNCP、SPD box QP、PCG、PDNN 对照、RK45 求解器对照、简单轨迹消融和噪声鲁棒性验证。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `code` | 公共代码、UR3e 运动学、CoppeliaSim remote API、共享方法模块和仿真场景。新实验不再直接堆在 `code` 根目录。 |
| `experiments` | 所有正式实验项目。每个实验尽量包含独立 `scripts`、`results`、`docs` 和 `README.md`。 |
| `notes` | 研究路线、方法脉络、导师汇报、问题分析和 Obsidian 相关笔记。 |
| `papers` | 论文草稿、LaTeX 工程、图表和投稿相关材料。 |
| `reference` | 本地参考论文、PDF、论文解析中间文件和文献资料。 |

## 主要实验分类

| 分类 | 说明 |
| --- | --- |
| `experiments/baseline_original_driftfree` | 原始三类 drift free baseline、Method 3a PCG 和 Method 1/2/3c DLCCZNN 版本。 |
| `experiments/dlccznn_tvqp` | 连续 TVQP 加 ELNCP 加 DLCCZNN 的重构方法，以及 Method 2 DLCCZNN 边界处理实验。 |
| `experiments/spd_box_qp_pcg` | Method 3 TVQP 加 ELNCP 加 PCG，以及 reduced SPD box QP 泛化研究。 |
| `experiments/solver_comparison_integration` | PDNN 对照、DLCCZNN 对照、RK45 对照和组合实验。 |
| `experiments/trajectory_ablation_tuning` | 简单轨迹、反馈去除、参数调节和 live 参数搜索结果。 |
| `experiments/robustness_noise` | 借鉴已有 ZNN 论文噪声设置的鲁棒性验证实验。 |
| `experiments/legacy_integrated_scripts` | 早期历史 baseline 入口和历史结果。只作为追溯使用，不建议作为新实验入口。 |

## 实验入口规则

当前正式实验遵循两个入口原则：

| 文件 | 用途 |
| --- | --- |
| `run_offline.py` 或 `run_method*_offline.py` | 只运行 offline 仿真、绘图、表格和 summary。 |
| `run_live.py` 或 `run_method*_live.py` | 只运行 CoppeliaSim live 仿真、绘图、表格和 summary。 |

每个正式入口都是完整可运行文件，不依赖单文件全集成 runner，也不使用 `--mode` 在同一脚本里切换 offline 和 live。

## 严谨实验输出规则

为了避免覆盖旧数据，正式实验默认会把每次运行保存到独立目录：

```text
results/.../runs/<运行模式>__<关键参数标签>__<时间戳>/
```

每次运行会生成：

| 文件或目录 | 说明 |
| --- | --- |
| `run_config.json` | 本次运行时间、脚本路径、输出目录和全部命令行参数。 |
| `*_summary.json` | 关键误差、漂移、运行时间和求解器指标。 |
| `*_summary.txt` | 可读 summary。部分脚本生成。 |
| `*_trajectory.png` | 期望轨迹与实际轨迹对比。 |
| `*_position_error.png` | 位置跟踪误差。 |
| `*_joint_drift.png` | 关节角漂移误差。 |
| `*_joint_angles.png` | 关节角时间序列。 |
| `*_joint_table.csv/png` | 初末关节角和漂移表。部分脚本生成。 |

如果手动传入 `--output-root`，脚本会尊重指定路径，不再自动加时间戳。做正式实验时建议不要传 `--output-root`，除非已经手动给出唯一目录。

## 典型运行方式

在对应实验的脚本目录中运行，例如：

```powershell
cd C:\Users\lyx\Desktop\sci\experiments\spd_box_qp_pcg\project_method3_tvqp_elncp_pcg_solver\scripts
python run_offline.py --tau 0.005 --pcg-max-iters 8
python run_live.py --tau 0.005 --pcg-max-iters 8
```

Method 2 DLCCZNN 示例：

```powershell
cd C:\Users\lyx\Desktop\sci\experiments\dlccznn_tvqp\project_method2_dlccznn_elncp_bounds\scripts
python run_offline.py --tau 0.005 --solver-substeps 60
python run_live.py --tau 0.005 --solver-substeps 60
```

简单轨迹消融示例：

```powershell
cd C:\Users\lyx\Desktop\sci\experiments\trajectory_ablation_tuning\project_simple_shape_trajectories\scripts
python run_method2_offline.py --trajectory-name circle
python run_method2_live.py --trajectory-name circle
```

## 环境依赖

建议环境：

| 组件 | 说明 |
| --- | --- |
| Python 3.10 或 3.11 | 当前脚本已通过 `py_compile` 语法检查。 |
| `numpy` | 数值计算。 |
| `matplotlib` | 绘图。 |
| `scipy` | RK45 和部分数值求解。 |
| CoppeliaSim | live 仿真实验需要。 |
| CoppeliaSim remote API | 已放在 `code/sim`。 |

安装常用 Python 包：

```powershell
pip install numpy matplotlib scipy
```

## 验证命令

整理或迁移后可运行：

```powershell
@'
import py_compile
from pathlib import Path
files = list(Path('code').rglob('*.py')) + list(Path('experiments').rglob('*.py'))
errors = []
for p in files:
    try:
        py_compile.compile(str(p), doraise=True)
    except Exception as exc:
        errors.append((p, exc))
print(f'compiled={len(files)} errors={len(errors)}')
for p, exc in errors:
    print(p, exc)
'@ | python -
```

当前整理后验证结果为：

```text
compiled=44 errors=0
pycache=0 pyc=0
```

## GitHub 同步说明

本仓库用于同步到：

```text
git@github.com:lyxmaskei-art/ros1_win-_control.git
```

本次同步目标是以当前 `sci` 文件夹为准覆盖远端内容。由于包含论文 PDF、实验图片和历史结果，仓库体积较大；如果后续转入 Linux 做 ROS 实机实验，建议优先使用 `experiments` 中的正式实验入口和 `code/sim`、`code/core` 里的公共代码。
