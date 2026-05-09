# Code Workspace

`code` 现在只作为共享代码区，不再存放项目实验、实验结果或公式文档。

## 子目录

| 目录 | 用途 |
| --- | --- |
| `core` | UR3e 运动学、通用轨迹、CoppeliaSim 连接、结果统计与绘图工具。 |
| `methods` | 可复用的方法级实现，例如连续 TVQP DLCCZNN、早期集成方法控制器。 |
| `sim` | CoppeliaSim remote API Python 文件和动态库。 |
| `.vscode` | 本地工具配置。 |

## 路径约定

实验脚本应通过向上搜索定位 `sci/code`，再把以下目录加入 `sys.path`：

| 路径 | 作用 |
| --- | --- |
| `code/core` | 通用基础模块，例如 `discrete_rmp_qp_common.py`。 |
| `code/methods` | 方法模块，例如 `continuous_tvqp_dlccznn.py`。 |
| `code/sim` | CoppeliaSim remote API。 |

不要再让新实验依赖“脚本必须位于 `code/project_xxx/scripts`”这种位置假设。

## 实验和文档位置

早期 `code/results/single_file_integrated_runs` 已迁移到 `../experiments/legacy_integrated_scripts/results`。

公式推导、实验说明和调参报告已迁移到对应的 `experiments/.../docs` 或 `notes/project_overview`。

后续规则：`code` 只保存可复用代码和仿真接口；offline、live、调参、鲁棒性和论文制表结果全部放到 `experiments` 对应实验分类中。
