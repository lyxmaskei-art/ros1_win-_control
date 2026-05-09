# Workspace Cleanup Status

整理日期：2026-05-09

## 已完成整理

| 原位置 | 新位置 | 说明 |
| --- | --- | --- |
| `qp.md` 等根目录散落 Markdown | `notes/project_overview` | 项目脉络、导师汇报、QP 说明等文档集中管理。 |
| `Pasted image *.png` | `notes/assets/pasted_images` | 粘贴图片集中管理。 |
| `paper_ras_discrete_driftfree` | `papers/paper_ras_discrete_driftfree` | 论文项目与代码区分开。 |
| `code/docs` | `experiments/.../docs` 和 `notes/project_overview` | 方法推导、调参报告和项目总览归到对应实验或总览目录。 |
| `code/results/single_file_integrated_runs` | `experiments/legacy_integrated_scripts/results` | 早期单文件集成结果归入历史实验目录。 |
| `ros1_win-_control` | `C:\Users\lyx\Desktop\ros1_win-_control` | GitHub 镜像移出 `sci`，避免和本地实验目录重复。 |

## 已删除的重复和临时内容

| 删除对象 | 原因 |
| --- | --- |
| `tmp` | 只包含 PDF 抽取文本、smoke 输出和临时脚本，不作为正式实验数据。 |
| `experiments/migration_validation_20260509` | 只用于迁移 smoke 和 GUI 验证，正式结果已在对应实验目录中。 |
| 各实验 `results/*smoke*`、`results/*_tmp*` | 迁移或连通性验证产物，不作为正式论文数据。 |
| `__pycache__` 和 `*.pyc` | Python 缓存，可自动再生成。 |
| `code/docs`、`code/results`、`code/.obsidian` | 与当前目录规范冲突，且内容已迁移或无正式用途。 |

## 保持原位的关键目录

| 目录 | 原因 |
| --- | --- |
| `code/core` | 公共代码依赖。 |
| `code/methods` | 可复用方法实现。 |
| `code/sim` | CoppeliaSim remote API 依赖。 |
| `reference` | 本地参考文献库，避免上传到 GitHub。 |
| `experiments` | 正式实验代码、结果和文档的唯一主目录。 |
| `.obsidian`, `.claude`, `.uploads` | 工具配置和状态目录。 |

## 仍然需要人工确认的历史 git 状态

外层 `sci` 仓库仍然显示大量历史删除和未跟踪文件。原因是该仓库早已处于 dirty 状态，本次整理没有执行 `git reset`、`git clean` 或批量删除。

建议后续单独处理：

1. 确认旧 PDF、PPT、Word 文件是否只保留在 `reference` 或 `papers`。
2. 若要让外层 `sci` git 变干净，应先决定是继续维护这个总仓库，还是只维护 `C:\Users\lyx\Desktop\ros1_win-_control` 这个发布仓库。

## 当前建议

短期内不要对外层 `sci` 执行 `git reset --hard` 或 `git clean -fd`。当前目录已经按实验归类，后续新增实验应直接建在 `experiments` 下，并把同一实验的代码、结果和文档保存在同一个项目文件夹中。
