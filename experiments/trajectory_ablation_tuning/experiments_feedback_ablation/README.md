# Feedback Ablation Experiments

这里保存从 `sci` 根目录移入的 TVQP 反馈消融实验结果。

## 子目录

| 目录 | 说明 |
| --- | --- |
| `experiments_tvqp_with_feedback` | 有反馈版本的早期 TVQP offline 实验。 |
| `experiments_tvqp_without_feedback` | 无反馈版本的早期 TVQP offline 实验。 |
| `experiments_tvqp_dlccznn_paper_with_feedback` | DLCCZNN 论文式推导加反馈，含 offline 和 live。 |
| `experiments_tvqp_dlccznn_paper_without_feedback` | DLCCZNN 论文式推导去反馈，含 offline 和 live。 |
| `experiments_tvqp_dlccznn_paper_singleupdate_with_feedback` | 单步更新加反馈版本。 |

这些结果用于支撑反馈结构、是否单步更新对 drift free 效果的影响分析。
