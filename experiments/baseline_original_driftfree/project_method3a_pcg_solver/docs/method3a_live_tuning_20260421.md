# Method 3a PCG Live 调参记录 2026-04-21

## 1. 调参规模

- offline broad: `243` 组
- offline refined: `243` 组
- live 候选与邻域补跑: `9` 组
- `Jacobi` 预条件器对照: `2` 组 offline

## 2. 本轮扫描的核心参数

$$
task\_gain,\quad
position\_weight,\quad
drift\_weight,\quad
regularization\_gain,\quad
solver\_max\_iters
$$

PCG 替代后的单周期主复杂度为

$$
\mathcal{O}(m n^2 + k n^2) = \mathcal{O}((m+k)n^2)
$$

其中：

$$
n = 6,\quad m = 3,\quad k = \text{平均 PCG 迭代次数}
$$

对当前推荐 live 配置，实测

$$
k \approx 4.6655
$$

## 3. 主要结论

### 3.1 `task_gain = 216` 不值得保留

- 在 broad sweep 中，`task_gain = 216` 会把误差和漂移同时推坏。
- 因此实际可用工作区主要在 `162/180/198`，其中 `198` 最稳。

### 3.2 stable 区域里，平均位置误差几乎是平的

对 top 几组 live 结果而言：

- mean position error 都在 `1.302e-4 m` 左右
- final position error 都在 `5.81e-5 m` 左右

这意味着：

- live 选参的真正判别量不是位置均值
- 而是最终漂移和内层迭代代价

### 3.3 live 中更好的漂移恢复来自“大 `position_weight` + 小 `drift_weight`”

这一点和直觉相反，但实验很清楚：

- `position_weight = 15000`
- `drift_weight = 5e-4`

这组在 live 中给出了当前最小的最终关节漂移。

解释是：

- 更强的任务通道约束让内层 SPD 线性系统的求解方向更稳定
- 过大的 `drift_weight` 会把实时闭环中的漂移补偿项直接压进求解器，反而放大了 live 扰动下的漂移残留

### 3.4 `solver_max_iters = 6` 是当前最好平衡点

- `iters = 5` 可以工作，也能保持较低复杂度
- 但在当前 best live 邻域里，`iters = 6` 的最终漂移更低
- `iters = 7` 在部分 offline 点上有效，但没有体现出稳定的 live 优势

### 3.5 `Jacobi` 预条件器在当前 `6 x 6` 小系统里不是决定项

对当前 best live 邻域点做 offline 对照：

- 有 `Jacobi`: mean iterations `4.77625`
- 无 `Jacobi`: mean iterations `4.8565`

两者差别很小，且误差指标几乎一致。

结论是：

- 在这个 `6 x 6` 小系统里，预条件器不是主导因素
- 权重组合比预条件器重要得多

## 4. 当前推荐保留配置

若以 drift-free 效果为主要目标，当前推荐：

$$
task\_gain = 198,\quad
position\_weight = 15000,\quad
drift\_weight = 5\times10^{-4},\quad
regularization\_gain = 10^{-10},\quad
solver\_max\_iters = 6
$$

对应 live 结果：

- mean position error: `1.302302e-4 m`
- final position error: `5.814786e-5 m`
- final joint drift norm: `3.197149e-4 rad`
- mean inner iterations: `4.6655`

相对旧保留 live 配置：

- mean position error 基本不变
- final position error 基本不变
- final joint drift 改善约 `1.273x`

## 5. 若只追求最小平均位置误差

若只看本轮 live 扫描中的最小平均位置误差，则更优的是：

$$
task\_gain = 198,\quad
position\_weight = 6000,\quad
drift\_weight = 0.004,\quad
regularization\_gain = 10^{-9},\quad
solver\_max\_iters = 6
$$

但这个点的 drift-free 能力不如推荐配置，因此不建议作为最终保留点。

## 6. 文件位置

- 调参结果目录：
  - `C:\Users\lyx\Desktop\sci\code\project_method3a_pcg_solver\results\tuning_20260421\method3a_pcg`
- 当前推荐 live 点：
  - `C:\Users\lyx\Desktop\sci\code\project_method3a_pcg_solver\results\tuning_20260421\method3a_pcg\live_selected_runs\tg198_pw1p5e04_dw0p0005_rg1em10_it6_jac1\live`
