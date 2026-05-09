# 2026-04-21 参数调优与阶段进展总结

## 1. 今天具体完成了什么

- 为 `Method 2 DLCCZNN` 和 `Method 3a PCG` 主脚本补上了安全扫参入口：
  - 支持自定义 `--output-root`
  - 支持 `--skip-figures`
  - 避免批量实验覆盖原有保留结果
- 新建了两套可复现的批量调参脚本：
  - `project_method3c_lccznn_solver/scripts/tune_method2_dlccznn.py`
  - `project_method3a_pcg_solver/scripts/tune_method3a_pcg.py`
- 完成了两条线的大规模离线调参：
  - Method 2: `81` 组 broad + `243` 组 refined = `324` 组 offline
  - Method 3a PCG: `243` 组 broad + `243` 组 refined + `2` 组预条件对照 = `488` 组 offline
- 完成了两条线的 live 调参：
  - Method 2: `9` 组 live
  - Method 3a PCG: `9` 组 live
- 针对 live 结果继续做了邻域补跑，而不是停留在第一轮候选。
- 验证了 `Method 3a PCG` 中 `Jacobi` 预条件器的边际作用。

## 2. PCG 替代后方法的时间复杂度

对 `Method 3a PCG` 而言，每个控制周期需要求解一个 `n x n` 对称正定线性系统，其中当前问题里：

$$
n = 6,\quad m = 3
$$

先构造

$$
Q = w_p J^\top J + w_d I + \lambda I
$$

其中 `J \in \mathbb{R}^{m \times n}`。构造 `J^\top J` 的主复杂度为

$$
\mathcal{O}(m n^2)
$$

PCG 每次迭代主要是一次矩阵向量乘和若干向量内积，其主复杂度为

$$
\mathcal{O}(n^2)
$$

若平均 PCG 迭代次数为 `k`，则单周期总复杂度可写成

$$
\mathcal{O}(m n^2 + k n^2) = \mathcal{O}((m+k)n^2)
$$

对当前推荐的 live 最优参数组合，实测平均内迭代次数约为

$$
k \approx 4.67
$$

因此当前问题上的主复杂度量级是

$$
\mathcal{O}((3 + 4.67)\cdot 6^2)
$$

也就是一个很小规模的二次复杂度迭代求解过程。

结论是：

- 从渐近形式看，PCG 替代后的内层求解复杂度是二次型 `\mathcal{O}((m+k)n^2)`。
- 但由于 `n=6` 很小，它的工程优势不在于“绝对 wall-clock 一定比直接解更快”，而在于：
  - 它是结构匹配的迭代替代器
  - 可以 warm start
  - 可以用有限迭代控制内层代价
  - 更适合以后向更高维冗余系统扩展

## 3. 为什么 LCCZNN 效果差

这里的“差”主要是指：放到 `Method 3a` 的内层后，位置精度还能维持，但 drift-free 恢复能力明显丢失。

核心原因不是代码写错，而是求解器和问题结构不匹配：

- `Method 3a` 的内层本质上是一个小规模、强结构、对精度极敏感的时变 SPD 线性方程组。
- LCCZNN 在这里做的是“残差跟踪式逼近”，不是每一步都把 SPD 线性系统高精度解到位。
- drift-free 目标对内层误差极敏感。位置误差小，不代表关节漂移恢复也小。
- LCCZNN 为了稳定通常还会引入：
  - 子步长离散
  - 幂次激活
  - 指数裁剪
  - 残差能量归一化
- 这些机制对 KKT 根跟踪类问题可以工作，但对 `Method 3a` 这种“每一步都需要很准”的内层 SPD 线性求解，误差会累积到漂移通道里。

一句话总结：

- `Method 3a` 需要的是“高精度、结构匹配的线性求解器”。
- LCCZNN 更像“低复杂度残差动态跟踪器”。
- 所以它能保位置，保不住最关键的 drift-free 恢复。

## 4. 今天最重要的发现

### 4.1 Method 2

- offline 最优区和 live 最优区并不一致。
- 如果只看 offline 平均位置误差，最优点会落在低 `task_gain` 区域。
- 但 live 真正最优的 drift-free 区域明显向更高 `task_gain`、更高 `mu_gain`、更高 `Ns`、更高 `solver_gamma_gain` 偏移。
- 在本次 live 扫描区间内，`mu_gain` 从 `12 -> 14 -> 16 -> 18 -> 20` 时，末端关节漂移是单调下降的。
- `activation_power = 0.85` 比 `0.90/0.95` 更适合当前 live 闭环环境。

### 4.2 Method 3a PCG

- `task_gain = 216` 基本会把稳定工作区推坏，offline 和 live 都不值得保留。
- 在稳定区内，前几名配置的平均位置误差几乎一样，差异主要体现在最终漂移。
- live 中更优的漂移恢复并不是来自“更大的 drift_weight”，反而来自：

$$
\text{更大的 } position\_weight + \text{更小的 } drift\_weight
$$

- 这说明 live 环境下更准确的任务通道求解，会间接减轻漂移补偿通道的累计压力。
- 对当前 `6 x 6` 小系统，`Jacobi` 预条件器不是决定性因素，权重组合比预条件器更重要。

## 5. 当前推荐的 live 参数组合

### 5.1 Method 2 推荐组合

若以 drift-free 效果为主要目标，当前推荐保留：

$$
task\_gain = 220,\quad
mu\_gain = 20,\quad
activation\_power = 0.85,\quad
solver\_substeps = 80,\quad
solver\_gamma\_gain = 4608
$$

对应 live 结果为：

- mean position error: `2.466074e-4 m`
- final position error: `1.043252e-4 m`
- final joint drift norm: `1.430160e-3 rad`

相对原保留 live 结果：

- mean position error 改善约 `1.364x`
- final position error 改善约 `1.362x`
- final joint drift 改善约 `3.178x`

### 5.2 Method 3a PCG 推荐组合

若以 drift-free 效果为主要目标，当前推荐保留：

$$
task\_gain = 198,\quad
position\_weight = 15000,\quad
drift\_weight = 5\times10^{-4},\quad
regularization\_gain = 10^{-10},\quad
solver\_max\_iters = 6
$$

对应 live 结果为：

- mean position error: `1.302302e-4 m`
- final position error: `5.814786e-5 m`
- final joint drift norm: `3.197149e-4 rad`
- mean inner iterations: `4.6655`

相对原保留 live 结果：

- mean position error 基本持平
- final position error 基本持平
- final joint drift 改善约 `1.273x`

## 6. 可以直接写进论文或汇报的点

- `Method 2` 存在明显的 offline/live 最优区偏移，说明只靠离线最小误差选参不可靠。
- `Method 2` 中 `mu_gain` 对 live drift-free 恢复是主导参数，而 `solver_substeps` 主要决定是否能把该恢复效果稳定兑现出来。
- `Method 3a PCG` 中，live 最优点并不在“大 drift_weight”区域，而是在“更强任务约束 + 更轻漂移权重”的组合上。
- 对 `6 x 6` 小规模 SPD 系统，PCG 的实际优势主要来自结构匹配与 warm start，而不是简单的渐近复杂度口号。

## 7. 今天的工程创新点

- 把原先分散、不可复现的手工试参，变成了可重跑的 broad/refine/live 三阶段调参流程。
- 把 live 结果真正纳入选参标准，而不是只看 offline。
- 对 Method 2 找到了比原保留配置明显更强的 live 参数带。
- 对 Method 3a PCG 找到了“位置精度几乎不变，但漂移显著更低”的新组合。

## 8. 相关文件

- 总结文档：`C:\Users\lyx\Desktop\sci\code\tuning_progress_20260421.md`
- Method 2 调参文档：`C:\Users\lyx\Desktop\sci\code\project_method3c_lccznn_solver\docs\method2_live_tuning_20260421.md`
- Method 3a 调参文档：`C:\Users\lyx\Desktop\sci\code\project_method3a_pcg_solver\docs\method3a_live_tuning_20260421.md`
