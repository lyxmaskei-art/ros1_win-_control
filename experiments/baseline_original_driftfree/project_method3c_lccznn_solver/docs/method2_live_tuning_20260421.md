# Method 2 Live 调参记录 2026-04-21

## 1. 调参规模

- offline broad: `81` 组
- offline refined: `243` 组
- live 候选与邻域补跑: `9` 组

## 2. 本轮扫描的核心参数

$$
task\_gain,\quad
mu\_gain,\quad
activation\_power,\quad
solver\_substeps,\quad
solver\_gamma\_gain
$$

其中复杂度上最敏感的是 `solver_substeps = N_s`，因为 Method 2 内层 DLCCZNN 的单周期复杂度近似正比于

$$
\mathcal{O}\!\left(N_s (n+m)^2\right)
$$

对当前问题有 `n=6, m=3`，因此 `N_s` 的增加会线性抬升单周期代价。

## 3. 主要结论

### 3.1 offline 最优区和 live 最优区不一致

- 若只看 offline 平均位置误差，最优点集中在较低 `task_gain` 区域。
- 但这些点放到 live 后，并不能给出最好的 drift-free 恢复。
- 真正的 live 最优区明显转向：

$$
\text{高 } task\_gain + \text{高 } mu\_gain + \text{高 } N_s + \text{高 } solver\_gamma\_gain
$$

### 3.2 `mu_gain` 是 live 漂移恢复的主导参数

在固定

$$
task\_gain = 220,\quad
activation\_power = 0.85,\quad
N_s = 80,\quad
solver\_gamma\_gain = 4608
$$

时，live 结果呈现出很清楚的单调趋势：

- `mu = 12`: final drift `2.609686e-3 rad`
- `mu = 14`: final drift `2.171138e-3 rad`
- `mu = 16`: final drift `1.854458e-3 rad`
- `mu = 18`: final drift `1.615190e-3 rad`
- `mu = 20`: final drift `1.430160e-3 rad`

结论是：

- 在本次扫描区间内，`mu_gain` 提升会持续增强 drift-free 恢复。
- 代价是平均位置误差会有小幅上升，但增幅远小于漂移收益。

### 3.3 `activation_power = 0.85` 比 `0.90/0.95` 更适合 live

在 refined top30 中，`activation_power = 0.85` 占绝对优势，说明：

- 稍弱的幂次非线性可以减少 live 中的补偿过激
- 对漂移通道更友好
- 同时没有把位置误差推坏

### 3.4 `N_s = 80` 比 `N_s = 60` 更稳

本轮补跑直接比较了：

$$
(mu, N_s) = (14, 80) \quad \text{vs} \quad (14, 60)
$$

结果表明：

- `N_s = 80` 的 final drift 更低
- `N_s = 60` 并没有带来可观的位置误差收益
- 因此当前 live 推荐仍应保留 `N_s = 80`

## 4. 推荐保留配置

若以 drift-free 为主要目标，当前推荐：

$$
task\_gain = 220,\quad
mu\_gain = 20,\quad
activation\_power = 0.85,\quad
solver\_substeps = 80,\quad
solver\_gamma\_gain = 4608
$$

对应 live 结果：

- mean position error: `2.466074e-4 m`
- final position error: `1.043252e-4 m`
- final joint drift norm: `1.430160e-3 rad`

相对旧保留 live 配置：

- mean position error 改善约 `1.364x`
- final position error 改善约 `1.362x`
- final joint drift 改善约 `3.178x`

## 5. 若只追求最小平均位置误差

若只看本轮 live 扫描中的最小平均位置误差，则更优的是：

$$
task\_gain = 220,\quad
mu\_gain = 12,\quad
activation\_power = 0.85,\quad
solver\_substeps = 80,\quad
solver\_gamma\_gain = 4608
$$

但该点的 final drift 为

$$
2.609686\times10^{-3}\ \text{rad}
$$

明显弱于 `mu = 20` 的 drift-free 恢复，因此不建议作为最终保留点。

## 6. 文件位置

- 调参结果目录：
  - `C:\Users\lyx\Desktop\sci\code\project_method3c_lccznn_solver\results\tuning_20260421\method2`
- 当前推荐 live 点：
  - `C:\Users\lyx\Desktop\sci\code\project_method3c_lccznn_solver\results\tuning_20260421\method2\live_extra\tg220_mu20p0_ap0p85_ns80_gg4608\live`
