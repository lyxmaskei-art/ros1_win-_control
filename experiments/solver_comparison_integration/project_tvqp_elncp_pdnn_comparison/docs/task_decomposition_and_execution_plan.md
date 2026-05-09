---
title: TVQP + ELNCP + PDNN 对照实验任务拆解
created: 2026-05-07
project: project_tvqp_elncp_pdnn_comparison
---

# TVQP + ELNCP + PDNN 对照实验任务拆解

## 1. 任务目标

本项目只在以下目录中执行：

`C:\Users\lyx\Desktop\sci\code\project_tvqp_elncp_pdnn_comparison`

目标是以现有的 Method 2 TVQP + ELNCP + DLCCZNN 为对照对象，新建一套同问题、同轨迹、同约束、同欧拉离散方式下的 PDNN 求解实验。实验不是重新设计上层控制问题，而是只替换底层神经动力学求解器。

## 2. 保持不变的部分

1. 机器人对象保持为 UR3e 位置级运动学模型。
2. 跟踪任务仍采用末端三维位置轨迹。
3. Method 2 仍采用标量能量型误差处理逻辑。
4. 速度边界仍采用动态边界。
5. 不等式约束仍采用 ELNCP 残差进入 KKT 系统。
6. 关节最终位置更新仍采用欧拉离散：

$$
q_{k+1}=q_k+\tau \dot q_k .
$$

7. 结果图保持旧实验格式：轨迹图、位置误差图、关节漂移误差图、关节角图、任务残差图、求解器残差图和关节恢复表。

## 3. 新增的 PDNN 求解器

记 ELNCP 后的残差为

$$
F(y,t)=
\begin{bmatrix}
W u+\mu(q-q_0)-J(q)^{T}\lambda+\sigma \omega\\
J(q)u-b(t,q)\\
\psi_{\varepsilon}(s(u),\omega)
\end{bmatrix},
$$

其中

$$
y=
\begin{bmatrix}
u\\
\lambda\\
\omega
\end{bmatrix}.
$$

DLCCZNN 对照组继续沿用原脚本中的标量能量下降式。PDNN 组采用残差能量梯度流：

$$
\dot y=-\rho F_y(y,t)^T F(y,t),
$$

并用同款显式欧拉离散：

$$
y_{k+1}=y_k-\tau \rho F_y(y_k,t_k)^T F(y_k,t_k).
$$

为测试连续模型数值积分精度，PDNN 支持每个控制周期内的 `pdnn_inner_steps` 个欧拉小步：

$$
h_p=\frac{\tau}{N_p},\qquad
y^{r+1}=y^r-h_p\rho F_y(y^r,t_k)^TF(y^r,t_k).
$$

当 `pdnn_inner_steps=1` 时，就是与控制周期同频的一步欧拉离散 PDNN。

## 4. 实验指标

每个实验至少记录以下指标：

1. 平均位置误差：

$$
\bar e_p=\frac{1}{N}\sum_{k=0}^{N-1}\lVert x(q_k)-x_d(t_k)\rVert_2 .
$$

2. 末端最终位置误差：

$$
e_p(T)=\lVert x(q_N)-x_d(T)\rVert_2 .
$$

3. 最终关节漂移范数：

$$
e_q(T)=\lVert q_N-q_0\rVert_2 .
$$

4. 最大关节漂移绝对值。
5. 平均任务残差与最终任务残差。
6. 平均求解器残差与最终求解器残差。
7. 单次实验总运行时间。
8. 单步平均计算时间：

$$
t_{\mathrm{step}}=\frac{t_{\mathrm{runtime}}}{N}.
$$

## 5. 实验矩阵

### 5.1 PDNN 连续模型精度测试

目的：判断 PDNN 在连续模型小步欧拉积分下是否可以把同一 TVQP + ELNCP 问题求到可接受精度。

配置：

1. 轨迹：heart。
2. 模式：offline。
3. 时长：20 s。
4. 采样周期：0.005 s。
5. PDNN 子步：`1, 5, 20, 50`。
6. PDNN 增益：先用稳定增益，再根据 smoke test 微调。

输出目录：

`results\pdnn_continuous_accuracy`

### 5.2 四组 PDNN 与 DLCCZNN 对照实验

第一组：同实时预算。

PDNN 使用 `pdnn_inner_steps=1`，DLCCZNN 使用原 Method 2 设置。比较在相同控制周期内谁能得到更低误差。

第二组：同精度所需计算成本。

扫描 PDNN 子步数，观察 PDNN 需要多少欧拉小步才能接近 DLCCZNN 的误差。

第三组：采样周期鲁棒性。

采样周期取：

$$
\tau\in\{0.002,0.005,0.01,0.02\}.
$$

比较两种求解器对离散周期放大的敏感性。

第四组：计算复杂度与运行时间趋势。

在相同轨迹和相同时长下，记录 PDNN 子步数变化时的运行时间和误差变化，并与 DLCCZNN 单步更新对照。

基础输出目录：

`results\comparison_heart`

### 5.3 简单图形筛选

目的：把 Method 2 中的 heart 轨迹替换为更平滑、更简单的轨迹，先只跑 offline，筛选精度最高的三个图形。

候选图形：

1. circle。
2. line。
3. ellipse。
4. figure8。
5. small_circle。

筛选指标：

先按平均位置误差排序，再按最终关节漂移范数排序。选出前三个。

输出目录：

`results\shape_screening`

### 5.4 三个优选图形上的重复对照

对 5.3 选出的前三个图形，重复 5.2 中四组对照实验。

输出目录：

`results\comparison_top_shapes`

## 6. Live 实验执行策略

Live 实验会连接 CoppeliaSim。为了避免实验矩阵过大，执行顺序为：

1. 先在 heart 上跑同实时预算 live 对照。
2. 再对三个优选图形分别跑代表性 live 对照。
3. 若连接失败或耗时过长，保留 offline 完整矩阵，并在最终报告中明确标记未完成的 live 项。

## 7. 验证检查点

### 7.1 代码检查

运行：

```powershell
python -m py_compile .\scripts\run_offline.py .\scripts\run_live.py
```

### 7.2 Smoke test

运行 0.2 s 或 0.5 s offline，验证：

1. 程序不崩溃。
2. summary json 生成。
3. 至少生成轨迹图、位置误差图和关节漂移图。
4. 残差和误差为有限数值。

### 7.3 正式 offline

运行完整 20 s offline 矩阵，所有结果保存到 `results`。

### 7.4 Live

连接 CoppeliaSim 后运行 10 s live 代表性实验，若可行再扩展至优选图形。

## 8. 主要风险

1. PDNN 梯度流对增益和欧拉步长敏感，过大可能发散，过小可能收敛慢。
2. PDNN 若必须依赖大量内部欧拉小步才能接近 DLCCZNN，反而可以作为凸显 DLCCZNN 离散结构优势的证据。
3. Live 实验受 CoppeliaSim 通信周期和仿真状态影响，可能与 offline 有明显差距。
4. 简单图形更平滑，但不保证漂移一定更低，因为漂移还受闭合路径、关节冗余分配和反馈项共同影响。

## 9. 最终交付物

1. 集成代码：

`scripts\run_offline.py` ? `scripts\run_live.py`

2. 实验图与 summary：

`results\...`

3. 公式推导与数据分析报告：

`docs\pdnn_vs_dlccznn_experiment_report.md`

4. 总表：

`results\pdnn_vs_dlccznn_all_summaries.csv`

