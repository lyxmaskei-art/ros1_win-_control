# Reduced SPD Box QP 与 PCG 方向实验数据汇总

本文档汇总当前围绕“连续时变 QP、ELNCP 边界处理、Reduced SPD box QP、PCG 求解”这一方向已经完成的主要实验数据，并列出与 Method2 DLCCZNN、原始 Method3a PCG、PDNN、RK45、噪声鲁棒性和泛化验证相关的关键对照结果。

原始数据主要位于：

- `C:\Users\lyx\Desktop\sci\code\project_method3_tvqp_elncp_pcg_solver`
- `C:\Users\lyx\Desktop\sci\code\project_spd_box_qp_pcg_generalization_research`
- `C:\Users\lyx\Desktop\sci\code\experiments_parameter_tuning`
- `C:\Users\lyx\Desktop\sci\code\project_tvqp_elncp_pdnn_comparison`
- `C:\Users\lyx\Desktop\sci\code\project_tvqp_method2_rk45_solver`
- `C:\Users\lyx\Desktop\sci\code\experiments_noise_robustness`

## 1. 主线方法：Method3 TVQP ELNCP PCG

该方法对应目前的新方向：将 Method3 的 QP 结构整理为连续时变形式，引入 ELNCP 边界处理，并把最终子问题落到 reduced SPD box QP 上，用 PCG 求解。

### 1.1 与原始 Method3a PCG 的直接对比

原始文件：

`C:\Users\lyx\Desktop\sci\code\project_method3_tvqp_elncp_pcg_solver\results\comparison_with_method3a_pcg.csv`

| 方法 | 模式 | 平均位置误差 m | 最大位置误差 m | 末端位置误差 m | 最终关节漂移范数 rad | 平均迭代数 | 最大迭代数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始 Method3a PCG | offline 20 s | 2.043367116e-4 | 3.074120350e-4 | 6.079450557e-7 | 2.284136557e-5 | 4.44375 | 5 |
| Method3 TVQP ELNCP PCG | offline 20 s | 2.043366877e-4 | 3.074013272e-4 | 6.079975504e-7 | 4.667180394e-7 | 9.51525 | 13 |
| 原始 Method3a PCG best live | live 10 s | 1.302302364e-4 | 2.311864250e-4 | 5.814786418e-5 | 3.197148818e-4 | 4.6655 | 6 |
| Method3 TVQP ELNCP PCG | live 10 s | 1.302305246e-4 | 2.312150653e-4 | 5.815954593e-5 | 4.170262651e-4 | 10.7005 | 13 |

结论：

- offline 中，新 Method3 TVQP ELNCP PCG 的平均位置误差与原始 Method3a PCG 几乎一致，但最终关节漂移显著更小。
- live 中，新方法的位置误差与原始 Method3a PCG 基本一致，但最终关节漂移略大于原始 Method3a PCG best live。
- 新方法的代价是平均 PCG 迭代数更高。

### 1.2 泛化研究项目中的复现实验

原始文件：

`C:\Users\lyx\Desktop\sci\code\project_spd_box_qp_pcg_generalization_research\results\method3_robot_validation\offline\method3_tvqp_elncp_pcg_summary.json`

`C:\Users\lyx\Desktop\sci\code\project_spd_box_qp_pcg_generalization_research\results\method3_robot_validation_live10\live\method3_tvqp_elncp_pcg_summary.json`

| 项目 | offline 20 s | live 10 s |
|---|---:|---:|
| 平均位置误差 m | 2.043366877e-4 | 1.302309931e-4 |
| 最大位置误差 m | 3.074013272e-4 | 2.312155385e-4 |
| 末端位置误差 m | 6.079975504e-7 | 5.815280268e-5 |
| 最终关节漂移范数 rad | 4.667180394e-7 | 4.172800843e-4 |
| 平均 ELNCP 残差 | 1.909520669e-8 | 5.879662314e-9 |
| 最大 ELNCP 残差 | 1.481118205e-7 | 1.620490439e-7 |
| 平均 stationarity 残差 | 1.905586282e-8 | 5.832409643e-9 |
| 平均 NCP 残差 | 8.670080869e-11 | 8.762897057e-11 |
| 平均 PCG 迭代数 | 9.51525 | 10.6775 |
| 最大 PCG 迭代数 | 13 | 13 |
| 平均 active 迭代数 | 1.5305 | 1.8815 |
| 最大 active 迭代数 | 2 | 2 |
| 最小边界 slack | 1.859659846 | 1.565330994 |
| 运行时间 s | 1.8180 | 743.2074 |

核心判断：

- 该方法的 KKT 和 NCP 残差很小，说明 reduced SPD box QP 的数值求解是稳定的。
- live 误差受 CoppeliaSim 通信周期、状态反馈滞后和执行噪声影响明显，但平均位置误差仍在 0.13 mm 左右。

## 2. 与历史最优 Method2 DLCCZNN 和 Method3a PCG 的比较

原始文件：

`C:\Users\lyx\Desktop\sci\code\experiments_parameter_tuning\final_live_results\all_best_summaries.json`

| 方法 | 模式 | 平均位置误差 m | 最大位置误差 m | 末端位置误差 m | 最终关节漂移范数 rad |
|---|---:|---:|---:|---:|---:|
| Method2 DLCCZNN best | offline | 4.357016116e-6 | 1.644390956e-5 | 3.255219447e-6 | 1.042481462e-5 |
| Method2 DLCCZNN best | live | 2.015899554e-4 | 3.586699581e-4 | 9.176250837e-5 | 9.561883887e-4 |
| Method3a PCG best | offline | 7.944719042e-7 | 1.618604240e-6 | 8.825280281e-7 | 2.082318300e-6 |
| Method3a PCG best | live | 1.302564879e-4 | 2.312085097e-4 | 5.841846757e-5 | 4.169294907e-4 |

核心判断：

- 单看当前已有 live 数据，Method3a PCG 和 Method3 TVQP ELNCP PCG 的位置误差明显优于 Method2 DLCCZNN。
- Method2 DLCCZNN 的理论卖点仍然是离散神经动力学结构，而不是当前 live 精度第一。
- Method3a PCG 是目前工程精度最强的基线之一。

## 3. Method2 clip 与 ELNCP 边界处理的对比

原始文件：

`C:\Users\lyx\Desktop\sci\code\project_method2_dlccznn_elncp_bounds\results\comparison_clip_vs_elncp.csv`

| 方法 | 模式 | 平均位置误差 m | 最大位置误差 m | 末端位置误差 m | 最终关节漂移范数 rad | 平均内部残差 | 最小边界 slack |
|---|---:|---:|---:|---:|---:|---:|---:|
| Method2 baseline clip | offline | 4.357016116e-6 | 1.644390956e-5 | 3.255219447e-6 | 1.042481462e-5 |  |  |
| Method2 baseline clip | live | 2.015899554e-4 | 3.586699581e-4 | 9.176250837e-5 | 9.561883887e-4 |  |  |
| Method2 ELNCP | offline | 6.730135140e-4 | 9.905431916e-4 | 2.609508211e-4 | 1.449875115e-3 | 0.168533296 | 1.861201722 |
| Method2 ELNCP | live | 1.104477981e-3 | 1.631679319e-3 | 3.533453060e-4 | 2.443879743e-3 | 0.218444415 | 1.569848646 |

核心判断：

- 对 Method2 来说，直接把 clip 换成 ELNCP 降维处理后精度明显下降。
- 这说明 ELNCP 不是天然提高精度，它的优势是约束解释更正规、可导性更好、便于构造 KKT 残差，但参数和结构必须匹配。

## 4. 连续 TVQP ELNCP DLCCZNN 重构实验

原始文件：

`C:\Users\lyx\Desktop\sci\code\project_continuous_tvqp_elncp_dlccznn\results_corrected_tuned\corrected_all_result_summary.csv`

| 实验 | 方法 | 模式 | 平均位置误差 m | 末端位置误差 m | 最终关节漂移范数 rad | 平均任务残差 | 最终求解器残差 |
|---|---|---:|---:|---:|---:|---:|---:|
| m1_pos_best_live | Method1 | live | 6.851562796e-4 | 3.827236587e-4 | 2.998454753e-1 | 0.055934447 | 0.042001983 |
| m2_pos_best_live | Method2 | live | 2.550504071e-3 | 1.294790385e-3 | 3.008779619e-1 | 0.191211857 | 0.122442700 |
| m2_drift_best_live | Method2 | live | 3.420562007e-3 | 2.393984104e-3 | 8.704344851e-2 | 0.255001145 | 0.205774974 |
| m2_lam2_g50_kp80 | Method2 | live | 3.966620647e-3 | 2.379002450e-3 | 4.621737356e-2 | 0.294780410 | 0.208631386 |
| m1_balanced | Method1 | offline | 8.408545201e-5 | 3.804576947e-5 | 1.579339653e-2 | 0.010055311 | 0.005043989 |
| m1_pos_best | Method1 | offline | 8.446830893e-5 | 3.771087909e-5 | 1.147576637e-2 | 0.010101540 | 0.004988011 |
| m2_pos_best | Method2 | offline | 2.933961530e-4 | 2.549093847e-5 | 1.179736599e-2 | 0.023841567 | 0.007289740 |
| m2_balanced | Method2 | offline | 8.090818723e-4 | 4.413833988e-4 | 2.704994001e-3 | 0.064756771 | 0.036554754 |

核心判断：

- 连续 TVQP ELNCP DLCCZNN 重构目前没有超过单步冻结式 DLCCZNN 或 PCG 方法。
- 它的失败主要体现在 live 中关节漂移偏大，说明连续动态重构并不自动适配 CoppeliaSim 实时控制链路。

## 5. Method2 RK45 与 Euler 对比

原始文件：

`C:\Users\lyx\Desktop\sci\code\project_tvqp_method2_rk45_solver\results\final_gamma4608_rtol1e-5\comparison_euler_vs_rk45.csv`

| 方法 | 模式 | 平均位置误差 m | 最大位置误差 m | 末端位置误差 m | 最终关节漂移范数 rad |
|---|---:|---:|---:|---:|---:|
| Euler | offline | 1.186804268e-5 | 1.440255632e-4 | 8.240221535e-7 | 2.144357899e-6 |
| RK45 | offline | 8.910018836e-7 | 1.275539883e-6 | 4.234357900e-7 | 1.267751781e-6 |
| Euler | live | 2.426504124e-4 | 4.781052235e-4 | 1.043578160e-4 | 1.416303859e-3 |
| RK45 | live | 2.356950556e-4 | 4.179968066e-4 | 1.041701455e-4 | 1.429924985e-3 |

核心判断：

- RK45 在 offline 中显著改善精度。
- RK45 在 live 中只小幅改善位置误差，并略微增加最终关节漂移。
- 这说明 live 误差主要不由积分格式决定，而由真实控制循环、通信和反馈滞后决定。

## 6. PDNN 与 DLCCZNN 对照实验

原始文件：

`C:\Users\lyx\Desktop\sci\code\project_tvqp_elncp_pdnn_comparison\results\pdnn_vs_dlccznn_live_summary.csv`

### 6.1 live 对照

| 轨迹 | 方法 | 平均位置误差 m | 最大位置误差 m | 末端位置误差 m | 最终关节漂移范数 rad | 内部步数 |
|---|---|---:|---:|---:|---:|---:|
| figure8 | DLCCZNN | 1.141998702e-3 | 3.874453127e-3 | 1.450269679e-4 | 7.895743540e-3 | 1 |
| figure8 | PDNN | 8.370742965e-4 | 2.595932560e-3 | 1.448184784e-4 | 7.899096910e-3 | 5 |
| heart | DLCCZNN | 6.796256286e-3 | 2.755685058e-2 | 1.114743435e-3 | 1.406845366e-2 | 1 |
| heart | PDNN | 3.458669322e-3 | 8.682536171e-3 | 2.176573075e-4 | 8.224428821e-3 | 5 |
| line | DLCCZNN | 6.828574143e-4 | 1.231297302e-3 | 2.299757028e-4 | 8.044274306e-3 | 1 |
| line | PDNN | 4.889347060e-4 | 7.380736324e-4 | 1.480894577e-4 | 7.915382402e-3 | 5 |
| small circle | DLCCZNN | 6.053593852e-4 | 1.852801280e-3 | 1.461748033e-4 | 7.909103442e-3 | 1 |
| small circle | PDNN | 4.654707136e-4 | 1.246810333e-3 | 1.448659072e-4 | 7.891627952e-3 | 5 |

结论：

- 这一组 live 对照中，PDNN 5 步的精度普遍优于 DLCCZNN 1 步。
- 但是这组不是最终最优 DLCCZNN 调参结果，不能作为 DLCCZNN 的最终性能上限。

### 6.2 ns 提升后的 DLCCZNN live 调参

原始文件：

`C:\Users\lyx\Desktop\sci\code\project_tvqp_elncp_pdnn_comparison\results\coupled_param_search\stage6_live_final_t300_d30\stage6_live_final_t300_d30_summary.csv`

参数：

- task gain = 300
- drift gain = 30
- solver gamma = 3072
- activation power = 0.8
- DLCCZNN internal steps = 60

| 轨迹 | 平均位置误差 m | 最大位置误差 m | 末端位置误差 m | 最终关节漂移范数 rad |
|---|---:|---:|---:|---:|
| figure8 | 9.836827552e-5 | 3.617448121e-4 | 7.705021249e-5 | 5.799531348e-4 |
| line | 7.807569663e-5 | 1.060426099e-4 | 7.646491390e-5 | 5.813398369e-4 |
| circle | 1.024894879e-4 | 2.453397968e-4 | 7.680007306e-5 | 5.813693568e-4 |
| small circle | 8.496805358e-5 | 1.463850275e-4 | 7.795099710e-5 | 5.817870624e-4 |
| ellipse | 1.089074342e-4 | 2.655042584e-4 | 7.796350967e-5 | 5.821505184e-4 |

结论：

- 提高 ns 后，DLCCZNN live 的平均位置误差可进入 0.08 mm 到 0.11 mm 区间。
- 但最终关节漂移约为 5.8e-4 rad，仍略高于 Method3a PCG 的 4.17e-4 rad。

## 7. 简单轨迹 offline 筛选

原始文件：

`C:\Users\lyx\Desktop\sci\code\project_tvqp_elncp_pdnn_comparison\results\formal_offline\shape_screening\shape_screening_summary.csv`

| 轨迹 | 方法 | 平均位置误差 m | 最大位置误差 m | 末端位置误差 m | 最终关节漂移范数 rad |
|---|---|---:|---:|---:|---:|
| small circle | DLCCZNN | 2.496004593e-4 | 8.751833261e-4 | 4.218741914e-7 | 2.238116667e-6 |
| line | DLCCZNN | 2.599948403e-4 | 5.243257182e-4 | 3.048650046e-5 | 1.560110765e-4 |
| figure8 | DLCCZNN | 4.733431190e-4 | 1.741540804e-3 | 1.226456003e-6 | 6.861215183e-6 |
| circle | DLCCZNN | 5.480463875e-4 | 2.146313188e-3 | 2.439911530e-6 | 1.150637824e-5 |
| ellipse | DLCCZNN | 6.162828198e-4 | 2.169466153e-3 | 1.134560591e-6 | 6.680889194e-6 |

结论：

- offline 中，小圆的最终角度漂移最好。
- line 的平均位置误差接近小圆，但最终角度漂移明显更大。

## 8. 噪声鲁棒性数据

原始文件：

`C:\Users\lyx\Desktop\sci\code\experiments_noise_robustness\noise_robustness_summary.csv`

重要说明：

这组噪声实验主要是早期 Method2 与 Method3a 的鲁棒性测试，不是新的 Method3 TVQP ELNCP PCG 的直接噪声实验。

### 8.1 actuator noise

| 方法 | 噪声强度 | 平均位置误差均值 m | 最终角度漂移均值 rad |
|---|---:|---:|---:|
| Method2 | 1e-5 | 4.067673636e-6 | 3.889993139e-6 |
| Method3a | 1e-5 | 8.032026182e-7 | 2.255308921e-6 |
| Method2 | 1e-4 | 4.081000001e-6 | 4.010378961e-6 |
| Method3a | 1e-4 | 8.917221027e-7 | 2.923394823e-6 |
| Method2 | 1e-3 | 5.135999006e-6 | 1.460690006e-5 |
| Method3a | 1e-3 | 3.385851563e-6 | 1.596815189e-5 |
| Method2 | 1e-2 | 2.776323190e-5 | 1.136680899e-4 |
| Method3a | 1e-2 | 3.044467757e-5 | 1.233701893e-4 |

结论：

- actuator noise 较小时，Method3a 精度更好。
- actuator noise 较大时，两者都会退化，Method2 与 Method3a 的差距缩小。

### 8.2 sensor noise

| 方法 | 噪声强度 | 平均位置误差均值 m | 最终角度漂移均值 rad |
|---|---:|---:|---:|
| Method2 | 1e-6 | 4.128500933e-6 | 4.090655842e-6 |
| Method3a | 1e-6 | 1.434225163e-6 | 3.782042935e-6 |
| Method2 | 1e-5 | 7.536834130e-6 | 2.659914925e-5 |
| Method3a | 1e-5 | 8.509755820e-6 | 2.946267632e-5 |
| Method2 | 1e-4 | 5.869364750e-5 | 1.940737427e-4 |
| Method3a | 1e-4 | 7.434873327e-5 | 2.745323127e-4 |
| Method2 | 5e-4 | 2.935128059e-4 | 7.131524270e-4 |
| Method3a | 5e-4 | 6.902536007e-1 | 4.629194874 |
| Method2 | 1e-3 | 1.534691680e-1 | 7.444966837e-1 |
| Method3a | 1e-3 | 6.994137435e-1 | 4.038227537 |

结论：

- sensor noise 中等时，Method2 的退化更慢。
- Method3a 在 5e-4 sensor noise 下出现明显崩溃。
- 这提示 Method3a PCG 虽然精度强，但对某些传感噪声可能更敏感。

## 9. 泛化边界与反例实验

原始文件：

`C:\Users\lyx\Desktop\sci\code\project_spd_box_qp_pcg_generalization_research\results\synthetic_qp_generalization_summary.csv`

| 测试项 | 是否可精确转化 | PCG 解误差 | 约束违背 | KKT 违背 | 结论 |
|---|---:|---:|---:|---:|---|
| 原生 SPD box QP | True | 3.608224830e-16 | 0 | 2.795719212e-11 | 支持 |
| 硬等式 QP 用 penalty 变成 SPD | False | 4.242428566e-5 | 5.999700015e-5 |  | 反驳精确等价 |
| 一般耦合线性不等式近似为 box | False | 7.071067812e-1 | 1.0 |  | 反驳一般耦合不等式 |
| 非凸 indefinite QP 变成 SPD | False | 1.0 | 0 |  | 反驳非凸 QP |

结论：

- “所有 QP 都能无损转化为 SPD box QP 再用 PCG 解”这个强命题不成立。
- 比较稳妥的命题是：一类机器人重复运动学中的特定漂移抑制 QP，如果能被结构化为 SPD box QP 或 reduced SPD box QP，则 PCG 具有精度、速度和工程实现优势。

## 10. 当前总判断

### 10.1 当前数据最强的方法

从已有数据看：

- offline 最强：Method3a PCG best，平均位置误差 7.944719042e-7 m，最终关节漂移 2.082318300e-6 rad。
- live 位置误差最强之一：Method3a PCG best 与 Method3 TVQP ELNCP PCG，平均位置误差约 1.302e-4 m。
- live 简单轨迹 DLCCZNN ns60 可达到 7.8e-5 m 到 1.09e-4 m 的平均位置误差，但最终关节漂移约 5.8e-4 rad。
- 抗 sensor noise 的趋势上，早期数据里 Method2 比 Method3a 更稳，但新 Method3 TVQP ELNCP PCG 还缺少直接噪声测试。

### 10.2 对论文叙事最有用的数据

如果以“工程有效性”为主线：

- Method3a PCG 和 Method3 TVQP ELNCP PCG 是最强数据支撑。
- Reduced SPD box QP 是更容易讲清楚结构优势的理论对象。
- PCG 是当前最有工程竞争力的求解器。
- DLCCZNN 可以作为离散神经动力学对照和理论延伸，但不能硬说当前所有数据都优于 PCG。

如果以“DLCCZNN”为主线：

- 需要强调离散控制结构、低复杂度、采样周期适配和神经动力学意义。
- 不能只靠现有 live 精度压过 PCG。
- 可以用 PDNN 对比来说明同样实时预算下，PDNN 对调参与内部步数更敏感，但目前必须谨慎，因为某些 PDNN 5 步数据确实优于 DLCCZNN 1 步。

### 10.3 仍缺的数据

- Method3 TVQP ELNCP PCG 的直接噪声鲁棒性实验。
- Method3 TVQP ELNCP PCG 在多轨迹上的系统 live 对比。
- 真实 UR3e 物理实验。
- 与 OSQP、active set QP、qpOASES、CasADi 类求解器的工程时间对比。
- 多个随机初始关节角下的统计实验。

