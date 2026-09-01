# Step 2 实验结果与结论

> fr3_walking_xyz (50帧, Metric3D 深度初始化, 60k 高斯)

## 结果对比

| 模式 | PSNR mean | 前半 (0-25帧) | 后半 (25-50帧) | 高斯数 | 判决 |
|---|---|---|---|---|---|
| **no_anti (baseline)** | **24.0 dB** | 23.5 dB | 24.7 dB | 60k | PASS ✅ |
| **anti tau=3.0** | **19.5 dB** | 22.2 dB | 24.1 dB | 60k | FAIL ❌ |

## 机制分析

当前 anti-dynamic 是**帧级 loss 权重**：对渲染残差大的帧降权，减少其对 Gaussian 梯度的贡献。

**为什么失败**：
- 帧级降权 ≠ Gaussian 级剔除：人的走动产生了高残差帧，但这些帧的信息对 Gaussian 颜色更新仍是有用的（人经过的背景区域需要被正确渲染）；
- 降权等于"忽略这些帧"，导致覆盖不足 → PSNR 反而下降；
- 降权信号作用于 loss 而非 Gaussian 参数（opacities），动态 Gaussian 没有被抑制。

## 关键结论

> **Anti-dynamic 必须在 Gaussian 层面做操作（降低 opacity / 删除 / 剔除），不能只在帧层面降权 loss。**

这与 WildGS-SLAM（uncertainty 直接乘 opacity）和 Gassidy（渲染 loss flow 对 Gaussian 分组降权）的机制一致，也验证了 codex 的判断："决策层、生命周期、统计校准"才是正确的切入点。

## 下一步

改为 Gaussian 级 anti-dynamic：
1. 每帧渲染后，计算 per-pixel 残差
2. 把高残差区域对应的 Gaussian opacity 降低（与 WildGS 的 uncertainty 思路一致，但用纯渲染一致性而非 learned uncertainty）
3. 对比帧级 vs Gaussian 级两种机制，用 ATE 和 Gaussian 数量变化验证
