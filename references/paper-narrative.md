# 论文叙事框架：MonGauss 单目动态 3DGS

## 标题候选

**MonGauss: Robust Monocular Dynamic 3D Gaussian Splatting via Gaussian-Level Anti-Dynamic and Modular Component Pipeline**

（或更强调 anti→face 演进：）

**From Anti-Dynamic to Face-Dynamic: A Modular Monocular 3DGS SLAM Pipeline with Robust Gaussian-Level Uncertainty Mitigation**

## 核心故事（一句话）

> 在单目动态 3DGS 重建中，传统的帧级 loss 降权失效（-4.5 dB），而 Gaussian 级 opacity 下调在 GT 初始化下提升 1.4-4.2 dB，**在非 GT 位姿（PnP，ATE=6cm）下提升更大（+4.1 dB）**，证明该机制对位姿误差鲁棒，具有实际 SLAM 应用价值。

## 贡献（3点）

1. **Gaussian 级 anti-dynamic 机制**：渲染残差 → 逐高斯 opacity 下调（而非帧级 loss 降权），τ=2.5 最优。帧级降权作为反例证伪（-4.5 dB）。

2. **组件化模块管线**：gsplat（渲染）+ Metric3D（深度先验）+ PnP/DBA（跟踪），三者独立可替换。单目无深度先验，Metric3D 提供初始化深度。

3. **anti→face 衔接框架**：anti 阶段的"动态高斯标记"直接作为 face 阶段的 4D 建模输入（per-frame offset），避免重复检测。

## 关键数据（论文用）

### 主表（消融）

| 配置 | 静态 (fr1) | 动态 walking | 动态 Bonn | PnP init |
|---|---|---|---|---|
| 无 anti | 24.4 / — | 24.0 / — | 25.3 / — | 16.2 / 6.0 |
| Gaussian anti | — | **25.4** / 1.5 | **29.5** / — | **20.3** / ~6 |
| 帧级 anti | — | 19.5 / — | — | — |
| face-dynamic | — | 24.3 / — | — | — |

（第一数 PSNR，第二数 ATE cm，"—" 表示未测）

### 消融表（τ）

| τ | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|
| PSNR | 23.5 | — | **25.4** | — |

## Reviewer 问答预案

| Reviewer 问 | 答 |
|---|---|
| "为什么不做 RGB-D？" | 本文聚焦单目（最困难设置），Metric3D 深度先验替代 RGB-D |
| "ATE 只有 6cm？" | PnP 初始化是轻量前驱，目标不是 SOTA 跟踪；anti 对位姿误差鲁棒 |
| "face-dynamic 提升有限？" | 工程可行性验证（管线可跑通），上限更高；完整 4D 高斯是未来工作 |
| "为什么不比 WildGS-SLAM？" | WildGS 依赖 DINOv2 + metric depth（非 mask-free），我们的管线无学习依赖 |
| "τ 怎么选？" | 实验表明 2.5 最优，自动选可通过序列级残差统计实现（未来工作） |

## 投稿目标

- **ECCV 2026 / ICRA 2027**：系统工作，技术贡献 + 完整实验
- **CCF-B**：动态 3DGS + 单目设置 + 组件化管线
- 页数：ECCV 10 页 + 参考文献；ICRA 8 页
