# Dynamic 3DGS SLAM Baseline 对比表

> 数据权威：`data/methods.json`。本文聚焦与本项目（RoGS-SLAM，单目序列）最相关的对比方法。

## 1. 单目动态 3DGS SLAM（直接竞品，最强相关）

| 方法 | Venue | 动态处理 | mask-free | 额外网络/先验 | 静态地图 | 备注 |
|---|---|---|---|---|---|---|
| **WildGS-SLAM** | CVPR 2025 | 不确定性加权 | ✅ | DINOv2 + uncertainty MLP + metric depth | ✅ | 我们的主要 baseline |
| **DGS-SLAM** | ICRA 2025 | 深度不确定 + 语义 mask | ❌ | 语义先验 | ✅ | 首个 3DGS 动态 SLAM 框架 |
| **Dy3DGS-SLAM** | ICRA 2025 | 光流+深度概率融合 mask | ❌ | 光流 | ✅ | 单目 RGB |
| **RoGS-SLAM (ours)** | MMM 2027 投稿 | 可靠性加权（几何） | ✅ | 无（frozen RAFT） | ✅ | 纯几何、无学习网络 |

## 2. RGB-D 动态 3DGS SLAM（传感器不同的参考）

| 方法 | Venue | 动态处理 | 备注 |
|---|---|---|---|
| DG-SLAM | NeurIPS 2024 | motion mask + adaptive Gaussians | hybrid pose optimization |
| GARAD-SLAM | ICRA 2025 | 高斯级分割 + 渲染惩罚 | 标题即 "Anti Dynamic" |
| Gassidy | ICRA 2025 | 鲁棒过滤 | — |
| BDGS-SLAM | Sensors 2025 | 贝叶斯概率更新 | — |
| PG-SLAM | TRO 2025 | motion mask | 照片级真实 |
| JPG-SLAM | ICRA 2025 | 点-高斯联合表示 | — |
| DAGS-SLAM | arXiv 2026 | 时空运动概率 + 按需语义 | 面向移动/边缘 |

## 3. Face-Dynamic（面向动态，概念对照）

| 方法 | Venue | 表示 | 说明 |
|---|---|---|---|
| D2GSLAM | arXiv 2025 | 静态3D + 动态4D 高斯 | 显式建模动态物体运动，跟踪可利用运动信息 |
| GS-DMSR | arXiv 2026 | 变形场 + 多尺度流形 | 离线动态重建（非 SLAM） |

## 4. 静态基座（动态方法的下限对比）

| 方法 | Venue | 传感器 | 备注 |
|---|---|---|---|
| MonoGS | CVPR 2024 | mono/stereo/RGB-D | 本项目基座 |
| SplaTAM | CVPR 2024 | RGB-D | silhouette 引导 |
| GS-SLAM | CVPR 2024 | RGB-D/mono | 自适应高斯扩展 |
| MonoGS++ | BMVC 2024 | mono | 快速单目 |
| Splat-SLAM | arXiv 2024 | mono | 全局优化 |

## 5. 对比建议（写论文用）

1. **主表**：单目设置下与 WildGS-SLAM、DGS-SLAM、Dy3DGS-SLAM 同列对比（若能复现）。Bonn/TUM 均有官方或社区结果。
2. **下限**：MonoGS（静态基座）必须列，体现动态处理增益。
3. **上限对照**：可提 face-dynamic（D2GSLAM）说明路线差异——我们不做动态物体重建，目标不同。
4. **注意传感器一致性**：RGB-D 方法（DG-SLAM 等）不能直接与单目方法比 ATE，需在叙述中显式区分。
