# WildGS-SLAM 深度分析（baseline）

> 数据源：`data/methods.json` → `wildgs-slam`。本文件回答核心问题：**WildGS-SLAM 是 anti-dynamic 还是 face-dynamic？——答案是 anti-dynamic。**

## 1. 一句话结论（三源交叉验证）

**WildGS-SLAM（CVPR 2025, arXiv:2504.03886）是 anti-dynamic（抗动态）方法。**

三个独立来源一致：

1. **论文摘要原话**（arXiv/CVPR page）：
   > "WildGS-SLAM accurately tracks the camera trajectory and reconstructs a 3D Gaussian map **for static elements, effectively removing all dynamic components**."
2. **方法机制**：uncertainty map（DINOv2 + shallow MLP）→ "guide **dynamic object removal** during both tracking and mapping"；"weighing error terms to **minimize the impact of moving objects**"。
3. **codex 独立意见**（2026-09-01）：
   > "anti-dynamic 将动态视为干扰并检测、剔除或降权，以维护静态地图……**WildGS-SLAM 属于 anti-dynamic（抗动态）方法**。"

## 2. 系统管线（按论文 + 本地源码）

```
RGB 序列（单目）
   │
   ├─ DINOv2 ────────────────► 3D 感知特征
   │                                │
   │                                ▼
   │                        uncertainty MLP ──► 逐像素不确定性 U
   │
   ├─ 前端 tracking：
   │   dense bundle adjustment (DBA)，以 U 作为误差权重
   │   + monocular metric depth 辅助位姿估计
   │
   └─ 后端 mapping：
       渲染 RGB/depth 与观测比较，loss 按 U 降权（不确定性加权 loss）
       → 增量构建静态场景 3D 高斯地图
```

本地源码佐证（`/data/WildGS-SLAM`）：

- `src/utils/dyn_uncertainty/uncertainty_model.py`：`MLPNetwork`（DINOv2 384 维特征 → 1 维不确定性，softplus 输出，He uniform 初始化，dropout 0.2）——与论文 "shallow MLP" 一致；
- `src/utils/dyn_uncertainty/mapping_utils.py`：mapping loss 中按 `uncertainty` 加权 RGB L1 / SSIM / depth 项（`rgb_l1_loss = |rendered_img * mask - gt_img * mask|`，depth mask 0.01~threshold），即 **动态像素的梯度贡献被不确定性压低**；
- 输出为**静态地图**：动态高斯不进入最终重建。

## 3. 为什么是 anti-dynamic 而不是 face-dynamic

| 判据 | WildGS-SLAM | face-dynamic 典型（如 D2GSLAM） |
|---|---|---|
| 动态物体有显式运动/变形表示？ | ❌ 无（只有不确定性标量） | ✅ 4D 高斯 / 变形场 |
| 动态物体的运动被重建/渲染？ | ❌ 被移除 | ✅ 显式时序建模 |
| 输出内容 | 静态地图 + 轨迹 | 时空场景 |
| 动态信息用于什么 | 仅用于降低其对位姿/建图的干扰 | 既是干扰抑制信号也是建模目标 |

WildGS-SLAM 的 uncertainty **不是对动态物体运动的描述**，而是一个"这片区域别信"的标量权重——这正是 anti-dynamic 的定义。

## 4. 对本项目（RoGS-SLAM）的对比意义

| 维度 | WildGS-SLAM (baseline) | RoGS-SLAM (ours) |
|---|---|---|
| 动态识别信号 | DINOv2 特征 + 学习的不确定性 MLP | 光流-重投影一致性 + 深度残差异常（纯几何） |
| 额外网络/先验 | DINOv2 + uncertainty MLP + monocular metric depth | 无（frozen RAFT 光流即可） |
| 对动态区域的处理 | 不确定性加权（可视为软降权） | Cauchy 软降权（floor 0.10） |
| mask-free？ | 是（无语义标签） | 是（无语义标签、无检测器） |
| 动态分离粒度 | 逐像素（特征级） | 逐像素（几何级） |
| 传感器 | 单目（与其配准一致） | 目前 RGB-D 为主，单目扩展中 |
| 单目深度依赖 | 依赖 metric depth 先验 | 不依赖 |

**论文叙事建议**：两家同属"抗动态 + 软加权"路线，但信号来源（学习特征 vs 几何一致性）与依赖（深度先验 vs 无）不同。related work 中应把 WildGS-SLAM 作为"不确定性加权"子类的代表，突出 RoGS-SLAM 的 mask-free + 无额外网络 + 纯几何定位。

## 5. 数据集

WildGS-SLAM 官方在 Wild-SLAM（自建）、Bonn、TUM、Replica、ScanNet 上评测。RoGS-SLAM 目前跑 Bonn/TUM/Replica——**评测面基本对齐**，可做同表对比（注意单目 vs RGB-D 设置差异）。

## 6. 事实核查与引用

- arXiv: [2504.03886](https://arxiv.org/abs/2504.03886)（CVPR 2025 Poster）
- Project: https://wildgs-slam.github.io/
- Code: https://github.com/GradientSpaces/WildGS-SLAM（本地 `/data/WildGS-SLAM`，29 commits）
- 作者：Zheng*, Zhu* (equal), Bieri, Pollefeys, Peng, Armeni（Stanford / ETH Zürich / Microsoft）
