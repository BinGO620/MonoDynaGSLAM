# Dynamic 3DGS SLAM 分类法：Anti-Dynamic vs Face-Dynamic

> 本文件定义本项目（dynamic 3DGS SLAM，单目序列）使用的方法分类法。数据权威在 `data/categories.json`，本文件是它的分析性扩展。

## 1. 两分法的本质

动态 3DGS SLAM 的全部方法都面临同一个问题：**相机在位姿估计与建图时，如何处理场景中的运动物体。** 答案沿一条轴展开——**动态物体在系统中是"要被压制的干扰"还是"要被建模的对象"**：

| 维度 | Anti-Dynamic（抗动态） | Face-Dynamic（面向动态） |
|---|---|---|
| 对动态物体态度 | 干扰源，剔除/降权 | 建模对象，显式重建其运动 |
| 输出 | 静态地图 + 相机轨迹 | 时空场景（静态+动态可渲染） |
| 动态信息去向 | 丢弃 | 编码进表示（4D/变形场） |
| 典型开销 | 低（mask/加权为主） | 高（运动网络、4D 表示） |
| 评测侧重 | ATE / 静态重建质量 | 动态重建质量 + ATE |
| 是否依赖语义/检测 | 部分依赖，可不依赖 | 通常需要动态-静态分离先验 |

**判定标准（一句话）**：方法里是否有一个**显式的运动/变形表示**来描述动态物体？有 → face-dynamic；只是"把动态像素的重量调低/去掉" → anti-dynamic。

## 2. Anti-Dynamic 子类

按"如何识别动态"分组：

1. **不确定性加权**（不确定网络输出权重，融入 loss）：WildGS-SLAM（DINOv2+MLP 预测逐像素不确定性）
2. **语义 mask**（分割网络/检测器给出动态类别区域，硬/软 mask）：Gassidy、Dy3DGS-SLAM（光流+深度概率融合）、DAGS-SLAM（YOLO 按需触发）
3. **光流/几何一致性**（光流 vs 重投影流不一致 → 动态）：RoGS-SLAM（本项目）、DG-SLAM
4. **高斯级动态分割**（在 3D 表示上直接分割动态高斯）：GARAD-SLAM（Gaussian pyramid 网络）
5. **概率/贝叶斯滤波**（多视角概率更新）：BDGS-SLAM

## 3. Face-Dynamic 子类

1. **静态-动态复合表示**：静态 3D 高斯 + 动态 4D 高斯，统一优化。代表：D2GSLAM
2. **变形场/运动场**：对动态高斯学习 deformation field（离线动态重建常见，如 GS-DMSR；SLAM 内较少见）

## 4. 对单目序列的特殊性

单目设置下 anti-dynamic 的识别信号比 RGB-D **弱得多**（没有真深度来算残差/遮挡），所以单目方法多依赖：

- 光流（RAFT 等，单目可算）：WildGS-SLAM、RoGS-SLAM 均依赖光流类信号；
- 语义/自监督特征（DINOv2）：WildGS-SLAM；
- 单目深度先验（metric depth）：WildGS-SLAM 用 monocular metric depth 辅助位姿。

这是本领域单目 vs RGB-D 方法最重要的分水岭之一，写论文的 related work 时必须区分传感器设置来叙述。

## 5. 本项目的定位

RoGS-SLAM（monogs-ours）属于 **anti-dynamic + 光流/几何一致性** 子类，且刻意选择 **mask-free + 软降权** 路线：

- 与 WildGS-SLAM 的差异：不需要 DINOv2/uncertainty 网络，不需要单目深度先验，纯几何一致性信号；
- 与 mask 类方法的差异：不硬删除像素（floor 0.10 保底），保留部分动态区域的证据；
- 核心卖点：mask-free（无语义标注、无检测器、无额外网络），泛化到任意动态类别。

## 6. 反方观点记录

**codex（gpt-5.6-sol，2026-09-01）** 对本分类法的意见：

> 本质区别：anti-dynamic 将动态视为干扰并检测、剔除或降权，以维护静态地图；face-dynamic 将动态物体/运动作为建模对象，显式重建或估计其时空变化。
> **WildGS-SLAM 属于 anti-dynamic。**

与本文件判定一致。
