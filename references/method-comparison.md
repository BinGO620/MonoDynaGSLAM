# Dynamic 3DGS SLAM 方法对比报告（WildGS-SLAM vs RoGS-SLAM vs 参照系）

> 生成：2026-09-01 · 工具：3dgs-method-compare（core-stance / methods-slam / methods-dynamic / output-rules）
> 数据源：本仓库 `data/methods.json`（论文摘要 + 源码交叉核实）；skill 知识库补充 WildGS-SLAM 定位。

## 核心判定（前置结论）

**WildGS-SLAM = anti-dynamic（抗动态）**。Skill 知识库 SLAM 片段对其描述为
"Dynamic environments, uncertainty-aware mapping via pretrained 3D priors"，
与本仓库三源核实一致：动态物体被移除/降权，只重建静态地图，无显式动态运动表示。

---

## Overview Table

| 维度 | WildGS-SLAM (CVPR'25) | RoGS-SLAM (ours, MMM'27 投稿) | DGS-SLAM (ICRA'25) | DG-SLAM (NeurIPS'24) | GARAD-SLAM (ICRA'25) | D2GSLAM (arXiv'25) |
|---|---|---|---|---|---|---|
| 分类 | **anti-dynamic** | **anti-dynamic** | anti-dynamic | anti-dynamic | anti-dynamic | **face-dynamic** |
| 传感器 | 单目 | RGB-D（单目扩展中） | 单目/RGB-D | RGB-D | 单目/RGB-D | 单目/RGB-D |
| 动态识别信号 | DINOv2 + uncertainty MLP（学习） | 光流-重投影一致性 + 深度残差异常（几何） | 深度不确定 + 语义 mask | motion mask + adaptive Gaussians | 高斯级分割网络 | 几何提示动态分离 |
| 对动态区域处理 | 不确定性加权（软降权） | Cauchy 软降权（floor 0.10，不硬删） | mask 移除 | mask 移除 | 动态高斯渲染惩罚 | 静态3D + 动态4D 复合建模 |
| mask-free？ | ✅（无语义标签） | ✅（无标签/检测器/网络） | ❌ | ❌ | ✅（网络直接分割高斯） | —（几何提示分离） |
| 额外依赖 | DINOv2 + metric depth 先验 | 无（frozen RAFT） | 语义先验 | 光流 | Gaussian pyramid 网络 | 无 |
| 动态物体是否建模 | ❌ 移除 | ❌ 不建模（仅降权） | ❌ 移除 | ❌ 移除 | ❌ 移除 | ✅ 4D 高斯显式建模 |
| 静态地图输出 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅（+动态） |
| 动态地图输出 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 单目深度依赖 | 依赖 metric depth | 不依赖 | — | — | — | — |
| 计算开销档位 | 中（DINOv2 推理） | 低（RAFT 一次前向） | 中 | 中 | 中 | 高（4D 表示） |
| 代码可用 | ✅ 开源 | ✅ 开源（ours） | 未开源 | 未开源 | 未开源 | 未开源 |

> 注：DGS-SLAM / DG-SLAM / GARAD-SLAM / D2GSLAM 的数值细节若未在知识库中出现，一律标 "数据不可得"，不做猜测。

---

## Detailed Analysis

### 1. 动态识别信号：学习特征 vs 几何一致性

- **WildGS-SLAM** 走"学习不确定性"路线：DINOv2 提供 3D 感知特征，shallow MLP 预测逐像素不确定性。优点：特征泛化到新场景/新动态类别（论文强调）；代价：需要 DINOv2 前向 + MLP 在线训练 + 单目 metric depth 先验。
- **RoGS-SLAM** 走"纯几何一致性"路线：光流与刚性重投影流的不一致 + 深度残差异常，合成可靠性信号 `s=(1-e_flow)(1-v·g)`。优点：零学习网络、零语义/深度先验、mask-free；代价：依赖光流质量，纹理贫乏或光照突变场景信号变弱。
- **设计权衡**：学习路线泛化上限高但依赖重；几何路线轻量普适但对信号质量敏感。两者都避免硬 mask，属于"软降权"族——这是与 DG-SLAM（mask 移除）的路线分界。

### 2. 动态区域处理：软降权 vs 硬移除 vs 显式建模

- **软降权**（WildGS-SLAM 不确定性加权 / RoGS-SLAM Cauchy 降权）：保留部分动态证据，不引入硬 mask 边界误差，对 mask 精度不敏感；
- **硬移除**（DG-SLAM / DGS-SLAM）：mask 误差会直接造成静态图缺损与位姿退化，对 mask 精度敏感；
- **显式建模**（D2GSLAM）：动态物体进入 4D 高斯，静态/动态统一优化，可输出时空场景——本质差异是"目标不同"，不是"更好的 anti-dynamic"。

### 3. 单目设置的特殊性（本项目主线）

- 单目下无真深度，深度残差信号不可用 → RoGS-SLAM 若转单目需依赖光流一致性（已有），但深度门控项 v·g 需要替代；
- WildGS-SLAM 用 monocular metric depth 补偿单目尺度歧义；
- 单目对比表（主表）应只列单目方法：WildGS-SLAM / DGS-SLAM / Dy3DGS-SLAM / RoGS-SLAM（单目版）。

### 4. 推荐与叙事建议

- **主 baseline**：WildGS-SLAM——同属"anti-dynamic + 软加权 + mask-free"，差异点干净（学习 vs 几何；依赖 vs 无依赖），是最佳对比对象；
- **related work 叙事**：把 WildGS-SLAM 归"不确定性加权"子类，DGS-SLAM/Dy3DGS-SLAM 归"mask 类"，DG-SLAM 归"motion mask + 高斯管理"，GARAD-SLAM 归"高斯级分割"，D2GSLAM 归"face-dynamic 路线对照"；
- **立场句**（可放 related work 末段）："Unlike uncertainty-learning approaches that rely on pre-trained priors and metric depth (e.g. WildGS-SLAM), we derive reliability purely from geometric consistency, requiring no semantic labels, no detectors, and no additional networks."

### 5. 反方/不确定项

- [UNCERTAIN] DG-SLAM 是否开源、DGS-SLAM 是否单目可跑：以官方 repo 为准，标"待核实"；
- [UNCERTAIN] WildGS-SLAM 的 DBA 权重形式（softmax 归一化 vs 直接乘）细节：需读论文 §3 确认；
- 上述 [UNCERTAIN] 项不影响 anti/face 分类判定（判据是"有无显式运动表示"，已由摘要+源码确认）。
