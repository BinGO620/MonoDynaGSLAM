# 组件化 3DGS SLAM 架构 — 从独立组件到连接层

> 关键洞察（用户提供 + 调研验证）：3DGS 渲染和 SLAM 是**独立发展**的两个领域，已有大量成熟独立组件。我们的研究空间在于**把它们连接起来**，而不是从零重写或在某个单体系统上改。

## 1. 可复用的独立组件（不绑定 SLAM）

### 3DGS 渲染后端（轴 A：效率）

| 组件 | 状态 | 说明 |
|---|---|---|
| **gsplat**（nerfstudio-project） | 成熟（v1.5.3+），Apache-2.0 | 独立的 CUDA 加速可微 3DGS 渲染库；批量光栅化、N-D 特征渲染、深度渲染、MCMC 致密化、位姿梯度；比官方 diff-gaussian-rasterization 快 2-4× 省显存；PyTorch 接口 |
| **Scaffold-GS** | 成熟 | 锚点紧凑表示（我们轴 A 的效率思想来源） |
| **SOG / FLAS** | MMM'27 教程 | 自组织高斯 + 排序压缩（离线静态） |

### SLAM 跟踪前端（轴 C：BA）

| 组件 | 状态 | 说明 |
|---|---|---|
| **DROID-SLAM DBA layer** | 成熟，NeurIPS 2021 | 可微稠密 BA 层；位姿+逆深度迭代优化；mono/stereo/RGB-D 通用；WildGS-SLAM 已复用它 |
| **DPVO** | 成熟 | 补丁级 DROID-SLAM，30+ FPS 实时 |
| **Lietorch** | 成熟 | SE(3)/SO(3) 李群运算库 |
| **Theseus**（Meta） | 成熟 | 可微非线性优化库 |

### 动态处理组件（轴 B：anti↔face）

| 组件 | 状态 | 说明 |
|---|---|---|
| RAFT（光流） | 成熟 | 光流提取（anti 信号来源） |
| DINOv2（特征） | 成熟 | 语义特征（WildGS-SLAM 路线） |
| 4D 高斯 / 变形场（D2GSLAM 等） | 研究代码 | face-dynamic 表示 |

## 2. 组件化架构设计（我们的连接层）

```
输入 RGB 序列
   │
   ├─[组件复用] DROID-SLAM DBA ──► 位姿 + 逆深度（跟踪/轴C-BA）
   │        │
   │        ▼
   ├─[组件复用] gsplat 渲染 ────► 可微光栅化（建图/轴C-渲染）
   │        │
   │        ▼
   ├─[我们的连接层] 动态状态机 ──► anti↔face 分治（轴B）
   │        静态/瞬态静止/运动/未知 状态估计
   │
   └─[我们的连接层] 地图管理 ───► 紧凑静态 + 4D 动态（轴A）
            anchor 紧凑表示（静态）
            4D 高斯/变形场（动态）
```

**分工明确**：
- 复用：DROID-DBA（跟踪）、gsplat（渲染）、RAFT（光流）——都是成熟独立组件
- 自研（连接层 = 研究贡献）：
  1. **动态状态机**：连接跟踪与渲染的动态策略分治
  2. **地图管理**：静态紧凑 + 动态 4D 的混合表示与生命周期
  3. **BA-渲染解耦**：BA 用 anti（静态一致性），渲染用 face（视觉完整）

## 3. 为什么这是更好的工程起点

| 方案 | 优点 | 缺点 |
|---|---|---|
| 在 monogs-ours 上改 | 已有资产 | 单体耦合；RGB-D 为主；历史包袱重；代码与 MMM 投稿绑定 |
| 在 MonoGS 上改 | 经典基线 | 同上 |
| **组件化新写（推荐）** | 用最新 gsplat（快 2-4×）；模块边界清晰；可控性强；研究贡献聚焦"连接层" | 需要重新集成（但 gsplat 提供标准 API） |

## 4. 演进路线：anti-dynamic → face-dynamic（用户方向）

```
阶段1: anti-dynamic 基线（连接 DROID-DBA + gsplat）
  ├─ BA 用 anti 权重（光流-重投影不一致 → 降权）
  └─ 渲染只建静态图
  → 验证: 单目 ATE 达标, 静态 PSNR 达标

阶段2: face-dynamic 扩展（+4D 动态表示）
  ├─ 动态区域检测 → 4D 高斯/变形场
  ├─ 静态用紧凑 anchor, 动态用全参数
  └─ BA 仍用 anti（不动摇跟踪）
  → 验证: 动态区 PSNR 提升, ghosting 下降

阶段3: 效率硬化（轴A完整）
  ├─ 自组织/压缩（SOG 思想）迁移到在线
  ├─ 关键帧/高斯生命周期管理
  └─ 长序列稳定性
  → 验证: 长序列 ATE 不漂移, #Gaussians 受控
```

## 5. 组件依赖清单（本地环境）

| 组件 | 来源 | 本地状态 |
|---|---|---|
| gsplat | pip（有 wheel，CUDA 12.8/13.2 需 torch 2.7+）| 安装中 |
| DROID-DBA | monogs-ours/WildGS-SLAM 已有 | ✅ |
| RAFT | monogs-ours 已有（frozen）| ✅ |
| lietorch | 独立 pip | ⚠️ 需装 |

## 6. 风险与对策

1. **gsplat 版本兼容**：新版本要求 torch 2.7+，而 monogs-ours 用 torch 2.1+cu118。对策：新组件化项目用独立 conda env（gsplat 官方 wheel 支持 cu118/cu121 旧版本，或用 +pt24）。
2. **从零集成工作量大**：gsplat 提供标准 API + examples，DROID-DBA 可单独抽取。第一阶段只做最小闭环（静态重建 + 位姿），逐步加动态。
3. **单目尺度歧义**：DROID 本身是单目，天然匹配我们的单目主线。

## 7. 立即可做的 Phase 0

在本地 2060 上：
1. 安装 gsplat（独立 env）
2. 跑通 gsplat 官方 example（静态 3DGS 训练，验证渲染闭环）
3. 用 Replica/TUM 单目序列，验证 gsplat 位姿优化（pose gradient）在单目下的可行性
4. 最小集成：gsplat 渲染 + 简单 BA → 静态重建 PSNR 基线
