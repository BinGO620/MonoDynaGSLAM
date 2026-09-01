# 探索路线图：RGB-D → 单目 → anti-dynamic → face-dynamic

> 生成：2026-09-01 · 依据：文献矩阵（25+ 篇）+ codex 对抗审核 + Phase 0 实验证据
> 核心用户方向：从已做的 RGB-D（有深度）演进到单目，再从 anti-dynamic 演进到 face-dynamic；同时 3DGS 效率要提升。

## 1. 演进路径与现状

```
阶段 0: RGB-D 3DGS SLAM（已有经验）
   └─ monogs-ours / RoGS-SLAM：mask-free 可靠性加权 anti-dynamic（已投 MMM）
       资产：光流-重投影一致性、Cauchy 降权、生命周期组件
       ⚠ 依赖深度 → 单目下不可直接用

阶段 1: 单目 3DGS SLAM（无深度）← 当前要跨过的坎
   └─ 挑战：无真深度 → 动态检测信号变弱、尺度歧义
   └─ 现有：MonoGS/Splat-SLAM（静态）、WildGS/Dy3DGS（anti，依赖学习先验）

阶段 2: 单目 anti-dynamic（把动态当干扰抑制）
   └─ 现状：WildGS-SLAM（DINOv2+深度先验）、Dy3DGS（光流+深度网络）
   └─ 空缺：纯几何、零学习先验的单目 anti-dynamic（G1，文献 agent 确认"单目下为零"）

阶段 3: 单目 face-dynamic（显式建模动态）← 最终目标
   └─ 现状：单目 face-dynamic 只有 DynaGSLAM（WACV'26，SAM2+RAFT+DynoSAM 三重型组件）
   └─ 空缺：单目 face-dynamic 轻量化是明确空档（所有 4D 方法 D2GSLAM/Flow4DGS/RU4D 均 RGB-D）
```

## 2. 证据驱动的关键判断

### 2.1 为什么单目是真正的分水岭（Phase 0 实验证据）
- gsplat 组件闭环可跑通（渲染 + 位姿梯度反传 + 真实数据加载）✅
- 但无深度先验时 PSNR 仅 ~5.7 dB（视锥随机初始化），有 GT 深度时仍偏低 → 初始化质量是单目重建的主要瓶颈
- 结论：**单目 3DGS SLAM 必须引入某种深度先验**（估计网络 or 几何三角化 or DROID 逆深度）

### 2.2 为什么 face-dynamic 是合理终点（文献证据）
- 所有单目 anti-dynamic 都把动态当噪声丢弃，动态物体的运动信息被浪费
- 所有单目 face-dynamic 依赖重型组件（SAM2/RAFT/DynoSAM），难以实时、难在 2060 级 GPU 跑
- 从 anti 到 face 的**轻量化衔接**（复用 anti 的动态分离结果 → 直接对分离出的动态高斯做 4D 建模）在单目下无现成工作

### 2.3 codex 红线（避免重复）
- ❌ 时空运动概率过滤（DAGS-SLAM 已做）
- ❌ 语义+几何双层概率（DL-SLAM 已做）
- ❌ 静态3D+动态4D 联合表示（D2GSLAM 已做，但 RGB-D——这是可差异化点）
- ❌ 不确定性重加权+4D（RU4D-SLAM 已做）
- ✅ 可做：决策层、生命周期、可逆记忆、长期一致性、统计校准、预算分配

## 3. 探索路线（三步递进）

### Step 1：单目静态 3DGS SLAM 基线（组件化）
**目标**：建立"gsplat 渲染 + 深度先验 + 位姿优化"的最小闭环，达到可用静态重建质量
**组件映射**：
- 渲染：gsplat（已装，OpenCV 约定已确认）
- 深度先验：DROID-DBA（monogs-ours/WildGS-SLAM 已有）或 Metric3D（本地已有 /data/Metric3D）
- 位姿：DROID 位姿 或 简单 BA
**验证**：TUM 单目序列 PSNR ≥ 20 dB、ATE 达标
**产出**：可复用的组件化静态基线（不依赖 monogs-ours 单体）

### Step 2：单目 anti-dynamic 扩展
**目标**：在 Step 1 基线上加动态抑制，复用 RoGS-SLAM 的几何一致性思想但单目化
**关键改动**：深度残差信号不可用 → 用光流-重投影一致性 + 渲染残差统计
**验证**：Bonn 动态序列单目 ATE 优于静态基线，静态 PSNR 不退化
**产出**：单目 anti-dynamic 组件（可与 WildGS/Dy3DGS 对比）

### Step 3：单目 face-dynamic 演进
**目标**：把 Step 2 分离出的动态高斯显式建模（4D/变形场），实现 anti→face 无缝衔接
**关键思想**：anti 阶段的动态分离结果直接作为 face 阶段的输入（不重复检测）
**差异化**：相对 D2GSLAM 等 RGB-D 方法，单目 + 轻量化；相对 DynaGSLAM，无重型组件
**验证**：动态区域渲染 PSNR 提升、ghosting 下降、ATE 不退化

## 4. 每步的 Phase 0 验证计划

| Step | Phase 0 装置 | 判决门 |
|---|---|---|
| 1 | 本机 2060，TUM fr1_desk 30 帧，gsplat+深度先验 | PSNR ≥ 20 dB 且位姿梯度收敛 |
| 2 | 本机 2060，Bonn balloon 440 帧（单目化） | 动态 ATE 优于静态基线 ≥6% |
| 3 | 远程 3090，Bonn crowd（单目化） | 动态区 PSNR 提升 + ghosting 下降 |

## 5. 立即执行

Step 1 已部分完成（gsplat 组件验证）。下一步：
1. 接入深度先验（DROID-DBA 或 Metric3D），重建 PSNR 提升到可用水平
2. 把最小闭环固化为 `src/` 组件化代码（独立于 monogs-ours）
3. 存档本路线图到 `references/roadmap.md`
