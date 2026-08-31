# Dynamic 3DGS SLAM（单目）研究 Idea 探索报告

> 生成：2026-09-01
> 方法：deep-research lit-review 模式 + nature-literature-pipeline gap-analysis + codex（gpt-5.6-sol）对抗审核 + 本地源码/数据交叉核实
> 数据源：`data/methods.json` + 文献全景 agent（25+ 篇，arXiv/venue 佐证）+ arXiv API + Web 检索
> 定位：RoGS-SLAM 仅作已有资产（组件可借鉴），不进入论文主线。

---

## 1. 领域现状速览（2024-2026）

动态 3DGS SLAM 两年内爆发 25+ 篇，两大路线已趋饱和：

- **anti-dynamic**（把动态当干扰移除/降权）：语义 mask 与不确定性/概率两大赛道**密集**（DG-SLAM/DGS-SLAM/SGF-SLAM/SLAM-X/WildGS-SLAM/UP-SLAM/DL-SLAM/DAGS-SLAM/MPDG-SLAM…）
- **face-dynamic**（显式建模动态，4D/变形场）：2025H2 密集出现，但**清一色 RGB-D**（D2GSLAM/Flow4DGS-SLAM/RU4D-SLAM/4DGS-SLAM/PG-SLAM…）

**单目（纯 RGB）动态 3DGS SLAM 现有**：WildGS-SLAM、Dy3DGS-SLAM、DyGS-SLAM(Sensors)、GGD-SLAM、DL-SLAM、DynaGSLAM、NRGS-SLAM、M³（feed-forward/非刚性）。其中**端到端在线、单目、无深度先验、纯几何一致性的 anti-dynamic 方法不存在**。

## 2. Knowledge Gaps（按优先级）

| # | Gap | 状态 | 优先级 |
|---|---|---|---|
| G1 | **单目 + 纯几何一致性（无 DINOv2/单目深度/学习式光流）的 anti-dynamic 3DGS SLAM 缺失**。纯几何代表 Gassidy 依赖 RGB-D；单目方法全部引入学习先验 | 方法空缺 | **高** |
| G2 | 动态感知的关键帧选择/地图管理（动态含量、运动边界、遮挡回填作为关键帧代价） | 方法空缺 | 中 |
| G3 | **动态物体生命周期管理**（出现→运动→静止→消失→重访 状态机 + 地图遗忘/重映射/实例记忆） | 方法空缺+机制不完整 | **高** |
| G4 | **anti ↔ face 自适应切换**（按动态程度/语义/算力预算在线切换丢弃或 4D 建模） | 框架空缺 | **高** |
| G5 | 长时间动态序列漂移控制 + 缺乏长时动态基准 | 方法+基准空缺 | **高** |
| G6 | 单目光流引导、类别无关、免深度先验的 anti-dynamic（Flow4DGS 机制单目化） | 方法空缺 | 中 |
| G7 | 边缘/嵌入式部署的单目动态 3DGS SLAM（Jetson 等） | 方法空缺 | 中 |
| G8 | object-centric face-dynamic（实例高斯簇 + 6DoF/轨迹 + 运动预测） | 方法空缺 | 低-中 |
| G9 | 单目动态评测协议（带 GT 单目动态序列、长序列、间歇动态标准化） | 基准空缺 | 中 |

## 3. codex 候选 Idea 排名（对抗审核后）

| 排名 | Idea | 新颖性 | 可行性 | 单目相关 | 综合 | codex 建议 |
|---|---|---|---|---|---|---|
| 1 | 动态状态驱动的 anti/face 自适应切换 | 5 | 4 | 5 | 14 | 首选（与 Idea2 组合成系统） |
| 2 | 动态感知的主动关键帧选择 | 5 | 4 | 5 | 14 | 首选（与 Idea1 组合） |
| 3 | mask-free 连续时间鲁棒高斯后验 | 4 | 5 | 5 | 14 | 最稳妥工程路线 |
| 4 | 动态物体生命周期与可逆地图记忆 | 5 | 3 | 5 | 13 | — |
| 5 | 静态锚点-动态变化日志（长漂移） | 4 | 3 | 5 | 12 | — |
| 6 | 纯几何可校准动态因果不确定性 | 4 | 3 | 5 | 12 | — |

## 4. 禁止重复的方向（codex 红线）

- 时空运动概率过滤动态高斯 → 已覆盖：DAGS-SLAM、MoPe
- 语义+几何双层动态概率 → 已覆盖：DL-SLAM
- 静态 3D + 动态 4D 联合表示 → 已覆盖：D2GSLAM、Flow4DGS-SLAM
- 光流引导 mask + 4D 形变 → 已覆盖：Flow4DGS-SLAM
- FIFO/时序模型提取动态特征 → 已覆盖：GGD-SLAM
- 渐进式动态 scaffold 重建 → 已覆盖：ProDyG
- 不确定性重加权 + 4D mapping → 已覆盖：RU4D-SLAM

**2026-2027 更有竞争力的切入点是决策层、生命周期、可逆记忆、长期一致性、统计校准和预算分配，而不是再增加一个动态分割网络。**

## 5. 建议切入方向（综合文献 agent + codex）

**主选组合（系统级贡献）**：G1/G3/G4 交汇处的"**具备生命周期管理的自适应单目动态 3DGS SLAM**"：
- 动态状态估计决定表示类型（anti/face 切换）；
- 主动关键帧策略决定何时更新哪类地图；
- 生命周期状态机处理间歇出现/瞬态静止；
- 全程单目 + 纯几何（G1 卖点：零外部模型、开箱即用）。

**最快落地（1 年内的稳妥结果）**：mask-free 鲁棒后验（codex Idea 3），直接基于 MonoGS/WildGS-SLAM 改造，避免训练大模型，实验闭环清晰。

## 6. 验证路径（三阶段）

- **Phase 0（机制自检，本机 2060）**：最短序列 × 1 seed × treatment vs control，只看机制诊断；
- **Phase 1（信号量级，本机/远程 3090）**：两个最短序列 × 1-3 seed，ATE 效应 ≥6% 才进 Phase 2；
- **Phase 2（全矩阵判决，远程 jiangwenheng 双 3090）**：5 序列 × 3 seed × 全臂。

数据集：Bonn（balloon/crowd/mv_no_box 等动态）、TUM（freiburg3_walking_*）、Replica（静态对照）。
指标：ATE/RPE（主）、静态 PSNR/SSIM/LPIPS、动态 mask F1、ghosting rate、#Gaussians、FPS。

## 7. 反方观点记录

**codex 对纯几何路线的警告**（诚实记录）："纯几何对大面积动态或低视差场景天然受限；不能宣称全面超过学习方法。更合理的定位是'无需语义的可靠性估计和失效边界分析'。"——若走 G1，论文叙事应定位为"无先验鲁棒性 + 失效边界分析"，而非通用 SOTA。
