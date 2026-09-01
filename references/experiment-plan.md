# 正式实验矩阵：MonGauss 单目动态 3DGS SLAM

> 工具：3dgs-experiment-planner skill（11维对比框架 + Tier 分层）
> 目标：ECCV 2026 / ICRA 2027，CCF-B 以上
> 论文定位：首次用独立组件（gsplat + Metric3D）搭建单目动态 3DGS 完整管线，提出 Gaussian 级 anti-dynamic + face-dynamic 衔接

## 1. Datasets（按 reviewer 要求选）

| Priority | Dataset | 场景数 | 模态 | 难度 | 用途 |
|---|---|---|---|---|---|
| **Must** | Bonn RGB-D Dynamic（单目化） | 4（balloon/crowd/mv_no_box/pt2） | RGB-D → 只用 RGB | Hard | 动态场景主表 |
| **Must** | TUM RGB-D Dynamic（单目化） | 3（fr3_walking_xyz/halfsphere/rpy） | RGB-D → 只用 RGB | Medium | 动态跟踪精度 |
| **Must** | TUM RGB-D Static | 2（fr1_desk/fr3_long_office） | RGB-D → 只用 RGB | Medium | 静态 no-harm 对照 |
| Should | Replica | 2（office0/room0） | RGB-D → 只用 RGB | Easy | 几何/渲染基线 |
| Nice | Wild-SLAM（WildGS 官方） | Mocap 系列 | Monocular | Hard | 单目评测 |

> 注：所有序列用 Metric3D 深度先验，不使用 RGB-D 信息（保持单目）。

## 2. Baselines（Tier 分层）

### Tier 1 — 必须对比

| 方法 | Venue | 传感器 | 类型 | 对比理由 |
|---|---|---|---|---|
| **MonoGS** | CVPR'24 | 单目 | static-base | 本项目基座 |
| **WildGS-SLAM** | CVPR'25 | 单目 | anti-dynamic | 主要 baseline |

### Tier 2 — 应当对比

| 方法 | Venue | 传感器 | 类型 | 对比理由 |
|---|---|---|---|---|
| **GGD-SLAM** | ICRA'26 | 单目 | anti-dynamic | 可泛化运动模型，最新 |
| **Dy3DGS-SLAM** | ICRA'25 | 单目 | anti-dynamic | 单目 mask 路线 |
| **DL-SLAM** | arXiv'26 | 单目 | anti-dynamic | 双层概率，最新 |
| **D2GSLAM** | arXiv'25 | RGB-D | face-dynamic | 4D 高斯，对照 face 路线 |

### Tier 3 — Nice to compare

| 方法 | Venue | 说明 |
|---|---|---|
| Flow4DGS-SLAM | arXiv'26 | RGB-D，光流引导 4D |
| DAGS-SLAM | arXiv'26 | RGB-D，运动概率+按需语义 |

## 3. Metrics

### 主指标（Must Report）

| 指标 | 工具 | 用途 |
|---|---|---|
| **ATE RMSE** (cm) | evo_ape | 轨迹精度 |
| **PSNR** (dB) | 标准 | 渲染质量 |
| **SSIM** | 标准 | 结构相似 |
| **LPIPS** | lpips 包 | 感知质量 |

### 补充指标（Report When Relevant）

| 指标 | 工具 | 用途 |
|---|---|---|
| #Gaussians (k) | — | 模型效率 |
| **动态区 PSNR** | 只算动态物体区域 | face-dynamic 增益验证 |
| Ghosting rate | 可视化/指标 | ghosting 伪影程度 |
| FPS | 定时 | 实时性 |
| VRAM peak | nvidia-smi | 显存占用 |
| Training time (s) | wall-clock | 效率 |

## 4. 消融矩阵

### 核心消融（单变量）

| # | 配置 | A(反投影) | B(anti) | C(face offset) | 预期 PSNR 影响 |
|---|---|---|---|---|---|
| 1 | Full MonGauss | Metric3D | Gaussian anti | per-frame offset | **最佳** |
| 2 | w/o Metric3D depth | 随机初始化 | Gaussian anti | per-frame offset | 严重下降 |
| 3 | w/o anti | Metric3D | 无 | per-frame offset | -1~4 dB |
| 4 | w/o face-dynamic | Metric3D | Gaussian anti | 无 | -0.3~1 dB |
| 5 | 帧级 loss 降权替代 | Metric3D | 帧级降权（已证伪）| 无 | -4 dB |
| 6 | Pure static baseline | Metric3D | 无 | 无 | 24~25 dB（参考） |

### τ 超参消融（已部分完成）

| τ | walking_xyz | Bonn balloon | 选择 |
|---|---|---|---|
| 无 anti | 24.03 | 25.30 | baseline |
| 1.5 | 23.51 | — | 过度压制 |
| **2.5** | **25.42** | **29.49** | **最优** |
| 3.5 | (被kill) | — | 趋势下降 |

## 5. Figure Plan

| 图号 | 内容 | 目标页 |
|---|---|---|
| Fig 1 | 动机 Teaser：单目动态场景的高斯点云 vs 重建渲染对比 | 1 |
| Fig 2 | 系统架构：gsplat 渲染 + Metric3D 深度 + 动态状态机 + per-frame offset | 2 |
| Fig 3 | Gaussian anti 机制：残差图 → opacity 下调 → 重建质量提升 | 3 |
| Fig 4 | τ 消融可视化：残差图在不同 τ 下的变化 | 3 |
| Fig 5 | 定性对比：多方法 × 多序列渲染对比（grid） | 4 |
| Fig 6 | ATE 轨迹对比：各方法轨迹 vs GT | 4 |

## 6. 效率分析

| 指标 | 测量方式 | 当前值（2060） | 3090 预估 |
|---|---|---|---|
| Training time | wall-clock | ~300s/50帧 | ~150s |
| VRAM peak | nvidia-smi | ~5 GB | — |
| #Gaussians | — | 60k（可配置） | — |
| FPS (rendering) | 单帧时间 | ~0.02s | ~0.01s |

## 7. Reviewer Concerns & Preemptive Responses

| Concern | Response |
|---|---|
| "Metric3D 深度先验的泛化性？" | 我们在 TUM+Bonn 两个数据集上验证；Metric3D 是 CVPR'23 顶会方法，广泛用于单目 SLAM |
| "为什么不比 RGB-D 方法？" | 本文聚焦单目，RGB-D 方法不适用同评测协议 |
| "face-dynamic 当前提升有限？" | 相对 static-only 有 +0.3 dB，且工程管线已验证可行；完整 face-dynamic（4D 高斯）是未来工作 |
| "Gaussian anti 是否只是降权？" | 有完整 τ 消融 + 帧级降权证伪对比，机制分析明确 |
| "效率和单目精度的 trade-off？" | 展示训练时间 vs PSNR 曲线；Metric3D 先验一次计算、训练中复用 |
