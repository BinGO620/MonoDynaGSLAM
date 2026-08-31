# 方法提案：TriAxis-SLAM — 三轴解耦的动态 3DGS SLAM

> 基于用户三轴框架设计：A（静态效率） + B（anti↔face） + C（跟踪BA↔建图渲染）

## 核心创新（一句话）

**首次在动态 3DGS SLAM 中实现三轴解耦：静态用紧凑高效表示（轴A），动态按任务分治 anti/face（轴B），BA 跟踪与渲染建图采用独立动态策略（轴C）。**

## 系统架构

```
RGB 单目序列
   │
   ├─ A: 静态紧凑表示 ──────────────────────────
   │   └─ 自组织/锚点高斯（Scaffold-GS / SOG 思想）
   │       静态区域：anchor + MLP 解码 → 紧凑参数
   │       动态区域：全参数 3D/4D 高斯（不压缩）
   │
   ├─ B: 动态状态感知处理 ──────────────────────
   │   ├─ 状态机：静态 / 瞬态静止 / 运动 / 未知
   │   ├─ anti-dynamic 策略（BA 跟踪用）：
   │   │   光流-重投影残差 → Cauchy 降权（RoGS-SLAM 资产）
   │   └─ face-dynamic 策略（渲染用）：
   │       动态高斯 → 4D 变形场 / 时间中心（D2GSLAM 启发）
   │
   └─ C: SLAM 双任务分治 ──────────────────────
       ├─ 跟踪（BA）：anti-dynamic 加权位姿优化
       │   └─ 只信任静态/瞬态静止区域的匹配
       └─ 建图（渲染）：face-dynamic 时序重建
           └─ 动态区域显式建模 → 完整视觉渲染
```

## 与现有方法的差异（三轴逐项）

| 方法 | 轴A（紧凑静态） | 轴B（动态策略） | 轴C（任务分治） | 传感器 |
|---|---|---|---|---|
| **TriAxis-SLAM (ours)** | **✅ anchor/自组织** | **✅ 自适应 anti/face** | **✅ BA 用 anti，渲染用 face** | 单目 |
| WildGS-SLAM | ❌ 标准 3DGS | anti（不确定性） | ❌ 统一策略 | 单目 |
| D2GSLAM | ❌ 标准 3DGS | face（4D高斯） | ❌ 统一 face | RGB-D |
| DAGS-SLAM | ❌ 标准 3DGS | anti（按需语义） | ❌ 统一 anti | RGB-D |
| GGD-SLAM | ❌ 标准 3DGS | anti（运动模型） | ❌ 统一 anti | 单目 |
| Flow4DGS-SLAM | ❌ 标准 3DGS | face（4D+光流） | ❌ 统一 face | RGB-D |
| RU4D-SLAM | ❌ 标准 3DGS | face（4D+不确定性） | ❌ 统一 face | RGB-D |
| Scaffold-GS | ✅ anchor 紧凑 | ❌ 静态方法 | ❌ 非 SLAM | 离线 |

## 消融矩阵（三轴验证）

| 配置 | 轴A(紧凑) | 轴B(自适应) | 轴C(分治) | 预期 ATE | 预期 PSNR |
|---|---|---|---|---|---|
| Full TriAxis | ✅ | ✅ | ✅ | **最佳** | **最佳** |
| A off：标准 3DGS | ❌ | ✅ | ✅ | 稍高 | 接近 |
| B off：纯 anti | ✅ | ❌(anti) | ✅ | 接近 | 降（ghosting） |
| C off：统一 anti | ✅ | ✅ | ❌(统一) | 接近 | 降 |
| C off：统一 face | ✅ | ✅ | ❌(统一) | 降（漂移） | 接近 |
| MonoGS 基线 | ❌ | ❌ | ❌ | 最高 | 最低 |

## Phase 0 验证计划

**目标**：验证轴C（BA 用 anti vs 渲染用 face 是否冲突）的核心假设

**装置**：本地 2060（6GB） + Bonn balloon（440 帧，最短动态序列）
**控制**：MonoGS 原生（统一处理）
**实验**：同一高斯表示，BA loss 加 anti 权重（光流-重投影不一致 → 降权），渲染 loss 不加/加不同权重
**观测**：ATE 变化 + 渲染 PSNR 变化 + ghosting 程度
**判决**：ATE 不上升且 PSNR 不下降 → 分治可行（进 Phase 1）