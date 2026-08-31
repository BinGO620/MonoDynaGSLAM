# 三轴交叉方法 — Novelty 核查

> 对照 `data/methods.json` + 文献全景 agent 矩阵（25+ 篇），确认三轴交叉点是否空白。

## 三轴定义（用户确认框架）

- **A: 静态重建效率** — 3DGS 紧凑/自组织/压缩表示（MMM'27 教程线：HAC / Compact3D / Scaffold-GS / FLAS / SOG）
- **B: 动态处理** — anti-dynamic（移除/降权）↔ face-dynamic（显式建模）
- **C: SLAM 双任务** — 跟踪（BA/位姿优化）↔ 建图（渲染/重建）

## 单轴覆盖现状（文献矩阵）

| 轴 | 代表方法 | 覆盖度 |
|---|---|---|
| A | MMM'27 教程、Scaffold-GS、SOG、FLAS（离线静态） | 静态离线成熟，**在线 SLAM 未用** |
| B-anti | WildGS-SLAM、Dy3DGS、DAGS-SLAM、GGD-SLAM、DL-SLAM、DG-SLAM、Gassidy | 密集 |
| B-face | D2GSLAM、Flow4DGS-SLAM、4DGS-SLAM、RU4D-SLAM、DynaGSLAM | 增长中，全 RGB-D |
| C | MonoGS、SplaTAM、GS-SLAM（静态基座） | 成熟 |

## 两轴交叉现状

| 交叉 | 代表方法 | 状态 |
|---|---|---|
| A×C | 无（在线 SLAM 无紧凑表示） | **空白** |
| B×C | 大多数动态 SLAM（anti/face 用于 BA+渲染，但**同一策略贯穿两任务**） | 单策略，未分治 |
| A×B | 无（动态 SLAM 无紧凑动态表示） | **空白** |

## 三轴交叉（A×B×C）— Novelty 结论

**没有任何现有方法**同时做到：在线 SLAM 中使用紧凑静态表示 + 动态区域自适应 anti/face + BA 与渲染采用分治策略。三个交叉点两两皆为空白。

## 关键差异化定位（vs 最近工作）

| 邻近工作 | 做的是什么 | 我们不同在哪 |
|---|---|---|
| DAGS-SLAM | 不确定性调度（B×C，按需语义） | 无紧凑表示（A 缺失）；语义调度而非表示分治 |
| GGD-SLAM | 可泛化运动模型（B） | 无紧凑表示、无双任务分治 |
| Flow4DGS-SLAM | 光流引导 4D（B-face） | RGB-D；无紧凑静态表示 |
| RU4D-SLAM | 不确定性重加权+4D（B-face×C） | 无紧凑表示、无双任务 anti/face 分治 |
| DL-SLAM | 双层动态概率（B） | 无紧凑表示 |
| Scaffold-GS / SOG | 紧凑静态表示（A，离线） | 非 SLAM、非动态 |

## Novelty 声明（论文可用）

> "To the best of our knowledge, we are the first to jointly address (i) compact/self-organizing static Gaussian representation in online SLAM, (ii) adaptive anti-dynamic vs face-dynamic handling across tracking (BA) and mapping (rendering), and (iii) decoupled dynamic strategies for the two SLAM tasks."

## 风险提示

1. 紧凑表示（Scaffold-GS 锚点/SOG 自组织）在**在线增量**设置下的更新机制未验证——这是最大技术风险；
2. BA 用 anti、渲染用 face 是否会导致**同一高斯两套权重冲突**，需 Phase 0 验证；
3. 单目下紧凑表示 + 动态处理的计算预算（2060 6GB）。
