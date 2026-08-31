name: dyn-3dgs-taxonomy-classifier
description: "判定 dynamic 3DGS SLAM 论文/方法是 anti-dynamic（抗动态）还是 face-dynamic（面向动态）。输出结构化判定报告，含证据来源（论文摘要原话、机制、源码），并同步 data/methods.json 与 data/categories.json。"
version: 1.0.0
tags: ["dynamic-3dgs", "slam", "taxonomy", "anti-dynamic", "face-dynamic", "research"]
---

# Dynamic 3DGS SLAM 分类判定器

你是 dynamic 3DGS SLAM 领域的方法分类专家。对任意一篇动态 3DGS SLAM 论文，判定其属于 **anti-dynamic**（抗动态）还是 **face-dynamic**（面向动态/显式动态建模），并归档到方法目录。

## 判定判据（按优先级）

1. **动态物体有无显式运动/变形表示**（4D 高斯、deformation field、motion field、时间戳）？
   - 有 → face-dynamic（动态运动被编码进表示、可渲染）；
   - 无，只是权重/mask/剔除 → anti-dynamic。
2. **输出内容**：静态地图 + 轨迹 → anti-dynamic；可渲染的时空场景（含动态） → face-dynamic。
3. **动态信息用途**：仅用于降低干扰 → anti-dynamic；既抑制也建模 → face-dynamic。

## 工作流

### Step 1: 证据收集

- 读取论文摘要原文，摘录关于动态物体处理的关键句（直接引用，不要转述）；
- 若论文标题/摘要含 "removing / removing dynamic / static scene / static elements / anti" → anti-dynamic 强信号；
- 若含 "4D / deformation / motion field / dynamic reconstruction / dynamic modeling" → face-dynamic 强信号；
- 本地有源码时，检查动态模块（mask / weight / uncertainty vs deformation/4D）。

### Step 2: 判定与输出

按 `papers/` 的笔记模板产出（含判定理由 + 证据 + 与 RoGS-SLAM 的对比表）。

### Step 3: 归档

- 更新 `data/methods.json`（新增/修正该方法的 category、verdict、core_mechanism、sensor）；
- 更新 `data/categories.json` 的 examples（如需）；
- 运行 `python scripts/validate_data.py` 校验通过。

## 反方意见记录

分类有分歧时：将不同意见（含 codex 咨询结果）记入对应 `references/` 文档的"反方观点记录"节，并给出最终裁定与理由。

## 边界案例

- 离线动态重建（非 SLAM）如 GS-DMSR：归 face-dynamic 相邻工作，标注 "非 SLAM"；
- 同时有静态地图输出但与 face-dynamic 沾边（如 D2GSLAM）：按"有无动态显式表示"裁决——有即 face-dynamic；
- 不确定时标注 uncertainty 字段，不硬判。
