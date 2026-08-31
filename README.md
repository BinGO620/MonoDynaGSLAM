# MonoDynaGSLAM

**Mono Dynamic 3D Gaussian Splatting SLAM** — 单目动态 3DGS SLAM 领域的结构化研究资产库。

## 项目定位

本仓库是 **dynamic 3DGS SLAM（单目序列）** 方向的**研究资产组织层**，参照两份开源 skills 仓库的工程方法论组织：

- **[Awesome-Gaussian-Skills](https://github.com/jaccen/Awesome-Gaussian-Skills)**：`data/` 单一数据源 + `references/` 知识库分析 + `skills/` SKILL.md 技能 + `scripts/` 校验脚本。
- **[academic-research-skills](https://github.com/Imbad0202/academic-research-skills)**：deep-research / academic-paper / academic-paper-reviewer / academic-pipeline 四大技能流水线，`shared/` 共享契约。

核心载体：**方法目录（`data/methods.json`）** 是唯一事实源，`references/` 分析文档、`skills/` 技能、`papers/` 论文笔记均从中派生。

## 核心结论（一句话）

> **WildGS-SLAM（CVPR 2025）属于 anti-dynamic（抗动态）方法**：它用 DINOv2 特征 + uncertainty MLP 预测逐像素不确定性，在 tracking 和 mapping 中**移除/降权动态物体**，只重建静态地图，**不显式建模动态物体的运动**。详见 [`references/wildgs-slam-analysis.md`](references/wildgs-slam-analysis.md) 与 [`references/taxonomy.md`](references/taxonomy.md)。

## 仓库结构

```
MonoDynaGSLAM/
├── data/                  # 单一数据源（方法目录、分类体系、数据集）
│   ├── methods.json       # 方法目录（唯一事实源）
│   ├── categories.json    # anti-dynamic / face-dynamic / 静态基座 分类体系
│   └── datasets.json      # 评测数据集清单（Bonn/TUM/Replica/Wild-SLAM…）
├── references/            # 知识库分析文档（从 data/ 派生）
│   ├── taxonomy.md        # anti-dynamic vs face-dynamic 分类法
│   ├── wildgs-slam-analysis.md  # baseline WildGS-SLAM 深度分析
│   ├── baselines.md       # 主要 baseline 对比表
│   └── benchmarks.md      # 评测基准与协议
├── skills/                # SKILL.md 格式领域技能（_contracts/ 定义 I/O 契约）
├── papers/                # 论文笔记（按 arXiv ID 组织）
├── code/                  # 代码资产索引（monogs-ours、WildGS-SLAM 等）
├── scripts/               # 校验/构建脚本
└── docs/                  # 决策记录与工程文档
```

## 研究方法（monogs-ours / RoGS-SLAM）

自有方法 **RoGS-SLAM**（[monogs-ours](https://github.com/BinGo620/RoGS-SLAM)）：mask-free 动态 3DGS SLAM，光流-重投影一致性 + 深度残差异常构成可靠性信号，Huber/Cauchy 降权（软权重、不硬删像素）。**已投稿 MMM 2027，录用结果未出。** 方法笔记见 `papers/` 与 `references/baselines.md`。

## 快速开始

```bash
# 校验数据源完整性（方法目录 ↔ 分析文档 ↔ 笔记）
python scripts/validate_data.py

# 按分类筛选方法
python scripts/query_methods.py --category anti-dynamic --sensor monocular
```

## 引用与许可

方法目录数据来源于公开论文；代码资产索引指向各自上游仓库。请遵守各上游仓库的 LICENSE。
