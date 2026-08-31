# 决策记录（ADR）

本项目采用轻量 ADR 记录影响仓库方向的判定，便于追溯"为什么这样组织/这样判定"。

## ADR-001: WildGS-SLAM 分类判定 = anti-dynamic

- **状态**: 已接受（2026-09-01）
- **问题**: baseline WildGS-SLAM 到底是 anti-dynamic 还是 face-dynamic？
- **判定**: **anti-dynamic（抗动态）**
- **证据**（三源交叉验证）:
  1. 论文摘要原话："reconstructs a 3D Gaussian map for **static elements**, effectively **removing all dynamic components**"；
  2. 方法机制：uncertainty MLP 权重只用于"降低动态干扰"，动态物体无显式运动/变形表示，不进入最终重建；
  3. codex（gpt-5.6-sol）独立意见：一致判 anti-dynamic。
- **影响**: related work 中 WildGS-SLAM 归"不确定性加权"子类；与 RoGS-SLAM 同为"anti-dynamic + 软加权 + mask-free"，差异在信号来源（学习 vs 几何）与依赖（metric depth vs 无）。
- **相关文档**: `references/wildgs-slam-analysis.md`、`references/taxonomy.md`、`references/method-comparison.md`

## ADR-002: 仓库组织方法论 = 双源参照

- **状态**: 已接受（2026-09-01）
- **问题**: 搜集到的资产如何科学组织？
- **判定**: 参照两个开源 skills 仓库的工程方法论：
  - **Awesome-Gaussian-Skills**: `data/` 单一数据源（methods.json/categories.json/datasets.json）+ `references/` 派生分析 + `scripts/` 校验（validate_data.py 保证一致性）；
  - **academic-research-skills**: 流水线式研究技能（deep-research / academic-paper / reviewer / pipeline），以 SKILL.md 契约化。
- **影响**: 本仓库成为"资产组织层"，代码留在 monogs-ours；方法目录唯一事实源，任何文档不得与之矛盾。
- **相关文档**: `CLAUDE.md`、`README.md`

## ADR-003: 实验纪律继承（monogs-ours → 本仓库）

- **状态**: 已接受（2026-09-01）
- **问题**: 实验预算如何分配？
- **判定**: 继承 monogs-ours 的三阶段递进纪律（Phase 0 机制自检 → Phase 1 信号量级 → Phase 2 全矩阵判决），Phase 1 效应 <6% 即停。
- **相关文档**: `references/benchmarks.md`、`CLAUDE.md`
