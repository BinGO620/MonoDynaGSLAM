# MonoDynaGSLAM — 项目硬约束

本文件承载**长期不变**的工程与研究纪律。当前活动计划在 `references/` 与关联项目 `monogs-ours`（RoGS-SLAM）中维护。

## 最高准则（所有决策的最终仲裁）

**不要为了赶进度写论文。实验方法达标、证明方法有用，才最重要。**

1. 截稿日不是硬约束，是自加的；方法未达标宁可押后一个窗口。
2. 每一篇要投的稿子必须先回答：方法贡献是"我们自己的"吗？对"动态 3DGS SLAM"题目方向有用吗？
3. headline 必须动态相关、框架通用、有自己的方法内核。
4. 写作是实验达标的自然产物，不是倒逼实验的方向盘。

## 领域定位

- 方向：**dynamic 3DGS SLAM，单目数据序列**（本项目主线）；
- 主 baseline：**WildGS-SLAM**（判定为 **anti-dynamic**，见 `references/wildgs-slam-analysis.md`）；
- 自有方法：RoGS-SLAM（`monogs-ours`），mask-free 可靠性加权路线，MMM 2027 已投稿、结果未出。

## 资产组织纪律（本仓库）

1. **单一事实源**：`data/methods.json` 是方法目录唯一权威；`references/` 分析、`papers/` 笔记、`skills/` 技能不得与它矛盾。修改方法信息先改 JSON。
2. **分类一致性**：anti-dynamic / face-dynamic / static-base 三分法（`data/categories.json`），任何文档不得发明新分类而不更新 JSON。
3. **事实可核查**：每个方法条目标注 arXiv ID / venue / code 链接；分析文档给出证据来源（论文原话、源码路径、codex 意见）。
4. **工程三阶段预算**：实验按 Phase 0/1/2 递进（见 `references/benchmarks.md`）。
5. **推送前校验**：`python scripts/validate_data.py` 通过才 commit。

## 咨询与对抗审核

- **codex**（本机 CLI，gpt-5.6-sol）用于方法机制、命名、结论、方向的不确定/分歧裁决；结论记录到对应 `references/` 文档的反方观点节。
- 联网核实用 Tavily/WebFetch；GitHub 仓库操作经 SSH（key: id_ed25519，账户 BinGO620）。
