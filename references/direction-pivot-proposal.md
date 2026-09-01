# 方向修正提案：从"动态压制"转向"动态感知高斯生命周期管理"

> 2026-09-02 · 触发：用户"效果不好就反思+找灵感" + 联网调研 + 本项目 n=8 审计证据链
> 状态：提案（待验证）

## 1. 三个反思（效果不好的根因）

1. **战场错位**：anti-dynamic 的主战场是跟踪（ATE），我们无跟踪闭环，只能在收益最弱的渲染指标上打转。DROID-SLAM in the Wild（CVPR 2026，WildGS 同团队续作）正是把不确定性打进 BA——印证此判断。
2. **压制范式失效的深层原因**（我们自己的审计发现）：动态高斯在多帧光度冲突下已被梯度优化自然压制（低 opacity），"识别→衰减/删除"没有附加价值。**正确的操作不是压制，而是管理高斯的生死。**
3. **天花板问题**：动态区 PSNR 21 vs 静态区 26——5 dB 空间在"被动态遮挡的背景如何恢复"，压制做不到。

## 2. 联网发现（2026 年领域信号）

| 论文 | Venue | 信号 |
|---|---|---|
| **TAD-GS**: Temporally Aware Densification for Dynamic 3DGS | **ECCV 2026** | "现有致密化在短寿命快速运动区域失效，动态物体模糊欠重建"——动态感知致密化被顶会认可；**但它是离线多视角（Neural 3D Video），不是在线 SLAM** |
| DROID-SLAM in the Wild | CVPR 2026 | 多视角特征不一致 → 逐像素不确定性 → 可微 BA（anti 打进跟踪端） |
| Dynamic Visual SLAM using a General 3D Prior | CVPR 2026 | feed-forward 模型滤动态 + patch BA（Stachniss 组）；**只做位姿+深度，不做 3DGS 建图** |
| Flow4DGS-SLAM | CVPR 2026 Highlight | 光流引导动静分离 + 4D 高斯，"static and dynamic 同时重建" |
| VGGT-SLAM 2.0 / MASt3R-SLAM / Flash-Mono (ICLR'26) | 多篇 | 几何基础模型前端是 2026 主线，动态处理是插入模块 |

## 3. 新方向：动态感知的高斯生命周期管理（在线单目 SLAM）

### 核心范式差异

| 范式 | 操作 | 代表 | 问题 |
|---|---|---|---|
| 检测→删除/压制 | 一次性二值/连续决策 | DGS-SLAM、Dy3DGS、GARAD、WildGS | 我们已审计证明：压制无附加价值 |
| 离线动态致密化 | 为动态区专门致密化 | TAD-GS（ECCV'26） | 离线、非 SLAM |
| **生命周期管理（本提案）** | **证据积累 → 延迟提交/回收/再生** | **无（在线单目）** | — |

### 机制设计（monogs-ours D 组件的单目化 + 扩展）

1. **候选池（candidate pool）**：新高斯不立即入图，以低权重渲染；跨 K 帧一致观测（多帧光度/几何一致）才转正（deferred commit）——动态物体路过产生的瞬时不一致高斯自然无法转正；
2. **回收池（eviction pool）**：与累积证据冲突的旧高斯（鬼影）进入回收池，跨帧确认后删除或**重定位**（relocate 到其证据支持的另一次观测）；
3. **再生（re-densification）**：动态物体移开后暴露的背景区域，由"曾被遮挡标记"触发补致密化——这是压制范式做不到的（信息已被删）。

### 为什么是这个方向（证据链支撑）

- **本项目的审计（n=8）**证明压制范式失效——这是新方向的动机实验，现成；
- **TAD-GS（ECCV'26）**证明动态致密化是领域痛点——但在线单目 SLAM 无生命周期范式（gap）；
- **monogs-ours D 组件**（lineage lifecycle: deferred commit/evict with evidence accumulation）是已验证的 RGB-D 原型——单目化 = 新贡献 + 资产复用；
- **静态误伤/复现性问题**的根因（无仲裁的即时决策）正好被生命周期框架修复：不确定的高斯被"延迟观察"而非"立即压制"。

### 需验证的核心假设

H1：动态序列的主要伪影是"鬼影高斯"（动态物体路过时错误提交的高斯），生命周期管理可量化减少它们（Bonn GT mask 可量化 ghosting）；
H2：延迟提交+回收 优于 压制/删除（对比 selective/random decay/删除三对照）；
H3：再生机制恢复被动态遮挡的背景（动态区 PSNR 提升的主要来源）。

### 验证路径（Phase 0，1-2 周）

1. 用现有组件化管线实现最小生命周期（候选池 + 延迟提交，约 200 行）；
2. Bonn balloon + walking_xyz：ghosting 量化（GT mask 区域的鬼影高斯计数）+ 动态区/静态区 PSNR；
3. 对照：baseline / selective decay / 生命周期。判决门：动态区 PSNR 提升 ≥1 dB 且鬼影高斯数下降 ≥30%。

## 4. 风险与备选

- 风险：生命周期调度（K 值、证据阈值）引入新超参——需要类似 τ 的敏感性分析纪律；
- 风险：TAD-GS 作者或他人可能已在做在线版——投稿前需再查新；
- 备选方向（若 H1-H3 证伪）：几何基础模型前端（VGGT）替换 PnP+Metric3D，解决位姿-外观退化（工程升级，novelty 弱但稳定）。

## 5. Phase 0 执行记录（2026-09-02）与载体修正

### 5.1 自建递增框架的教训
lifecycle_incremental.py（递增式训练 + 候选池）跑通但地图质量天花板 10 dB（全帧管线 24.7）：
- 修复轨迹：颜色归一化 bug → scales 可优化 + 窗口联合优化 → 致密化提速，6→10 dB；
- 剩余差距需重写完整 3DGS 致密化体系（clone&split / opacity reset / 多分辨率）——
  **这不是机制自检，是造轮子**。按三阶段纪律停止。

### 5.2 正确载体发现：monogs-ours deferred_commit.py（661 行）
- **三臂生命周期消融已实现**：`lifecycle_mode ∈ {immediate, prune, deferred}`——
  immediate=vanilla 对照、prune=insert-then-remove 对照、deferred=ours（延迟提交）；
- 对称证据 C± 计数、lineage allocator、static evidence、reliability 确认通道；
- arms 间唯一 allowed_config_diff 的复现纪律。
- **codex 建议的三臂拆分在资产中已存在**；EXP51-54 已有 RGB-D 消融数据。

### 5.3 修正后的验证路径
1. **H1/H2 的正确载体**：monogs-ours 三臂框架（RGB-D 成熟环境）跑 ghosting 量化
   （Bonn GT mask：deferred vs immediate vs prune 的地图污染率）——直接复用；
2. **MonoDynaGSLAM 的贡献点**：单目 evidence 信号（多帧渲染一致性，已实现+已审计）
   替换 RGB-D 可靠性信号接入三臂框架 = 单目 lifecycle（G1 + G3 的交汇）；
3. codex 的闭环退化警告（位姿误差污染 evidence）作为设计约束纳入。

### 5.4 三方综合结论（codex 意见 + 独立验证 + 项目资产）
方向成立但 gap 需收窄（"在线单目"→"单目 evidence 驱动的 lifecycle 三臂对照"）；
Phase 0 教训确认了 codex 的"先窄后宽"建议；载体从自建框架修正为 monogs-ours。
