# 代码资产索引

本目录记录与本仓库关联的本地代码仓库位置与用途。**本仓库不包含代码**，代码在上游/本地仓库维护，此处仅做索引。

## 核心项目

| 仓库 | 本地路径 | 用途 | 远程 |
|---|---|---|---|
| **monogs-ours (RoGS-SLAM)** | `/data/monogs-ours` | 自有方法：mask-free 可靠性加权动态 3DGS SLAM；MMM 2027 已投稿 | https://github.com/BinGo620/RoGS-SLAM |
| **WildGS-SLAM** | `/data/WildGS-SLAM` | baseline（anti-dynamic，不确定性加权）；CVPR 2025 | https://github.com/GradientSpaces/WildGS-SLAM |

## 静态基座

| 仓库 | 本地路径 | 用途 | 远程 |
|---|---|---|---|
| SplaTAM | `/data/SplaTAM` | 静态 RGB-D 3DGS SLAM（对比下限） | https://github.com/spla-tam/SplaTAM |

## 其他相关（本机环境）

| 仓库 | 本地路径 | 备注 |
|---|---|---|
| DynaGSLAM_official | `/data/DynaGSLAM_official` | 动态 GS 相关 |
| dynamic-3dgs-slam-bak | `/data/dynamic-3dgs-slam-bak` | 早期动态 3DGS SLAM 备份 |
| DynaSLAM | `/data/DynaSLAM` | 经典动态 SLAM（语义 mask 路线） |
| DynOSAM | `/data/DynOSAM` | 动态 SLAM（多模态） |
| NGD-SLAM | `/data/NGD-SLAM` | 神经高斯相关 |
| Rodyn-SLAM | `/data/Rodyn-SLAM` | 动态 SLAM |
| ORB_SLAM3 / nice-slam / OneFormer / Metric3D | `/data/...` | 依赖与工具 |

## 数据资产

| 数据集 | 本地路径 | 说明 |
|---|---|---|
| Bonn / TUM / Replica | `/data/Datasets/{Bonn,TUM,Replica}` | 评测数据（详见 `data/datasets.json`） |

## 参考 skills 仓库（工程方法论来源）

| 仓库 | 本地路径 | 说明 |
|---|---|---|
| Awesome-Gaussian-Skills | `/data/Awesome-Gaussian-Skills` | 3DGS 方法目录 + skills（`data/` 单一数据源方法论来源） |
| academic-research-skills | `/tmp/ars-ref`（clone 副本） | 研究流水线方法论（deep-research / academic-paper 等） |

> 注意：`/tmp/ars-ref` 是临时 clone，长期参考请使用 `/home/cb/.zcode/skills/` 下已安装的 skills。
