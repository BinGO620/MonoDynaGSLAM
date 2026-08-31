# 评测基准与协议

> 本仓库聚焦 **单目序列**，评测协议需与传感器设置严格对齐。

## 1. 指标

| 指标 | 用途 | 工具 |
|---|---|---|
| ATE RMSE (cm) | 轨迹精度（主指标） | `evo_ape tum` |
| RPE | 局部漂移 | `evo_rpe` |
| PSNR / SSIM / LPIPS | 渲染质量（静态地图） | 3DGS 标准评测 |
| FPS / GPU memory | 实时性（可选） | — |

## 2. 序列选择原则（单目重点）

- **静态长序列**（Bonn f2_xyz / f3_long_office_household）：检验无动态时方法不退化（no-harm gate）；
- **人物动态**（Bonn crowd / crowd2 / balloon）：最常见动态；
- **非人物动态**（Bonn mv_no_box / obox）：检验无人物先验的泛化（mask-free 卖点所在）。

## 3. 本项目实验纪律（继承 monogs-ours，永久生效）

实验预算按三阶段递进，前一阶段不过不进下一阶段：

- **Phase 0（机制自检，1-2 run）**：最短序列 × 1 seed × treatment vs control，只看机制诊断不看 ATE；
- **Phase 1（信号量级，~6 run）**：两个最短序列 × 1-3 seed × 1 臂 vs 1 control，ATE 效应 ≥6%（>2× 噪声地板）才进 Phase 2；
- **Phase 2（全矩阵判决）**：5 序列 × 3 seed × 全臂。

> 教训（exp32）：Phase 1 未过就进 Phase 2 烧了两天 3090。**好的方法在短序列上就能看到效果。**

## 4. 对照组角色

- **Public baseline**：公开 MonoGS / WildGS-SLAM 结果（外部复现），用于绝对竞争力判断；
- **Diagnostic reference**：历史原型（如 tagged V1 `v1.0-maskboth-densekf`），定位已知成败机制；
- **Experimental control**：针对已批准 hypothesis 冻结的 module-off 配置，用于因果归因。

## 5. GPU 预算

- 本地 RTX 2060 6GB（开发/机制自检）
- 远程双 RTX 3090（正式实验；Bonn balloon 440 帧 ~40min/run）
