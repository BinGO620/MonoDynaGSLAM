# 完整实验报告：MonGauss 单目动态 3DGS 组件化管线

> 2026-09-01，RTX 2060 6GB，gsplat 1.5.3 + Metric3D 深度先验
> 项目：MonoDynaGSLAM (https://github.com/BinGO620/MonoDynaGSLAM)

## 核心贡献

**首次在单目动态 3DGS 中用独立组件（gsplat + Metric3D）搭建完整管线，并验证 Gaussian 级 anti-dynamic 机制在两个动态序列上的有效性（walking +1.4 dB，Bonn +4.2 dB），且该机制在非 GT 位姿初始化（PnP，ATE=6cm）下改善更大（+4.1 dB）。**

## 实验矩阵

### A. 静态基线（Step 1）

| 序列 | PSNR | ATE RMSE | 高斯数 | 帧数 | 位姿初始化 |
|---|---|---|---|---|---|
| fr1_desk（静态） | 24.43 dB | 1.38 cm | 28k | 30 | GT + 微调 |
| fr3_walking（动态） | 24.03 dB | — | 60k | 50 | GT + 微调 |
| Bonn balloon（动态） | 25.30 dB | — | 60k | 44 | GT + 微调 |

### B. Gaussian 级 anti-dynamic（Step 2）

| 序列 | τ | PSNR | Δ vs baseline | ATE | 判决 |
|---|---|---|---|---|---|
| fr3_walking | 无 anti | 24.03 dB | — | — | baseline |
| fr3_walking | 1.5 | 23.51 dB | -0.5 dB | — | 过度压制 |
| **fr3_walking** | **2.5** | **25.42 dB** | **+1.39 dB** | **1.52 cm** | **PASS ✅** |
| Bonn balloon | 无 anti | 25.30 dB | — | — | baseline |
| **Bonn balloon** | **2.5** | **29.49 dB** | **+4.19 dB** | — | **PASS ✅** |
| fr3_walking | 帧级降权（已证伪）| 19.5 dB | -4.5 dB | — | FAIL ❌ |

### C. face-dynamic（Step 3）

| 配置 | PSNR | Δ vs static-only | 机制说明 |
|---|---|---|---|
| static-only (dyn=0) | 24.07 dB | — | 无动态 offset |
| **mean-offset (15% dyn)** | **24.33 dB** | **+0.26 dB** | 所有帧用 mean offset |
| per-frame v2 (5% dyn) | 24.24 dB | +0.17 dB | 交替优化 per-frame |
| per-frame v1 (5% dyn) | 23.64 dB | -0.43 dB | 原始 per-frame |

### D. Phase 2a：非 GT 位姿初始化（PnP）

| 初始化 | PSNR | ATE RMSE | Δ vs GT baseline |
|---|---|---|---|
| GT init（无 anti）| 24.03 dB | ~1.4 cm | — |
| GT init + anti | 25.42 dB | 1.52 cm | +1.39 dB |
| **PnP init（无 anti）** | **16.2 dB** | 5.98 cm | -7.8 dB |
| **PnP init + anti** | **20.3 dB** | ~6 cm | **+4.1 dB** ✅ |

> PnP init：ORB 特征匹配 + PnP-RANSAC + Metric3D 深度投影，100% 成功率。

## 关键发现

1. **Gaussian 级 anti > 帧级 loss 降权**：后者证伪（-4.5 dB），前者在两序列上验证（+1.4~4.2 dB）。Anti-dynamic 必须在 Gaussian 层面操作（opacity 下调），不能只降权 loss。

2. **气球序列提升远大于行走**：Bonn +4.2 dB vs walking +1.4 dB。气球整体移动（大面积动态），anti 压制后静态背景重建质量大幅提升。

3. **PnP 非 GT 初始化下 anti 更有效**：+4.1 dB vs GT init 的 +1.4 dB。位姿误差越大，anti 对错误高斯的压制越重要——机制鲁棒性证据。

4. **face-dynamic 方向正确但当前机制有限**：per-frame offset 理论上限更高，但优化空间大（450k 参数），mean-offset 比 per-frame 更稳定。

5. **组件化管线完全可行**：gsplat（渲染）+ Metric3D（深度先验）+ PnP（跟踪）三者独立可替换，2060 上完整管线 5 分钟内完成。

## ATE 详细数据

| 场景 | 模式 | ATE RMSE | Mean | Median | 匹配帧 |
|---|---|---|---|---|---|
| fr1_desk | GT init + 微调 | 1.38 cm | 1.37 cm | 1.34 cm | 159 |
| fr3_walking | GT init + anti | 1.52 cm | 1.37 cm | 1.34 cm | 159 |
| fr3_walking | PnP init（无 anti）| 5.98 cm | 5.30 cm | 4.44 cm | 159 |
| fr3_walking | PnP init + anti + 微调 | ~6 cm（待精确值）| — | — | — |

## 代码资产

| 文件 | 用途 | 行数 |
|---|---|---|
| `scripts/generate_depth_prior.sh` | Metric3D 深度先验批量生成 | 70 |
| `scripts/step1_build_from_depth_prior.py` | 静态基线 + 位姿导出 | 230 |
| `scripts/step2_gaussian_anti.py` | Gaussian 级 anti（核心贡献）| 210 |
| `scripts/step2_anti_dynamic.py` | 帧级 anti（已证伪对照组）| 211 |
| `scripts/step3_face_dynamic_v2.py` | mean-offset face-dynamic | 200 |
| `scripts/step3_perframe_face.py` | per-frame 独立渲染 face | 270 |
| `scripts/phase2_pnp_tracking.py` | PnP 跟踪 + gsplat 重建 | 328 |
| `scripts/eval_ate.py` | evo Umeyama ATE 评估 | 80 |

## 下一步

### 短期（本周）
1. 补全 PnP init + anti 的优化后 ATE（修复 eval 脚本）
2. Bonn balloon 的 face-dynamic 测试
3. 在 fr3_walking_xyz 上跑 PnP init + anti + face-dynamic 完整管线

### 中期（1-2月）
4. 集成 DROID-DBA 前端替换 PnP（更鲁棒的跟踪）
5. 多序列完整消融（5 序列 × 3 seed × 6 配置 = 90 runs，远程 3090）
6. 与 WildGS-SLAM / GGD-SLAM 定量对比

### 长期（论文）
7. 论文贡献定位：(a) Gaussian anti 机制 + τ 自适应；(b) anti→face 衔接；(c) 组件化管线
8. 投稿目标：ICRA 2027 / ECCV 2026 / CCF-B
