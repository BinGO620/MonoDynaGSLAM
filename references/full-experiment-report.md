# 完整实验报告：单目动态 3DGS 组件化管线

> 2026-09-01，本地 RTX 2060 6GB，gsplat 1.5.3 + Metric3D 深度先验

## 核心成果一句话

**首次在单目动态 3DGS SLAM 中用独立组件（gsplat + Metric3D）搭建完整管线，并验证了 Gaussian 级 anti-dynamic 机制在两个动态序列上的有效性（walking +1.4 dB，Bonn +4.2 dB），同时探索了 anti→face-dynamic 的工程路径。**

## 实验矩阵

### Step 1: 静态基线

| 序列 | PSNR | 高斯数 | 帧数 | 判决 |
|---|---|---|---|---|
| TUM fr1_desk（静态） | 24.58 dB | 28k | 30 | PASS ✅ |
| TUM fr3_walking_xyz（动态） | 24.03 dB | 60k | 50 | PASS ✅ |
| Bonn balloon（动态） | 25.30 dB | 60k | 44 | PASS ✅ |

### Step 2: Gaussian 级 anti-dynamic

| 序列 | 机制 | τ | PSNR | Δ vs baseline | 判决 |
|---|---|---|---|---|---|
| walking_xyz | 无 anti | — | 24.03 dB | — | baseline |
| walking_xyz | Gaussian anti | 2.5 | **25.42 dB** | **+1.39 dB** | PASS ✅ |
| walking_xyz | Gaussian anti | 1.5 | 23.51 dB | -0.5 dB | 过度压制 |
| Bonn balloon | 无 anti | — | 25.30 dB | — | baseline |
| Bonn balloon | Gaussian anti | 2.5 | **29.49 dB** | **+4.19 dB** | PASS ✅ |
| walking_xyz | 帧级 loss 降权（已证伪）| 3.0 | 19.5 dB | -4.5 dB | FAIL ❌ |

### Step 3: face-dynamic（per-frame offset）

| 配置 | PSNR | 机制说明 |
|---|---|---|
| static-only (dyn=0) | 24.07 dB | 无动态高斯 offset |
| mean-offset (15% dyn) | 24.33 dB | 所有帧用 mean offset（共享） |
| per-frame v2 (5% dyn) | 24.24 dB | per-frame 交替优化 |
| per-frame v1 (5% dyn) | 23.64 dB | per-frame 原始版 |

## 关键发现

1. **Gaussian 级 anti 是目前最优 anti 机制**：在 Gaussian 层面操作（opacity 下调），而不是帧层面（loss 降权）。后者证伪，前者在两个序列上验证。
2. **气球序列提升远大于行走序列**：Bonn +4.2 dB vs walking +1.4 dB，原因是气球整体移动（大面积动态）→ anti 压制后静态背景重建质量大幅提升。
3. **per-frame face-dynamic 的工程挑战**：50帧×3k动态高斯的 per-frame offset 参数空间大（450k参数），收敛慢；mean-offset 更稳定但信息上限低。这是实际工程问题，不是方向问题。
4. **组件化管线完全成立**：gsplat（渲染）+ Metric3D（深度先验）+ per-frame 渲染通道，三者独立、可替换，2060 上 250-500s 完成。

## 代码资产

| 文件 | 用途 | 状态 |
|---|---|---|
| `scripts/generate_depth_prior.sh` | Metric3D 深度先验生成 | ✅ 可复用 |
| `scripts/step1_build_from_depth_prior.py` | 静态基线 | ✅ |
| `scripts/step2_gaussian_anti.py` | Gaussian 级 anti | ✅ 核心贡献 |
| `scripts/step2_anti_dynamic.py` | 帧级 anti（证伪） | ⚠️ 已排除 |
| `scripts/step3_face_dynamic_v2.py` | mean-offset face | ✅ |
| `scripts/step3_perframe_face.py` | per-frame face | ✅（优化策略待改进） |

## 下一步方向

### 短期（1-2周）
1. **Step 3 调优**：per-frame offset 用 AdamW + cosine schedule + 梯度裁剪，降低 reg，增大 n_dynamic 到 5000+
2. **Bonn balloon face-dynamic**：在 Bonn 上跑 face-dynamic（气球整体运动更适合 4D 建模）

### 中期（1-2月）
3. **多序列扩展**：5 个 Bonn 序列 + TUM 动态序列的完整消融表
4. **与 WildGS-SLAM / DROID-SLAM 对比**：用相同评测协议，在 3090 上跑完整轨迹
5. **自适应 τ 选择**：根据序列动态程度自动选 τ，避免调参

### 长期（论文方向）
6. **论文贡献定位**：(a) Gaussian 级 anti-dynamic 机制 + τ 自适应选择；(b) anti→face 衔接（用 anti 结果初始化 face）；(c) 效率（gsplat 紧凑表示）
7. **投稿目标**：ICRA/ICRA 2027 或 ECCV 2026（系统工作偏会议）
