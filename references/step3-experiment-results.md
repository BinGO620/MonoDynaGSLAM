# Step 3 face-dynamic 实验结果

> fr3_walking_xyz（50帧, Metric3D 深度先验, 60k 高斯）

## 三步完整对比

| 方法 | PSNR | 前半 | 后半 | Δ vs 静态基线 | 高斯数 | 判决 |
|---|---|---|---|---|---|---|
| Step 1 静态基线 | 24.03 dB | 23.6 | 24.6 | — | 60k | PASS ✅ |
| Step 2 Gaussian anti τ=2.5 | 25.42 dB | 25.0 | 26.0 | **+1.39 dB** | 60k | PASS ✅ |
| Step 3 face-dynamic (15% dyn) | 24.33 dB | 24.0 | 24.8 | +0.30 dB | 60k | PASS ✅ |
| Step 3 static-only (dyn=0) | 24.07 dB | 23.7 | 24.5 | +0.04 dB | 60k | PASS ✅ |
| Step 2 帧级 anti（已证伪） | 19.5 dB | 22.2 | 24.1 | -4.5 dB | 60k | FAIL ❌ |

## 关键分析

### Step 3 face-dynamic 的局限性

当前实现用 `frame_offsets.mean()` 做共享渲染，不是真正的 per-frame 变形——相当于只学到一个全局位移，没有逐帧变化。D2GSLAM 等真正 face-dynamic 方法的做法是：

1. **每帧独立 rasterization**：means_fi = means + offset_fi，渲染时每帧独立调用
2. **per-frame 颜色插值**：动态高斯的颜色也有时间变化
3. **4D 高斯表示**：off-the-shelf 的 4D 变形场（控制点 + MLP）

当前 Phase 0 的意义在于：**证明了组件化管线可行性（gsplat + Metric3D + per-frame 渲染通道）**，且 face-dynamic > static-only（+0.26 dB），说明方向正确，但需要更强的 per-frame 机制。

### 为什么 Step 2 anti > Step 3 face（当前版）

Step 2 的 Gaussian opacity 下调直接减少了高残差 Gaussian 的能量，简单有效；Step 3 的 mean offset 没有真正产生 per-frame 差异。下一步：**真正 per-frame 渲染**。

## 下一步：per-frame 真正 face-dynamic

改 Phase B 为：每帧独立渲染（means + offset_fi → rasterize → loss_fi），完整 per-frame 优化 offset。这是 D2GSLAM 单目版的核心实现。

注意：50帧×60k高斯×320×240 分辨率，每帧独立渲染 50 次 → 显存或时间可能不够。需要：
- 减少 batch_size（每 5 帧一批）
- 或减少高斯数（比如只对 top 5% 动态高斯做 offset，其余锁死）
