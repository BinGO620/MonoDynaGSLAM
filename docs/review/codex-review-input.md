# 模拟审稿材料（诚实版，含实现细节披露）

> 提交给 codex 模拟审稿。请以 CVPR/ICRA 审稿人视角严格评审。
> 声明：以下全部如实，包括实现与声称不符之处。

## 1. 方法声称（论文叙事）

**MonGauss**：单目动态 3DGS 组件化管线，核心贡献是"Gaussian 级 anti-dynamic"：
- 渲染残差 → 逐高斯 opacity 下调（声称 Gaussian 级操作）
- 与帧级 loss 降权对比，证明帧级失效（19.5dB vs 25.4dB）
- 声称 anti 在非 GT 初始化下改善更大（PnP init +4.1dB vs GT init +1.4dB）

## 2. 实际实现（⚠️ 与声称的差距）

```python
@torch.no_grad()
def anti_update():
    # 每 200 iter 调用一次
    rends = render(all frames)
    residual = |pred - gt|.mean(per pixel)   # (N,H,W)
    for fi in frames:
        med = residual[fi].median(); std = ...
        dyn_frac = (residual[fi] > med + tau*std).mean()  # 该帧动态像素比例（标量！）
        if dyn_frac > 0.01:
            opacities.data.mul_(max(0.9, 1.0 - 0.02*dyn_frac*10))  # ⚠️ 全局乘！
```

**关键问题**：
1. `opacities.data.mul_(decay)` 是**全局衰减所有高斯**，不是选择性压制动态高斯。dynamic_frac 只是一个标量调制衰减幅度。
2. 没有把残差定位到具体高斯（没有用 gsplat 的 Gaussian ID rasterization 或 alpha 反传）。
3. "Gaussian 级"的声称在当前实现中**不成立**——实际是"帧级信号调制的全局衰减"。
4. 效果来源可能只是"降低全体 opacity → 隐式正则化"，而非动态高斯识别。

## 3. 实验数据（全部真实）

### GT pose 初始化 + 微调（500 iter 或 3000 iter）

| 序列 | 配置 | PSNR | ATE |
|---|---|---|---|
| fr1_desk (静态) | baseline | 24.43 dB | 1.38 cm |
| fr3_walking_xyz | baseline | 24.03 dB | 1.52 cm |
| fr3_walking_xyz | anti τ=2.5 | 25.42 dB | — |
| fr3_walking_xyz | anti τ=1.5 | 23.51 dB | — |
| Bonn balloon | baseline | 25.30 dB | — |
| Bonn balloon | anti τ=2.5 | 29.49 dB | — |
| fr3_walking_xyz | 帧级 loss 降权 | 19.54 dB | — |

### PnP 初始化（非 GT）

| 序列 | 配置 | PSNR | ATE |
|---|---|---|---|
| fr3_walking_xyz | PnP only（无 anti，无优化）| 16.2 dB | 5.98 cm |
| fr3_walking_xyz | PnP + anti + 位姿微调 | 20.3 dB | 6.6 cm（恶化 0.6）|
| Bonn balloon | PnP only | — | 2.57 cm |
| Bonn balloon | PnP + anti + 微调 | 27.0 dB | 3.4 cm（恶化 0.8）|

### 实验设置

- 分辨率 320×240（原始 640×480 的 1/2）
- 50 帧（walking）/ 44 帧（Bonn），从序列开头截取
- 60k 高斯（Bonn PnP 用 40k）
- Metric3D vit_small 深度先验（离线预计算）
- 单 seed，无重复
- 2060 6GB

## 4. 帧级降权对照的实现

```python
# 帧级降权（证伪对照）
per_frame_mse = mse(pred, gt, per frame)
median_psnr = per_frame_psnr.median()
weight = sigmoid((per_frame_psnr - median_psnr + tau) / 0.5)
recon_loss = (per_frame_mse * weight.detach()).mean()
```

⚠️ 这个对照把高残差帧整体降权。审稿人可能说"你的帧级基线做弱了"（比如更好的做法是 per-pixel 帧内加权，或直接 mask 掉高残差像素）。

## 5. 我们自己已知的疑点（请重点审查）

1. **全局 opacity 衰减 vs 选择性压制的区分实验没做**：全局衰减也可能带来同样提升（作为隐式正则化），那"Gaussian 级"叙事就是错的。
2. **对照组公平性**：帧级降权是不是故意做弱？
3. **没有 per-pixel mask 对照**：最直接的对照是"把高残差像素的 loss mask 掉"（Dy3DGS 类方法的标准做法），我们没做这个对照。
4. **单 seed、单分辨率、短序列**：50 帧 320×240，结论的统计强度弱。
5. **PSNR 评估的是重建训练帧自身**（没有 held-out novel view），PSNR 提升可能只是过拟合差异。
6. **与 WildGS/Gassidy 无同协议数字对比**。
7. Gassidy (ICRA'25, RGB-D) 已经用"渲染 loss flow 过滤动态"，我们的"渲染残差过滤"在单目下做同样的事——novelty 是"单目 + 无学习先验"还是机制本身？

## 6. 请 codex 输出

1. 模拟 3 个审稿人（R1 严格、R2 方法论、R3 领域专家）的评分和意见（Reject/Borderline/Accept）
2. 指出当前声称中最站不住脚的点，按致命程度排序
3. 如果要救活"Gaussian 级"叙事，最少需要补哪些实验（具体到对照组设计）
4. novelty 的真实定位：这个工作最诚实的一句话定位是什么
