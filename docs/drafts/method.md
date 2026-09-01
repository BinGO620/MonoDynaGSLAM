# Method（§2+§3 初稿）

## §2 System Overview: Modular MonGauss Pipeline

MonGauss decomposes monocular dynamic 3DGS into three independent components (Figure 2):

**Component A — Monocular Depth Prior (Metric3D).** For each input RGB frame, Metric3D predicts a per-pixel metric depth map. These depth maps are back-projected using camera intrinsics and estimated poses to initialize a 3D Gaussian point cloud with per-pixel RGB color. This is a one-time offline computation, independent of downstream rendering.

**Component B — Pose Estimation.** MonGauss accepts any pose source: (i) PnP-RANSAC with Metric3D depth for initialization, (ii) DROID-DBA for learned dense tracking, or (iii) ground-truth poses for ablation. Pose optimization during rendering (via gsplat's differentiable rasterization gradient) further refines initial poses. The modular design allows upgrading the pose estimator without changing other components.

**Component C — Gaussian Splatting Renderer (gsplat).** The independent gsplat library (v1.5.3) provides differentiable tile-based rasterization with analytic gradients for Gaussian parameters and camera poses. It handles SH-based color, anisotropic covariance, and depth rendering — all needed for our anti-dynamic module.

## §3 Gaussian-Level Anti-Dynamic (Core Contribution)

### Problem

In dynamic environments, moving objects create Gaussian primitives that are consistent in only a subset of frames. When all frames contribute equally to Gaussian optimization, these dynamic Gaussians receive incorrect gradient signals from conflicting observations, degrading reconstruction quality for both the moving objects and the static background.

### Key Insight

**Suppress dynamic content at the Gaussian level, not the frame level.** Frame-level loss down-weighting simply ignores corrupted frames — but those frames contain valid static background that should still contribute. Gaussian-level suppression selectively reduces the influence of specific Gaussians that exhibit multi-frame inconsistency.

### Mechanism

**Step 1: Multi-frame rendering consistency check.** At periodic intervals (every τ anti-period iterations), MonGauss renders each frame independently and computes per-pixel residual:

$$r_i = |I_i^{pred} - I_i^{gt}|$$

For each frame, the median residual $m_i$ and standard deviation $\sigma_i$ define a threshold: pixels with residual $> m_i + \tau \cdot \sigma_i$ are marked as "rendering-inconsistent" — indicating they likely belong to dynamic objects or regions where Gaussians cannot simultaneously explain all observations.

**Step 2: Gaussian opacity reduction.** Gaussians that contribute to high-residual regions are suppressed by multiplying their opacity values by a decay factor:

$$\alpha_g \leftarrow \alpha_g \cdot \max(0.9, 1.0 - \beta \cdot f_{dyn})$$

where $f_{dyn}$ is the fraction of rendering-inconsistent pixels in the current frame, and $\beta = 0.2$ is a step size parameter. The floor of 0.9 prevents complete suppression (hard removal), maintaining partial evidence from partially-dynamic Gaussians.

### Why Frame-Level Down-Weighting Fails

We experimentally showed (Table 1) that reducing the per-frame loss weight for high-residual frames **degrades** PSNR by 4.5 dB — because it simply ignores valid static background content in those frames. The Gaussian-level approach selectively suppresses only the problematic Gaussians while preserving valid static contributions.

### Comparison to Existing Anti-Dynamic Methods

| Method | Anti signal | Level | Learned prior | Monocular |
|---|---|---|---|---|
| WildGS-SLAM | DINOv2 uncertainty | Gaussian | Yes (DINOv2+MLP) | Yes |
| Dy3DGS-SLAM | OF+depth mask | Frame | Yes (flow+depth nets) | Yes |
| GGD-SLAM | Generalizable motion | Gaussian | Yes (attention+LSTM) | Yes |
| DL-SLAM | Dual-level probability | Pixel+instance | Yes (semantic+geo) | Yes |
| **MonGauss (ours)** | **Rendering residual** | **Gaussian** | **No (pure geometric)** | **Yes** |

Our method is the only **geometry-only, learned-prior-free** anti-dynamic approach for monocular 3DGS.

## §3.x Quantitative Results (Summary)

| Init | Sequence | Config | PSNR | ATE |
|---|---|---|---|---|
| GT | fr1_desk | Baseline | 24.4 | 1.4 cm |
| GT | fr3_walking | Baseline | 24.0 | — |
| GT | fr3_walking | Anti (τ=2.5) | **25.4** | 1.5 cm |
| GT | Bonn balloon | Baseline | 25.3 | — |
| GT | Bonn balloon | Anti (τ=2.5) | **29.5** | — |
| GT | fr3_walking | Frame-level anti | 19.5 | — |
| **PnP** | **fr3_walking** | **Baseline** | **16.2** | **6.0 cm** |
| **PnP** | **fr3_walking** | **Anti (τ=2.5)** | **20.3** | **6.0 cm** |

Key finding: Gaussian anti provides **+4.1 dB** from PnP init vs **+1.4 dB** from GT init — anti-dynamic is more valuable when tracking is imperfect.
