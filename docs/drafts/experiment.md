# Experiment（§4 初稿）

## §4.1 Setup

**Datasets:** TUM fr3_walking_xyz (50 frames, dynamic), Bonn balloon (44 frames, dynamic), TUM fr1_desk (30 frames, static for reference).

**Metrics:** PSNR (dB), ATE RMSE (cm), SSIM, LPIPS.

**Ablation components:** (A) Metric3D depth prior (vs. random initialization), (B) Gaussian anti-dynamic (τ ablation: 1.5/2.5/3.5), (C) Frame-level anti (ablation), (D) Pose source (GT init vs. PnP init).

**All results on single RTX 2060 (6 GB), resolution 320×240.**

## §4.2 Main Results

### Gaussian Anti-Dynamic is Effective (Table 1)

On walking_xyz: anti improves PSNR from 24.0 → 25.4 dB (+1.4 dB).
On Bonn balloon: anti improves PSNR from 25.3 → 29.5 dB (+4.2 dB).
Frame-level anti degrades PSNR by 4.5 dB (19.5 dB), confirming Gaussian-level operation is necessary.

### Anti is More Valuable Under Pose Error (Table 2)

With PnP-initialized poses (ATE = 6 cm), anti provides +4.1 dB (16.2 → 20.3 dB).
With GT-initialized poses (ATE = 1.5 cm), anti provides +1.4 dB (24.0 → 25.4 dB).
The improvement ratio is nearly 3× higher under pose error.

## §4.3 Ablation Study

### τ Sensitivity

| τ | 1.5 | 2.5 | 3.5 |
|---|---|---|---|
| PSNR (dB) | 23.5 | **25.4** | [pending] |

τ=2.5 balances dynamic suppression (high enough to exclude motion artifacts) with static preservation (low enough to avoid suppressing valid background).

### Pose Source

| Source | ATE (cm) | PSNR (dB) | Trade-off |
|---|---|---|---|
| GT init + opt | 1.5 | 25.4 | Upper bound |
| PnP init + opt | 6.6* | 20.3 | Practical lower bound |

*ATE slightly worsens after optimization due to appearance-pose ambiguity in monocular setting (see §5).

## §4.4 Qualitative Results

[Figure 5: Per-frame PSNR curves — walking_xyz showing anti (25.4) consistently above baseline (24.0); Bonn balloon showing anti (29.5) vs baseline (25.3).]

[Figure 6: Residual maps before/after anti — showing reduction in dynamic artifact regions.]
