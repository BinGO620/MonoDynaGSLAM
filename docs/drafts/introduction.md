# Introduction（初稿）

## 草稿

3D Gaussian Splatting (3DGS) has emerged as a powerful scene representation for real-time novel view synthesis, achieving unprecedented rendering quality and speed. However, its integration into **Simultaneous Localization and Mapping (SLAM)** systems operating in **dynamic environments** with **monocular input** remains challenging: moving objects corrupt both camera tracking and scene reconstruction.

Existing approaches fall into two categories: **anti-dynamic** methods that identify and suppress dynamic content (WildGS-SLAM, Dy3DGS-SLAM, GGD-SLAM), and **face-dynamic** methods that explicitly model moving objects with 4D representations (D2GSLAM, Flow4DGS-SLAM). However, (1) all monocular anti-dynamic methods rely on learned priors (DINOv2, metric depth networks, optical flow networks) — none achieve purely geometric anti-dynamic processing, (2) face-dynamic 4D methods are predominantly RGB-D, with no efficient monocular variant, and (3) the processing pipeline is tightly coupled: single methods cannot be independently replaced or upgraded.

We present **MonGauss**, a modular monocular dynamic 3DGS pipeline that decouples three components: a differentiable Gaussian rasterizer (gsplat), a monocular metric depth prior (Metric3D), and a Gaussian-level anti-dynamic module. Our key insight is that dynamic objects must be suppressed **at the Gaussian level** — reducing per-Gaussian opacity based on multi-frame rendering consistency — rather than at the frame level by down-weighting reconstruction loss. We show this Gaussian-level approach works significantly better when camera poses are not ground-truth: on the TUM fr3_walking_xyz benchmark, it improves PSNR by **+4.1 dB** from a PnP-initialized trajectory (ATE=6 cm), compared to +1.4 dB from a ground-truth-initialized trajectory. This robustness to pose error makes MonGauss practically relevant for real SLAM deployment, where initial poses are never perfect.

We further explore a **anti→face-dynamic** bridge: the dynamic Gaussians identified by our anti-dynamic module serve directly as input to a lightweight 4D offset module, demonstrating a modular pathway from suppression to explicit modeling without redundant detection.

**Contributions:**

1. **Gaussian-level anti-dynamic** (§3): Per-Gaussian opacity reduction guided by multi-frame rendering residuals, with extensive ablation showing frame-level loss down-weighting is ineffective.

2. **Modular pipeline** (§2): Independent components (gsplat rendering, Metric3D depth prior, PnP/DBA tracking) that can be upgraded separately — demonstrated by replacing PnP with DROID-DBA without changing the rendering or anti-dynamic modules.

3. **Anti→face bridge** (§4): A modular transition from anti-dynamic suppression to face-dynamic 4D offset modeling, using the same dynamic Gaussian markers for both.

**Note:** This is a preliminary draft — see `references/full-experiment-report.md` for the latest data, `references/paper-narrative.md` for the complete narrative framework, and `references/experiment-plan.md` for the full experimental design.
