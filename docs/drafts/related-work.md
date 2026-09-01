# Related Work（初稿）

## 3DGS-Based SLAM

MonGS (CVPR 2024) first demonstrated that 3D Gaussians can serve as the sole scene representation for monocular SLAM, achieving real-time tracking and mapping. Following works improved scalability (Splat-SLAM, MonoGS++, GS-SLAM) and added dense depth (SplaTAM). These systems assume **static scenes** and degrade significantly when moving objects are present — e.g., MonoGS achieves only 86.5 cm ATE on the Bonn crowd sequence (RoGS-SLAM baseline data).

## Dynamic 3DGS SLAM: Anti-Dynamic

Anti-dynamic methods detect and suppress dynamic content to maintain static reconstruction quality. **WildGS-SLAM** (CVPR 2025) introduces DINOv2 features + uncertainty MLP for monocular RGB, while **Dy3DGS-SLAM** (ICRA 2025) fuses optical flow and depth masks. **GGD-SLAM** (ICRA 2026) proposes a generalizable motion model without semantic annotations. **DAGS-SLAM** (arXiv 2026) introduces spatiotemporal motion probability with on-demand semantic scheduling for efficiency. **DL-SLAM** (arXiv 2026) uses dual-level probability (pixel + instance) for dynamic filtering.

However, all monocular methods rely on **learned priors** (DINOv2, metric depth, optical flow networks) — no purely geometric anti-dynamic approach exists for monocular input. Our work fills this gap.

## Dynamic 3DGS SLAM: Face-Dynamic

Face-dynamic methods explicitly model moving objects. **D2GSLAM** (arXiv 2025) uses static 3D + dynamic 4D Gaussians but requires RGB-D. **Flow4DGS-SLAM** (arXiv 2026) uses optical flow-guided 4D Gaussians, also RGB-D. **RU4D-SLAM** (CVPR 2026 Findings) adds uncertainty reweighting to 4D reconstruction. **DynaGSLAM** (WACV 2026) is the only monocular face-dynamic method but depends on SAM2+RAFT+DynoSAM (heavy).

**Key gap:** All efficient 4D methods are RGB-D; monocular face-dynamic lacks lightweight variants.

## Gap Analysis

| Aspect | Existing | MonGauss |
|---|---|---|
| Monocular anti-dynamic | Relies on learned priors | Pure geometric (rendering residuals) |
| Pipeline modularity | Tightly coupled | Independent components (gsplat/Metric3D/PnP) |
| Frame-level vs Gaussian-level | Frame-level loss (ineffective) | Gaussian-level opacity (effective) |
| Anti→face transition | No bridge exists | Modular (shared dynamic markers) |

## Note

This is a preliminary draft. The full Related Work will be completed using `3dgs-paper-reader` and `nature-literature-pipeline` skills after downloading and reading the relevant papers from `papers/pdf/`.
