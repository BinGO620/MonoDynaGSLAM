# Limitation & Future Work（§5+§6 初稿）

## §5 Limitations

**1. Appearance-pose ambiguity in monocular setting.** When camera poses are estimated from PnP and then optimized jointly with Gaussians, the system suffers from the fundamental monocular ambiguity: different (pose, Gaussian) combinations can produce similar renders. Our experiments show ATE worsening from 6.0 → 6.6 cm after optimization — the optimizer trades small pose errors for better rendering by distorting Gaussians. This is a well-known problem in monocular SLAM, not unique to our method, but honest reporting is important.

**2. Gaussian anti is geometry-only, not semantic.** Our anti-dynamic module does not know *what* is moving — only that certain Gaussians produce inconsistent renders. This works well when motion is geometric (translation/rotation of rigid objects), but may fail for deformable motion (cloth, fluids) where Gaussians change shape without translation.

**3. face-dynamic module is incomplete.** The per-frame offset approach demonstrated feasibility (PSNR 24.3 dB, PASS) but did not exceed anti-only results. The per-frame 4D Gaussian approach (D2GSLAM style) requires significant engineering effort beyond our Phase 0 exploration.

**4. Scale dependency on Metric3D.** Metric3D's monocular depth has inherent scale ambiguity. We use its canonical-to-real scale calibration ($\alpha = f_x / 1000$), but accuracy degrades for far-field objects or unusual camera geometries.

## §6 Future Work

**1. Decoupled tracking + rendering.** Use DROID-SLAM (or VGGT-SLAM) for pose estimation separately, feed into MonGauss for rendering only. This avoids the appearance-pose ambiguity entirely.

**2. Semantic-aware Gaussian anti.** Combine our geometric anti signal with lightweight semantic priors (SAM2, depth Anything) to distinguish "moving because dynamic" from "inconsistent because of lighting/occlusion."

**3. Progressive anti→face transition.** In the first N frames, use anti-dynamic. After dynamic Gaussians are identified, switch to face-dynamic modeling (4D Gaussians) for the final reconstruction.

**4. Long-sequence deployment.** Integrate SLAM-specific components: loop closure, map management, Gaussian lifecycle (insertion/pruning). Our component-based architecture makes each addition independent.

**5. Multi-sequence formal evaluation.** Full ablation table on 5 Bonn + 3 TUM sequences with 3 seeds (90 runs), on RTX 3090 (remote jiangwenheng cluster).
