#!/usr/bin/env python3
"""Phase 2d：PnP init → gsplat + Gaussian anti → 精确 ATE。

完整闭环：PnP 跟踪 → Metric3D 反投影点云 → gsplat + anti → 位姿优化 → ATE。

用法：
  python3 scripts/phase2d_pnp_full.py \
      --pnp results/phase2_pnp_anti/pnp_trajectory.txt \
      --prior results/depth_prior_walking \
      --iters 2500 --anti --tau 2.5 --out results/phase2d_pnp_full
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def load_tum_poses(path):
    ts_l, poses = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip(): continue
        p = line.split(); t = float(p[0]); T = np.eye(4)
        T[:3, 3] = [float(p[1]), float(p[2]), float(p[3])]
        qx, qy, qz, qw = [float(p[i]) for i in [4, 5, 6, 7]]
        T[0, 0] = 1 - 2*(qy**2+qz**2); T[0, 1] = 2*(qx*qy-qz*qw); T[0, 2] = 2*(qx*qz+qy*qw)
        T[1, 0] = 2*(qx*qy+qz*qw); T[1, 1] = 1 - 2*(qx**2+qz**2); T[1, 2] = 2*(qy*qz-qx*qw)
        T[2, 0] = 2*(qx*qz-qy*qw); T[2, 1] = 2*(qy*qz+qx*qw); T[2, 2] = 1 - 2*(qx**2+qy**2)
        ts_l.append(t); poses.append(T)
    return np.array(ts_l), np.array(poses)


def save_tum_trajectory(path, poses, timestamps):
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    for T, ts in zip(poses, timestamps):
        q = T[:3, :3]; tx, ty, tz = T[:3, 3]
        trace = np.trace(q)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w, x, y, z = 0.25/s, (q[2,1]-q[1,2])*s, (q[0,2]-q[2,0])*s, (q[1,0]-q[0,1])*s
        else:
            if q[0,0] > q[1,1] and q[0,0] > q[2,2]:
                s = 2*np.sqrt(1+q[0,0]-q[1,1]-q[2,2])
                w, x, y, z = (q[2,1]-q[1,2])/s, .25*s, (q[0,1]+q[1,0])/s, (q[0,2]+q[2,0])/s
            elif q[1,1] > q[2,2]:
                s = 2*np.sqrt(1+q[1,1]-q[0,0]-q[2,2])
                w, x, y, z = (q[0,2]-q[2,0])/s, (q[0,1]+q[1,0])/s, .25*s, (q[1,2]+q[2,1])/s
            else:
                s = 2*np.sqrt(1+q[2,2]-q[0,0]-q[1,1])
                w, x, y, z = (q[1,0]-q[0,1])/s, (q[0,2]+q[2,0])/s, (q[1,2]+q[2,1])/s, .25*s
        lines.append(f"{ts} {tx} {ty} {tz} {x} {y} {z} {w}")
    Path(path).write_text("\n".join(lines))


def compute_ate(est_poses, est_ts, gt_path, max_diff=0.02):
    gt_ts, gt_poses = load_tum_poses(gt_path)
    est_matched, gt_matched = [], []
    for gt_t, gt_p in zip(gt_ts, gt_poses):
        idx = np.argmin(np.abs(est_ts - gt_t))
        if np.abs(est_ts[idx] - gt_t) < max_diff:
            est_matched.append(est_poses[idx][:3, 3])
            gt_matched.append(gt_p[:3, 3])
    if len(est_matched) < 3:
        return 0, 0, 0, 0
    est_m, gt_m = np.array(est_matched), np.array(gt_matched)
    sc, dc = est_m.mean(0), gt_m.mean(0)
    H = (est_m - sc).T @ (gt_m - dc)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[-1] *= -1; R = Vt.T @ U.T
    est_aligned = (R @ est_m.T).T + dc - R @ sc
    errors = np.linalg.norm(est_aligned - gt_m, axis=1)
    return np.sqrt(np.mean(errors**2)), np.mean(errors), np.median(errors), len(est_m)


def voxel_downsample(pts, colors, voxel=0.02, max_points=60000):
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx, inverse, counts = np.unique(keys, axis=0, return_index=True,
                                         return_inverse=True, return_counts=True)
    n = len(idx)
    sp = np.zeros((n, 3)); sc = np.zeros((n, 3))
    np.add.at(sp, inverse, pts); np.add.at(sc, inverse, colors)
    mp, mc = sp / counts[:, None], sc / counts[:, None]
    if n > max_points:
        sel = np.random.choice(n, max_points, replace=False)
        mp, mc = mp[sel], mc[sel]
    return mp, mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pnp", required=True, help="PnP 位姿轨迹路径 (TUM 格式)")
    ap.add_argument("--prior", required=True, help="深度先验目录")
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--max_points", type=int, default=60000)
    ap.add_argument("--anti", action="store_true")
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--pose_lr", type=float, default=5e-5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        tag = "pnp_anti" if args.anti else "pnp"
        args.out = f"results/phase2d_{tag}"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    device = "cuda"
    meta = json.loads((Path(args.prior) / "meta.json").read_text())
    depths = np.load(Path(args.prior) / "depths.npy")
    seq_dir = Path(meta["seq_dir"])
    K_orig = np.array(meta["K"])
    n_frames = len(meta["timestamps"])
    gt_path = str(seq_dir / "groundtruth.txt")

    W, H = args.res, int(args.res * 0.75)
    scale = W / 640.0
    K = K_orig.copy(); K[:2] *= scale
    Ks = torch.from_numpy(K).float().to(device).unsqueeze(0).expand(1, n_frames, 3, 3).contiguous()

    # 加载 PnP 位姿
    pnp_ts, pnp_poses = load_tum_poses(args.pnp)
    viewmats_pnp = torch.tensor(np.linalg.inv(pnp_poses), dtype=torch.float32).unsqueeze(0).to(device)

    # PnP ATE
    pnp_ate, _, _, _ = compute_ate(pnp_poses, pnp_ts, gt_path)
    print(f"PnP init ATE: {pnp_ate*100:.1f} cm")

    # 加载图像
    images = []
    for ts, rel in zip(meta["timestamps"], meta["image_paths"]):
        img = cv2.cvtColor(cv2.imread(str(seq_dir / rel)), cv2.COLOR_BGR2RGB)
        images.append(torch.from_numpy(cv2.resize(img, (W, H)).transpose(2, 0, 1)).float() / 255.0)
    images = torch.stack(images).to(device)

    # 反投影点云
    fx_o, fy_o, cx_o, cy_o = K_orig[0, 0], K_orig[1, 1], K_orig[0, 2], K_orig[1, 2]
    all_pts, all_cols = [], []
    for i in range(n_frames):
        dep = depths[i]; yi, xi = np.mgrid[0:dep.shape[0]:4, 0:dep.shape[1]:4]
        d = dep[::4, ::4]; v = (d > 0.2) & (d < 8)
        if v.sum() == 0: continue
        yc, xc, dc = yi[v], xi[v], d[v]
        pts_cam = np.stack([(xc - cx_o)/fx_o*dc, (yc - cy_o)/fy_o*dc, dc], axis=-1)
        img_o = cv2.cvtColor(cv2.imread(str(seq_dir/meta["image_paths"][i])), cv2.COLOR_BGR2RGB)
        cols = img_o[yc, xc].astype(np.float32) / 255.0
        pts_w = (pnp_poses[i][:3, :3] @ pts_cam.T).T + pnp_poses[i][:3, 3]
        all_pts.append(pts_w); all_cols.append(cols)
    pts, cols = voxel_downsample(np.concatenate(all_pts), np.concatenate(all_cols), 0.02, args.max_points)
    n_g = len(pts)
    print(f"高斯数: {n_g} | 帧: {n_frames}")

    pts_t = torch.from_numpy(pts).float().to(device)
    cols_t = torch.from_numpy(cols).float().to(device)
    means = pts_t.unsqueeze(0).requires_grad_(True)
    quats = torch.zeros(1, n_g, 4, device=device); quats[..., 0] = 1.0; quats.requires_grad_(True)
    scales = torch.full((1, n_g, 3), 0.02, device=device).requires_grad_(True)
    opacities = torch.full((1, n_g), 0.9, device=device).requires_grad_(True)
    colors = cols_t.unsqueeze(0).clone().requires_grad_(True)
    viewmats_param = viewmats_pnp.clone().requires_grad_(True)

    from gsplat import rasterization

    @torch.no_grad()
    def anti_update():
        if not args.anti: return
        rends, _, _ = rasterization(means, quats, scales, opacities.detach(), colors,
                                    viewmats_param.detach(), Ks, W, H)
        residual = (rends.squeeze(0).permute(0, 3, 1, 2) - images).abs().mean(dim=1)
        for fi in range(n_frames):
            med = residual[fi].median(); std = residual[fi].std() + 1e-8
            dyn_frac = (residual[fi] > med + args.tau * std).float().mean().item()
            if dyn_frac > 0.01:
                opacities.data.mul_(max(0.9, 1.0 - 0.02 * dyn_frac * 10))

    opt = torch.optim.Adam([
        {"params": [means, quats, scales, opacities, colors], "lr": 1e-2},
        {"params": [viewmats_param], "lr": args.pose_lr},
    ])
    print(f"训练 {args.iters} iters | anti={args.anti} tau={args.tau}")
    for it in range(args.iters):
        if args.anti and it > 0 and it % 200 == 0:
            anti_update()
        opt.zero_grad()
        r, _, _ = rasterization(means, quats, scales, opacities, colors,
                                viewmats_param, Ks, W, H)
        loss = torch.nn.functional.mse_loss(r.squeeze(0).permute(0, 3, 1, 2), images)
        loss.backward(); opt.step()
        if it % 500 == 0 or it == args.iters - 1:
            print(f"  {it}: {(-10*torch.log10(loss+1e-8)).item():.1f} dB")

    # 评估
    with torch.no_grad():
        r, _, _ = rasterization(means.detach(), quats.detach(), scales.detach(),
                                opacities.detach(), colors.detach(),
                                viewmats_param.detach(), Ks, W, H)
        psnr = -10 * np.log10(
            torch.nn.functional.mse_loss(r.squeeze(0).permute(0, 3, 1, 2), images).item() + 1e-8)

    # 导出优化后位姿并算 ATE
    pose_opt = viewmats_param.detach().squeeze(0).cpu().numpy()
    c2w_opt = np.array([np.linalg.inv(pose_opt[i]) for i in range(n_frames)])
    save_tum_trajectory(str(out_dir / "optimized_trajectory.txt"), c2w_opt,
                        [float(ts) for ts in meta["timestamps"]])
    opt_ate, opt_mean, opt_med, n_match = compute_ate(c2w_opt, np.array([float(ts) for ts in meta["timestamps"]]), gt_path)
    print(f"\n=== Phase 2d 完整结果 ===")
    print(f"PnP  init ATE: {pnp_ate*100:.1f} cm")
    print(f"优化后 ATE:   {opt_ate*100:.1f} cm (改善 {pnp_ate*100 - opt_ate*100:.1f} cm)")
    print(f"PSNR: {psnr:.1f} dB | 匹配帧: {n_match}")
    print(f"总耗时: {time.time()-t0:.0f}s")

    summary = {
        "pnp_ate_cm": round(pnp_ate*100, 1), "optimized_ate_cm": round(opt_ate*100, 1),
        "ate_improvement_cm": round((pnp_ate - opt_ate)*100, 1),
        "psnr": round(float(psnr), 2), "n_gauss": n_g, "n_frames": n_frames
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
