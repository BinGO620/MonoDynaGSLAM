#!/usr/bin/env python3
"""Step 1: 单目静态 3DGS 重建基线（gsplat + Metric3D 深度先验，多帧融合初始化）。

流程（Phase 0）：
  1. 加载预生成的 Metric3D 深度先验（scripts/generate_depth_prior.sh 产出）
  2. 全部帧深度反投影 → 世界系点云（带像素颜色），体素下采样
  3. gsplat 3DGS 训练（颜色优化 + 位姿微调）
  4. 评估 PSNR / 简化 ATE

依赖环境：系统 python3（gsplat 1.5.3, torch 2.4.1+cu121）

用法：
  python3 scripts/step1_build_from_depth_prior.py \
      --prior results/depth_prior --iters 3000 --max_points 60000
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def voxel_downsample(pts: np.ndarray, colors: np.ndarray, voxel: float = 0.02,
                     max_points: int = 60000):
    """体素下采样：每体素取平均。"""
    keys = np.floor(pts / voxel).astype(np.int64)
    # 用唯一键聚合
    _, idx, inverse, counts = np.unique(
        keys, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    n_vox = len(idx)
    sum_pts = np.zeros((n_vox, 3))
    sum_col = np.zeros((n_vox, 3))
    np.add.at(sum_pts, inverse, pts)
    np.add.at(sum_col, inverse, colors)
    mean_pts = sum_pts / counts[:, None]
    mean_col = sum_col / counts[:, None]
    if n_vox > max_points:
        sel = np.random.choice(n_vox, max_points, replace=False)
        mean_pts, mean_col = mean_pts[sel], mean_col[sel]
    return mean_pts, mean_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="results/depth_prior",
                    help="深度先验目录（depths.npy + meta.json）")
    ap.add_argument("--res", type=int, default=320, help="训练渲染宽度")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--max_points", type=int, default=60000)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--pose_lr", type=float, default=5e-5)
    ap.add_argument("--no_pose_opt", action="store_true")
    ap.add_argument("--out", default="results/step1")
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((Path(args.prior) / "meta.json").read_text())
    depths = np.load(Path(args.prior) / "depths.npy")  # (N,480,640) 米
    seq_dir = Path(meta["seq_dir"])
    K_orig = np.array(meta["K"])  # 640 分辨率内参
    n_frames = len(meta["image_paths"])
    print(f"序列: {seq_dir.name} | 帧: {n_frames} | 深度: {depths.shape}")

    # ---- 训练分辨率 ----
    W = args.res
    H = int(args.res * 0.75)
    scale = W / 640.0
    K = K_orig.copy()
    K[:2] *= scale
    Ks = torch.from_numpy(K).float().to(device).unsqueeze(0).expand(1, n_frames, 3, 3).contiguous()

    # ---- 加载图像 + GT 位姿 ----
    fx_o, fy_o, cx_o, cy_o = K_orig[0, 0], K_orig[1, 1], K_orig[0, 2], K_orig[1, 2]

    def quat_to_R(qx, qy, qz, qw):
        return np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])

    gt = {}
    for line in (seq_dir / "groundtruth.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        t = float(p[0])
        R = quat_to_R(*[float(x) for x in p[4:8]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [float(p[1]), float(p[2]), float(p[3])]
        gt[t] = T

    gt_ts = sorted(gt.keys())

    def match_gt(t):
        idx = np.searchsorted(gt_ts, t)
        best, best_dt = None, 0.02
        for i in [idx - 1, idx, idx + 1]:
            if 0 <= i < len(gt_ts) and abs(gt_ts[i] - t) < best_dt:
                best, best_dt = gt_ts[i], abs(gt_ts[i] - t)
        return gt[best] if best is not None else None

    images = []
    poses = []
    for i, (ts, rel) in enumerate(zip(meta["timestamps"], meta["image_paths"])):
        img = cv2.imread(str(seq_dir / rel))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))
        images.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        T = match_gt(float(ts))
        if T is None:
            raise KeyError(f"帧 {ts} 无 20ms 内 GT 匹配")
        poses.append(torch.from_numpy(T).float())
    images = torch.stack(images).to(device)   # (N,3,H,W)
    poses_gt = torch.stack(poses).to(device)  # (N,4,4) c2w
    print(f"图像 {tuple(images.shape)} | 加载 {time.time()-t0:.1f}s")

    # ---- 多帧深度反投影 + 颜色采样 → 初始点云 ----
    all_pts = []
    all_cols = []
    for i in range(n_frames):
        dep = depths[i]
        h_o, w_o = dep.shape
        yi, xi = np.mgrid[0:h_o:4, 0:w_o:4]  # 1/4 采样
        d = dep[::4, ::4]
        valid = (d > 0.2) & (d < 8.0)
        if valid.sum() == 0:
            continue
        yi, xi, di = yi[valid], xi[valid], d[valid]
        x_cam = (xi - cx_o) / fx_o * di
        y_cam = (yi - cy_o) / fy_o * di
        pts_cam = np.stack([x_cam, y_cam, di], axis=-1)  # (M,3)

        # 颜色：从对应原图取
        img_o = cv2.cvtColor(cv2.imread(str(seq_dir / meta["image_paths"][i])), cv2.COLOR_BGR2RGB)
        cols = img_o[yi, xi].astype(np.float32) / 255.0

        pose = match_gt(float(meta["timestamps"][i]))
        pts_world = (pose[:3, :3] @ pts_cam.T).T + pose[:3, 3]
        all_pts.append(pts_world)
        all_cols.append(cols)
    pts = np.concatenate(all_pts, 0)
    cols = np.concatenate(all_cols, 0)
    print(f"反投影原始点数: {len(pts)}")
    pts, cols = voxel_downsample(pts, cols, args.voxel, args.max_points)
    print(f"体素下采样后: {len(pts)} @ voxel={args.voxel}")

    pts_t = torch.from_numpy(pts).float().to(device)
    cols_t = torch.from_numpy(cols).float().to(device)

    n_gauss = len(pts)
    means = pts_t.unsqueeze(0).requires_grad_(True)
    quats = torch.zeros(1, n_gauss, 4, device=device)
    quats[..., 0] = 1.0
    quats.requires_grad_(True)
    scales = torch.full((1, n_gauss, 3), 0.02, device=device).requires_grad_(True)
    opacities = torch.full((1, n_gauss), 0.9, device=device).requires_grad_(True)
    colors = cols_t.unsqueeze(0).clone().requires_grad_(True)  # 初始颜色=像素色

    viewmats_gt = poses_gt.inverse().unsqueeze(0)  # (1,N,4,4) w2c
    if args.no_pose_opt:
        viewmats_param = viewmats_gt
    else:
        viewmats_param = viewmats_gt.clone().requires_grad_(True)

    opt = torch.optim.Adam([
        {"params": [means, quats, scales, opacities, colors], "lr": 1e-2},
        {"params": [viewmats_param], "lr": args.pose_lr},
    ])

    from gsplat import rasterization
    print(f"训练 {args.iters} iters | 高斯 {n_gauss} | 位姿优化 {not args.no_pose_opt}")
    hist = []
    for it in range(args.iters):
        opt.zero_grad()
        rends, _, _ = rasterization(means, quats, scales, opacities, colors,
                                    viewmats_param, Ks, W, H)
        loss = torch.nn.functional.mse_loss(rends.squeeze(0).permute(0, 3, 1, 2), images)
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == args.iters - 1:
            psnr = -10 * torch.log10(loss + 1e-8).item()
            hist.append((it, loss.item(), psnr))
            print(f"  iter {it}: loss={loss.item():.4f} psnr={psnr:.1f} dB")

    # ---- 评估 ----
    with torch.no_grad():
        rends, _, _ = rasterization(means.detach(), quats.detach(), scales.detach(),
                                    opacities.detach(), colors.detach(),
                                    viewmats_param.detach(), Ks, W, H)
        mse = torch.nn.functional.mse_loss(rends.squeeze(0).permute(0, 3, 1, 2), images).item()
        psnr = -10 * np.log10(mse + 1e-8)
        # 每帧 PSNR
        per_frame = []
        for i in range(n_frames):
            m = torch.nn.functional.mse_loss(
                rends.squeeze(0)[i].permute(2, 0, 1), images[i]).item()
            per_frame.append(-10 * np.log10(m + 1e-8))
        # 简化 ATE：位姿微调量（Umeyama 前的粗指标）
        ate = (viewmats_param[:, :, :3, 3] - viewmats_gt[:, :, :3, 3]).norm(dim=-1)
        ate = ate.squeeze(0).mean().item() * 100

    # 保存渲染样例（rends: (1,N,H,W,3) NHWC）
    step = max(1, n_frames // 6)
    for j, i in enumerate(range(0, n_frames, step)):
        s = (rends.squeeze(0)[i].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # (H,W,3)
        cv2.imwrite(str(out_dir / f"render_{j}.png"), cv2.cvtColor(s, cv2.COLOR_RGB2BGR))

    summary = {
        "seq": seq_dir.name, "n_frames": n_frames, "res": W,
        "n_gauss": n_gauss, "iters": args.iters,
        "psnr_mean": round(float(psnr), 2),
        "psnr_first_half": round(float(np.mean(per_frame[: n_frames // 2])), 2),
        "psnr_second_half": round(float(np.mean(per_frame[n_frames // 2:])), 2),
        "pose_drift_cm": round(float(ate), 2),
        "train_seconds": round(time.time() - t0, 1),
        "per_frame_psnr": [round(float(p), 2) for p in per_frame],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Step 1 结果 ===")
    print(f"PSNR mean={summary['psnr_mean']} dB (前半 {summary['psnr_first_half']} / 后半 {summary['psnr_second_half']})")
    print(f"位姿漂移 {summary['pose_drift_cm']} cm | 高斯 {n_gauss} | 耗时 {summary['train_seconds']}s")
    print(f"判决门: PSNR >= 20 dB → {'PASS ✅' if summary['psnr_mean'] >= 20 else 'FAIL ❌'}")
    print(f"样例图与 summary.json 已存 {out_dir}")


if __name__ == "__main__":
    main()
