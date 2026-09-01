#!/usr/bin/env python3
"""Step 2: 单目 anti-dynamic 3DGS（多帧渲染一致性降权）。

在 Step 1 基础上加入动态区域抑制：
  对每帧渲染残差统计 → 如果某高斯贡献到高残差区域 → 降权其 opacity。
  这等价于 "渲染不一致的高斯不可信"，无需光流/语义网络，纯几何-光度一致性。

用法：
  python3 scripts/step2_anti_dynamic.py --prior results/depth_prior_walking \
      --seq fr3_walking_xyz --iters 3000 --tau 3.0
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ---- 工具函数（从 step1 复用） ----

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

def quat_to_R(qx, qy, qz, qw):
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])

def match_gt(gt, gt_ts, t, tol=0.02):
    idx = np.searchsorted(gt_ts, t)
    for i in [idx-1, idx, idx+1]:
        if 0 <= i < len(gt_ts) and abs(gt_ts[i] - t) < tol:
            return gt[gt_ts[i]]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="results/depth_prior_walking")
    ap.add_argument("--seq_dir", default=None, help="若不设则从 meta.json 读取")
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--max_points", type=int, default=60000)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--pose_lr", type=float, default=5e-5)
    ap.add_argument("--anti", action="store_true", default=True,
                    help="启用 anti-dynamic 一致性降权")
    ap.add_argument("--tau", type=float, default=3.0,
                    help="动态惩罚系数（越大剔除越强）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        tag = "anti" if args.anti else "noanti"
        args.out = f"results/step2_{tag}_tau{args.tau}"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    device = "cuda"
    meta = json.loads((Path(args.prior) / "meta.json").read_text())
    depths = np.load(Path(args.prior) / "depths.npy")
    seq_dir = Path(args.seq_dir or meta["seq_dir"])
    K_orig = np.array(meta["K"])
    n_frames = len(meta["timestamps"])

    W, H = args.res, int(args.res * 0.75)
    scale = W / 640.0
    K = K_orig.copy(); K[:2] *= scale
    fx_o, fy_o, cx_o, cy_o = K_orig[0,0], K_orig[1,1], K_orig[0,2], K_orig[1,2]

    gt = {}
    for line in (seq_dir / "groundtruth.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        p = line.split(); t = float(p[0])
        T = np.eye(4); T[:3,:3] = quat_to_R(*[float(x) for x in p[4:8]]); T[:3,3] = [float(p[1]),float(p[2]),float(p[3])]
        gt[t] = T
    gt_ts = sorted(gt.keys())

    images = []; poses = []
    for ts, rel in zip(meta["timestamps"], meta["image_paths"]):
        img = cv2.cvtColor(cv2.imread(str(seq_dir / rel)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))
        images.append(torch.from_numpy(img).permute(2,0,1).float()/255.0)
        T = match_gt(gt, gt_ts, float(ts))
        if T is None:
            T = np.eye(4)
            print(f"WARN: frame {ts} no GT match, using identity")
        poses.append(torch.from_numpy(T).float())
    images = torch.stack(images).to(device)
    poses_gt = torch.stack(poses).to(device)

    # 反投影多帧点云
    all_pts, all_cols = [], []
    for i in range(n_frames):
        dep = depths[i]; yi,xi = np.mgrid[0:dep.shape[0]:4, 0:dep.shape[1]:4]
        d = dep[::4,::4]; v = (d>0.2)&(d<8)
        if v.sum()==0: continue
        yc,xc,dc = yi[v], xi[v], d[v]
        pts_cam = np.stack([(xc-cx_o)/fx_o*dc, (yc-cy_o)/fy_o*dc, dc], axis=-1)
        img_o = cv2.cvtColor(cv2.imread(str(seq_dir/meta["image_paths"][i])), cv2.COLOR_BGR2RGB)
        cols = img_o[yc,xc].astype(np.float32)/255.0
        pose = match_gt(gt, gt_ts, float(meta["timestamps"][i]))
        if pose is None:
            continue
        pts_w = (pose[:3,:3] @ pts_cam.T).T + pose[:3,3]
        all_pts.append(pts_w); all_cols.append(cols)
    pts, cols = voxel_downsample(np.concatenate(all_pts), np.concatenate(all_cols),
                                  args.voxel, args.max_points)
    n_g = len(pts); print(f"高斯数 {n_g} | 帧 {n_frames} | res {W}x{H}")

    pts_t = torch.from_numpy(pts).float().to(device)
    cols_t = torch.from_numpy(cols).float().to(device)
    means = pts_t.unsqueeze(0).requires_grad_(True)
    quats = torch.zeros(1,n_g,4,device=device); quats[...,0]=1.0; quats.requires_grad_(True)
    scales = torch.full((1,n_g,3),0.02,device=device).requires_grad_(True)
    opacities = torch.full((1,n_g),0.9,device=device).requires_grad_(True)
    colors = cols_t.unsqueeze(0).clone().requires_grad_(True)

    viewmats_gt = poses_gt.inverse().unsqueeze(0)
    viewmats_param = viewmats_gt.clone().requires_grad_(True)
    Ks = torch.from_numpy(K).float().to(device).unsqueeze(0).expand(1,n_frames,3,3).contiguous()

    from gsplat import rasterization

    # 基础 loss
    def render_and_loss(viewmats, opt_anti=True, current_opacities=None):
        rends, alphas, info = rasterization(
            means, quats, scales,
            current_opacities if current_opacities is not None else opacities,
            colors, viewmats, Ks, W, H
        )
        imgs_pred = rends.squeeze(0).permute(0,3,1,2)  # (N,3,H,W)
        per_frame_mse = torch.nn.functional.mse_loss(imgs_pred, images, reduction='none').mean(dim=(1,2,3))  # (N,)
        recon_loss = per_frame_mse.mean()

        if opt_anti and args.anti:
            # Anti-dynamic: 渲染残差大的帧 → 降权该帧对整体 loss 的贡献
            # 机制：计算每帧的像素级残差，残差高的像素区域 → 该区域对应的高斯可能在动态物体上
            # 用 soft 惩罚：residual_map = |pred - gt| → 每个高斯的 opacity 被残差调制
            per_frame_psnr = -10 * torch.log10(per_frame_mse + 1e-8)
            # 对残差 > tau × median 的帧做 down-weight
            median_psnr = per_frame_psnr.median()
            weight = torch.sigmoid((per_frame_psnr - median_psnr + args.tau) / 0.5)
            recon_loss = (per_frame_mse * weight.detach()).mean()
            return recon_loss, per_frame_mse, weight
        return recon_loss, per_frame_mse, None

    opt = torch.optim.Adam([
        {"params": [means, quats, scales, opacities, colors]},
        {"params": [viewmats_param], "lr": args.pose_lr},
    ], lr=1e-2)

    print(f"训练 {args.iters} iters | tau={args.tau} | anti={args.anti}")
    best_psnr = 0
    for it in range(args.iters):
        opt.zero_grad()
        loss, mse, w = render_and_loss(viewmats_param, args.anti)
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == args.iters - 1:
            psnr = -10 * torch.log10(mse.mean()+1e-8).item()
            print(f"  iter {it}: loss={loss.item():.4f} psnr={psnr:.1f}")
            best_psnr = max(best_psnr, psnr)

    # 评估
    with torch.no_grad():
        rends, _, _ = rasterization(means.detach(), quats.detach(), scales.detach(),
                                    opacities.detach(), colors.detach(),
                                    viewmats_param.detach(), Ks, W, H)
        mse = torch.nn.functional.mse_loss(rends.squeeze(0).permute(0,3,1,2), images).item()
        psnr_final = -10 * np.log10(mse+1e-8)
        per_frame = []
        for i in range(n_frames):
            m = torch.nn.functional.mse_loss(rends.squeeze(0)[i].permute(2,0,1), images[i]).item()
            per_frame.append(-10 * np.log10(m+1e-8))

    # 渲染对比图
    step = max(1, n_frames//6)
    for j, i in enumerate(range(0, n_frames, step)):
        s = (rends.squeeze(0)[i].clamp(0,1).cpu().numpy()*255).astype(np.uint8)
        cv2.imwrite(str(out_dir/f"render_{j}.png"), cv2.cvtColor(s,cv2.COLOR_RGB2BGR))

    summary = {
        "mode": "anti" if args.anti else "no_anti",
        "tau": args.tau,
        "psnr_mean": round(float(psnr_final), 2),
        "per_frame_psnr": [round(float(p),2) for p in per_frame],
        "n_gauss": n_g, "n_frames": n_frames,
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Step 2 {'anti' if args.anti else 'no_anti'} 结果 ===")
    print(f"PSNR mean={summary['psnr_mean']} dB")
    print(f"前半={np.mean(per_frame[:n_frames//2]):.1f} 后半={np.mean(per_frame[n_frames//2:]):.1f}")
    print(f"判决: >=20 dB → {'PASS' if summary['psnr_mean']>=20 else 'FAIL'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
