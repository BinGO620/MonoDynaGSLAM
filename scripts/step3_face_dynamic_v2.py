#!/usr/bin/env python3
"""Step 3: 单目 face-dynamic（动态高斯 per-frame 变形 + 静态锚定）。

gsplat 不支持 per-frame means，所以 face-dynamic 的正确做法是：
  对每个帧，将动态高斯的 means 加上该帧的 offset，然后独立渲染。
  静态高斯 means 保持不变。

这等价于 D2GSLAM 的 static-3D + dynamic-4D 的单目轻量化实现。

用法：
  python3 scripts/step3_face_dynamic_v2.py --prior results/depth_prior_walking \
      --seq_dir /data/Datasets/TUM/rgbd_dataset_freiburg3_walking_xyz \
      --iters 2500 --dynamic_ratio 0.15
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch


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
    ap.add_argument("--seq_dir", default=None)
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--max_points", type=int, default=60000)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--pose_lr", type=float, default=5e-5)
    ap.add_argument("--dynamic_ratio", type=float, default=0.15)
    ap.add_argument("--anti", action="store_true")
    ap.add_argument("--anti_period", type=int, default=200)
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--offset_lr", type=float, default=5e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        tag = "face" if args.dynamic_ratio > 0 else "static"
        args.out = f"results/step3v2_{tag}"
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
        if T is None: T = np.eye(4)
        poses.append(torch.from_numpy(T).float())
    images = torch.stack(images).to(device)
    poses_gt = torch.stack(poses).to(device)

    # ---- 多帧反投影 + 动态标记 ----
    all_frame_pts = []
    for i in range(n_frames):
        dep = depths[i]; yi,xi = np.mgrid[0:dep.shape[0]:4, 0:dep.shape[1]:4]
        d = dep[::4,::4]; v = (d>0.2)&(d<8)
        if v.sum()==0: continue
        yc,xc,dc = yi[v], xi[v], d[v]
        pts_cam = np.stack([(xc-cx_o)/fx_o*dc, (yc-cy_o)/fy_o*dc, dc], axis=-1)
        pose = match_gt(gt, gt_ts, float(meta["timestamps"][i]))
        if pose is None: continue
        pts_w = (pose[:3,:3] @ pts_cam.T).T + pose[:3,3]
        img_o = cv2.cvtColor(cv2.imread(str(seq_dir/meta["image_paths"][i])), cv2.COLOR_BGR2RGB)
        cols = img_o[yc,xc].astype(np.float32)/255.0
        all_frame_pts.append((pts_w, cols))

    # 体素下采样
    pts_cat = np.concatenate([p for p,_ in all_frame_pts], 0)
    cols_cat = np.concatenate([c for _,c in all_frame_pts], 0)
    pts, cols = voxel_downsample(pts_cat, cols_cat, args.voxel, args.max_points)
    n_g = len(pts)

    # 动态标记：跨帧体素频率
    g_vox = np.floor(pts / args.voxel).astype(np.int64)
    g_vox_set = set(map(tuple, g_vox))
    frame_freq = np.zeros(n_g, dtype=np.float64)
    for pts_w, _ in all_frame_pts:
        pv = np.floor(pts_w / args.voxel).astype(np.int64)
        pv_set = set(map(tuple, pv))
        for gi, gv in enumerate(g_vox):
            if tuple(gv) in pv_set:
                frame_freq[gi] += 1
    n_dyn = max(1, int(n_g * args.dynamic_ratio))
    dyn_idx = np.argsort(frame_freq)[-n_dyn:]
    dyn_mask = np.zeros(n_g, dtype=bool); dyn_mask[dyn_idx] = True
    print(f"高斯 {n_g} | 动态 {n_dyn} ({n_dyn/n_g*100:.1f}%) | 帧 {n_frames} | res {W}x{H}")

    # ---- 初始化 ----
    pts_t = torch.from_numpy(pts).float().to(device)
    cols_t = torch.from_numpy(cols).float().to(device)

    means = pts_t.unsqueeze(0).requires_grad_(True)
    quats = torch.zeros(1,n_g,4,device=device); quats[...,0]=1.0; quats.requires_grad_(True)
    scales = torch.full((1,n_g,3),0.02,device=device).requires_grad_(True)
    opacities = torch.full((1,n_g),0.9,device=device).requires_grad_(True)
    colors = cols_t.unsqueeze(0).clone().requires_grad_(True)

    # 动态偏移：per-frame 3D offset
    dyn_mask_t = torch.from_numpy(dyn_mask).to(device)  # (n_g,) bool
    frame_offsets = torch.zeros(n_frames, n_g, 3, device=device).requires_grad_(True)

    viewmats_gt = poses_gt.inverse().unsqueeze(0)
    viewmats_param = viewmats_gt.clone().requires_grad_(True)
    Ks = torch.from_numpy(K).float().to(device).unsqueeze(0).expand(1,n_frames,3,3).contiguous()

    from gsplat import rasterization

    @torch.no_grad()
    def anti_update():
        if not args.anti: return
        # 用 means (不加 offset) 做快速渲染，算残差
        rends, _, _ = rasterization(means, quats, scales, opacities.detach(),
                                    colors, viewmats_param.detach(), Ks, W, H)
        residual = (rends.squeeze(0).permute(0,3,1,2) - images).abs().mean(dim=1)
        for fi in range(n_frames):
            med = residual[fi].median(); std = residual[fi].std() + 1e-8
            dyn_frac = (residual[fi] > med + args.tau * std).float().mean().item()
            if dyn_frac > 0.01:
                opacities.data.mul_(max(0.9, 1.0 - 0.02 * dyn_frac * 10))

    # 静态帧优化器（迭代 0,1,...K：只优化 means/quats/scales/opacities/colors/pose）
    # 动态帧：每帧渲染时 means 加 frame_offsets[dyn_idx]
    opt_static = torch.optim.Adam([
        {"params": [means, quats, scales, opacities, colors], "lr": 1e-2},
        {"params": [viewmats_param], "lr": args.pose_lr},
    ])
    opt_offset = torch.optim.Adam([{"params": [frame_offsets], "lr": args.offset_lr}])

    # 先训练静态基线 K iters（不用 offset），建立良好初始化
    K_STATIC = 800
    print(f"Phase A: 静态基线 {K_STATIC} iters")
    for it in range(K_STATIC):
        opt_static.zero_grad()
        rends, _, _ = rasterization(means, quats, scales, opacities, colors,
                                    viewmats_param, Ks, W, H)
        loss = torch.nn.functional.mse_loss(rends.squeeze(0).permute(0,3,1,2), images)
        loss.backward(); opt_static.step()
        if it % 200 == 0:
            print(f"  static {it}: loss={loss.item():.4f} psnr={-10*torch.log10(loss+1e-8).item():.1f}")

    # Phase B: face-dynamic — 逐帧渲染加 offset，交替优化静态参数 + offset
    print(f"Phase B: face-dynamic {args.iters-K_STATIC} iters | anti={args.anti}")
    for it in range(K_STATIC, args.iters):
        if args.anti and it > 0 and it % args.anti_period == 0:
            anti_update()

        # 交替优化
        if (it - K_STATIC) % 2 == 0:
            # 优化静态参数：用 means + mean_offset 做共享渲染
            opt_static.zero_grad()
            avg_offset = frame_offsets.mean(dim=0).unsqueeze(0)  # (1,n_g,3)
            means_deformed = means + avg_offset * dyn_mask_t.unsqueeze(0).unsqueeze(-1)
            rends, _, _ = rasterization(means_deformed, quats, scales, opacities, colors,
                                        viewmats_param, Ks, W, H)
            loss = torch.nn.functional.mse_loss(rends.squeeze(0).permute(0,3,1,2), images)
            loss.backward(); opt_static.step()
        else:
            # 优化 offset：逐帧渲染（只对动态高斯有偏移）
            # 逐帧开销大，用 mini-batch：每 iter 随机选 5 帧
            opt_offset.zero_grad()
            batch_idx = np.random.choice(n_frames, min(5, n_frames), replace=False)
            batch_loss = 0
            for fi in batch_idx:
                offset_fi = frame_offsets[fi].unsqueeze(0)  # (1,n_g,3)
                means_fi = means + offset_fi * dyn_mask_t.unsqueeze(0).unsqueeze(-1)
                rends_fi, _, _ = rasterization(means_fi, quats, scales, opacities, colors,
                                               viewmats_param[:,fi:fi+1], Ks[:,fi:fi+1], W, H)
                loss_fi = torch.nn.functional.mse_loss(
                    rends_fi.reshape(H,W,3), images[fi].permute(1,2,0))
                batch_loss = batch_loss + loss_fi
            batch_loss = batch_loss / len(batch_idx)
            # 正则化：offset 不能太大
            reg = frame_offsets.pow(2).mean() * 0.01
            (batch_loss + reg).backward(); opt_offset.step()

        if it % 500 == 0 or it == args.iters - 1:
            with torch.no_grad():
                avg = frame_offsets.mean(dim=0).unsqueeze(0)
                md = means + avg * dyn_mask_t.unsqueeze(0).unsqueeze(-1)
                r, _, _ = rasterization(md, quats, scales, opacities, colors,
                                        viewmats_param, Ks, W, H)
                psnr = -10*torch.log10(torch.nn.functional.mse_loss(
                    r.squeeze(0).permute(0,3,1,2), images)+1e-8).item()
            print(f"  iter {it}: psnr={psnr:.1f}")

    # 评估：用 mean offset 做整体渲染
    with torch.no_grad():
        avg = frame_offsets.mean(dim=0).unsqueeze(0)
        md = means + avg * dyn_mask_t.unsqueeze(0).unsqueeze(-1)
        rends, _, _ = rasterization(md, quats, scales, opacities.detach(), colors.detach(),
                                    viewmats_param.detach(), Ks, W, H)
        mse = torch.nn.functional.mse_loss(rends.squeeze(0).permute(0,3,1,2), images).item()
        psnr = -10 * np.log10(mse+1e-8)
        per_frame = [round(-10*np.log10(torch.nn.functional.mse_loss(
            rends.squeeze(0)[i].permute(2,0,1), images[i]).item()+1e-8), 2)
            for i in range(n_frames)]

    step = max(1, n_frames//6)
    for j, i in enumerate(range(0, n_frames, step)):
        s = (rends.squeeze(0)[i].clamp(0,1).cpu().numpy()*255).astype(np.uint8)
        cv2.imwrite(str(out_dir/f"render_{j}.png"), cv2.cvtColor(s,cv2.COLOR_RGB2BGR))

    summary = {"mode": "face_dynamic", "dynamic_ratio": args.dynamic_ratio,
               "n_dynamic": n_dyn, "psnr_mean": round(float(psnr),2),
               "per_frame_psnr": per_frame, "n_gauss": n_g, "n_frames": n_frames}
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Step 3 face-dynamic ===")
    print(f"PSNR={summary['psnr_mean']} dB 前半={np.mean(per_frame[:n_frames//2]):.1f} 后半={np.mean(per_frame[n_frames//2:]):.1f}")
    print(f"判决: >=20 → {'PASS' if summary['psnr_mean']>=20 else 'FAIL'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
