#!/usr/bin/env python3
"""Step 3 加深：per-frame 独立渲染 face-dynamic（单目动态 4D 高斯完整实现）。

核心：每个帧独立渲染 dynamic gaussians，带有 per-frame offset。
静态高斯 means 共享，动态高斯每个帧一个 offset。
这就是 D2GSLAM 的单目轻量化版本。

显存优化：动态高斯数限制在 3000 以内（每帧独立渲染 50 帧）。

用法：
  python3 scripts/step3_perframe_face.py --prior results/depth_prior_walking \
      --seq_dir /data/Datasets/TUM/rgbd_dataset_freiburg3_walking_xyz \
      --iters 2500 --n_dynamic 3000 --anti --tau 2.5
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
    ap.add_argument("--n_dynamic", type=int, default=3000,
                    help="动态高斯数量上限（per-frame 渲染）")
    ap.add_argument("--anti", action="store_true")
    ap.add_argument("--anti_period", type=int, default=200)
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--offset_lr", type=float, default=3e-3)
    ap.add_argument("--batch_frames", type=int, default=10,
                    help="per-frame 渲染每批帧数（控制显存）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"results/step3_perframe_n{args.n_dynamic}"
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
    pts_cat = np.concatenate([p for p,_ in all_frame_pts], 0)
    cols_cat = np.concatenate([c for _,c in all_frame_pts], 0)
    pts, cols = voxel_downsample(pts_cat, cols_cat, args.voxel, args.max_points)
    n_g = len(pts)

    g_vox = np.floor(pts / args.voxel).astype(np.int64)
    frame_freq = np.zeros(n_g, dtype=np.float64)
    for pts_w, _ in all_frame_pts:
        pv = set(map(tuple, np.floor(pts_w / args.voxel).astype(np.int64)))
        for gi, gv in enumerate(g_vox):
            if tuple(gv) in pv: frame_freq[gi] += 1
    n_dyn = min(args.n_dynamic, n_g)
    dyn_idx = np.argsort(frame_freq)[-n_dyn:]
    dyn_mask = np.zeros(n_g, dtype=bool); dyn_mask[dyn_idx] = True
    print(f"高斯 {n_g} | 动态 {n_dyn} ({n_dyn/n_g*100:.1f}%) | 帧 {n_frames}")

    pts_t = torch.from_numpy(pts).float().to(device)
    cols_t = torch.from_numpy(cols).float().to(device)
    dyn_mask_t = torch.from_numpy(dyn_mask).to(device)

    means = pts_t.unsqueeze(0).requires_grad_(True)  # (1,n_g,3) shared
    quats = torch.zeros(1,n_g,4,device=device); quats[...,0]=1.0; quats.requires_grad_(True)
    scales = torch.full((1,n_g,3),0.02,device=device).requires_grad_(True)
    opacities = torch.full((1,n_g),0.9,device=device).requires_grad_(True)
    colors = cols_t.unsqueeze(0).clone().requires_grad_(True)

    # per-frame offset：只动态高斯有
    # 存为 (n_frames, n_dyn, 3)，映射回 n_g
    frame_offsets = torch.zeros(n_frames, n_dyn, 3, device=device).requires_grad_(True)
    viewmats_gt = poses_gt.inverse().unsqueeze(0)
    viewmats_param = viewmats_gt.clone().requires_grad_(True)
    Ks = torch.from_numpy(K).float().to(device).unsqueeze(0).expand(1,n_frames,3,3).contiguous()

    from gsplat import rasterization

    def make_means_fi(fi):
        """第 fi 帧的 means：static means + per-frame offset for dynamic."""
        offset_full = torch.zeros(1, n_g, 3, device=device)
        offset_full[:, dyn_idx, :] = frame_offsets[fi:fi+1]  # (1, n_dyn, 3)
        return means + offset_full

    @torch.no_grad()
    def anti_update():
        if not args.anti: return
        with torch.no_grad():
            rends, _, _ = rasterization(means, quats, scales, opacities.detach(),
                                        colors, viewmats_param.detach(), Ks, W, H)
            residual = (rends.squeeze(0).permute(0,3,1,2) - images).abs().mean(dim=1)
            for fi in range(n_frames):
                med = residual[fi].median(); std = residual[fi].std() + 1e-8
                dyn_frac = (residual[fi] > med + args.tau * std).float().mean().item()
                if dyn_frac > 0.01:
                    opacities.data.mul_(max(0.9, 1.0 - 0.02 * dyn_frac * 10))

    # Phase A: 短暂静态初始化（500 iters，给 shared params 一个合理起点）
    K_STATIC = 500
    opt_shared = torch.optim.Adam([
        {"params": [means, quats, scales, opacities, colors], "lr": 1e-2},
        {"params": [viewmats_param], "lr": args.pose_lr},
    ])
    print(f"Phase A: 静态初始化 {K_STATIC} iters")
    for it in range(K_STATIC):
        opt_shared.zero_grad()
        r, _, _ = rasterization(means, quats, scales, opacities, colors,
                                viewmats_param, Ks, W, H)
        loss = torch.nn.functional.mse_loss(r.squeeze(0).permute(0,3,1,2), images)
        loss.backward(); opt_shared.step()
        if it % 100 == 0:
            print(f"  {it}: {(-10*torch.log10(loss+1e-8)).item():.1f} dB")

    # Phase B: 联合优化 — 每帧独立渲染所有帧，offset 与 shared params 同步优化
    # 显存优化：把 batch_frames 提高，每次渲染所有 batch_frames 帧
    opt_joint = torch.optim.Adam([
        {"params": [means, quats, scales, opacities, colors], "lr": 5e-3},
        {"params": [viewmats_param], "lr": args.pose_lr},
        {"params": [frame_offsets], "lr": args.offset_lr},
    ])
    remaining = args.iters - K_STATIC
    print(f"Phase B: 联合 per-frame 优化 {remaining} iters (batch={args.batch_frames})")
    for it in range(remaining):
        global_it = K_STATIC + it
        if args.anti and it > 0 and it % args.anti_period == 0:
            anti_update()

        # 逐帧独立渲染（关键：每帧有自己的 means）
        opt_joint.zero_grad()
        # 随机采 batch_frames 帧
        batch_idx = np.random.choice(n_frames, min(args.batch_frames, n_frames), replace=False)
        batch_loss = 0
        for fi in batch_idx:
            mf = make_means_fi(fi)
            rf, _, _ = rasterization(mf, quats, scales, opacities, colors,
                                     viewmats_param[:,fi:fi+1], Ks[:,fi:fi+1], W, H)
            loss_fi = torch.nn.functional.mse_loss(
                rf.reshape(H,W,3), images[fi].permute(1,2,0))
            batch_loss = batch_loss + loss_fi
        batch_loss = batch_loss / len(batch_idx)
        # 轻量 offset 正则化
        reg = frame_offsets.pow(2).mean() * 0.0005
        (batch_loss + reg).backward(); opt_joint.step()

        if it % 500 == 0 or it == remaining - 1:
            with torch.no_grad():
                # 用 mean offset 评估整体
                avg = frame_offsets.mean(dim=0).unsqueeze(0)
                off_f = torch.zeros(1,n_g,3,device=device)
                off_f[:,dyn_idx,:] = avg
                md = means + off_f
                r, _, _ = rasterization(md, quats, scales, opacities, colors,
                                        viewmats_param, Ks, W, H)
                psnr = -10*torch.log10(torch.nn.functional.mse_loss(
                    r.squeeze(0).permute(0,3,1,2), images)+1e-8).item()
            print(f"  {global_it}: {psnr:.1f} dB")

    # 评估（mean offset）
    with torch.no_grad():
        avg = frame_offsets.mean(dim=0).unsqueeze(0)
        off_f = torch.zeros(1,n_g,3,device=device)
        off_f[:,dyn_idx,:] = avg
        md = means + off_f
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

    summary = {"mode": "perframe_face", "n_dynamic": n_dyn,
               "psnr_mean": round(float(psnr),2),
               "per_frame_psnr": per_frame, "n_gauss": n_g, "n_frames": n_frames}
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Step 3 per-frame face ===")
    print(f"PSNR={summary['psnr_mean']} dB 前半={np.mean(per_frame[:n_frames//2]):.1f} 后半={np.mean(per_frame[n_frames//2:]):.1f}")
    print(f"判决: >=20 → {'PASS' if summary['psnr_mean']>=20 else 'FAIL'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
