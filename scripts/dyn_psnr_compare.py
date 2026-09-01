#!/usr/bin/env python3
"""动态区域局部 PSNR 对比：检验"残差归因"是否在动态区有局部增益（对 PSNR 稀释论证的实证）。

三种模式（none / selective / random_gauss）在 Bonn balloon 上训练，
评估时分区报告：全局 PSNR / 动态 mask 内 PSNR / 静态区 PSNR。
若 selective 的动态区 PSNR > random_gauss 且 > none，则归因有效（全局 PSNR 只是稀释）。

用法：
  python3 scripts/dyn_psnr_compare.py --mode none --out results/dynpsnr_none
"""
import argparse
import json
import sys
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
    ap.add_argument("--prior", default="results/depth_prior_bonn_balloon")
    ap.add_argument("--seq_dir", default="/data/Datasets/Bonn/rgbd_bonn_balloon")
    ap.add_argument("--mode", required=True,
                    choices=["none", "selective", "random_gauss"])
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--max_points", type=int, default=40000)
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--anti_period", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    t0 = time.time()
    device = "cuda"
    meta = json.loads((Path(args.prior) / "meta.json").read_text())
    depths = np.load(Path(args.prior) / "depths.npy")
    seq_dir = Path(args.seq_dir)
    K_orig = np.array(meta["K"])
    n_frames = len(meta["timestamps"])

    W, H = args.res, int(args.res * 0.75)
    scale = W / 640.0
    K = K_orig.copy(); K[:2] *= scale
    Ks = torch.from_numpy(K).float().to(device).unsqueeze(0).expand(1, n_frames, 3, 3).contiguous()

    gt = {}
    for line in (seq_dir / "groundtruth.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        p = line.split(); t = float(p[0])
        T = np.eye(4); T[:3,:3] = quat_to_R(*[float(x) for x in p[4:8]])
        T[:3,3] = [float(p[1]),float(p[2]),float(p[3])]
        gt[t] = T
    gt_ts = sorted(gt.keys())

    images = []; poses = []
    for ts, rel in zip(meta["timestamps"], meta["image_paths"]):
        img = cv2.cvtColor(cv2.imread(str(seq_dir / rel)), cv2.COLOR_BGR2RGB)
        images.append(torch.from_numpy(cv2.resize(img,(W,H)).transpose(2,0,1)).float()/255.0)
        T = match_gt(gt, gt_ts, float(ts))
        poses.append(torch.from_numpy(T).float() if T is not None else torch.eye(4))
    images = torch.stack(images).to(device)
    poses_gt = torch.stack(poses).to(device)

    # GT 动态 mask（按帧序对应）
    cand = sorted((seq_dir / "dynamic_mask_gtmc").glob("*.png"))
    dyn_masks = []
    for i in range(n_frames):
        if i < len(cand):
            m = cv2.imread(str(cand[i]), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            dyn_masks.append((m > 127).astype(np.float32))
        else:
            dyn_masks.append(np.zeros((H, W), dtype=np.float32))
    dyn_t = torch.from_numpy(np.stack(dyn_masks)).to(device)  # (N,H,W)
    print(f"GT 动态占比: {dyn_t.mean().item()*100:.1f}%")

    fx_o, fy_o, cx_o, cy_o = K_orig[0,0], K_orig[1,1], K_orig[0,2], K_orig[1,2]
    all_pts, all_cols = [], []
    for i in range(n_frames):
        dep = depths[i]; yi,xi = np.mgrid[0:dep.shape[0]:4, 0:dep.shape[1]:4]
        d = dep[::4,::4]; v = (d>0.2)&(d<8)
        if v.sum()==0: continue
        yc,xc,dc = yi[v], xi[v], d[v]
        pts_cam = np.stack([(xc-cx_o)/fx_o*dc,(yc-cy_o)/fy_o*dc,dc], axis=-1)
        img_o = cv2.cvtColor(cv2.imread(str(seq_dir/meta["image_paths"][i])), cv2.COLOR_BGR2RGB)
        cols = img_o[yc,xc].astype(np.float32)/255.0
        pose = match_gt(gt, gt_ts, float(meta["timestamps"][i]))
        if pose is None: continue
        pts_w = (pose[:3,:3] @ pts_cam.T).T + pose[:3,3]
        all_pts.append(pts_w); all_cols.append(cols)
    pts, cols = voxel_downsample(np.concatenate(all_pts), np.concatenate(all_cols), 0.02, args.max_points)
    n_g = len(pts)

    pts_t = torch.from_numpy(pts).float().to(device)
    cols_t = torch.from_numpy(cols).float().to(device)
    means = pts_t.unsqueeze(0).requires_grad_(True)
    quats = torch.zeros(1,n_g,4,device=device); quats[...,0]=1.0; quats.requires_grad_(True)
    scales = torch.full((1,n_g,3),0.02,device=device).requires_grad_(True)
    opacities = torch.full((1,n_g),0.9,device=device).requires_grad_(True)
    colors = cols_t.unsqueeze(0).clone().requires_grad_(True)
    viewmats_gt = poses_gt.inverse().unsqueeze(0)
    viewmats_param = viewmats_gt.clone().requires_grad_(True)

    from gsplat import rasterization

    @torch.no_grad()
    def project_gaussians():
        mw = means.squeeze(0)
        ones = torch.ones(n_g, 1, device=device)
        cam_pts = torch.einsum("nij,gj->ngi", viewmats_param.detach().squeeze(0),
                               torch.cat([mw, ones], 1))
        depth = cam_pts[..., 2]; valid = depth > 0.05
        fx_, fy_ = K[0,0].item(), K[1,1].item()
        cx_, cy_ = K[0,2].item(), K[1,2].item()
        u = (fx_ * cam_pts[...,0] / depth.clamp(min=1e-6) + cx_).clamp(0, W-1).long()
        v = (fy_ * cam_pts[...,1] / depth.clamp(min=1e-6) + cy_).clamp(0, H-1).long()
        return u, v, valid

    @torch.no_grad()
    def selective_select():
        rends, _, _ = rasterization(means, quats, scales, opacities.detach(), colors,
                                    viewmats_param.detach(), Ks, W, H)
        residual = (rends.squeeze(0).permute(0,3,1,2) - images).abs().mean(dim=1)
        thr = torch.tensor([residual[fi].median() + args.tau * residual[fi].std()
                            for fi in range(n_frames)], device=device).median()
        u, v, valid = project_gaussians()
        res_at_g = residual.view(n_frames, -1).gather(1, v * W + u)
        dyn_obs = (res_at_g > thr) & valid
        return (((dyn_obs.float().sum(0) >= 3) & (valid.float().sum(0) > 3))
                .nonzero(as_tuple=True)[0])

    opt = torch.optim.Adam([
        {"params": [means, quats, scales, opacities, colors], "lr": 1e-2},
        {"params": [viewmats_param], "lr": 5e-5},
    ])
    print(f"mode={args.mode} | {n_g} 高斯 | {args.iters} iters")
    for it in range(args.iters):
        if args.mode != "none" and it > 0 and it % args.anti_period == 0:
            if args.mode == "selective":
                sel = selective_select()
                if len(sel): opacities.data.view(n_g)[sel] *= 0.9
            elif args.mode == "random_gauss":
                # 数量匹配 selective（近似：用同比例 ~22%）
                k = int(n_g * 0.22)
                idx = torch.randperm(n_g, device=device)[:k]
                opacities.data.view(n_g)[idx] *= 0.9
        opt.zero_grad()
        rends, _, _ = rasterization(means, quats, scales, opacities, colors,
                                    viewmats_param, Ks, W, H)
        loss = torch.nn.functional.mse_loss(rends.squeeze(0).permute(0,3,1,2), images)
        loss.backward(); opt.step()

    # 分区评估
    with torch.no_grad():
        rends, _, _ = rasterization(means.detach(), quats.detach(), scales.detach(),
                                    opacities.detach(), colors.detach(),
                                    viewmats_param.detach(), Ks, W, H)
        preds = rends.squeeze(0).permute(0,3,1,2)  # (N,3,H,W)
        sq = (preds - images) ** 2  # (N,3,H,W)
        sq_map = sq.mean(dim=1)  # (N,H,W)
        dyn = (dyn_t > 0.5)
        stat = ~dyn
        glob = (-10 * torch.log10(sq_map.mean() + 1e-8)).item()
        dyn_p = (-10 * torch.log10(sq_map[dyn].mean() + 1e-8)).item()
        st_p = (-10 * torch.log10(sq_map[stat].mean() + 1e-8)).item()

    summary = {"mode": args.mode, "seed": args.seed,
               "psnr_global": round(glob, 2), "psnr_dynamic": round(dyn_p, 2),
               "psnr_static": round(st_p, 2), "n_gauss": n_g}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== mode={args.mode} ===")
    print(f"全局 PSNR={glob:.2f} | 动态区={dyn_p:.2f} | 静态区={st_p:.2f} dB")
    print(f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
