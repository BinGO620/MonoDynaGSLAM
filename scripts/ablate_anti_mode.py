#!/usr/bin/env python3
"""Anti 机制对照消融：区分"残差信号的真实贡献"与"隐式正则化"。

四种模式：
  none            — baseline（无 anti）
  global_residual — 当前实现：帧级动态比例调制的全局 opacity 衰减
  random_decay    — 对照B：不看残差，固定衰减全部 opacity（检验残差信号是否有信息量）
  pixel_mask      — 对照A：高残差像素从 loss 中 mask 掉（Dy3DGS 类标准做法）

用法：
  python3 scripts/ablate_anti_mode.py --prior results/depth_prior_walking \
      --seq_dir /data/Datasets/TUM/rgbd_dataset_freiburg3_walking_xyz \
      --mode pixel_mask --iters 2500 --out results/abl_pixel_mask
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
    ap.add_argument("--prior", required=True)
    ap.add_argument("--seq_dir", default=None)
    ap.add_argument("--mode", required=True,
                    choices=["none", "global_residual", "random_decay", "pixel_mask",
                             "selective", "random_gauss", "combined"])
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--max_points", type=int, default=60000)
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--anti_period", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1, help="训练帧间隔，>1 时评估全部帧")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
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
    def project_gaussians_to_pixels(viewmats):
        """把高斯中心投影到每帧像素坐标，返回像素坐标 (N,n_g) 与有效深度掩码 (N,n_g)。"""
        mw = means.squeeze(0)  # (n_g,3)
        ones = torch.ones(n_g, 1, device=device)
        mw_h = torch.cat([mw, ones], dim=1)  # (n_g,4)
        vm = viewmats.squeeze(0)  # (N,4,4)
        cam_pts = torch.einsum("nij,gj->ngi", vm, mw_h)  # (N, n_g, 4)
        depth = cam_pts[..., 2]  # (N, n_g)
        valid = depth > 0.05
        fx_, fy_ = K[0, 0].item(), K[1, 1].item()
        cx_, cy_ = K[0, 2].item(), K[1, 2].item()
        u = fx_ * cam_pts[..., 0] / depth.clamp(min=1e-6) + cx_
        v = fy_ * cam_pts[..., 1] / depth.clamp(min=1e-6) + cy_
        return u, v, valid

    @torch.no_grad()
    def selective_gaussian_mask():
        """F/D 共用：残差归因选择高斯。返回被选中的高斯索引列表。"""
        rends, _, _ = rasterization(means, quats, scales, opacities.detach(), colors,
                                    viewmats_param.detach(), Ks, W, H)
        residual = (rends.squeeze(0).permute(0, 3, 1, 2) - images).abs().mean(dim=1)  # (N,H,W)
        # 全局归一化的残差阈值（每帧 median+tau*std 的中位数，保证跨帧一致）
        threshs = []
        for fi in range(n_frames):
            med = residual[fi].median(); std = residual[fi].std() + 1e-8
            threshs.append(med + args.tau * std)
        thr_global = torch.tensor(threshs, device=device).median()
        u, v, valid = project_gaussians_to_pixels(viewmats_param.detach())
        # 采样残差：线性索引 gather（低显存）
        u_c = u.clamp(0, W - 1).long()
        v_c = v.clamp(0, H - 1).long()
        flat = residual.view(n_frames, -1)  # (N, H*W)
        lin = v_c * W + u_c  # (N, n_g)
        res_at_g = flat.gather(1, lin)  # (N, n_g)
        dyn_obs = (res_at_g > thr_global) & valid  # (N, n_g) 该高斯在有效深度下落在高残差像素
        # 聚合跨帧：出现在高残差像素的帧占比 > 50% 的帧数
        dyn_frame_count = dyn_obs.float().sum(dim=0)  # (n_g,)
        # 选择标准：高斯至少在 3 帧中被观察到高残差，且观察帧数 > 0
        observed = valid.float().sum(dim=0)  # (n_g,) 有效观察帧数
        selected = ((dyn_frame_count >= 3) & (observed > 5)).nonzero(as_tuple=True)[0]
        return selected

    @torch.no_grad()
    def anti_update_selective():
        """F: 真正的 selective Gaussian anti（投影归因 + 逐高斯衰减）。"""
        selected = selective_gaussian_mask()
        if len(selected) == 0: return
        opacities.data.view(n_g)[selected] *= 0.9  # 每个被选中高斯独立衰减
        return len(selected)

    @torch.no_grad()
    def anti_update_random_gauss():
        """D: 随机选择与 F 相同数量的高斯衰减。"""
        selected_true = selective_gaussian_mask()
        k = len(selected_true)
        if k == 0: return 0
        rand_idx = torch.randperm(n_g, device=device)[:k]
        opacities.data.view(n_g)[rand_idx] *= 0.9
        return k

    @torch.no_grad()
    def anti_update_global_residual():
        """当前实现：帧级动态比例 → 全局 opacity 衰减。"""
        rends, _, _ = rasterization(means, quats, scales, opacities.detach(), colors,
                                    viewmats_param.detach(), Ks, W, H)
        residual = (rends.squeeze(0).permute(0,3,1,2) - images).abs().mean(dim=1)
        for fi in range(n_frames):
            med = residual[fi].median(); std = residual[fi].std() + 1e-8
            df = (residual[fi] > med + args.tau * std).float().mean().item()
            if df > 0.01:
                opacities.data.mul_(max(0.9, 1.0 - 0.02 * df * 10))

    @torch.no_grad()
    def anti_update_random_decay():
        """对照B：固定衰减，无残差信息。幅度与 global_residual 的平均衰减匹配。"""
        # 估计平均 dyn_frac（用当前残差，但衰减不依赖它——只是幅度匹配）
        rends, _, _ = rasterization(means, quats, scales, opacities.detach(), colors,
                                    viewmats_param.detach(), Ks, W, H)
        residual = (rends.squeeze(0).permute(0,3,1,2) - images).abs().mean(dim=1)
        dfs = []
        for fi in range(n_frames):
            med = residual[fi].median(); std = residual[fi].std() + 1e-8
            dfs.append((residual[fi] > med + args.tau * std).float().mean().item())
        avg_df = np.mean(dfs)
        # 关键区别：衰减幅度固定，与动态比例无关
        opacities.data.mul_(max(0.9, 1.0 - 0.02 * avg_df * 10))

    opt = torch.optim.Adam([
        {"params": [means, quats, scales, opacities, colors], "lr": 1e-2},
        {"params": [viewmats_param], "lr": 5e-5},
    ])

    train_frames = list(range(0, n_frames, args.stride))
    if args.stride > 1:
        print(f"HOLD-OUT 协议: 训练 {len(train_frames)}/{n_frames} 帧 (stride={args.stride})")
    print(f"mode={args.mode} | 高斯 {n_g} | 帧 {n_frames} | tau {args.tau}")
    for it in range(args.iters):
        do_anti = (args.mode != "none" and it > 0 and it % args.anti_period == 0)
        if do_anti:
            if args.mode == "global_residual":
                anti_update_global_residual()
            elif args.mode == "random_decay":
                anti_update_random_decay()
            elif args.mode == "selective":
                n_sel = anti_update_selective()
                if it % 500 == 0: print(f"    [selective] {n_sel} gaussians suppressed")
            elif args.mode == "combined":
                anti_update_global_residual()
                n_sel = anti_update_selective()
                if it % 500 == 0: print(f"    [combined] {n_sel} selective + global")
            elif args.mode == "random_gauss":
                n_sel = anti_update_random_gauss()

        opt.zero_grad()
        rends, _, _ = rasterization(means, quats, scales, opacities, colors,
                                    viewmats_param, Ks, W, H)
        imgs_pred = rends.squeeze(0).permute(0,3,1,2)

        if args.stride > 1:
            tidx = torch.tensor(train_frames, device=device)
            loss = torch.nn.functional.mse_loss(imgs_pred[tidx], images[tidx])
        elif args.mode == "pixel_mask" and do_anti:
            # 对照A：高残差像素 mask 掉（在 anti_update 时机生成 mask，持续到下个周期）
            with torch.no_grad():
                residual = (imgs_pred - images).abs().mean(dim=1)  # (N,H,W)
                med = residual.median(); std = residual.std() + 1e-8
                mask = (residual <= med + args.tau * std).float().unsqueeze(1)  # (N,1,H,W)
            main._pixel_mask = mask
            loss = (torch.nn.functional.mse_loss(imgs_pred, images, reduction='none').mean(dim=1, keepdim=True) * mask).mean()
        elif args.mode == "pixel_mask":
            mask = getattr(main, "_pixel_mask", None)
            if mask is not None:
                loss = (torch.nn.functional.mse_loss(imgs_pred, images, reduction='none').mean(dim=1, keepdim=True) * mask).mean()
            else:
                loss = torch.nn.functional.mse_loss(imgs_pred, images)
        else:
            loss = torch.nn.functional.mse_loss(imgs_pred, images)

        loss.backward(); opt.step()
        if it % 500 == 0 or it == args.iters - 1:
            psnr_full = -10*torch.log10(torch.nn.functional.mse_loss(imgs_pred, images)+1e-8).item()
            print(f"  {it}: loss={loss.item():.4f} psnr(full)={psnr_full:.1f}")

    with torch.no_grad():
        rends, _, _ = rasterization(means.detach(), quats.detach(), scales.detach(),
                                    opacities.detach(), colors.detach(),
                                    viewmats_param.detach(), Ks, W, H)
        psnr = -10*np.log10(torch.nn.functional.mse_loss(
            rends.squeeze(0).permute(0,3,1,2), images).item()+1e-8)
        per_frame = [round(-10*np.log10(torch.nn.functional.mse_loss(
            rends.squeeze(0)[i].permute(2,0,1), images[i]).item()+1e-8), 2) for i in range(n_frames)]

    summary = {"mode": args.mode, "tau": args.tau, "psnr_mean": round(float(psnr),2),
               "per_frame_psnr": per_frame, "n_gauss": n_g, "n_frames": n_frames,
               "iters": args.iters}
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== mode={args.mode} === PSNR={summary['psnr_mean']} dB | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
