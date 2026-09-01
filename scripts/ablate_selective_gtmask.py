#!/usr/bin/env python3
"""Selective anti + 动态 GT mask 量化归因（Bonn 专用，有 dynamic_mask_gtmc）。

量化：
1. 每轮 anti_update 记录被选中高斯 ID
2. 训练结束后，把选中高斯投影到各帧，统计其落在 GT 动态 mask 内的比例
3. 与随机选择同数量高斯对比 precision

用法：
  python3 scripts/ablate_selective_gtmask.py --prior results/depth_prior_bonn_balloon \
      --seq_dir /data/Datasets/Bonn/rgbd_bonn_balloon --iters 2500 \
      --out results/abl_gtmask
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ablate_anti_mode import voxel_downsample, quat_to_R, match_gt  # 复用工具


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="results/depth_prior_bonn_balloon")
    ap.add_argument("--seq_dir", default="/data/Datasets/Bonn/rgbd_bonn_balloon")
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--max_points", type=int, default=40000)
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--anti_period", type=int, default=200)
    ap.add_argument("--out", default="results/abl_gtmask")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

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

    # 加载 GT 动态 mask（resize 到训练分辨率）
    dyn_masks = []
    for ts, rel in zip(meta["timestamps"], meta["image_paths"]):
        mask_path = seq_dir / "dynamic_mask_gtmc" / f"{float(ts):.5f}.png"
        # 文件名格式不同，直接按 rgb 时间戳找最近的 mask
        cand = sorted((seq_dir / "dynamic_mask_gtmc").glob("*.png"))
        if not cand:
            print("无 GT mask"); return
        # 用帧序号对应（mask 序列与 rgb 同帧率采样时近似对应）
        idx = len(dyn_masks)
        if idx < len(cand):
            m = cv2.imread(str(cand[idx]), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            dyn_masks.append((m > 127).astype(np.uint8))
        else:
            dyn_masks.append(np.zeros((H, W), dtype=np.uint8))
    dyn_masks = np.stack(dyn_masks)  # (N,H,W)
    print(f"GT mask 平均动态占比: {dyn_masks.mean()*100:.1f}%")

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
        depth = cam_pts[..., 2]
        valid = depth > 0.05
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
        thr_global = torch.tensor([residual[fi].median() + args.tau * residual[fi].std()
                                   for fi in range(n_frames)], device=device).median()
        u, v, valid = project_gaussians()
        flat = residual.view(n_frames, -1)
        res_at_g = flat.gather(1, v * W + u)
        dyn_obs = (res_at_g > thr_global) & valid
        dyn_frame_count = dyn_obs.float().sum(dim=0)
        observed = valid.float().sum(dim=0)
        return ((dyn_frame_count >= 3) & (observed > 3)).nonzero(as_tuple=True)[0]

    # 动态 mask 张量
    dyn_t = torch.from_numpy(dyn_masks).float().to(device)  # (N,H,W)

    @torch.no_grad()
    def gt_precision(selected_ids):
        """选中高斯投影落在 GT 动态 mask 内的比例（跨帧平均）。"""
        if len(selected_ids) == 0: return 0.0
        u, v, valid = project_gaussians()
        u_s = u[:, selected_ids]; v_s = v[:, selected_ids]
        valid_s = valid[:, selected_ids]
        mvals = dyn_t.view(n_frames, -1).gather(1, v_s * W + u_s)  # (N,k)
        in_dyn = (mvals > 0.5) & valid_s
        precision = (in_dyn.float().sum() / valid_s.float().sum().clamp(min=1)).item()
        return precision

    opt = torch.optim.Adam([
        {"params": [means, quats, scales, opacities, colors], "lr": 1e-2},
        {"params": [viewmats_param], "lr": 5e-5},
    ])

    sel_history = []
    print(f"训练 {args.iters} iters | selective + GT mask 量化")
    for it in range(args.iters):
        if it > 0 and it % args.anti_period == 0:
            selected = selective_select()
            if len(selected) > 0:
                opacities.data.view(n_g)[selected] *= 0.9
            sel_history.append({"iter": it, "n_selected": len(selected)})
            # 每 500 iter 做一次 GT precision 量化
            if it % 500 == 0:
                prec = gt_precision(selected)
                print(f"  it={it} selected={len(selected)} ({len(selected)/n_g*100:.0f}%) GT-precision={prec*100:.1f}%")
        opt.zero_grad()
        rends, _, _ = rasterization(means, quats, scales, opacities, colors,
                                    viewmats_param, Ks, W, H)
        loss = torch.nn.functional.mse_loss(rends.squeeze(0).permute(0,3,1,2), images)
        loss.backward(); opt.step()

    # 最终评估
    with torch.no_grad():
        rends, _, _ = rasterization(means.detach(), quats.detach(), scales.detach(),
                                    opacities.detach(), colors.detach(),
                                    viewmats_param.detach(), Ks, W, H)
        psnr = -10*np.log10(torch.nn.functional.mse_loss(
            rends.squeeze(0).permute(0,3,1,2), images).item()+1e-8)

    # 最终选中集的 GT precision + 随机对照
    selected_final = selective_select()
    prec_f = gt_precision(selected_final)
    rand_ids = torch.randperm(n_g, device=device)[:len(selected_final)]
    prec_rand = gt_precision(rand_ids)

    summary = {
        "psnr": round(float(psnr), 2),
        "n_selected_final": int(len(selected_final)),
        "gt_precision_selective": round(prec_f*100, 1),
        "gt_precision_random": round(prec_rand*100, 1),
        "gt_mask_coverage": round(float(dyn_masks.mean()*100), 1),
        "sel_history": sel_history,
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Selective + GT mask 量化 ===")
    print(f"PSNR={summary['psnr']} dB")
    print(f"选中 {len(selected_final)}/{n_g} ({len(selected_final)/n_g*100:.1f}%)")
    print(f"GT precision: selective={summary['gt_precision_selective']}% vs random={summary['gt_precision_random']}%")
    print(f"GT mask 全局动态占比: {summary['gt_mask_coverage']}%")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
