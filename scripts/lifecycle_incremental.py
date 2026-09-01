#!/usr/bin/env python3
"""Phase 0: 递增式训练 + 高斯生命周期管理（最小实现）。

与全帧联合优化的本质区别：
  - 帧 0 初始化 → 帧依次处理（tracking + mapping + 致密化 + 生命周期仲裁）
  - 鬼影高斯可被"新证据"回收；候选高斯需跨帧一致才转正

对照组设计（同框架内公平对比）：
  incremental       — 递增式训练（无生命周期，致密化即时提交）= 递增 baseline
  lifecycle         — 递增 + 生命周期（候选池延迟提交 + 冲突回收）

指标：全局 PSNR / 动态区 PSNR（Bonn GT mask）/ 鬼影高斯计数。

用法：
  python3 scripts/lifecycle_incremental.py --mode incremental --out results/lc_incr
  python3 scripts/lifecycle_incremental.py --mode lifecycle  --out results/lc_life
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch


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


def densify_from_residual(residual, image, depth, pose, K, n_add, voxel_set):
    """从高残差像素反投影生成新高斯（候选）。返回 (pts_w, cols)。"""
    H, W = residual.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    # 只在深度有效的高残差像素加
    flat = residual.flatten()
    k = min(n_add, len(flat))
    top_idx = np.argpartition(flat, -k)[-k:]
    ys, xs = np.unravel_index(top_idx, (H, W))
    pts_w, cols = [], []
    for y, x in zip(ys, xs):
        d = depth[y*2, x*2] if depth is not None else 0  # depth 是原始分辨率
        key = (int(x // 4), int(y // 4), int(d*10) if d > 0 else -1)
        if key in voxel_set:
            continue
        voxel_set.add(key)
        if d <= 0.2 or d > 8:
            continue
        pts_cam = np.array([(x*2-cx)/fx*d, (y*2-cy)/fy*d, d])
        pts_w.append(pose[:3,:3] @ pts_cam + pose[:3,3])
        cols.append(image[y, x].astype(np.float32) / 255.0)
    if not pts_w:
        return np.zeros((0,3)), np.zeros((0,3))
    return np.array(pts_w), np.array(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="results/depth_prior_bonn_balloon")
    ap.add_argument("--seq_dir", default="/data/Datasets/Bonn/rgbd_bonn_balloon")
    ap.add_argument("--mode", required=True, choices=["incremental", "lifecycle"])
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--map_iters", type=int, default=100, help="每帧 mapping 迭代数")
    ap.add_argument("--max_points", type=int, default=30000)
    ap.add_argument("--evidence_K", type=int, default=3, help="候选转正所需一致帧数")
    ap.add_argument("--tau", type=float, default=2.5)
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
    fx_o, fy_o, cx_o, cy_o = K_orig[0,0], K_orig[1,1], K_orig[0,2], K_orig[1,2]

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
        images.append(cv2.resize(img, (W, H)))
        T = match_gt(gt, gt_ts, float(ts))
        poses.append(T if T is not None else np.eye(4))
    images_t = torch.stack([torch.from_numpy(im.transpose(2,0,1)).float()/255.0 for im in images]).to(device)
    poses_gt = np.array(poses)
    viewmats_gt = np.linalg.inv(poses_gt)
    viewmats = torch.tensor(viewmats_gt, dtype=torch.float32).unsqueeze(0).to(device).requires_grad_(True)  # (1,N,4,4)
    Ks = torch.from_numpy(K).float().to(device).unsqueeze(0).expand(1, n_frames, 3, 3).contiguous()

    # GT 动态 mask
    cand = sorted((seq_dir / "dynamic_mask_gtmc").glob("*.png"))
    dyn_masks = []
    for i in range(n_frames):
        if i < len(cand):
            m = cv2.imread(str(cand[i]), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            dyn_masks.append((m > 127).astype(np.float32))
        else:
            dyn_masks.append(np.zeros((H, W), dtype=np.float32))
    dyn_t = torch.from_numpy(np.stack(dyn_masks)).to(device)

    # ---- 高斯池（动态增删）----
    # 初始化：帧 0 深度反投影
    dep0 = depths[0]
    yi, xi = np.mgrid[0:dep0.shape[0]:4, 0:dep0.shape[1]:4]
    d0 = dep0[::4, ::4]; v0 = (d0 > 0.2) & (d0 < 8)
    yc0, xc0, dc0 = yi[v0], xi[v0], d0[v0]
    pts_cam0 = np.stack([(xc0-cx_o)/fx_o*dc0, (yc0-cy_o)/fy_o*dc0, dc0], axis=-1)
    img0 = cv2.cvtColor(cv2.imread(str(seq_dir / meta["image_paths"][0])), cv2.COLOR_BGR2RGB)
    cols0 = img0[yc0, xc0].astype(np.float32) / 255.0
    pts_w0 = (poses_gt[0][:3,:3] @ pts_cam0.T).T + poses_gt[0][:3,3]

    # 下采样初始化
    if len(pts_w0) > args.max_points // 2:
        sel = np.random.choice(len(pts_w0), args.max_points // 2, replace=False)
        pts_w0, cols0 = pts_w0[sel], cols0[sel]

    means_list = [torch.tensor(pts_w0, dtype=torch.float32, device=device)]
    colors_list = [torch.tensor(cols0, dtype=torch.float32, device=device)]
    scales_list = [torch.full((len(pts_w0), 3), 0.02, device=device)]
    status = ["established"] * len(pts_w0)   # established / candidate
    evidence = [args.evidence_K] * len(pts_w0)
    voxel_set = set((int(x//8), int(y//8)) for x, y in zip(xc0, yc0))

    from gsplat import rasterization

    opt = None  # 每帧重建 optimizer（池会变）
    def rebuild_opt(means, quats, scales, opacities, colors):
        return torch.optim.Adam([
            {"params": [means, quats, scales, opacities, colors], "lr": 1e-2},
            {"params": [viewmats], "lr": 5e-5},
        ])

    n_frames_proc = min(n_frames, 44)
    print(f"mode={args.mode} | 递增处理 {n_frames_proc} 帧 | init {len(status)} 高斯")

    for t in range(1, n_frames_proc):
        n_est = status.count("established")
        means = means_list[0] if len(means_list) == 1 else torch.cat(means_list, 0)
        colors = colors_list[0] if len(colors_list) == 1 else torch.cat(colors_list, 0)
        scales = scales_list[0] if len(scales_list) == 1 else torch.cat(scales_list, 0)
        n_g = means.shape[0]
        quats = torch.zeros(n_g, 4, device=device); quats[:, 0] = 1.0
        opacities = torch.where(torch.tensor([s == "candidate" for s in status], device=device),
                                torch.full((n_g,), 0.3, device=device),
                                torch.full((n_g,), 0.9, device=device))
        means = means.detach().requires_grad_(True)
        colors = colors.detach().requires_grad_(True)
        scales = scales.detach().requires_grad_(True)
        quats = quats.requires_grad_(True)
        opacities = opacities.requires_grad_(True)
        means_list = [means]; colors_list = [colors]; scales_list = [scales]

        opt = rebuild_opt(means, quats, scales, opacities, colors)
        vm_t = viewmats[:, t:t+1].detach().clone().requires_grad_(True)
        Kt = Ks[:, t:t+1]

        # mapping：当前帧 + 随机历史帧窗口联合优化（防灾难性遗忘）
        hist_idx = list(np.random.choice(t, min(5, t), replace=False)) if t > 0 else []
        for _ in range(args.map_iters):
            opt.zero_grad()
            loss = 0
            for fi in [t] + hist_idx:
                vm_f = viewmats[:, fi:fi+1].detach() if fi != t else vm_t
                r_f, _, _ = rasterization(means.unsqueeze(0), quats.unsqueeze(0), scales.unsqueeze(0),
                                          opacities.unsqueeze(0), colors.unsqueeze(0),
                                          vm_f, Ks[:, fi:fi+1], W, H)
                loss = loss + torch.nn.functional.mse_loss(r_f.reshape(3, H, W), images_t[fi])
            loss = loss / (1 + len(hist_idx))
            loss.backward(); opt.step()

        # 致密化：当前帧高残差区域 → 候选高斯
        with torch.no_grad():
            r, _, _ = rasterization(means.unsqueeze(0), quats.unsqueeze(0), scales.unsqueeze(0),
                                    opacities.unsqueeze(0), colors.unsqueeze(0),
                                    vm_t, Kt, W, H)
            residual = (r.reshape(3, H, W) - images_t[t]).abs().mean(dim=0).cpu().numpy()
            med = np.median(residual); std = residual.std() + 1e-8
            new_pts, new_cols = densify_from_residual(
                residual, images[t], depths[t], poses_gt[t], K_orig,
                n_add=min(2000, int((residual > med + args.tau*std).sum() * 0.5)) + 10,
                voxel_set=voxel_set)
            if len(new_pts) > 0:
                means_list.append(torch.tensor(new_pts, dtype=torch.float32, device=device))
                colors_list.append(torch.tensor(new_cols, dtype=torch.float32, device=device))
                scales_list.append(torch.full((len(new_pts), 3), 0.05, device=device))
                status += ["candidate"] * len(new_pts)
                evidence += [0] * len(new_pts)

        # 生命周期仲裁
        if args.mode == "lifecycle":
            with torch.no_grad():
                u = (K[0,0].item() * (vm_t[0,0,0,0]*0 + 1))  # 占位，实际用投影
                # 投影所有高斯到当前帧，采样残差更新 evidence
                mw = means.detach()
                ones = torch.ones(n_g, 1, device=device)
                cam_pts = torch.einsum("ij,gj->gi", torch.cat([vm_t[0,0].detach(), torch.eye(4, device=device)])[:3,:4] * 0 + vm_t[0,0].detach(), torch.cat([mw, ones], 1))
                depth_c = cam_pts[:, 2]
                valid = depth_c > 0.05
                uu = (K[0,0].item() * cam_pts[:,0] / depth_c.clamp(min=1e-6) + K[0,2].item()).clamp(0, W-1).long()
                vv = (K[1,1].item() * cam_pts[:,1] / depth_c.clamp(min=1e-6) + K[1,2].item()).clamp(0, H-1).long()
                res_at_g = torch.from_numpy(residual).to(device)[vv, uu]
                med_t = np.median(residual); std_t = residual.std() + 1e-8
                consistent = (res_at_g <= med_t + args.tau * std_t) & valid
                for gi in range(n_g):
                    if not valid[gi]:
                        continue
                    if status[gi] == "candidate":
                        evidence[gi] = evidence[gi] + 1 if consistent[gi] else evidence[gi] - 1
                        if evidence[gi] >= args.evidence_K:
                            status[gi] = "established"
                            opacities.data[gi] = 0.9
                        elif evidence[gi] <= -args.evidence_K:
                            status[gi] = "dead"
                            opacities.data[gi] = 0.0
                    # established 不降级（Phase 0 简化）

        if t % 10 == 0:
            print(f"  frame {t}: {n_g} gaussians ({status.count('candidate')} candidate) | loss {loss.item():.4f}")

    # ---- 最终评估：渲染全部帧 ----
    means = torch.cat(means_list, 0)
    colors = torch.cat(colors_list, 0)
    scales = torch.cat(scales_list, 0)
    n_g = means.shape[0]
    alive = torch.tensor([s != "dead" for s in status], device=device)
    quats = torch.zeros(n_g, 4, device=device); quats[:, 0] = 1.0
    opacities = torch.where(alive & torch.tensor([s == "established" for s in status], device=device),
                            torch.full((n_g,), 0.9, device=device),
                            torch.where(alive, torch.full((n_g,), 0.3, device=device),
                                        torch.zeros(n_g, device=device)))
    with torch.no_grad():
        rends, _, _ = rasterization(means.unsqueeze(0), quats.unsqueeze(0), scales.unsqueeze(0),
                                    opacities.unsqueeze(0), colors.unsqueeze(0),
                                    viewmats[:, :n_frames_proc].detach(), Ks[:, :n_frames_proc], W, H)
        preds = rends.squeeze(0).permute(0, 3, 1, 2)
        sq_map = ((preds - images_t[:n_frames_proc])**2).mean(dim=1)
        dyn = (dyn_t[:n_frames_proc] > 0.5)
        glob = (-10 * torch.log10(sq_map.mean() + 1e-8)).item()
        dyn_p = (-10 * torch.log10(sq_map[dyn].mean() + 1e-8)).item()
        st_p = (-10 * torch.log10(sq_map[~dyn].mean() + 1e-8)).item()

    n_dead = status.count("dead")
    n_cand = status.count("candidate")
    summary = {"mode": args.mode, "psnr_global": round(glob, 2),
               "psnr_dynamic": round(dyn_p, 2), "psnr_static": round(st_p, 2),
               "n_gauss": n_g, "n_dead": n_dead, "n_candidate": n_cand,
               "n_frames": n_frames_proc}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== mode={args.mode} ===")
    print(f"全局={glob:.2f} 动态区={dyn_p:.2f} 静态区={st_p:.2f} dB")
    print(f"高斯: {n_g} 总 / {n_dead} dead / {n_cand} candidate | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
