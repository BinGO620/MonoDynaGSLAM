#!/usr/bin/env python3
"""Phase 2a: PnP 跟踪 + Metric3D 深度先验 + gsplat 重建（非 GT 初始化）。

前端：Metric3D 单帧深度 → 特征匹配 → PnP 帧间位姿估计
后端：gsplat + Gaussian anti（Step 2 管线）
评估：ATE + PSNR

这就是"真正从图像估计位姿"的最小闭环。

用法：
  python3 scripts/phase2_pnp_tracking.py --prior results/depth_prior_walking \
      --seq_dir /data/Datasets/TUM/rgbd_dataset_freiburg3_walking_xyz --iters 2500
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


def pnp_pose_estimation(img_curr, img_prev, depth_prev, K, max_features=2000):
    """ORB 特征匹配 + PnP-RANSAC 估计帧间位姿（T_cam_curr ← T_cam_prev）。"""
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(img_prev, cv2.COLOR_RGB2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY), None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return None, len(good)

    pts_3d = []; pts_2d = []
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    for m in good:
        u, v = int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])
        d = depth_prev[v, u]
        if d <= 0.2 or d > 8: continue
        x = (u - cx) / fx * d
        y = (v - cy) / fy * d
        pts_3d.append([x, y, d])
        pts_2d.append(kp2[m.trainIdx].pt)
    pts_3d = np.array(pts_3d, dtype=np.float64)
    pts_2d = np.array(pts_2d, dtype=np.float64)

    if len(pts_3d) < 8:
        return None, len(pts_3d)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K, None,
        iterationsCount=200, reprojectionError=4.0,
        confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success or inliers is None or len(inliers) < 6:
        return None, len(inliers) if inliers is not None else 0

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = tvec.flatten()
    return T, len(inliers)


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
    ap.add_argument("--prior", required=True)
    ap.add_argument("--seq_dir", default=None)
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--max_points", type=int, default=60000)
    ap.add_argument("--anti", action="store_true")
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        tag = "pnp_anti" if args.anti else "pnp"
        args.out = f"results/phase2_{tag}"
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

    gt = {}
    for line in (seq_dir / "groundtruth.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        p = line.split(); t = float(p[0])
        T = np.eye(4); T[:3,:3] = quat_to_R(*[float(x) for x in p[4:8]]); T[:3,3] = [float(p[1]),float(p[2]),float(p[3])]
        gt[t] = T
    gt_ts = sorted(gt.keys())

    # ---- 加载图像 + 深度 + GT ----
    images = []; all_depths = []; poses_gt = []
    for ts, rel in zip(meta["timestamps"], meta["image_paths"]):
        img = cv2.cvtColor(cv2.imread(str(seq_dir / rel)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))
        images.append(torch.from_numpy(img).permute(2,0,1).float()/255.0)
        T = match_gt(gt, gt_ts, float(ts))
        poses_gt.append(torch.from_numpy(T).float() if T is not None else torch.eye(4))
    images = torch.stack(images).to(device)
    poses_gt = torch.stack(poses_gt).to(device)

    # ---- PnP 跟踪（在原始分辨率上做） ----
    print(f"PnP 跟踪: {n_frames} 帧, {W}x{H}")
    pnp_poses = [np.eye(4)]  # 第一帧为单位阵
    n_good = 0; n_fail = 0
    for i in range(1, n_frames):
        img_prev = cv2.cvtColor(cv2.imread(str(seq_dir / meta["image_paths"][i-1])), cv2.COLOR_BGR2RGB)
        img_curr = cv2.cvtColor(cv2.imread(str(seq_dir / meta["image_paths"][i])), cv2.COLOR_BGR2RGB)
        dep_prev = depths[i-1]
        T_rel, n_inliers = pnp_pose_estimation(img_curr, img_prev, dep_prev, K_orig)
        if T_rel is not None:
            # T_rel: cam_curr ← cam_prev (in cam_prev coords)
            # 世界系位姿: T_world_curr = T_world_prev @ T_rel
            pnp_poses.append(pnp_poses[-1] @ T_rel)
            n_good += 1
        else:
            pnp_poses.append(pnp_poses[-1].copy())  # 失败时保持上一帧位姿
            n_fail += 1
    print(f"PnP: {n_good} 成功 / {n_fail} 失败")

    # 保存 PnP 位姿为 TUM 格式
    pnp_lines = ["# timestamp tx ty tz qx qy qz qw"]
    for i, (ts, _) in enumerate(zip(meta["timestamps"], meta["image_paths"])):
        T = pnp_poses[i]
        q = T[:3,:3]; tx, ty, tz = T[:3, 3]
        trace = np.trace(q)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw, qx, qy, qz = 0.25/s, (q[2,1]-q[1,2])*s, (q[0,2]-q[2,0])*s, (q[1,0]-q[0,1])*s
        else:
            if q[0,0]>q[1,1] and q[0,0]>q[2,2]:
                s = 2.0*np.sqrt(1.0+q[0,0]-q[1,1]-q[2,2])
                qw, qx, qy, qz = (q[2,1]-q[1,2])/s, 0.25*s, (q[0,1]+q[1,0])/s, (q[0,2]+q[2,0])/s
            elif q[1,1]>q[2,2]:
                s = 2.0*np.sqrt(1.0+q[1,1]-q[0,0]-q[2,2])
                qw, qx, qy, qz = (q[0,2]-q[2,0])/s, (q[0,1]+q[1,0])/s, 0.25*s, (q[1,2]+q[2,1])/s
            else:
                s = 2.0*np.sqrt(1.0+q[2,2]-q[0,0]-q[1,1])
                qw, qx, qy, qz = (q[1,0]-q[0,1])/s, (q[0,2]+q[2,0])/s, (q[1,2]+q[2,1])/s, 0.25*s
        pnp_lines.append(f"{ts} {tx} {ty} {tz} {qx} {qy} {qz} {qw}")
    (out_dir / "pnp_trajectory.txt").write_text("\n".join(pnp_lines))
    print(f"PnP 位姿已保存: {out_dir}/pnp_trajectory.txt")

    # ---- ATE 评估 PnP 位姿（Umeyama 对齐） ----
    def tum_poses_from_file(path):
        ts_l, ps = [], []
        for line in open(path):
            if line.startswith("#") or not line.strip(): continue
            p = line.split()
            t = float(p[0]); T = np.eye(4)
            T[:3, 3] = [float(p[1]), float(p[2]), float(p[3])]
            qx, qy, qz, qw = [float(p[i]) for i in [4,5,6,7]]
            R = quat_to_R(qx, qy, qz, qw); T[:3,:3] = R
            ts_l.append(t); ps.append(T)
        return np.array(ts_l), np.array(ps)

    gt_ts_arr, gt_poses_arr = tum_poses_from_file(str(seq_dir / "groundtruth.txt"))
    pnp_ts_arr, pnp_poses_arr = tum_poses_from_file(str(out_dir / "pnp_trajectory.txt"))

    est_matched = []; gt_matched = []
    for gt_t, gt_p in zip(gt_ts_arr, gt_poses_arr):
        idx = np.argmin(np.abs(pnp_ts_arr - gt_t))
        if np.abs(pnp_ts_arr[idx] - gt_t) < 0.02:
            est_matched.append(pnp_poses_arr[idx][:3, 3])
            gt_matched.append(gt_p[:3, 3])
    est_matched = np.array(est_matched); gt_matched = np.array(gt_matched)
    if len(est_matched) >= 3:
        src_c, dst_c = est_matched.mean(0), gt_matched.mean(0)
        H = (est_matched-src_c).T @ (gt_matched-dst_c)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0: Vt[-1]*=-1; R = Vt.T @ U.T
        est_aligned = ((R @ est_matched.T).T + dst_c - R @ src_c)
        ate = np.sqrt(np.mean(np.sum((est_aligned - gt_matched)**2, axis=1)))
        print(f"\n=== PnP ATE (非GT初始化) ===")
        print(f"  RMSE: {ate*100:.1f} cm")
        print(f"  匹配帧: {len(est_matched)}")
    else:
        print("匹配帧数不足")
        ate = 0

    # ---- 用 PnP 位姿做 gsplat 重建（Gaussian anti） ----
    print(f"\ngsplat 重建（用 PnP 位姿初始化）...")
    Ks = torch.from_numpy(K).float().to(device).unsqueeze(0).expand(1, n_frames, 3, 3).contiguous()
    poses_pnp = torch.tensor(np.array(pnp_poses), dtype=torch.float32).to(device)
    viewmats_pnp = poses_pnp.inverse().unsqueeze(0).to(device)

    # 多帧反投影点云
    all_pts, all_cols = [], []
    for i in range(n_frames):
        dep = depths[i]; yi,xi = np.mgrid[0:dep.shape[0]:4, 0:dep.shape[1]:4]
        d = dep[::4,::4]; v = (d>0.2)&(d<8)
        if v.sum()==0: continue
        fx_o, fy_o, cx_o, cy_o = K_orig[0,0], K_orig[1,1], K_orig[0,2], K_orig[1,2]
        yc,xc,dc = yi[v], xi[v], d[v]
        pts_cam = np.stack([(xc-cx_o)/fx_o*dc, (yc-cy_o)/fy_o*dc, dc], axis=-1)
        img_o = cv2.cvtColor(cv2.imread(str(seq_dir/meta["image_paths"][i])), cv2.COLOR_BGR2RGB)
        cols = img_o[yc,xc].astype(np.float32)/255.0
        pose = pnp_poses[i]
        pts_w = (pose[:3,:3] @ pts_cam.T).T + pose[:3,3]
        all_pts.append(pts_w); all_cols.append(cols)
    pts, cols = voxel_downsample(np.concatenate(all_pts), np.concatenate(all_cols), 0.02, args.max_points)
    n_g = len(pts); print(f"高斯数 {n_g}")

    pts_t = torch.from_numpy(pts).float().to(device)
    cols_t = torch.from_numpy(cols).float().to(device)
    means = pts_t.unsqueeze(0).requires_grad_(True)
    quats = torch.zeros(1,n_g,4,device=device); quats[...,0]=1.0; quats.requires_grad_(True)
    scales = torch.full((1,n_g,3),0.02,device=device).requires_grad_(True)
    opacities = torch.full((1,n_g),0.9,device=device).requires_grad_(True)
    colors = cols_t.unsqueeze(0).clone().requires_grad_(True)
    viewmats_param = viewmats_pnp.clone().requires_grad_(True)

    from gsplat import rasterization

    @torch.no_grad()
    def anti_update():
        if not args.anti: return
        rends, _, _ = rasterization(means, quats, scales, opacities.detach(), colors,
                                    viewmats_param.detach(), Ks, W, H)
        residual = (rends.squeeze(0).permute(0,3,1,2) - images).abs().mean(dim=1)
        for fi in range(n_frames):
            med = residual[fi].median(); std = residual[fi].std()+1e-8
            dyn_frac = (residual[fi] > med + args.tau*std).float().mean().item()
            if dyn_frac > 0.01:
                opacities.data.mul_(max(0.9, 1.0 - 0.02*dyn_frac*10))

    opt = torch.optim.Adam([
        {"params": [means, quats, scales, opacities, colors], "lr": 1e-2},
        {"params": [viewmats_param], "lr": 5e-5},
    ])
    for it in range(args.iters):
        if args.anti and it > 0 and it % 200 == 0: anti_update()
        opt.zero_grad()
        r, _, _ = rasterization(means, quats, scales, opacities, colors,
                                viewmats_param, Ks, W, H)
        loss = torch.nn.functional.mse_loss(r.squeeze(0).permute(0,3,1,2), images)
        loss.backward(); opt.step()
        if it % 500 == 0 or it == args.iters-1:
            print(f"  {it}: {(-10*torch.log10(loss+1e-8)).item():.1f} dB")

    # 评估
    with torch.no_grad():
        rends, _, _ = rasterization(means.detach(), quats.detach(), scales.detach(),
                                    opacities.detach(), colors.detach(),
                                    viewmats_param.detach(), Ks, W, H)
        mse = torch.nn.functional.mse_loss(rends.squeeze(0).permute(0,3,1,2), images).item()
        psnr = -10*np.log10(mse+1e-8)

    # 优化后 ATE
    pose_opt = viewmats_param.detach().squeeze(0).cpu().numpy()
    est_opt = []; gt_opt = []
    for i in range(n_frames):
        c2w = np.linalg.inv(pose_opt[i])
        ts = float(meta["timestamps"][i])
        idx = np.argmin(np.abs(gt_ts_arr - ts))
        if np.abs(gt_ts_arr[idx] - ts) < 0.02:
            est_opt.append(c2w[:3,3]); gt_opt.append(gt_poses_arr[idx][:3,3])
    est_opt = np.array(est_opt); gt_opt = np.array(gt_opt)
    if len(est_opt) >= 3:
        src_c, dst_c = est_opt.mean(0), gt_opt.mean(0)
        H = (est_opt-src_c).T @ (gt_opt-dst_c)
        U,S,Vt = np.linalg.svd(H); R = Vt.T @ U.T
        if np.linalg.det(R)<0: Vt[-1]*=-1; R=Vt.T @ U.T
        ate_opt = np.sqrt(np.mean(np.sum(((R @ est_opt.T).T + dst_c - R @ src_c - gt_opt)**2, axis=1)))
    else:
        ate_opt = 0

    summary = {
        "pnp_ate_cm": round(float(ate*100), 1),
        "optimized_ate_cm": round(float(ate_opt*100), 1),
        "psnr": round(float(psnr), 2),
        "pnp_success_rate": round(n_good / (n_frames-1) * 100, 1),
        "n_gauss": n_g, "n_frames": n_frames
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Phase 2a 完整结果 ===")
    print(f"PnP 位姿 ATE: {summary['pnp_ate_cm']} cm (跟踪成功率 {summary['pnp_success_rate']}%)")
    print(f"优化后 ATE: {summary['optimized_ate_cm']} cm")
    print(f"PSNR: {summary['psnr']} dB")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
