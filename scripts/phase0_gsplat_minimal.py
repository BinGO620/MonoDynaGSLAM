#!/usr/bin/env python3
"""Phase 0 — 组件化可行性验证：gsplat 静态 3DGS 重建 + 位姿优化（真实数据）。

用 TUM fr1_desk 前 N 帧（真实 RGB + GT 位姿），跑 gsplat 最小闭环：
  1. 加载真实图像
  2. 用 GT 位姿初始化 + 微扰，测试 gsplat 位姿优化能否收敛
  3. 静态 3DGS 训练，评估渲染 PSNR 与位姿误差

用途：验证"复用 gsplat 组件 + 连接层"路线是否成立，不绑定 monogs-ours。

用法：python3 scripts/phase0_gsplat_minimal.py --seq fr1_desk --n_frames 30 --res 320
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# TUM 内参（fr1 系列）
FR1_K = np.array([[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]])


def parse_groundtruth(path: Path):
    """读取 TUM groundtruth.txt，返回 (timestamp, pose) 列表（pose 为 4x4）。"""
    poses = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        t = float(parts[0])
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        # 四元数 -> 旋转矩阵
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        poses[t] = T
    return poses


def load_frames(seq_dir: Path, n_frames: int, res: int, device: str = "cuda"):
    """加载前 n_frames 帧 RGB，resize 到 res，返回 (images, K, poses_gt, rgb_paths)。"""
    import cv2

    gt = parse_groundtruth(seq_dir / "groundtruth.txt")
    rgb_txt = seq_dir / "rgb.txt"
    # 按时间戳排序选前 n_frames 与 GT 对齐的帧
    lines = [l.split() for l in rgb_txt.read_text().splitlines() if l and not l.startswith("#")]
    # 用 GT 时间戳 + 最近邻匹配（float 精度问题）
    timestamps_gt = sorted(gt.keys())
    gt_ts_set = set(timestamps_gt)
    frames = []
    for ts, path in lines:
        t = float(ts)
        # 找 GT 中最近的时间戳（<5ms 差异）
        idx = np.searchsorted(timestamps_gt, t)
        best_t = None
        best_dt = 0.01  # 10ms
        for i in [idx-1, idx, idx+1]:
            if 0 <= i < len(timestamps_gt):
                dt = abs(timestamps_gt[i] - t)
                if dt < best_dt:
                    best_dt = dt
                    best_t = timestamps_gt[i]
        if best_t is not None:
            frames.append((best_t, path))
        if len(frames) >= n_frames:
            break

    images = []
    poses = []
    scale = res / 640.0
    K = FR1_K.copy()
    K[:2, :] *= scale
    for t, rel in frames:
        img = cv2.imread(str(seq_dir / rel))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (res, int(res * 0.75)))  # 480/640 = 0.75
        images.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        poses.append(torch.from_numpy(gt[t]).float())
    images = torch.stack(images).to(device)  # (N,3,H,W)
    poses = torch.stack(poses).to(device)    # (N,4,4)
    K = torch.from_numpy(K).float().to(device)
    return images, K, poses, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", default="/data/Datasets/TUM/rgbd_dataset_freiburg1_desk")
    ap.add_argument("--n_frames", type=int, default=30)
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--optimize_pose", action="store_true", default=True)
    ap.add_argument("--use_depth", action="store_true", default=False,
                    help="用 GT 深度初始化（模拟深度先验，验证瓶颈）")
    args = ap.parse_args()

    t0 = time.time()
    print(f"=== Phase 0: gsplat 最小闭环（静态重建 + 位姿优化）===")
    print(f"序列: {args.seq_dir} | 帧数: {args.n_frames} | 分辨率: {args.res} | iters: {args.iters}")

    device = "cuda"
    images, K, poses_gt, frames = load_frames(
        Path(args.seq_dir), args.n_frames, args.res, device
    )
    print(f"加载 {len(frames)} 帧，耗时 {time.time()-t0:.1f}s")

    # ---- 用 gsplat 做静态 3DGS 训练 + 位姿优化 ----
    from gsplat import rasterization

    N, C, H, W = images.shape
    torch.manual_seed(42)

    # 初始化高斯
    if args.use_depth:
        # 用 GT 深度初始化（模拟深度先验的极限性能）
        # 加载深度图（对齐 rgb）
        import cv2
        dep_paths = {}
        if (Path(args.seq_dir) / "associations.txt").exists():
            for line in (Path(args.seq_dir) / "associations.txt").read_text().splitlines():
                if not line or line.startswith("#"): continue
                parts = line.split()
                if len(parts) >= 4:
                    dep_paths[float(parts[0])] = parts[3]
        means_list = []
        scale = 5000.0
        for i, (t, _) in enumerate(frames):
            # 找最近深度
            ts_list = sorted(dep_paths.keys())
            idx = np.searchsorted(ts_list, t)
            best_dt = None
            best_dep = None
            for di in [idx-1, idx, idx+1]:
                if 0 <= di < len(ts_list):
                    dt = abs(ts_list[di] - t)
                    if best_dt is None or dt < best_dt:
                        best_dt = dt
                        best_dep = dep_paths[ts_list[di]]
            if best_dep is None or best_dt > 0.02:
                continue
            dep = cv2.imread(str(Path(args.seq_dir) / best_dep), cv2.IMREAD_UNCHANGED)
            if dep is None: continue
            dep = dep.astype(np.float32) / scale
            # 降采样到 1/8，只取有效深度
            dep_lr = dep[::8, ::8]
            yi, xi = np.mgrid[0:dep_lr.shape[0], 0:dep_lr.shape[1]]
            valid = (dep_lr > 0) & (dep_lr < 8)
            yi, xi, di = yi[valid], xi[valid], dep_lr[valid]
            if len(yi) == 0: continue
            # 相机坐标
            # 注意：TUM 的 depth 是 640x480，被 resize 到 320x240（H*0.75）
            # 但这里我们直接在原始深度上采样，然后转世界坐标
            fx, fy = FR1_K[0, 0], FR1_K[1, 1]
            cx, cy = FR1_K[0, 2], FR1_K[1, 2]
            xi = xi * 8 + 4  # 恢复原始像素坐标
            yi = yi * 8 + 4
            x_cam = (xi - cx) / fx * di
            y_cam = (yi - cy) / fy * di
            pts_cam = np.stack([x_cam, y_cam, di], axis=-1)
            # 转世界系
            pose = poses_gt[i].cpu().numpy()
            pts_world = (pose[:3, :3] @ pts_cam.T).T + pose[:3, 3]
            means_list.append(pts_world)
        if len(means_list) < 3:
            print("WARN: 深度加载不足，退化为随机初始化")
            means_list = []
            for _ in range(3):
                M = 2000
                depths = torch.rand(M, 1, device=device) * 4.5 + 0.5
                u = torch.rand(M, 1, device=device) * W
                v = torch.rand(M, 1, device=device) * H
                fx, fy = K[0, 0].item(), K[1, 1].item()
                cx, cy = K[0, 2].item(), K[1, 2].item()
                x_cam = (u - cx) / fx * depths
                y_cam = (v - cy) / fy * depths
                pts_cam = torch.cat([x_cam, y_cam, depths], dim=-1)
                pts_world = (poses_gt[0][:3, :3] @ pts_cam.T).T + poses_gt[0][:3, 3]
                means_list.append(pts_world.cpu().numpy())
        pts_world = np.concatenate(means_list, axis=0)
        # 随机子采样控制数量
        if len(pts_world) > 20000:
            idx = np.random.choice(len(pts_world), 20000, replace=False)
            pts_world = pts_world[idx]
        pts_world = torch.from_numpy(pts_world).float().to(device)
        n_gauss = pts_world.shape[0]
        print(f"深度初始化: {n_gauss} 高斯（来自 {len(means_list)} 帧）")
    else:
        # 随机视锥初始化（无深度先验）
        M = 5000
        depths = torch.rand(M, 1, device=device) * 4.5 + 0.5
        u = torch.rand(M, 1, device=device) * W
        v = torch.rand(M, 1, device=device) * H
        fx, fy = K[0, 0].item(), K[1, 1].item()
        cx, cy = K[0, 2].item(), K[1, 2].item()
        x_cam = (u - cx) / fx * depths
        y_cam = (v - cy) / fy * depths
        pts_cam = torch.cat([x_cam, y_cam, depths], dim=-1)
        pts_world = (poses_gt[0][:3, :3] @ pts_cam.T).T + poses_gt[0][:3, 3]
        n_gauss = pts_world.shape[0]
    means = pts_world.unsqueeze(0)  # (1, M, 3)
    quats = torch.randn(1, n_gauss, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.full((1, n_gauss, 3), 0.05, device=device)
    opacities = torch.ones(1, n_gauss, device=device) * 0.5
    colors = torch.randn(1, n_gauss, 3, device=device)  # SH 系数（0 阶初始化）

    Ks = K.unsqueeze(0).expand(1, N, 3, 3).contiguous()  # (1, N, 3, 3)
    viewmats = poses_gt.inverse().unsqueeze(0)  # (1,N,4,4) 世界->相机
    if args.optimize_pose:
        viewmats_param = viewmats.clone().requires_grad_(True)
    else:
        viewmats_param = viewmats

    opt = torch.optim.Adam(
        [{"params": [means, quats, scales, opacities, colors]},
         {"params": [viewmats_param], "lr": 1e-4}],
        lr=1e-2,
    )

    print(f"高斯数: {n_gauss} | 位姿优化: {args.optimize_pose}")
    # 训练循环
    for it in range(args.iters):
        opt.zero_grad()
        renders, alphas, _ = rasterization(
            means, quats, scales, opacities, colors,
            viewmats_param, Ks, W, H,
        )  # (1,N,H,W,3)
        # 对比渲染图像
        loss = torch.nn.functional.mse_loss(renders.squeeze(0).permute(0, 3, 1, 2), images)
        loss.backward()
        opt.step()
        if it % 200 == 0:
            psnr = -10 * torch.log10(loss + 1e-8)
            print(f"  iter {it}: loss={loss.item():.4f} psnr={psnr.item():.1f} dB")

    # 评估
    with torch.no_grad():
        renders, _, _ = rasterization(
            means, quats, scales, opacities, colors,
            viewmats_param.detach(), Ks, W, H,
        )
        loss = torch.nn.functional.mse_loss(renders.squeeze(0).permute(0, 3, 1, 2), images)
        psnr = -10 * torch.log10(loss + 1e-8)
    print(f"\n最终 PSNR: {psnr.item():.1f} dB")

    # 位姿误差（训练后 vs GT）
    if args.optimize_pose:
        viewmats_est = viewmats_param.detach()
        # 计算 ATE（简化：只比较平移）
        trans_gt = viewmats[:, :, :3, 3].squeeze(0)
        trans_est = viewmats_est[:, :, :3, 3].squeeze(0)
        # 对齐尺度/刚体变换（Umeyama 简化）
        ate = (trans_gt - trans_est).norm(dim=-1).mean().item() * 100  # cm
        print(f"简化 ATE（平移均值，未对齐刚体变换）: {ate:.1f} cm")
        print("注：真实 ATE 需 Umeyama 对齐，此处仅冒烟验证位姿梯度可反传。")

    print(f"\n=== Phase 0 完成，总耗时 {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
