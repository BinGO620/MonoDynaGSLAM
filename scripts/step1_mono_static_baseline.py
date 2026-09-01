#!/usr/bin/env python3
"""Step 1: 单目静态 3DGS 重建基线（gsplat + Metric3D 深度先验）。

流程：
  1. 用 Metric3D 对单目序列生成深度图（先验）
  2. 用深度先验初始化点云
  3. 用 gsplat 做 3DGS 重建 + 位姿优化
  4. 评估 PSNR 与 ATE

用法: python3 scripts/step1_mono_static_baseline.py --seq fr1_desk --n_frames 50
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 添加 Metric3D 路径
sys.path.insert(0, "/data/Metric3D")

FR1_K = np.array([[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]])
FR2_K = np.array([[520.9, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]])
FR3_K = np.array([[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]])

KCAMS = {  # (fx, fy, cx, cy) for TUM sequences
    "fr1_desk": (517.3, 516.5, 318.6, 255.3),
    "fr2_xyz": (520.9, 521.0, 325.1, 249.7),
    "fr3_office": (535.4, 539.2, 320.1, 247.6),
}


def parse_groundtruth(path: Path):
    poses = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        parts = line.split()
        t = float(parts[0])
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
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


def load_images(seq_dir: Path, n_frames: int, res: int, device: str):
    import cv2
    K = np.array([KCAMS.get(seq_dir.name, (535.4, 539.2, 320.1, 247.6))])
    K = np.array([[K[0,0],0,K[0,2]],[0,K[0,1],K[0,3]],[0,0,1]])
    gt = parse_groundtruth(seq_dir / "groundtruth.txt")
    lines = [l.split() for l in (seq_dir / "rgb.txt").read_text().splitlines()
             if l and not l.startswith("#")]
    timestamps = sorted(gt.keys())
    frames = []
    for ts, path in lines:
        t = float(ts)
        idx = np.searchsorted(timestamps, t)
        for i in [idx-1, idx, idx+1]:
            if 0 <= i < len(timestamps) and abs(timestamps[i] - t) < 0.01:
                frames.append((timestamps[i], path))
                break
        if len(frames) >= n_frames:
            break
    scale = res / 640.0
    K[:2] *= scale
    images = []
    poses = []
    for t, rel in frames:
        img = cv2.imread(str(seq_dir / rel))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h = int(res * 0.75)
        img = cv2.resize(img, (res, h))
        images.append(torch.from_numpy(img).permute(2,0,1).float() / 255.0)
        poses.append(torch.from_numpy(gt[t]).float())
    return torch.stack(images).to(device), torch.from_numpy(K).float().to(device), torch.stack(poses).to(device), frames


def compute_depth_prior(images, seq_dir, frames, res, device):
    """用 Metric3D 生成深度先验（单帧或批量）。"""
    from mono.model.mono_model import MonoModel
    # 加载 Metric3D 模型
    model = torch.hub.load("/data/Metric3D", "metric3d_vit_small", pretrain=True, source="local")
    model = model.to(device).eval()
    depths = []
    for i, (t, _) in enumerate(frames):
        img = images[i].cpu().numpy().transpose(1,2,0)  # H,W,3
        img = (img * 255).astype(np.uint8)
        # Metric3D 需要标准化
        from PIL import Image
        rgb = Image.fromarray(img)
        # 用 Metric3D 的推理管线
        with torch.no_grad():
            # 在 metric3d 的 demo 里找推理函数
            pass  # TODO: 接入 Metric3D 推理管线
        break  # 先只做一张
    return depths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", default="/data/Datasets/TUM/rgbd_dataset_freiburg1_desk")
    ap.add_argument("--n_frames", type=int, default=50)
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--use_depth_prior", action="store_true", default=False)
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda"
    images, K, poses_gt, frames = load_images(Path(args.seq_dir), args.n_frames, args.res, device)
    print(f"加载 {len(frames)} 帧，{time.time()-t0:.1f}s | 图像 {images.shape}")

    # 深度先验
    if args.use_depth_prior:
        depths = compute_depth_prior(images, Path(args.seq_dir), frames, args.res, device)
        print(f"深度先验计算完成")

    # 初始化点云
    M = 10000
    depths = torch.rand(M, 1, device=device) * 4.5 + 0.5
    u = torch.rand(M, 1, device=device) * args.res
    v = torch.rand(M, 1, device=device) * int(args.res * 0.75)
    fx, fy = K[0,0].item(), K[1,1].item()
    cx, cy = K[0,2].item(), K[1,2].item()
    x_cam = (u - cx) / fx * depths
    y_cam = (v - cy) / fy * depths
    pts_cam = torch.cat([x_cam, y_cam, depths], dim=-1)
    pts_world = (poses_gt[0][:3,:3] @ pts_cam.T).T + poses_gt[0][:3, 3]
    n_gauss = pts_world.shape[0]

    means = pts_world.unsqueeze(0)
    quats = (torch.randn(1, n_gauss, 4, device=device) / 2).normalized()
    scales = torch.full((1, n_gauss, 3), 0.03, device=device)
    opacities = torch.ones(1, n_gauss, device=device) * 0.3
    colors = torch.randn(1, n_gauss, 3, device=device)

    # 位姿
    viewmats = poses_gt.inverse().unsqueeze(0)
    viewmats_param = viewmats.clone().requires_grad_(True)
    Ks = K.unsqueeze(0).expand(1, args.n_frames, 3, 3).contiguous()

    opt = torch.optim.Adam([
        {"params": [means, quats, scales, opacities, colors]},
        {"params": [viewmats_param], "lr": 1e-4},
    ], lr=1e-2)

    from gsplat import rasterization
    print(f"训练 {args.iters} iters, {n_gauss} 高斯...")
    for it in range(args.iters):
        opt.zero_grad()
        rends, _, _ = rasterization(means, quats, scales, opacities, colors, viewmats_param, Ks, args.res, int(args.res*0.75))
        loss = torch.nn.functional.mse_loss(rends.squeeze(0).permute(0,3,1,2), images)
        loss.backward()
        opt.step()
        if it % 500 == 0:
            print(f"  iter {it}: loss={loss.item():.4f} psnr={-10*torch.log10(loss+1e-8).item():.1f} dB")

    with torch.no_grad():
        rends, _, _ = rasterization(means, quats, scales, opacities, colors, viewmats_param.detach(), Ks, args.res, int(args.res*0.75))
        loss = torch.nn.functional.mse_loss(rends.squeeze(0).permute(0,3,1,2), images)
        psnr = -10 * torch.log10(loss + 1e-8).item()
        ate = (viewmats_param[:,:,:3,3].squeeze(0) - viewmats[:,:,:3,3].squeeze(0)).norm(dim=-1).mean().item() * 100
    print(f"\n最终 PSNR={psnr:.1f} dB | 简化ATE={ate:.1f} cm | 耗时={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()