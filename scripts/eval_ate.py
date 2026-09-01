#!/usr/bin/env python3
"""ATE 评估脚本：计算估计轨迹 vs GT 的 ATE RMSE（evo Umeyama 对齐）。

用法：
  python3 scripts/eval_ate.py --est results/step2_gauss_anti/estimated_trajectory.txt \
      --gt /data/Datasets/TUM/rgbd_dataset_freiburg3_walking_xyz/groundtruth.txt
"""
import argparse
import numpy as np
from pathlib import Path


def load_tum_trajectory(path):
    """加载 TUM 格式轨迹，返回 (timestamps, poses Nx4x4)。"""
    ts_list = []; poses = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            p = line.split()
            t = float(p[0])
            tx, ty, tz = float(p[1]), float(p[2]), float(p[3])
            qx, qy, qz, qw = float(p[4]), float(p[5]), float(p[6]), float(p[7])
            R = np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])
            T = np.eye(4); T[:3,:3]=R; T[:3,3]=[tx,ty,tz]
            ts_list.append(t); poses.append(T)
    return np.array(ts_list), np.array(poses)


def umeyama_align(src, dst):
    """Umeyama 刚体对齐（scale=1），返回对齐后的 src。"""
    src_c = src.mean(axis=0); dst_c = dst.mean(axis=0)
    H = (src-src_c).T @ (dst-dst_c)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    t = dst_c - R @ src_c
    src_aligned = (R @ src.T).T + t
    return src_aligned


def interpolate_poses(est_ts, est_poses, gt_ts):
    """用最近邻匹配估计位姿到 GT 时间戳。"""
    aligned = []
    for gt_t in gt_ts:
        idx = np.argmin(np.abs(est_ts - gt_t))
        if np.abs(est_ts[idx] - gt_t) < 0.02:  # 20ms tolerance
            aligned.append(est_poses[idx])
        else:
            aligned.append(None)
    return aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--max_diff", type=float, default=0.02, help="时间戳最大差异（秒）")
    args = ap.parse_args()

    est_ts, est_poses = load_tum_trajectory(args.est)
    gt_ts, gt_poses = load_tum_trajectory(args.gt)
    print(f"估计轨迹: {len(est_ts)} 帧 | GT轨迹: {len(gt_ts)} 帧")

    # 最近邻匹配
    est_matched = []; gt_matched = []
    for gt_t, gt_p in zip(gt_ts, gt_poses):
        idx = np.argmin(np.abs(est_ts - gt_t))
        if np.abs(est_ts[idx] - gt_t) < args.max_diff:
            est_matched.append(est_poses[idx][:3, 3])
            gt_matched.append(gt_p[:3, 3])
    est_matched = np.array(est_matched)
    gt_matched = np.array(gt_matched)
    print(f"匹配帧数: {len(est_matched)} / {len(gt_ts)}")

    if len(est_matched) < 3:
        print("匹配帧数不足，无法计算 ATE")
        return

    # Umeyama 对齐
    est_aligned = umeyama_align(est_matched, gt_matched)

    # ATE RMSE
    errors = np.linalg.norm(est_aligned - gt_matched, axis=1)
    ate_rmse = np.sqrt(np.mean(errors**2))
    ate_mean = np.mean(errors)
    ate_median = np.median(errors)
    ate_std = np.std(errors)
    ate_max = np.max(errors)

    print(f"\n=== ATE 结果 ===")
    print(f"  RMSE:   {ate_rmse*100:.2f} cm")
    print(f"  Mean:   {ate_mean*100:.2f} cm")
    print(f"  Median: {ate_median*100:.2f} cm")
    print(f"  Std:    {ate_std*100:.2f} cm")
    print(f"  Max:    {ate_max*100:.2f} cm")
    print(f"  匹配帧: {len(est_matched)}")


if __name__ == "__main__":
    main()
