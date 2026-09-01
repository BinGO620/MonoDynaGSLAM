#!/usr/bin/env python3
"""完整 3DGS 引擎跑 TUM/Bonn 序列：gsplat 官方 simple_trainer + TUM Parser 适配。

效果优先路线第一步：把基线从"阉割版训练器"对齐到官方完整引擎
（clone&split 致密化、SH、SSIM、优化调度、MCMC/DefaultStrategy）。

初始点云：Metric3D 深度反投影（替代 COLMAP SfM）——保持单目自洽。

用法：
  python3 scripts/train_full_3dgs.py --prior results/depth_prior_fr1_50 \
      --steps 7000 --out results/full3dgs_fr1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ---- 挂载官方 examples 到 path（在 import simple_trainer 之前） ----
GSPLAT_EXAMPLES = "/tmp/gsplat_repo/examples"
sys.path.insert(0, GSPLAT_EXAMPLES)

import datasets.colmap as colmap_module


class TUMParser:
    """兼容 gsplat examples Parser 接口：从 meta.json + depths.npy + GT pose 构建。"""

    def __init__(self, data_dir: str, factor: int = 1, test_every: int = 8,
                 normalize: bool = False, **kwargs):
        d = Path(data_dir)
        meta = json.loads((d / "meta.json").read_text())
        depths = np.load(d / "depths.npy")
        seq_dir = Path(meta["seq_dir"])
        K_orig = np.array(meta["K"])
        self.n_frames = len(meta["timestamps"])

        # GT 位姿（最近邻匹配）
        gt = {}
        for line in (seq_dir / "groundtruth.txt").read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            t = float(p[0])
            qx, qy, qz, qw = [float(x) for x in p[4:8]]
            R = np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [float(p[1]), float(p[2]), float(p[3])]
            gt[t] = T
        gt_ts = sorted(gt.keys())

        # 每帧构建
        self.image_names = []
        self.image_paths = []
        camtoworlds = []
        depths_out = []
        image_sizes = set()
        for i, (ts, rel) in enumerate(zip(meta["timestamps"], meta["image_paths"])):
            idx = np.searchsorted(gt_ts, float(ts))
            T = None
            for j in [idx-1, idx, idx+1]:
                if 0 <= j < len(gt_ts) and abs(gt_ts[j] - float(ts)) < 0.02:
                    T = gt[gt_ts[j]]
                    break
            if T is None:
                continue
            self.image_names.append(f"frame_{i:05d}")
            self.image_paths.append(str(seq_dir / rel))
            camtoworlds.append(T)
            depths_out.append(depths[i])
            image_sizes.add((640, 480))

        self.camtoworlds = np.array(camtoworlds, dtype=np.float32)  # (N,4,4)
        n = len(self.image_names)
        # 单相机
        cam_id = 0
        self.camera_ids = [cam_id] * n
        self.camera_id_to_idx = {cam_id: 0}
        self.Ks_dict = {cam_id: K_orig.astype(np.float32)}
        self.params_dict = {cam_id: []}  # 无畸变（fr1/f3）
        self.imsize_dict = {cam_id: (640, 480)}
        self.mask_dict = {cam_id: None}
        self.mapx_dict = {cam_id: None}
        self.mapy_dict = {cam_id: None}
        self.roi_undist_dict = {cam_id: None}
        self.test_every = test_every

        # 深度 → 初始点云（Metric3D 反投影 + 颜色采样）
        pts_all, rgb_all = [], []
        fx_o, fy_o, cx_o, cy_o = K_orig[0,0], K_orig[1,1], K_orig[0,2], K_orig[1,2]
        for i, dep in enumerate(depths_out):
            yi, xi = np.mgrid[0:dep.shape[0]:2, 0:dep.shape[1]:2]
            dd = dep[::2, ::2]
            v = (dd > 0.2) & (dd < 8.0)
            if v.sum() == 0:
                continue
            yc, xc, dc = yi[v], xi[v], dd[v]
            pts_cam = np.stack([(xc-cx_o)/fx_o*dc, (yc-cy_o)/fy_o*dc, dc], axis=-1)
            img = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
            cols = img[yc, xc].astype(np.float32)
            pts_w = (self.camtoworlds[i][:3,:3] @ pts_cam.T).T + self.camtoworlds[i][:3,3]
            pts_all.append(pts_w)
            rgb_all.append(cols)
        pts_all = np.concatenate(pts_all, 0)
        rgb_all = np.concatenate(rgb_all, 0)
        # 体素下采样（保持颜色）
        if len(pts_all) > 150000:
            keys = np.floor(pts_all / 0.02).astype(np.int64)
            _, uidx = np.unique(keys, axis=0, return_index=True)
            pts_all, rgb_all = pts_all[uidx], rgb_all[uidx]
        if len(pts_all) > 150000:
            sel = np.random.choice(len(pts_all), 150000, replace=False)
            pts_all, rgb_all = pts_all[sel], rgb_all[sel]
        self.points = pts_all.astype(np.float32)
        self.points_rgb = rgb_all.astype(np.uint8)
        self.points_err = np.zeros(len(pts_all), dtype=np.float32)

        # 场景 scale（colmap parser 语义）
        extents = self.points.max(0) - self.points.min(0)
        self.scene_scale = float(np.linalg.norm(extents))

        # 兼容属性
        self.indices = np.arange(n)
        self.point_indices = []
        self.exposure_values = None
        self.load_exposure = False
        self.load_depths = False
        self.normalize = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True, help="深度先验目录（meta.json + depths.npy）")
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--res", type=int, default=320, help="训练分辨率宽（0=原始640）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--disable_viewer", action="store_true", default=True)
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ---- 写适配数据目录：把 meta+depths 链接到一个 data_dir ----
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for f in ["meta.json", "depths.npy"]:
        src = Path(args.prior) / f
        if not (data_dir / f).exists():
            (data_dir / f).symlink_to(src.resolve())

    # ---- monkeypatch Parser ----
    colmap_module.Parser = TUMParser

    # ---- 运行官方 trainer ----
    from simple_trainer import Config, Runner
    from dataclasses import replace as dc_replace

    cfg = Config(
        data_dir=str(data_dir),
        data_factor=1,
        result_dir=str(out_dir),
        steps_scaler=args.steps / 30000.0,  # 官方默认 30k
        disable_viewer=True,
        init_type="sfm",
        ssim_lambda=0.2,
    )
    # 调整总步数
    cfg = dc_replace(cfg, steps_scaler=args.steps / 30000.0)

    device = torch.device("cuda")
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    runner.train()

    psnr = runner.psnrs[-1].item() if hasattr(runner, "psnrs") and len(runner.psnrs) else -1
    print(f"\n=== 完整引擎结果 ===\n最终 train PSNR: {psnr:.2f} dB | 耗时 {time.time()-t0:.0f}s")
    (out_dir / "summary.json").write_text(json.dumps(
        {"psnr": round(float(psnr), 2), "steps": args.steps,
         "seconds": round(time.time()-t0, 1)}, indent=2))


if __name__ == "__main__":
    main()
