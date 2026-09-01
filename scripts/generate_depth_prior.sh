#!/usr/bin/env bash
# 生成深度先验：在 wildgs-slam 环境用 Metric3D 对 TUM 序列推理
# 用法: bash scripts/generate_depth_prior.sh /data/Datasets/TUM/rgbd_dataset_freiburg1_desk 50
set -e
SEQ_DIR="${1:-/data/Datasets/TUM/rgbd_dataset_freiburg1_desk}"
N_FRAMES="${2:-50}"
OUT_DIR="${3:-/data/MonoDynaGSLAM/results/depth_prior}"
mkdir -p "$OUT_DIR"

PY="$(ls -d /data/conda_envs/wildgs-slam/bin/python 2>/dev/null | head -1)"
if [ -z "$PY" ]; then
  echo "ERROR: wildgs-slam python not found"
  exit 1
fi

# 复用 Metric3D 推理：用 WildGS 的 depth estimator 接口
"$PY" -c "
import sys, os, json, numpy as np, torch
from pathlib import Path
import cv2

sys.path.insert(0, '/data/WildGS-SLAM')
from src.utils.mono_priors.metric_depth_estimators import (
    get_metric_depth_estimator, predict_metric_depth
)

# 读取 rgb.txt
seq_dir = Path('$SEQ_DIR')
n_frames = int($N_FRAMES)
out_dir = Path('$OUT_DIR')
out_dir.mkdir(parents=True, exist_ok=True)

# 加载第一帧的 K（从 config 或 TUM 内参）
# TUM fr1 系列内参
K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]])

# 读取 rgb 帧（取前 n_frames）
lines = [l.split() for l in (seq_dir / 'rgb.txt').read_text().splitlines()
         if l and not l.startswith('#')][:n_frames]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {device} | 帧数: {len(lines)}')

# 加载 Metric3D 模型
cfg = {'mono_prior': {'depth': 'metric3d_vit_small'}, 'cam': {'fx': K[0,0]}, 'data': {'output': '.'}, 'scene': 'tum', 'device': device}
model = get_metric_depth_estimator(cfg)
model.eval()
print('模型加载完成')

# 逐帧推理
depth_maps = []
for i, (ts, rel) in enumerate(lines):
    img = cv2.imread(str(seq_dir / rel))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        depth = predict_metric_depth(model, i, img_t, cfg, device, save_depth=False)
    depth_maps.append(depth.cpu().numpy())
    if (i+1) % 10 == 0:
        print(f'  {i+1}/{len(lines)}')

# 保存
out = np.stack(depth_maps, axis=0)  # (N,H,W)
np.save(str(out_dir / 'depths.npy'), out)
# 保存元数据
meta = {'seq_dir': str(seq_dir), 'n_frames': n_frames, 'K': K.tolist(), 'timestamps': [l[0] for l in lines], 'image_paths': [l[1] for l in lines]}
with open(str(out_dir / 'meta.json'), 'w') as f:
    json.dump(meta, f)
print(f'深度先验保存: {out_dir}/depths.npy shape={out.shape}')
" 2>&1