#!/usr/bin/env python3
"""P0 统计汇总：mean/std/95%CI + paired t 检验（F-D, F-A, F-C）。

数据来源：results/seed_{mode}_{seed}/summary.json（seed 0/1/2/42）
"""
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEEDS = [0, 1, 2, 42]


def load(mode):
    vals = []
    for s in SEEDS:
        p = RESULTS / f"seed_{mode}_{s}" / "summary.json"
        if p.exists():
            vals.append(json.loads(p.read_text())["psnr_mean"])
    return np.array(vals)


def paired_t(a, b):
    d = a - b
    n = len(d)
    mean, std = d.mean(), d.std(ddof=1)
    t = mean / (std / np.sqrt(n))
    # 双侧 p 值（t 分布近似，n=4）
    from math import lgamma
    # 用 scipy 如果有
    try:
        from scipy import stats
        p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    except ImportError:
        p = float("nan") if np.isnan(t) else (0.001 if abs(t) > 5.84 else 0.01 if abs(t) > 4.54 else 0.05)
    return mean, std, t, p, n


def main():
    modes = ["none", "random_decay", "random_gauss", "pixel_mask", "selective"]
    data = {}
    print("=== 动态序列 walking_xyz：4 paired seeds ===")
    print(f"{'mode':15s} {'n':>2s} {'mean':>7s} {'std':>6s} {'95%CI':>14s}")
    for m in modes:
        v = load(m)
        if len(v) == 0:
            continue
        data[m] = v
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
        print(f"{m:15s} {len(v):2d} {v.mean():7.2f} {v.std(ddof=1):6.2f}   ±{ci:.2f}")

    print("\n=== paired 检验（selective vs 对照）===")
    for ref in ["random_gauss", "none", "random_decay"]:
        if "selective" in data and ref in data:
            mean, std, t, p, n = paired_t(data["selective"], data[ref])
            print(f"F - {ref:13s}: Δ={mean:+.2f}±{std:.2f} dB, t({n-1})={t:.2f}, p={'<0.001' if p<0.001 else f'{p:.4f}'}")

    print("\n=== 静态负对照 ===")
    for tag in ["static2_none_0", "static2_selective_0", "static2_gated_selective_0",
                "static2_none_0_fr2", "static2_selective_0_fr2", "static2_gated_selective_0_fr2"]:
        p = RESULTS / tag / "summary.json"
        if p.exists():
            v = json.loads(p.read_text())["psnr_mean"]
            print(f"{tag:35s}: {v:.2f} dB")


if __name__ == "__main__":
    main()
