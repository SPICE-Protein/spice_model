#!/usr/bin/env python3
import sys

if sys.version_info[0] < 3:
    sys.exit("需要 Python 3：先 conda activate spice，或用 ~/miniconda3/envs/spice/bin/python")

"""核时探针：一个蛋白 build + N 步 MD 在给定线程数下的墙钟。

提交 coverage 轴前先跑这个，回答两个问题：
  1. 单 RL run 吃几核？（定超算 --cpus-per-task）
  2. 单蛋白墙钟多久？（定 --time）

背景：引擎是 SoA/SIMD 单线程向量化，config 里 path_a_threads=2 是死配置（代码从未引用），
所以多数情况 1 核就够；若引擎内部有 rayon 则多核有真实加速。用墙钟对比说话。

用法：
  python scripts/experiments/slurm/fetch_atoms.py --pdb-id 2lyz --out data/parquet_hpc
  python scripts/experiments/slurm/probe_threads.py --pdb-id 2lyz \
      --parquet-dir data/parquet_hpc --threads 1 --steps 20
  python scripts/experiments/slurm/probe_threads.py --pdb-id 2lyz \
      --parquet-dir data/parquet_hpc --threads 8 --steps 20
加速比 = t(1)/t(8)：≈1 → 不吃核，cpus-per-task=1~2；明显 >1 → 按比例给核。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb-id", required=True)
    ap.add_argument("--parquet-dir", default="data/parquet_hpc")
    ap.add_argument("--threads", type=int, default=1, help="引擎线程数（设 RAYON_NUM_THREADS）")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--relax-iters", type=int, default=50)  
    args = ap.parse_args()

    os.environ["RAYON_NUM_THREADS"] = str(args.threads)

    from spice_rl.env import load_structure_with_atoms
    from spice_rl.env.quick_check import quick_check

    struct, base_atoms = load_structure_with_atoms(
        args.parquet_dir, args.pdb_id.upper(), max_residues=150)
    print(f"pdb={args.pdb_id.upper()} threads={args.threads} steps={args.steps} "
          f"n_res={struct.residue_count()} relax_iters={args.relax_iters}")

    t0 = time.time()
    out = quick_check(struct, ph=7.0, temp=298.0,
                      relax_iters=args.relax_iters, n_steps=args.steps,
                      equilibrate=True)
    dt = time.time() - t0
    ok = out.get("ok", False)
    print(f"RESULT ok={ok} wall={dt:.1f}s "
          f"(build+equilibrate+{args.steps} steps) => 每步 {dt/max(1, args.steps):.2f}s")
    if not ok:
        print("  reason:", out.get("reason"))

    est_min = 30 * dt * 3.0 / 60.0
    print(f"粗估单蛋白 RL 墙钟 ≈ {est_min:.0f} min "
          f"（30 集 × 探针 dt × 3× 安全系数；--time 建议 = K×此值×1.5）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
