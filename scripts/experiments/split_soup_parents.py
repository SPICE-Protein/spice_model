#!/usr/bin/env python3
"""把 soup_screen 幸存者按 m5 中位切 low/high 两桶，打成 shard 目录，并生成 per-bucket n=0 RL config（输出目录隔离）。

用法（HPC，model 根目录，只需 pyyaml+csv，无需 TF/引擎）:
  python scripts/experiments/split_soup_parents.py \
      --csv  ../data/data_efficiency/soup_screen/soup_denovo_n200_len80_prebiotic.csv \
      --mmifs ../data/data_efficiency/soup_screen/mmifs \
      --out  ../data/data_efficiency/soup_rl/parents \
      --base-config configs/HPC/posttrain_soup.yaml \
      --shard-size 10

生成:
  <out>/low/shard_00..NN/   （m5 低半，100 条）
  <out>/high/shard_00..NN/  （m5 高半，100 条）
  <out>/{low,high}/posttrain_soup.yaml
    = base n=0 配置 + 输出目录(ckpt/log/pseudo_label/pseudo_tfrecord/phase_map)隔离到各自桶
    → low/high 并发跑不互覆，且能按桶归因存活/伪标签。

之后（每桶一个数组作业，10 亲本/任务）:
  BUCKET=low  sbatch --array=0-9 run_soup_rl_full.sbatch
  BUCKET=high sbatch --array=0-9 run_soup_rl_full.sbatch

验证 n=0（随机初始化，不碰 45k 先验）:
  grep -c "Successfully loaded Pre-train weights" slurm_logs/soup_rl_full_*.out   # 必须全 0
"""
import argparse
import csv
import math
import os
import shutil

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="soup_screen 输出 CSV")
    ap.add_argument("--mmifs", required=True, help="mmCIF RL 亲本目录")
    ap.add_argument("--root", default="../data/data_efficiency/soup_rl",
                    help="配置输出根（config 目录据此隔离到 {root}/{bucket}/...；跑前生物对照时换独立根，如 soup_rl_prebiotic）")
    ap.add_argument("--out", default=None,
                    help="亲本 shard 输出目录（默认 {root}/parents）")
    ap.add_argument("--mutation-alphabet", default="full20",
                    help="ES 突变字母表：full20（默认全 20 aa）或显式字符串如 GAVDESPTRK（前生物限制）")
    ap.add_argument("--base-config", default="configs/HPC/posttrain_soup.yaml")
    ap.add_argument("--shard-size", type=int, default=10, help="每个 shard 亲本数（=数组作业单任务量）")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(args.root, "parents")

    if not os.path.exists(args.base_config):
        print(f"[error] 找不到 base config: {args.base_config}（需先上传 configs/HPC/posttrain_soup.yaml）")
        return 1

    rows = [r for r in csv.DictReader(open(args.csv))
            if r.get("ok") == "True" and r.get("survive") == "20" and r.get("m5")]
    rows.sort(key=lambda r: float(r["m5"]))
    n = len(rows)
    if n == 0:
        print("[error] CSV 里没有全存活(20步)且有 m5 的行")
        return 1
    half = n // 2
    buckets = {"low": rows[:half], "high": rows[half:]}
    print(f"总 {n} 条 | 中位切分: low {len(buckets['low'])} / high {len(buckets['high'])}")

    import yaml  # noqa: PLC0415
    for bucket, blist in buckets.items():
        if not blist:
            print(f"[warn] {bucket} 空桶，跳过")
            continue
        bdir = os.path.join(args.out, bucket)
        # 复制亲本到 shard
        copied, missing = 0, 0
        for i, r in enumerate(blist):
            src = os.path.join(args.mmifs, f"{r['label']}_s20.cif")
            if not os.path.exists(src):
                print(f"[warn] 缺 mmCIF: {r['label']}")
                missing += 1
                continue
            shard = f"shard_{i // args.shard_size:02d}"
            os.makedirs(os.path.join(bdir, shard), exist_ok=True)
            shutil.copy(src, os.path.join(bdir, shard, os.path.basename(src)))
            copied += 1
        nshard = math.ceil(len(blist) / args.shard_size)
        print(f"[{bucket}] {copied} 亲本 -> {bdir}/shard_XX/ ({nshard} shards) | "
              f"m5 {float(blist[0]['m5']):.4f}~{float(blist[-1]['m5']):.4f} | 缺 {missing}")
        # per-bucket n=0 config（改输出目录 + 突变字母表）
        cfg = yaml.safe_load(open(args.base_config))
        pre = f"{args.root}/{bucket}"
        cfg["post"]["ckpt_dir"] = f"{pre}/checkpoints/posttrain"
        cfg["post"]["log_dir"] = f"{pre}/runs/posttrain"
        cfg["post"]["pseudo_label_dir"] = f"{pre}/pseudo_labels"
        cfg["post"]["pseudo_tfrecord_path"] = f"{pre}/pseudo.tfrecord"
        cfg["post"]["phase_map_dir"] = f"{pre}/phase_maps"
        cfg["es"]["mutation_alphabet"] = args.mutation_alphabet
        out_cfg = os.path.join(bdir, "posttrain_soup.yaml")
        with open(out_cfg, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"[config] {out_cfg}")

    nshard = math.ceil(half / args.shard_size)
    print(f"\n提交（先 mkdir -p slurm_logs）:")
    print(f"  BUCKET=low  sbatch --array=0-{nshard-1} run_soup_rl_full.sbatch")
    print(f"  BUCKET=high sbatch --array=0-{nshard-1} run_soup_rl_full.sbatch")
    print(f"  非默认根时补 SOUP_ROOT=$HOME/spice/model/{args.out}（sbatch 的 SOUP_ROOT 默认指向 soup_rl/parents）")
    print(f"  突变字母表: {args.mutation_alphabet}")
    print("运行中验证 n=0: grep -c 'Successfully loaded Pre-train weights' slurm_logs/soup_rl_full_*.out")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
