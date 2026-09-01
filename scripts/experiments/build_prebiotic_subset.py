#!/usr/bin/env python3
"""从 full20 幸存者数据挑 top-N 亲本 → prebiotic 字母表子集跑（便宜的决定性对照）。

用法（HPC，model 根目录）:
  python scripts/experiments/build_prebiotic_subset.py \
      --high-candidates ../data/data_efficiency/soup_rl/high/runs/posttrain/pathb_candidates.csv \
      --low-candidates  ../data/data_efficiency/soup_rl/low/runs/posttrain/pathb_candidates.csv \
      --parents-dir     ../data/data_efficiency/soup_rl/parents \
      --root            ../data/data_efficiency/soup_rl_prebiotic \
      --per-bucket 4 --max-episodes 15

生成:
  <root>/subset/{low,high}/denovo{NNN}_s20.cif   (top-N 亲本 mmCIF，按 full20 幸存数排序)
  <root>/subset/{low,high}/posttrain_soup.yaml   (mutation_alphabet=GAVDESPTRK，输出隔离)

提交（每桶一个任务）:
  SUBSET_ROOT=$HOME/spice/<root>/subset BUCKET=low  MAX_EPISODES=15 \
      sbatch scripts/experiments/slurm/run_soup_prebiotic_subset.sbatch
  SUBSET_ROOT=$HOME/spice/<root>/subset BUCKET=high MAX_EPISODES=15 ...

判定: prebiotic 限制下（同批亲本、同 n=0）是否还出幸存者 → 坐实/证伪"前生物 10 AA 足够"。
"""
import argparse
import csv
import glob
import os
import shutil
from collections import Counter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--high-candidates", required=True, help="full20 high runs/posttrain/pathb_candidates.csv")
    ap.add_argument("--low-candidates", required=True, help="full20 low 同")
    ap.add_argument("--parents-dir", default="../data/data_efficiency/soup_rl/parents",
                    help="full20 亲本 shard 根（含 {high,low}/shard_XX/*.cif）")
    ap.add_argument("--root", default="../data/data_efficiency/soup_rl_prebiotic")
    ap.add_argument("--per-bucket", type=int, default=4, help="每桶挑几个 top 亲本")
    ap.add_argument("--alphabet", default="GAVDESPTRK")
    ap.add_argument("--max-episodes", type=int, default=15, help="仅用于成本估算/提示，实际由 sbatch MAX_EPISODES 传入")
    ap.add_argument("--base-config", default="configs/HPC/posttrain_soup.yaml")
    args = ap.parse_args()

    import yaml  # noqa: PLC0415
    if not os.path.exists(args.base_config):
        print(f"[error] 缺 base config: {args.base_config}")
        return 1

    for bucket, cand_path in (("high", args.high_candidates), ("low", args.low_candidates)):
        surv = [r for r in csv.DictReader(open(cand_path)) if r["survived"] == "1"]
        cnt = Counter(r["tag"] for r in surv)
        top = [t for t, _ in cnt.most_common(args.per_bucket)]
        print(f"[{bucket}] 幸存候选 {len(surv)}（{len(cnt)} 父本）→ 取 top {len(top)}: {top}")
        bdir = os.path.join(args.root, "subset", bucket)
        os.makedirs(bdir, exist_ok=True)
        n_copy = 0
        for tag in top:
            # tag='soup_denovo135_s20'，mmCIF 文件名为 denovo135_s20.cif（去掉 soup_ 前缀）
            fname = (tag[5:] if tag.startswith("soup_") else tag) + ".cif"
            src = glob.glob(os.path.join(args.parents_dir, bucket, "**", fname), recursive=True)
            if not src:
                print(f"  [warn] 找不到 {fname} in {args.parents_dir}/{bucket}")
                continue
            shutil.copy(src[0], os.path.join(bdir, fname))
            n_copy += 1
        print(f"  复制 {n_copy} 亲本 -> {bdir}/")
        cfg = yaml.safe_load(open(args.base_config))
        pre = f"{args.root}/subset/{bucket}"
        cfg["post"]["ckpt_dir"] = f"{pre}/checkpoints/posttrain"
        cfg["post"]["log_dir"] = f"{pre}/runs/posttrain"
        cfg["post"]["pseudo_label_dir"] = f"{pre}/pseudo_labels"
        cfg["post"]["pseudo_tfrecord_path"] = f"{pre}/pseudo.tfrecord"
        cfg["post"]["phase_map_dir"] = f"{pre}/phase_maps"
        cfg["es"]["mutation_alphabet"] = args.alphabet
        out_cfg = os.path.join(bdir, "posttrain_soup.yaml")
        yaml.safe_dump(cfg, open(out_cfg, "w"), sort_keys=False)
        print(f"  [config] {out_cfg}  (mutation_alphabet={args.alphabet})")

    n_ep = args.per_bucket * 2 * args.max_episodes
    est_core_hrs = n_ep * 0.5  # full20 实测 ~0.5 核时/集
    print(f"\n提交（2 个任务，各 ~4 亲本 × {args.max_episodes} 集）:")
    for b in ("low", "high"):
        print(f"  SUBSET_ROOT=$HOME/spice/model/{args.root}/subset BUCKET={b} MAX_EPISODES={args.max_episodes} "
              f"sbatch scripts/experiments/slurm/run_soup_prebiotic_subset.sbatch")
    print(f"\n预计: ~{n_ep} 集 ≈ {est_core_hrs:.0f} 核时 ≈ ¥{est_core_hrs*0.1:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
