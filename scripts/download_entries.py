#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np
import polars as pl


def is_cloud() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return os.path.isdir("/kaggle")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/parquet/entries_all.parquet")
    ap.add_argument("--max-seq-len", type=int, default=None,
                    help="额外打印 ≤ 该长度 aa 的蛋白构成")
    ap.add_argument("--endpoint", default=None, help="hf 端点，默认自动：云端官方 / 本地镜像")
    ap.add_argument("--repo", default="SPICE-Protein/spice_protein")
    args = ap.parse_args()

    ep = args.endpoint or ("https://huggingface.co" if is_cloud() else "https://hf-mirror.com")
    os.environ["HF_ENDPOINT"] = ep
    import huggingface_hub.constants as hf_constants
    hf_constants.ENDPOINT = ep  
    from huggingface_hub import hf_hub_download, list_repo_files

    print(f"HF endpoint: {ep}")
    entries = sorted(f for f in list_repo_files(args.repo, repo_type="dataset")
                     if f.startswith("entries_shard_") and f.endswith(".parquet"))
    print(f"共 {len(entries)} 个 entries shard，逐个下载并合并…")
    frames = []
    for f in entries:
        p = hf_hub_download(args.repo, f, repo_type="dataset")
        frames.append(pl.read_parquet(p).with_columns(pl.lit(f).alias("_src")))
    en_all = pl.concat(frames)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    en_all.write_parquet(args.out)
    print(f"已写入: {args.out}（{en_all.height} 条，唯一 pdb {en_all['pdb_id'].n_unique()}）")

    print(f"总条目: {en_all.height}")
    if "has_env" in en_all.columns:
        n_env = int(en_all["has_env"].sum())
        print(f"有真实 env(pH+temp+ionic) 的条目: {n_env} ({n_env / en_all.height * 100:.1f}%)")
    print("全长分箱(n_residues):",
          dict(zip(*[["<50", "50-99", "100-149", "150-199", "200-255", ">=256"],
                     np.histogram(en_all["n_residues"], bins=[0, 50, 100, 150, 200, 256, 10**9])[0]])))

    if args.max_seq_len is not None:
        sub = en_all.filter(pl.col("n_residues") <= args.max_seq_len)
        print(f"\n≤{args.max_seq_len} aa 的蛋白: {sub.height} 个"
              f"（占全量 {sub.height / en_all.height * 100:.1f}%）")
        print("其长度分箱:",
              dict(zip(*[["<30", "30-59", "60-89", "90-119", "120-149", "150"],
                         np.histogram(sub["n_residues"],
                                      bins=[0, 30, 60, 90, 120, 150, args.max_seq_len + 1])[0]])))
        if "has_env" in sub.columns:
            n_env = int(sub["has_env"].sum())
            print(f"其中含真实 env 的: {n_env} ({n_env / sub.height * 100:.1f}%)")
        if "method" in sub.columns:
            print("按实验方法:")
            for row in (sub.group_by("method").len()
                        .sort("len", descending=True).head(6).iter_rows()):
                print(f"  {str(row[0])[:40]:<42} {row[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
