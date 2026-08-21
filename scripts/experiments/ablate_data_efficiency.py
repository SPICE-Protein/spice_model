#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def _dotted_set(d, path, val):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = val


def _run(cmd):
    print(">>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def _scale_dirs(out, n):
    d = os.path.join(out, f"n{n}")
    os.makedirs(d, exist_ok=True)
    return {
        "yaml": os.path.join(d, "pretrain.yaml"),
        "tfrecord": os.path.join(d, "tfrecords"),
        "ckpt": os.path.join(d, "checkpoints"),
        "log": os.path.join(d, "runs"),
        "ckpt_file": os.path.join(d, "checkpoints", "best_weights.weights.h5"),
    }


def _make_pretrain_yaml(base_yaml, n, dirs, target_steps=0):
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    _dotted_set(cfg, "data.max_chains", n)
    _dotted_set(cfg, "data.tfrecord_dir", dirs["tfrecord"])
    _dotted_set(cfg, "train.ckpt_dir", dirs["ckpt"])
    _dotted_set(cfg, "train.log_dir", dirs["log"])
    if target_steps > 0:
        _dotted_set(cfg, "train.max_steps", target_steps)
        _dotted_set(cfg, "train.warmup_steps", max(20, target_steps // 10))
    with open(dirs["yaml"], "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return dirs["yaml"]


def _parse_casp_out(text):
    m = re.search(r"AUC\s+([\d.]+)", text)
    p = re.search(r"P@L/5\s+([\d.]+)%", text)
    return (
        float(m.group(1)) if m else float("nan"),
        float(p.group(1)) if p else float("nan"),
    )


def _write_csv(out, rows):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "results.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["scale"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"-> {path}")


def build(base_yaml, scales, out, target_steps=0):
    for n in scales:
        dirs = _scale_dirs(out, n)
        y = _make_pretrain_yaml(base_yaml, n, dirs, target_steps)
        _run([sys.executable, "-m", "spice_pre.data.dataset", "--config", y, "build"])
        _run([sys.executable, "-m", "spice_pre.train_pretrain", "--config", y])
    print(f"[done] build: 尺度 {scales} 的 TFRecord + prior 完成")


def evaluate(scales, out):
    rows = []
    for n in scales:
        dirs = _scale_dirs(out, n)
        if not os.path.exists(dirs["yaml"]) or not os.path.exists(dirs["ckpt_file"]):
            print(f"[skip] n={n} 缺 config/ckpt（先跑 build）")
            continue
        log = os.path.join(out, f"n{n}", "casp.log")
        with open(log, "w") as f:
            subprocess.check_call(
                [sys.executable, "-m", "spice_pre.eval_casp",
                 "--config", dirs["yaml"], "--weights", dirs["ckpt_file"]],
                cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        auc, p5 = _parse_casp_out(open(log).read())
        rows.append({"scale": n, "n_records": "?", "casp_auc": auc,
                     "casp_p5": p5, "ckpt": dirs["ckpt_file"]})
        print(f"[n={n}] CASP AUC={auc:.3f} P@L/5={p5:.1f}%", flush=True)
    _write_csv(out, rows)
    print("[done] eval: 汇总写入 results.csv（先跑 build 拿到 n_records）")


def rl_gen(base_post_yaml, scales, out):
    for n in scales:
        dirs = _scale_dirs(out, n)
        with open(base_post_yaml) as f:
            cfg = yaml.safe_load(f)
        _dotted_set(cfg, "post.pretrain_ckpt", dirs["ckpt_file"])
        _dotted_set(cfg, "post.log_dir", os.path.join(dirs["log"], "posttrain"))
        _dotted_set(cfg, "post.ckpt_dir", os.path.join(dirs["ckpt"], "posttrain"))
        _dotted_set(cfg, "post.pseudo_label_dir", os.path.join(out, f"n{n}", "pseudo_labels"))
        _dotted_set(cfg, "post.pseudo_tfrecord_path", os.path.join(out, f"n{n}", "pseudo.tfrecord"))
        _dotted_set(cfg, "post.phase_map_dir", os.path.join(out, f"n{n}", "phase_maps"))
        post_y = os.path.join(out, f"n{n}", "posttrain.yaml")
        with open(post_y, "w") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
        print(f"[n={n}] 固定 RL 环配置: {post_y}")
        print(f"   Kaggle/HPC: python -m spice_rl.train_post --config {post_y} --pdb-id <id>")
        print("   （同一组蛋白 × 每个尺度 = 干净的数据效率消融）")
    print("[done] rl-gen: 每个尺度一份 posttrain.yaml，RL 配置完全固定，只有 prior 尺度变")


def gen_only(base_yaml, scales, out, target_steps=0):
    for n in scales:
        dirs = _scale_dirs(out, n)
        y = _make_pretrain_yaml(base_yaml, n, dirs, target_steps)
        print(f"[gen] {y}  (max_chains={n}, max_steps={target_steps or 'inherit'})")
    print("[done] gen-only: 配置已生成，可分发到 Colab/Kaggle 并行训 prior")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["gen-only", "build", "eval", "rl-gen"], required=True)
    ap.add_argument("--scales", default="10,100,1000,45000")
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--post-config", default="configs/posttrain.yaml")
    ap.add_argument("--out", default="runs/ablation/data_efficiency")
    ap.add_argument("--target-steps", type=int, default=3000,
                    help="每尺度固定优化器步数（数据效率公平性；0=继承 base config）")
    args = ap.parse_args()
    scales = [int(x) for x in args.scales.split(",") if x]
    if args.phase == "gen-only":
        gen_only(args.config, scales, args.out, args.target_steps)
    elif args.phase == "build":
        build(args.config, scales, args.out, args.target_steps)
    elif args.phase == "eval":
        evaluate(scales, args.out)
    else:
        rl_gen(args.post_config, scales, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
