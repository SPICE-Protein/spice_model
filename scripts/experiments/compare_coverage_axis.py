#!/usr/bin/env python
"""Coverage-axis data-efficiency ablation report: n10 (weakest seed) vs n45000 (full prior).

Reads the fixed-RL-loop posttrain outputs for the two end-member prior scales and
produces (1) a console summary (NaN/watchdog check, SAC alpha, survivor counts per
protein, Q statistics, beyond-PDB coverage), (2) a Mann-Whitney U comparison of
survivor Q distributions, (3) a two-panel figure, and (4) a CSV report.

The ablation is a deliberate two-point end-member comparison: both scales use the
SAME current-pretrain pipeline and the SAME fixed RL loop; intermediate prior scales
are redundant (bounded by the extremes) and are not part of this ablation.

Usage:
  python scripts/experiments/compare_coverage_axis.py \
      --base /path/to/seventh_mut --scales n10,n45000 \
      --labels 'n10=N=10;n45000=N=45,000'
Re-runnable per seed / per round: just point --base at the new output root.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import Counter, defaultdict

import numpy as np

try:
    from scipy import stats as _stats
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # noqa: BLE001
    HAVE_MPL = False

PROTEIN_ORDER = ["7qf3", "6qqe", "8d8f", "1jvt"]
PROTEIN_LABEL = {
    "7qf3": "7QF3", "6qqe": "6QQE (lysozyme)", "8d8f": "8D8F (lysozyme)",
    "1jvt": "1JVT (RNase A)",
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="/Users/redelectricity/Documents/Projects/SPICE/data/seventh_mut")
    ap.add_argument("--scales", default="n10,n45000", help="comma-separated scale dir names")
    ap.add_argument("--labels", default="n10=N=10;n45000=N=45,000", help="; map dir->axis label")
    ap.add_argument("--subdir", default="runs/posttrain", help="posttrain outputs subdir")
    ap.add_argument("--out-dir", default=None, help="figure/report output dir (default: runs/figures)")
    return ap.parse_args()


def load(base, scale, subdir, name):
    p = os.path.join(base, scale, subdir, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def protein_blocks(metrics):
    """Split metrics rows into per-protein blocks (ep resets to 0 between proteins)."""
    blocks, cur = [], []
    for r in metrics:
        if int(r["ep"]) == 0 and cur:
            blocks.append(cur)
            cur = []
        cur.append(r)
    if cur:
        blocks.append(cur)
    return blocks


def summarize(base, scale, subdir, label):
    m = load(base, scale, subdir, "metrics.csv")
    c = load(base, scale, subdir, "pathb_candidates.csv")
    cf = load(base, scale, subdir, "pathb_candidates_failures.csv")
    cov = load(base, scale, subdir, "coverage.csv")
    bl = load(base, scale, subdir, "explosive_blacklist.csv")

    out = {"label": label}
    if m is None:
        return out
    # NaN / watchdog check
    nan_eps = [
        r["ep"] for r in m
        if any("nan" in r[k].lower() for k in ("critic_loss", "actor_loss", "alpha_loss", "conf_loss")
               if r[k] not in ("", "nan"))
    ]
    out["n_eps"] = len(m)
    out["nan_eps"] = nan_eps
    # alpha per protein block
    alpha = []
    for bi, b in enumerate(protein_blocks(m)):
        al = [float(r["alpha"]) for r in b if r["alpha"] not in ("", "nan")]
        if al:
            alpha.append((bi, al[0], al[-1]))
    out["alpha_blocks"] = alpha
    # survivors per protein
    surv = defaultdict(list)
    if c:
        for r in c:
            if r["survived"] == "1":
                surv[r["tag"]].append(float(r["q"]))
    out["survivors"] = {k: (len(v), v) for k, v in surv.items()}
    # failures
    out["failures"] = Counter(r["reason"][:40] for r in cf) if cf else Counter()
    # coverage
    if cov:
        kinds = defaultdict(list)
        for r in cov:
            kinds[r["kind"]].append((float(r["ph"]), float(r["temp"])))
        out["coverage"] = {k: v for k, v in kinds.items()}
    out["blacklist"] = len(bl) if bl else 0
    return out


def print_summary(s):
    print(f"--- {s['label']} ---")
    if "n_eps" not in s:
        print("  (no metrics)")
        return
    print(f"  episodes={s['n_eps']}  NaN eps: {s['nan_eps'] if s['nan_eps'] else 'none'}")
    print(f"  alpha per protein (start->end): "
          + ", ".join(f"p{bi} {a:.3f}->{z:.3f}" for bi, a, z in s["alpha_blocks"]))
    for tag in PROTEIN_ORDER:
        if tag in s["survivors"]:
            n, qs = s["survivors"][tag]
            print(f"  [{tag}] survivors={n:>4}  Q med={statistics.median(qs):.3f} "
                  f"(min {min(qs):.3f}, max {max(qs):.3f})")
    cov = s.get("coverage", {})
    if cov:
        env = cov.get("env_fail", [])
        if env:
            phs = [p[0] for p in env]; ts = [p[1] for p in env]
            print(f"  coverage env_fail: {len(env)} pts, pH {min(phs):.1f}-{max(phs):.1f}, "
                  f"T {min(ts):.0f}-{max(ts):.0f}")
        print(f"  coverage anchor/pathA: "
              + ", ".join(f"{k}={len(v)}" for k, v in cov.items() if k != "env_fail"))
    if s["failures"]:
        print(f"  failures: {dict(s['failures'].most_common(3))}")
    print(f"  blacklist entries: {s['blacklist']}")


def stats_table(sa, sb):
    """Mann-Whitney U on survivor Q per protein, n10 vs n45000."""
    rows = []
    tags = [t for t in PROTEIN_ORDER if t in sa["survivors"] or t in sb["survivors"]]
    for t in tags:
        a = sa["survivors"].get(t, (0, []))[1]
        b = sb["survivors"].get(t, (0, []))[1]
        row = {"tag": t, "n10_n": len(a), "n45_n": len(b),
               "n10_med": statistics.median(a) if a else float("nan"),
               "n45_med": statistics.median(b) if b else float("nan")}
        if HAVE_SCIPY and len(a) > 3 and len(b) > 3:
            row["p"] = float(_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        else:
            row["p"] = float("nan")
        rows.append(row)
    # combined
    a = [q for t in tags for q in sa["survivors"].get(t, (0, []))[1]]
    b = [q for t in tags for q in sb["survivors"].get(t, (0, []))[1]]
    row = {"tag": "ALL", "n10_n": len(a), "n45_n": len(b),
           "n10_med": statistics.median(a) if a else float("nan"),
           "n45_med": statistics.median(b) if b else float("nan")}
    if HAVE_SCIPY and len(a) > 3 and len(b) > 3:
        row["p"] = float(_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    else:
        row["p"] = float("nan")
    rows.append(row)
    return rows


def parse_muts(s):
    """Parse '43:L>I;75:K>C' -> [(43,'L','I'), (75,'K','C')]."""
    out = []
    for tok in str(s).split(";"):
        tok = tok.strip()
        if not tok or ">" not in tok or ":" not in tok:
            continue
        pos, rest = tok.split(":", 1)
        w, m = rest.split(">", 1)
        out.append((int(pos), w, m))
    return out


def product_summary(base, scale, subdir, label):
    """Aggregate the designed products (surviving mutants) per protein."""
    c = load(base, scale, subdir, "pathb_candidates.csv")
    subs = defaultdict(Counter)   # tag -> Counter('pos:W>M' -> count)
    n_mut = Counter()             # #mutations per survivor
    seen = defaultdict(set)       # tag -> set(mut_seq)
    n_surv = Counter()            # tag -> survivor count
    if c:
        for r in c:
            if r["survived"] != "1":
                continue
            tag = r["tag"]
            n_surv[tag] += 1
            seen[tag].add(r["mut_seq"])
            muts = parse_muts(r["mutations"])
            n_mut[len(muts)] += 1
            for pos, w, m in muts:
                subs[tag][f"{pos}:{w}>{m}"] += 1
    return {"label": label, "subs": subs, "n_mut": n_mut,
            "uniq": {t: len(s) for t, s in seen.items()}, "n_surv": n_surv}


def print_product(pa, pb, top_n=8):
    print("\n=== Product analysis (survivor mutation chemistry): n10 vs n45000 ===")
    tags = [t for t in PROTEIN_ORDER if t in pa["subs"] or t in pb["subs"]]
    for t in tags:
        a, b = pa["subs"].get(t, Counter()), pb["subs"].get(t, Counter())
        ta = [k for k, _ in a.most_common(top_n)]
        tb = [k for k, _ in b.most_common(top_n)]
        shared = set(ta) & set(tb)
        print(f"[{t}] n10: {pa['n_surv'].get(t,0)} surv / {pa['uniq'].get(t,0)} uniq | "
              f"n45: {pb['n_surv'].get(t,0)} surv / {pb['uniq'].get(t,0)} uniq")
        print(f"  top{top_n} n10: {ta}")
        print(f"  top{top_n} n45: {tb}")
        print(f"  shared top-{top_n}: {sorted(shared) if shared else 'none'}")
    print(f"  mutations/survivor: n10={dict(pa['n_mut'])}  n45={dict(pb['n_mut'])}")


def make_product_figure(pa, pb, out_png, top_n=8):
    """Per-protein survivor substitution chemistry, both scales, shared marked *."""
    if not HAVE_MPL:
        return
    tags = [t for t in PROTEIN_ORDER if t in pa["subs"] or t in pb["subs"]]
    ncol = 2
    nrow = max(1, (len(tags) + 1) // 2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.6 * nrow), squeeze=False)
    for j, t in enumerate(tags):
        ax = axes[j // ncol][j % ncol]
        a, b = pa["subs"].get(t, Counter()), pb["subs"].get(t, Counter())
        keys = set(a) | set(b)
        scored = sorted(keys, key=lambda k: max(a.get(k, 0), b.get(k, 0)), reverse=True)[:top_n]
        y = np.arange(len(scored))
        ca = [a.get(k, 0) for k in scored]
        cb = [b.get(k, 0) for k in scored]
        h = 0.38
        ax.barh(y + h / 2, ca, h, color="#4C72B0", label=pa["label"])
        ax.barh(y - h / 2, cb, h, color="#DD8452", label=pb["label"])
        for i, k in enumerate(scored):
            if k in a and k in b:
                ax.scatter(max(ca[i], cb[i]) + 0.2, i, marker="*", s=110, c="#1a9850", zorder=5)
        ax.set_yticks(y)
        ax.set_yticklabels(scored, fontsize=8)
        ax.set_xlabel("survivor count")
        ax.set_title(f"{PROTEIN_LABEL.get(t, t)}   (* = shared)")
        ax.legend(fontsize=7)
    for j in range(len(tags), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"[product figure] saved -> {out_png}")


def make_figure(sa, sb, out_png):
    if not HAVE_MPL:
        return
    labels = [sa["label"], sb["label"]]
    series = [sa, sb]
    tags = [t for t in PROTEIN_ORDER if any(t in s["survivors"] for s in series)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: survivor counts per protein, grouped bars
    x = np.arange(len(tags))
    w = 0.38
    for i, s in enumerate(series):
        counts = [s["survivors"].get(t, (0, []))[0] for t in tags]
        axes[0].bar(x + (i - 0.5) * w, counts, w, label=s["label"],
                    color="#4C72B0" if i == 0 else "#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([PROTEIN_LABEL[t] for t in tags], rotation=15, ha="right")
    axes[0].set_ylabel("surviving mutants")
    axes[0].set_title("Survivor yield (Path B)")
    axes[0].legend()

    # Panel 2: Q distribution per protein, grouped box plots
    pos = []
    flier = dict(marker=".", markersize=2, alpha=0.4)
    for j, t in enumerate(tags):
        for i, s in enumerate(series):
            qs = s["survivors"].get(t, (0, []))[1]
            pp = j * 2 + i
            axes[1].boxplot(qs, positions=[pp], widths=0.7,
                            patch_artist=True, showfliers=True, flierprops=flier,
                            boxprops=dict(facecolor="#4C72B0" if i == 0 else "#DD8452", alpha=0.85))
            pos.append(pp)
    axes[1].set_xticks([j * 2 + 0.5 for j in range(len(tags))])
    axes[1].set_xticklabels([PROTEIN_LABEL[t] for t in tags], rotation=15, ha="right")
    axes[1].set_ylabel("survivor Q (native-contact agreement)")
    axes[1].axhline(0.5, ls="--", c="grey", lw=0.8)
    axes[1].text(len(tags) - 0.9, 0.52, "Q gate", c="grey", fontsize=8)
    axes[1].set_title("Survivor quality (Q)")
    # fake legend
    axes[1].plot([], [], color="#4C72B0", label=sa["label"])
    axes[1].plot([], [], color="#DD8452", label=sb["label"])
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"[figure] saved -> {out_png}")


def main():
    args = parse_args()
    labels = dict(kv.split("=", 1) for kv in args.labels.split(";") if "=" in kv)
    scales = [s for s in args.scales.split(",") if s]

    summaries = {}
    for sc in scales:
        label = labels.get(sc, sc)
        s = summarize(args.base, sc, args.subdir, label)
        summaries[sc] = s
        print_summary(s)
        print()

    if len(summaries) >= 2:
        sa, sb = summaries[scales[0]], summaries[scales[1]]
        print("=== Mann-Whitney U (survivor Q: n10 vs n45000) ===")
        rows = stats_table(sa, sb)
        print(f"{'tag':<8}{'n10_n':>6}{'n45_n':>6}{'n10_med':>9}{'n45_med':>9}{'p':>12}")
        for r in rows:
            print(f"{r['tag']:<8}{r['n10_n']:>6}{r['n45_n']:>6}"
                  f"{r['n10_med']:>9.3f}{r['n45_med']:>9.3f}{r['p']:>12.4g}")

        out_dir = args.out_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "figures")
        os.makedirs(out_dir, exist_ok=True)
        tag = os.path.basename(os.path.normpath(args.base))
        make_figure(sa, sb, os.path.join(out_dir, f"coverage_axis_{tag}.png"))
        # CSV report
        rep = os.path.join(out_dir, f"coverage_axis_{tag}.csv")
        with open(rep, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[report] saved -> {rep}")

        # product (mutant chemistry) analysis
        pa = product_summary(args.base, scales[0], args.subdir, summaries[scales[0]]["label"])
        pb = product_summary(args.base, scales[1], args.subdir, summaries[scales[1]]["label"])
        print_product(pa, pb)
        make_product_figure(pa, pb, os.path.join(out_dir, f"product_chemistry_{tag}.png"))
        prod = []
        for t in [x for x in PROTEIN_ORDER if x in pa["subs"] or x in pb["subs"]]:
            a, b = pa["subs"].get(t, Counter()), pb["subs"].get(t, Counter())
            for k in set(a) | set(b):
                prod.append({"tag": t, "substitution": k, "n10": a.get(k, 0),
                             "n45": b.get(k, 0), "shared": k in a and k in b})
        prodp = os.path.join(out_dir, f"product_chemistry_{tag}.csv")
        with open(prodp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tag", "substitution", "n10", "n45", "shared"])
            w.writeheader()
            w.writerows(sorted(prod, key=lambda r: (-r["n10"] - r["n45"], r["tag"])))
        print(f"[product report] saved -> {prodp}")


if __name__ == "__main__":
    main()
