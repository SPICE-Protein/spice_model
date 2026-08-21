#!/usr/bin/env python3
"""Meltome Atlas CSV → SQLite 流式转换（2026-08-14）。

把 `cross-species.csv` / `human.csv`（~300 万行，长格式）流式灌进 SQLite：
- csv 模块逐行读（O(1) 内存），按配置做类型转换（REAL/INTEGER/TEXT），NA→NULL
- 批插（默认 10000 行）+ 每文件一个事务
- 末尾生成每蛋白汇总表（n、min/max/mean melt_point 等）

用法：
    python3 scripts/meltome_to_sqlite.py \
        --dir data/stability_benchmark/meltomeatlas \
        --db  data/stability_benchmark/meltomeatlas/meltome.sqlite3
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import time

# csv文件名 -> (表名, [(csv表头, sqlite列名, sqlite类型)])
SPECS = {
    "cross-species.csv": {
        "table": "meltome_cross_species",
        "columns": [
            ("run_name", "run_name", "TEXT"),
            ("Protein_ID", "protein_id", "TEXT"),
            ("gene_name", "gene_name", "TEXT"),
            ("meltPoint", "melt_point", "REAL"),
            ("channel", "channel", "TEXT"),
            ("fold_change", "fold_change", "REAL"),
            ("temperature", "temperature", "INTEGER"),
        ],
    },
    "human.csv": {
        "table": "meltome_human",
        "columns": [
            ("gene_name", "gene_name", "TEXT"),
            ("cell_line_or_type", "cell_line_or_type", "TEXT"),
            ("fold_change", "fold_change", "REAL"),
            ("temperature", "temperature", "INTEGER"),
            ("meltPoint", "melt_point", "REAL"),
            ("quan_norm_meltPoint", "quan_norm_melt_point", "REAL"),
        ],
    },
}


def convert(raw: str, typ: str):
    s = (raw or "").strip()
    if s == "" or s.upper() in ("NA", "N/A", "NAN", "NULL", "NONE"):
        return None
    if typ == "INTEGER":
        try:
            return int(float(s))
        except ValueError:
            return None
    if typ == "REAL":
        try:
            return float(s)
        except ValueError:
            return None
    return raw


def load_one(con, cur, csv_path: str, spec: dict, batch: int) -> int:
    table = spec["table"]
    cols = [(h, db, t) for h, db, t in spec["columns"]]
    db_cols = [c[1] for c in cols]
    type_by_header = {h: t for h, _, t in cols}
    cur.execute(
        'CREATE TABLE IF NOT EXISTS "%s" (%s)'
        % (table, ", ".join('"%s" %s' % (db, t) for _, db, t in cols))
    )
    n = 0
    buf = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = [header.index(h) for h, _, _ in cols]
        for row in reader:
            rec = []
            for i, (h, db, t) in zip(idx, cols):
                rec.append(convert(row[i] if i < len(row) else "", type_by_header[h]))
            buf.append(rec)
            if len(buf) >= batch:
                cur.executemany(
                    'INSERT INTO "%s" (%s) VALUES (%s)'
                    % (table, ",".join('"%s"' % c for c in db_cols), ",".join("?" * len(db_cols))),
                    buf,
                )
                n += len(buf)
                buf = []
    if buf:
        cur.executemany(
            'INSERT INTO "%s" (%s) VALUES (%s)'
            % (table, ",".join('"%s"' % c for c in db_cols), ",".join("?" * len(db_cols))),
            buf,
        )
        n += len(buf)
    con.commit()
    return n


def build_summaries(con, cur):
    # cross-species：按 protein_id 汇总（run_name 含物种）
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meltome_cross_species_summary AS
        SELECT protein_id,
               MAX(gene_name) AS gene_name,
               MAX(run_name) AS run_name,
               COUNT(*) AS n_rows,
               COUNT(DISTINCT melt_point) AS n_distinct_mp,
               MIN(melt_point) AS min_mp,
               MAX(melt_point) AS max_mp,
               ROUND(AVG(melt_point), 3) AS mean_mp
        FROM meltome_cross_species
        WHERE melt_point IS NOT NULL
        GROUP BY protein_id
        """
    )
    # human：按 gene+cell_line 汇总
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meltome_human_summary AS
        SELECT gene_name,
               cell_line_or_type,
               COUNT(*) AS n_rows,
               COUNT(DISTINCT melt_point) AS n_distinct_mp,
               MIN(melt_point) AS min_mp,
               MAX(melt_point) AS max_mp,
               ROUND(AVG(melt_point), 3) AS mean_mp,
               ROUND(AVG(quan_norm_melt_point), 3) AS mean_quan_norm
        FROM meltome_human
        WHERE melt_point IS NOT NULL
        GROUP BY gene_name, cell_line_or_type
        """
    )
    con.commit()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="meltomeatlas 目录（含两份 csv）")
    ap.add_argument("--db", required=True, help="输出 SQLite 路径")
    ap.add_argument("--batch", type=int, default=10000)
    args = ap.parse_args()

    t0 = time.time()
    con = sqlite3.connect(args.db)
    cur = con.cursor()
    cur.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=OFF;
        PRAGMA cache_size=-200000;
        PRAGMA temp_store=MEMORY;
        """
    )
    for fn, spec in SPECS.items():
        p = os.path.join(args.dir, fn)
        if not os.path.exists(p):
            print(f"跳过（不存在）: {p}", flush=True)
            continue
        n = load_one(con, cur, p, spec, args.batch)
        print(f"  ✔ {spec['table']}: {n:,} 行 ({time.time()-t0:.1f}s)", flush=True)
    build_summaries(con, cur)
    print("  汇总表已建", flush=True)
    con.close()
    print(f"完成，用时 {time.time()-t0:.1f}s，DB={args.db}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
