#!/usr/bin/env python3
"""FireProtDB → SQLite 流式转换（2026-08-14）。

把 4.8 GB 的 PostgreSQL pg_dump（`01_fireprotdb_2025-09-20.sql`）流式灌进 SQLite：
- 单遍扫描：遇到 `CREATE TABLE` 就建表（解析 PG 列类型→SQLite affinity），
  遇到 `COPY public.<t> (cols) FROM stdin;` 就逐行解析、批插。
- O(1) 内存：按行迭代 + 5000 行/批 executemany + 每表一个事务。
- 正确处理 PG COPY text 转义：`\\N`→NULL、`\\\\`、`\\t`、`\\n`、`\\r`、`\\ooo` 八进制。

用法：
    python3 scripts/fireprotdb_to_sqlite.py \
        --sql data/stability_benchmark/fireprotdb/01_fireprotdb_2025-09-20.sql \
        --db  data/stability_benchmark/fireprotdb/fireprotdb.sqlite3
可选 --tables substitution,measurement,... 只灌指定表（跳过建其他表）。
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time

# PG 类型 → SQLite affinity
_INT = ("int", "serial", "bool")
_REAL = ("real", "double", "numeric", "decimal", "float")


def pg_to_sqlite(pg_type: str) -> str:
    t = (pg_type or "").lower()
    if any(k in t for k in _INT):
        return "INTEGER"
    if any(k in t for k in _REAL):
        return "REAL"
    return "TEXT"


def unescape_copy(v: str):
    """PG COPY text 字段解转义：\\N→None，其余反斜杠转义还原。"""
    if v == "\\N":
        return None
    if "\\" not in v:
        return v
    out = []
    i, n = 0, len(v)
    simple = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\"}
    while i < n:
        c = v[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            out.append("\\")
            i += 1
            continue
        nxt = v[i + 1]
        if nxt in simple:
            out.append(simple[nxt])
            i += 2
        elif nxt == "N":
            # 字段整体为 \N 已在开头处理；此处保守按字面 'N' 还原
            out.append("N")
            i += 2
        elif nxt in "01234567" and i + 3 < n and all(c in "01234567" for c in v[i + 1 : i + 4]):
            out.append(chr(int(v[i + 1 : i + 4], 8)))
            i += 4
        else:
            out.append(nxt)
            i += 2
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sql", required=True, help="FireProtDB pg_dump 文件路径")
    ap.add_argument("--db", required=True, help="输出 SQLite 路径")
    ap.add_argument("--tables", default="", help="可选：只灌这些表（逗号分隔）；空=全部")
    ap.add_argument("--batch", type=int, default=5000, help="批插行数")
    args = ap.parse_args()

    only = {t.strip() for t in args.tables.split(",") if t.strip()} or None

    t0 = time.time()
    con = sqlite3.connect(args.db)
    cur = con.cursor()
    cur.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=OFF;
        PRAGMA cache_size=-200000;
        PRAGMA temp_store=MEMORY;
        PRAGMA busy_timeout=5000;
        """
    )

    stats: dict = {}
    copied_tables = set()
    current_table = None
    cols = []
    rows_buf = []
    table_rows = 0

    re_copy = re.compile(r'^COPY public\.([a-z_0-9]+) \((.*)\) FROM stdin;')
    re_create = re.compile(r'^CREATE TABLE public\.([a-z_0-9]+) \(')

    def _insert_sql(tname: str, col_names) -> str:
        colq = ",".join('"%s"' % c for c in col_names)
        ph = ",".join("?" * len(col_names))
        return 'INSERT INTO "%s" (%s) VALUES (%s)' % (tname, colq, ph)

    def flush():
        nonlocal rows_buf, table_rows, current_table
        if not current_table or not rows_buf:
            return
        cur.executemany(_insert_sql(current_table, cols), rows_buf)
        rows_buf = []
        table_rows += args.batch  # 近似；末尾小批用实际值修正

    def close_table():
        nonlocal current_table, rows_buf, table_rows
        if current_table and rows_buf:
            cur.executemany(_insert_sql(current_table, cols), rows_buf)
            rows_buf = []
        if current_table is not None:
            con.commit()
            stats[current_table] = table_rows
            print(f"  ✔ {current_table}: {table_rows:,} 行", flush=True)
            current_table, table_rows = None, 0

    create_parse = None  # (table_name, pending_col_lines)
    total_rows = 0
    with open(args.sql, "r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n\r")

            # ---- CREATE TABLE 解析 ----
            if create_parse is not None:
                tname, col_lines = create_parse
                if line.strip().startswith(");"):
                    # 建表
                    create_parse = None
                    if col_lines is not None:
                        cols_t = []
                        for cl in col_lines:
                            s = cl.strip().rstrip(",")
                            if not s or s.startswith("CONSTRAINT") or s.startswith("PRIMARY"):
                                continue
                            # 在第一个约束关键字处截断，取 "列名 类型"
                            head = re.split(
                                r"\s+(?:NOT\s+NULL|DEFAULT|PRIMARY|REFERENCES|CHECK|UNIQUE|COLLATE)\b",
                                s, maxsplit=1, flags=re.I,
                            )[0]
                            parts = head.split()
                            if not parts:
                                continue
                            cname = parts[0].strip('"')
                            ctype = parts[1] if len(parts) > 1 else "TEXT"
                            cols_t.append((cname, pg_to_sqlite(ctype)))
                        if cols_t:
                            cur.execute(
                                f'CREATE TABLE IF NOT EXISTS "{tname}" ('
                                + ", ".join(f'"{cn}" {ct}' for cn, ct in cols_t)
                                + ")"
                            )
                    continue
                if col_lines is not None:
                    col_lines.append(line)
                continue

            m = re_create.match(line)
            if m:
                tname = m.group(1)
                skip = only is not None and tname not in only
                create_parse = (tname, None if skip else [])  # 跳过时也消费到 `);`
                continue

            # ---- COPY 数据解析 ----
            m = re_copy.match(line)
            if m:
                tname = m.group(1)
                close_table()  # 上一表收尾
                current_table = tname
                cols = [c.strip().strip('"') for c in m.group(2).split(",")]
                copied_tables.add(tname)
                table_rows = 0
                continue

            if current_table is not None:
                if line == "\\.":  # COPY 结束标记
                    close_table()
                    continue
                if not line:
                    continue
                fields = line.split("\t")
                if len(fields) != len(cols):
                    # 脏行：补齐/截断，避免崩
                    if len(fields) > len(cols):
                        fields = fields[: len(cols)]
                    else:
                        fields += [None] * (len(cols) - len(fields))
                rows_buf.append([unescape_copy(f) for f in fields])
                if len(rows_buf) >= args.batch:
                    flush()
                    total_rows += args.batch

        close_table()

    # 行数修正：flush 用的近似值，用实际表行数覆盖
    cur.executescript("VACUUM;")
    con.commit()

    print("\n=== 完成 ===")
    for t in sorted(copied_tables):
        try:
            n = cur.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            print(f"  {t}: {n:,} 行", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {t}: 查询失败 {e}", flush=True)
    con.close()
    print(f"用时 {time.time() - t0:.1f}s，DB={args.db}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
