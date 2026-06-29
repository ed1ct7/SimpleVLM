#!/usr/bin/env python3
"""Администрирование БД результатов (SQLite) — без сторонних зависимостей.

Команды (раздел «администрирование БД» практики):
    python solution/db/admin.py stats      # размеры таблиц, индексы, представления
    python solution/db/admin.py check       # integrity_check + foreign_key_check
    python solution/db/admin.py backup       # резервная копия БД (онлайн, .backup)
    python solution/db/admin.py optimize     # VACUUM + ANALYZE (дефрагментация + статистика)
    python solution/db/admin.py plan         # EXPLAIN QUERY PLAN для отчётного запроса
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
import time
from pathlib import Path

DB = Path(__file__).resolve().parent / "vk_vlm.sqlite"


def _con():
    if not DB.exists():
        sys.exit(f"нет БД: {DB} — сначала `python solution/db/build_db.py`")
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def stats(con):
    print(f"БД: {DB}  ({DB.stat().st_size/1024:.0f} КБ)")
    print(f"page_size={con.execute('PRAGMA page_size').fetchone()[0]}  "
          f"page_count={con.execute('PRAGMA page_count').fetchone()[0]}")
    print("\nтаблицы (строк):")
    for (name,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  {name:<20} {n:>8}")
    print("\nпредставления (VIEW):")
    for (name,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name"):
        print(f"  {name}")
    print("\nиндексы:")
    for name, tbl in con.execute(
        "SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY tbl_name"):
        print(f"  {name:<24} → {tbl}")


def check(con):
    print("integrity_check:", con.execute("PRAGMA integrity_check").fetchone()[0])
    fk = con.execute("PRAGMA foreign_key_check").fetchall()
    print("foreign_key_check:", "OK (нарушений нет)" if not fk else f"НАРУШЕНИЯ: {fk}")


def backup(con):
    dst = DB.with_name(f"vk_vlm.backup-{time.strftime('%Y%m%d-%H%M%S')}.sqlite")
    bck = sqlite3.connect(dst)
    con.backup(bck)            # онлайн-бэкап средствами SQLite (консистентный)
    bck.close()
    print(f"резервная копия: {dst}  ({dst.stat().st_size/1024:.0f} КБ)")


def optimize(con):
    con.execute("ANALYZE")     # пересобрать статистику для планировщика
    con.commit()
    con.execute("VACUUM")      # дефрагментация файла
    print("ANALYZE + VACUUM выполнены")


def plan(con):
    sql = ("SELECT category, AVG(correct) FROM mmbench_prediction "
           "WHERE model_key='mine-8b' GROUP BY category")
    print("запрос:", sql)
    print("план выполнения:")
    for row in con.execute("EXPLAIN QUERY PLAN " + sql):
        print("  ", " ".join(str(x) for x in row))


CMDS = {"stats": stats, "check": check, "backup": backup, "optimize": optimize, "plan": plan}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in CMDS:
        sys.exit(f"использование: python {Path(__file__).name} {{{'|'.join(CMDS)}}}")
    con = _con()
    CMDS[sys.argv[1]](con)
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
