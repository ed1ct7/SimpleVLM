#!/usr/bin/env python3
"""REST API + веб-дашборд над БД результатов — на стандартной библиотеке (без зависимостей).

Маленькое приложение поверх SQLite: отдаёт метрики/срезы по HTTP в JSON и рисует дашборд.
Запросы параметризованы (защита от SQL-инъекций), имя модели валидируется по справочнику.

    python solution/db/api.py            # http://127.0.0.1:8000
Эндпоинты:
    GET /                      — HTML-дашборд
    GET /api/leaderboard       — сводная таблица метрик (VIEW v_leaderboard)
    GET /api/models            — справочник моделей
    GET /api/skills?model=...  — точность по навыкам MMBench (VIEW v_skill_accuracy)
    GET /api/errors?model=...&limit=N — примеры ошибок GQA
"""
from __future__ import annotations

import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

DB = Path(__file__).resolve().parent / "vk_vlm.sqlite"


def query(sql, params=()):
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def valid_models():
    return {r["model_key"] for r in query("SELECT model_key FROM model")}


DASHBOARD = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>VK-VLM — результаты</title><style>
body{font:15px system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.3rem} table{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}
th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left}
th{background:#f3f3f3} select{padding:.3rem}</style></head><body>
<h1>VK-VLM — результаты оценки</h1>
<h2>Лидерборд</h2><div id="lb">…</div>
<h2>Точность по навыкам (MMBench)</h2>
<select id="m"></select><div id="sk"></div>
<script>
const tbl=rows=>{if(!rows.length)return'<p>нет данных</p>';
 const c=Object.keys(rows[0]);return'<table><tr>'+c.map(k=>'<th>'+k+'</th>').join('')+
 '</tr>'+rows.map(r=>'<tr>'+c.map(k=>'<td>'+(r[k]??'')+'</td>').join('')+'</tr>').join('')+'</table>';};
fetch('/api/leaderboard').then(r=>r.json()).then(d=>lb.innerHTML=tbl(d));
fetch('/api/models').then(r=>r.json()).then(d=>{m.innerHTML=d.map(x=>
 '<option value="'+x.model_key+'">'+x.label+'</option>').join('');loadSkills();});
m.onchange=loadSkills;
function loadSkills(){fetch('/api/skills?model='+encodeURIComponent(m.value))
 .then(r=>r.json()).then(d=>sk.innerHTML=tbl(d));}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj, ensure_ascii=False), "application/json", code)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self._send(DASHBOARD, "text/html")
            if u.path == "/api/leaderboard":
                return self._json(query("SELECT * FROM v_leaderboard ORDER BY is_ours, mmbench"))
            if u.path == "/api/models":
                return self._json(query("SELECT model_key, label, is_ours FROM model"))
            if u.path == "/api/skills":
                model = q.get("model", [""])[0]
                if model not in valid_models():
                    return self._json({"error": "unknown model"}, 400)
                return self._json(query(
                    "SELECT category, n, accuracy FROM v_skill_accuracy "
                    "WHERE model_key=? ORDER BY accuracy DESC", (model,)))
            if u.path == "/api/errors":
                model = q.get("model", [""])[0]
                if model not in valid_models():
                    return self._json({"error": "unknown model"}, 400)
                limit = min(int(q.get("limit", ["10"])[0]), 100)
                return self._json(query(
                    "SELECT question, gold, pred FROM gqa_prediction "
                    "WHERE model_key=? AND correct_extracted=0 LIMIT ?", (model, limit)))
            return self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001 — вернуть 500 как JSON, не падать
            return self._json({"error": str(e)}, 500)

    def log_message(self, *_):  # тише в консоль
        pass


def main():
    if not DB.exists():
        raise SystemExit(f"нет БД: {DB} — сначала `python solution/db/build_db.py`")
    srv = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("API на http://127.0.0.1:8000  (Ctrl+C — стоп)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
