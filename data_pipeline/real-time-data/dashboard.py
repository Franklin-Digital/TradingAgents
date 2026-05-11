"""Real-time scanner dashboard (Flask + React).

Serves a single-page React dashboard backed by a Flask JSON API that reads
from QuestDB over its HTTP `/exec` endpoint.

Run:
    pip install flask flask-cors requests
    python dashboard.py --env prod
    python dashboard.py --env nonprod --questdb-host localhost --port 8052
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

log = logging.getLogger("dashboard")


# ---------------------------------------------------------------------------
# QuestDB client
# ---------------------------------------------------------------------------

class QuestDB:
    """Thin wrapper over QuestDB's HTTP /exec endpoint."""

    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def exec(self, query: str) -> dict[str, Any]:
        r = requests.get(
            f"{self.base_url}/exec",
            params={"query": query},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def rows(self, query: str) -> list[dict[str, Any]]:
        """Run a query and return a list of dict rows keyed by column name."""
        data = self.exec(query)
        cols = [c["name"] for c in data.get("columns", [])]
        return [dict(zip(cols, row)) for row in data.get("dataset", [])]


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def create_app(env: str, qdb: QuestDB) -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "env": env})

    @app.get("/api/universe")
    def api_universe():
        try:
            rows = qdb.rows(
                "SELECT * FROM active_universe LATEST ON ts PARTITION BY conid"
            )
            return jsonify({"rows": rows, "count": len(rows)})
        except Exception as exc:  # noqa: BLE001
            log.exception("universe query failed")
            return jsonify({"error": str(exc), "rows": [], "count": 0}), 500

    @app.get("/api/stats")
    def api_stats():
        try:
            data = qdb.rows(
                "SELECT count() AS cnt, max(ts) AS last_scan "
                "FROM scanner_results "
                "WHERE ts > dateadd('h', -1, now())"
            )
            row = data[0] if data else {"cnt": 0, "last_scan": None}
            return jsonify({
                "count_last_hour": row.get("cnt", 0),
                "last_scan": row.get("last_scan"),
            })
        except Exception as exc:  # noqa: BLE001
            log.exception("stats query failed")
            return jsonify({
                "error": str(exc),
                "count_last_hour": 0,
                "last_scan": None,
            }), 500

    @app.get("/api/sectors")
    def api_sectors():
        try:
            rows = qdb.rows(
                "SELECT sector, count() AS n, avg(score) AS avg_score "
                "FROM (SELECT * FROM active_universe "
                "      LATEST ON ts PARTITION BY conid) "
                "WHERE sector IS NOT NULL "
                "GROUP BY sector ORDER BY n DESC"
            )
            return jsonify({"rows": rows})
        except Exception as exc:  # noqa: BLE001
            log.exception("sectors query failed")
            return jsonify({"error": str(exc), "rows": []}), 500

    @app.get("/")
    def index():
        return INDEX_HTML.replace("__ENV__", env)

    return app


# ---------------------------------------------------------------------------
# Frontend (single-file React via CDN + Babel)
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Scanner Dashboard</title>
<style>
  :root {
    --bg: #0f1117;
    --panel: #161a23;
    --panel-2: #1d222e;
    --border: #262c3a;
    --text: #e6e9ef;
    --muted: #8a93a6;
    --accent: #4f8cff;
    --green: #2ecc71;
    --amber: #f5a623;
    --red: #ef4444;
    --blue: #3b82f6;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
  }
  .wrap { max-width: 1500px; margin: 0 auto; padding: 18px 24px 48px; }

  header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px;
  }
  header h1 { font-size: 18px; font-weight: 600; margin: 0; letter-spacing: .2px; }
  .env-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 999px;
    font-size: 12px; font-weight: 600; letter-spacing: .5px;
    border: 1px solid var(--border);
  }
  .env-prod    { color: var(--green); border-color: rgba(46,204,113,.4); background: rgba(46,204,113,.08); }
  .env-nonprod { color: var(--amber); border-color: rgba(245,166,35,.4); background: rgba(245,166,35,.08); }

  .stats {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
    margin-bottom: 18px;
  }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px;
  }
  .card .label { color: var(--muted); font-size: 11px; text-transform: uppercase;
                 letter-spacing: .8px; margin-bottom: 6px; }
  .card .value { font-size: 22px; font-weight: 600; }

  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
  .chip {
    padding: 6px 12px; border-radius: 999px; cursor: pointer;
    background: var(--panel); border: 1px solid var(--border);
    color: var(--text); font-size: 12px; user-select: none;
  }
  .chip.active { background: var(--accent); border-color: var(--accent); color: #fff; }
  .chip .n { color: var(--muted); margin-left: 6px; }
  .chip.active .n { color: rgba(255,255,255,.85); }

  .controls {
    display: flex; gap: 10px; align-items: center; margin-bottom: 14px;
    flex-wrap: wrap;
  }
  .controls input, .controls select {
    background: var(--panel); border: 1px solid var(--border); color: var(--text);
    padding: 8px 10px; border-radius: 8px; font-size: 13px; outline: none;
  }
  .controls input { min-width: 220px; }
  .controls input:focus, .controls select:focus { border-color: var(--accent); }
  .toggle-btn {
    background: var(--panel); border: 1px solid var(--border); color: var(--text);
    padding: 8px 12px; border-radius: 8px; cursor: pointer; font-size: 13px;
  }
  .toggle-btn.on { background: var(--accent); border-color: var(--accent); color: #fff; }

  table { width: 100%; border-collapse: collapse; background: var(--panel);
          border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
  thead th {
    text-align: left; font-size: 11px; text-transform: uppercase;
    letter-spacing: .6px; color: var(--muted); font-weight: 600;
    padding: 10px 12px; border-bottom: 1px solid var(--border);
    cursor: pointer; user-select: none; white-space: nowrap;
  }
  thead th .arrow { color: var(--accent); margin-left: 4px; }
  tbody td {
    padding: 10px 12px; border-bottom: 1px solid var(--border);
    font-size: 13px; vertical-align: middle;
  }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover { background: var(--panel-2); }
  td.sym { font-weight: 600; }
  td.rank { color: var(--muted); width: 50px; }

  .score-cell { min-width: 140px; }
  .score-bar { background: var(--border); height: 6px; border-radius: 3px; overflow: hidden;
               position: relative; }
  .score-fill { height: 100%; background: linear-gradient(90deg, #4f8cff, #2ecc71); }
  .score-num { font-size: 11px; color: var(--muted); margin-top: 2px; }

  .pill {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    background: var(--panel-2); border: 1px solid var(--border);
    font-size: 11px; color: var(--muted);
  }
  .tier { font-weight: 600; font-size: 11px; }
  .tier-1 { color: var(--green); }
  .tier-2 { color: var(--accent); }
  .tier-3 { color: var(--amber); }

  .badges { display: flex; flex-wrap: wrap; gap: 4px; }
  .badge {
    padding: 2px 7px; border-radius: 4px; font-size: 10px; font-weight: 600;
    letter-spacing: .3px; color: #fff;
  }
  .b-GAPUP  { background: var(--green); }
  .b-GAPDN  { background: var(--red); }
  .b-VOL    { background: var(--blue); }
  .b-HOT    { background: var(--amber); }
  .b-PCT_UP { background: #16a34a; }
  .b-PCT_DN { background: #dc2626; }
  .b-MOM    { background: #8b5cf6; }
  .b-NEWS   { background: #06b6d4; }
  .b-OTHER  { background: #475569; }

  .empty { color: var(--muted); text-align: center; padding: 40px; }
  .err   { color: var(--red); margin-top: 10px; font-size: 12px; }
</style>
</head>
<body>
<div id="root"></div>

<script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

<script type="text/babel">
const ENV = "__ENV__";
const REFRESH_MS = 30000;

const SCANNER_BADGES = {
  GAPUP:  { cls: "b-GAPUP",  label: "GAP↑" },
  GAPDN:  { cls: "b-GAPDN",  label: "GAP↓" },
  VOL:    { cls: "b-VOL",    label: "VOL"  },
  HOT:    { cls: "b-HOT",    label: "HOT"  },
  PCT_UP: { cls: "b-PCT_UP", label: "%↑"   },
  PCT_DN: { cls: "b-PCT_DN", label: "%↓"   },
  MOM:    { cls: "b-MOM",    label: "MOM"  },
  NEWS:   { cls: "b-NEWS",   label: "NEWS" },
};

function classifyScanner(name) {
  if (!name) return "OTHER";
  const n = name.toUpperCase();
  if (n.includes("GAP") && (n.includes("UP") || n.includes("HIGH"))) return "GAPUP";
  if (n.includes("GAP") && (n.includes("DN") || n.includes("DOWN") || n.includes("LOW"))) return "GAPDN";
  if (n.includes("HOT")) return "HOT";
  if (n.includes("VOL")) return "VOL";
  if (n.includes("PCT") && (n.includes("UP") || n.includes("GAIN"))) return "PCT_UP";
  if (n.includes("PCT") && (n.includes("DN") || n.includes("LOSE"))) return "PCT_DN";
  if (n.includes("MOM")) return "MOM";
  if (n.includes("NEWS")) return "NEWS";
  return "OTHER";
}

function parseScannerList(row) {
  const raw = row.scanners ?? row.scanner_list ?? row.scanner ?? "";
  if (Array.isArray(raw)) return raw.filter(Boolean);
  return String(raw)
    .split(/[,;|]/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function fmtTime(ts) {
  if (!ts) return "—";
  const d = new Date(ts);
  if (isNaN(d.getTime())) return String(ts);
  return d.toLocaleTimeString();
}

function Badge({ scanner }) {
  const key = classifyScanner(scanner);
  const meta = SCANNER_BADGES[key] || { cls: "b-OTHER", label: scanner };
  return <span className={"badge " + meta.cls} title={scanner}>{meta.label}</span>;
}

function StatCard({ label, value }) {
  return (
    <div className="card">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}

function ScoreBar({ score }) {
  const v = Math.max(0, Math.min(100, Number(score) || 0));
  return (
    <div className="score-cell">
      <div className="score-bar"><div className="score-fill" style={{ width: v + "%" }} /></div>
      <div className="score-num">{v.toFixed(1)}</div>
    </div>
  );
}

function App() {
  const [universe, setUniverse] = React.useState([]);
  const [stats, setStats]       = React.useState({ count_last_hour: 0, last_scan: null });
  const [sectors, setSectors]   = React.useState([]);
  const [err, setErr]           = React.useState(null);

  const [search, setSearch]     = React.useState("");
  const [sectorSel, setSectorSel] = React.useState(null);
  const [scannerSel, setScannerSel] = React.useState("");
  const [multiOnly, setMultiOnly] = React.useState(false);
  const [sortKey, setSortKey]   = React.useState("score");
  const [sortDir, setSortDir]   = React.useState("desc");

  const fetchAll = React.useCallback(async () => {
    try {
      const [u, s, sec] = await Promise.all([
        fetch("/api/universe").then((r) => r.json()),
        fetch("/api/stats").then((r) => r.json()),
        fetch("/api/sectors").then((r) => r.json()),
      ]);
      setUniverse(u.rows || []);
      setStats(s || {});
      setSectors(sec.rows || []);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  React.useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, REFRESH_MS);
    return () => clearInterval(id);
  }, [fetchAll]);

  const allScanners = React.useMemo(() => {
    const set = new Set();
    universe.forEach((r) => parseScannerList(r).forEach((x) => set.add(x)));
    return Array.from(set).sort();
  }, [universe]);

  const rows = React.useMemo(() => {
    const q = search.trim().toUpperCase();
    return universe.filter((r) => {
      if (q && !String(r.symbol || "").toUpperCase().includes(q)) return false;
      if (sectorSel && r.sector !== sectorSel) return false;
      const scs = parseScannerList(r);
      if (scannerSel && !scs.includes(scannerSel)) return false;
      if (multiOnly && scs.length < 2) return false;
      return true;
    });
  }, [universe, search, sectorSel, scannerSel, multiOnly]);

  const sorted = React.useMemo(() => {
    const dir = sortDir === "asc" ? 1 : -1;
    const get = (r) => {
      if (sortKey === "scanner_count") return parseScannerList(r).length;
      const v = r[sortKey];
      return v == null ? (typeof v === "number" ? 0 : "") : v;
    };
    return [...rows].sort((a, b) => {
      const av = get(a), bv = get(b);
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * dir;
      return String(av).localeCompare(String(bv)) * dir;
    });
  }, [rows, sortKey, sortDir]);

  const topSector = sectors.length ? sectors[0].sector : "—";
  const avgScore = universe.length
    ? (universe.reduce((s, r) => s + (Number(r.score) || 0), 0) / universe.length).toFixed(1)
    : "—";

  function toggleSort(k) {
    if (sortKey === k) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(k); setSortDir("desc"); }
  }
  const arrow = (k) => sortKey === k ? <span className="arrow">{sortDir === "asc" ? "↑" : "↓"}</span> : null;

  const isProd = ENV === "prod";

  return (
    <div className="wrap">
      <header>
        <h1>Real-Time Scanner Dashboard</h1>
        <span className={"env-badge " + (isProd ? "env-prod" : "env-nonprod")}>
          {isProd ? "● PROD LIVE" : "◐ NONPROD"}
        </span>
      </header>

      <div className="stats">
        <StatCard label="Active Symbols" value={universe.length} />
        <StatCard label="Top Sector"     value={topSector} />
        <StatCard label="Avg Score"      value={avgScore} />
        <StatCard label="Last Scan"      value={fmtTime(stats.last_scan)} />
      </div>

      <div className="chips">
        <span className={"chip " + (sectorSel === null ? "active" : "")}
              onClick={() => setSectorSel(null)}>
          All <span className="n">{universe.length}</span>
        </span>
        {sectors.map((s) => (
          <span key={s.sector}
                className={"chip " + (sectorSel === s.sector ? "active" : "")}
                onClick={() => setSectorSel(sectorSel === s.sector ? null : s.sector)}>
            {s.sector} <span className="n">{s.n}</span>
          </span>
        ))}
      </div>

      <div className="controls">
        <input
          type="text" placeholder="Search symbol…"
          value={search} onChange={(e) => setSearch(e.target.value)}
        />
        <select value={scannerSel} onChange={(e) => setScannerSel(e.target.value)}>
          <option value="">All scanners</option>
          {allScanners.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <button
          className={"toggle-btn " + (multiOnly ? "on" : "")}
          onClick={() => setMultiOnly((v) => !v)}
        >
          Multi-scanner only
        </button>
        <span style={{ color: "var(--muted)", fontSize: 12, marginLeft: "auto" }}>
          {sorted.length} of {universe.length} · refresh 30s
        </span>
      </div>

      {err && <div className="err">Error: {err}</div>}

      <table>
        <thead>
          <tr>
            <th onClick={() => toggleSort("rank")}>Rank{arrow("rank")}</th>
            <th onClick={() => toggleSort("symbol")}>Symbol{arrow("symbol")}</th>
            <th onClick={() => toggleSort("score")}>Score{arrow("score")}</th>
            <th onClick={() => toggleSort("sector")}>Sector{arrow("sector")}</th>
            <th onClick={() => toggleSort("tier")}>Tier{arrow("tier")}</th>
            <th onClick={() => toggleSort("scanner_count")}>#Scanners{arrow("scanner_count")}</th>
            <th>Scanners</th>
            <th onClick={() => toggleSort("exchange")}>Exchange{arrow("exchange")}</th>
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 && (
            <tr><td colSpan="8" className="empty">No symbols match the current filters.</td></tr>
          )}
          {sorted.map((r, i) => {
            const scs = parseScannerList(r);
            const tier = r.tier ?? "—";
            return (
              <tr key={r.conid || r.symbol || i}>
                <td className="rank">{r.rank ?? i + 1}</td>
                <td className="sym">{r.symbol}</td>
                <td><ScoreBar score={r.score} /></td>
                <td><span className="pill">{r.sector || "—"}</span></td>
                <td className={"tier tier-" + tier}>{tier !== "—" ? "T" + tier : "—"}</td>
                <td>{scs.length}</td>
                <td><div className="badges">{scs.map((s) => <Badge key={s} scanner={s} />)}</div></td>
                <td><span className="pill">{r.exchange || "—"}</span></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _defaults(env: str) -> dict[str, int]:
    if env == "prod":
        return {"questdb_port": 9000, "dashboard_port": 8051}
    return {"questdb_port": 19000, "dashboard_port": 8052}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time scanner dashboard")
    p.add_argument("--env", choices=["prod", "nonprod"], default="nonprod")
    p.add_argument("--questdb-host", default="localhost")
    p.add_argument("--questdb-port", type=int, default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=None)
    args = p.parse_args(argv)

    d = _defaults(args.env)
    if args.questdb_port is None:
        args.questdb_port = d["questdb_port"]
    if args.port is None:
        args.port = d["dashboard_port"]
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
    )
    args = parse_args(argv)
    qdb = QuestDB(args.questdb_host, args.questdb_port)
    app = create_app(args.env, qdb)
    log.info(
        "starting dashboard env=%s questdb=%s:%d listen=%s:%d",
        args.env, args.questdb_host, args.questdb_port, args.host, args.port,
    )
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
