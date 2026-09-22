#!/usr/bin/env python3
"""Smoke test: boot uvicorn on :8123 and verify /api/analyze for ADBE.

Usage:  python3 test_smoke.py
Needs:  pip install -r requirements.txt  (and internet for yfinance)
"""
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request

PORT = 8123
failures = []

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  << {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)

def get(path, params=None):
    url = f"http://127.0.0.1:{PORT}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.status, json.loads(r.read().decode())

srv = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(PORT)],
    cwd=".", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try:
            st, body = get("/api/health")
            if st == 200 and body.get("ok"):
                break
        except Exception:
            time.sleep(1)
    else:
        check("server boot", False, "uvicorn did not come up")
        raise SystemExit(1)
    check("health", True)

    st, d = get("/api/analyze", {"ticker": "ADBE", "mode": "stock"})
    check("stock 200", st == 200)
    check("headline mentions ADBE + a price",
          "ADBE" in d.get("verdict_headline", "") and "$" in d.get("verdict_headline", ""),
          d.get("verdict_headline"))
    check("regime summary composed from values",
          "SPY" in d["regime"]["summary"] and "VIX" in d["regime"]["summary"],
          d["regime"]["summary"][:140])
    check("technicals composed", "RSI" in d["technicals"]["summary"],
          d["technicals"]["summary"][:140])
    check("verdict in set", d.get("verdict") in ("GO", "CAUTION", "NO-GO"), d.get("verdict"))

    st, d2 = get("/api/analyze", {"ticker": "ADBE", "mode": "entry"})
    check("entry 200", st == 200)
    check("entry evaluation differs from stock",
          d2["mode_evaluation"]["summary"] != d["mode_evaluation"]["summary"])

    st, d3 = get("/api/analyze", {"ticker": "ADBE", "mode": "csp",
                                  "strike": 240, "credit": 3.10, "delta": 0.22,
                                  "iv": 0.45, "dte": 34, "contracts": 1})
    m = d3["mode_evaluation"]
    check("csp 200", st == 200)
    check("csp summary names contract terms",
          "240" in m["summary"] and "breakeven" in m["summary"].lower(),
          m["summary"][:180])
    check("csp summary != stock summary", m["summary"] != d["mode_evaluation"]["summary"])
    check("csp metrics present", bool(m.get("metrics", {}).get("breakeven")),
          json.dumps(m.get("metrics"))[:140])

    # Same-response-bug regression: two different tickers must not share verdict text.
    st, d4 = get("/api/analyze", {"ticker": "INTC", "mode": "stock"})
    check("INTC 200", st == 200)
    check("tickers read differently",
          d4["verdict_headline"] != d["verdict_headline"]
          or d4["technicals"]["summary"] != d["technicals"]["summary"],
          "ADBE vs INTC produced byte-identical prose")

    check("sections timestamped",
          all(d[s].get("as_of") for s in ("regime", "technicals", "valuation", "catalysts")))
finally:
    srv.terminate()

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("All smoke checks passed.")
