"""
Trading Analyst Agent — self-hostable single service.

FastAPI backend serving a single-page UI and a JSON analysis endpoint.
Live market data comes from yfinance only (no API keys).
All conclusion prose is composed from the computed values of each run —
no canned verdict strings — so different tickers/modes genuinely read
differently. Any missing data renders as "unavailable", never invented.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TypeVar

import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_DIR = __file__.rsplit("/", 1)[0]
STATIC_DIR = f"{APP_DIR}/static"

UA = "trading-analyst-selfhost/1.0"


def _yf_session():
    """Build a shared curl_cffi session that honors proxy env vars.

    yfinance's default curl_cffi Session sets proxies={} explicitly, which
    overrides the environment — that breaks hosts that require an egress
    proxy (sandboxes, corporate networks). If YF_PROXY (or HTTPS_PROXY /
    https_proxy) is set, use it; otherwise return None for default behavior.
    """
    proxy = (os.environ.get("YF_PROXY") or os.environ.get("HTTPS_PROXY")
             or os.environ.get("https_proxy"))
    if not proxy:
        return None
    from curl_cffi import requests as creq
    s = creq.Session(impersonate="chrome")
    s.proxies = {"http": proxy, "https": proxy}
    return s


_YF_SESSION = _yf_session()


def _ticker(sym: str):
    return yf.Ticker(sym, session=_YF_SESSION) if _YF_SESSION is not None else yf.Ticker(sym)

# ---------------------------------------------------------------------------
# Small numeric helpers (pure python, no pandas needed)
# ---------------------------------------------------------------------------

def ema(values: list[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    k = 2 / (n + 1)
    e = sum(values[:n]) / n
    for v in values[n:]:
        e = v * k + e * (1 - k)
    return e


def sma(values: list[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def rsi(closes: list[float], n: int = 14) -> Optional[float]:
    if len(closes) < n + 1:
        return None
    gains, losses = [], []
    for i in range(1, n + 1):
        ch = closes[-i] - closes[-i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    avg_g = sum(gains) / n
    avg_l = sum(losses) / n
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100 - 100 / (1 + rs)


def atr(highs: list[float], lows: list[float], closes: list[float], n: int = 14) -> Optional[float]:
    if len(closes) < n + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    return sum(trs[-n:]) / n


def pct_change(values: list[float], n: int) -> Optional[float]:
    if len(values) < n + 1:
        return None
    base = values[-(n + 1)]
    if base == 0:
        return None
    return (values[-1] - base) / base * 100


def median(values: list[float]) -> Optional[float]:
    vs = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vs:
        return None
    m = len(vs) // 2
    return vs[m] if len(vs) % 2 else (vs[m - 1] + vs[m]) / 2


def f2(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:,.2f}"


def fpct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:+.1f}%"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# yfinance data layer — every fetch is failure-tolerant
# ---------------------------------------------------------------------------

T = TypeVar("T")


def _parallel(**jobs: Callable[[], T]) -> dict[str, Optional[T]]:
    """Run independent fetch jobs concurrently; a failed job yields None."""
    out: dict[str, Optional[T]] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
        futs = {k: ex.submit(fn) for k, fn in jobs.items()}
        for k, f in futs.items():
            try:
                out[k] = f.result()
            except Exception:
                out[k] = None
    return out


def _finite(x: Any) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def daily_history(ticker: str, period: str = "9mo", interval: str = "1d") -> Optional[dict[str, Any]]:
    """Return aligned OHLCV data as plain lists, or None on failure."""
    try:
        source_interval = "1h" if interval == "4h" else interval
        df = _ticker(ticker).history(period=period, interval=source_interval, auto_adjust=False)
        if df is None or df.empty:
            return None
        # Drop incomplete rows so chart candles and indicators stay aligned.
        rows = [(o, h, l, c, v, str(date)) for date, o, h, l, c, v in zip(
            df.index, df["Open"].tolist(), df["High"].tolist(),
            df["Low"].tolist(), df["Close"].tolist(), df["Volume"].tolist())
            if all(_finite(value) is not None for value in (o, h, l, c, v))]
        if interval == "4h":
            grouped = []
            days = {}
            for row in rows:
                days.setdefault(row[-1].split(" ")[0], []).append(row)
            for day_rows in days.values():
                for start in range(0, len(day_rows), 4):
                    chunk = day_rows[start:start + 4]
                    if len(chunk) < 4:
                        continue
                    grouped.append((chunk[0][0], max(row[1] for row in chunk),
                                    min(row[2] for row in chunk), chunk[-1][3],
                                    sum(row[4] for row in chunk), chunk[-1][5]))
            rows = grouped
        if len(rows) < 30:
            return None
        closes = [float(c) for _, _, _, c, _, _ in rows]
        return {
            "closes": closes,
            "opens": [float(o) for o, _, _, _, _, _ in rows],
            "highs": [float(h) for _, h, _, _, _, _ in rows],
            "lows": [float(l) for _, _, l, _, _, _ in rows],
            "volumes": [float(v) for _, _, _, _, v, _ in rows],
            "dates": [date for _, _, _, _, _, date in rows],
            "last_date": rows[-1][-1],
        }
    except Exception:
        return None


def quote_info(ticker: str) -> dict[str, Any]:
    """Fast quote + info fields; missing fields stay None."""
    out: dict[str, Any] = {"price": None, "name": None, "sector": None,
                           "industry": None, "currency": None}
    t = _ticker(ticker)  # one Ticker object reused for every lookup below
    try:
        fi = t.fast_info
        out["price"] = _finite(fi.last_price)
    except Exception:
        pass
    try:
        info = t.info or {}
        out["name"] = info.get("longName") or info.get("shortName")
        out["sector"] = info.get("sector")
        out["industry"] = info.get("industry")
        out["currency"] = info.get("currency")
        for k in ("trailingPE", "forwardPE", "pegRatio", "priceToBook",
                  "marketCap", "trailingEps", "forwardEps",
                  "earningsGrowth", "revenueGrowth", "dividendYield",
                  "fiftyTwoWeekHigh", "fiftyTwoWeekLow"):
            out[k] = _finite(info.get(k))
        # earnings date, if published
        try:
            cal = t.calendar
            if cal is not None and not cal.empty and "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"].iloc[0]
                out["earnings_date"] = str(ed)
            else:
                out["earnings_date"] = None
        except Exception:
            out["earnings_date"] = None
        # recent news headlines, if the feed returns any
        try:
            news = t.news or []
            out["news"] = [
                {"title": n.get("title"), "publisher": n.get("publisher"),
                 "link": (n.get("link") or "")[:200]}
                for n in news[:5] if n.get("title")
            ]
        except Exception:
            out["news"] = []
    except Exception:
        pass
    return out


# Sector -> representative ETF for a rough sector-trend read.
SECTOR_ETFS = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Basic Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}


# ---------------------------------------------------------------------------
# Analysis building blocks — every sentence below is data-composed
# ---------------------------------------------------------------------------

def build_regime() -> dict[str, Any]:
    """Market regime gate: SPY/QQQ 20-session trend, VIX vs 90d median, 10Y."""
    h = _parallel(
        spy=lambda: daily_history("SPY", "6mo"),
        qqq=lambda: daily_history("QQQ", "6mo"),
        vix=lambda: daily_history("^VIX", "6mo"),
        tnx=lambda: daily_history("^TNX", "6mo"),
    )
    spy, qqq, vix, tnx = h["spy"], h["qqq"], h["vix"], h["tnx"]

    spy20 = pct_change(spy["closes"], 20) if spy else None
    qqq20 = pct_change(qqq["closes"], 20) if qqq else None
    vix_now = vix["closes"][-1] if vix else None
    vix_med = median(vix["closes"][-90:]) if vix else None
    # ^TNX quotes in hundredths of a percent (e.g. 415 = 4.15%)
    tnx_now = (tnx["closes"][-1] / 100) if tnx and tnx["closes"][-1] > 50 else (tnx["closes"][-1] if tnx else None)
    if tnx:
        raw = [c / 100 if c > 50 else c for c in tnx["closes"][-90:]]
        tnx_med = median(raw)
    else:
        tnx_med = None

    votes, notes = [], []
    if spy20 is not None:
        if spy20 > 0:
            votes.append("on"); notes.append(f"SPY {fpct(spy20)} over 20 sessions (trend up)")
        else:
            votes.append("off"); notes.append(f"SPY {fpct(spy20)} over 20 sessions (trend down)")
    else:
        notes.append("SPY 20-session trend unavailable")
    if qqq20 is not None:
        if qqq20 > 0:
            votes.append("on"); notes.append(f"QQQ {fpct(qqq20)} over 20 sessions (trend up)")
        else:
            votes.append("off"); notes.append(f"QQQ {fpct(qqq20)} over 20 sessions (trend down)")
    else:
        notes.append("QQQ 20-session trend unavailable")
    if vix_now is not None and vix_med is not None:
        if vix_now <= vix_med:
            votes.append("on"); notes.append(f"VIX {vix_now:.2f} at/below 90-day median {vix_med:.2f} (fear contained)")
        else:
            votes.append("off"); notes.append(f"VIX {vix_now:.2f} above 90-day median {vix_med:.2f} (fear elevated)")
    else:
        notes.append("VIX read unavailable")
    if tnx_now is not None and tnx_med is not None:
        notes.append(f"10Y yield {tnx_now:.2f}% vs 90-day median {tnx_med:.2f}%")

    if not votes:
        label = "unknown"
    elif votes.count("on") >= votes.count("off") and votes.count("on") >= 2:
        label = "risk-on"
    elif votes.count("off") >= 2:
        label = "risk-off"
    else:
        label = "mixed"

    # Composed, never canned: the numbers above are this run's numbers.
    summary = (
        f"Regime read ({now_iso()}): " + "; ".join(notes)
        + f". Net signal: {votes.count('on')} risk-on votes vs {votes.count('off')} risk-off votes "
        + f"→ classified {label}."
    )
    return {"label": label, "summary": summary, "notes": notes,
            "spy_20d_pct": spy20, "qqq_20d_pct": qqq20,
            "vix": vix_now, "vix_median90": vix_med,
            "tnx_pct": tnx_now, "tnx_median90": tnx_med, "as_of": now_iso()}


def build_technicals(ticker: str, hist: dict[str, list[float]]) -> dict[str, Any]:
    closes, highs, lows, vols = hist["closes"], hist["highs"], hist["lows"], hist["volumes"]
    price = closes[-1]
    ema20, ema50 = ema(closes, 20), ema(closes, 50)
    sma20, sma50 = sma(closes, 20), sma(closes, 50)
    r = rsi(closes)
    a = atr(highs, lows, closes)
    vol = vols[-1]
    vol_avg = sma(vols, 20)
    vol_ratio = vol / vol_avg if vol_avg else None
    hi52, lo52 = max(highs[-252:]), min(lows[-252:])
    pos52 = (price - lo52) / (hi52 - lo52) * 100 if hi52 > lo52 else None
    sup = min(lows[-20:])   # recent 20-session low, labeled as such
    res = max(highs[-20:])  # recent 20-session high, labeled as such
    chg20 = pct_change(closes, 20)

    stack_bits = []
    if price is not None and ema20: stack_bits.append(f"price {f2(price)} {'above' if price > ema20 else 'below'} 20-day EMA {f2(ema20)}")
    if ema20 and ema50: stack_bits.append(f"20-day EMA {'above' if ema20 > ema50 else 'below'} 50-day EMA {f2(ema50)}")
    stack = "bullish stack" if (price and ema20 and ema50 and price > ema20 > ema50) else \
            "bearish stack" if (price and ema20 and ema50 and price < ema20 < ema50) else "mixed stack"

    if r is None: rsi_read = "RSI unavailable"
    elif r >= 70: rsi_read = f"RSI {r:.0f} — overbought zone (≥70), momentum stretched"
    elif r <= 30: rsi_read = f"RSI {r:.0f} — oversold zone (≤30), washout risk"
    else: rsi_read = f"RSI {r:.0f} — neutral band"

    vol_read = (f"volume {vol_ratio:.1f}× its 20-day average — conviction behind the move"
                if vol_ratio and vol_ratio >= 1.5 else
                f"volume {vol_ratio:.1f}× its 20-day average — no unusual participation"
                if vol_ratio else "volume read unavailable")

    summary = (
        f"{ticker} technicals ({hist['last_date']}): {', '.join(stack_bits) if stack_bits else 'EMA stack unavailable'} "
        f"→ {stack}. {rsi_read}. {vol_read}. "
        f"ATR(14) ${f2(a)} ({(a/price*100):.1f}% of price) — daily swing budget. "
        f"Price sits {pos52:.0f}% up its 52-week range (${f2(lo52)}–${f2(hi52)}). "
        f"Recent 20-session support ≈ ${f2(sup)}, resistance ≈ ${f2(res)}; "
        f"20-session move {fpct(chg20)}."
    )
    return {"summary": summary, "price": price, "ema20": ema20, "ema50": ema50,
            "sma20": sma20, "sma50": sma50, "rsi": r, "atr": a,
            "vol_ratio": vol_ratio, "pos52": pos52, "support20": sup,
            "resistance20": res, "chg20": chg20, "stack": stack,
            "as_of": hist["last_date"]}


def build_valuation(ticker: str, info: dict[str, Any], tech: dict[str, Any]) -> dict[str, Any]:
    pe_t, pe_f = info.get("trailingPE"), info.get("forwardPE")
    pb, peg = info.get("priceToBook"), info.get("pegRatio")
    mcap = info.get("marketCap")
    lines = []
    if pe_t: lines.append(f"trailing P/E {pe_t:.1f}×")
    if pe_f: lines.append(f"forward P/E {pe_f:.1f}×")
    if pe_t and pe_f and pe_f > 0:
        g = (pe_t - pe_f) / pe_f * 100
        lines.append(f"market prices roughly {g:.0f}% earnings growth (trailing vs forward P/E)")
    if pb: lines.append(f"P/B {pb:.1f}×")
    if peg: lines.append(f"PEG {peg:.2f}")
    if mcap: lines.append(f"market cap ${mcap/1e9:,.1f}B")
    pos52 = tech.get("pos52")
    if pos52 is not None:
        lines.append(f"price {pos52:.0f}% up its 52-week range — valuation context, not a signal by itself")

    if lines:
        summary = f"{ticker} valuation snapshot: " + "; ".join(lines) + ". " \
            "Peer-relative multiples are not available from the free quote feed, so no sector-relative claim is made."
    else:
        summary = (f"{ticker} valuation: multiples unavailable from the quote feed right now "
                   f"(no trailing/forward P/E returned) — refusing to guess; treat any valuation read as incomplete.")
    return {"summary": summary, "trailing_pe": pe_t, "forward_pe": pe_f,
            "peg": peg, "price_to_book": pb, "market_cap": mcap, "as_of": now_iso()}


def build_catalysts(ticker: str, info: dict[str, Any]) -> dict[str, Any]:
    ed = info.get("earnings_date")
    news = info.get("news") or []
    parts = []
    if ed:
        parts.append(f"next earnings date on file: {ed} (verify with the company IR calendar before trading around it)")
    else:
        parts.append("earnings date unavailable from the feed — check the company IR calendar before holding through an announcement")
    if news:
        heads = "; ".join(f"“{n['title']}” ({n['publisher']})" for n in news)
        parts.append(f"recent headlines: {heads}")
    else:
        parts.append("no recent headlines returned by the feed")
    sector = info.get("sector") or "sector unknown"
    summary = f"{ticker} catalysts ({sector}): " + ". ".join(parts) + "."
    return {"summary": summary, "earnings_date": ed, "news": news, "as_of": now_iso()}


# ---------------------------------------------------------------------------
# Mode-specific evaluation — each mode composes its own verdict text
# ---------------------------------------------------------------------------

def evaluate_entry(ticker: str, tech: dict[str, Any], regime: dict[str, Any]) -> dict[str, Any]:
    price, ema20, ema50 = tech["price"], tech["ema20"], tech["ema50"]
    r, vol_ratio = tech["rsi"], tech["vol_ratio"]
    sup, res = tech["support20"], tech["resistance20"]
    checks, positives, negatives = [], [], []

    if price and ema20 and ema50:
        if price > ema20 > ema50:
            positives.append(f"price ${f2(price)} holds above a rising EMA stack (20d ${f2(ema20)} > 50d ${f2(ema50)})")
        elif price < ema20 < ema50:
            negatives.append(f"price ${f2(price)} sits under a falling EMA stack — buying here is catching a downtrend")
        else:
            checks.append(f"EMA stack mixed (price ${f2(price)}, 20d ${f2(ema20)}, 50d ${f2(ema50)}) — no clean trend to join")
    else:
        checks.append("EMA stack incomplete — entry read degraded")

    if r is not None:
        if r >= 70: negatives.append(f"RSI {r:.0f} overbought — chasing risks buying the top of the swing")
        elif r <= 30: positives.append(f"RSI {r:.0f} oversold — weak hands may be flushed, bounce candidate if trend intact")
        else: checks.append(f"RSI {r:.0f} neutral — momentum neither stretched nor washed out")
    if vol_ratio is not None:
        if vol_ratio >= 1.5: positives.append(f"volume {vol_ratio:.1f}× average confirms participation")
        else: checks.append(f"volume {vol_ratio:.1f}× average — thin conviction")

    if sup and price and price - sup > 0:
        checks.append(f"nearest recent support ${f2(sup)} is {((price-sup)/price*100):.1f}% below — natural stop/trigger zone")
    if res and price and res - price > 0:
        checks.append(f"nearest recent resistance ${f2(res)} is {((res-price)/price*100):.1f}% above — breakout trigger lives there")

    if regime["label"] == "risk-off":
        negatives.append(f"market regime is {regime['label']} — headwind for any new long")
    elif regime["label"] == "risk-on":
        positives.append("market regime risk-on — tailwind for new longs")

    if negatives and not positives:
        verdict, why = "NO-GO", "negatives dominate"
    elif positives and not negatives:
        verdict, why = "GO", "positives dominate"
    else:
        verdict, why = "CAUTION", "evidence conflicts"

    detail = " Positives: " + "; ".join(positives) + "." if positives else ""
    detail += " Negatives: " + "; ".join(negatives) + "." if negatives else ""
    detail += " Watch: " + "; ".join(checks) + "." if checks else ""
    summary = (f"{ticker} entry read ({tech['as_of']}): verdict {verdict} because {why}.{detail} "
               f"This is a trigger-based read, not a prediction — the levels above are where the trade gets confirmed or killed.")
    return {"verdict": verdict, "summary": summary, "positives": positives,
            "negatives": negatives, "checks": checks}


class CspTerms(BaseModel):
    strike: float
    credit: float          # premium received per share
    delta: Optional[float] = None   # e.g. 0.22
    iv: Optional[float] = None      # e.g. 0.45 for 45%
    dte: int = 30
    contracts: int = 1


class AskRequest(BaseModel):
    ticker: str
    question: str
    period: str = "3mo"
    interval: str = "4h"


def answer_chart_question(request: AskRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    chart = chart_payload(request.ticker, request.period, request.interval)
    candles = chart["candles"]
    latest = candles[-1]
    index = len(candles) - 1
    levels = chart["levels"]
    phase = chart["phases"][index]
    ema20 = chart["ema20"][index]
    ema50 = chart["ema50"][index]
    cmf = chart["cmf20"][index]
    ad_line = chart["ad_line"][index]
    lower = question.lower()

    def money(value: Optional[float]) -> str:
        return "unavailable" if value is None else f"${value:,.2f}"

    if "phase" in lower or any(word in lower for word in ("accumulation", "distribution", "markup", "markdown")):
        answer = (f"The latest {chart['interval']} candle ({latest['dates']}) is classified as {phase}. "
                  f"The classification uses EMA trend structure, 20-bar price change, A/D direction, and CMF.")
    elif "entry" in lower or "buy" in lower:
        answer = (f"The chart's best pullback entry is the primary support retest at {money(levels['best_entry'])}. "
                  f"The breakout confirmation level is {money(levels['entry_trigger'])}. "
                  "These are technical reference levels, not a guarantee or personalized advice.")
    elif "support" in lower or "resistance" in lower or "level" in lower:
        answer = (f"Primary support is {money(levels['support'])} and primary resistance is {money(levels['resistance'])}. "
                  f"Secondary support is {money(levels['secondary_support'])}; secondary resistance is {money(levels['secondary_resistance'])}.")
    elif "rsi" in lower or "overbought" in lower or "oversold" in lower:
        rsi_value = chart["rsi"]
        answer = f"RSI is {rsi_value:.1f}. It is {'oversold' if rsi_value <= 30 else 'overbought' if rsi_value >= 70 else 'in the neutral band'} by the 30/70 screen." if rsi_value is not None else "RSI is unavailable for this view."
    elif "cmf" in lower or "money flow" in lower or "buying pressure" in lower or "selling pressure" in lower:
        answer = f"CMF20 is {cmf:.3f} and the latest phase is {phase}; this reads as {'buying' if cmf is not None and cmf > 0 else 'selling' if cmf is not None else 'unavailable'} pressure. A/D is {ad_line:,.0f}."
    elif "ema" in lower or "trend" in lower or "direction" in lower:
        relation = "above" if ema20 is not None and latest["closes"] > ema20 else "below"
        stack = "bullish" if ema20 is not None and ema50 is not None and latest["closes"] > ema20 > ema50 else "bearish" if ema20 is not None and ema50 is not None and latest["closes"] < ema20 < ema50 else "mixed"
        answer = f"The close is {relation} EMA20 ({money(ema20)}); EMA50 is {money(ema50)}. The current EMA structure is {stack}."
    elif "candle" in lower or "open" in lower or "close" in lower or "volume" in lower:
        answer = (f"The latest candle is {latest['dates']}: open {money(latest['opens'])}, high {money(latest['highs'])}, "
                  f"low {money(latest['lows'])}, close {money(latest['closes'])}, volume {latest['volumes']:,.0f}.")
    else:
        answer = (f"{chart['ticker']} is at {money(latest['closes'])} on the latest {chart['interval']} candle, "
                  f"classified as {phase}. Best pullback entry/support is {money(levels['best_entry'])}; "
                  f"resistance/trigger is {money(levels['entry_trigger'])}. Ask about phase, entry, levels, RSI, trend, CMF, or the latest candle for a focused answer.")
    return {"ticker": chart["ticker"], "question": question, "answer": answer, "as_of": chart["as_of"]}


def evaluate_csp(ticker: str, tech: dict[str, Any], regime: dict[str, Any], t: CspTerms) -> dict[str, Any]:
    spot = tech["price"]
    if spot is None or spot <= 0:
        return {"verdict": "NO-GO",
                "summary": f"{ticker} CSP: cannot evaluate — spot price unavailable, and the contract math needs a real spot. No invented numbers.",
                "metrics": {}}
    breakeven = t.strike - t.credit
    buffer_pct = (spot - breakeven) / spot * 100
    capital_at_risk = breakeven * 100 * t.contracts
    premium_total = t.credit * 100 * t.contracts
    roc = premium_total / capital_at_risk * 100 if capital_at_risk > 0 else None
    roc_ann = roc * (365 / t.dte) if roc is not None and t.dte > 0 else None

    positives, negatives, checks = [], [], []
    if t.strike >= spot:
        negatives.append(f"strike ${f2(t.strike)} is at/above spot ${f2(spot)} — this is not a cash-secured put discount, assignment starts underwater")
    else:
        positives.append(f"strike ${f2(t.strike)} is {((spot-t.strike)/spot*100):.1f}% below spot — genuine discount entry")

    checks.append(f"breakeven ${f2(breakeven)} sits {buffer_pct:.1f}% below spot — that is the full downside cushion")
    sup = tech.get("support20")
    if sup:
        if breakeven < sup:
            positives.append(f"breakeven ${f2(breakeven)} is under recent 20-session support ${f2(sup)} — structure backs the short put")
        else:
            negatives.append(f"breakeven ${f2(breakeven)} is above recent support ${f2(sup)} — a retest of support puts the put in the money")

    if t.delta is not None:
        d = abs(t.delta)
        if d <= 0.30:
            positives.append(f"delta {t.delta:.2f} passes the ≤0.30 screen (≈{d*100:.0f}% modeled assignment odds)")
        else:
            negatives.append(f"delta {t.delta:.2f} fails the ≤0.30 screen (≈{d*100:.0f}% modeled assignment odds — too hot for a CSP)")
    else:
        checks.append("no delta supplied — cannot screen assignment odds; get it from the broker chain")
    if t.iv is not None:
        checks.append(f"IV {t.iv*100:.0f}% supplied — richer IV means richer premium for the same strike")
    else:
        checks.append("no IV supplied — premium richness cannot be judged")

    if roc is not None:
        checks.append(f"max return {roc:.2f}% on ${capital_at_risk:,.0f} at risk over {t.dte} days"
                      + (f" ({roc_ann:.1f}% annualized)" if roc_ann else ""))
    if regime["label"] == "risk-off":
        negatives.append("market regime risk-off — short premium into a fearful tape demands wider margin")

    if negatives and not positives:
        verdict = "NO-GO"
    elif positives and not negatives:
        verdict = "GO"
    else:
        verdict = "CAUTION"

    detail = (" Positives: " + "; ".join(positives) + ".") if positives else ""
    detail += (" Negatives: " + "; ".join(negatives) + ".") if negatives else ""
    detail += (" Notes: " + "; ".join(checks) + ".") if checks else ""
    summary = (f"{ticker} cash-secured put — SELL {t.contracts}× ${t.strike:.0f}P, ${t.credit:.2f} credit, {t.dte} DTE, "
               f"spot ${f2(spot)}: verdict {verdict}.{detail} "
               f"Contract-specific math only — no options chain was invented; confirm these exact terms on the broker chain.")
    return {"verdict": verdict, "summary": summary, "positives": positives,
            "negatives": negatives, "checks": checks,
            "metrics": {"spot": spot, "breakeven": round(breakeven, 2),
                        "downside_buffer_pct": round(buffer_pct, 2),
                        "capital_at_risk": round(capital_at_risk, 2),
                        "premium_total": round(premium_total, 2),
                        "return_on_risk_pct": round(roc, 2) if roc else None,
                        "annualized_pct": round(roc_ann, 1) if roc_ann else None}}


def risk_veto(ticker: str, mode: str, tech: dict[str, Any], regime: dict[str, Any],
              mode_eval: dict[str, Any], csp_terms: Optional[CspTerms]) -> dict[str, Any]:
    blockers, warnings = [], []

    if tech.get("price") is None:
        blockers.append(f"{ticker}: no quote data returned — the risk manager vetoes any verdict built on a missing price")
    if regime["label"] == "unknown":
        warnings.append("regime unreadable (all gate inputs failed) — size any trade as if the tape is hostile")
    if regime["label"] == "risk-off":
        warnings.append(f"regime {regime['label']}: {'; '.join(regime['notes'][:2])} — new risk needs a wider margin of safety")
    a, price = tech.get("atr"), tech.get("price")
    if a and price and a / price > 0.05:
        warnings.append(f"ATR ${f2(a)} is {(a/price*100):.1f}% of price — single-day swings can erase a tight stop")
    if mode == "csp" and csp_terms and price and csp_terms.strike >= price:
        blockers.append(f"CSP strike ${f2(csp_terms.strike)} ≥ spot ${f2(price)} — uncapped-downside shape on a put sold at-the-money; vetoed")
    r = tech.get("rsi")
    if r is not None and r >= 75:
        warnings.append(f"RSI {r:.0f} extremely stretched — late entries get punished first on a reversal")

    if blockers:
        final, why = "NO-GO", "hard blocker(s) fired"
    elif mode_eval["verdict"] == "NO-GO":
        final, why = "NO-GO", f"{mode} evaluation itself concluded NO-GO"
    elif warnings or mode_eval["verdict"] == "CAUTION":
        final = "CAUTION"
        why = "warnings present" if warnings else f"{mode} evaluation concluded CAUTION"
    else:
        final, why = "GO", "no blockers, no warnings, evaluation clean"

    # The veto narrative is assembled from what actually fired — never a fixed line.
    fired = [f"BLOCKER: {b}" for b in blockers] + [f"warning: {w}" for w in warnings]
    summary = (f"Risk manager on {ticker} ({mode} mode): {len(blockers)} blocker(s), {len(warnings)} warning(s). "
               + (" ".join(fired) + " " if fired else "Nothing fired. ")
               + f"Final verdict: {final} — {why}.")
    return {"final_verdict": final, "summary": summary,
            "blockers": blockers, "warnings": warnings}


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def analyze(ticker: str, mode: str, csp_terms: Optional[CspTerms]) -> dict[str, Any]:
    ticker = ticker.strip().upper()
    if not ticker or not ticker.replace(".", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid ticker")
    if mode not in ("stock", "entry", "csp"):
        raise HTTPException(status_code=400, detail="mode must be stock, entry, or csp")
    if mode == "csp" and csp_terms is None:
        raise HTTPException(status_code=400, detail="csp mode requires strike, credit, dte, contracts")

    fetched = _parallel(
        regime=build_regime,
        hist=lambda: daily_history(ticker),
        info=lambda: quote_info(ticker),
    )
    regime: dict[str, Any] = fetched["regime"] or {"label": "unknown", "summary": "regime fetch failed", "notes": [], "as_of": now_iso()}
    hist = fetched["hist"]
    info: dict[str, Any] = fetched["info"] or {}

    if hist is None:
        # Honest degradation: no invented price, verdict forced NO-GO.
        tech = {"summary": f"{ticker} technicals: price history unavailable from the feed — no indicators computed, nothing guessed.",
                "price": None, "as_of": now_iso()}
        val = {"summary": f"{ticker} valuation: unavailable — no quote data to anchor it.", "as_of": now_iso()}
        cat = {"summary": f"{ticker} catalysts: unavailable — no quote data.", "as_of": now_iso()}
        mode_eval = {"verdict": "NO-GO",
                     "summary": f"{ticker} ({mode} mode): NO-GO — the feed returned no price history, so there is no analysis to give."}
    else:
        tech = build_technicals(ticker, hist)
        val = build_valuation(ticker, info, tech)
        cat = build_catalysts(ticker, info)
        if mode == "entry":
            mode_eval = evaluate_entry(ticker, tech, regime)
        elif mode == "csp":
            mode_eval = evaluate_csp(ticker, tech, regime, csp_terms)  # type: ignore[arg-type]
        else:
            # stock mode: read the tape without an entry trigger
            stack, r = tech["stack"], tech["rsi"]
            bits = [f"trend structure reads {stack}", f"RSI {r:.0f}" if r is not None else "RSI unavailable"]
            if regime["label"] == "risk-on": bits.append("market tailwind (risk-on)")
            elif regime["label"] == "risk-off": bits.append("market headwind (risk-off)")
            mode_eval = {"verdict": "CAUTION",
                         "summary": (f"{ticker} stock read ({tech['as_of']}): " + ", ".join(bits)
                                     + ". This is a situational read, not an entry call — switch to entry mode for trigger levels.")}

    risk = risk_veto(ticker, mode, tech, regime, mode_eval, csp_terms)

    # Verdict headline is composed from this run's facts — the anti-canned-text guarantee.
    price_txt = f"${tech['price']:,.2f}" if tech.get("price") else "price unavailable"
    rsi_txt = f"RSI {tech['rsi']:.0f}" if tech.get("rsi") is not None else "RSI n/a"
    headline = (f"{ticker} [{mode}] → {risk['final_verdict']}: {price_txt}, {rsi_txt}, "
                f"regime {regime['label']}, {len(risk['blockers'])} blocker(s)/{len(risk['warnings'])} warning(s)")

    return {
        "ticker": ticker, "mode": mode, "as_of": now_iso(),
        "company": info.get("name"), "sector": info.get("sector"),
        "verdict": risk["final_verdict"], "verdict_headline": headline,
        "regime": regime, "technicals": tech, "valuation": val,
        "catalysts": cat, "mode_evaluation": mode_eval, "risk": risk,
        "csp_terms": csp_terms.model_dump() if csp_terms else None,
        "data_note": "All figures from yfinance free quote feed; timestamps are UTC. Missing fields render as unavailable — never estimated.",
    }


def indicator_series(values: list[float], window: int) -> list[Optional[float]]:
    """Return an EMA for every point where its lookback is available."""
    if len(values) < window:
        return [None] * len(values)
    k = 2 / (window + 1)
    result: list[Optional[float]] = [None] * (window - 1)
    current = sum(values[:window]) / window
    result.append(current)
    for value in values[window:]:
        current = value * k + current * (1 - k)
        result.append(current)
    return result


def accumulation_distribution(hist: dict[str, Any]) -> list[float]:
    """Return the cumulative Chaikin Accumulation/Distribution Line."""
    total = 0.0
    result = []
    for high, low, close, volume in zip(hist["highs"], hist["lows"], hist["closes"], hist["volumes"]):
        spread = high - low
        multiplier = 0.0 if spread == 0 else ((close - low) - (high - close)) / spread
        total += multiplier * volume
        result.append(total)
    return result


def chaikin_money_flow(hist: dict[str, Any], window: int = 20) -> list[Optional[float]]:
    """Return CMF, measuring buying/selling pressure over a rolling window."""
    flow = []
    for high, low, close, volume in zip(hist["highs"], hist["lows"], hist["closes"], hist["volumes"]):
        spread = high - low
        multiplier = 0.0 if spread == 0 else ((close - low) - (high - close)) / spread
        flow.append((multiplier * volume, volume))
    result: list[Optional[float]] = [None] * len(flow)
    for index in range(window - 1, len(flow)):
        money_flow = sum(item[0] for item in flow[index - window + 1:index + 1])
        volume = sum(item[1] for item in flow[index - window + 1:index + 1])
        result[index] = money_flow / volume if volume else None
    return result


def market_phases(closes: list[float], ema20: list[Optional[float]],
                  ema50: list[Optional[float]], ad_line: list[float],
                  cmf20: list[Optional[float]]) -> list[str]:
    """Classify each bar from trend structure and volume-flow confirmation."""
    phases = []
    for index, close in enumerate(closes):
        if index < 20 or ema20[index] is None or ema50[index] is None or cmf20[index] is None:
            phases.append("Insufficient data")
            continue
        price_change = close - closes[index - 20]
        ad_change = ad_line[index] - ad_line[max(0, index - 5)]
        trending_up = close > ema20[index] > ema50[index] and price_change > 0
        trending_down = close < ema20[index] < ema50[index] and price_change < 0
        if trending_up:
            phases.append("Markup")
        elif trending_down:
            phases.append("Markdown")
        elif ad_change > 0 and cmf20[index] > 0:
            phases.append("Accumulation")
        elif ad_change < 0 and cmf20[index] < 0:
            phases.append("Distribution")
        else:
            phases.append("Transition")
    return phases


def chart_payload(ticker: str, period: str = "3mo", interval: str = "4h") -> dict[str, Any]:
    ticker = ticker.strip().upper()
    if not ticker or not ticker.replace(".", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid ticker")
    requested_period = period
    view_days = {"1d": 1, "5d": 5, "7d": 7, "10d": 10, "1mo": 22,
                 "3mo": 66, "6mo": 132, "9mo": 198, "1y": 252, "2y": 520}
    bars_per_day = {"1m": 390, "5m": 78, "15m": 26, "30m": 13, "1h": 7, "4h": 2, "1d": 1}[interval]
    view_bars = min(3000, view_days[period] * bars_per_day)
    if interval == "1m":
        source_period = "7d"
    elif interval in ("5m", "15m", "30m"):
        source_period = "60d"
    elif interval in ("1h", "4h"):
        source_period = "1y"
    elif period in ("1d", "5d", "10d", "1mo"):
        source_period = "3mo"
    else:
        source_period = period
    hist = daily_history(ticker, source_period, interval)
    if hist is None:
        raise HTTPException(status_code=404, detail="Price history unavailable")
    closes = hist["closes"]
    candles = [
        {key: hist[key][index] for key in ("dates", "opens", "highs", "lows", "closes", "volumes")}
        for index in range(len(closes))
    ]
    # Keep enough history for the selected view and indicator lookbacks.
    history_bars = min(len(closes), max(520, view_bars))
    candles = candles[-history_bars:]
    close_window = closes[-history_bars:]
    ema20 = indicator_series(close_window, 20)
    ema50 = indicator_series(close_window, 50)
    ad_line = accumulation_distribution(hist)[-history_bars:]
    cmf20 = chaikin_money_flow(hist)[-history_bars:]
    phases = market_phases(close_window, ema20, ema50, ad_line, cmf20)
    support = min(hist["lows"][-20:])
    resistance = max(hist["highs"][-20:])
    prior_lows = hist["lows"][-60:-20]
    prior_highs = hist["highs"][-60:-20]
    secondary_support = min(prior_lows) if prior_lows else None
    secondary_resistance = max(prior_highs) if prior_highs else None
    rsi_value = rsi(closes)
    latest = closes[-1]
    trend = "above" if ema20[-1] is not None and latest > ema20[-1] else "below"
    cmf_value = cmf20[-1]
    flow = "buying pressure" if cmf_value is not None and cmf_value > 0 else "selling pressure"
    insight = (f"{ticker} is trading {trend} its 20-day EMA; RSI is "
               f"{rsi_value:.0f}. Candles and volume use {interval} Yahoo Finance data." 
               if rsi_value is not None else
               f"{ticker} is trading {trend} its 20-day EMA. RSI is unavailable from the current {interval} history.")
    insight += f" CMF20 indicates {flow}."
    if view_bars < len(candles):
        candles = candles[-view_bars:]
        ema20 = ema20[-view_bars:]
        ema50 = ema50[-view_bars:]
        ad_line = ad_line[-view_bars:]
        cmf20 = cmf20[-view_bars:]
        phases = phases[-view_bars:]
    levels = {"buy_entry": support, "sell_entry": resistance,
              "best_entry": support, "entry_trigger": resistance,
              "support": support, "resistance": resistance,
              "secondary_support": secondary_support,
              "secondary_resistance": secondary_resistance,
              "method": "Best entry is the primary support retest; entry trigger is a breakout above primary resistance."}
    return {"ticker": ticker, "period": requested_period, "source_period": source_period, "interval": interval,
            "candles": candles, "ema20": ema20, "ema50": ema50,
            "ad_line": ad_line, "cmf20": cmf20, "rsi": rsi_value,
            "phases": phases, "levels": levels, "insight": insight, "as_of": now_iso()}


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

app = FastAPI(title="Trading Analyst Agent (self-hosted)")


def _json_safe(x: Any) -> Any:
    """Final guarantee: no NaN/Inf may reach the JSON encoder."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


@app.get("/api/analyze")
def api_analyze(
    ticker: str = Query(..., description="Ticker symbol, e.g. ADBE"),
    mode: str = Query("stock", description="stock | entry | csp"),
    strike: Optional[float] = Query(None),
    credit: Optional[float] = Query(None),
    delta: Optional[float] = Query(None),
    iv: Optional[float] = Query(None),
    dte: int = Query(30),
    contracts: int = Query(1),
):
    csp = None
    if mode == "csp":
        if strike is None or credit is None:
            raise HTTPException(status_code=400, detail="csp mode requires strike and credit")
        csp = CspTerms(strike=strike, credit=credit, delta=delta, iv=iv, dte=dte, contracts=contracts)
    return _json_safe(analyze(ticker, mode, csp))


@app.get("/api/chart")
def api_chart(ticker: str = Query(...), period: str = Query("3mo"), interval: str = Query("4h")):
    if period not in ("1d", "5d", "7d", "10d", "1mo", "3mo", "6mo", "9mo", "1y", "2y"):
        raise HTTPException(status_code=400, detail="invalid chart period")
    if interval not in ("1m", "5m", "15m", "30m", "1h", "4h", "1d"):
        raise HTTPException(status_code=400, detail="interval must be 1m, 5m, 15m, 30m, 1h, 4h, or 1d")
    return _json_safe(chart_payload(ticker, period, interval))


@app.post("/api/ask")
def api_ask(request: AskRequest):
    if request.period not in ("1d", "5d", "7d", "10d", "1mo", "3mo", "6mo", "9mo", "1y", "2y"):
        raise HTTPException(status_code=400, detail="invalid chart period")
    if request.interval not in ("1m", "5m", "15m", "30m", "1h", "4h", "1d"):
        raise HTTPException(status_code=400, detail="invalid chart interval")
    return _json_safe(answer_chart_question(request))


@app.get("/api/health")
def health():
    return {"ok": True, "as_of": now_iso()}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(f"{STATIC_DIR}/index.html")
