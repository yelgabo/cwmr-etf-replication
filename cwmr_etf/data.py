"""Public data: Yahoo Finance daily bars, FRED's 3-month T-bill rate, the NYSE calendar.

Downloads are cached under data/ so reruns use the same snapshot. Yahoo's adjusted closes
include dividends and splits; the open is scaled by the same day's adjustment factor.
"""
import json
import time
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

CACHE = Path(__file__).resolve().parent.parent / "data"
START = date(1995, 1, 3)
UA = {"User-Agent": "Mozilla/5.0"}


def sessions(start: date, end: date) -> list[date]:
    """NYSE sessions from start to end inclusive."""
    sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    return [d.date() for d in sched.index]


def yahoo(symbol: str) -> pd.DataFrame:
    """Daily adjusted open and close for one symbol, from its first trading day."""
    path = CACHE / f"{symbol}.csv"
    if not path.exists():
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               "?period1=0&period2=4000000000&interval=1d&events=div,split")
        r = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                             timeout=60))["chart"]["result"][0]
        time.sleep(0.5)
        days = [datetime.fromtimestamp(t + r["meta"]["gmtoffset"], UTC).date()
                for t in r["timestamp"]]
        q = r["indicators"]["quote"][0]
        df = pd.DataFrame({"date": days, "open": q["open"], "raw_close": q["close"],
                           "close": r["indicators"]["adjclose"][0]["adjclose"]}).dropna()
        df["open"] = df["open"] * df["close"] / df["raw_close"]
        CACHE.mkdir(exist_ok=True)
        df[["date", "open", "close"]].to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["date"]).assign(date=lambda d: d["date"].dt.date)


def tbill() -> tuple[list[date], list[float]]:
    """FRED DTB3: dates and annual rates in percent, holidays dropped."""
    path = CACHE / "DTB3.csv"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        path.write_bytes(urllib.request.urlopen(urllib.request.Request(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3", headers=UA),
            timeout=60).read())
    t = pd.read_csv(path, na_values=".").dropna()
    t.columns = ["date", "rate"]
    return [date.fromisoformat(d) for d in t["date"]], t["rate"].astype(float).tolist()


def panel(symbols: list[str], end: date) -> tuple[list[date], dict[str, np.ndarray],
                                                  dict[str, np.ndarray]]:
    """Sessions from START to before end, and per symbol arrays of adjusted close and open on
    those sessions: NaN before the symbol's first trading day; a session Yahoo skipped repeats
    the previous close as its open and close."""
    days = [d for d in sessions(START, end) if d < end]
    index = {d: k for k, d in enumerate(days)}
    closes, opens = {}, {}
    for s in symbols:
        px = yahoo(s)
        c = np.full(len(days), np.nan)
        o = np.full(len(days), np.nan)
        for d, op, cl in px[["date", "open", "close"]].itertuples(index=False):
            k = index.get(d)
            if k is not None:
                c[k], o[k] = cl, op
        first = max(px["date"].min(), START)
        k0 = index[min(d for d in days if d >= first)]
        for k in range(k0 + 1, len(days)):
            if np.isnan(c[k]):
                c[k] = c[k - 1]
                o[k] = c[k]
        closes[s], opens[s] = c, o
    return days, closes, opens
