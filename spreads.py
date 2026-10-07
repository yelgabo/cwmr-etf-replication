"""Estimated half-spreads from daily open, high, low and close prices (EDGE; Ardia, Guidotti
and Kroencke, 2024), by ETF and year, checked against half-spreads measured from quotes at
15:45 in 2016-2024.

    uv run python spreads.py
"""
import numpy as np
from bidask import edge

from cwmr_etf import data

# Median half-spread in bp at 15:45 New York time, 45 sessions in 2016-2024, from Alpaca SIP
# quotes (the paper's Table 2).
MEASURED = {"SPY": 0.17, "QQQ": 0.26, "IWM": 0.30, "DIA": 0.29, "EFA": 0.73, "EEM": 1.20,
            "XLK": 0.50, "XLF": 1.68, "XLV": 0.50, "XLE": 0.74, "XLI": 0.64, "XLY": 0.41,
            "XLP": 0.80, "XLU": 0.81, "XLB": 0.82, "XLRE": 1.35}


def half_spread_bp(px, year):
    y = px[[d.year == year for d in px["date"]]]
    if len(y) < 120:
        return None
    s = edge(y["open"].to_numpy(), y["high"].to_numpy(), y["low"].to_numpy(),
             y["close"].to_numpy(), sign=True)
    return None if s is None or np.isnan(s) else s / 2 * 1e4


def table(years):
    rows = {}
    for sym in data.ETFS:
        px = data.yahoo(sym)
        rows[sym] = {y: half_spread_bp(px, y) for y in years}
    return rows


def fmt(v):
    return "     ." if v is None else f"{v:6.1f}"


if __name__ == "__main__":
    years = list(range(1999, 2025))
    est = table(years)
    print("EDGE half-spread estimates, bp (signed; negative means below what the method resolves)")
    print(f"{'':6}" + "".join(f"{y % 100:>6}" for y in years))
    for sym, row in est.items():
        print(f"{sym:6}" + "".join(fmt(row[y]) for y in years))
    print("\nCheck against measured half-spreads, 2016-2024")
    print(f"{'':6}{'measured':>10}{'EDGE median':>13}")
    for sym, row in est.items():
        vals = [row[y] for y in range(2016, 2025) if row[y] is not None]
        print(f"{sym:6}{MEASURED[sym]:10.2f}{np.median(vals):13.1f}")
