"""Section 8: next-day autocorrelation of each ETF's daily return in excess of SPY's, 1999-2024.
Reports the median across the 15 non-SPY ETFs.

    uv run python autocorrelation.py
"""
from datetime import date

import numpy as np

from cwmr_etf import data

days, closes, _ = data.panel(data.ETFS, date(2025, 1, 1))
spy = closes["SPY"]
for lo, hi in ((1999, 2024),):
    ks = [k for k, d in enumerate(days) if lo <= d.year <= hi]
    values = []
    for s in data.ETFS[1:]:
        c, p = closes[s][ks], spy[ks]
        ok = ~np.isnan(c)
        excess = np.diff(c[ok]) / c[ok][:-1] - np.diff(p[ok]) / p[ok][:-1]
        if len(excess) > 100:
            values.append(np.corrcoef(excess[:-1], excess[1:])[0, 1])
    print(f"{lo}-{hi}: median {np.median(values):+.3f} over {len(values)} ETFs")
