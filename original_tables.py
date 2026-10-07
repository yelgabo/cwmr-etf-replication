"""Section 3 and Table 3 of the paper: the original paper's results from its own code and data.

    uv run python original_tables.py

Clones Lahanis, Liu and Zhou's repository at the version we tested into authors-repo/ and runs
their functions unchanged on their 93 NASDAQ-100 stocks, 1998-2009.
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cwmr_etf.stats import paired_sharpe_test

REPO = "https://github.com/nglahani/Online-Quantitative-Trading-Strategies"
COMMIT = "7c2e88d763f0eb3d4807e60d5414ce512d06688e"
HERE = Path(__file__).resolve().parent / "authors-repo"

if not HERE.exists():
    subprocess.run(["git", "clone", "-q", REPO, str(HERE)], check=True)
subprocess.run(["git", "-C", str(HERE), "checkout", "-q", COMMIT], check=True)
sys.path[:0] = [str(HERE / "Scripts"), str(HERE / "Scripts" / "Strategies")]
import follow_the_loser as ftl  # noqa: E402
import follow_the_winner as ftw  # noqa: E402

df = pd.read_csv(HERE / "Data" / "Price Relative Vectors" / "price_relative_vectors.csv",
                 index_col=0)
rel = df.to_numpy(float)
n = rel.shape[1]
b0 = np.full(n, 1 / n)
crp = np.tile(b0, (len(rel), 1))
cwmr = np.asarray(ftl.cwmr(b0, rel))
pamr = np.asarray(ftl.pamr(b0, rel))
print(f"{n} stocks, {len(rel)} days, {df.index[0]} to {df.index[-1]}")


def daily_returns(b, x, cost_bps=0.0):
    """Hold weights b[t] over day t; pay cost_bps on every dollar moved from the drifted
    weights. A one-day delay is daily_returns(b[:-1], x[1:])."""
    out, prev = [], None
    for t in range(len(x)):
        c = 0.0
        if prev is not None and cost_bps:
            drift = prev * x[t - 1]
            c = np.abs(b[t] - drift / drift.sum()).sum() * cost_bps / 1e4
        out.append((1 - c) * float(b[t] @ x[t]) - 1)
        prev = b[t]
    return np.array(out)


def wealth(r):
    return np.concatenate([[1.0], np.cumprod(1 + r)])


wealth_rows = {"CWMR": cwmr, "PAMR": pamr, "Anticor": np.asarray(ftl.anticor(b0, rel)),
               "FTRL": np.asarray(ftw.follow_the_regularized_leader(b0, rel)), "CRP": crp}
print("Final wealth (paper's Tables 3-4):",
      ", ".join(f"{k} {wealth(daily_returns(b, rel))[-1]:,.2f}x" for k, b in wealth_rows.items()))

base, base_delay = daily_returns(crp, rel), daily_returns(crp[:-1], rel[1:])
rows = [("CRP", base, None),
        ("CWMR", daily_returns(cwmr, rel), base),
        ("CWMR, 10 bp", daily_returns(cwmr, rel, 10), base),
        ("CWMR, one-day delay", daily_returns(cwmr[:-1], rel[1:]), base_delay),
        ("PAMR, one-day delay", daily_returns(pamr[:-1], rel[1:]), base_delay),
        ("PAMR, one-day delay and 5 bp", daily_returns(pamr[:-1], rel[1:], 5), base_delay)]
print("\nTable 3. The authors' data, 1998-2009")
print(f"{'':30}{'CAGR':>8}{'p vs CRP':>10}")
for name, r, ref in rows:
    w = wealth(r)
    growth = w[-1] ** (252 / len(r)) - 1
    p = "" if ref is None else f"{paired_sharpe_test(wealth(r), wealth(ref))[1]:.4f}"
    print(f"{name:30}{growth:8.1%}{p:>10}")
