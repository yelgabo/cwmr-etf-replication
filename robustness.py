"""Checks quoted in Sections 2, 4, 6 and 8 of the paper, 1999-2024:

- bootstrap p-values at several block lengths, and Holm-adjusted p-values at each block length
  for the nine comparisons of Table 5;
- why longer blocks give smaller p-values (autocorrelation of the return difference);
- the correlation range of the paired tests, and trades per year in 1999-2001;
- how concentrated weekly bars is, and how often the CWMR update does not fire;
- where the weekly reversal happens: each ETF's return in excess of SPY's from the signal's
  close to the next open, and from that open to the open after, regressed on the week's
  excess return.

    uv run python robustness.py
"""
import numpy as np

import etf_tables as T
from cwmr_etf import stats
from cwmr_etf.cwmr import CWMR, OnlineCWMR

BLOCKS = (1, 5, 21, 63, 126)
res = T.res


def holm(block):
    p = np.array([stats.block_bootstrap(res[a].equity, res[b].equity, block=block)[1]
                  for a, b in T.PAIRS])
    adjusted, running = np.empty(len(p)), 0.0
    for rank, k in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[k]))
        adjusted[k] = running
    return p, adjusted


raw, adj = zip(*(holm(k) for k in BLOCKS))
print(f"Bootstrap p by block length {BLOCKS}, then Holm-adjusted over the {len(T.PAIRS)} "
      "comparisons at each block length")
for j, (a, b) in enumerate(T.PAIRS):
    print(f"  {a + ' vs ' + b:38} " + ", ".join(f"{r[j]:.3f}" for r in raw)
          + "  | Holm " + ", ".join(f"{h[j]:.3f}" for h in adj))


def daily_returns(r):
    return r.equity[1:] / r.equity[:-1] - 1


print("\nLag-1 autocorrelation of the daily return difference (negative: longer blocks give "
      "smaller p)")
for name in T.CWMR_NAMES:
    diff = daily_returns(res[name]) - daily_returns(res["Equal weight"])
    print(f"  {name} minus equal weight: {np.corrcoef(diff[1:], diff[:-1])[0, 1]:+.3f}")
rhos = [np.corrcoef(daily_returns(res[a]), daily_returns(res[b]))[0, 1] for a, b in T.PAIRS]
print(f"Correlation of daily returns across the paired tests: {min(rhos):.2f} to {max(rhos):.2f}")
w = res["Weekly bars"].fills_by_year
print("Weekly bars fills per year: " + ", ".join(f"{y} {w[y]}" for y in (1999, 2000, 2001)))

algo = OnlineCWMR(T.UNIVERSE, T.CLOSES, T.DAYS, bars="weekly")
updates = [0, 0]
original_update = CWMR.update


def counted(self, x):
    updates[0] += 1
    updates[1] += np.dot(self.b, x) <= self.epsilon
    return original_update(self, x)


CWMR.update = counted
held, top = [], []
for i, d in enumerate(T.DAYS):
    if d >= T.START and d in T.weekly_on:
        w = algo.weights(i)
        if w:
            held.append(sum(1 for v in w.values() if v > 0.01))
            top.append(max(w.values()))
CWMR.update = original_update
held, top = np.array(held), np.array(top)
print(f"\nWeekly bars: {len(held)} decisions; median ETFs held {np.median(held):.0f}; "
      f"largest weight 90% or more in {np.mean(top >= 0.9):.0%} of weeks, 50% or more in "
      f"{np.mean(top >= 0.5):.0%}")
print(f"CWMR updates computed (learning from 1995): {updates[0]}; not firing: {updates[1]} "
      f"({updates[1] / updates[0]:.2%})")

print("\nNext returns per 1% weekly move in excess of SPY, bp")
closes, opens, days = T.CLOSES, T.OPENS, T.DAYS
ends = [i for i, d in enumerate(days) if d in T.weekly_on and i + 2 < len(days)]
x, overnight, next_day = [], [], []
prev = None
for i in ends:
    if days[i] >= T.START and prev is not None:
        for s in T.UNIVERSE[1:]:
            c, o, sc, so = closes[s], opens[s], closes["SPY"], opens["SPY"]
            if np.isnan(c[prev]) or np.isnan(o[i + 2]):
                continue
            x.append((c[i] / c[prev] - 1) - (sc[i] / sc[prev] - 1))
            overnight.append((o[i + 1] / c[i] - 1) - (so[i + 1] / sc[i] - 1))
            next_day.append((o[i + 2] / o[i + 1] - 1) - (so[i + 2] / so[i + 1] - 1))
    prev = i
x = np.array(x)
b1 = np.polyfit(x, np.array(overnight), 1)[0] * 100
b2 = np.polyfit(x, np.array(next_day), 1)[0] * 100
print(f"  close to next open {b1:+.2f}; next open to the open after {b2:+.2f}")
