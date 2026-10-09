"""Checks quoted in Sections 2, 4, 5, 6 and 9 of the paper, 1999-2024:

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
from datetime import date

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
for name in ("Weekly bars", "Daily"):
    w = res[name].fills_by_year
    print(f"{name} fills per year: " + ", ".join(f"{y} {w[y]}" for y in (1999, 2000, 2001)))


def turnover_by_year(r):
    out = {}
    for y, dollars in r.traded_by_year.items():
        eq = [e for d, e in zip(r.days, r.equity) if d.year == y]
        if eq:
            out[y] = dollars / np.mean(eq)
    return out


for name in ("Daily", "Weekly bars"):
    tby = turnover_by_year(res[name])
    print(f"{name} turnover by year (dollars traded over that year's mean equity): "
          f"mean {np.mean(list(tby.values())):.0f}x, 1999-2005 "
          f"{min(tby[y] for y in range(1999, 2006)):.0f}x to {max(tby[y] for y in range(1999, 2006)):.0f}x")


print("CAGR by part of the window, to show where each lead comes from")
for name in ("Daily", "Weekly bars", "Equal weight", "SPY"):
    r = res[name]
    cells = [stats.cagr(*stats.window(r.days, r.equity, lo, hi))
             for lo, hi in ((T.START, date(2009, 1, 1)), (date(2009, 1, 1), T.END))]
    print(f"  {name:14} 1999-2008 {cells[0]:6.2%}  2009-2024 {cells[1]:6.2%}")


def variance_ratio(diff, k):
    sums = np.convolve(diff, np.ones(k), "valid")
    return sums.var() / (k * diff.var())


diff = daily_returns(res["Weekly bars"]) - daily_returns(res["Equal weight"])
print("Variance ratios of the weekly bars minus equal weight return difference: "
      + ", ".join(f"{k}-day {variance_ratio(diff, k):.2f}" for k in (5, 21, 63, 126)))


def lead_bootstrap(a, b, block=21, draws=10_000, seed=0):
    """Annual log-return lead of a over b, with a moving-block bootstrap 95% interval."""
    la, lb = np.log(a.equity[1:] / a.equity[:-1]), np.log(b.equity[1:] / b.equity[:-1])
    d = la - lb
    n, k = len(d), len(d) // block
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for j in range(draws):
        idx = (rng.integers(0, n - block + 1, k)[:, None] + np.arange(block)).ravel()
        out[j] = d[idx].mean() * 252
    lead = d.mean() * 252
    p = float(np.mean(np.abs(out - out.mean()) >= abs(lead)))
    return lead, np.percentile(out, 2.5), np.percentile(out, 97.5), p


print("\nAnnual log-return lead, 21-day block bootstrap: lead [95% interval], p, Holm-adjusted p")
leads = [lead_bootstrap(res[a], res[b]) for a, b in T.PAIRS]
lead_p = np.array([x[3] for x in leads])
lead_holm, running = np.empty(len(lead_p)), 0.0
for rank, k in enumerate(np.argsort(lead_p)):
    running = max(running, min(1.0, (len(lead_p) - rank) * lead_p[k]))
    lead_holm[k] = running
for (a, b), (lead, lo, hi, p), h in zip(T.PAIRS, leads, lead_holm):
    print(f"  {a + ' vs ' + b:38} {lead:+.2%} [{lo:+.2%}, {hi:+.2%}]  p {p:.3f}  Holm {h:.3f}")


def holm_adjust(p):
    adjusted, running = np.empty(len(p)), 0.0
    for rank, k in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[k]))
        adjusted[k] = running
    return adjusted


by_block = [np.array([lead_bootstrap(res[a], res[b], block=k)[3] for a, b in T.PAIRS])
            for k in BLOCKS]
print(f"\nReturn-lead bootstrap p by block length {BLOCKS}, then Holm-adjusted over the "
      f"{len(T.PAIRS)} comparisons at each block length")
for j, (a, b) in enumerate(T.PAIRS):
    print(f"  {a + ' vs ' + b:38} " + ", ".join(f"{ps[j]:.3f}" for ps in by_block)
          + "  | Holm " + ", ".join(f"{holm_adjust(ps)[j]:.3f}" for ps in by_block))

algo = OnlineCWMR(T.UNIVERSE, T.CLOSES, T.DAYS, bars="weekly")
updates = [0, 0]
original_update = CWMR.update


def counted(self, x):
    # Each decision also updates a throwaway copy with the week so far; count only the
    # algorithm's own updates.
    if self is algo.algo:
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
print(f"Weekly CWMR updates (learning from 1995): {updates[0]}; not firing: {updates[1]} "
      f"({updates[1] / updates[0]:.2%})")

print("\nNext returns per 1% weekly move in excess of SPY, bp")
closes, opens, days = T.CLOSES, T.OPENS, T.DAYS
ends = [i for i, d in enumerate(days) if d in T.weekly_on and i + 2 < len(days)]
x, overnight, next_day, week = [], [], [], []
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
            week.append(i)
    prev = i
x = np.array(x)
weeks = np.array(week)


def slope_clustered(x, y, groups):
    """OLS slope with a standard error clustered by week (ETFs in a week move together)."""
    xc = x - x.mean()
    beta = (xc @ (y - y.mean())) / (xc @ xc)
    resid = y - y.mean() - beta * xc
    scores = np.array([np.sum(xc[groups == g] * resid[groups == g]) for g in np.unique(groups)])
    return beta, np.sqrt(np.sum(scores ** 2)) / (xc @ xc)


for label, y in (("close to next open", overnight), ("next open to the open after", next_day)):
    beta, se = slope_clustered(x, np.array(y), weeks)
    print(f"  {label}: {beta * 100:+.2f} (s.e. {se * 100:.2f}, clustered by week)")
