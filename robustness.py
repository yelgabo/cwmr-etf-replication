"""Checks quoted in Sections 2, 6 and 8 of the paper:

- bootstrap p-values for weekly bars against SPY and equal weight at several block lengths,
  and Holm-adjusted p-values for the five comparisons of Tables 6 and 7;
- how concentrated weekly bars is, and how often the CWMR update does not fire;
- where the weekly reversal happens: each ETF's return in excess of SPY's from the signal's
  close to the next open, and from that open to the open after, regressed on the week's
  excess return.

    uv run python robustness.py
"""
import contextlib
import io
from datetime import date

import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import etf_tables as T
from cwmr_etf import stats
from cwmr_etf.cwmr import CWMR, OnlineCWMR

res = {n: T.run(T.S2016, f, on) for n, (f, on) in T.strategies(T.S2016).items()}

print("Bootstrap p by block length, 1999-2015")
for a, b in (("Weekly bars", "SPY"), ("Weekly bars", "Equal weight"), ("Monthly bars", "SPY"),
             ("Monthly bars", "Equal weight")):
    ps = [stats.block_bootstrap(res[a].equity, res[b].equity, block=k)[1] for k in (1, 5, 21, 63, 126)]
    print(f"  {a} vs {b}: " + ", ".join(f"{k}-day {p:.3f}" for k, p in zip((1, 5, 21, 63, 126), ps)))

pairs = (("Weekly bars", "SPY"), ("Weekly bars", "Equal weight"), ("Monthly bars", "SPY"),
         ("Monthly bars", "Equal weight"), ("Equal weight", "SPY"))


def holm(runs, block=21):
    p = np.array([stats.block_bootstrap(runs[a].equity, runs[b].equity, block=block)[1]
                  for a, b in pairs])
    adjusted, running = np.empty(len(p)), 0.0
    for rank, k in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[k]))
        adjusted[k] = running
    return adjusted


print("\nHolm-adjusted bootstrap p (21-day blocks), 1999-2015")
for (a, b), x in zip(pairs, holm(res)):
    print(f"  {a} vs {b}: {x:.3f}")

whole = {n: T.run(T.S2025, f, on) for n, (f, on) in T.strategies(T.S2025).items()}
print("\nBootstrap p by block length, 1999-2024 (Table 7)")
for a, b in pairs:
    ps = [stats.block_bootstrap(whole[a].equity, whole[b].equity, block=k)[1]
          for k in (1, 5, 21, 63, 126)]
    print(f"  {a} vs {b}: " + ", ".join(f"{k}-day {p:.3f}" for k, p in zip((1, 5, 21, 63, 126), ps)))
print("Holm-adjusted bootstrap p (21-day blocks), 1999-2024")
for (a, b), x in zip(pairs, holm(whole)):
    print(f"  {a} vs {b}: {x:.3f}")

print("\nHolm-adjusted bootstrap p at each block length (1, 5, 21, 63, 126 days)")
for label, runs in (("1999-2015", res), ("1999-2024", whole)):
    by_block = [holm(runs, k) for k in (1, 5, 21, 63, 126)]
    for j, (a, b) in enumerate(pairs):
        print(f"  {label} {a} vs {b}: " + ", ".join(f"{h[j]:.3f}" for h in by_block))


def daily_returns(r):
    return r.equity[1:] / r.equity[:-1] - 1


print("\nWhy longer blocks give smaller p: lag-1 autocorrelation of the daily return difference")
for label, runs in (("1999-2015", res), ("1999-2024", whole)):
    diff = daily_returns(runs["Weekly bars"]) - daily_returns(runs["Equal weight"])
    print(f"  {label} weekly bars minus equal weight: {np.corrcoef(diff[1:], diff[:-1])[0, 1]:+.3f}")
corr = np.corrcoef(daily_returns(res["Equal weight"]), daily_returns(res["SPY"]))[0, 1]
print(f"Correlation of daily returns, equal weight and SPY, 1999-2015: {corr:.3f}")
w = whole["Weekly bars"].fills_by_year
print("Weekly bars fills per year: " + ", ".join(f"{y} {w[y]}" for y in (1999, 2000, 2001)))

days, closes, opens = T.PANELS[T.S2025]
algo = OnlineCWMR(T.UNIVERSE, closes, days, bars="weekly")
updates = [0, 0]
original_update = CWMR.update


def counted(self, x):
    updates[0] += 1
    updates[1] += np.dot(self.b, x) <= self.epsilon
    return original_update(self, x)


CWMR.update = counted
held, top = [], []
for i, d in enumerate(days):
    if d >= T.S1999 and d in T.weekly_on:
        w = algo.weights(i)
        if w:
            held.append(sum(1 for v in w.values() if v > 0.01))
            top.append(max(w.values()))
CWMR.update = original_update
held, top = np.array(held), np.array(top)
print(f"\nWeekly bars, 1999-2024: {len(held)} decisions; median ETFs held {np.median(held):.0f}; "
      f"largest weight 90% or more in {np.mean(top >= 0.9):.0%} of weeks, 50% or more in "
      f"{np.mean(top >= 0.5):.0%}")
print(f"CWMR updates computed: {updates[0]}; not firing: {updates[1]} ({updates[1] / updates[0]:.2%})")

print("\nNext returns per 1% weekly move in excess of SPY, bp")
ends = [i for i, d in enumerate(days) if d in T.weekly_on and i + 2 < len(days)]
for lo, hi in ((1999, 2016), (2016, 2025)):
    x, overnight, next_day = [], [], []
    prev = None
    for i in ends:
        if lo <= days[i].year < hi and prev is not None:
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
    print(f"  {lo}-{hi - 1}: close to next open {b1:+.2f}; next open to the open after {b2:+.2f}")
