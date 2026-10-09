"""Tables 5 to 9 and Section 6.4 of the paper: CWMR on weekly and monthly bars over 16 ETFs, 1999-2024.

    uv run python etf_tables.py

Downloads Yahoo and FRED data into data/ on the first run (about a minute).
"""
from datetime import date

from cwmr_etf import backtest, data, stats
from cwmr_etf.cwmr import OnlineCWMR

UNIVERSE = data.ETFS
S1999, S2016, S2025 = date(1999, 1, 1), date(2016, 1, 1), date(2025, 1, 1)
RATES = data.tbill()
CALENDAR = data.sessions(data.START, date(2027, 12, 31))


def cwmr(days, closes, bars, block=None):
    algo = OnlineCWMR(UNIVERSE, closes, days, bars=bars, block=block)
    return lambda i, account: algo.weights(i)


def equal_weight(closes):
    """Equal weight in every symbol with two prices, reset on the last session of each month
    (and on the first session, when nothing is held)."""
    month_end = backtest.decision_days(CALENDAR, "month_end")

    def decide(i, account, days=None):
        if days[i] not in month_end and account.positions:
            return None
        live = [s for s in UNIVERSE if i > 0 and closes[s][i - 1] == closes[s][i - 1]]
        return {s: 1.0 / len(live) for s in live} if live else None
    return decide


def spy(i, account):
    """Buy SPY with everything; buy again only when cash builds up past 0.1% of equity."""
    if account.positions.get("SPY") and account.cash < 0.001 * account.equity:
        return None
    return {"SPY": 1.0}


def offset_days(days, k):
    return {d for i, d in enumerate(days) if (i + 1 - k) % 5 == 0}


def run(end, decide, decide_on, cost_bps=0.7, start=S1999):
    days, closes, opens = PANELS[end]
    return backtest.run(days, closes, opens, decide, decide_on, start, end, RATES, cost_bps)


def describe(r):
    return (stats.cagr(r.days, r.equity), stats.sharpe(r.days, r.equity, RATES),
            stats.max_drawdown(r.equity), stats.turnover(r.days, r.equity, r.traded))


PANELS = {end: data.panel(UNIVERSE, end) for end in (S2016, S2025)}
weekly_on = backtest.decision_days(CALENDAR, "weekly")
monthly_on = backtest.decision_days(CALENDAR, "month_end")
daily_on = backtest.decision_days(CALENDAR, "daily")


def strategies(end):
    days, closes, _ = PANELS[end]
    ew = equal_weight(closes)
    return {
        "Weekly bars": (cwmr(days, closes, "weekly"), weekly_on),
        "Monthly bars": (cwmr(days, closes, "monthly"), monthly_on),
        "Equal weight": (lambda i, a: ew(i, a, days), daily_on),
        "SPY": (spy, daily_on),
    }


PAIRS = (("Weekly bars", "SPY"), ("Weekly bars", "Equal weight"), ("Monthly bars", "SPY"),
         ("Monthly bars", "Equal weight"), ("Equal weight", "SPY"))

print("Table 5. Pre-registered test, 1999-2015, 0.7 bp per dollar traded")
print(f"{'':14}{'CAGR':>8}{'Sharpe':>8}{'Max DD':>9}{'Turnover':>10}")
res = {name: run(S2016, f, on) for name, (f, on) in strategies(S2016).items()}
for name, r in res.items():
    c, sh, dd, to = describe(r)
    print(f"{name:14}{c:8.2%}{sh:8.2f}{dd:9.1%}{to:9.1f}x")
print("\nTable 6. Paired Sharpe tests, 1999-2015: annual Sharpe difference, Jobson-Korkie-Memmel"
      " p, 21-day block bootstrap p and 95% interval")
for a, b in PAIRS:
    jk = stats.paired_sharpe_test(res[a].equity, res[b].equity)[1]
    d, p, (lo, hi) = stats.block_bootstrap(res[a].equity, res[b].equity)
    print(f"{a + ' vs ' + b:28}{d:+6.2f}{jk:8.2f}{p:8.3f}   [{lo:+.2f}, {hi:+.2f}]")

print("\nTable 7. Exploratory: the whole period, 1999-2024, 0.7 bp; paired tests as in Table 6")
print(f"{'':14}{'CAGR':>8}{'Sharpe':>8}{'Max DD':>9}{'Turnover':>10}")
whole = {name: run(S2025, f, on) for name, (f, on) in strategies(S2025).items()}
for name, r in whole.items():
    c, sh, dd, to = describe(r)
    print(f"{name:14}{c:8.2%}{sh:8.2f}{dd:9.1%}{to:9.1f}x")
for a, b in PAIRS:
    jk = stats.paired_sharpe_test(whole[a].equity, whole[b].equity)[1]
    d, p, (lo, hi) = stats.block_bootstrap(whole[a].equity, whole[b].equity)
    print(f"{a + ' vs ' + b:28}{d:+6.2f}{jk:8.2f}{p:8.3f}   [{lo:+.2f}, {hi:+.2f}]")

print("\nTable 8. Weekly bars by where the week ends, 1999-2024, 0.7 bp, CAGR")
print(f"{'':28}{'1999-2015':>11}{'2016-2024':>11}{'1999-2024':>11}")
days25, closes25, _ = PANELS[S2025]
versions = {"Calendar weeks (Friday)": (cwmr(days25, closes25, "weekly"), weekly_on)}
for k in range(5):
    versions[f"5-session blocks, offset {k}"] = (cwmr(days25, closes25, None, block=5),
                                                 offset_days(days25, k))
for name in ("Equal weight", "SPY"):
    versions[name] = strategies(S2025)[name]
for name, (f, on) in versions.items():
    r = run(S2025, f, on)
    cells = [stats.cagr(*stats.window(r.days, r.equity, lo, hi))
             for lo, hi in ((S1999, S2016), (S2016, S2025), (S1999, S2025))]
    print(f"{name:28}" + "".join(f"{c:11.1%}" for c in cells))

print("\nTable 9. Weekly bars at higher costs per dollar traded, CAGR")
print(f"{'':12}" + "".join(f"{bp:>9} bp" for bp in (0.7, 5, 10, 20)))
for end in (S2016, S2025):
    days, closes, _ = PANELS[end]
    row = []
    for bp in (0.7, 5, 10, 20):
        r = run(end, cwmr(days, closes, "weekly"), weekly_on, cost_bps=bp)
        row.append(stats.cagr(r.days, r.equity))
    print(f"1999-{end.year - 1:<7}" + "".join(f"{c:12.2%}" for c in row))

print("\nSection 6.4. Weekly bars traded one session late, 1999-2015, 0.7 bp")
days, closes, _ = PANELS[S2016]
late = OnlineCWMR(UNIVERSE, closes, days, bars="weekly")
after_week_end = {days[i + 1] for i, d in enumerate(days[:-1]) if d in weekly_on}
r = run(S2016, lambda i, account: late.weights(i - 1), after_week_end)
c, sh, dd, to = describe(r)
p_ew = stats.block_bootstrap(r.equity, res["Equal weight"].equity)[1]
print(f"{'Signal at week end, trade a session later':42}{c:8.2%}{sh:8.2f}   bootstrap p vs EW {p_ew:.2f}")

ew = res["Equal weight"]
print(f"\nBreak-even against equal weight, 1999-2015 (equal weight {stats.cagr(ew.days, ew.equity):.2%})")
for bp in (4, 4.5):
    r = run(S2016, cwmr(*PANELS[S2016][:2], "weekly"), weekly_on, cost_bps=bp)
    print(f"Weekly bars at {bp} bp: {stats.cagr(r.days, r.equity):.2%}")
