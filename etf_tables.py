"""Tables 5, 6 and 7 of the paper: CWMR on weekly and monthly bars over 16 ETFs, 1999-2024.

    uv run python etf_tables.py

Downloads Yahoo and FRED data into data/ on the first run (about a minute).
"""
from datetime import date

from cwmr_etf import backtest, data, stats
from cwmr_etf.cwmr import OnlineCWMR

UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY",
            "XLP", "XLU", "XLB", "XLRE"]
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


print("Table 5. Pre-registered test, 1999-2015, 0.7 bp per dollar traded")
print(f"{'':14}{'CAGR':>8}{'Sharpe':>8}{'Max DD':>9}{'Turnover':>10}{'p vs SPY':>10}{'p vs EW':>9}")
res = {name: run(S2016, f, on) for name, (f, on) in strategies(S2016).items()}
for name, r in res.items():
    c, sh, dd, to = describe(r)
    p_spy = "" if name == "SPY" else f"{stats.paired_sharpe_test(r.equity, res['SPY'].equity)[1]:.2f}"
    p_ew = (f"{stats.paired_sharpe_test(r.equity, res['Equal weight'].equity)[1]:.2f}"
            if name.endswith("bars") else "")
    print(f"{name:14}{c:8.2%}{sh:8.2f}{dd:9.1%}{to:9.1f}x{p_spy:>10}{p_ew:>9}")

print("\nTable 6. Weekly bars by the session that ends each week, 1999-2024, 0.7 bp, CAGR")
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

print("\nTable 7. Weekly bars at higher costs per dollar traded, CAGR")
print(f"{'':12}" + "".join(f"{bp:>9} bp" for bp in (0.7, 5, 10, 20)))
for end in (S2016, S2025):
    days, closes, _ = PANELS[end]
    row = []
    for bp in (0.7, 5, 10, 20):
        r = run(end, cwmr(days, closes, "weekly"), weekly_on, cost_bps=bp)
        row.append(stats.cagr(r.days, r.equity))
    print(f"1999-{end.year - 1:<7}" + "".join(f"{c:12.2%}" for c in row))
