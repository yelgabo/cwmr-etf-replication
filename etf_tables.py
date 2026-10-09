"""Tables 4 to 7 and Sections 4 to 7 of the paper: CWMR on 16 ETFs, 1999-2024, daily data,
deciding on each session's close and filling at the next session's open.

    uv run python etf_tables.py

Downloads Yahoo and FRED data into data/ on the first run (about a minute).
"""
from datetime import date

from cwmr_etf import backtest, data, stats
from cwmr_etf.cwmr import OnlineCWMR

UNIVERSE = data.ETFS
START, END = date(1999, 1, 1), date(2025, 1, 1)
# The window the 1999-2015 plan registered (Section 7).
REGISTERED_END = date(2016, 1, 1)
RATES = data.tbill()
CALENDAR = data.sessions(data.START, date(2027, 12, 31))
DAYS, CLOSES, OPENS = data.panel(UNIVERSE, END)


def cwmr(bars, block=None):
    algo = OnlineCWMR(UNIVERSE, CLOSES, DAYS, bars=bars, block=block)
    return lambda i, account: algo.weights(i)


def equal_weight():
    """Equal weight in every symbol with two prices, reset on the last session of each month
    (and on the first session, when nothing is held)."""
    month_end = backtest.decision_days(CALENDAR, "month_end")

    def decide(i, account):
        if DAYS[i] not in month_end and account.positions:
            return None
        live = [s for s in UNIVERSE if i > 0 and CLOSES[s][i - 1] == CLOSES[s][i - 1]]
        return {s: 1.0 / len(live) for s in live} if live else None
    return decide


def spy(i, account):
    """Buy SPY with everything; buy again only when cash builds up past 0.1% of equity."""
    if account.positions.get("SPY") and account.cash < 0.001 * account.equity:
        return None
    return {"SPY": 1.0}


def offset_days(k):
    return {d for i, d in enumerate(DAYS) if (i + 1 - k) % 5 == 0}


def run(decide, decide_on, cost_bps=0.7, end=END):
    return backtest.run(DAYS, CLOSES, OPENS, decide, decide_on, START, end, RATES, cost_bps)


def describe(r):
    return (stats.cagr(r.days, r.equity), stats.sharpe(r.days, r.equity, RATES),
            stats.max_drawdown(r.equity), stats.turnover(r.days, r.equity, r.traded))


def breakeven(make, decide_on, target_cagr, hi=30.0):
    """Cost in bp per dollar traded at which the strategy's CAGR falls to target_cagr."""
    lo = 0.0
    for _ in range(14):
        mid = (lo + hi) / 2
        r = run(make(), decide_on, cost_bps=mid)
        lo, hi = (mid, hi) if stats.cagr(r.days, r.equity) > target_cagr else (lo, mid)
    return (lo + hi) / 2


daily_on = backtest.decision_days(CALENDAR, "daily")
weekly_on = backtest.decision_days(CALENDAR, "weekly")
monthly_on = backtest.decision_days(CALENDAR, "month_end")
SCHEDULES = {
    "Daily": (lambda: cwmr("daily"), daily_on),
    "Monthly, daily signal": (lambda: cwmr("daily"), monthly_on),
    "Monthly bars": (lambda: cwmr("monthly"), monthly_on),
    "Weekly bars": (lambda: cwmr("weekly"), weekly_on),
    "Equal weight": (equal_weight, daily_on),
    "SPY": (lambda: spy, daily_on),
}
CWMR_NAMES = ["Daily", "Monthly, daily signal", "Monthly bars", "Weekly bars"]
PAIRS = ([(n, "Equal weight") for n in CWMR_NAMES] + [(n, "SPY") for n in CWMR_NAMES]
         + [("Equal weight", "SPY")])

res = {name: run(make(), on) for name, (make, on) in SCHEDULES.items()}

if __name__ == "__main__":
    print("Table 4. 1999-2024, 0.7 bp per dollar traded")
    print(f"{'':24}{'CAGR':>8}{'Sharpe':>8}{'Max DD':>9}{'Turnover':>10}")
    for name, r in res.items():
        c, sh, dd, to = describe(r)
        print(f"{name:24}{c:8.2%}{sh:8.2f}{dd:9.1%}{to:9.1f}x")

    print("\nTable 5. Paired Sharpe tests: annual Sharpe difference, Jobson-Korkie-Memmel p,"
          " 21-day block bootstrap p and 95% interval")
    for a, b in PAIRS:
        jk = stats.paired_sharpe_test(res[a].equity, res[b].equity)[1]
        d, p, (lo, hi) = stats.block_bootstrap(res[a].equity, res[b].equity)
        print(f"{a + ' vs ' + b:38}{d:+6.2f}{jk:8.2f}{p:8.3f}   [{lo:+.2f}, {hi:+.2f}]")

    print("\nTable 6. Costs: CAGR at each cost in bp per dollar traded, and break-even costs")
    print(f"{'':24}" + "".join(f"{bp:>9} bp" for bp in (0.7, 5, 10, 20))
          + f"{'vs EW':>9}{'vs SPY':>9}")
    ew_cagr, spy_cagr = (stats.cagr(res[n].days, res[n].equity) for n in ("Equal weight", "SPY"))
    for name in CWMR_NAMES:
        make, on = SCHEDULES[name]
        cells = []
        for bp in (0.7, 5, 10, 20):
            r = res[name] if bp == 0.7 else run(make(), on, cost_bps=bp)
            cells.append(stats.cagr(r.days, r.equity))
        be = [breakeven(make, on, target) for target in (ew_cagr, spy_cagr)]
        print(f"{name:24}" + "".join(f"{c:12.2%}" for c in cells)
              + "".join(f"{b:8.2f}bp" if cells[0] > t else f"{'none':>9}"
                        for b, t in zip(be, (ew_cagr, spy_cagr))))

    print("\nTable 7. Weekly bars by where the week ends, 0.7 bp; bootstrap p against equal weight")
    print(f"{'':28}{'CAGR':>8}{'Sharpe':>8}{'p vs EW':>9}")
    versions = {"Calendar weeks (Friday)": res["Weekly bars"]}
    for k in range(5):
        versions[f"5-session blocks, offset {k}"] = run(cwmr(None, block=5), offset_days(k))
    for name, r in versions.items():
        c, sh, _, _ = describe(r)
        p = stats.block_bootstrap(r.equity, res["Equal weight"].equity)[1]
        print(f"{name:28}{c:8.2%}{sh:8.2f}{p:9.3f}")

    print("\nSection 6. Weekly bars traded one session late, 0.7 bp")
    late = OnlineCWMR(UNIVERSE, CLOSES, DAYS, bars="weekly")
    after_week_end = {DAYS[i + 1] for i, d in enumerate(DAYS[:-1]) if d in weekly_on}
    r = run(lambda i, account: late.weights(i - 1), after_week_end)
    c, sh, _, _ = describe(r)
    p = stats.block_bootstrap(r.equity, res["Equal weight"].equity)[1]
    print(f"{'Signal at week end, trade a session later':42}{c:8.2%}{sh:8.2f}"
          f"   bootstrap p vs EW {p:.3f}")

    print("\nSection 7. The registered 1999-2015 test: CAGR and Sharpe over T-bills, 0.7 bp")
    for name in ("Weekly bars", "Monthly bars", "Equal weight", "SPY"):
        make, on = SCHEDULES[name]
        r = run(make(), on, end=REGISTERED_END)
        c, sh, _, _ = describe(r)
        print(f"{name:24}{c:8.2%}{sh:8.2f}")
