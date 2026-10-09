# Monthly CWMR on ETFs spliced with stand-in funds, 1995-2015 (paper Section 8, superseded)

| | |
|---|---|
| Commit | `d9a912d9040327a5e8fdeda4911611086ed73a77` |
| Commit time (author's machine) | 2026-10-06 01:12:10 -07:00 |
| Received by GitHub (push event) | 2026-10-06 08:12:12 UTC |

Note added for publication: this test filled orders at the decision's own close and filled the
years before each ETF's launch with stand-in funds. The paper no longer relies on it (Section 8).
Paths refer to files in https://github.com/yelgabo/btest.

## Text as committed

> ## Long-history test of monthly CWMR, 1995-2015 (pre-registered)
>
> Written and committed before running. The 2016-2024 study chose monthly CWMR; this asks whether
> its lead over SPY also appears in the 21 years before, which no strategy here has seen.
>
> **Data** (`long_history.py`, check in `outputs/long_history_check.out`). Each of the 16 slots
> uses its ETF from launch and a stand-in before, all as daily total returns (Yahoo adjusted
> closes, dividends reinvested). No slot is empty at any point, so the universe never changes.
>
> | Slot | Stand-in until ETF launch | ETF from | Daily corr. with ETF, launch-2024 |
> |---|---|---|---|
> | SPY | none | 1993-01-29 | |
> | QQQ | Rydex NASDAQ-100 (RYOCX) | 1999-03-10 | 0.978 |
> | IWM | Vanguard Small-Cap Index (NAESX) | 2000-05-26 | 0.981 |
> | DIA | Dow price index plus 2% a year (no dividend series found) | 1998-01-20 | 0.987 |
> | EFA | 60% Vanguard European (VEURX), 40% Pacific (VPACX) | 2001-08-27 | 0.968 |
> | EEM | Vanguard Emerging Markets Index (VEIEX) | 2003-04-14 | 0.938 |
> | XLK, XLF, XLV, XLE | Fidelity Select FSPTX, FIDSX, FSPHX, FSENX | 1998-12-22 | 0.949, 0.965, 0.810, 0.970 |
> | XLI | Ken French 12-industry "Manuf" (Fidelity's FCYIX has no history on Yahoo) | 1998-12-22 | 0.942 |
> | XLY, XLP, XLU, XLB | Fidelity Select FSCPX, FDFAX, FSUTX, FSDPX | 1998-12-22 | 0.933, 0.869, 0.878, 0.939 |
> | XLRE | Fidelity Real Estate Investment (FRESX) | 2015-10-08 | 0.971 |
>
> The Fidelity Select funds are actively managed, so they track their sectors loosely (0.81 to
> 0.97). Most of the window runs on stand-ins: until 1998 every slot but SPY is one.
>
> **Pipeline check** (`long_history_run.py validate`, `outputs/long_history_validate.out`). On
> 2016-2024, where every slot is the ETF, this data gives monthly CWMR 18.40%, Sharpe 0.87,
> turnover 17.8x; btest gave 18.3%, 0.87, 18x. SPY differs: 14.58% here against btest's 13.8%,
> because btest's buy-and-hold baseline keeps SPY's dividends as cash (8.6% of the account by the
> end of 2024) instead of reinvesting them. SPY's total return is the right benchmark, so this
> test uses it; the earlier comparisons with btest's SPY overstate CWMR's lead by about 0.8
> points a year.
>
> **Test.**
>
> - Strategy: lab `olps/cwmr_monthly` v1, unchanged (epsilon 0.89, theta 0.92, eta 0.93), learning
>   from scratch on 1995-01-03.
> - Window: 1995-01-03 to 2015-12-31.
> - Timing: daily closes only, so it decides on the close of each month's last session and fills
>   at that close (btest decides at 15:30 and fills at 15:45; the check above shows they agree).
> - Settings: $20,000, fractional shares, idle cash at the 3-month T-bill rate (FRED DTB3),
>   benchmark SPY with dividends reinvested.
> - Pass: CAGR and Sharpe ratio both above SPY's at 0.7 bp per dollar traded. Also reported: the
>   paired Sharpe test, max drawdown, turnover, and results at 5, 10 and 20 bp.
> - Limits: the stand-ins are investable but were not tradable this cheaply. Fidelity Select
>   funds charged short-term trading fees then, so this tests whether the signal exists in
>   these return series, not whether it could have been traded profitably in 1995.
> - Run once. Whatever the outcome, the strategy and the data are not changed and rerun on this
>   window.
