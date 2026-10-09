# CWMR on weekly and monthly bars, 1999-2015 (paper Section 8)

| | |
|---|---|
| Commit | `97b0d4a98b81ddbe1492ab3907539f07bc0377db` |
| Commit time (author's machine) | 2026-10-06 18:06:29 -07:00 |
| First results committed | `548da5c`, 2026-10-06 18:08:22 -07:00 |

GitHub's event log has no record of this push, so the order rests on the repository history.
Note added for publication: the plan says "decide and fill at the close". Orders were later
moved to fill at the next session's open and both variants rerun once, then once more after a
correction to how next-open fills were booked (paper Section 8).
The plan's "Neither has been run before 1999" refers to data before 1999; neither variant had
been run on any data before 2016 either, as the paper states in Section 2.4.

## Text as committed

> ## CWMR on monthly and weekly bars, 1999-2015 (pre-registered)
>
> Written and committed before running. Variant A above learns from daily moves but holds for a
> month. These variants judge winners and losers over the same period they hold: B learns from
> month-end (or week-end) closes and trades at each month (or week) end. In 2016-2024, monthly
> bars returned 11.6% and weekly bars 17.6% against SPY's 14.5%; weekly bars failed the offset
> check. Neither has been run before 1999. The 1999-2015 window has been seen only by variant A.
>
> - Strategies: lab `olps/cwmr_monthly_bars` and `olps/cwmr_weekly_bars` v1 (CWMR with epsilon
>   0.89, theta 0.92, eta 0.93; the 16 ETFs; bars = "monthly" with rebalance = "month_end", and
>   bars = "weekly" with rebalance = "weekly").
> - Data: btest long history (Yahoo day bars), each ETF from its first trading day; QQQ joins in
>   March 1999, IWM in 2000, EFA in 2001, EEM in 2003, XLRE in October 2015.
> - Window: 1999-01-01 to 2015-12-31. Decide and fill at the close; $20,000, 0.7 bp per dollar
>   traded, fractional shares, idle cash at the T-bill rate, benchmark SPY with dividends.
> - Pass, judged separately for each: CAGR and Sharpe ratio both above SPY's. Also reported: max
>   drawdown, turnover.
> - Run once each from the website. Whatever the outcome, neither is changed and rerun on this
>   window.
