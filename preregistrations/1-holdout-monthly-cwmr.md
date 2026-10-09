# Holdout test of monthly CWMR, 2025-2026 (paper Section 7)

| | |
|---|---|
| Commit | `7f9a8b09d6b314ede1f48e5a36ee811d418145a9` |
| Commit time (author's machine) | 2026-10-05 03:06:04 -07:00 |
| Received by GitHub (push event) | 2026-10-05 10:06:06 UTC |
| Run created (backtest database) | 2026-10-05 10:06:32 UTC |

Note added for publication: the plan says no run of any strategy had used the holdout window.
That was wrong: two unrelated single-ETF strategies (a volatility-targeting rule and a 200-day
trend filter) had been run on it two days earlier. No portfolio selection strategy had.

## Text as committed

> ## Holdout test of monthly CWMR (pre-registered)
>
> Written and committed before running. Hypothesis from the 2016-2024 study
> (`outputs/monthly_robustness.out`): the paper's CWMR, learning from daily prices but trading
> only at month end, beats SPY after realistic costs.
>
> - Strategy: lab `olps/cwmr_monthly` v1, unchanged (epsilon 0.89, theta 0.92, eta 0.93; 16 ETFs;
>   decide 15:30, fill 15:45 on the last session of each month).
> - Window: 2025-01-01 to the latest data (sessions through 2026-10-02). btest's holdout; no run
>   of any strategy has used it.
> - Settings: $20,000, 0.7 bp per dollar traded, fractional shares, idle cash at the T-bill rate,
>   benchmark SPY.
> - Pass: CAGR and Sharpe ratio both above SPY's over the same window. Also reported: the paired
>   Sharpe test against SPY (not expected to be significant over 21 months).
> - Run once. Whatever the outcome, the strategy is not modified and rerun on this window.
>
> Result: pending.
