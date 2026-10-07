# Replication code: mean-reversion portfolio selection on US-listed ETFs

Code for Gabo (2026), "Mean-Reversion Portfolio Selection on US-Listed ETFs, 1999-2026: A
Replication with Pre-Registered Tests". It reproduces the paper's tables from public data.

## Run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or later.

```sh
uv run python etf_tables.py                                    # Tables 5 to 8
uv run python autocorrelation.py                               # Section 8 autocorrelations
uv run python spreads.py                                       # Section 6.3 spread estimates
uv run --with scipy --with cvxpy python original_tables.py     # Section 3 and Table 3
```

The first run downloads Yahoo Finance daily bars and the FRED 3-month T-bill series into
`data/` (about a minute); later runs reuse that snapshot. `original_tables.py` clones Lahanis,
Liu and Zhou's repository at the version the paper tested (commit `7c2e88d`) and runs their
functions unchanged. The outputs we got are in `expected/`.

## What is here

| File | Contents |
|---|---|
| `cwmr_etf/cwmr.py` | The CWMR variant from the original authors' code, and the rule for symbols that start trading mid-sample |
| `cwmr_etf/backtest.py` | Daily backtest: decide on a session's close, fill at the next session's open, costs per dollar traded, SEC fee on sales, T-bill interest on cash |
| `cwmr_etf/data.py` | Yahoo daily bars (dividend-adjusted open and close), FRED DTB3, NYSE calendar |
| `cwmr_etf/stats.py` | CAGR, Sharpe ratio over T-bills, max drawdown, turnover, Jobson-Korkie test with Memmel's correction, block bootstrap |
| `etf_tables.py` | CWMR on weekly and monthly bars over 16 ETFs, 1999-2024, with SPY and equal-weight benchmarks, paired tests and a one-session-late variant |
| `autocorrelation.py` | Next-day autocorrelation of each ETF's return in excess of SPY's, by period |
| `spreads.py` | Yearly half-spreads estimated from daily prices (EDGE, via the authors' `bidask` package), checked against spreads measured from 2016-2024 quotes |
| `preregistrations/` | The three test plans, word for word as committed, with commit hashes and times |
| `original_tables.py` | The original paper's results from its own data and code |

## Not included

Sections 4 and 5 of the paper (daily trading and the 2025-2026 holdout) used minute bars from
Alpaca, which need an Alpaca account, and a backtester that trades at 15:45 New York time. Those
results are not reproduced here.

The Yahoo download uses the chart endpoint behind finance.yahoo.com, which is not an official
API; it may change or stop working, and Yahoo's terms limit the data to personal use. The data is
downloaded on each machine and not redistributed here.

Yahoo revises its data from time to time. Adjusted price ratios do not change when a new
dividend is paid, but if a download differs from ours the numbers may differ in the last digit.

## License

MIT. `cwmr_etf/cwmr.py` ports the CWMR function from Lahanis, Liu and Zhou's MIT-licensed code;
their notice is kept in `LICENSE`. `original_tables.py` fetches their repository and runs it
unchanged.
