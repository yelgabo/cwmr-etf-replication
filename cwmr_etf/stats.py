"""Performance measures and the paired Sharpe ratio test."""
import bisect
import math
from datetime import date

import numpy as np

TRADING_DAYS = 252


def cagr(days: list[date], equity: np.ndarray) -> float:
    years = (days[-1] - days[0]).days / 365.25
    return (equity[-1] / equity[0]) ** (1 / years) - 1


def sharpe(days: list[date], equity: np.ndarray, rates) -> float:
    """Annualised mean daily return in excess of the T-bill rate, over the daily volatility."""
    rate_dates, rate_values = rates
    ret = equity[1:] / equity[:-1] - 1
    rf = []
    for d in days[1:]:
        k = bisect.bisect_right(rate_dates, d)
        rf.append(rate_values[k - 1] / 100 / TRADING_DAYS if k > 0 else 0.0)
    return float((ret - np.array(rf)).mean() / ret.std(ddof=1) * math.sqrt(TRADING_DAYS))


def max_drawdown(equity: np.ndarray) -> float:
    return float((equity / np.maximum.accumulate(equity) - 1).min())


def turnover(days: list[date], equity: np.ndarray, traded: float) -> float:
    """Dollars bought and sold per year over average equity."""
    return traded / equity.mean() / ((days[-1] - days[0]).days / 365.25)


def paired_sharpe_test(eq1: np.ndarray, eq2: np.ndarray) -> tuple[float, float]:
    """Jobson-Korkie test with Memmel's correction on daily returns over the same days.
    Returns the annualised Sharpe difference and the two-sided p-value."""
    r1, r2 = eq1[1:] / eq1[:-1] - 1, eq2[1:] / eq2[:-1] - 1
    s1, s2 = r1.mean() / r1.std(), r2.mean() / r2.std()
    rho = np.corrcoef(r1, r2)[0, 1]
    var = (2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / len(r1)
    z = (s1 - s2) / math.sqrt(var)
    return (s1 - s2) * math.sqrt(TRADING_DAYS), 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def window(days: list[date], equity: np.ndarray, lo: date, hi: date):
    """The part of one run between lo and hi, starting from the equity on the session before."""
    ks = [k for k, d in enumerate(days) if lo <= d < hi]
    if ks[0] > 0:
        ks = [ks[0] - 1] + ks
    return [days[k] for k in ks], equity[ks]
