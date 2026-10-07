"""Daily backtest with next-open fills.

On each decision session the strategy sees closes through that session and returns target
weights. Orders are sized at that close as dollar amounts (target weight times equity, minus
the position's value) and fill at the next session's open, which is the first price after the
decision. Sells go first; buys are scaled down if cash runs short. There is no leverage and no
short selling. Every fill pays a flat cost per dollar traded, sales also pay the SEC fee, and
idle cash earns the previous day's 3-month T-bill rate. Prices are dividend-adjusted, so
dividends need no separate handling.
"""
import bisect
import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

SEC_FEE_RATE = 20.6e-6
MIN_TRADE = 1.0


@dataclass
class Account:
    """What a strategy sees of the account at a decision: cash, equity at today's close and
    shares held."""
    cash: float
    equity: float
    positions: dict[str, float]


@dataclass
class Result:
    days: list[date]
    equity: np.ndarray
    traded: float
    fills: int
    fill_dates: list[date] = field(default_factory=list)


def decision_days(calendar: list[date], schedule: str) -> set[date]:
    """Sessions a schedule decides on. calendar must run past the backtest's last day so the
    last session of a week or month is known."""
    out = set()
    for k, d in enumerate(calendar[:-1]):
        nxt = calendar[k + 1]
        if (schedule == "daily" or (schedule == "weekly"
                                    and nxt.isocalendar()[1] != d.isocalendar()[1])
                or (schedule == "month_end" and nxt.month != d.month)):
            out.add(d)
    return out


def run(days, closes, opens, decide, decide_on: set[date], start: date, end: date,
        rates: tuple[list[date], list[float]], cost_bps: float, cash: float = 20_000.0) -> Result:
    symbols = list(closes)
    col = {s: k for k, s in enumerate(symbols)}
    close = np.column_stack([closes[s] for s in symbols])
    # The next session's open; the last session in the data falls back to its own close.
    fill = np.vstack([np.column_stack([opens[s] for s in symbols])[1:], close[-1:]])
    rate_dates, rate_values = rates
    slip = cost_bps / 10_000
    qty = np.zeros(len(symbols))
    traded, nfills, prev = 0.0, 0, None
    equity, fill_dates = [], []
    trading = [i for i, d in enumerate(days) if start <= d < end]
    for i in trading:
        d = days[i]
        if prev is not None:
            k = bisect.bisect_left(rate_dates, d)
            r = rate_values[k - 1] / 100 if k > 0 else 0.0
            cash += max(cash, 0.0) * r * (d - prev).days / 365
        if d in decide_on:
            held = {s for s in symbols if qty[col[s]]}
            eq = cash + sum(qty[col[s]] * close[i, col[s]] for s in held)
            target = decide(i, Account(cash, eq, {s: qty[col[s]] for s in held}))
            if target is not None:
                px = {s: close[i, col[s]] for s in set(target) | held
                      if not math.isnan(fill[i, col[s]])}
                exits, orders = set(), {}
                for s, p in px.items():
                    if s not in target:
                        exits.add(s)
                        continue
                    delta = target[s] * eq - qty[col[s]] * p
                    if abs(delta) >= MIN_TRADE:
                        orders[s] = delta
                for s in sorted(exits | {s for s, v in orders.items() if v < 0}):
                    f, have = fill[i, col[s]], qty[col[s]]
                    q = -have if s in exits else max(orders[s] / f, -have)
                    if q == 0:
                        continue
                    price = f * (1 - slip)
                    cash += abs(q) * price * (1 - SEC_FEE_RATE)
                    qty[col[s]] += q
                    traded += abs(q) * price
                    nfills += 1
                buys = {s: v for s, v in orders.items() if v > 0}
                need = sum(buys.values()) * (1 + slip)
                scale = min(1.0, max(cash, 0.0) / need) if need > 0 else 0.0
                for s, dollars in sorted(buys.items()):
                    price = fill[i, col[s]] * (1 + slip)
                    q = dollars * scale / price
                    if q <= 0 or q * price < MIN_TRADE:
                        continue
                    cash -= q * price
                    qty[col[s]] += q
                    traded += q * price
                    nfills += 1
                fill_dates.append(d)
        held = qty != 0
        equity.append(cash + float(qty[held] @ close[i, held]))
        prev = d
    return Result([days[i] for i in trading], np.array(equity), traded, nfills, fill_dates)
