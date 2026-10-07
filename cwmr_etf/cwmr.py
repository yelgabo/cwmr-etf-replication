"""The CWMR variant in Lahanis, Liu and Zhou's code (cwmr() in Strategies/follow_the_loser.py),
and the rule that turns it into a strategy over a universe whose members start trading at
different times.
"""
import copy
from datetime import date

import numpy as np

EPS = 1e-15


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto non-negative weights that sum to one."""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u > cssv / np.arange(1, len(v) + 1))[0][-1]
    return np.maximum(v - cssv[rho] / (rho + 1.0), 0)


class CWMR:
    """Weights b and covariance sigma. update(x) takes one period's price ratios."""

    def __init__(self, n: int, epsilon=0.89, theta=0.92, eta=0.93):
        self.n, self.epsilon, self.theta, self.eta = n, epsilon, theta, eta
        self.b = np.full(n, 1.0 / n)
        self.sigma = np.eye(n)

    def update(self, x: np.ndarray) -> None:
        mean = np.dot(self.b, x)
        denom = x @ (self.sigma @ x)
        lam = self.eta * max(0, (mean - self.epsilon) / (denom + EPS)) if denom > 0 else 0.0
        mu = self.b - lam * (self.sigma @ x)
        inv = np.linalg.inv(self.sigma + np.eye(self.n) * 1e-12)
        inv += 2 * lam * self.theta * np.outer(x, x)
        self.sigma = np.linalg.inv(inv)
        self.b = project_to_simplex(mu)


def _period(bars: str):
    return {"weekly": lambda d: d.isocalendar()[:2], "monthly": lambda d: (d.year, d.month)}[bars]


class OnlineCWMR:
    """CWMR over a fixed universe. At a decision on session i it has seen every completed bar
    since the first session in the data (daily, weekly or monthly closes) plus today's close.

    A symbol with no price ratio yet (not trading) gets the mean ratio of those that have one.
    Holding it is then the same as holding the trading symbols in equal parts, so its weight
    is split equally among them. That lets the algorithm run on the whole universe from the
    first bar and keep its state when a symbol starts trading.
    """

    def __init__(self, symbols, closes, days: list[date], bars="daily", block=None):
        self.symbols = symbols
        self.c = np.column_stack([closes[s] for s in symbols])
        self.days, self.bars, self.block = days, bars, block
        self.algo, self.done, self.first = CWMR(len(symbols)), 0, None
        if bars != "daily" and not block:
            keys = [_period(bars)(d) for d in days]
            self.ends = [k for k in range(len(days) - 1) if keys[k + 1] != keys[k]]

    def _rows(self, i: int) -> list[int]:
        if self.block:
            return list(range(i % self.block, i + 1, self.block))
        if self.bars == "daily":
            return list(range(i + 1))
        # The last session of each completed period, then today.
        return [k for k in self.ends if k < i] + [i]

    def weights(self, i: int) -> dict[str, float] | None:
        rows = self.c[self._rows(i)]
        with np.errstate(invalid="ignore"):
            rel = rows[1:] / rows[:-1]
        ok = np.isfinite(rel)
        first = int(np.argmax(ok.any(axis=1))) if ok.any() else len(rel)
        rel, ok = rel[first:], ok[first:]
        if not len(rel) or ok[-1].sum() < 2:
            return None
        mean = np.nanmean(np.where(ok, rel, np.nan), axis=1, keepdims=True)
        rel = np.where(ok, rel, mean)
        done = rel[:-1]
        if self.first != first or self.done > len(done):
            self.algo, self.done, self.first = CWMR(len(self.symbols)), 0, first
        for x in done[self.done:]:
            self.algo.update(x)
        self.done = len(done)
        today = copy.deepcopy(self.algo)
        today.update(rel[-1])
        b, live = today.b, ok[-1]
        w = np.where(live, b, 0.0) + live * b[~live].sum() / live.sum()
        return {s: float(x) for s, x, t in zip(self.symbols, w, live) if t and x > 1e-6}
