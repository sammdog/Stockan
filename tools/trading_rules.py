"""Chronological sensitivity tests for manual Stockan trading rules."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from tools.portfolio_backtest import simulate

SWITCH = pd.Timestamp("2026-06-01")
HORIZON = 20
DISCOUNTS = (0.0025, 0.005, 0.01, 0.015)
TARGETS = (0.02, 0.03, 0.04, 0.05)


@dataclass(frozen=True)
class Rules:
    cooldown: int = 0
    exposure: float = 1.0
    ev_threshold: float | None = None
    rank_expected: bool = False
    min_candidates: int = 0
    discount: float = 0.005
    take_profit: float = 0.03
    adaptive_entry: bool = False
    adaptive_target: bool = False
    review_target: int = 0
    max_age: int = 0
    stop: float = 0
    weak_exit: bool = False
    market: str = ""
    market_exit: float = 0
    downside_exit: float = 0


def variants():
    return {
        "cooldown": {f"cooldown_{n}": {"cooldown": n} for n in (10, 20, 30)},
        "budget": {f"exposure_{n}": {"exposure": n / 100} for n in (50, 70)},
        "ev": {
            **{f"ev_min_{n}": {"ev_threshold": n / 100} for n in (0, 0.25, 0.5, 1)},
            "ev_rank_min_0": {"rank_expected": True, "ev_threshold": 0.0},
            "ev_rank_min_0.5": {"rank_expected": True, "ev_threshold": 0.005},
            "ev_rank_at_least_3": {"rank_expected": True, "ev_threshold": 0.005, "min_candidates": 3},
        },
        "entry": {
            **{f"entry_{n}bp": {"discount": n / 10_000} for n in (25, 100, 150)},
            "entry_adaptive": {"adaptive_entry": True},
        },
        "target": {
            **{f"target_{n}pct": {"take_profit": n / 100} for n in (2, 4, 5)},
            "target_per_ticker": {"adaptive_target": True},
            "target_review_10": {"adaptive_target": True, "review_target": 10},
        },
        "time_exit": {f"time_{n}": {"max_age": n} for n in (15, 30, 60)},
        "risk_exit": {
            **{f"stop_{n}pct": {"stop": n / 100} for n in (5, 8, 12)},
            "weak_relative_exit": {"weak_exit": True},
        },
        "market": {
            "market_half_20": {"market": "half20"},
            "market_skip_20": {"market": "skip20"},
            "market_half_50": {"market": "half50"},
            "market_skip_5pct": {"market": "skip5pct"},
            "half_every_day": {"market": "half_always"},
        },
    }


def order_statistics(market, sessions, tickers, fee):
    """Estimate net return per submitted order using only resolved past orders."""
    keys = [(entry, target) for entry in DISCOUNTS for target in TARGETS]
    months = pd.DatetimeIndex(sessions).to_period("M").unique()
    stats = {}
    for ticker in tickers:
        bars = np.array([market[day].get(ticker, (np.nan,) * 4) for day in sessions])
        if len(bars) <= HORIZON + 1:
            continue
        close = bars[:-(HORIZON + 1), 2]
        next_low = bars[1:-HORIZON, 0]
        high_windows = sliding_window_view(bars[2:, 1], HORIZON)
        high = np.where(np.isfinite(high_windows), high_windows, -np.inf).max(axis=1)
        expiry = bars[HORIZON + 1:, 2]
        valid = np.isfinite(close) & np.isfinite(next_low) & np.isfinite(expiry)
        for entry, target in keys:
            buy = close * (1 - entry)
            sell = np.where(high >= buy * (1 + target), buy * (1 + target), expiry)
            pnl = np.where(next_low <= buy, sell / buy * (1 - fee) - 1 - fee, 0.0)
            pnl[~valid] = np.nan
            for month in months:
                stop = np.searchsorted(sessions, month.start_time) - HORIZON - 1
                if stop <= 0:
                    continue
                past = pnl[max(0, stop - 240):stop]
                n = np.isfinite(past).sum()
                if n >= 30:
                    stats.setdefault((month, entry, target), {})[ticker] = (n, float(np.nansum(past)))
    estimates = {}
    for key, by_ticker in stats.items():
        total_n = sum(n for n, _ in by_ticker.values())
        total_value = sum(total for _, total in by_ticker.values())
        prior = total_value / total_n
        for ticker, (n, total) in by_ticker.items():
            estimates[(*key, ticker)] = (total + 30 * prior) / (n + 30)
    return estimates


def market_context(index, sessions):
    close = index.set_index("begin")["close"].sort_index()
    # If this market session has no index value, use the last observed index value.
    returns = close.pct_change(20).reindex(sessions, method="ffill")
    long_returns = close.pct_change(50).reindex(sessions, method="ffill")
    return {
        day: (float(returns.loc[day]), float(long_returns.loc[day]))
        for day in sessions
    }


def downside_probabilities(market, sessions, tickers, contexts):
    """Forecast a further 5% close decline over twenty sessions."""
    calendar = pd.DatetimeIndex(sessions)
    regime = np.array([contexts[day][0] < 0 for day in calendar])
    months = calendar.to_period("M")
    histories = {}
    for ticker in tickers:
        close = pd.Series([market[day].get(ticker, (None, None, np.nan))[2]
                           for day in calendar], index=calendar, dtype=float)
        momentum = close / close.shift(20) - 1
        bucket = np.digitize(momentum.to_numpy(), [-0.1, 0, 0.1]) + 4 * regime
        future = close.shift(-20) / close - 1
        histories[ticker] = bucket, (future <= -0.05).to_numpy(), (future.notna() & momentum.notna()).to_numpy(), momentum.notna().to_numpy()
    probabilities = {}
    for month in months.unique():
        end = calendar.searchsorted(month.start_time) - 20
        if end < 60:
            continue
        start = max(0, end - 240)
        summaries = {}
        total_count = np.zeros(8)
        total_hit = np.zeros(8)
        for ticker, (bucket, sell, valid, known) in histories.items():
            valid = valid[start:end]
            states = bucket[start:end][valid]
            counts = np.bincount(states, minlength=8)[:8]
            hits = np.bincount(states, weights=sell[start:end][valid], minlength=8)[:8]
            summaries[ticker] = counts, hits
            total_count += counts
            total_hit += hits
        prior = (total_hit + 1) / (total_count + 2)
        for ticker, (bucket, _, _, known) in histories.items():
            counts, hits = summaries[ticker]
            rates = (hits + 30 * prior) / (counts + 30)
            for i in np.flatnonzero((months == month) & known):
                probabilities[(calendar[i], ticker)] = float(rates[bucket[i]])
    return probabilities


def audit_rules(predictions, market, sessions, index):
    contexts = market_context(index, sessions)
    tickers = sorted(predictions.ticker.unique())
    estimates = order_statistics(market, sessions, tickers, 0.0005)
    downside = downside_probabilities(market, sessions, tickers, contexts)
    if not estimates or not downside:
        raise ValueError("Insufficient history for exit rules")
    ranks = predictions.sort_values(["date", "median", "ticker"], ascending=[True, False, True])
    base = Rules(ev_threshold=0.0025)
    choices = {
        "baseline": base,
        "stop_5": replace(base, stop=0.05),
        "stop_8": replace(base, stop=0.08),
        "stop_12": replace(base, stop=0.12),
        "time_30": replace(base, max_age=30),
        "market_exit_5": replace(base, market_exit=0.05),
        "market_exit_10": replace(base, market_exit=0.1),
        "downside_45": replace(base, downside_exit=0.45),
        "downside_60": replace(base, downside_exit=0.6),
        "market_5_and_downside_45": replace(base, market_exit=0.05, downside_exit=0.45),
    }
    periods = {
        "full": (ranks, sessions),
        "selection": (ranks[ranks.date < SWITCH], [d for d in sessions if d < SWITCH]),
        "from_june": (ranks[ranks.date >= SWITCH], sessions),
    }
    reports = []
    for period, (source, calendar) in periods.items():
        for name, rules in choices.items():
            report, _, _ = simulate(source, market, calendar, 50_000, 5,
                                    rules=rules, estimates=estimates, contexts=contexts,
                                    cadence=1, downside=downside)
            row = {"period": period, "rule": name, **report}
            print("EXIT_RISK " + json.dumps(row), flush=True)
            reports.append(row)
    Path("exit_risk_summary.json").write_text(json.dumps(reports, indent=2) + "\n")
