"""Chronological sensitivity tests for manual Stockan trading rules."""

import json
from dataclasses import asdict, dataclass, replace
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


def audit_rules(predictions, market, sessions, index):
    contexts = market_context(index, sessions)
    estimates = order_statistics(market, sessions, sorted(predictions.ticker.unique()), 0.0005)
    if not estimates:
        raise ValueError("No resolved history for estimating order returns")
    methods = ("median", "relative_l1", "pooled_relative_l1", "double_tail_weight")
    candidates = variants()
    rows = []
    chosen = []

    def run(ranks, rules, period, phase):
        if period == "selection":
            source = ranks[ranks.date < SWITCH]
            calendar = [day for day in sessions if day < SWITCH]
        elif period == "validation":
            source = ranks[ranks.date >= SWITCH]
            calendar = sessions
        else:
            source, calendar = ranks, sessions
        return simulate(source, market, calendar, 50_000, 5, phase, rules, estimates, contexts)[0]

    def record(ranks, method, group, name, rules, period, phase=0):
        summary = run(ranks, rules, period, phase)
        row = {"method": method, "group": group, "variant": name, "period": period,
               "phase": phase, "rules": asdict(rules), **summary}
        rows.append(row)
        return summary

    for method in methods:
        ranks = predictions.sort_values(["date", method, "ticker"], ascending=[True, False, True])
        baseline = Rules()
        base_score = record(ranks, method, "baseline", "baseline", baseline, "selection")["net_pnl"]
        record(ranks, method, "baseline", "baseline", baseline, "full")
        for phase in (0, 1):
            record(ranks, method, "baseline", "baseline", baseline, "validation", phase)
        winners = []
        for group, options in candidates.items():
            results = []
            for name, configuration in options.items():
                rules = replace(baseline, **configuration)
                results.append((record(ranks, method, group, name, rules, "selection")["net_pnl"],
                                name, configuration))
                record(ranks, method, group, name, rules, "full")
                record(ranks, method, group, name, rules, "validation")
            best = max(results, key=lambda result: result[0])
            if best[0] > base_score:
                winners.append((best[0], group, best[1], best[2]))
        combined = baseline
        score = base_score
        accepted = []
        # Prefer the largest individual selection improvement, keeping a rule
        # only where its incremental contribution is positive on selection.
        for _, group, name, changes in sorted(winners, key=lambda item: -item[0]):
            trial = replace(combined, **changes)
            trial_score = run(ranks, trial, "selection", 0)["net_pnl"]
            if trial_score > score:
                score = trial_score
                combined = trial
                accepted.append(name)
        chosen.append({"method": method, "included": accepted, "rules": asdict(combined)})
        for period in ("selection", "full"):
            record(ranks, method, "combined", "combined", combined, period)
        for phase in (0, 1):
            record(ranks, method, "combined", "combined", combined, "validation", phase)

    Path("trading_rules_summary.json").write_text(json.dumps({"selection_end": str(SWITCH.date()), "chosen": chosen, "runs": rows}, indent=2) + "\n")
    details = []
    for row in rows:
        details.extend({"method": row["method"], "group": row["group"], "variant": row["variant"],
                        "period": row["period"], "phase": row["phase"], "month": month, "pnl": pnl}
                       for month, pnl in row["monthly_pnl"].items())
        if row["group"] in {"baseline", "combined"} or (row["period"] == "full" and row["method"] == "median"):
            message = {k: row[k] for k in ("method", "group", "variant", "period", "phase",
                                           "net_pnl", "max_drawdown_pct", "filled", "sold",
                                           "rule_skips", "risk_exit_count", "open_cost")}
            if row["group"] in {"baseline", "combined"}:
                message["monthly_pnl"] = row["monthly_pnl"]
            print("TRADING_RULE " + json.dumps(message), flush=True)
    for item in chosen:
        print("TRADING_CHOSEN " + json.dumps(item), flush=True)
    pd.DataFrame(details).to_csv("trading_rules_monthly.csv", index=False)
