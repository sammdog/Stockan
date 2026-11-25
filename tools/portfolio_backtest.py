"""Compare Stockan rankings using a cash-constrained limit-order simulation."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("double_tail_weight", "median", "relative_l1", "pooled_relative_l1")
ORDER_VALUE = 1000
ENTRY_FACTOR = 0.995
EXIT_FACTOR = 1.03


@dataclass
class Position:
    ticker: str
    opened: pd.Timestamp
    quantity: float
    cost: float
    target: float
    close: float


def simulate(ranks, market, sessions, capital, fee_bps, phase=0):
    fee = fee_bps / 10_000
    signals = {
        day: list(group["ticker"].head(10))
        for day, group in ranks.groupby("date", sort=True)
    }
    session_number = {day: i for i, day in enumerate(sessions)}
    first, last = min(signals), max(signals)
    eligible = [day for day in sessions if first <= day <= last]
    schedule = set(eligible[phase::2])
    cash = capital
    positions = []
    queued = []
    counts = dict(signals=0, filled=0, expired=0, cash_skips=0, missing=0, sold=0, sold_within_14_days=0)
    realized = 0.0
    trades, equity = [], []
    for day in sessions[session_number[first]:]:
        bars = market.get(day, {})
        buying_power = cash
        remaining = []
        for position in positions:
            bar = bars.get(position.ticker)
            if bar is not None and bar[1] >= position.target:
                proceeds = position.quantity * position.target * (1 - fee)
                cash += proceeds
                realized += proceeds - position.cost
                counts["sold"] += 1
                counts["sold_within_14_days"] += (day - position.opened).days <= 14
                trades.append((day, position.ticker, "sell", position.target, proceeds, proceeds - position.cost))
            else:
                if bar is not None:
                    position.close = bar[2]
                remaining.append(position)
        positions = remaining

        for ticker, buy_limit in queued:
            bar = bars.get(ticker)
            cost = ORDER_VALUE * (1 + fee)
            if bar is None or bar[0] > buy_limit:
                counts["expired"] += 1
                continue
            if buying_power + 1e-8 < cost:
                counts["cash_skips"] += 1
                continue
            cash -= cost
            buying_power -= cost
            positions.append(Position(ticker, day, ORDER_VALUE / buy_limit, cost, buy_limit * EXIT_FACTOR, bar[2]))
            counts["filled"] += 1
            trades.append((day, ticker, "buy", buy_limit, -cost, None))

        queued = []
        if day in schedule and day in signals:
            for ticker in signals[day]:
                counts["signals"] += 1
                bar = bars.get(ticker)
                if bar is None:
                    counts["missing"] += 1
                else:
                    queued.append((ticker, bar[2] * ENTRY_FACTOR))

        value = sum(position.quantity * position.close * (1 - fee) for position in positions)
        open_cost = sum(position.cost for position in positions)
        equity.append((day, cash + value, cash, value, open_cost, len(positions)))

    daily = pd.DataFrame(equity, columns=["date", "equity", "cash", "open_value", "open_cost", "open_positions"])
    last = daily.iloc[-1]
    peaks = daily["equity"].cummax().clip(lower=capital)
    drawdown = (daily["equity"] / peaks - 1).min()
    summary = {
        **counts,
        "net_pnl": round(float(last.equity - capital), 2),
        "return_pct": round(float((last.equity / capital - 1) * 100), 3),
        "realized_pnl": round(realized, 2),
        "open_pnl": round(float(last.open_value - last.open_cost), 2),
        "open_cost": round(float(last.open_cost), 2),
        "open_positions": int(last.open_positions),
        "max_drawdown_pct": round(float(drawdown * 100), 3),
        "peak_open_cost": round(float(daily["open_cost"].max()), 2),
    }
    by_month = daily.groupby(daily["date"].dt.to_period("M"))["equity"].last()
    summary["monthly_pnl"] = {
        str(month): round(float(delta), 2)
        for month, delta in (by_month - by_month.shift(fill_value=capital)).items()
    }
    assert counts["signals"] == counts["filled"] + counts["expired"] + counts["cash_skips"] + counts["missing"]
    assert abs(summary["net_pnl"] - summary["realized_pnl"] - summary["open_pnl"]) <= 0.021
    return summary, daily, pd.DataFrame(trades, columns=["date", "ticker", "side", "price", "cash_delta", "realized_pnl"])


def audit_portfolios(predictions, candles, as_of):
    market = {}
    latest_completed = pd.Timestamp(as_of)
    for ticker, frame in candles.items():
        for day, low, high, close in frame.itertuples(index=False, name=None):
            if day >= latest_completed:
                continue
            if np.isfinite([low, high, close]).all() and 0 < low <= close <= high:
                market.setdefault(day, {})[ticker] = (low, high, close)
    sessions = sorted(market)
    scenarios = [(50_000, 5, 0), (100_000, 5, 0), (50_000, 20, 0), (50_000, 5, 1)]
    reports = []
    all_daily, all_trades = [], []
    for method in METHODS:
        ranks = predictions.sort_values(["date", method, "ticker"], ascending=[True, False, True])
        for capital, bps, phase in scenarios:
            result, daily, trades = simulate(ranks, market, sessions, capital, bps, phase)
            identity = {"method": method, "capital": capital, "fee_bps": bps, "phase": phase}
            report = {**identity, **result}
            reports.append(report)
            print("PORTFOLIO " + json.dumps(report), flush=True)
            all_daily.append(daily.assign(**identity))
            all_trades.append(trades.assign(**identity))
    print(f"PORTFOLIO_PERIOD signals {predictions.date.min().date()} to {predictions.date.max().date()}, "
          f"positions marked through {sessions[-1].date()}", flush=True)
    Path("portfolio_summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    pd.concat(all_daily, ignore_index=True).to_csv("portfolio_equity.csv", index=False)
    pd.concat(all_trades, ignore_index=True).to_csv("portfolio_trades.csv", index=False)
