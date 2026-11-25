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
    opened_session: int = 0
    risk_pending: bool = False
    cooldown_seen: bool = False
    last_review: int = 0


def simulate(ranks, market, sessions, capital, fee_bps, phase=0, rules=None, estimates=None, contexts=None):
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
    cooldown_until = {}
    positions = []
    queued = []
    counts = dict(signals=0, filled=0, expired=0, cash_skips=0, missing=0, sold=0, sold_within_14_days=0)
    counts.update(rule_skips=0, risk_exit_count=0)
    realized = 0.0
    trades, equity = [], []
    for day in sessions[session_number[first]:]:
        bars = market.get(day, {})
        ordinal = session_number[day]
        month = day.to_period("M")
        trend20, trend50 = contexts.get(day, (np.nan, np.nan)) if contexts else (np.nan, np.nan)
        remaining = []
        for position in positions:
            bar = bars.get(position.ticker)
            if bar is not None and (position.risk_pending or bar[1] >= position.target):
                price = bar[3] if position.risk_pending else position.target
                proceeds = position.quantity * price * (1 - fee)
                cash += proceeds
                realized += proceeds - position.cost
                counts["sold"] += 1
                counts["sold_within_14_days"] += (day - position.opened).days <= 14
                counts["risk_exit_count"] += position.risk_pending
                trades.append((day, position.ticker, "sell", price, proceeds, proceeds - position.cost))
            else:
                if bar is not None:
                    position.close = bar[2]
                remaining.append(position)
        positions = remaining

        for ticker, buy_limit, target, amount in queued:
            bar = bars.get(ticker)
            if bar is None or bar[0] > buy_limit:
                counts["expired"] += 1
                continue
            cost = amount * (1 + fee)
            cash -= cost
            positions.append(Position(ticker, day, amount / buy_limit, cost,
                                      buy_limit * (1 + target), bar[2], opened_session=ordinal,
                                      last_review=ordinal))
            counts["filled"] += 1
            trades.append((day, ticker, "buy", buy_limit, -cost, None))

        if rules is not None:
            for position in positions:
                if position.ticker not in bars:
                    continue
                entry_price = position.cost / (position.quantity * (1 + fee))
                drawdown = position.close / entry_price - 1
                age = ordinal - position.opened_session
                if rules.cooldown and drawdown <= -0.08 and not position.cooldown_seen:
                    cooldown_until[position.ticker] = max(
                        ordinal + rules.cooldown, cooldown_until.get(position.ticker, -1)
                    )
                    position.cooldown_seen = True
                if ((rules.stop and drawdown <= -rules.stop)
                        or (rules.max_age and age >= rules.max_age)
                        or (rules.weak_exit and age >= 10 and drawdown <= -0.05 and trend20 <= -0.05)):
                    position.risk_pending = True
                if rules.review_target and ordinal - position.last_review >= rules.review_target:
                    options = [(estimates.get((month, rules.discount, target, position.ticker), -np.inf), target)
                               for target in (0.02, 0.03, 0.04, 0.05)]
                    best, target = max(options)
                    if np.isfinite(best):
                        position.target = entry_price * (1 + target)
                    position.last_review = ordinal

        value = sum(position.quantity * position.close * (1 - fee) for position in positions)
        open_cost = sum(position.cost for position in positions)
        queued = []
        reserved = 0.0
        if day in schedule and day in signals:
            for ticker in signals[day]:
                counts["signals"] += 1
                bar = bars.get(ticker)
                if bar is None:
                    counts["missing"] += 1
                    continue
                amount = ORDER_VALUE
                discount, target = 1 - ENTRY_FACTOR, EXIT_FACTOR - 1
                if rules is not None:
                    discount, target = rules.discount, rules.take_profit
                    entries = (0.0025, 0.005, 0.01, 0.015) if rules.adaptive_entry else (discount,)
                    targets = (0.02, 0.03, 0.04, 0.05) if rules.adaptive_target else (target,)
                    expected, chosen_entry, chosen_target = max(
                        (estimates.get((month, e, t, ticker), -np.inf), e, t)
                        for e in entries for t in targets
                    )
                    if np.isfinite(expected):
                        discount, target = chosen_entry, chosen_target
                    if (ordinal < cooldown_until.get(ticker, -1)
                            or (rules.ev_threshold is not None and expected < rules.ev_threshold)
                            or (rules.market == "skip20" and trend20 < 0)
                            or (rules.market == "skip5pct" and trend20 < -0.05)):
                        counts["rule_skips"] += 1
                        continue
                    if (rules.market == "half_always" or (rules.market == "half20" and trend20 < 0)
                            or (rules.market == "half50" and trend50 < 0)):
                        amount /= 2
                cost = amount * (1 + fee)
                if (rules is not None and rules.exposure < 1
                        and value + reserved + cost > rules.exposure * (cash + value)):
                    counts["rule_skips"] += 1
                    continue
                if cash + 1e-8 < reserved + cost:
                    counts["cash_skips"] += 1
                    continue
                queued.append((ticker, bar[2] * (1 - discount), target, amount))
                reserved += cost
        equity.append((day, cash + value, cash, value, open_cost, len(positions)))

    daily = pd.DataFrame(equity, columns=["date", "equity", "cash", "open_value", "open_cost", "open_positions"])
    last = daily.iloc[-1]
    peaks = daily["equity"].cummax().clip(lower=capital)
    drawdown = (daily["equity"] / peaks - 1).min()
    summary = {
        **counts,
        "pending": len(queued),
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
    assert counts["signals"] == counts["filled"] + counts["expired"] + counts["cash_skips"] + counts["missing"] + counts["rule_skips"] + len(queued)
    assert abs(summary["net_pnl"] - summary["realized_pnl"] - summary["open_pnl"]) <= 0.021
    return summary, daily, pd.DataFrame(trades, columns=["date", "ticker", "side", "price", "cash_delta", "realized_pnl"])


def audit_portfolios(predictions, candles, as_of, index):
    market = {}
    latest_completed = pd.Timestamp(as_of)
    for ticker, frame in candles.items():
        for day, low, high, close, opening in frame.itertuples(index=False, name=None):
            if day >= latest_completed:
                continue
            if np.isfinite([low, high, close, opening]).all() and 0 < low <= close <= high and low <= opening <= high:
                market.setdefault(day, {})[ticker] = (low, high, close, opening)
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

    from tools.trading_rules import audit_rules

    audit_rules(predictions, market, sessions, index)
