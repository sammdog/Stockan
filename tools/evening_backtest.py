"""Compare evening limit orders using a common, latency-adjusted information cutoff."""

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import pandas as pd

from tools.trading_rules import order_statistics

FIRST_DAY = pd.Timestamp("2026-07-01")
FEE = 0.0005
CAPITAL = 50_000
ORDER_VALUE = 1000
POLICIES = ("previous_close", "evening_price", "previous_with_filter", "evening_with_filter")


@dataclass
class Holding:
    ticker: str
    quantity: float
    target: float
    last: float
    cost: float


def intraday(get, ticker, first, last):
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
    offset = 0
    seen = set()
    bars = defaultdict(list)
    while True:
        response = get(url, params={
            "from": first.date().isoformat(), "till": last.date().isoformat(),
            "interval": 10, "iss.only": "candles", "candles.columns": "begin,end,close,low",
            "start": offset,
        }, timeout=25)
        response.raise_for_status()
        block = response.json()["candles"]
        rows = block["data"]
        if not rows:
            break
        columns = [block["columns"].index(key) for key in ("begin", "end", "close", "low")]
        for row in rows:
            begin, end, close, low = (row[i] for i in columns)
            start = pd.Timestamp(begin)
            if start in seen:
                raise ValueError(f"Repeated candle page for {ticker} at {start}")
            seen.add(start)
            bars[start.normalize()].append((start.time(), pd.Timestamp(end).time(), float(close), float(low)))
        offset += len(rows)
    print(f"EVENING_SOURCE {ticker} {len(seen)} bars {len(bars)} sessions", flush=True)
    return bars


def information(bars):
    known = [bar for bar in bars if time(18, 10) <= bar[1] <= time(18, 40)]
    if not known:
        return None
    _, stamp, price, _ = known[-1]
    earlier = [bar for bar in bars if 30 <= (
        stamp.hour * 60 + stamp.minute - bar[1].hour * 60 - bar[1].minute
    ) <= 35]
    recent = price / earlier[-1][2] - 1 if earlier else None
    later = [bar[3] for bar in bars if bar[0] >= time(19)]
    return price, recent, min(later) if later else None


def simulate(opportunities, market, sessions, policy):
    use_evening = policy.startswith("evening")
    use_filter = policy.endswith("filter")
    cash = CAPITAL
    holdings = []
    counts = Counter()
    snapshots = []
    realized = 0
    for day in sessions:
        available = cash
        bars = market[day]
        remaining = []
        for h in holdings:
            bar = bars.get(h.ticker)
            if bar is not None and bar[1] >= h.target:
                proceeds = h.quantity * h.target * (1 - FEE)
                cash += proceeds
                realized += proceeds - h.cost
                counts["sold"] += 1
            else:
                if bar is not None:
                    h.last = bar[2]
                remaining.append(h)
        holdings = remaining
        for ticker, close, price, recent, later in opportunities.get(day, ()):
            counts["candidates"] += 1
            drop = price / close - 1 <= -.02 or (recent is not None and recent <= -.01)
            if use_filter and drop:
                counts["filtered"] += 1
                continue
            cost = ORDER_VALUE * (1 + FEE)
            if available < cost:
                counts["cash_skips"] += 1
                continue
            # The order was submitted even if its limit never gets touched.
            available -= cost
            buy = (price if use_evening else close) * .995
            if later is None or later > buy:
                counts["unfilled"] += 1
                continue
            cash -= cost
            counts["bought"] += 1
            holdings.append(Holding(ticker, ORDER_VALUE / buy, buy * 1.03, bars[ticker][2], cost))
        open_value = sum(h.quantity * h.last * (1 - FEE) for h in holdings)
        snapshots.append((day, cash + open_value))
    equity = pd.Series(dict(snapshots))
    month = equity.groupby(equity.index.to_period("M")).last()
    return {
        **counts,
        "net_pnl": round(float(equity.iloc[-1] - CAPITAL), 2),
        "drawdown_pct": round(float((equity / equity.cummax().clip(lower=CAPITAL) - 1).min() * 100), 3),
        "realized_pnl": round(realized, 2),
        "open_pnl": round(float(equity.iloc[-1] - CAPITAL - realized), 2),
        "open_positions": len(holdings),
        "monthly_pnl": {str(k): round(float(v), 2) for k, v in month.diff().fillna(month - CAPITAL).items()},
    }


def audit(predictions, candles, as_of, get):
    market = defaultdict(dict)
    for ticker, frame in candles.items():
        for day, low, high, close, opening in frame.itertuples(index=False, name=None):
            if day < pd.Timestamp(as_of) and 0 < low <= close <= high:
                market[day][ticker] = (low, high, close, opening)
    sessions = sorted(market)
    estimates = order_statistics(market, sessions, sorted(candles), FEE)
    ranks = {d: list(g.sort_values(["median", "ticker"], ascending=[False, True]).ticker.head(10))
             for d, g in predictions.groupby("date")}
    picks = {}
    for prior, day in zip(sessions, sessions[1:]):
        if day < FIRST_DAY or prior not in ranks:
            continue
        month = prior.to_period("M")
        picks[day] = [(t, market[prior][t][2]) for t in ranks[prior]
                      if t in market[prior] and estimates.get((month, .005, .03, t), -1) >= .0025]
    if not picks or not any(picks.values()):
        raise ValueError("No eligible evening candidates")
    tickers = sorted({t for rows in picks.values() for t, _ in rows})
    with ThreadPoolExecutor(max_workers=3) as pool:
        bars_by_ticker = dict(zip(tickers, pool.map(lambda t: intraday(get, t, min(picks), max(picks)), tickers)))
    opportunities = {}
    proposed = observed = has_later = 0
    for day, symbols in picks.items():
        rows = []
        for ticker, close in symbols:
            proposed += 1
            snapshot = information(bars_by_ticker[ticker].get(day, ()))
            if snapshot is not None and ticker in market[day]:
                observed += 1
                has_later += snapshot[2] is not None
                rows.append((ticker, close, *snapshot))
        opportunities[day] = rows
    if observed == 0:
        raise ValueError("No paired snapshots before the evening decision")
    print(f"EVENING_COVERAGE {min(picks).date()} to {max(picks).date()}, mark through {sessions[-1].date()}, "
          f"{len(tickers)} tickers, {proposed} candidates, {observed} snapshots, {has_later} with later bars", flush=True)
    reports = []
    for policy in POLICIES:
        result = simulate(opportunities, market, [d for d in sessions if d >= min(picks)], policy)
        reports.append({"policy": policy, **result})
        print("EVENING_RESULT " + json.dumps(reports[-1]), flush=True)
    Path("evening_summary.json").write_text(json.dumps(reports, indent=2) + "\n")
