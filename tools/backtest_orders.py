"""Evaluate next-session limit orders from archived Stockan forecasts."""

import argparse
import csv
import json
import math
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

MOSCOW = timezone(timedelta(hours=3))


def read_candles(path):
    candles = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            ticker, day = row["ticker"], date.fromisoformat(row["date"])
            low, high, close = (float(row[key]) for key in ("low", "high", "close"))
            if not all(math.isfinite(price) for price in (low, high, close)) or not 0 < low <= close <= high:
                raise ValueError(f"Invalid candle for {ticker} on {day}")
            if day in candles[ticker]:
                raise ValueError(f"Duplicate candle for {ticker} on {day}")
            candles[ticker][day] = (low, high, close)
    return {ticker: sorted(by_day.items()) for ticker, by_day in candles.items()}


def read_signals(directory):
    daily = {}
    for path in sorted(directory.rglob("latest_results.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(snapshot["timestamp"].replace("Z", "+00:00"))
        if timestamp.utcoffset() is None:
            raise ValueError(f"Expected a timestamp with a timezone in {path}")
        day = timestamp.astimezone(MOSCOW).date()
        previous = daily.get(day)
        if previous is None or timestamp > previous[0]:
            daily[day] = timestamp, snapshot

    for day, (_, snapshot) in sorted(daily.items()):
        forecasts = snapshot["medium_potential"] | snapshot["high_potential"]
        inputs = snapshot.get("prediction_inputs", {})
        for ticker, forecast in sorted(forecasts.items()):
            yield day, ticker, forecast, inputs.get(ticker)


def evaluate(day, inputs, candles, args):
    if inputs is None:
        return "missing_forecast_input", None, None, None
    if not candles:
        return "missing_candles", None, None, None
    source_day = date.fromisoformat(inputs["data_date"])
    position = bisect_right(candles, day, key=lambda candle: candle[0])
    if not position or candles[position - 1][0] != source_day:
        return "stale_or_unmatched_input", None, None, None
    reference = float(inputs["reference_close"])
    if not math.isfinite(reference) or reference <= 0:
        return "invalid_reference_price", None, None, None
    if not math.isclose(reference, candles[position - 1][1][2], rel_tol=1e-8):
        return "reference_close_mismatch", None, None, None
    entry_limit = reference * (1 - args.entry_discount / 100)
    exit_limit = entry_limit * (1 + args.take_profit / 100)
    fee = args.fee_bps / 10_000
    entry_window = candles[position:position + args.entry_sessions]
    for entry_index, (entry_day, (low, _, _)) in enumerate(entry_window, start=position):
        if low > entry_limit:
            continue
        exit_window = candles[entry_index + 1:entry_index + 1 + args.hold_sessions]
        for exit_day, (_, high, _) in exit_window:
            if high >= exit_limit:
                net = exit_limit * (1 - fee) / (entry_limit * (1 + fee)) - 1
                return "target_hit", entry_day, exit_day, 100 * net
        if len(exit_window) < args.hold_sessions:
            return "open_at_end", entry_day, None, None
        exit_day, (_, _, close) = exit_window[-1]
        net = close * (1 - fee) / (entry_limit * (1 + fee)) - 1
        return "time_exit", entry_day, exit_day, 100 * net
    status = "unfilled" if len(entry_window) == args.entry_sessions else "pending_at_end"
    return status, None, None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--entry-discount", type=float, default=0.5)
    parser.add_argument("--take-profit", type=float, default=3.0)
    parser.add_argument("--entry-sessions", type=int, default=1)
    parser.add_argument("--hold-sessions", type=int, default=10)
    parser.add_argument("--fee-bps", type=float, required=True, help="Commission per side in basis points")
    args = parser.parse_args()
    if not (0 <= args.entry_discount < 100 and math.isfinite(args.take_profit)
            and args.take_profit > 0 and 0 <= args.fee_bps < 10_000
            and min(args.entry_sessions, args.hold_sessions) >= 1):
        parser.error("Invalid order parameters")

    candles = read_candles(args.candles)
    signals = list(read_signals(args.snapshots))
    if not signals:
        parser.error("No signals in latest_results.json snapshots")

    counts = Counter()
    returns = []
    with args.trades.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(("signal_date", "ticker", "forecast_pct", "status", "entry_date", "exit_date", "net_return_pct"))
        for day, ticker, forecast, inputs in signals:
            status, entry, exit_day, net = evaluate(day, inputs, candles.get(ticker), args)
            counts[status] += 1
            if net is not None:
                returns.append(net)
            writer.writerow((day, ticker, forecast, status, entry, exit_day, net))
    print(json.dumps({
        "signals": len(signals),
        "outcomes": dict(counts),
        "target_rate_closed": counts["target_hit"] / len(returns) if returns else None,
        "mean_net_return_pct_closed": mean(returns) if returns else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
