"""Production median ranking and limit-order recommendations."""

from decimal import Decimal, ROUND_FLOOR

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DISCOUNT = 0.005
TAKE_PROFIT = 0.03
FEE = 0.0005
MIN_EDGE = 0.0025
HORIZON = 20
MAX_CANDIDATES = 10


def median_forecasts(frames, signal_date, horizon):
    cutoff = pd.Timestamp(signal_date).to_period("Q").start_time
    for ticker, frame in frames.items():
        close = frame["ticker"]
        if close.empty or close.index[-1] != signal_date:
            continue
        label_end = close.index.to_series().shift(-horizon)
        future_max = close.shift(-1).rolling(horizon).max().shift(-(horizon - 1))
        target = (future_max / close - 1) * 100
        training = target.loc[label_end < cutoff].dropna()
        if len(training) >= 120:
            yield ticker, float(training.median()), float(close.iloc[-1])


def expected_returns(candles, signal_date):
    cutoff = pd.Timestamp(signal_date).to_period("M").start_time
    sessions = sorted({day for frame in candles.values() for day in frame.index if day < cutoff})
    stop = len(sessions) - HORIZON - 1
    if stop <= 0:
        return {}
    history = slice(max(0, stop - 240), stop)
    samples = {}
    for ticker, frame in candles.items():
        low, high, close = frame.reindex(sessions)[["low", "high", "close"]].to_numpy().T
        reference = close[:stop]
        next_low = low[1:-HORIZON]
        windows = sliding_window_view(high[2:], HORIZON)
        future_max = np.where(np.isfinite(windows), windows, -np.inf).max(axis=1)
        expiry = close[HORIZON + 1:]
        valid = np.isfinite(reference) & (reference > 0) & np.isfinite(next_low) & np.isfinite(expiry)
        buy = np.where(valid, reference * (1 - DISCOUNT), np.nan)
        target = buy * (1 + TAKE_PROFIT)
        sale = np.where(future_max >= target, target, expiry)
        returns = np.where(next_low <= buy, sale / buy * (1 - FEE) - 1 - FEE, 0.0)
        past = returns[history][valid[history]]
        if len(past) >= 30:
            samples[ticker] = (len(past), float(past.sum()))
    if not samples:
        return {}
    prior = sum(total for _, total in samples.values()) / sum(n for n, _ in samples.values())
    return {ticker: (total + 30 * prior) / (n + 30) for ticker, (n, total) in samples.items()}


def board_details(get, tickers):
    remaining = set(tickers)
    found = {}
    offset = 0
    seen = set()
    while remaining:
        response = get(
            "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json",
            params={"iss.only": "securities", "securities.columns": "SECID,SECNAME,SHORTNAME,LOTSIZE,MINSTEP,CURRENCYID", "start": offset},
            timeout=20,
        )
        response.raise_for_status()
        block = response.json()["securities"]
        page = block["data"]
        if not page:
            break
        offset += len(page)
        new_page = False
        for row in page:
            item = dict(zip(block["columns"], row))
            ticker = item["SECID"]
            new_page |= ticker not in seen
            seen.add(ticker)
            if ticker not in remaining:
                continue
            remaining.remove(ticker)
            lot, tick = item.get("LOTSIZE"), item.get("MINSTEP")
            # Режим TQBR номинирован в рублях (код расчётов SUR или RUB).
            if lot and tick and lot > 0 and tick > 0 and item.get("CURRENCYID") in (None, "SUR", "RUB"):
                found[ticker] = (item.get("SECNAME") or item.get("SHORTNAME") or ticker, int(lot), str(tick))
        if not new_page:
            break
    return found


def recommendations(ranking, edge, details):
    result = []
    for ticker, median, reference in ranking[:MAX_CANDIDATES]:
        if edge.get(ticker, -np.inf) < MIN_EDGE or ticker not in details:
            continue
        name, lot, step = details[ticker]
        tick = Decimal(step)
        unrounded = Decimal(str(reference)) * Decimal("0.995")
        limit = (unrounded / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
        if limit <= 0:
            continue
        result.append({
            "ticker": ticker, "name": name, "lot_size": lot, "price_step": step,
            "buy_limit": str(limit), "reference_close": reference,
            "take_profit_pct": 3.0, "expected_order_return_pct": round(edge[ticker] * 100, 3),
            "median_growth_pct": round(median, 3),
        })
    return result
