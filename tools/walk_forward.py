"""Compare Stockan-style forecasts against a historical median by date."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMRegressor

FOLD_BOUNDARIES = pd.date_range("2025-10-01", "2027-01-01", freq="QS")
MIN_TRAIN_ROWS = 120


def historical_usd(start, end):
    """Get official USD rates by effective date, including valid carry-forward days."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    response = requests.get(
        "https://www.cbr.ru/scripts/XML_dynamic.asp",
        params={
            "date_req1": (start - pd.Timedelta(days=31)).strftime("%d/%m/%Y"),
            "date_req2": end.strftime("%d/%m/%Y"),
            "VAL_NM_RQ": "R01235",
        },
        timeout=30,
    )
    response.raise_for_status()
    rates = pd.Series({
        pd.to_datetime(record.attrib["Date"], format="%d.%m.%Y"):
        float(record.findtext("Value").replace(",", ".")) / int(record.findtext("Nominal"))
        for record in ET.fromstring(response.content).findall("Record")
    }, dtype=float).sort_index()
    if rates.empty or rates.index.min() > start or rates.index.max() < end - pd.Timedelta(days=10):
        raise ValueError("CBR history does not cover the requested dates")
    daily = rates.reindex(pd.date_range(start, end), method="ffill")
    print(f"WALK_FORWARD_USD {len(daily)} daily rates from {start.date()} to {end.date()}", flush=True)
    return daily.rename("usdt").rename_axis("begin").reset_index()


def refit_indices(sources, cutoff, end):
    indices = {}
    for sector, (features, target, history) in sources.items():
        train = features.index < cutoff
        if train.sum() < 30:
            continue
        model = LGBMRegressor(random_state=1, verbosity=-1, n_jobs=2)
        model.fit(features.loc[train], target.loc[train])
        known_history = history.loc[history.index < end]
        indices[sector] = pd.Series(model.predict(known_history), index=known_history.index)
    return indices


def dated_features(frame, original_dates, sector_index, market_index):
    features = frame.drop(columns="target").copy()
    features["sector_x"] = sector_index.reindex(frame.index)
    market = market_index.reindex(original_dates)
    features["sector_y"] = market.reindex(frame.index)
    for name in features.columns:
        if name.startswith("sector_y_lag_"):
            lag = int(name.rsplit("_", 1)[-1])
            features[name] = market.shift(lag).reindex(frame.index)
    if features.isna().any().any():
        raise ValueError("Missing point-in-time features after rebuilding indices")
    return features


def model_forecast(train, target, test, weighted):
    model = LGBMRegressor(random_state=1, n_estimators=200, learning_rate=0.1,
                          verbosity=-1, n_jobs=2)
    weights = np.where(target > 12, 2, 1) if weighted else None
    model.fit(train, target, sample_weight=weights)
    return model.predict(test)


def summarize(results):
    metrics = {}
    methods = ("median", "equal_weight", "double_tail_weight")
    for name in methods:
        residual = results[name] - results["actual"]
        high = results[name] >= 20
        top = (results.sort_values(["date", name, "ticker"], ascending=[True, False, True])
               .groupby("date", sort=False).head(10))
        metrics[name] = {
            "mae": round(float(residual.abs().mean()), 3),
            "bias": round(float(residual.mean()), 3),
            "forecast_ge20": int(high.sum()),
            "realized_ge20_when_forecast_ge20": int((results.loc[high, "actual"] >= 20).sum()),
            "top10_mean_realized": round(float(top["actual"].mean()), 3),
            "top10_realized_ge3_pct": round(float(100 * (top["actual"] >= 3).mean()), 3),
        }
    return metrics


def audit(data_frames, horizon, index_sources, index_dates):
    all_results = []
    for cutoff, end in zip(FOLD_BOUNDARIES[:-1], FOLD_BOUNDARIES[1:]):
        indices = refit_indices(index_sources, cutoff, end)
        fold = []
        for sector, tickers in data_frames.items():
            if sector not in indices or "IMOEX2" not in indices:
                continue
            for ticker, frame in tickers.items():
                dates = frame.index.to_series()
                label_end = dates.shift(-horizon)
                train = label_end < cutoff
                # Leave the newest source row out because it can be an unfinished candle.
                test = (dates >= cutoff) & (dates < end) & (label_end < dates.iloc[-1])
                if train.sum() < MIN_TRAIN_ROWS or not test.any():
                    continue
                selected = frame.loc[train | test]
                features = dated_features(selected, index_dates[sector, ticker],
                                          indices[sector], indices["IMOEX2"])
                y = frame.loc[train, "target"]
                x_train, x_test = features.loc[train.loc[features.index]], features.loc[test.loc[features.index]]
                fold.append(pd.DataFrame({
                    "date": x_test.index,
                    "ticker": ticker,
                    "actual": frame.loc[test, "target"].to_numpy(),
                    "median": y.median(),
                    "equal_weight": model_forecast(x_train, y, x_test, weighted=False),
                    "double_tail_weight": model_forecast(x_train, y, x_test, weighted=True),
                }))
        if fold:
            results = pd.concat(fold, ignore_index=True)
            all_results.append(results)
            print_report(cutoff.date().isoformat(), results)
    if not all_results:
        raise ValueError("No complete walk-forward folds")
    results = pd.concat(all_results, ignore_index=True)
    print_report("all", results)
    stress = results.loc[results["date"].between("2026-03-01", "2026-05-31")]
    if not stress.empty:
        print_report("mar-may-2026", stress)
    results.to_csv("walk_forward_predictions.csv", index=False)


def print_report(period, results):
    print("WALK_FORWARD " + json.dumps({
        "period": period,
        "rows": len(results),
        "tickers": int(results["ticker"].nunique()),
        "days": int(results["date"].nunique()),
        "actual_ge20": int((results["actual"] >= 20).sum()),
        "metrics": summarize(results),
    }), flush=True)
