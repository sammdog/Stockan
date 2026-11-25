"""Compare Stockan-style forecasts against a historical median by date."""

import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

FOLD_BOUNDARIES = pd.date_range("2025-10-01", "2027-01-01", freq="QS")
MIN_TRAIN_ROWS = 120


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


MODEL_PARAMETERS = {
    "equal_weight": {},
    "double_tail_weight": {},
    "l1": {"objective": "regression_l1"},
    "compact_l1": {
        "objective": "regression_l1", "num_leaves": 7, "max_depth": 3,
        "min_child_samples": 60, "reg_lambda": 3, "n_estimators": 100,
        "learning_rate": 0.05,
    },
}
METHODS = ("median", *MODEL_PARAMETERS, "relative_l1", "pooled_relative_l1")


def model_forecast(train, target, test, name):
    config = {
        "random_state": 1, "n_estimators": 200, "learning_rate": 0.1,
        "verbosity": -1, "n_jobs": 2,
    }
    config.update(MODEL_PARAMETERS["compact_l1" if name.endswith("relative_l1") else name])
    if name == "pooled_relative_l1":
        config["min_child_samples"] = 300
    model = LGBMRegressor(**config)
    weights = np.where(target > 12, 2, 1) if name == "double_tail_weight" else None
    model.fit(train, target, sample_weight=weights)
    return model.predict(test)


def relative_features(features):
    result = {}
    for field in ("ticker", "sector_y", "usdt", "brent"):
        for lag in (7, 14, 28):
            result[f"{field}_change_{lag}"] = features[field] / features[f"{field}_lag_{lag}"] - 1
    result["volume_to_average"] = features["volume"] / features["volume_rolling5"]
    return pd.DataFrame(result, index=features.index).replace([np.inf, -np.inf], np.nan)


def summarize(results):
    metrics = {}
    for name in METHODS:
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


def audit(data_frames, horizon, index_sources, index_dates, candles, as_of, actual_index, moex_get):
    results = []
    for cutoff, end in zip(FOLD_BOUNDARIES[:-1], FOLD_BOUNDARIES[1:]):
        for tickers in data_frames.values():
            for ticker, frame in tickers.items():
                dates = frame.index.to_series()
                label_end = dates.shift(-horizon)
                train = label_end < cutoff
                test = (dates >= cutoff) & (dates < end) & (label_end < dates.iloc[-1])
                if train.sum() >= MIN_TRAIN_ROWS and test.any():
                    results.append(pd.DataFrame({
                        "date": frame.index[test], "ticker": ticker,
                        "median": frame.loc[train, "target"].median(),
                    }))
    from tools.evening_backtest import audit as evening_audit

    evening_audit(pd.concat(results, ignore_index=True), candles, as_of, moex_get)


def print_report(period, results):
    print("WALK_FORWARD " + json.dumps({
        "period": period,
        "rows": len(results),
        "tickers": int(results["ticker"].nunique()),
        "days": int(results["date"].nunique()),
        "actual_ge20": int((results["actual"] >= 20).sum()),
        "metrics": summarize(results),
    }), flush=True)
