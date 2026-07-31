"""
Инференс.

Пример:
  python predict.py --data-dir data --model models/best_model.joblib --horizon 12 --output forecast.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.train import (
    forecast_naive,
    forecast_moving_average,
    forecast_sarima,
    forecast_prophet,
    recursive_forecast_ml,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.joblib")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--store-id", type=int, default=None)
    parser.add_argument("--department", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--output", default="forecast.csv")
    args = parser.parse_args()

    path = Path(args.model)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    bundle = joblib.load(path)
    model = bundle["model"]
    meta = bundle.get("meta", {})
    model_type = meta.get("model_type", "unknown")

    hist = load_data(
        data_dir=args.data_dir,
        store_id=args.store_id,
        department=args.department,
    )
    hist = hist.sort_values("ds").reset_index(drop=True)
    horizon = args.horizon
    last_date = pd.to_datetime(hist["ds"].iloc[-1])
    fut_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W-FRI")

    if model_type == "naive":
        pred = forecast_naive(hist.set_index("ds")["y"], horizon)
    elif model_type == "ma":
        pred = forecast_moving_average(
            hist.set_index("ds")["y"], horizon, window=meta.get("window", 4)
        )
    elif model_type == "sarima":
        pred, _, _ = forecast_sarima(model, horizon)
    elif model_type == "prophet":
        fc = forecast_prophet(model, periods=horizon, freq="W-FRI")
        pred = fc.tail(horizon)["yhat"].values
    elif model_type in ("xgboost", "lightgbm"):
        feat_cols = meta.get("feature_cols")
        if not feat_cols:
            raise ValueError("meta['feature_cols'] required")
        pred = recursive_forecast_ml(model, hist, horizon, feature_cols=feat_cols)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    out = pd.DataFrame({"ds": fut_dates, "yhat": pred})
    out.to_csv(args.output, index=False)
    print(f"Saved {horizon}-week forecast → {args.output}")
    print(out.head())


if __name__ == "__main__":
    main()
