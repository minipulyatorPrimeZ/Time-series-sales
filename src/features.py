"""Feature engineering для недельного retail-ряда."""

from __future__ import annotations

import numpy as np
import pandas as pd


# для weekly данных лаги в неделях
DEFAULT_LAGS = (1, 2, 4, 8, 12, 52)
DEFAULT_WINDOWS = (4, 8, 12)


def add_calendar_features(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    out["month"] = dt.dt.month
    out["quarter"] = dt.dt.quarter
    out["year"] = dt.dt.year
    out["weekofyear_sin"] = np.sin(2 * np.pi * out["weekofyear"] / 52)
    out["weekofyear_cos"] = np.cos(2 * np.pi * out["weekofyear"] / 52)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "y",
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "y",
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Rolling по прошлому (shift 1), без leakage."""
    out = df.copy()
    shifted = out[target_col].shift(1)
    for w in windows:
        out[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=max(1, w // 2)).mean()
        out[f"roll_std_{w}"] = shifted.rolling(w, min_periods=max(1, w // 2)).std()
    return out


def build_ml_frame(
    df: pd.DataFrame,
    date_col: str = "ds",
    target_col: str = "y",
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Календарь + лаги + rolling + внешние регрессоры (temperature, cpi, ...).
    extra_cols — список уже существующих колонок, которые оставляем как признаки.
    """
    base_cols = [date_col, target_col]
    if extra_cols is None:
        candidates = [
            "is_holiday",
            "temperature",
            "fuel_price",
            "cpi",
            "unemployment",
            "markdown_1",
            "markdown_2",
            "markdown_3",
            "markdown_4",
            "markdown_5",
            "store_size",
        ]
        extra_cols = [c for c in candidates if c in df.columns]

    keep = base_cols + [c for c in extra_cols if c in df.columns]
    out = df[keep].copy()
    out = add_calendar_features(out, date_col=date_col)
    out = add_lag_features(out, target_col=target_col, lags=lags)
    out = add_rolling_features(out, target_col=target_col, windows=windows)

    # store_type / region — one-hot если есть
    for cat in ("store_type", "region"):
        if cat in df.columns:
            dummies = pd.get_dummies(df[cat], prefix=cat, dummy_na=False)
            out = pd.concat([out.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    return out


def get_feature_columns(
    df: pd.DataFrame, target_col: str = "y", date_col: str = "ds"
) -> list[str]:
    exclude = {target_col, date_col}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols
