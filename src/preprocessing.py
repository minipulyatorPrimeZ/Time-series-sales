"""Time-based split и подготовка выборок (weekly)."""

from __future__ import annotations

import pandas as pd

from .features import build_ml_frame, get_feature_columns


def time_split(
    df: pd.DataFrame,
    date_col: str = "ds",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Сплит строго по времени."""
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    return train, val, test


def prepare_ml_matrices(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str = "ds",
    target_col: str = "y",
):
    """
    Признаки строятся на полном ряду, затем режутся по датам —
    лаги на стыке train/val корректны.
    """
    full = pd.concat([train, val, test], ignore_index=True)
    full = build_ml_frame(full, date_col=date_col, target_col=target_col)

    train_end = train[date_col].max()
    val_end = val[date_col].max()
    feat_cols = get_feature_columns(full, target_col=target_col, date_col=date_col)

    train_ml = full[full[date_col] <= train_end].dropna(subset=feat_cols + [target_col])
    val_ml = full[(full[date_col] > train_end) & (full[date_col] <= val_end)]
    test_ml = full[full[date_col] > val_end]

    val_ml = val_ml.dropna(subset=feat_cols + [target_col])
    test_ml = test_ml.dropna(subset=feat_cols + [target_col])

    return (
        train_ml[feat_cols],
        train_ml[target_col],
        val_ml[feat_cols],
        val_ml[target_col],
        test_ml[feat_cols],
        test_ml[target_col],
        feat_cols,
        train_ml,
        val_ml,
        test_ml,
    )


def to_series(df: pd.DataFrame, date_col: str = "ds", value_col: str = "y") -> pd.Series:
    """Series с DatetimeIndex, частота W (неделя)."""
    tmp = df[[date_col, value_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.sort_values(date_col).set_index(date_col)
    # не форсируем asfreq — даты в Walmart-like данных уже weekly Friday
    return tmp[value_col]
