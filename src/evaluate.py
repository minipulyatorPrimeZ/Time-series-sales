"""Метрики, графики, сохранение модели."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import evaluate_forecast, RANDOM_STATE


def print_metrics(name: str, metrics: dict) -> None:
    print(f"{name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame(results).T.round(4)
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE")
    return df


def plot_series(df: pd.DataFrame, date_col: str = "ds", value_col: str = "y", title: str = "Weekly sales"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[date_col], df[value_col], lw=1.0, color="steelblue")
    if "is_holiday" in df.columns:
        hol = df[df["is_holiday"] == 1]
        if len(hol):
            ax.scatter(hol[date_col], hol[value_col], color="red", s=18, zorder=3, label="holiday")
            ax.legend()
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel(value_col)
    plt.tight_layout()
    return fig


def plot_decomposition(result, title: str = "Decomposition"):
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


def plot_acf_pacf(series: pd.Series, lags: int = 52):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=min(lags, len(series.dropna()) // 2 - 1), ax=axes[1], method="ywm")
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    plt.tight_layout()
    return fig


def plot_forecast_compare(
    history: pd.DataFrame,
    forecasts: dict[str, np.ndarray],
    actual: np.ndarray | None = None,
    date_col: str = "ds",
    value_col: str = "y",
    title: str = "Forecast comparison",
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    freq: str = "W-FRI",
):
    fig, ax = plt.subplots(figsize=(13, 5))
    tail_n = min(80, len(history))
    hist_tail = history.iloc[-tail_n:]
    ax.plot(hist_tail[date_col], hist_tail[value_col], label="history", color="gray", lw=1)

    last_date = pd.to_datetime(history[date_col].iloc[-1])
    for name, pred in forecasts.items():
        fut_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=len(pred), freq=freq)
        ax.plot(fut_dates, pred, label=name, lw=1.5)
        if intervals and name in intervals:
            lo, hi = intervals[name]
            ax.fill_between(fut_dates, lo, hi, alpha=0.2)

    if actual is not None:
        actual = np.asarray(actual)
        fut_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=len(actual), freq=freq)
        ax.plot(fut_dates, actual, label="actual", color="black", lw=1.2, linestyle="--")

    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title: str = "Residuals"):
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(resid, lw=0.8)
    axes[0].axhline(0, color="red", ls="--", lw=1)
    axes[0].set_title(f"{title} (time)")
    axes[1].hist(resid, bins=25, color="steelblue", edgecolor="white")
    axes[1].set_title(f"{title} (hist)")
    plt.tight_layout()
    return fig


def save_model_bundle(path: str | Path, model: Any, meta: dict | None = None) -> Path:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta or {}}, path)
    return path
