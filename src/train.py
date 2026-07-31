"""Обучение baseline, SARIMA, Prophet, boosting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import RANDOM_STATE, evaluate_forecast


def forecast_naive(history: pd.Series, horizon: int) -> np.ndarray:
    last = float(history.dropna().iloc[-1])
    return np.full(horizon, last)


def forecast_moving_average(history: pd.Series, horizon: int, window: int = 4) -> np.ndarray:
    tail = history.dropna().iloc[-window:]
    return np.full(horizon, float(tail.mean()))


def train_sarima(
    y_train: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 0, 52),
):
    """
    SARIMA. Для weekly retail обычно s=52 (годовая сезонность).
    На коротких рядах (<2 лет) seasonal может быть нестабилен — тогда s=1 или без seasonal.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = y_train.copy()
    y.index = pd.to_datetime(y.index)
    y = y.sort_index().interpolate(limit_direction="both")

    # если ряд короче 2*s — упрощаем seasonal
    s = seasonal_order[3]
    if len(y) < 2 * s + 10:
        seasonal_order = (0, 0, 0, 0)

    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=100)
    return fitted


def forecast_sarima(fitted, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = fitted.get_forecast(steps=horizon)
    mean = np.asarray(pred.predicted_mean)
    ci = pred.conf_int(alpha=0.05)
    lower = np.asarray(ci.iloc[:, 0])
    upper = np.asarray(ci.iloc[:, 1])
    return mean, lower, upper


def select_sarima_order(
    y_train: pd.Series,
    p_range=range(0, 3),
    d_range=range(0, 2),
    q_range=range(0, 3),
    seasonal_period: int = 52,
) -> tuple[tuple, tuple, float]:
    """Грубый grid по AIC. На длинных weekly-рядах s=52 дорого — ограничиваем."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = y_train.copy()
    y.index = pd.to_datetime(y.index)
    y = y.sort_index().interpolate(limit_direction="both")

    use_seasonal = len(y) >= 2 * seasonal_period + 10
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_seasonal = (1, 1, 0, seasonal_period) if use_seasonal else (0, 0, 0, 0)

    for p in p_range:
        for d in d_range:
            for q in q_range:
                seas = (1, 1, 0, seasonal_period) if use_seasonal else (0, 0, 0, 0)
                try:
                    m = SARIMAX(
                        y,
                        order=(p, d, q),
                        seasonal_order=seas,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = m.fit(disp=False, maxiter=50)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                        best_seasonal = seas
                except Exception:
                    continue
    return best_order, best_seasonal, float(best_aic if best_aic < np.inf else -1)


def train_prophet(train_df: pd.DataFrame, yearly_seasonality: bool = True):
    from prophet import Prophet

    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,  # данные уже weekly
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.95,
    )
    if "is_holiday" in train_df.columns:
        # простые holiday-флаги как доп. регрессор
        fit_df = train_df[["ds", "y"]].copy()
        fit_df["ds"] = pd.to_datetime(fit_df["ds"])
        fit_df["is_holiday"] = train_df["is_holiday"].astype(float).values
        m.add_regressor("is_holiday")
        m.fit(fit_df)
    else:
        fit_df = train_df[["ds", "y"]].copy()
        fit_df["ds"] = pd.to_datetime(fit_df["ds"])
        m.fit(fit_df)
    return m


def forecast_prophet(
    model,
    periods: int,
    freq: str = "W-FRI",
    future_regressors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=periods, freq=freq)
    if future_regressors is not None and "is_holiday" in future_regressors.columns:
        # merge regressor на future dates
        fr = future_regressors.copy()
        fr["ds"] = pd.to_datetime(fr["ds"])
        future = future.merge(fr[["ds", "is_holiday"]], on="ds", how="left")
        future["is_holiday"] = future["is_holiday"].fillna(0)
    elif "is_holiday" in model.extra_regressors:
        future["is_holiday"] = 0
    return model.predict(future)


def train_xgboost(X_train, y_train, X_val, y_val, n_estimators: int = 500):
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=40,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, n_estimators: int = 800):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    return model


def recursive_forecast_ml(
    model,
    history_df: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
    date_col: str = "ds",
    target_col: str = "y",
    freq: str = "W-FRI",
) -> np.ndarray:
    """
    Рекурсивный multi-step на horizon недель.
    Внешние регрессоры (temperature и т.д.) на будущие даты неизвестны —
    для простоты протягиваем последнее известное значение.

    FIXME: для markdown/holiday на горизонте лучше подставлять календарь акций,
    иначе модель недооценивает праздничные пики.
    """
    from .features import build_ml_frame

    hist = history_df.copy()
    # какие extra колонки тащить вперёд
    extra = [
        c
        for c in hist.columns
        if c
        not in (
            date_col,
            target_col,
            "n_rows",
        )
        and pd.api.types.is_numeric_dtype(hist[c])
    ]

    preds = []
    last_date = pd.to_datetime(hist[date_col].iloc[-1])

    for step in range(horizon):
        next_date = last_date + pd.Timedelta(weeks=step + 1)
        row = {date_col: next_date, target_col: np.nan}
        for c in extra:
            row[c] = hist[c].iloc[-1]  # last observation carried forward
        tmp = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        feats = build_ml_frame(tmp, date_col=date_col, target_col=target_col)
        x = feats.iloc[[-1]][feature_cols]
        if x.isna().any(axis=None):
            x = x.ffill(axis=0).fillna(0)
        y_hat = float(model.predict(x)[0])
        preds.append(y_hat)
        row[target_col] = y_hat
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)

    return np.array(preds)
