"""
Загрузка Retail Store Sales Forecasting Dataset.

Ожидаемая структура data/:
  sales.csv     — store_id, department, date, weekly_sales, is_holiday
  stores.csv    — store_id, store_type, store_size, region
  features.csv  — store_id, date, temperature, fuel_price, markdown_1..5,
                  cpi, unemployment, is_holiday, (+ holiday_name, season если есть)

На выходе — агрегированный недельный ряд (ds, y) + внешние регрессоры по дате.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import RANDOM_STATE, ensure_datetime


# возможные имена колонок в разных выгрузках
_DATE_ALIASES = ("date", "Date", "ds", "week", "Week")
_SALES_ALIASES = ("weekly_sales", "Weekly_Sales", "sales", "y", "revenue")
_STORE_ALIASES = ("store_id", "Store", "store", "Store_ID")
_DEPT_ALIASES = ("department", "Dept", "dept", "Department")
_HOLIDAY_ALIASES = ("is_holiday", "IsHoliday", "holiday")


def _pick_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _normalize_sales(df: pd.DataFrame) -> pd.DataFrame:
    date_c = _pick_col(df, _DATE_ALIASES)
    sales_c = _pick_col(df, _SALES_ALIASES)
    store_c = _pick_col(df, _STORE_ALIASES)
    dept_c = _pick_col(df, _DEPT_ALIASES)
    hol_c = _pick_col(df, _HOLIDAY_ALIASES)

    if date_c is None or sales_c is None:
        raise ValueError(
            f"sales.csv: нужны колонки date и weekly_sales. Есть: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["date"] = ensure_datetime(df[date_c])
    out["weekly_sales"] = pd.to_numeric(df[sales_c], errors="coerce")
    out["store_id"] = df[store_c] if store_c else 1
    out["department"] = df[dept_c] if dept_c else 1
    if hol_c:
        out["is_holiday"] = df[hol_c].astype(int)
    else:
        out["is_holiday"] = 0
    return out.dropna(subset=["date", "weekly_sales"])


def _normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    date_c = _pick_col(df, _DATE_ALIASES)
    store_c = _pick_col(df, _STORE_ALIASES)
    if date_c is None:
        raise ValueError(f"features.csv: нет колонки date. Есть: {list(df.columns)}")

    out = df.copy()
    out = out.rename(columns={date_c: "date"})
    if store_c:
        out = out.rename(columns={store_c: "store_id"})
    out["date"] = ensure_datetime(out["date"])

    # унифицируем is_holiday
    hol_c = _pick_col(out, _HOLIDAY_ALIASES)
    if hol_c and hol_c != "is_holiday":
        out["is_holiday"] = out[hol_c].astype(int)

    # markdown / numeric
    rename_map = {}
    for c in out.columns:
        cl = c.lower().replace(" ", "_")
        if cl.startswith("markdown"):
            rename_map[c] = cl
        elif cl in ("temperature", "fuel_price", "cpi", "unemployment"):
            rename_map[c] = cl
    out = out.rename(columns=rename_map)
    return out


def _normalize_stores(df: pd.DataFrame) -> pd.DataFrame:
    store_c = _pick_col(df, _STORE_ALIASES)
    if store_c is None:
        raise ValueError(f"stores.csv: нет store_id. Есть: {list(df.columns)}")
    out = df.copy()
    out = out.rename(columns={store_c: "store_id"})
    # type / size / region
    lower = {c.lower(): c for c in out.columns}
    for want, aliases in {
        "store_type": ("store_type", "type", "Type"),
        "store_size": ("store_size", "size", "Size"),
        "region": ("region", "Region"),
    }.items():
        for a in aliases:
            if a in out.columns:
                out = out.rename(columns={a: want})
                break
            if a.lower() in lower:
                out = out.rename(columns={lower[a.lower()]: want})
                break
    return out


def load_raw(
    data_dir: str | Path = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читает три CSV из data_dir."""
    data_dir = Path(data_dir)
    sales_path = data_dir / "sales.csv"
    features_path = data_dir / "features.csv"
    stores_path = data_dir / "stores.csv"

    if not sales_path.exists():
        raise FileNotFoundError(
            f"Не найден {sales_path}. "
            "Скачайте датасет и положите sales.csv, features.csv, stores.csv в data/"
        )

    sales = _normalize_sales(pd.read_csv(sales_path))
    features = (
        _normalize_features(pd.read_csv(features_path))
        if features_path.exists()
        else pd.DataFrame()
    )
    stores = (
        _normalize_stores(pd.read_csv(stores_path))
        if stores_path.exists()
        else pd.DataFrame()
    )
    return sales, features, stores


def build_weekly_series(
    sales: pd.DataFrame,
    features: pd.DataFrame | None = None,
    stores: pd.DataFrame | None = None,
    store_id: int | None = None,
    department: int | None = None,
) -> pd.DataFrame:
    """
    Собирает недельный ряд для прогнозирования.

    По умолчанию — сумма weekly_sales по всем магазинам/отделам на каждую дату.
    Можно отфильтровать store_id / department.

    Внешние признаки из features.csv усредняются по store_id на дату
    (temperature, fuel_price, cpi, unemployment, markdowns, is_holiday).
    """
    s = sales.copy()
    if store_id is not None:
        s = s[s["store_id"] == store_id]
    if department is not None:
        s = s[s["department"] == department]
    if s.empty:
        raise ValueError("После фильтрации sales пуст — проверьте store_id / department")

    # агрегат продаж
    g = (
        s.groupby("date", as_index=False)
        .agg(
            y=("weekly_sales", "sum"),
            is_holiday=("is_holiday", "max"),
            n_rows=("weekly_sales", "size"),
        )
        .sort_values("date")
    )

    # внешние фичи
    if features is not None and len(features):
        f = features.copy()
        if store_id is not None and "store_id" in f.columns:
            f = f[f["store_id"] == store_id]

        num_cols = [
            c
            for c in (
                "temperature",
                "fuel_price",
                "cpi",
                "unemployment",
                "markdown_1",
                "markdown_2",
                "markdown_3",
                "markdown_4",
                "markdown_5",
            )
            if c in f.columns
        ]
        agg = {c: "mean" for c in num_cols}
        if "is_holiday" in f.columns:
            agg["is_holiday"] = "max"

        if agg:
            f_agg = f.groupby("date", as_index=False).agg(agg)
            # is_holiday из features не перезаписывает sales, если уже есть
            if "is_holiday" in f_agg.columns and "is_holiday" in g.columns:
                f_agg = f_agg.rename(columns={"is_holiday": "is_holiday_feat"})
            g = g.merge(f_agg, on="date", how="left")
            if "is_holiday_feat" in g.columns:
                g["is_holiday"] = g[["is_holiday", "is_holiday_feat"]].max(axis=1)
                g = g.drop(columns=["is_holiday_feat"])

    # store metadata — константы, если выбран один магазин
    if stores is not None and len(stores) and store_id is not None:
        row = stores[stores["store_id"] == store_id]
        if len(row):
            for col in ("store_type", "store_size", "region"):
                if col in row.columns:
                    g[col] = row.iloc[0][col]

    g = g.rename(columns={"date": "ds"})
    g["ds"] = ensure_datetime(g["ds"])
    g = g.sort_values("ds").reset_index(drop=True)

    # markdown NaN → 0 (нет акции)
    for c in g.columns:
        if c.startswith("markdown"):
            g[c] = g[c].fillna(0.0)

    return g


def load_data(
    data_dir: str | Path = "data",
    store_id: int | None = None,
    department: int | None = None,
) -> pd.DataFrame:
    """
    Главная точка входа.

    Returns
    -------
    DataFrame с колонками:
      ds, y, is_holiday, [temperature, fuel_price, cpi, unemployment, markdown_*]
    """
    data_dir = Path(data_dir)
    if not (data_dir / "sales.csv").exists():
        # заглушка только если данных нет — чтобы CI/демо не падали
        return _synthetic_weekly(random_state=RANDOM_STATE)

    sales, features, stores = load_raw(data_dir)
    return build_weekly_series(
        sales, features, stores, store_id=store_id, department=department
    )


def _synthetic_weekly(random_state: int = 42, n_weeks: int = 260) -> pd.DataFrame:
    """Минимальная заглушка, если data/ пуст. Не использовать для отчёта."""
    rng = np.random.RandomState(random_state)
    dates = pd.date_range("2010-02-05", periods=n_weeks, freq="W-FRI")
    t = np.arange(n_weeks)
    y = 1.5e6 + 800 * t + 1.2e5 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 4e4, n_weeks)
    return pd.DataFrame(
        {
            "ds": dates,
            "y": np.maximum(y, 1e4),
            "is_holiday": 0,
            "temperature": 60 + 20 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 3, n_weeks),
            "fuel_price": 3.2 + 0.002 * t + rng.normal(0, 0.05, n_weeks),
            "cpi": 170 + 0.05 * t,
            "unemployment": 7.5 - 0.005 * t,
        }
    )
