from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def read_daily_ohlcv(path: str | Path, stock: str | None = None) -> pd.DataFrame:
    """Read one stock daily OHLCV file.

    Expected columns: date, open, high, low, close, volume.
    """
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    if stock is not None:
        frame["stock"] = str(stock).zfill(6)
    required = {"date", "stock", *OHLCV_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return frame.sort_values(["stock", "date"]).reset_index(drop=True)


def add_future_return(frame: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Add close-to-close future return for each stock."""
    out = frame.sort_values(["stock", "date"]).copy()
    out["future_return"] = out.groupby("stock")["close"].shift(-horizon) / out["close"] - 1.0
    return out


def build_price_windows(
    frame: pd.DataFrame,
    window: int = 7,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build rolling OHLCV windows and aligned metadata.

    Returns:
        windows: shape [n_samples, window, n_features]
        meta: columns date, stock, future_return
    """
    feature_cols = feature_cols or OHLCV_COLUMNS
    windows = []
    rows = []
    for stock, group in frame.sort_values(["stock", "date"]).groupby("stock"):
        values = group[feature_cols].astype(float).to_numpy()
        dates = group["date"].to_numpy()
        returns = group["future_return"].to_numpy()
        for end in range(window - 1, len(group)):
            if not np.isfinite(returns[end]):
                continue
            chunk = values[end - window + 1 : end + 1]
            if not np.isfinite(chunk).all():
                continue
            windows.append(chunk)
            rows.append({"date": dates[end], "stock": stock, "future_return": returns[end]})
    return np.asarray(windows, dtype=np.float32), pd.DataFrame(rows)


def split_by_date(
    frame: pd.DataFrame,
    train_end: str = "2023-12-31",
    val_end: str = "2025-04-30",
    test_end: str = "2026-05-26",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation/test split."""
    train = frame[frame["date"] <= train_end].copy()
    val = frame[(frame["date"] > train_end) & (frame["date"] <= val_end)].copy()
    test = frame[(frame["date"] > val_end) & (frame["date"] <= test_end)].copy()
    return train, val, test


