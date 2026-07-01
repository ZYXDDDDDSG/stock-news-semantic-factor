from __future__ import annotations

import numpy as np
import pandas as pd


def calc_alpha158_like(frame: pd.DataFrame) -> pd.DataFrame:
    """A compact Alpha158-style price/volume factor baseline.

    This is not a byte-for-byte Qlib implementation. It provides common
    rolling return, volatility, volume, and price-location features with a
    similar factor-model interface.
    """
    out = frame.sort_values(["stock", "date"]).copy()
    group = out.groupby("stock", group_keys=False)

    out["ret_1"] = group["close"].pct_change(1)
    for window in [3, 5, 10, 20, 30, 60]:
        out[f"ret_{window}"] = group["close"].pct_change(window)
        out[f"ma_ratio_{window}"] = out["close"] / group["close"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean()
        ) - 1.0
        out[f"volatility_{window}"] = group["close"].pct_change().transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).std()
        )
        out[f"volume_ratio_{window}"] = out["volume"] / group["volume"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean()
        ) - 1.0
        high = group["high"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).max())
        low = group["low"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).min())
        out[f"price_position_{window}"] = (out["close"] - low) / (high - low).replace(0, np.nan)

    out["intraday_ret"] = out["close"] / out["open"] - 1.0
    out["high_low_spread"] = out["high"] / out["low"] - 1.0
    out["close_to_high"] = out["close"] / out["high"] - 1.0
    out["close_to_low"] = out["close"] / out["low"] - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def factor_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"date", "stock", "open", "high", "low", "close", "volume", "future_return"}
    return [c for c in frame.columns if c not in excluded]


