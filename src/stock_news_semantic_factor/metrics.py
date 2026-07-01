from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def safe_corr(a, b, method: str = "spearman") -> float:
    """Robust Pearson/Spearman correlation for noisy daily cross sections."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).correlation)
    raise ValueError(f"Unknown correlation method: {method}")


def daily_rank_metrics(
    frame: pd.DataFrame,
    score_col: str,
    ret_col: str = "future_return",
    date_col: str = "date",
    min_names: int = 20,
    topk: int = 5,
) -> tuple[dict, pd.DataFrame]:
    """Compute daily Pearson IC, RankIC, top-k return, and long-short return."""
    rows = []
    for date, group in frame.dropna(subset=[score_col, ret_col]).groupby(date_col):
        if len(group) < min_names:
            continue
        ranked = group.sort_values(score_col, ascending=False)
        k = min(topk, max(1, len(ranked) // 5))
        top = ranked.head(k)
        bottom = ranked.tail(k)
        rows.append(
            {
                "date": date,
                "n": len(ranked),
                "ic": safe_corr(ranked[score_col], ranked[ret_col], "pearson"),
                "rankic": safe_corr(ranked[score_col], ranked[ret_col], "spearman"),
                "topk_ret": float(top[ret_col].mean()),
                "bottomk_ret": float(bottom[ret_col].mean()),
                "long_short": float(top[ret_col].mean() - bottom[ret_col].mean()),
                "topk_hit": float((top[ret_col] > 0).mean()),
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return {"n_dates": 0}, daily
    summary = {"n_dates": int(len(daily)), "avg_names": float(daily["n"].mean())}
    for col in ["ic", "rankic", "topk_ret", "bottomk_ret", "long_short", "topk_hit"]:
        summary[f"{col}_mean"] = float(daily[col].mean())
        summary[f"{col}_std"] = float(daily[col].std(ddof=0))
    summary["rankic_ir"] = float(daily["rankic"].mean() / (daily["rankic"].std(ddof=0) + 1e-12))
    return summary, daily


def zscore_by_date(frame: pd.DataFrame, value_col: str, date_col: str = "date") -> pd.Series:
    group = frame.groupby(date_col)[value_col]
    mean = group.transform("mean")
    std = group.transform("std").replace(0, np.nan).fillna(1.0)
    return ((frame[value_col] - mean) / std).astype(float)


