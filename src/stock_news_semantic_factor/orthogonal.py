from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def orthogonalize_by_date(
    frame: pd.DataFrame,
    text_col: str,
    base_col: str,
    date_col: str = "date",
    min_names: int = 10,
) -> pd.Series:
    """Remove same-day linear exposure to a baseline factor score."""
    residual = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.dropna(subset=[text_col, base_col]).groupby(date_col):
        if len(group) < min_names or group[base_col].std() < 1e-12:
            continue
        model = LinearRegression()
        x = group[[base_col]].to_numpy(dtype=float)
        y = group[text_col].to_numpy(dtype=float)
        model.fit(x, y)
        residual.loc[group.index] = y - model.predict(x)
    return residual


def blend_scores(
    frame: pd.DataFrame,
    base_col: str,
    text_col: str,
    lambdas: list[float],
    metric_fn,
    ret_col: str = "future_return",
) -> tuple[float, pd.DataFrame]:
    """Select a linear blend weight on validation data."""
    rows = []
    best_lambda = 0.0
    best_rankic = -np.inf
    for lam in lambdas:
        tmp = frame.copy()
        tmp["blend_score"] = tmp[base_col].astype(float) + lam * tmp[text_col].astype(float)
        summary, _ = metric_fn(tmp.dropna(subset=["blend_score", ret_col]), "blend_score")
        rankic = summary.get("rankic_mean", np.nan)
        rows.append({"lambda": lam, **summary})
        if np.isfinite(rankic) and rankic > best_rankic:
            best_rankic = rankic
            best_lambda = lam
    return best_lambda, pd.DataFrame(rows)


