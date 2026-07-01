from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_data_exposure(
    frame: pd.DataFrame,
    primitive_col: str,
    ret_col: str = "future_return",
    stock_col: str = "stock",
    min_obs: int = 60,
    shrink: float = 0.5,
) -> pd.Series:
    """Estimate stock exposure to a primitive using train-set correlation.

    The shrink parameter pulls noisy stock-level estimates toward zero.
    """
    exposures = {}
    for stock, group in frame.dropna(subset=[primitive_col, ret_col]).groupby(stock_col):
        if len(group) < min_obs or group[primitive_col].std() < 1e-12 or group[ret_col].std() < 1e-12:
            exposures[stock] = 0.0
            continue
        corr = float(group[primitive_col].corr(group[ret_col]))
        exposures[stock] = shrink * corr
    return pd.Series(exposures, name=f"exposure_{primitive_col}")


def combine_exposures(
    llm_prior: pd.Series | None,
    data_exposure: pd.Series | None,
    industry_exposure: pd.Series | None = None,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> pd.Series:
    """Combine LLM prior, data-estimated, and industry exposure signals."""
    sources = []
    for source in [llm_prior, data_exposure, industry_exposure]:
        if source is not None:
            sources.append(source.astype(float))
    if not sources:
        raise ValueError("At least one exposure source is required.")
    index = sources[0].index
    for source in sources[1:]:
        index = index.union(source.index)
    prior = llm_prior.reindex(index).fillna(0.0) if llm_prior is not None else 0.0
    data = data_exposure.reindex(index).fillna(0.0) if data_exposure is not None else 0.0
    industry = industry_exposure.reindex(index).fillna(0.0) if industry_exposure is not None else 0.0
    a, b, c = weights
    out = a * prior + b * data + c * industry
    return pd.Series(np.clip(out, -1.0, 1.0), index=index, name="combined_exposure")


