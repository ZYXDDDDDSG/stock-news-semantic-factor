from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_CONCEPTS = {
    "policy": "macro policy support, regulation, fiscal policy, monetary policy",
    "export_control": "export control, sanctions, trade restriction, supply chain risk",
    "ai_compute": "AI infrastructure, computing power, semiconductor, cloud capex",
    "consumer": "consumer demand, retail recovery, household spending",
    "finance_realestate": "banking, insurance, interest rate, real estate credit risk",
    "global_trade": "foreign trade, tariff, export demand, global logistics",
    "risk_appetite": "market risk appetite, liquidity, equity market sentiment",
    "energy_materials": "energy price, metals, raw materials, commodity cycle",
    "infrastructure": "infrastructure investment, construction, public projects",
    "tourism_travel": "tourism, aviation, hotels, mobility, holiday travel",
}


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between row vectors in a and b."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def build_semantic_scores(
    news_embeddings: np.ndarray,
    concept_embeddings: np.ndarray,
    concept_names: list[str],
) -> pd.DataFrame:
    """Convert news embeddings into named semantic primitive scores."""
    sim = cosine_similarity_matrix(news_embeddings, concept_embeddings)
    return pd.DataFrame(sim, columns=[f"sem_{name}" for name in concept_names])


def apply_stock_exposure(
    frame: pd.DataFrame,
    concept_names: list[str],
    exposure_table: pd.DataFrame,
    stock_col: str = "stock",
) -> pd.DataFrame:
    """Create stock-level text factors: semantic score times stock exposure."""
    out = frame.copy()
    exposure = exposure_table.set_index(stock_col)
    for name in concept_names:
        sem_col = f"sem_{name}"
        exp_col = f"exposure_{name}"
        dst_col = f"stock_sem_{name}"
        if sem_col not in out.columns or exp_col not in exposure.columns:
            continue
        out[dst_col] = out[sem_col].astype(float) * out[stock_col].map(exposure[exp_col]).astype(float)
    stock_cols = [c for c in out.columns if c.startswith("stock_sem_")]
    if stock_cols:
        out["stock_sem_sum"] = out[stock_cols].sum(axis=1)
        out["stock_sem_abs_sum"] = out[stock_cols].abs().sum(axis=1)
    return out


def add_time_variants(
    frame: pd.DataFrame,
    factor_cols: list[str],
    stock_col: str = "stock",
    date_col: str = "date",
) -> tuple[pd.DataFrame, list[str]]:
    """Generate lag/diff/deviation/z-score variants for text primitive factors."""
    out = frame.sort_values([stock_col, date_col]).copy()
    made = []
    for col in factor_cols:
        if col not in out.columns:
            continue
        made.append(col)
        group = out.groupby(stock_col)[col]
        out[f"{col}_d1"] = group.diff(1)
        out[f"{col}_d5"] = group.diff(5)
        ma5 = group.transform(lambda s: s.rolling(5, min_periods=3).mean())
        ma20 = group.transform(lambda s: s.rolling(20, min_periods=10).mean())
        std20 = group.transform(lambda s: s.rolling(20, min_periods=10).std())
        out[f"{col}_dev5"] = out[col] - ma5
        out[f"{col}_dev20"] = out[col] - ma20
        out[f"{col}_z20"] = (out[col] - ma20) / std20.replace(0, np.nan)
        made.extend([f"{col}_d1", f"{col}_d5", f"{col}_dev5", f"{col}_dev20", f"{col}_z20"])
    return out, [c for c in made if c in out.columns]


