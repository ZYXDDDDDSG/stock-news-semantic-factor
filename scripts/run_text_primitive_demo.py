from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stock_news_semantic_factor.metrics import daily_rank_metrics
from stock_news_semantic_factor.semantic_text_factors import (
    DEFAULT_CONCEPTS,
    apply_stock_exposure,
    build_semantic_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--news-embeddings", required=True, help="NumPy array [n_dates, dim]")
    parser.add_argument("--concept-embeddings", required=True, help="NumPy array [n_concepts, dim]")
    parser.add_argument("--meta", required=True, help="CSV with date, stock, future_return")
    parser.add_argument("--exposure", required=True, help="CSV with stock and exposure_<concept> columns")
    parser.add_argument("--output", default="text_primitive_daily.csv")
    args = parser.parse_args()

    news = np.load(args.news_embeddings)
    concepts = np.load(args.concept_embeddings)
    names = list(DEFAULT_CONCEPTS)
    sem = build_semantic_scores(news, concepts, names)

    meta = pd.read_csv(args.meta)
    if len(sem) != meta["date"].nunique():
        raise ValueError("Demo expects one news embedding per unique date in meta.")
    date_sem = pd.DataFrame({"date": sorted(meta["date"].unique())}).join(sem)
    data = meta.merge(date_sem, on="date", how="left")
    exposure = pd.read_csv(args.exposure)
    data = apply_stock_exposure(data, names, exposure)
    score_cols = [c for c in data.columns if c.startswith("stock_sem_")]
    data["text_score"] = data[score_cols].mean(axis=1)
    summary, daily = daily_rank_metrics(data, "text_score")
    daily.to_csv(args.output, index=False)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()


