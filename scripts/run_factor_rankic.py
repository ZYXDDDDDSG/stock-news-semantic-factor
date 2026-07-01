from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_news_semantic_factor.factor_baseline import calc_alpha158_like, factor_columns
from stock_news_semantic_factor.metrics import daily_rank_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with date, stock, OHLCV, future_return")
    parser.add_argument("--score-col", default=None, help="Optional existing score column")
    parser.add_argument("--output", default="factor_rankic_daily.csv")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    if args.score_col:
        score_col = args.score_col
    else:
        data = calc_alpha158_like(data)
        cols = factor_columns(data)
        score_col = "simple_factor_score"
        data[score_col] = data[cols].rank(pct=True).mean(axis=1)
    summary, daily = daily_rank_metrics(data, score_col)
    daily.to_csv(args.output, index=False)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()


