"""Template for RankIC experiments.

This file intentionally avoids hard-coded private paths. Adapt the loader
functions to your local data layout or fill in configs/paths.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from stock_news_semantic_factor.metrics import daily_rank_metrics


def load_scores(path: Path) -> pd.DataFrame:
    """Expected columns: date, stock, score, future_return."""
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_rankic.yaml")
    parser.add_argument("--scores", required=True, help="CSV with date, stock, score, future_return")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    scores = load_scores(Path(args.scores))
    summary, daily = daily_rank_metrics(
        scores,
        "score",
        min_names=config["evaluation"]["min_names_per_day"],
        topk=config["evaluation"]["topk"],
    )
    print(pd.Series(summary).to_string())
    daily.to_csv("daily_rankic.csv", index=False)


if __name__ == "__main__":
    main()


