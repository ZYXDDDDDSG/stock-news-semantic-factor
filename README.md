# Stock News Semantic Factor

This repository contains research code for testing whether Chinese financial news text can form incremental stock-level factors beyond price/volume factor baselines.

The project has two stages:

## Stage 1: Text + Time-Series Direction Prediction

Task: use a 7-trading-day OHLCV window and prior news text to predict next-trading-day up/down movement.

Model structure:

```text
7-day OHLCV
-> GRU temporal expert
-> price hidden

News text
-> Qwen3 embedding
-> Text2Comp MLP
-> computation vector

price hidden + computation vector
-> FiLM fusion
-> classifier
-> next-day up/down
```

This stage shows the original PIERN-style idea: the LLM is not the final numerical predictor. It converts natural language into computation variables, while the expert model handles numerical prediction.

## Stage 2: Semantic Text Factors for RankIC / Pearson IC

Task: test whether news text can provide stock-level ranking factors under Pearson IC and RankIC.

Pipeline:

```text
News text
-> Qwen3 embedding
-> similarity to financial semantic prompts
-> daily semantic variables
-> semantic variables x stock exposure
-> stock-level semantic text factors
-> RankIC / Pearson IC / orthogonal tests
```

Semantic primitives include policy, export control, AI compute, consumer demand, finance/real estate, global trade, risk appetite, energy/materials, infrastructure, and tourism/travel.

The core question is not whether text helps every stock every day. The stronger finding is conditional: under specific semantic regimes, text factors can contain information not explained by Alpha158 or JoinQuant factor scores.

## Key Experiment Summary

Recent ablations on a 39-stock news-sensitive universe:

| Experiment | Test Pearson IC | Test RankIC | Note |
|---|---:|---:|---|
| Alpha158 | 0.0969 | 0.0720 | Strong pure factor baseline |
| Text primitives | -0.0059 | 0.0082 | Weak standalone signal |
| Raw Qwen embedding PCA | NaN | NaN | Degenerates for cross-sectional ranking |
| Alpha158 + text primitives | 0.0879 | 0.0677 | Direct full concatenation does not improve |
| Alpha158 + raw PCA | 0.0826 | 0.0656 | Direct embedding hurts baseline |

Interpretation: raw text embeddings are not enough for cross-sectional ranking because the same macro news is shared across stocks. Text must be converted into stock-aware computation variables or semantic factors.

## Repository Layout

```text
configs/
  paths.example.yaml
  experiment_rankic.yaml
src/stock_news_semantic_factor/
  data.py
  factor_baseline.py
  exposure.py
  metrics.py
  semantic_text_factors.py
  orthogonal.py
  text2comp_gru.py
scripts/
  run_rankic_template.py
  run_factor_rankic.py
  run_text_primitive_demo.py
```

## Notes

Raw market data, raw news data, model checkpoints, embeddings, experiment reports, private server paths, and account credentials are intentionally excluded.

