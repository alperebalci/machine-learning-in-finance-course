# Machine Learning in Finance Course

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is the primary umbrella repository for this Jors Academy research area. Related projects have been consolidated under `projects/` so the methods, implementations, experiments, and case studies can be maintained and explored from one place.

### Included projects

- [`mean-variance-portfolio-cardinality-optimization`](projects/mean-variance-portfolio-cardinality-optimization/)

Each consolidated project keeps its own files and a `SOURCE_REPOSITORY.md` provenance record. The snapshot preserves the source repository's default-branch files at consolidation time; repository-level history and metadata remain separate from the snapshot.
<!-- portfolio-umbrella:end -->

Companion code and Jupyter notebooks for the Jors Academy course **Machine Learning in Finance: A Comprehensive Course**.

This repository focuses on practical, reproducible implementations of financial data processing, time-series analysis, machine learning, deep learning, portfolio management, risk management, algorithmic trading, NLP, and advanced financial applications.

## Repository Structure

- `01_python_financial_analysis/` — financial data structures and time-series manipulation
- `02_financial_data_processing/` — data sources, cleaning, alignment, normalization, and feature engineering
- `03_time_series/` — stationarity, ACF/PACF, smoothing, ARIMA, SARIMA, GARCH, and hybrid models
- `04_ml_market_prediction/` — classification and regression models for market prediction
- `05_deep_learning_in_finance/` — neural networks, CNNs, RNNs, LSTMs, GRUs, and attention
- `06_portfolio_management/` — portfolio analytics, optimization, clustering, neural allocation, and reinforcement learning
- `07_risk_management/` — VaR, expected shortfall, stress testing, Monte Carlo, scenario analysis, and ML-based risk assessment
- `08_algorithmic_trading/` — execution, market microstructure, transaction costs, backtesting, momentum, mean reversion, statistical arbitrage, pairs trading, and HFT concepts; includes an [optional PyBroker ML walk-forward exercise](08_algorithmic_trading/README.md#optional-pybroker-ml-walk-forward-exercise)
- `09_nlp_in_finance/` — financial news collection, preprocessing, sentiment, NER, topic modeling, embeddings, FinBERT, GPT-style workflows, and text-driven trading signals
- `data/` — notes about data sources and local datasets

## Important Methodology Notes

Financial machine learning is highly sensitive to data leakage and invalid validation procedures. The examples in this repository use chronological train/test splits where appropriate and avoid random shuffling for time-series forecasting tasks.

Whenever a target depends on future prices or returns, predictors are constructed only from information that would have been available at the prediction timestamp.

For cross-sectional credit-risk examples, preprocessing is fitted only on training data and out-of-time validation is preferred when the objective is to approximate future deployment conditions.

Portfolio clustering examples standardize heterogeneous K-Means features and use a proper correlation-distance representation for hierarchical clustering rather than treating correlation-matrix rows as ordinary Euclidean observations.

Algorithmic-trading examples explicitly lag positions when appropriate and include transaction-cost assumptions so that a same-period signal is not credited with returns that occurred before the signal could have been executed.

Financial NLP examples preserve source provenance and publication timestamps. Sentiment, entity extraction, and generative-model outputs are treated as intermediate features rather than assumed trading signals, and downstream strategies use delayed positions to avoid look-ahead bias.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

The FinBERT example has optional dependencies that are not required by the rest of the repository:

```bash
pip install transformers torch
```

## Educational Use

The material is intended for educational and non-commercial use. See `LICENSE` for details.

## Disclaimer

This repository is for educational purposes only. It does not constitute investment advice, financial advice, credit advice, or a recommendation to trade, lend, borrow, or purchase any financial instrument. Historical performance, model estimates, and backtests do not guarantee future results.

## Continuous integration coverage

The [deterministic foundations CI](.github/workflows/foundations-ci.yml) executes both Python examples in `01_python_financial_analysis/` on Python 3.11 and 3.12. It validates generated CSV content and time-series example reproducibility without external market-data downloads.

**Scope:** this CI does not yet test the higher-level ML, deep learning, trading, XGBoost or TensorFlow chapters. The course's full requirements are intentionally not installed in this small smoke suite; these other modules require separate task-specific, data-safe validation.
