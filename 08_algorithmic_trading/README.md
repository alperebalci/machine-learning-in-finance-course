# Algorithmic Trading

This section introduces core concepts and practical building blocks for algorithmic trading. The material is designed for education and emphasizes realistic methodology, especially transaction costs, execution assumptions, chronological backtesting, and the distinction between a research signal and a deployable trading strategy.

## Topics

- introduction to algorithmic trading;
- order types and execution;
- market microstructure;
- trading costs and slippage;
- backtesting framework design;
- momentum strategies;
- mean-reversion strategies;
- statistical arbitrage;
- pairs trading;
- high-frequency trading basics.

## Methodology principles

A strategy should not be evaluated only on raw returns. At minimum, consider:

- transaction costs;
- bid-ask spread;
- slippage;
- turnover;
- position constraints;
- look-ahead bias;
- survivorship bias;
- data snooping;
- latency and execution assumptions;
- out-of-sample testing.

Signals should be generated using only information available at the decision timestamp. Positions should normally be shifted so that a signal computed at time t is applied to returns realized after that decision.

## Backtest structure

A simple vectorized backtest typically follows:

1. compute features using historical information;
2. generate a signal;
3. convert the signal into a target position;
4. lag the position if execution occurs after signal formation;
5. calculate turnover;
6. subtract transaction costs;
7. calculate performance and risk metrics.

The code in this section is deliberately compact and inspectable. It is not intended to replace a production-grade event-driven backtester or broker execution system.

## Optional PyBroker ML walk-forward exercise

[PyBroker](https://www.pybroker.com/en/latest/) is a third-party Python framework for event-driven strategy backtests, model-based signals, walk-forward analysis, and trading-performance evaluation. It complements the small vectorized engine in [`05_backtesting_framework.py`](05_backtesting_framework.py); it does **not** replace the course examples or provide live broker integration here.

The optional [`11_pybroker_ml_walkforward.py`](11_pybroker_ml_walkforward.py) demonstrates:

- deterministic **synthetic** daily OHLCV data, with no historical-data download or credentials;
- a five-bar return indicator as a past-only feature;
- scikit-learn linear regression trained separately within chronological walk-forward training windows to predict the next bar's return;
- a one-bar prediction horizon with `lookahead=1`, and simple long/flat decisions during out-of-sample windows;
- a printout of PyBroker's backtest metrics and executed-trade count.

Use Python **3.11+** and install the third-party framework separately (its PyPI package is `lib-pybroker`, while the import name is `pybroker`):

```bash
python -m pip install lib-pybroker scikit-learn
python 08_algorithmic_trading/11_pybroker_ml_walkforward.py
```

PyBroker is intentionally **not** in the course-wide `requirements.txt` or existing CI. The demo's Python syntax and synthetic OHLCV consistency were checked, but execution with the optional PyBroker dependency has not been verified in CI.

**Limitations:** the synthetic series has no realistic predictive edge; this simplified demo does **not** model commissions, spread, slippage, borrow costs, or latency. Walk-forward splitting helps reduce look-ahead leakage but does not by itself address selection bias, overfitting, or live-trading feasibility. Before interpreting performance, add explicit execution costs and compare to a lagged-signal baseline.

References: [PyBroker documentation](https://www.pybroker.com/en/latest/), [training a model and walk-forward testing](https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html), and [using pandas data as a source](https://www.pybroker.com/en/latest/notebooks/7.%20Creating%20a%20Custom%20Data%20Source.html).

## High-frequency trading note

The HFT material is conceptual. Real high-frequency systems depend on exchange connectivity, queue position, order-book dynamics, timestamp precision, latency measurement, colocation, market data normalization, and detailed exchange rules. A toy notebook cannot reproduce those conditions faithfully.

All material is educational and does not constitute investment advice.