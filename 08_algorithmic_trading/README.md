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

## Optional vectorbt research exercise

The optional [`12_vectorbt_cost_aware_backtest.py`](12_vectorbt_cost_aware_backtest.py) uses the open-source [vectorbt](https://vectorbt.dev/) library as an independent benchmark for the simpler, inspectable [vectorized backtester](05_backtesting_framework.py). It provides a reproducible synthetic 20/60-day moving-average crossover example with **one-bar-delayed orders**, a fixed final **30% holdout**, explicit **5 bp fees + 5 bp slippage per order**, and a **buy-and-hold baseline**. Strategy parameters are set in advance rather than selected on the holdout.

Install separately; the framework is **not** part of the course-wide requirements or CI:

```bash
python -m pip install vectorbt
python 08_algorithmic_trading/12_vectorbt_cost_aware_backtest.py
```

This is a *vectorized daily-bars illustration*, not a full exchange or market-microstructure simulator. It does not model intrabar order-book queues, liquidity, borrow costs, or live execution. The synthetic prices imply no predictable alpha. Optional-library runtime execution is not covered by the repository's CI.

Documentation: [vectorbt portfolio signal API](https://vectorbt.dev/api/portfolio/base/#vectorbt.portfolio.base.Portfolio.from_signals).

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

## High-frequency trading note

The HFT material is conceptual. Real high-frequency systems depend on exchange connectivity, queue position, order-book dynamics, timestamp precision, latency measurement, colocation, market data normalization, and detailed exchange rules. A toy notebook cannot reproduce those conditions faithfully.

All material is educational and does not constitute investment advice.