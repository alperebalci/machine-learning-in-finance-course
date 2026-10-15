"""Optional vectorbt backtest with next-bar signals and explicit trading costs.

Synthetic daily close prices keep the example deterministic and independent of
market-data services. Strategy parameters are fixed before the holdout window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_close(periods: int = 900, seed: int = 19) -> pd.Series:
    """Return strictly positive, reproducible business-day close prices."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.0002, scale=0.012, size=periods)
    prices = 100.0 * np.exp(log_returns.cumsum())
    dates = pd.bdate_range("2020-01-02", periods=periods)
    return pd.Series(prices, index=dates, name="synthetic_close")


def delayed_crossover_signals(
    close: pd.Series, fast_window: int = 20, slow_window: int = 60
) -> tuple[pd.Series, pd.Series]:
    """Determine crossovers at close t, submit simulated orders at close t+1."""
    if not 0 < fast_window < slow_window:
        raise ValueError("Require 0 < fast_window < slow_window")
    fast = close.rolling(fast_window, min_periods=fast_window).mean()
    slow = close.rolling(slow_window, min_periods=slow_window).mean()
    crossing_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    crossing_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    return (
        crossing_up.shift(1, fill_value=False).astype(bool),
        crossing_down.shift(1, fill_value=False).astype(bool),
    )


def main() -> None:
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise SystemExit("Optional dependency missing: python -m pip install vectorbt") from exc

    close = synthetic_close()
    # Indicators use only past/current closes. Signals are delayed one bar.
    entries, exits = delayed_crossover_signals(close)

    # The final 30% is held out; signal parameters are *not* selected on it.
    split = int(0.7 * len(close))
    holdout = close.iloc[split:]
    fees = 0.0005       # 5 basis points of each order value
    slippage = 0.0005   # 5 basis points against the simulated fill
    portfolio = vbt.Portfolio.from_signals(
        holdout,
        entries=entries.iloc[split:],
        exits=exits.iloc[split:],
        init_cash=10_000,
        fees=fees,
        slippage=slippage,
    )
    buy_and_hold = vbt.Portfolio.from_holding(
        holdout, init_cash=10_000, fees=fees, slippage=slippage
    )
    print(f"Holdout periods:         {len(holdout)}")
    print(f"Strategy total return:   {portfolio.total_return():.2%}")
    print(f"Strategy max drawdown:   {portfolio.max_drawdown():.2%}")
    print(f"Completed/active trades: {portfolio.trades.count()}")
    print(f"Buy-and-hold return:     {buy_and_hold.total_return():.2%}")
    print("Synthetic data only; no parameter tuning or intraday fill modeling.")


if __name__ == "__main__":
    main()
