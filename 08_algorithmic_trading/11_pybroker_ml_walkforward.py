"""Optional PyBroker walk-forward demonstration using synthetic daily OHLCV bars.

This example is educational, not a trading recommendation. It requires the
optional lib-pybroker package and scikit-learn; no data download is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def synthetic_bars(n_bars: int = 800, seed: int = 42) -> pd.DataFrame:
    """Create deterministic, internally consistent synthetic OHLCV bars."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_bars)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n_bars)))
    open_ = np.r_[100.0, close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n_bars))
    return pd.DataFrame(
        {
            "symbol": "SYNTH",
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n_bars, 100_000),
        }
    )


def five_bar_return(bar_data) -> np.ndarray:
    """Use past/current prices only, never future prices, as features."""
    return pd.Series(bar_data.close).pct_change(5).to_numpy()


def train_one_bar_model(_symbol: str, train_data: pd.DataFrame, _test_data: pd.DataFrame):
    """Fit on the training window only; drop the unknown final target bar."""
    train = train_data.copy()
    train["next_return"] = train["close"].pct_change().shift(-1)
    train = train.dropna(subset=["ret_5", "next_return"])
    model = LinearRegression()
    model.fit(train[["ret_5"]], train["next_return"])
    return model, ["ret_5"]


def trade_on_prediction(ctx) -> None:
    """Submit orders from out-of-sample next-bar predictions."""
    prediction = ctx.preds("next_return")[-1]
    if not np.isfinite(prediction):
        return
    if not ctx.long_pos() and prediction > 0.002:
        ctx.buy_shares = 10
    elif ctx.long_pos() and prediction <= 0:
        ctx.sell_all_shares()


def main() -> None:
    try:
        import pybroker
        from pybroker import Strategy, StrategyConfig
    except ImportError as exc:
        raise SystemExit("Install the optional dependency: pip install lib-pybroker") from exc

    data = synthetic_bars()
    indicator = pybroker.indicator("ret_5", five_bar_return)
    model = pybroker.model("next_return", train_one_bar_model, indicators=[indicator])
    strategy = Strategy(
        data,
        start_date=data["date"].iloc[0],
        end_date=data["date"].iloc[-1],
        config=StrategyConfig(initial_cash=100_000),
    )
    strategy.add_execution(trade_on_prediction, "SYNTH", models=model)
    result = strategy.walkforward(windows=3, train_size=0.5, lookahead=1, warmup=6)
    print(result.metrics_df.to_string(index=False))
    print(f"Executed trades: {len(result.trades)}")
    print("Synthetic data; execution costs and slippage are not modeled in this demo.")


if __name__ == "__main__":
    main()
