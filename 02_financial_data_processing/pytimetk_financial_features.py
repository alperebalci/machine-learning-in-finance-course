"""Optional pytimetk grouped features for synthetic multi-asset daily returns.

The feature set at date t uses returns no newer than t-1, to illustrate
chronological model preparation without data downloads or future leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_panel(n_days: int = 320, seed: int = 11) -> pd.DataFrame:
    """Produce a reproducible, independently indexed panel for three assets."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    common_factor = rng.normal(0.0001, 0.008, size=n_days)
    parts = []
    for i, symbol in enumerate(("ASSET_A", "ASSET_B", "ASSET_C")):
        noise = rng.normal(0.0, 0.006, size=n_days)
        close = 100.0 * np.exp(np.cumsum((0.8 + 0.2 * i) * common_factor + noise))
        returns = pd.Series(close).pct_change().to_numpy()
        parts.append(pd.DataFrame({"symbol": symbol, "date": dates, "return_1d": returns}))
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"])


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate lagged and rolling predictors separately for each symbol."""
    try:
        import pytimetk as tk
    except ImportError as exc:
        raise SystemExit("Optional dependency missing: python -m pip install pytimetk") from exc

    ordered = data.sort_values(["symbol", "date"]).reset_index(drop=True)
    lagged = tk.augment_lags(
        ordered.groupby("symbol"),
        date_column="date",
        value_column="return_1d",
        lags=[1, 5],
    )
    features = tk.augment_rolling(
        lagged.groupby("symbol"),
        date_column="date",
        value_column="return_1d_lag_1",
        window=20,
        window_func="std",
        threads=1,
    ).sort_values(["symbol", "date"]).reset_index(drop=True)

    # This is a *label*, never a model predictor. Last observation per symbol
    # has no known future return and must be excluded from supervised training.
    features["target_next_return"] = (
        features.groupby("symbol")["return_1d"].shift(-1)
    )
    predictors = [
        "return_1d_lag_1",
        "return_1d_lag_5",
        "return_1d_lag_1_rolling_std_win_20",
    ]
    return features.dropna(subset=[*predictors, "target_next_return"])


def main() -> None:
    panel = synthetic_panel()
    result = prepare_features(panel)
    print(result.groupby("symbol", observed=True).size().to_string())
    print(result.head(8).to_string(index=False))
    print("Feature values come from historical bars; target is the next daily return.")
    print("Use a chronological train/test split before fitting any predictive model.")


if __name__ == "__main__":
    main()
