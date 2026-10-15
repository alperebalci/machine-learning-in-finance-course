"""Train synthetic credit response/risk models, then optimize offers with SciPy/HiGHS.

All records, coefficients and outcomes are synthetic. Research use only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import expit
from scipy.sparse import coo_matrix, csr_matrix, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ("credit_score", "debt_ratio", "income_index", "rate", "limit_ratio", "month")


@dataclass(frozen=True)
class Decision:
    chosen: tuple[int, ...]
    expected_profit: float
    expected_exposure: float
    expected_loss: float
    status: str
    mip_gap: float | None


def probabilities(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Hidden data-generating propensities; never inputs to model fitting."""
    credit = df["credit_score"].to_numpy()
    debt = df["debt_ratio"].to_numpy()
    income = df["income_index"].to_numpy()
    rate = df["rate"].to_numpy()
    size = df["limit_ratio"].to_numpy()
    month = df["month"].to_numpy()
    accept = expit(1.3 + 0.45 * income - 5 * (rate - 0.09)
                   + 0.45 * (size - 1) - 0.9 * debt)
    default = expit(-2.6 - 0.012 * (credit - 650) + 1.8 * debt
                    + 0.55 * (size - 1) + 1.5 * (rate - 0.09)
                    + 0.02 * month)
    return np.clip(accept, 0.01, 0.99), np.clip(default, 0.001, 0.95)


def historical_data(n: int = 6500, seed: int = 7) -> pd.DataFrame:
    """Historical randomized offers; default labels observed only for accepted loans."""
    if n < 100:
        raise ValueError("Need at least 100 historical offers")
    rng = np.random.default_rng(seed)
    # Each customer appears once; month is a strictly historical decision timestamp.
    df = pd.DataFrame({
        "customer_id": np.arange(n),
        "month": np.arange(n) * 24 // n,
        "credit_score": np.clip(rng.normal(665, 75, n), 330, 830),
        "debt_ratio": rng.uniform(0.05, 1.4, n),
        "income_index": rng.lognormal(-0.1, 0.35, n),
        "rate": rng.choice([0.09, 0.12, 0.15], n),
        "limit_ratio": rng.choice([0.8, 1.2], n),
    })
    accept_prob, pd_true = probabilities(df)
    accepted = rng.binomial(1, accept_prob)
    default = rng.binomial(1, pd_true)
    df["accepted"] = accepted
    df["default_if_accepted"] = np.where(accepted == 1, default, np.nan)
    return df


def temporal_split(df: pd.DataFrame, cutoff_month: int = 18) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[df["month"] < cutoff_month].copy()
    test = df.loc[df["month"] >= cutoff_month].copy()
    if train.empty or test.empty or train["month"].max() >= test["month"].min():
        raise ValueError("invalid chronological split")
    if set(train["customer_id"]) & set(test["customer_id"]):
        raise ValueError("customer leakage across temporal split")
    return train, test


def train_models(train: pd.DataFrame):
    """Fit offer acceptance on all offers, PD on accepted historical offers only."""
    accepted_loans = train.loc[train["accepted"] == 1]
    if train["accepted"].nunique() != 2 or accepted_loans["default_if_accepted"].nunique() != 2:
        raise ValueError("training classes absent; increase sample size")
    acceptance_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600))
    pd_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600))
    acceptance_model.fit(train[list(FEATURES)], train["accepted"])
    pd_model.fit(accepted_loans[list(FEATURES)], accepted_loans["default_if_accepted"].astype(int))
    return acceptance_model, pd_model


def validation_metrics(validation: pd.DataFrame, acceptance_model, pd_model) -> dict[str, float]:
    accept_probs = acceptance_model.predict_proba(validation[list(FEATURES)])[:, 1]
    observed = validation.loc[validation["accepted"] == 1]
    default_probs = pd_model.predict_proba(observed[list(FEATURES)])[:, 1]
    return {
        "accept_brier": float(brier_score_loss(validation["accepted"], accept_probs)),
        "accept_auc": float(roc_auc_score(validation["accepted"], accept_probs)),
        "default_brier": float(brier_score_loss(observed["default_if_accepted"], default_probs)),
        "default_auc": float(roc_auc_score(observed["default_if_accepted"], default_probs)),
        "default_label_count": int(len(observed)),
    }


def prospective_offers(n_customers: int = 120, seed: int = 91) -> pd.DataFrame:
    """Fresh future customers with 3 rates x 2 amounts; no overlap with history."""
    if n_customers < 1:
        raise ValueError("n_customers must be positive")
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_customers) + 10_000_000, 6)
    option = np.tile(np.arange(6), n_customers)
    score = np.clip(rng.normal(665, 75, n_customers), 330, 830)
    debt = rng.uniform(0.05, 1.4, n_customers)
    income = rng.lognormal(-0.1, 0.35, n_customers)
    base_limit = rng.uniform(8, 75, n_customers)
    df = pd.DataFrame({
        "customer_id": ids, "month": 24,
        "credit_score": score[np.repeat(np.arange(n_customers), 6)],
        "debt_ratio": debt[np.repeat(np.arange(n_customers), 6)],
        "income_index": income[np.repeat(np.arange(n_customers), 6)],
        "rate": np.array([0.09, 0.12, 0.15])[option // 2],
        "limit_ratio": np.array([0.8, 1.2])[option % 2],
    })
    df["drawn"] = base_limit[np.repeat(np.arange(n_customers), 6)] * df["limit_ratio"] * 0.70
    df["lgd"] = 0.45
    return df


def score_offers(offers: pd.DataFrame, accept_model, pd_model) -> pd.DataFrame:
    """Economics from estimates; hidden probabilities only for simulation evaluation."""
    df = offers.copy()
    df["pred_accept"] = accept_model.predict_proba(df[list(FEATURES)])[:, 1]
    df["pred_pd"] = pd_model.predict_proba(df[list(FEATURES)])[:, 1]
    df["exposure"] = df["pred_accept"] * df["drawn"]
    df["expected_loss"] = df["exposure"] * df["pred_pd"] * df["lgd"]
    df["pred_profit"] = df["exposure"] * (df["rate"] - 0.045) - df["expected_loss"] - 0.35 * df["pred_accept"]
    true_accept, true_default = probabilities(df)
    df["truth_profit"] = true_accept * (df["drawn"] * (df["rate"] - 0.045)
                                        - df["drawn"] * true_default * df["lgd"] - 0.35)
    return df


def solve_offers(df: pd.DataFrame, *, exposure_cap: float, loss_cap: float,
                 objective: str = "pred_profit", time_limit: float = 20.0) -> Decision:
    """License-free, sparse multiple-choice portfolio MILP."""
    if exposure_cap < 0 or loss_cap < 0 or objective not in ("pred_profit", "truth_profit"):
        raise ValueError("invalid budget/objective")
    n = len(df)
    code, labels = pd.factorize(df["customer_id"], sort=False)
    group = coo_matrix((np.ones(n), (code, np.arange(n))), shape=(len(labels), n)).tocsr()
    rows = vstack([csr_matrix(df[["exposure", "expected_loss"]].to_numpy(float).T), group], format="csr")
    result = milp(c=-df[objective].to_numpy(float), integrality=np.ones(n, dtype=np.int32),
                  bounds=Bounds(np.zeros(n), np.ones(n)),
                  constraints=LinearConstraint(rows, -np.inf,
                                               np.r_[exposure_cap, loss_cap, np.ones(len(labels))]),
                  options={"time_limit": float(time_limit), "mip_rel_gap": 0.001})
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(f"No valid MILP incumbent: {result.message}")
    chosen = tuple(map(int, np.flatnonzero(result.x > .5)))
    selected = df.iloc[list(chosen)]
    if selected["customer_id"].duplicated().any():
        raise RuntimeError("multiple offers per customer")
    exposure = float(selected["exposure"].sum())
    loss = float(selected["expected_loss"].sum())
    if exposure > exposure_cap + 1e-5 or loss > loss_cap + 1e-5:
        raise RuntimeError("budget violated")
    gap = getattr(result, "mip_gap", None)
    return Decision(chosen, float(selected["pred_profit"].sum()), exposure, loss,
                    "gap_met" if result.status == 0 else "feasible_unproven",
                    None if gap is None else float(gap))


def run(seed: int = 7, historical: int = 6500, customers: int = 120) -> dict:
    past = historical_data(historical, seed)
    train, validation = temporal_split(past)
    accept_model, pd_model = train_models(train)
    metrics = validation_metrics(validation, accept_model, pd_model)
    offers = score_offers(prospective_offers(customers, seed + 1), accept_model, pd_model)
    exposure_cap = float(offers.groupby("customer_id")["exposure"].max().sum() * 0.60)
    loss_cap = float(offers.groupby("customer_id")["expected_loss"].max().sum() * 0.55)
    decision = solve_offers(offers, exposure_cap=exposure_cap, loss_cap=loss_cap)
    selected = offers.iloc[list(decision.chosen)]
    return {"historical_training_rows": len(train), "historical_validation_rows": len(validation),
            "default_training_rows": int(train["accepted"].sum()),
            "metrics": metrics, "prospective_customers": customers,
            "selected_customers": len(selected), "expected_profit": decision.expected_profit,
            "synthetic_truth_expected_profit": float(selected["truth_profit"].sum()),
            "predicted_expected_loss": decision.expected_loss,
            "exposure_cap": exposure_cap, "loss_cap": loss_cap,
            "status": decision.status, "mip_gap": decision.mip_gap,
            "solver": "SciPy/HiGHS (open source)",
            "disclaimer": "Entirely synthetic; not suitable for actual credit policy."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--historical", type=int, default=6500)
    parser.add_argument("--customers", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.historical, args.customers), indent=2))


if __name__ == "__main__":
    main()
