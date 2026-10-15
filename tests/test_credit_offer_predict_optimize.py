"""Leakage and feasibility tests for the synthetic learn-then-optimize lab."""

import importlib.util
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "projects" / "credit-offer-predict-then-optimize" / "experiment.py"
spec = importlib.util.spec_from_file_location("credit_offer_experiment", MODULE)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_temporal_partition_does_not_reuse_customers():
    history = module.historical_data(1800, 4)
    train, val = module.temporal_split(history)
    assert train["month"].max() < val["month"].min()
    assert set(train["customer_id"]).isdisjoint(val["customer_id"])
    assert history.loc[history["accepted"] == 0, "default_if_accepted"].isna().all()


def test_predictive_models_are_trained_and_probabilities_are_valid():
    history = module.historical_data(2400, 2)
    train, val = module.temporal_split(history)
    a, p = module.train_models(train)
    metrics = module.validation_metrics(val, a, p)
    assert 0 <= metrics["accept_auc"] <= 1
    assert 0 <= metrics["default_auc"] <= 1
    assert metrics["default_label_count"] < len(val)
    offers = module.score_offers(module.prospective_offers(12), a, p)
    assert offers["pred_accept"].between(0, 1).all()
    assert offers["pred_pd"].between(0, 1).all()


def test_sparse_milp_respects_exposure_loss_and_one_offer():
    history = module.historical_data(2300, 6)
    train, _ = module.temporal_split(history)
    a, p = module.train_models(train)
    df = module.score_offers(module.prospective_offers(30), a, p)
    exp = float(df.groupby("customer_id")["exposure"].max().sum() * .6)
    loss = float(df.groupby("customer_id")["expected_loss"].max().sum() * .6)
    decision = module.solve_offers(df, exposure_cap=exp, loss_cap=loss)
    assert 0 < len(decision.chosen) <= 30
    assert decision.expected_exposure <= exp + 1e-6
    assert decision.expected_loss <= loss + 1e-6
    assert df.iloc[list(decision.chosen)]["customer_id"].is_unique


def test_end_to_end_is_reproducible():
    a = module.run(seed=3, historical=2000, customers=25)
    b = module.run(seed=3, historical=2000, customers=25)
    for key in ("selected_customers", "expected_profit", "synthetic_truth_expected_profit"):
        assert np.isclose(a[key], b[key])
    assert a["solver"] == "SciPy/HiGHS (open source)"
