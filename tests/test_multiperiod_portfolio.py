"""Regression tests for the standalone institutional rebalancing LP."""
from pathlib import Path
import runpy
import numpy as np
import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    "projects/multiperiod-portfolio-rebalancing/model.py"))
solve = MODULE["solve"]
FIRST_RETURNS = MODULE["FIRST_RETURNS"]
SECOND_RETURNS = MODULE["SECOND_RETURNS"]
LEAF_NODE = MODULE["LEAF_NODE"]


def test_self_financing_and_nonanticipative_nodes():
    r = solve()
    cost = 0.005
    assert np.isclose(r.initial_risky.sum() * (1 + cost) + r.initial_cash, 100)
    assert (r.initial_risky >= -1e-8).all()
    assert r.initial_cash >= -1e-8
    assert (r.initial_risky <= 0.6 * (r.initial_risky.sum() + r.initial_cash) + 1e-7).all()
    for n in range(2):
        old = r.initial_risky * FIRST_RETURNS[n]
        new = r.node_risky[n]
        assert np.isclose(new.sum() + r.node_cash[n] +
                          cost * np.abs(new - old).sum(),
                          old.sum() + r.initial_cash * 1.005, atol=1e-6)
        assert np.all(new <= 0.6 * (new.sum() + r.node_cash[n]) + 1e-7)
    expected_leaf_wealth = np.array([
        np.dot(r.node_risky[LEAF_NODE[s]], SECOND_RETURNS[s])
        + 1.005 * r.node_cash[LEAF_NODE[s]] for s in range(4)
    ])
    np.testing.assert_allclose(r.terminal_wealth, expected_leaf_wealth)


def test_risk_aversion_does_not_increase_optimal_tail_loss():
    neutral = solve(risk_aversion=0.0)
    conservative = solve(risk_aversion=1.0)
    assert conservative.cvar_loss <= neutral.cvar_loss + 1e-5
    assert np.isclose(conservative.expected_wealth, conservative.terminal_wealth.mean())
    assert np.isclose(conservative.objective,
                      conservative.expected_wealth - conservative.cvar_loss)


@pytest.mark.parametrize("params", [
    {"trading_cost": -0.01}, {"max_asset_weight": 0},
    {"risk_aversion": -1}, {"confidence": 1},
])
def test_invalid_inputs(params):
    with pytest.raises(ValueError):
        solve(**params)
