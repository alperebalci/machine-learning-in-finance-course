"""Two-stage, scenario-tree portfolio rebalancing with CVaR and trading costs.

Small transparent linear program: rebalance once after observing the first
return, before the second return. Shared tree nodes enforce nonanticipativity.
All returns are synthetic gross return multipliers, not forecasts of markets.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog

INITIAL_WEALTH = 100.0
FIRST_RETURNS = np.array([[1.09, 1.02], [0.92, 1.06]])
SECOND_RETURNS = np.array([[1.10, 1.02], [0.94, 1.07],
                           [1.06, 1.03], [0.88, 1.08]])
LEAF_NODE = np.array([0, 0, 1, 1])
PROBABILITIES = np.full(4, 0.25)


@dataclass(frozen=True)
class PortfolioResult:
    initial_risky: np.ndarray
    initial_cash: float
    node_risky: np.ndarray
    node_cash: np.ndarray
    terminal_wealth: np.ndarray
    first_turnover: float
    second_turnover: np.ndarray
    expected_wealth: float
    cvar_loss: float
    objective: float


def discrete_cvar(losses: np.ndarray, confidence: float = 0.75) -> float:
    """Rockafellar-Uryasev upper-tail expected loss on equiprobable leaves."""
    losses = np.asarray(losses, dtype=float)
    if losses.size != 4 or not np.all(np.isfinite(losses)):
        raise ValueError("Expected four finite scenario losses")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    return float(min(t + np.mean(np.maximum(losses - t, 0)) / (1 - confidence)
                     for t in losses))


def solve(
    trading_cost: float = 0.005,
    risk_aversion: float = 0.3,
    max_asset_weight: float = 0.60,
    confidence: float = 0.75,
) -> PortfolioResult:
    """Maximize expected terminal wealth minus risk-aversion times loss CVaR.

    Trading cost is a fraction of traded notional at t=0 and t=1. There
    is no terminal liquidation, leverage, shorting, or transaction-size lot.
    """
    if not (0 <= trading_cost < 1 and risk_aversion >= 0
            and 0 < max_asset_weight <= 1 and 0 < confidence < 1):
        raise ValueError("Invalid cost, aversion, weight, or confidence")

    names = ([f"x0_{a}" for a in range(2)] + ["cash0"]
             + [f"u0_{a}" for a in range(2)]
             + [f"x1_{n}_{a}" for n in range(2) for a in range(2)]
             + [f"cash1_{n}" for n in range(2)]
             + [f"u1_{n}_{a}" for n in range(2) for a in range(2)]
             + ["eta"] + [f"z_{s}" for s in range(4)])
    ix = {n: i for i, n in enumerate(names)}
    ub, ub_rhs, eq, eq_rhs = [], [], [], []
    def add(store, bounds, coefficients, rhs):
        v = np.zeros(len(names))
        for k, value in coefficients.items():
            v[ix[k]] = value
        store.append(v)
        bounds.append(float(rhs))

    # At inception buy assets with initial cash and pay explicit turnover cost.
    add(eq, eq_rhs, {"x0_0": 1, "x0_1": 1, "cash0": 1,
                     "u0_0": trading_cost, "u0_1": trading_cost}, INITIAL_WEALTH)
    for a in range(2):
        add(ub, ub_rhs, {f"x0_{a}": 1, f"u0_{a}": -1}, 0)
        add(ub, ub_rhs, {f"x0_{a}": 1 - max_asset_weight,
                         f"x0_{1-a}": -max_asset_weight,
                         "cash0": -max_asset_weight}, 0)

    for n in range(2):
        # Dollar account balance after first-period returns and rebalancing.
        balance = {f"x1_{n}_{a}": 1 for a in range(2)}
        balance[f"cash1_{n}"] = 1
        balance["cash0"] = -1.005  # one-period synthetic cash return
        for a in range(2):
            balance[f"x0_{a}"] = -float(FIRST_RETURNS[n, a])
            balance[f"u1_{n}_{a}"] = trading_cost
        add(eq, eq_rhs, balance, 0)
        for a in range(2):
            previous = float(FIRST_RETURNS[n, a])
            add(ub, ub_rhs, {f"x1_{n}_{a}": 1, f"x0_{a}": -previous,
                             f"u1_{n}_{a}": -1}, 0)
            add(ub, ub_rhs, {f"x1_{n}_{a}": -1, f"x0_{a}": previous,
                             f"u1_{n}_{a}": -1}, 0)
            add(ub, ub_rhs, {f"x1_{n}_{a}": 1 - max_asset_weight,
                             f"x1_{n}_{1-a}": -max_asset_weight,
                             f"cash1_{n}": -max_asset_weight}, 0)

    obj = np.zeros(len(names))
    for s in range(4):
        n = int(LEAF_NODE[s])
        wealth = {f"cash1_{n}": 1.005}
        for a in range(2):
            wealth[f"x1_{n}_{a}"] = float(SECOND_RETURNS[s, a])
        # z_s >= INITIAL_WEALTH - W_s - eta
        add(ub, ub_rhs,
            {**{k: -v for k, v in wealth.items()}, "eta": -1, f"z_{s}": -1},
            -INITIAL_WEALTH)
        for k, coef in wealth.items():
            obj[ix[k]] -= PROBABILITIES[s] * coef
        obj[ix[f"z_{s}"]] = risk_aversion * PROBABILITIES[s] / (1 - confidence)
    obj[ix["eta"]] = risk_aversion
    bounds = [(0, None)] * len(names)
    bounds[ix["eta"]] = (None, None)
    result = linprog(obj, A_ub=np.array(ub), b_ub=np.array(ub_rhs),
                     A_eq=np.array(eq), b_eq=np.array(eq_rhs),
                     bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Portfolio LP failed: {result.message}")
    x = dict(zip(names, result.x))
    initial = np.array([x[f"x0_{a}"] for a in range(2)])
    at_node = np.array([[x[f"x1_{n}_{a}"] for a in range(2)] for n in range(2)])
    node_cash = np.array([x[f"cash1_{n}"] for n in range(2)])
    terminal = np.array([1.005 * node_cash[int(LEAF_NODE[s])] +
                         np.dot(at_node[int(LEAF_NODE[s])], SECOND_RETURNS[s])
                         for s in range(4)])
    turnover = np.sum(np.abs(at_node - FIRST_RETURNS * initial), axis=1)
    expected = float(np.dot(PROBABILITIES, terminal))
    cvar = discrete_cvar(INITIAL_WEALTH - terminal, confidence)
    return PortfolioResult(initial, float(x["cash0"]), at_node, node_cash,
                           terminal, float(initial.sum()), turnover, expected,
                           cvar, expected - risk_aversion * cvar)


def main() -> None:
    p = solve()
    print("Initial risky holdings:", p.initial_risky.round(3))
    print("After up/down observation:", p.node_risky.round(3))
    print(f"Expected final wealth: {p.expected_wealth:.4f}")
    print(f"Loss CVaR: {p.cvar_loss:.4f}")
    print("Scenario terminal wealth:", p.terminal_wealth.round(3))
    print("Educational synthetic model; not investment advice.")


if __name__ == "__main__":
    main()
