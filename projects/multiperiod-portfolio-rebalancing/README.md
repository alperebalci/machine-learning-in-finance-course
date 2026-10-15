# Two-Stage Portfolio Rebalancing with CVaR

An independently testable **Operations Research** extension of the course's single-period mean-variance examples. It uses `scipy.optimize.linprog` with seeded **fixed synthetic scenario multipliers**; it is not a historical trading result.

## Decisions and chronology

- **t = 0:** choose two nonnegative risky-dollar positions and a cash position from 100 units of wealth, paying proportional purchase costs.
- **t = 1:** observe an *up* or *down* first-period return, then rebalance the risky and cash positions once, paying absolute-dollar trading costs. There are only two distinct trading decision nodes.
- **t = 2:** observe one of four equally probable terminal return paths; record terminal risky value plus cash. No further trades occur.

Because both leaves under the same t=1 information node use **the same holdings**, the optimization enforces nonanticipativity structurally: it never chooses an order based on t=2 returns that have not yet occurred. Terminal prices do affect the *ex-ante scenario optimization*, which assumes the distribution is given, not the realized leaf.

## Linear objective

For terminal wealth `W_s`, initial wealth `W_0=100`, tail probability `1-alpha`, and scenario probabilities `p_s`:

```text
maximize  E[W_s] - lambda * CVaR_alpha(W_0 - W_s)

CVaR_alpha(loss) = min_eta eta + sum_s p_s * max(loss_s - eta, 0)/(1-alpha)
```

The LP introduces auxiliary variables `z_s >= W_0 - W_s - eta`, nonnegative absolute-trade auxiliaries, and self-financing constraints. Maximum 60% per risky asset, no leverage or short positions, and a 0.5% illustrative cost per traded notional apply at both decision dates. Cash earns an illustrative 0.5% per period.

## Run and test

From the repository root, with NumPy and SciPy already available in the course requirements:

```bash
python projects/multiperiod-portfolio-rebalancing/model.py
python -m pytest -q tests/test_multiperiod_portfolio.py
```

The tests check self-financing at both decision dates, the per-asset concentration cap, terminal wealth accounting, and risk-penalty behavior.

## Extension path

- Introduce a **rolling out-of-sample evaluation** where the scenario distribution is fitted only using past market data.
- Add factor/sector exposure matrices, multiple asset universes, liquidity and minimum-lot constraints; use MILP or mixed-integer convex solvers when necessary.
- Compare dynamic policy with a properly costed static-rebalance baseline, equal-weight, and a robust min-variance baseline.
- Evaluate expected utility, CVaR, turnover, scenario regret, prediction error sensitivity, and scenario-calibration uncertainty.

**Limitations:** this is a *two-trading-decision* scenario-tree exercise, not a full institutional trading system, production risk engine, or real-data walk-forward backtest. The return scenarios and risk preferences are synthetic; the realized paths would not be known at the time of rebalancing. Cash funding, taxes, lots, borrow constraints, liquidity depth, terminal liquidation costs, market-impact curves, and model uncertainty are omitted. Educational use only; not investment advice.
