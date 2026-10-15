# Credit Offer Predict-then-Optimize Lab

A fully synthetic, reproducible educational exercise that **fits** separate
logistic-regression models for credit-offer acceptance and conditional default,
then uses a license-free SciPy/HiGHS MILP to select rate/limit offers under
exposure and expected-credit-loss budgets.

## Run

From the repository root, using the dependencies listed in requirements.txt:

    python projects/credit-offer-predict-then-optimize/experiment.py --historical 6500 --customers 120
    pytest -q tests/test_credit_offer_predict_optimize.py

## Learning protocol

1. Generate historical customers with **randomized** offered rates/limits, chronological month stamps and synthetic outcomes.
2. Observe acceptance for all historical offers; observe default only for **accepted** historical offers. Unaccepted loans have missing default labels, not zero defaults.
3. Fit preprocessing and logistic estimators using only months 0–17; compute acceptance and conditional-default AUC/Brier scores on the later months 18–23.
4. Generate entirely new month-24 customer IDs and six alternative loan offers per customer; score them with the fitted estimators.
5. Maximize predicted expected contribution with a sparse binary MILP, subject to one offer (or none) per customer, predicted exposure budget and predicted expected-loss budget.
6. Evaluate the selected policy with hidden **synthetic** true probabilities, unavailable to training or MILP optimization.

AUC evaluates ranking and Brier score evaluates probabilistic accuracy; neither alone demonstrates good lending policy. The output exposes both predicted and synthetic-truth expected profits so the prediction–decision gap is visible. A future research extension could compute oracle decision regret, calibrate probabilities, and assess robust policies.

## Caveats

- This is simulated behavior, **not** empirical out-of-time validation. Historical labels are synthetically mature; no real repayment lags are modeled.
- Default models are trained only on accepted loans. Even with randomized historic offers, deployment requires selection-bias and counterfactual-identification analysis, missing-outcome handling, and calibration.
- There are no protected attributes, fairness assessment, borrower-affordability rules, lender policy checks, regulatory RWA calculations, uncertainty intervals, prepayment, capital charges, or production controls.
- The funding cost, loss-given-default, exposures and constraint budgets are illustrative; financial units are arbitrary. High-dimensional reality may require more elaborate estimation and optimization.
- Solver optimality is claimed only at its configured MIP-gap tolerance; a run can instead terminate with a feasible but unproven incumbent, or no incumbent.

**Research/education only. No commercial solver, confidential data, underwriting recommendation, or regulatory compliance claim.**
