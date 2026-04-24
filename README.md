# creditdefpipeline

End-to-end credit risk decision engine with ML pipeline and profit optimization.

## Data Dictionary

The project uses LendingClub accepted-loan data and frames the prediction task as:

- `0`: repaid
- `1`: default

The finalized modeling set contains 46 features and about 1.3M loans.

## Model Setup

This is a binary classification problem with moderate dimensionality, a large sample, and an imbalanced target at roughly 80/20 non-default/default. The repo uses a train/validation/test split configured in `config.yaml` and compares three baseline models:

- Logistic Regression
- XGBoost
- Neural Network

## Baseline Model Results

Measured by AUC:

- XGBoost: `0.722`
- Neural Network: `0.715`
- Logistic Regression: `0.711`

XGBoost was selected for the downstream profit optimization workflow.

## Profit Optimization

Notebook `notebooks/04_profit_optimization.ipynb` builds the business decision layer on top of the model. The repo now uses a single shared profit module, [`src/engine/profit.py`](/c:/Users/micro/Documents/creditdefpipeline/src/engine/profit.py:1), so notebook logic, evaluation scripts, and the decision engine all score loans the same way.

### What Was Wrong

The original pipeline had two separate problems:

1. The threshold curve in notebook `04` summed model-estimated expected profit rather than realized held-out profit.
2. The unit economics were too optimistic. Successful loans were credited with simple-interest revenue on the full balance for the full term, while defaults only lost unrecovered principal.

That made the portfolio curve look unrealistically strong at very high approval rates, including near-100% approval.

### What Changed

- [`src/data/profit_config.py`](/c:/Users/micro/Documents/creditdefpipeline/src/data/profit_config.py:1) now computes cashflow-based payoff assumptions from the raw CSV.
- `config.yaml` now stores:
  - `avg_net_revenue_if_repaid`
  - `avg_net_loss_if_default`
  - `servicing_cost_per_loan`
  - `false_negative_loss_multiplier`
- [`notebooks/04_profit_optimization.ipynb`](/c:/Users/micro/Documents/creditdefpipeline/notebooks/04_profit_optimization.ipynb:1) now plots realized portfolio P&L across thresholds and includes the `1.00` threshold explicitly.
- [`src/engine/calibrate.py`](/c:/Users/micro/Documents/creditdefpipeline/src/engine/calibrate.py:1), [`src/engine/decision.py`](/c:/Users/micro/Documents/creditdefpipeline/src/engine/decision.py:1), [`src/engine/uncalibrate_scale_pos_removed.py`](/c:/Users/micro/Documents/creditdefpipeline/src/engine/uncalibrate_scale_pos_removed.py:1), and [`evaluation/eval.py`](/c:/Users/micro/Documents/creditdefpipeline/evaluation/eval.py:1) now all use the same shared profit math.

## Updated Economics

After regenerating the config from the raw data:

- `avg_net_revenue_if_repaid`: `$2,323.41`
- `avg_net_loss_if_default`: `$7,403.41`
- `false_negative_loss_multiplier`: `1.25x`

This puts the break-even default probability at about `20.1%`, versus roughly `40.2%` under the earlier simplified assumptions.

## Updated Results

Running [`src/engine/calibrate.py`](/c:/Users/micro/Documents/creditdefpipeline/src/engine/calibrate.py:1) after the fix produced:

- Brier score (raw): `0.1436`
- Brier score (calibrated): `0.1443`
- Optimal threshold: `0.18`
- Max portfolio profit: `$160,392,472`
- Approval rate at optimal: `58.6%`
- Profit at 100% approval: `$3,281,396`

The key improvement is the last line: approve-all is no longer "hugely profitable." It is now close to flat, which is much more plausible for this dataset and payoff model.

Running [`evaluation/eval.py`](/c:/Users/micro/Documents/creditdefpipeline/evaluation/eval.py:1) reports:

- AUC: `0.7194`
- Brier score: `0.1436`
- Optimal threshold: `0.18`
- Approval rate: `51.3%`
- Portfolio profit: `$157,317,507`
- 100% approval P&L: `$3,281,396`

## Calibration Notes

The earlier XGBoost configuration also over-penalized the positive class because of `scale_pos_weight=(all y=0)/(all y=1)`. For this use case, removing `scale_pos_weight` improved the score distribution and produced more useful probabilities for profit-based decisions.

Calibration is still part of the workflow, but the repo now treats these as separate concerns:

- calibration fixes probability alignment
- profit modeling fixes payoff assumptions

The calibration curve is saved at:

![Calibration Plot](evaluation/plots/calibration_curve.png)

## What Was Learned

- Calibration alone does not validate a business decision system.
- Portfolio backtests should be reported on realized held-out outcomes, not just expected value from the model scores.
- Simple-interest revenue and principal-only LGD can materially overstate lender profitability on amortizing consumer loans.
- Class weighting should be used carefully when the downstream objective is portfolio P&L rather than recall or F1.
