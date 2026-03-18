# Results Summary

## Headline Result

Optimizing the decision threshold from 0.50 → 0.34 increased portfolio 
profit by **$99.5M (2.6x improvement)** on the held-out test set.

## Model Performance

| Model | Validation AUC |

|---|---|

| XGBoost | 0.7222 |
| Neural Network | 0.7149 |
| Logistic Regression | 0.7114 |

## Profit Optimization

| Metric | Value |

|---|---|

| Optimal threshold | 0.34 |
| Max portfolio profit | $137,776,656 |
| Profit at default threshold (0.50) | $38,282,676 |
| Profit improvement | $99,493,980 |
| Improvement multiplier | 2.6x |

## Key Business Insights

- Loans with P(default) < 0.34 are approved
- Loans with P(default) ≥ 0.34 are denied
- At optimal threshold, ~33% of applicants are approved
- Using 0.50 as threshold leaves $99.5M on the table

## SHAP Feature Importance (Business Translation)

| Feature | Avg Impact Per Loan |

|---|---|

| Interest rate | $3,968 |
| Loan term | $1,999 |
| Debt-to-income ratio | $1,379 |
| FICO score | $1,221 |
| Loan-to-income ratio | $908 |

Key finding: Interest rate is the strongest predictor of default,
likely because LendingClub already prices risk into the rate, 
making it both a cause and a signal of default risk.