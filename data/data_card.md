# Data Card

## Dataset: LendingClub Loan Data (2007–2018)

**Source:** [Kaggle — LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)  
**Rows:** ~2.26 million loans  
**Columns:** 150+ features  

## Key Features Used

| Feature | Description |
|---|---|
| `loan_amnt` | Requested loan amount |
| `int_rate` | Interest rate on the loan |
| `annual_inc` | Borrower annual income |
| `dti` | Debt-to-income ratio |
| `loan_status` | Target variable (Fully Paid / Charged Off) |
| `emp_length` | Employment length in years |
| `fico_range_low` | Borrower FICO score (low) |
| `revol_util` | Revolving credit utilization rate |

## Target Variable

`loan_status` is binarized:  

- `1` = Default (Charged Off)  
- `0` = Repaid (Fully Paid)

## Notes

- Raw data is excluded from version control (see `.gitignore`)
- To reproduce: download from Kaggle and place in `data/raw/`
