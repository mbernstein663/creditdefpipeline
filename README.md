# creditdefpipeline

End-to-end credit risk decision engine with ML pipeline and profit optimization.

## Data Dictionary


## The Model Setup

We define our binary target variable as "Repaid" (0) or "Default" (1). Approximately 20% of all loans are defaulted, therefore we need to be cautious about using accuracy as a success proxy.

We then performed some basic EDA and could quickly notice some correlation between default rates and loan grade & debt-to-income ratio. Higher grade loans are defaulted less, and higher DTI loans are defaulted more. No surprises there.

We then cleaned column formatting, removed nulls, and engineered some new features like Loan-to-income (LTI), FICO midpoint, and some behavioral flags like derogatory public record.

Our finalized dataset has 46 columns with ~1.3M documented loans.

## Model Selection + Results

We now have a binary classification problem with reasonable dimensionality, lots of examples, and class 80-20 imbalance. In our `config.yaml` we set test size to 20% and validation to 10% with 5 cross-validation folds. We then use a standard scaler to scale the validation and test set using the spread of `X_train`. 

We kept the modelling relatively conservative & used three binary classification staples: standard log regression, XGBoost, and a neural network classifier. In a real business environment, we would repeat with more models and a hyperparameter search for more effective profit maximization.

## The Model Results

We evaluated our models using AUC:
XGBoost: 0.722
Neural Network: 0.715
Logistic Regression: 0.711

We then saved our pytorch and scikit models.

## Model Re-Evaluation

In notebook 4, we chose XGBoost by AUC and aimed to optimize profits by performing more rigorous evaluation. In `profit_config.py` we defined the average loan amount, interest rate, loan length, and resulting loss from a default, then relayed results to `config.yaml`. We calculated average interest revenue and loss amounts using those numbers, then used our model trained in `03_baseline_models.ipynb` to get soft default probabilities on our test set. We then calculated expected profits per loan and created the profit curve, which plots the portfolio profits vs. decision threshold to find the optimized threshold. 

We then use our results to devise a decision engine that calculates a softmax probability of defaulting, and decides whether to approve/deny the loan based on our predicted profits.

## Feedback + Errors

Our original analysis yielded a contextually valid SHAP analysis but poor calibration analysis. Our results yielded:

--- Calibration Check ---
Brier score: 0.2144 (lower is better, 0.25 = random)
Predicted 0.08 → Actual 0.02 (overpredicting by 0.06)
Predicted 0.16 → Actual 0.04 (overpredicting by 0.11)
Predicted 0.25 → Actual 0.08 (overpredicting by 0.18)
Predicted 0.35 → Actual 0.12 (overpredicting by 0.23)
Predicted 0.45 → Actual 0.17 (overpredicting by 0.28)
Predicted 0.55 → Actual 0.24 (overpredicting by 0.31)
Predicted 0.65 → Actual 0.32 (overpredicting by 0.33)
Predicted 0.74 → Actual 0.43 (overpredicting by 0.31)
Predicted 0.83 → Actual 0.57 (overpredicting by 0.26)
Predicted 0.91 → Actual 0.75 (overpredicting by 0.16)

Our calibration check is a robust check on the test net that indicates our model's tendency to overpredict probabilities. Our model was being too conservative with default rates (only 24% defaulted at an estimated 55% rate), which would have significantly reduced profit margins.

--- Residual Analysis ---

Mean residual by DTI bucket (negative = overpredicting default):
dti_bucket
0-10    -0.2288
10-20   -0.2472
20-30   -0.2739
30-40   -0.2908
40+     -0.2944

Mean residual by FICO bucket:
fico_bucket
620-660   -0.2874
660-700   -0.2785
700-740   -0.2269
740+      -0.1629

Mean residual by term:
term
36   -0.2411
60   -0.3033

Overall mean residual: -0.2561
Residual analysis complete.

Our residual analysis corroborates this. The mean residual error overly penalizes by dti buckets, term length, and fico buckets. (i.e. the model tells us for 60-year term `actual default probability` - `estimated default probability` = -0.30, meaning the actual default rate was 30% lower than we estimated).
The model shows clear indication of over-conservative defaulting estimates- which could lead to significant profit reduction. We will troubleshoot this by analyzing class imbalances (there are less defaulters than non-defaulters, which may be affecting model results).

We fixed this by including class calibration, which adjusts the predicions ...
This yielded the following calibration curves:
![Calibration Plot](evaluation/plots/calibration_curve.png)

We can see that the calibrated data is fitting extremely well until the probabilities get high- this calibrated model is overfitting because we are using the same data for calbration and did not create an additional calibration set.

## Model Development Notes

### Calibration Fix

Initial XGBoost probabilities were systematically overconfident 
(predicted 0.55 default rate where actual was 0.24). 

Post-calibration results:
- Brier score: [add updated score]
- Optimal threshold: 0.34
- Portfolio profit: $659M (+4.7x vs. uncalibrated)

The profit improvement reflects the uncalibrated model 
incorrectly rejecting ~57% of profitable loans (86% approval 
post-calibration vs. 29% pre-calibration).

### Was Calibration Even Necessary?

It turns out that our problem diagnosis was backwards. We misatrributed the 

### What happened?

AUC:              0.7194
Brier score:      0.1436
Optimal threshold:0.34
Approval rate:    86.3%
Portfolio profit: $741,673,984
Test set size:    269,062 loans

## Model retraining

In initial coding, we used model retraining and test/train re-splitting at each stage. We adapted and changed to .joblib and .pt structure in \src\models to preserve reproducibility and save time.

# What Was Learned

- A calibration check is a diagnostic tool, not a tell all about underlying errors. Don't jump straight into isotonic/sigmoid calibration methods without performing EDA/hyperparameter searches unless you have a strong AUC. This is because the AUC tells us the probability of ranking a random positive over a random negative (i.e. how frequently is my model saying this [actual 1] is more likely to be a 0 than this [actual 0]- and vice versa). With a strong AUC (let's say, 0.85) the model is mostly (85% of the time) correctly ranking probabilities, meaning the activation function may just need some calibration to properly boost the raw probabilities and correctly assign a 1 or 0.
- Do not use `scale_pos_weight` ("tipping the scale") unless there is certainty about the distribution of 1s and 0s in the actual target variable. If strong scale adjustment is needed (i.e. there's a very low chance of defaulting, like with fraud) we may want to try anomaly detection as an alternative.