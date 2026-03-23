# creditdefpipeline

End-to-end credit risk decision engine with ML pipeline and profit optimization.

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

Our calibration check is a robust check on the test net that indicates our model's tendency to overpredict probabilities. Our model was beibg too conservative with default rates (only 24% defaulted at an estimated 55% rate), which would have significantly reduced profit margins.

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

Our residual analysis corroborates this. The mean residual error overly penalizes by dti buckets, term length, and fico buckets. (i.e. the model tells us for 60-year term `actual default probability` - `estimated default probability` = -0.30, meaning the actual default rate was 30% lower than we estimated)
The model shows clear indication of over-conservative defaulting estimates- which could lead to significant profit reduction. We will troubleshoot this by analyzing class imbalances (there are less defaulters than non-defaulters, which may be affecting model results).

We fixed this by including class calibration, which adjusts the predicions ...
This yielded the following calibration curves:
![Calibration Plot](evaluation/plots/calibration_curve.png)


We can see that the calibrated data is fitting extremely well until the probabilities get high- this calibrated model is overfitting because we are using the same data for calbration and did not create an additional calibration set.




2) Model retraining

In initial coding, we used model retraining and test/train re-splitting at each stage. We adapted and changed to .joblib and .pt structure in \src\models to preserve reproducibility and save time.
