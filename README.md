# creditdefpipeline
End-to-end credit risk decision engine with ML pipeline and profit optimization.





### Feedback + Errors::

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

Our calibration check is a robust check on the test net that indicates our model's tendency to overpredict probabilities. Our model was beign too conservative with default rates (only 24% defaulted at an estimated 55% rate), which would have significantly reduced profit margins. 
