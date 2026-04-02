# quick_eval.py — run once, copy the output into README
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from matplotlib import pyplot as plt

model = joblib.load("src/models/xgboost_model.joblib")
threshold = joblib.load("src/models/optimal_threshold.joblib")
X_test = pd.read_parquet("src/models/X_test.parquet")
y_test = pd.read_parquet("src/models/y_test.parquet")["default"]

probs = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, probs)
brier = brier_score_loss(y_test, probs)
approval_rate = (probs < threshold).mean()

# Profit calculation
AVG_LOAN = 15000
INTEREST_REVENUE = AVG_LOAN * 0.13 * 3   # match your config
LOSS_AMOUNT = AVG_LOAN * 0.70

approved = probs < threshold
p = probs[approved]
profit = ((1 - p) * INTEREST_REVENUE - p * LOSS_AMOUNT).sum()

print(f"AUC:              {auc:.4f}")
print(f"Brier score:      {brier:.4f}")
print(f"Optimal threshold:{threshold:.2f}")
print(f"Approval rate:    {approval_rate:.1%}")
print(f"Portfolio profit: ${profit:,.0f}")
print(f"Test set size:    {len(X_test):,} loans")

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.sample(5000, random_state=42))
shap.summary_plot(shap_values, X_test.sample(5000, random_state=42), 
                  plot_type="bar", max_display=8,
                  show=False)
plt.savefig("evaluation/plots/shap_importance.png", dpi=150, 
            bbox_inches="tight")