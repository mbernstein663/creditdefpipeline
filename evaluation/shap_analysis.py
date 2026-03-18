import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Setup
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

import joblib

model = joblib.load("src/models/xgboost_model.joblib")
optimal_threshold = joblib.load("src/models/optimal_threshold.joblib")
print(f"Model loaded. Threshold: {optimal_threshold:.2f}")



# SHAP values
print("Computing SHAP values (this takes a few minutes)...")
sample = X_test.sample(n=2000, random_state=SEED)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)
print("SHAP values computed.")

os.makedirs(config["paths"]["plots_dir"], exist_ok=True)

# Plot 1 — Feature importance (beeswarm)
print("Generating plots...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, sample,
    max_display=15,
    show=False
)
plt.title("SHAP feature importance — impact on default probability")
plt.tight_layout()
plt.savefig(config["paths"]["plots_dir"] + "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved shap_summary.png")

# Plot 2 — Bar chart of mean absolute SHAP values
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values, sample,
    plot_type="bar",
    max_display=15,
    show=False
)
plt.title("Mean absolute SHAP values — top 15 features")
plt.tight_layout()
plt.savefig(config["paths"]["plots_dir"] + "shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved shap_bar.png")

# Business translation
print("\n--- Business Insights ---")
shap_df = pd.DataFrame(shap_values, columns=sample.columns)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

print("\nTop 10 features by impact on default probability:")
print(mean_abs_shap.head(10).round(4).to_string())

# Quantify business impact for top features
print("\n--- Dollar Impact Translation ---")
AVG_LOAN = config["profit"]["avg_loan_amount"]
LGD      = config["profit"]["loss_given_default"]
LOSS     = AVG_LOAN * LGD

top_features = mean_abs_shap.head(5).index.tolist()
for feat in top_features:
    avg_impact = mean_abs_shap[feat]
    dollar_impact = avg_impact * LOSS
    print(f"{feat:30s} → avg SHAP {avg_impact:.4f} → ~${dollar_impact:,.0f} impact per loan")

print("\nSHAP analysis complete.")





from sklearn.calibration import calibration_curve

print("\n--- Calibration Check ---")
prob_true, prob_pred = calibration_curve(
    y_test, 
    model.predict_proba(X_test)[:, 1], 
    n_bins=10
)

plt.figure(figsize=(7, 5))
plt.plot(prob_pred, prob_true, marker="o", color="#4C9BE8", 
         linewidth=2, label="XGBoost")
plt.plot([0, 1], [0, 1], linestyle="--", color="#888", 
         linewidth=1, label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of actual defaults")
plt.title("Calibration curve — predicted vs actual default rate")
plt.legend()
plt.tight_layout()
plt.savefig(config["paths"]["plots_dir"] + "calibration.png", 
            dpi=150, bbox_inches="tight")
plt.close()

# Quantify calibration error
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_test, model.predict_proba(X_test)[:, 1])
print(f"Brier score: {brier:.4f} (lower is better, 0.25 = random)")

for pred, true in zip(prob_pred, prob_true):
    diff = true - pred
    direction = "underpredicting" if diff > 0 else "overpredicting"
    print(f"Predicted {pred:.2f} → Actual {true:.2f} ({direction} by {abs(diff):.2f})")