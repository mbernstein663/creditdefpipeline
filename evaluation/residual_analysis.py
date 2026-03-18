import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

import joblib

model = joblib.load("src/models/xgboost_model.joblib")
optimal_threshold = joblib.load("src/models/optimal_threshold.joblib")
print(f"Model loaded. Threshold: {optimal_threshold:.2f}")

# Build residuals dataframe
p_default = model.predict_proba(X_test)[:, 1]
residuals = pd.DataFrame({
    "p_default":      p_default,
    "actual":         y_test.values,
    "residual":       y_test.values - p_default,
    "abs_residual":   np.abs(y_test.values - p_default),
    "int_rate":       X_test["int_rate"].values,
    "dti":            X_test["dti"].values,
    "term":           X_test["term"].values,
    "fico_range_low": X_test["fico_range_low"].values,
    "loan_to_income": X_test["loan_to_income"].values,
})

# DTI buckets
residuals["dti_bucket"] = pd.cut(
    residuals["dti"],
    bins=[0, 10, 20, 30, 40, 100],
    labels=["0-10", "10-20", "20-30", "30-40", "40+"]
)

# FICO buckets
residuals["fico_bucket"] = pd.cut(
    residuals["fico_range_low"],
    bins=[0, 620, 660, 700, 740, 850],
    labels=["<620", "620-660", "660-700", "700-740", "740+"]
)

os.makedirs(config["paths"]["plots_dir"], exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1 — Residuals vs predicted probability
axes[0,0].scatter(residuals["p_default"], residuals["residual"],
                  alpha=0.1, color="#4C9BE8", s=1)
axes[0,0].axhline(0, color="#E8593C", linewidth=1.5, linestyle="--")
axes[0,0].set_xlabel("Predicted P(default)")
axes[0,0].set_ylabel("Residual (actual - predicted)")
axes[0,0].set_title("Residuals vs predicted probability")

# Plot 2 — Mean residual by DTI bucket
dti_resid = residuals.groupby("dti_bucket", observed=True)["residual"].mean()
axes[0,1].bar(dti_resid.index, dti_resid.values, color="#4C9BE8")
axes[0,1].axhline(0, color="#E8593C", linewidth=1.5, linestyle="--")
axes[0,1].set_xlabel("DTI bucket")
axes[0,1].set_ylabel("Mean residual")
axes[0,1].set_title("Mean residual by DTI — is model biased by DTI?")

# Plot 3 — Mean residual by FICO bucket
fico_resid = residuals.groupby("fico_bucket", observed=True)["residual"].mean()
axes[1,0].bar(fico_resid.index, fico_resid.values, color="#4C9BE8")
axes[1,0].axhline(0, color="#E8593C", linewidth=1.5, linestyle="--")
axes[1,0].set_xlabel("FICO bucket")
axes[1,0].set_ylabel("Mean residual")
axes[1,0].set_title("Mean residual by FICO — is model biased by credit score?")

# Plot 4 — Mean residual by term
term_resid = residuals.groupby("term", observed=True)["residual"].mean()
axes[1,1].bar(term_resid.index.astype(str), term_resid.values, color="#4C9BE8")
axes[1,1].axhline(0, color="#E8593C", linewidth=1.5, linestyle="--")
axes[1,1].set_xlabel("Loan term (months)")
axes[1,1].set_ylabel("Mean residual")
axes[1,1].set_title("Mean residual by term")

plt.suptitle("Residual analysis — where is the model systematically wrong?",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(config["paths"]["plots_dir"] + "residual_analysis.png",
            dpi=150, bbox_inches="tight")
plt.close()

# Print findings
print("--- Residual Analysis ---")
print("\nMean residual by DTI bucket (negative = overpredicting default):")
print(dti_resid.round(4).to_string())
print("\nMean residual by FICO bucket:")
print(fico_resid.round(4).to_string())
print("\nMean residual by term:")
print(term_resid.round(4).to_string())
print("\nOverall mean residual:", residuals["residual"].mean().round(4))
print("Residual analysis complete.")