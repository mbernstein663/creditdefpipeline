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

SEED = config["model"]["random_seed"]

# Load data
df = pd.read_parquet(config["data"]["features_dir"] + "features.parquet")
X = df.drop(columns=["default"])
y = df["default"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=config["model"]["test_size"],
    random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=config["model"]["val_size"],
    random_state=SEED, stratify=y_temp
)

# Train XGBoost
print("Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=SEED,
    eval_metric="auc",
    early_stopping_rounds=20,
    verbosity=0
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("Model trained.")

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