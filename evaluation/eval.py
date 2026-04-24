# quick_eval.py — run once, copy the output into README
import joblib
import pandas as pd
import sys
import yaml
from pathlib import Path
import matplotlib
from sklearn.metrics import roc_auc_score, brier_score_loss
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.engine.profit import portfolio_profit_curve, profit_params

model = joblib.load("src/models/xgboost_model.joblib")
threshold = joblib.load("src/models/optimal_threshold.joblib")
X_test = pd.read_parquet("src/models/X_test.parquet")
y_test = pd.read_parquet("src/models/y_test.parquet")["default"]
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

probs = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, probs)
brier = brier_score_loss(y_test, probs)
approval_rate = (probs < threshold).mean()

INTEREST_REVENUE, LOSS_AMOUNT, SERVICE_COST, FN_LOSS_MULTIPLIER = profit_params(config)
curve = portfolio_profit_curve(
    probs,
    y_test.to_numpy(),
    [threshold, 1.0],
    INTEREST_REVENUE,
    LOSS_AMOUNT,
    servicing_cost=SERVICE_COST,
    fn_loss_multiplier=FN_LOSS_MULTIPLIER,
)
profit = float(curve.loc[curve["threshold"] == float(threshold), "portfolio_profit"].iloc[0])
approve_all_profit = float(curve.loc[curve["threshold"] == 1.0, "portfolio_profit"].iloc[0])

print(f"AUC:              {auc:.4f}")
print(f"Brier score:      {brier:.4f}")
print(f"Optimal threshold:{threshold:.2f}")
print(f"Approval rate:    {approval_rate:.1%}")
print(f"Portfolio profit: ${profit:,.0f}")
print(f"100% approval P&L:${approve_all_profit:,.0f}")
print(f"Test set size:    {len(X_test):,} loans")

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.sample(5000, random_state=42))
shap.summary_plot(shap_values, X_test.sample(5000, random_state=42), 
                  plot_type="bar", max_display=8,
                  show=False)
plt.savefig("evaluation/plots/shap_importance.png", dpi=150, 
            bbox_inches="tight")
