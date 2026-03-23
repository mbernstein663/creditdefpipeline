import pandas as pd
import numpy as np
import joblib
from torchgen import model
import yaml
import os
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

from sklearn.isotonic import IsotonicRegression

def calibrate_and_save(config):
    print("Loading base model and data...")
    model = joblib.load("src/models/xgboost_model.joblib")
    X_test = pd.read_parquet("src/models/X_test.parquet")
    y_test = pd.read_parquet("src/models/y_test.parquet")["default"]

    # Load val set for calibration fitting
    df = pd.read_parquet(config["data"]["features_dir"] + "features.parquet")
    X = df.drop(columns=["default"])
    y = df["default"]

    SEED = config["model"]["random_seed"]
    X_temp, _, y_temp, _ = train_test_split(
        X, y, test_size=config["model"]["test_size"],
        random_state=SEED, stratify=y
    )
    _, X_val, _, y_val = train_test_split(
        X_temp, y_temp, test_size=config["model"]["val_size"],
        random_state=SEED, stratify=y_temp
    )

    # Get raw probabilities on val set
    print("Fitting isotonic calibration on validation set...")
    val_probs = model.predict_proba(X_val)[:, 1]

    # Fit isotonic regression to map raw probs to calibrated probs
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs, y_val)

    # Save both
    joblib.dump(iso, "src/models/isotonic_calibrator.joblib")
    print("Saved isotonic_calibrator.joblib")

    return model, iso, X_test, y_test

def optimize_threshold(model, iso, X_test, y_test, config):
    print("Reoptimizing threshold on calibrated probabilities...")

    AVG_LOAN = config["profit"]["avg_loan_amount"]
    INT_RATE = config["profit"]["avg_interest_rate"]
    TERM_YRS = config["profit"]["avg_loan_term_years"]
    LGD      = config["profit"]["loss_given_default"]

    INTEREST_REVENUE = AVG_LOAN * INT_RATE * TERM_YRS
    LOSS_AMOUNT      = AVG_LOAN * LGD

    raw_probs = model.predict_proba(X_test)[:, 1]
    p_default = iso.predict(raw_probs)

    thresholds = np.arange(0.01, 0.99, 0.01)
    portfolio_profits = []

    for thresh in thresholds:
        approved = p_default < thresh
        if approved.sum() == 0:
            portfolio_profits.append(0)
            continue
        p_approved = p_default[approved]
        profit = ((1 - p_approved) * INTEREST_REVENUE - p_approved * LOSS_AMOUNT).sum()
        portfolio_profits.append(profit)

    portfolio_profits = np.array(portfolio_profits)
    optimal_idx = np.argmax(portfolio_profits)
    optimal_threshold = thresholds[optimal_idx]
    max_profit = portfolio_profits[optimal_idx]

    joblib.dump(optimal_threshold, "src/models/optimal_threshold_calibrated.joblib")

    print(f"Old threshold:         0.34")
    print(f"New threshold:         {optimal_threshold:.2f}")
    print(f"Max portfolio profit:  ${max_profit:,.0f}")

    return optimal_threshold, max_profit




def make_decision(loan_features: dict, config_path="config.yaml") -> dict:
    config = load_config(config_path)

    model     = joblib.load("src/models/xgboost_model.joblib")
    iso       = joblib.load("src/models/isotonic_calibrator.joblib")
    threshold = joblib.load("src/models/optimal_threshold_calibrated.joblib")

    AVG_LOAN = config["profit"]["avg_loan_amount"]
    INT_RATE = config["profit"]["avg_interest_rate"]
    TERM_YRS = config["profit"]["avg_loan_term_years"]
    LGD      = config["profit"]["loss_given_default"]

    INTEREST_REVENUE = AVG_LOAN * INT_RATE * TERM_YRS
    LOSS_AMOUNT      = AVG_LOAN * LGD

    X = pd.DataFrame([loan_features])
    raw_prob  = model.predict_proba(X)[0, 1]
    p_default = float(iso.predict([raw_prob])[0])
    expected_profit = (1 - p_default) * INTEREST_REVENUE - p_default * LOSS_AMOUNT

    return {
        "decision":        "APPROVE" if p_default < threshold else "DENY",
        "p_default":       round(p_default, 4),
        "expected_profit": round(expected_profit, 2),
        "threshold_used":  round(float(threshold), 2)
    }

if __name__ == "__main__":
    config = load_config()
    model, iso, X_test, y_test = calibrate_and_save(config)
    optimal_threshold, max_profit = optimize_threshold(model, iso, X_test, y_test, config)
    print("\nDecision engine ready.")
    print(f"Using calibrated model with threshold: {optimal_threshold:.2f}")


    # Add this temporarily to the bottom of decision.py to diagnose
raw_probs = model.predict_proba(X_test)[:, 1]
cal_probs = iso.predict(raw_probs)

import numpy as np
print(f"\nRaw probabilities:")
print(f"  Mean:   {raw_probs.mean():.4f}")
print(f"  Median: {np.median(raw_probs):.4f}")
print(f"  Min:    {raw_probs.min():.4f}")
print(f"  Max:    {raw_probs.max():.4f}")

print(f"\nCalibrated probabilities:")
print(f"  Mean:   {cal_probs.mean():.4f}")
print(f"  Median: {np.median(cal_probs):.4f}")
print(f"  Min:    {cal_probs.min():.4f}")
print(f"  Max:    {cal_probs.max():.4f}")

print(f"\nLoans approved at threshold 0.34:")
print(f"  Raw:        {(raw_probs < 0.34).mean():.1%}")
print(f"  Calibrated: {(cal_probs < 0.34).mean():.1%}")


AVG_LOAN = config["profit"]["avg_loan_amount"]
INT_RATE = config["profit"]["avg_interest_rate"]
TERM_YRS = config["profit"]["avg_loan_term_years"]
LGD      = config["profit"]["loss_given_default"]

INTEREST_REVENUE = AVG_LOAN * INT_RATE * TERM_YRS
LOSS_AMOUNT      = AVG_LOAN * LGD



print("\nProfit at different calibrated thresholds:")
for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.34, 0.40, 0.45, 0.50]:
    approved = cal_probs < t
    if approved.sum() == 0:
        continue
    p = cal_probs[approved]
    profit = ((1 - p) * INTEREST_REVENUE - p * LOSS_AMOUNT).sum()
    rate = approved.mean()
    print(f"  Threshold {t:.2f} → approval {rate:.1%} → profit ${profit:,.0f}")



# Add to decision.py or a new notebook: 05_calibration_analysis.py
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

# Raw model
frac_pos_raw, mean_pred_raw = calibration_curve(y_test, raw_probs, n_bins=10)
ax.plot(mean_pred_raw, frac_pos_raw, 's-', label='XGBoost (raw)')

# Calibrated
frac_pos_cal, mean_pred_cal = calibration_curve(y_test, cal_probs, n_bins=10)
ax.plot(mean_pred_cal, frac_pos_cal, 's-', label='XGBoost (calibrated)')

# Perfect calibration line
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration curve: raw vs. calibrated')
ax.legend()
plt.tight_layout()
plt.savefig('evaluation/plots/calibration_curve.png')
