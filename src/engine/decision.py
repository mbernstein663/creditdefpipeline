import os
import yaml
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calibrate_and_save(config):
    print("Loading data for CV-based calibration...")

    df = pd.read_parquet(config["data"]["features_dir"] + "features.parquet")
    X = df.drop(columns=["default"])
    y = df["default"]

    SEED = config["model"]["random_seed"]
    TEST_SIZE = config["model"]["test_size"]
    CV_FOLDS = config["model"]["cv_folds"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )

    # Load the already-chosen XGBoost hyperparameters if you have them saved somewhere.
    # Otherwise define them here to match your original trained model as closely as possible.
    base_model = XGBClassifier(
        random_state=SEED,
        eval_metric="logloss"
    )

    cal_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=CV_FOLDS
    )

    print("Fitting CV-calibrated model...")
    cal_model.fit(X_train, y_train)

    os.makedirs("src/models", exist_ok=True)
    joblib.dump(cal_model, "src/models/calibrated_xgboost_model.joblib")
    X_test.to_parquet("src/models/X_test.parquet", index=False)
    y_test.to_frame(name="default").to_parquet("src/models/y_test.parquet", index=False)

    print("Saved calibrated_xgboost_model.joblib")
    return cal_model, X_test, y_test


def optimize_threshold(cal_model, X_test, y_test, config):
    print("Reoptimizing threshold on calibrated probabilities...")

    AVG_LOAN = config["profit"]["avg_loan_amount"]
    INT_RATE = config["profit"]["avg_interest_rate"]
    TERM_YRS = config["profit"]["avg_loan_term_years"]
    LGD = config["profit"]["loss_given_default"]

    INTEREST_REVENUE = AVG_LOAN * INT_RATE * TERM_YRS
    LOSS_AMOUNT = AVG_LOAN * LGD

    p_default = cal_model.predict_proba(X_test)[:, 1]

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

    print(f"New threshold:         {optimal_threshold:.2f}")
    print(f"Max portfolio profit:  ${max_profit:,.0f}")

    return optimal_threshold, max_profit



def make_decision(loan_features: dict, config_path="config.yaml") -> dict:
    config = load_config(config_path)

    cal_model = joblib.load("src/models/calibrated_xgboost_model.joblib")
    threshold = joblib.load("src/models/optimal_threshold_calibrated.joblib")

    AVG_LOAN = config["profit"]["avg_loan_amount"]
    INT_RATE = config["profit"]["avg_interest_rate"]
    TERM_YRS = config["profit"]["avg_loan_term_years"]
    LGD = config["profit"]["loss_given_default"]

    INTEREST_REVENUE = AVG_LOAN * INT_RATE * TERM_YRS
    LOSS_AMOUNT = AVG_LOAN * LGD

    X = pd.DataFrame([loan_features])
    p_default = float(cal_model.predict_proba(X)[0, 1])
    expected_profit = (1 - p_default) * INTEREST_REVENUE - p_default * LOSS_AMOUNT

    return {
        "decision": "APPROVE" if p_default < threshold else "DENY",
        "p_default": round(p_default, 4),
        "expected_profit": round(expected_profit, 2),
        "threshold_used": round(float(threshold), 2)
    }

if __name__ == "__main__":
    config = load_config()
    cal_model, X_test, y_test = calibrate_and_save(config)
    optimal_threshold, max_profit = optimize_threshold(cal_model, X_test, y_test, config)

    print("\nDecision engine ready.")
    print(f"Using calibrated model with threshold: {optimal_threshold:.2f}")


probs = cal_model.predict_proba(X_test)[:, 1]
print(probs[:10])

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import joblib
import pandas as pd

cal_model = joblib.load("src/models/calibrated_xgboost_model.joblib")
X_test = pd.read_parquet("src/models/X_test.parquet")
y_test = pd.read_parquet("src/models/y_test.parquet")["default"]

cal_probs = cal_model.predict_proba(X_test)[:, 1]

frac_pos_cal, mean_pred_cal = calibration_curve(y_test, cal_probs, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_pred_cal, frac_pos_cal, "s-", label="XGBoost (CV-calibrated, sigmoid)")
plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration curve")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/plots/calibration_curve_cv_sigmoid.png")
plt.show()


import numpy as np
import pandas as pd

probs = cal_model.predict_proba(X_test)[:, 1]

bins = np.linspace(0, 1, 11)
bin_ids = np.digitize(probs, bins) - 1

summary = []
for i in range(10):
    mask = bin_ids == i
    n = mask.sum()
    if n > 0:
        mean_pred = probs[mask].mean()
        frac_pos = y_test[mask].mean()
    else:
        mean_pred = np.nan
        frac_pos = np.nan
    summary.append([i, n, mean_pred, frac_pos])

bin_df = pd.DataFrame(summary, columns=["bin", "count", "mean_pred", "frac_pos"])
print(bin_df)