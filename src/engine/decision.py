import os
import yaml
import joblib
import numpy as np
import pandas as pd

from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def profit_params(config):
    avg_loan = config["profit"]["avg_loan_amount"]
    int_rate = config["profit"]["avg_interest_rate"]
    term_yrs = config["profit"]["avg_loan_term_years"]
    lgd = config["profit"]["loss_given_default"]
    fn_loss_multiplier = config["profit"].get("false_negative_loss_multiplier", 1.0)

    interest_revenue = avg_loan * int_rate * term_yrs
    loss_amount = avg_loan * lgd
    return interest_revenue, loss_amount, fn_loss_multiplier


def calibrate_and_save(config):
    print("Loading data for calibration...")

    df = pd.read_parquet(config["data"]["features_dir"] + "features.parquet")
    X = df.drop(columns=["default"])
    y = df["default"]

    SEED = config["model"]["random_seed"]
    TEST_SIZE = config["model"]["test_size"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=config["model"]["val_size"],
        random_state=SEED,
        stratify=y_temp
    )

    model_path = "src/models/xgboost_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Missing src/models/xgboost_model.joblib. "
            "Run notebooks/03_baseline_models.ipynb first."
        )
    base_model = joblib.load(model_path)

    cal_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method="sigmoid"
    )

    print("Fitting calibrated model on held-out validation set...")
    cal_model.fit(X_val, y_val)

    os.makedirs("src/models", exist_ok=True)
    joblib.dump(cal_model, "src/models/calibrated_xgboost_model.joblib")
    X_test.to_parquet("src/models/X_test.parquet", index=False)
    y_test.to_frame(name="default").to_parquet("src/models/y_test.parquet", index=False)

    print("Saved calibrated_xgboost_model.joblib")
    return cal_model, X_test, y_test


def optimize_threshold(cal_model, X_test, y_test, config):
    print("Reoptimizing threshold on calibrated probabilities...")

    INTEREST_REVENUE, LOSS_AMOUNT, FN_LOSS_MULTIPLIER = profit_params(config)

    p_default = cal_model.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy()

    thresholds = np.arange(0.01, 0.99, 0.01)
    portfolio_profits = []

    for thresh in thresholds:
        approved = p_default < thresh
        if approved.sum() == 0:
            portfolio_profits.append(0)
            continue

        approved_defaults = y_true[approved] == 1
        approved_non_defaults = ~approved_defaults

        # Realized profit on held-out data:
        # non-default approvals earn interest, false-negative defaults incur amplified loss.
        gains = approved_non_defaults.sum() * INTEREST_REVENUE
        losses = approved_defaults.sum() * LOSS_AMOUNT * FN_LOSS_MULTIPLIER
        profit = gains - losses
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

    INTEREST_REVENUE, LOSS_AMOUNT, FN_LOSS_MULTIPLIER = profit_params(config)

    X = pd.DataFrame([loan_features])
    p_default = float(cal_model.predict_proba(X)[0, 1])
    expected_profit = (
        (1 - p_default) * INTEREST_REVENUE
        - p_default * LOSS_AMOUNT * FN_LOSS_MULTIPLIER
    )

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