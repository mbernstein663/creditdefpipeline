import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.engine.profit import (
    expected_profit_per_loan,
    portfolio_profit_curve,
    profit_params,
)


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    joblib.dump(cal_model, "src/models/calibrated_model.joblib")
    X_test.to_parquet("src/models/X_test.parquet", index=False)
    y_test.to_frame(name="default").to_parquet("src/models/y_test.parquet", index=False)

    print("Saved calibrated_model.joblib")
    return cal_model, X_test, y_test


def optimize_threshold(cal_model, X_test, y_test, config):
    print("Reoptimizing threshold on calibrated probabilities...")

    INTEREST_REVENUE, LOSS_AMOUNT, SERVICE_COST, FN_LOSS_MULTIPLIER = profit_params(config)

    p_default = cal_model.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy()

    thresholds = np.arange(0.01, 1.01, 0.01)
    curve = portfolio_profit_curve(
        p_default,
        y_true,
        thresholds,
        INTEREST_REVENUE,
        LOSS_AMOUNT,
        servicing_cost=SERVICE_COST,
        fn_loss_multiplier=FN_LOSS_MULTIPLIER,
    )

    optimal_idx = int(curve["portfolio_profit"].to_numpy().argmax())
    optimal_threshold = float(curve.loc[optimal_idx, "threshold"])
    max_profit = float(curve.loc[optimal_idx, "portfolio_profit"])
    approve_all_profit = float(
        curve.loc[np.isclose(curve["threshold"], 1.0), "portfolio_profit"].iloc[0]
    )

    joblib.dump(optimal_threshold, "src/models/optimal_threshold.joblib")

    print(f"New threshold:         {optimal_threshold:.2f}")
    print(f"Max portfolio profit:  ${max_profit:,.0f}")
    print(f"Profit at 100% approve:${approve_all_profit:,.0f}")

    return optimal_threshold, max_profit



def make_decision(loan_features: dict, config_path="config.yaml") -> dict:
    config = load_config(config_path)

    cal_model = joblib.load("src/models/calibrated_model.joblib")
    threshold = joblib.load("src/models/optimal_threshold.joblib")

    INTEREST_REVENUE, LOSS_AMOUNT, SERVICE_COST, FN_LOSS_MULTIPLIER = profit_params(config)

    X = pd.DataFrame([loan_features])
    p_default = float(cal_model.predict_proba(X)[0, 1])
    expected_profit = float(
        expected_profit_per_loan(
            [p_default],
            INTEREST_REVENUE,
            LOSS_AMOUNT,
            servicing_cost=SERVICE_COST,
            fn_loss_multiplier=FN_LOSS_MULTIPLIER,
        )[0]
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
