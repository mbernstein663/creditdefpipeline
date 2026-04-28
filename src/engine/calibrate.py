# calibrate.py — clean version

import os
import sys
import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import matplotlib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.frozen import FrozenEstimator
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.engine.profit import portfolio_profit_curve, profit_params


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_and_calibrate(config):
    df = pd.read_parquet(config["data"]["features_dir"] + "features.parquet")
    X, y = df.drop(columns=["default"]), df["default"]

    SEED = config["model"]["random_seed"]

    # Reproduce the exact train/val/test split from notebook 03
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"],
        random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config["model"]["val_size"],
        random_state=SEED, stratify=y_temp
    )

    model_path = "src/models/xgboost_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Missing src/models/xgboost_model.joblib. "
            "Run notebooks/03_baseline_models.ipynb first to train and save the model."
        )
    base = joblib.load(model_path)
    print("Loaded base XGBoost model from src/models/xgboost_model.joblib")

    # ── Calibrate using isotonic on held-out validation set ───────────────
    # We calibrate on val, evaluate on test — no leakage
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(base),
        method="sigmoid"  # or "isotonic"
    )

    calibrated.fit(X_val, y_val)
    os.makedirs("src/models", exist_ok=True)
    joblib.dump(calibrated, "src/models/calibrated_model.joblib")

    # Save test set for downstream evaluation
    X_test.to_parquet("src/models/X_test.parquet", index=False)
    y_test.to_frame(name="default").to_parquet("src/models/y_test.parquet", index=False)

    return calibrated, base, X_val, y_val, X_test, y_test


def plot_calibration(base, calibrated, X_test, y_test):
    raw_probs = base.predict_proba(X_test)[:, 1]
    cal_probs = calibrated.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    for probs, label, style in [
        (raw_probs, "XGBoost (raw)", "s-"),
        (cal_probs, "XGBoost (sigmoid calibrated)", "o-"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=20)
        ax.plot(mean_pred, frac_pos, style, label=label)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve: raw vs. sigmoid")
    ax.legend()

    # Print Brier scores so they appear in the README
    from sklearn.metrics import brier_score_loss
    print(f"Brier score (raw):        {brier_score_loss(y_test, raw_probs):.4f}")
    print(f"Brier score (calibrated): {brier_score_loss(y_test, cal_probs):.4f}")

    os.makedirs("evaluation/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("evaluation/plots/calibration_curve.png", dpi=150)
    plt.close(fig)


def optimize_threshold(calibrated, X_test, y_test, config):
    INTEREST_REVENUE, LOSS_AMOUNT, SERVICE_COST, FN_LOSS_MULTIPLIER = profit_params(config)

    p_default = calibrated.predict_proba(X_test)[:, 1]
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

    best_idx = int(curve["portfolio_profit"].to_numpy().argmax())
    best_t = float(curve.loc[best_idx, "threshold"])
    best_profit = float(curve.loc[best_idx, "portfolio_profit"])
    approve_all_profit = float(
        curve.loc[np.isclose(curve["threshold"], 1.0), "portfolio_profit"].iloc[0]
    )

    joblib.dump(best_t, "src/models/optimal_threshold.joblib")
    print(f"Optimal threshold: {best_t:.2f}")
    print(f"Max portfolio profit: ${best_profit:,.0f}")
    print(f"Approval rate at optimal: {(p_default < best_t).mean():.1%}")
    print(f"Profit at 100% approval: ${approve_all_profit:,.0f}")

    return best_t, best_profit


if __name__ == "__main__":
    config = load_config()
    calibrated, base, X_val, y_val, X_test, y_test = build_and_calibrate(config)
    plot_calibration(base, calibrated, X_test, y_test)
    optimize_threshold(calibrated, X_test, y_test, config)
