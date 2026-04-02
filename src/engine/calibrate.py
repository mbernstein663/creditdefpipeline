# calibrate.py — clean version

import os
import joblib
import numpy as np
import pandas as pd
from sklearn import base
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def load_config(path="config.yaml"):
    with open(path) as f:
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

    # ── Train base model with your original hyperparameters ──────────────
    # NOTE: Remove scale_pos_weight — this was the calibration bias source.
    # Use class_weight-aware eval metric instead.
    base = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        # scale_pos_weight REMOVED — this was causing overestimation
        random_state=SEED,
        eval_metric="auc",
        early_stopping_rounds=20,
        verbosity=0
    )
    base.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             verbose=False)

    print(f"Base model AUC: check notebook 03 benchmark")

    # ── Calibrate using isotonic on held-out validation set ───────────────
    # We calibrate on val, evaluate on test — no leakage
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

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
        (cal_probs, "XGBoost (isotonic calibrated)", "o-"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=20)
        ax.plot(mean_pred, frac_pos, style, label=label)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve: raw vs. isotonic")
    ax.legend()

    # Print Brier scores so they appear in the README
    from sklearn.metrics import brier_score_loss
    print(f"Brier score (raw):        {brier_score_loss(y_test, raw_probs):.4f}")
    print(f"Brier score (calibrated): {brier_score_loss(y_test, cal_probs):.4f}")

    os.makedirs("evaluation/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("evaluation/plots/calibration_curve.png", dpi=150)
    plt.show()


def optimize_threshold(calibrated, X_test, y_test, config):
    AVG_LOAN = config["profit"]["avg_loan_amount"]
    INTEREST_REVENUE = AVG_LOAN * config["profit"]["avg_interest_rate"] \
                                * config["profit"]["avg_loan_term_years"]
    LOSS_AMOUNT = AVG_LOAN * config["profit"]["loss_given_default"]

    p_default = calibrated.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.01, 0.99, 0.01)

    profits = []
    for t in thresholds:
        approved = p_default < t
        if approved.sum() == 0:
            profits.append(0)
            continue
        p = p_default[approved]
        profits.append(((1 - p) * INTEREST_REVENUE - p * LOSS_AMOUNT).sum())

    profits = np.array(profits)
    best_t = thresholds[np.argmax(profits)]
    best_profit = profits[np.argmax(profits)]

    joblib.dump(best_t, "src/models/optimal_threshold.joblib")
    print(f"Optimal threshold: {best_t:.2f}")
    print(f"Max portfolio profit: ${best_profit:,.0f}")
    print(f"Approval rate at optimal: {(p_default < best_t).mean():.1%}")

    return best_t, best_profit


if __name__ == "__main__":
    config = load_config()
    calibrated, base, X_val, y_val, X_test, y_test = build_and_calibrate(config)
    plot_calibration(base, calibrated, X_test, y_test)
    optimize_threshold(calibrated, X_test, y_test, config)