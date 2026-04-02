import os
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def train_and_save(config):
    df = pd.read_parquet(config["data"]["features_dir"] + "features.parquet")
    X, y = df.drop(columns=["default"]), df["default"]

    SEED = config["model"]["random_seed"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"],
        random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config["model"]["val_size"],
        random_state=SEED, stratify=y_temp
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        # scale_pos_weight intentionally removed — was causing
        # probability overestimation on imbalanced classes
        random_state=SEED,
        eval_metric="auc",
        early_stopping_rounds=20,
        verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    os.makedirs("src/models", exist_ok=True)
    joblib.dump(model, "src/models/xgboost_model.joblib")
    X_test.to_parquet("src/models/X_test.parquet", index=False)
    y_test.to_frame(name="default").to_parquet(
        "src/models/y_test.parquet", index=False
    )

    print(f"Val AUC: {model.best_score:.4f}")
    return model, X_val, y_val, X_test, y_test


def evaluate_calibration(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]

    brier = brier_score_loss(y_test, probs)
    print(f"Brier score: {brier:.4f}  (0.25 = random, lower is better)")

    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)

    print("\nCalibration check:")
    print(f"{'Predicted':>12}  {'Actual':>8}  {'Gap':>8}")
    for p, a in zip(mean_pred, frac_pos):
        gap = p - a
        flag = " ← large gap" if abs(gap) > 0.05 else ""
        print(f"  {p:.2f}  →  {a:.2f}  (off by {gap:+.2f}){flag}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_pred, frac_pos, "s-", label="XGBoost (no scale_pos_weight)")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration curve  |  Brier score: {brier:.4f}")
    ax.legend()

    os.makedirs("evaluation/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("evaluation/plots/calibration_curve1.png", dpi=150)
    plt.show()

    return brier


def optimize_threshold(model, X_test, y_test, config):
    AVG_LOAN = config["profit"]["avg_loan_amount"]
    INTEREST_REVENUE = (AVG_LOAN
                        * config["profit"]["avg_interest_rate"]
                        * config["profit"]["avg_loan_term_years"])
    LOSS_AMOUNT = AVG_LOAN * config["profit"]["loss_given_default"]

    p_default = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.01, 0.99, 0.01)

    profits = []
    for t in thresholds:
        approved = p_default < t
        if approved.sum() == 0:
            profits.append(0)
            continue
        p = p_default[approved]
        profits.append(
            ((1 - p) * INTEREST_REVENUE - p * LOSS_AMOUNT).sum()
        )

    profits = np.array(profits)
    best_t = thresholds[np.argmax(profits)]
    best_profit = profits[np.argmax(profits)]

    joblib.dump(best_t, "src/models/optimal_threshold.joblib")

    print(f"\nOptimal threshold:    {best_t:.2f}")
    print(f"Max portfolio profit: ${best_profit:,.0f}")
    print(f"Approval rate:        {(p_default < best_t).mean():.1%}")

    return best_t, best_profit


if __name__ == "__main__":
    config = load_config()
    model, X_val, y_val, X_test, y_test = train_and_save(config)
    brier = evaluate_calibration(model, X_test, y_test)
    optimize_threshold(model, X_test, y_test, config)

