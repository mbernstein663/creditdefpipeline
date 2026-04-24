import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


xgb_model = joblib.load("src/models/xgboost_model.joblib")
X_test = pd.read_parquet("src/models/X_test.parquet")
y_test = pd.read_parquet("src/models/y_test.parquet")["default"]

# Get default probabilities on the held-out test set saved from notebook 3.
p_default = xgb_model.predict_proba(X_test)[:, 1]



INTEREST_REVENUE = (
    config["profit"]["avg_loan_amount"]
    * config["profit"]["avg_interest_rate"]
    * config["profit"]["avg_loan_term_years"]
)

LOSS_AMOUNT = (
    config["profit"]["avg_loan_amount"]
    * config["profit"]["loss_given_default"]
)

break_even_pd = INTEREST_REVENUE / (INTEREST_REVENUE + LOSS_AMOUNT)

print("INTEREST_REVENUE:", INTEREST_REVENUE)
print("LOSS_AMOUNT:", LOSS_AMOUNT)
print("break_even_pd:", break_even_pd)
print("max p_default:", p_default.max())
print("mean p_default:", p_default.mean())
print("share above break-even:", (p_default > break_even_pd).mean())