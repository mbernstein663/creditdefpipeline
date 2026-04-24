import os
from typing import Tuple
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_profit_columns(csv_path: str) -> pd.DataFrame:
    usecols = [
        "loan_amnt",
        "int_rate",
        "term",
        "loan_status",
        "total_rec_prncp",
        "total_pymnt",
    ]
    return pd.read_csv(csv_path, usecols=usecols, low_memory=False)


def parse_interest_rate(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = series.astype(str).str.replace("%", "", regex=False).str.strip()
        numeric = pd.to_numeric(cleaned, errors="coerce")
    else:
        numeric = pd.to_numeric(series, errors="coerce")

    # LendingClub `int_rate` is usually in percent points (e.g. 13.56).
    return numeric.where(numeric <= 1.0, numeric / 100.0)


def parse_term_years(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        months = pd.to_numeric(series, errors="coerce")
    else:
        months = pd.to_numeric(
            series.astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )
    return months / 12.0


def compute_loss_given_default(df: pd.DataFrame) -> float:
    status = df["loan_status"].astype(str).str.lower().fillna("")
    default_mask = status.str.contains(r"charged off|default", regex=True)

    defaults = df.loc[default_mask, ["loan_amnt", "total_rec_prncp"]].copy()
    defaults["loan_amnt"] = pd.to_numeric(defaults["loan_amnt"], errors="coerce")
    defaults["total_rec_prncp"] = pd.to_numeric(
        defaults["total_rec_prncp"], errors="coerce"
    )
    defaults = defaults[defaults["loan_amnt"] > 0]

    # LGD = unrecovered principal / original principal, clipped to [0, 1].
    lgd = (defaults["loan_amnt"] - defaults["total_rec_prncp"]) / defaults["loan_amnt"]
    lgd = lgd.clip(lower=0.0, upper=1.0)
    return float(lgd.mean())


def compute_cashflow_profit_inputs(df: pd.DataFrame) -> Tuple[float, float]:
    status = df["loan_status"].astype(str).str.lower().fillna("")
    loan_amnt = pd.to_numeric(df["loan_amnt"], errors="coerce")
    total_pymnt = pd.to_numeric(df["total_pymnt"], errors="coerce")

    fully_paid = df.loc[status.eq("fully paid"), ["loan_amnt", "total_pymnt"]].copy()
    fully_paid["loan_amnt"] = pd.to_numeric(fully_paid["loan_amnt"], errors="coerce")
    fully_paid["total_pymnt"] = pd.to_numeric(fully_paid["total_pymnt"], errors="coerce")
    fully_paid = fully_paid.dropna()
    fully_paid = fully_paid[fully_paid["loan_amnt"] > 0]

    defaults = df.loc[
        status.str.contains(r"charged off|default", regex=True),
        ["loan_amnt", "total_pymnt"],
    ].copy()
    defaults["loan_amnt"] = pd.to_numeric(defaults["loan_amnt"], errors="coerce")
    defaults["total_pymnt"] = pd.to_numeric(defaults["total_pymnt"], errors="coerce")
    defaults = defaults.dropna()
    defaults = defaults[defaults["loan_amnt"] > 0]

    avg_net_revenue_if_repaid = (fully_paid["total_pymnt"] - fully_paid["loan_amnt"]).clip(lower=0.0)
    avg_net_loss_if_default = (defaults["loan_amnt"] - defaults["total_pymnt"]).clip(lower=0.0)

    return float(avg_net_revenue_if_repaid.mean()), float(avg_net_loss_if_default.mean())


def compute_profit_inputs(df: pd.DataFrame) -> Tuple[float, float, float, float, float, float]:
    loan_amnt = pd.to_numeric(df["loan_amnt"], errors="coerce")
    int_rate = parse_interest_rate(df["int_rate"])
    term_years = parse_term_years(df["term"])
    lgd = compute_loss_given_default(df)
    avg_net_revenue_if_repaid, avg_net_loss_if_default = compute_cashflow_profit_inputs(df)

    return (
        float(loan_amnt.mean()),
        float(int_rate.mean()),
        float(term_years.mean()),
        lgd,
        avg_net_revenue_if_repaid,
        avg_net_loss_if_default,
    )


def update_profit_config(config: dict, values: Tuple[float, float, float, float, float, float]) -> dict:
    (
        avg_loan_amount,
        avg_interest_rate,
        avg_loan_term_years,
        loss_given_default,
        avg_net_revenue_if_repaid,
        avg_net_loss_if_default,
    ) = values

    config.setdefault("profit", {})
    config["profit"]["avg_loan_amount"] = round(avg_loan_amount, 2)
    config["profit"]["avg_interest_rate"] = round(avg_interest_rate, 6)
    config["profit"]["avg_loan_term_years"] = round(avg_loan_term_years, 4)
    config["profit"]["loss_given_default"] = round(loss_given_default, 6)
    config["profit"]["avg_net_revenue_if_repaid"] = round(avg_net_revenue_if_repaid, 2)
    config["profit"]["avg_net_loss_if_default"] = round(avg_net_loss_if_default, 2)
    config["profit"].setdefault("servicing_cost_per_loan", 0.0)
    config["profit"].setdefault("false_negative_loss_multiplier", 1.0)
    return config


def save_config(config: dict, config_path: str = "config.yaml") -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    config_path = "config.yaml"
    config = load_config(config_path)

    csv_path = os.path.join(config["data"]["raw_dir"], config["data"]["lending_club_file"])
    df = load_profit_columns(csv_path)
    values = compute_profit_inputs(df)

    updated_config = update_profit_config(config, values)
    save_config(updated_config, config_path)

    print("Updated config.yaml with empirical profit assumptions:")
    print(f"  avg_loan_amount: {updated_config['profit']['avg_loan_amount']}")
    print(f"  avg_interest_rate: {updated_config['profit']['avg_interest_rate']}")
    print(f"  avg_loan_term_years: {updated_config['profit']['avg_loan_term_years']}")
    print(f"  loss_given_default: {updated_config['profit']['loss_given_default']}")
    print(f"  avg_net_revenue_if_repaid: {updated_config['profit']['avg_net_revenue_if_repaid']}")
    print(f"  avg_net_loss_if_default: {updated_config['profit']['avg_net_loss_if_default']}")


if __name__ == "__main__":
    main()
