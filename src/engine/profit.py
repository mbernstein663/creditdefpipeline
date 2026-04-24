import numpy as np
import pandas as pd


def profit_params(config):
    profit_cfg = config["profit"]

    service_cost = float(profit_cfg.get("servicing_cost_per_loan", 0.0))
    fn_loss_multiplier = float(profit_cfg.get("false_negative_loss_multiplier", 1.0))

    if (
        "avg_net_revenue_if_repaid" in profit_cfg
        and "avg_net_loss_if_default" in profit_cfg
    ):
        revenue = float(profit_cfg["avg_net_revenue_if_repaid"])
        loss = float(profit_cfg["avg_net_loss_if_default"])
    else:
        avg_loan = float(profit_cfg["avg_loan_amount"])
        int_rate = float(profit_cfg["avg_interest_rate"])
        term_yrs = float(profit_cfg["avg_loan_term_years"])
        lgd = float(profit_cfg["loss_given_default"])

        revenue = avg_loan * int_rate * term_yrs
        loss = avg_loan * lgd

    return revenue, loss, service_cost, fn_loss_multiplier


def expected_profit_per_loan(
    p_default,
    revenue_if_repaid,
    loss_if_default,
    servicing_cost=0.0,
    fn_loss_multiplier=1.0,
):
    p_default = np.asarray(p_default, dtype=float)
    p_repay = 1.0 - p_default
    return (
        p_repay * revenue_if_repaid
        - p_default * loss_if_default * fn_loss_multiplier
        - servicing_cost
    )


def realized_profit_per_loan(
    y_true,
    revenue_if_repaid,
    loss_if_default,
    servicing_cost=0.0,
    fn_loss_multiplier=1.0,
):
    y_true = np.asarray(y_true, dtype=int)
    repaid = y_true == 0
    defaulted = ~repaid
    return (
        repaid.astype(float) * revenue_if_repaid
        - defaulted.astype(float) * loss_if_default * fn_loss_multiplier
        - servicing_cost
    )


def portfolio_profit_curve(
    p_default,
    y_true,
    thresholds,
    revenue_if_repaid,
    loss_if_default,
    servicing_cost=0.0,
    fn_loss_multiplier=1.0,
):
    p_default = np.asarray(p_default, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    thresholds = np.asarray(thresholds, dtype=float)

    rows = []
    for threshold in thresholds:
        approved = p_default < threshold
        n_approved = int(approved.sum())

        if n_approved == 0:
            rows.append(
                {
                    "threshold": float(threshold),
                    "portfolio_profit": 0.0,
                    "approval_rate": 0.0,
                    "approved_count": 0,
                }
            )
            continue

        profit = realized_profit_per_loan(
            y_true[approved],
            revenue_if_repaid,
            loss_if_default,
            servicing_cost=servicing_cost,
            fn_loss_multiplier=fn_loss_multiplier,
        ).sum()

        rows.append(
            {
                "threshold": float(threshold),
                "portfolio_profit": float(profit),
                "approval_rate": float(approved.mean()),
                "approved_count": n_approved,
            }
        )

    return pd.DataFrame(rows)
