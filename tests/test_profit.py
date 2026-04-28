import re

import numpy as np
import pandas as pd
import pytest

from src.engine.profit import (
    expected_profit_per_loan,
    portfolio_profit_curve,
    profit_params,
    realized_profit_per_loan,
)


def test_profit_params_prefers_empirical_cashflow_inputs():
    config = {
        "profit": {
            "avg_loan_amount": 10000,
            "avg_interest_rate": 0.12,
            "avg_loan_term_years": 3,
            "loss_given_default": 0.5,
            "avg_net_revenue_if_repaid": 1800,
            "avg_net_loss_if_default": 4200,
            "servicing_cost_per_loan": 25,
            "false_negative_loss_multiplier": 1.1,
        }
    }

    assert profit_params(config) == (1800.0, 4200.0, 25.0, 1.1)


def test_expected_and_realized_profit_use_same_unit_economics():
    p_default = np.array([0.1, 0.6])
    revenue = 2000
    loss = 5000
    servicing_cost = 50
    multiplier = 1.2

    expected = expected_profit_per_loan(
        p_default,
        revenue,
        loss,
        servicing_cost=servicing_cost,
        fn_loss_multiplier=multiplier,
    )
    realized = realized_profit_per_loan(
        np.array([0, 1]),
        revenue,
        loss,
        servicing_cost=servicing_cost,
        fn_loss_multiplier=multiplier,
    )

    np.testing.assert_allclose(expected, np.array([1150.0, -2850.0]))
    np.testing.assert_allclose(realized, np.array([1950.0, -6050.0]))


def test_portfolio_profit_curve_uses_realized_outcomes_for_approved_loans():
    curve = portfolio_profit_curve(
        p_default=np.array([0.05, 0.2, 0.8]),
        y_true=np.array([0, 1, 1]),
        thresholds=np.array([0.1, 0.5, 1.0]),
        revenue_if_repaid=1000,
        loss_if_default=4000,
        servicing_cost=100,
        fn_loss_multiplier=1.0,
    )

    expected = pd.DataFrame(
        {
            "threshold": [0.1, 0.5, 1.0],
            "portfolio_profit": [900.0, -3200.0, -7300.0],
            "approval_rate": [1 / 3, 2 / 3, 1.0],
            "approved_count": [1, 2, 3],
        }
    )

    pd.testing.assert_frame_equal(curve.reset_index(drop=True), expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "p_default": [0.2, np.nan],
                "revenue_if_repaid": 1,
                "loss_if_default": 1,
            },
                re.escape("p_default contains NaN values."),
            ),
            (
                {
                    "p_default": [-0.1, 0.2],
                    "revenue_if_repaid": 1,
                    "loss_if_default": 1,
                },
                re.escape("p_default must stay within [0, 1]."),
            ),
        ],
)
def test_expected_profit_validates_probability_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        expected_profit_per_loan(**kwargs)


def test_portfolio_profit_curve_validates_input_lengths():
    with pytest.raises(ValueError, match="same length"):
        portfolio_profit_curve(
            p_default=[0.1, 0.2],
            y_true=[0],
            thresholds=[0.5],
            revenue_if_repaid=1,
            loss_if_default=1,
        )
