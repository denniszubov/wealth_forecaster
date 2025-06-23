import math
from dataclasses import dataclass

import pandas as pd
import streamlit as st

# Constants
MAX_AGE = 120  # simulation horizon if goal never reached

# Default values
DEFAULT_AGE = 30
DEFAULT_NET_WORTH = 0.0
DEFAULT_INCOME = 60000.0
DEFAULT_INCOME_GROWTH = 0.03  # 3%
DEFAULT_BASE_SAVINGS_RATE = 0.20  # 20%
DEFAULT_MAX_SAVINGS_RATE = 0.50  # 50%
DEFAULT_SAVINGS_ELASTICITY = 1.0
DEFAULT_EMPLOYER_MATCH = 0.0  # 0%
DEFAULT_EMPLOYER_MATCH_CAP = 0.0  # 0%
DEFAULT_ANNUAL_RETURN = 0.09  # 9%
DEFAULT_INCOME_TAX = 0.35  # 30%
DEFAULT_CAPITAL_GAINS_TAX = 0.20  # 20%
DEFAULT_INFLATION = 0.03  # 3%
DEFAULT_SPENDING_GOAL = 60000.0
DEFAULT_WITHDRAWAL_RATE = 0.04  # 4%

# UI formatting
PERCENTAGE_FORMAT = "%.1f"
CURRENCY_FORMAT = "%.0f"

YEARLY_COLUMNS = [
    "Age",
    "Gross Income",
    "Effective Savings Rate",
    "Annual Contributions",
    "Net Worth",
    "Investment Gains",
]

# Savings elasticity descriptions
ELASTICITY_DESCRIPTION = (
    "**Savings elasticity** controls how fast your savings‑rate glide path rises from the *base* toward the *max* as your income grows."
    "\n• **0.3 – Slow burner:** Halfway to max after ~90 % pay growth."
    "\n• **1.0 – Balanced (default):** Halfway after ~40 % pay bump."
    "\n• **2.0 – Turbo:** Halfway once income is ~20 % above today." 
    "\nPick a smaller value if expenses rise with income; larger if raises go mostly to savings."
)

@dataclass
class FinancialInputs:
    # Core profile
    current_age: int
    current_net_worth: float
    current_gross_income: float

    # Growth & saving dynamics
    annual_income_growth_rate: float
    base_savings_rate: float
    max_savings_rate: float
    savings_elasticity: float  # how quickly savings ramps toward max

    # Employer + market
    employer_match_rate: float
    employer_match_cap: float
    expected_annual_return_rate: float

    # Taxes & inflation
    income_tax_rate: float
    capital_gains_tax_rate: float
    inflation_rate: float

    # Retirement target
    desired_annual_retirement_spending: float  # today‑dollar spending goal
    withdrawal_rate: float  # safe‑withdrawal %


# ---------- helper functions ----------

def effective_savings_rate(
    base: float,
    max_rate: float,
    elasticity: float,
    income_now: float,
    income_base: float,
) -> float:
    """Smoothly increase savings rate toward max as income rises."""
    growth_factor = (income_now / income_base) - 1  # 0 when income == base
    rate = base + (max_rate - base) * (1 - math.exp(-elasticity * growth_factor))
    return min(rate, max_rate)


# ---------- projection engine ----------

def build_projection(params: FinancialInputs):
    rows: list[list[float]] = []

    age = params.current_age
    income = params.current_gross_income
    income_base = income  # reference point for savings‑rate ramp

    net_worth = params.current_net_worth
    after_tax_return = params.expected_annual_return_rate * (1 - params.capital_gains_tax_rate)

    cumulative_contributions = 0.0
    initial_net_worth = net_worth
    annual_contribution = 0.0  # first year placeholder

    target_age = None
    target_capital = None
    target_spending_nominal = None

    while age <= MAX_AGE:
        years_from_start = age - params.current_age
        spending_nominal = params.desired_annual_retirement_spending * (1 + params.inflation_rate) ** years_from_start
        required_capital = spending_nominal / params.withdrawal_rate

        # --- record current year ---
        rows.append(
            [
                age,
                income,
                effective_savings_rate(
                    params.base_savings_rate,
                    params.max_savings_rate,
                    params.savings_elasticity,
                    income,
                    income_base,
                ),
                annual_contribution,
                net_worth,
                net_worth - initial_net_worth - cumulative_contributions,
            ]
        )

        # Check if goal hit after updating previous year
        if net_worth >= required_capital and target_age is None:
            target_age, target_capital, target_spending_nominal = age, required_capital, spending_nominal
            break

        # --- advance to next year ---
        age += 1
        if age > MAX_AGE:
            break

        # update income
        income *= 1 + params.annual_income_growth_rate
        # dynamic savings rate
        save_rate = effective_savings_rate(
            params.base_savings_rate,
            params.max_savings_rate,
            params.savings_elasticity,
            income,
            income_base,
        )
        # contributions
        after_tax_income = income * (1 - params.income_tax_rate)
        eligible_match_base = min(params.employer_match_cap, save_rate) * income
        annual_contribution = save_rate * after_tax_income + params.employer_match_rate * eligible_match_base

        # portfolio growth
        net_worth *= 1 + after_tax_return
        net_worth += annual_contribution
        cumulative_contributions += annual_contribution

    df = pd.DataFrame(rows, columns=YEARLY_COLUMNS)
    return df, {
        "age": target_age,
        "required_capital": target_capital,
        "spending_nominal": target_spending_nominal,
    }


# ---------- streamlit UI ----------

def app():
    st.set_page_config(page_title="Wealth Forecast", layout="wide")
    st.title("Wealth Forecast")

    # ----- sidebar inputs -----
    with st.sidebar:
        st.header("Profile & Income")
        current_age = st.number_input("Current Age", 18, 80, DEFAULT_AGE, step=1)
        st.caption("Your age today. Starting point for the simulation.")

        current_net_worth = st.number_input("Current Net Worth ($)", 0.0, step=1000.0, format=CURRENCY_FORMAT, value=DEFAULT_NET_WORTH)
        st.caption("Total assets minus debt — pretax dollars.")

        current_income = st.number_input("Current Gross Income ($)", 0.0, step=1000.0, format=CURRENCY_FORMAT, value=DEFAULT_INCOME)
        st.caption("Annual salary + bonus before tax.")

        income_growth = st.slider("Annual Income Growth (%)", 0.0, 15.0, DEFAULT_INCOME_GROWTH*100) / 100
        st.caption("Average raise each year. 3 % doubles income in ~24 yr.")

        st.header("Saving Behaviour")
        base_save = st.slider("Base Savings Rate (% of take‑home)", 0.0, 100.0, DEFAULT_BASE_SAVINGS_RATE*100) / 100
        st.caption("Minimum savings rate today.")

        max_save = st.slider("Max Savings Rate (%)", int(base_save * 100), 100, int(DEFAULT_MAX_SAVINGS_RATE*100)) / 100
        st.caption("Upper‑limit savings rate once income is high.")

        elasticity = st.slider(
            "Savings Elasticity (rate ramp speed)",
            0.1,
            3.0,
            DEFAULT_SAVINGS_ELASTICITY,
            0.1,
        )
        st.caption(ELASTICITY_DESCRIPTION)

        st.header("Employer & Returns")
        employer_match_rate = st.slider("Employer Match (%)", 0.0, 100.0, DEFAULT_EMPLOYER_MATCH*100) / 100
        employer_match_cap = st.slider("Match Cap (% of income)", 0.0, 100.0, DEFAULT_EMPLOYER_MATCH_CAP*100) / 100
        expected_return = st.slider("Expected Annual Return (%)", -5.0, 15.0, DEFAULT_ANNUAL_RETURN*100) / 100

        st.header("Taxes & Inflation")
        income_tax = st.slider("Marginal Income Tax (%)", 0.0, 50.0, DEFAULT_INCOME_TAX*100) / 100
        cg_tax = st.slider("Capital‑Gains Tax (%)", 0.0, 40.0, DEFAULT_CAPITAL_GAINS_TAX*100) / 100
        inflation = st.slider("Inflation (%)", 0.0, 10.0, DEFAULT_INFLATION*100) / 100

        st.markdown("---")
        st.subheader("Retirement Goal")
        spend_goal = st.number_input("Desired Annual Spending (today $)", 10000.0, step=1000.0, format=CURRENCY_FORMAT, value=DEFAULT_SPENDING_GOAL)
        withdrawal = st.slider("Withdrawal Rate (%)", 2.0, 10.0, DEFAULT_WITHDRAWAL_RATE*100) / 100

    # build params
    params = FinancialInputs(
        current_age=current_age,
        current_net_worth=current_net_worth,
        current_gross_income=current_income,
        annual_income_growth_rate=income_growth,
        base_savings_rate=base_save,
        max_savings_rate=max_save,
        savings_elasticity=elasticity,
        employer_match_rate=employer_match_rate,
        employer_match_cap=employer_match_cap,
        expected_annual_return_rate=expected_return,
        income_tax_rate=income_tax,
        capital_gains_tax_rate=cg_tax,
        inflation_rate=inflation,
        desired_annual_retirement_spending=spend_goal,
        withdrawal_rate=withdrawal,
    )

    projection, details = build_projection(params)

    # ----- summary -----
    st.subheader("Retirement Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Spending Goal (today $)", f"${spend_goal:,.0f}")
    col2.metric("Withdrawal Rate", f"{withdrawal*100:.1f}%")
    col3.metric("Required Portfolio ($)", f"${details['required_capital']:,.0f}")
    col4.metric("Age Reached", details['age'] or "Not reached")

    if details["age"] is not None:
        st.info(
            f"At age **{details['age']}**, portfolio target is **\${details['required_capital']:,.0f}**, providing **\${details['spending_nominal']:,.0f}** per year in that year's dollars."
        )

    # ----- charts -----
    st.subheader("Income Growth vs Age")
    st.line_chart(projection.set_index("Age")["Gross Income"], height=300)
    
    st.subheader("Savings Rate Progression")
    st.line_chart(projection.set_index("Age")["Effective Savings Rate"], height=300)

    st.subheader("Net Worth Projection")
    st.line_chart(projection.set_index("Age")["Net Worth"], height=300)
    
    st.subheader("Contributions vs Investment Gains")
    st.area_chart(
        projection.set_index("Age")[["Annual Contributions", "Investment Gains"]], height=300
    )

    # ----- detailed table -----
    st.subheader("Detailed Projection (rounded)")
    rounded = projection.copy()
    for col in ["Gross Income", "Annual Contributions", "Net Worth", "Investment Gains"]:
        rounded[col] = rounded[col].round(0).apply(lambda x: f"{x:,.0f}")
    rounded["Effective Savings Rate"] = (
        rounded["Effective Savings Rate"] * 100
    ).round(1).apply(lambda x: f"{x}%")

    st.dataframe(rounded.set_index("Age"), use_container_width=True)


# ---------- run ----------

if __name__ == "__main__":
    app()
