import streamlit as st
from dataclasses import dataclass
import pandas as pd

MAX_AGE = 120  # fallback horizon if goal never reached

@dataclass
class FinancialInputs:
    current_age: int
    current_net_worth: float
    current_gross_income: float
    annual_income_growth_rate: float
    savings_rate: float
    employer_match_rate: float
    employer_match_cap: float
    expected_annual_return_rate: float
    income_tax_rate: float
    capital_gains_tax_rate: float
    inflation_rate: float
    desired_annual_retirement_spending: float  # today’s dollars
    withdrawal_rate: float  # decimal


def build_projection(params: FinancialInputs) -> tuple[pd.DataFrame, dict]:
    """Simulate year-by-year until target portfolio is met or MAX_AGE."""
    ages: list[int] = []
    net_worth_list: list[float] = []
    contributions_list: list[float] = []
    gains_list: list[float] = []

    age = params.current_age
    gross_income = params.current_gross_income
    net_worth = params.current_net_worth
    after_tax_return = params.expected_annual_return_rate * (1 - params.capital_gains_tax_rate)

    initial_net_worth = net_worth
    cumulative_contributions = 0.0

    target_age: int | None = None
    target_capital: float | None = None
    target_spending_nominal: float | None = None

    while age <= MAX_AGE:
        # Calculate this year's required capital (inflation-adjusted spending / withdrawal)
        years_from_now = age - params.current_age
        spending_nominal = params.desired_annual_retirement_spending * (
            1 + params.inflation_rate
        ) ** years_from_now
        required_capital = spending_nominal / params.withdrawal_rate

        ages.append(age)
        net_worth_list.append(net_worth)
        contributions_list.append(0.0 if age == params.current_age else annual_contribution)  # type: ignore[name-defined]
        gains_list.append(net_worth - initial_net_worth - cumulative_contributions)

        # Check if goal reached this year (post-growth and contributions from previous year)
        if net_worth >= required_capital and target_age is None:
            target_age = age
            target_capital = required_capital
            target_spending_nominal = spending_nominal
            break  # stop projection once goal reached

        # --- advance to next year ---
        age += 1
        if age > MAX_AGE:
            break

        # Grow income
        gross_income *= 1 + params.annual_income_growth_rate
        # Contributions
        after_tax_income = gross_income * (1 - params.income_tax_rate)
        eligible_match_base = min(params.employer_match_cap, params.savings_rate) * gross_income
        annual_contribution = params.savings_rate * after_tax_income + params.employer_match_rate * eligible_match_base
        # Market return, then add contribution
        net_worth *= 1 + after_tax_return
        net_worth += annual_contribution
        cumulative_contributions += annual_contribution

    # If not reached, final required_capital from last loop iteration
    if target_age is None:
        target_age = None
        target_capital = required_capital  # type: ignore
        target_spending_nominal = None

    projection_df = pd.DataFrame(
        {
            "Age": ages,
            "Net Worth": net_worth_list,
            "Annual Contributions": contributions_list,
            "Investment Gains": gains_list,
        }
    )

    details = {
        "age": target_age,
        "required_capital": target_capital,
        "spending_nominal": target_spending_nominal,
    }
    return projection_df, details


def app():
    st.set_page_config(page_title="Wealth Forecast", layout="wide")
    st.title("Wealth Forecast")

    # ----- SIDEBAR INPUTS -----
    with st.sidebar:
        st.header("Profile & Income")
        current_age = st.number_input("Current Age", 18, 80, 23, step=1)
        current_net_worth = st.number_input("Current Net Worth ($)", 0.0, step=1000.0, format="%.0f")
        current_gross_income = st.number_input("Current Gross Income ($)", 0.0, step=1000.0, format="%.0f", value=60000.0)
        annual_income_growth_rate = st.slider("Annual Income Growth (%)", 0.0, 15.0, 3.0) / 100

        st.header("Saving & Investing")
        savings_rate = st.slider("Savings Rate (% of after-tax income)", 0.0, 100.0, 20.0) / 100
        employer_match_rate = st.slider("Employer Match Rate (%)", 0.0, 100.0, 0.0) / 100
        employer_match_cap = st.slider("Employer Match Cap (% of income)", 0.0, 100.0, 0.0) / 100
        expected_annual_return_rate = st.slider("Expected Annual Return (%)", -5.0, 15.0, 7.0) / 100

        st.header("Taxes & Inflation")
        income_tax_rate = st.slider("Marginal Income Tax Rate (%)", 0.0, 50.0, 30.0) / 100
        capital_gains_tax_rate = st.slider("Capital Gains Tax Rate (%)", 0.0, 40.0, 20.0) / 100
        inflation_rate = st.slider("Inflation Rate (%)", 0.0, 10.0, 3.0) / 100

        st.markdown("---")
        st.subheader("Retirement Goal")
        desired_annual_retirement_spending = st.number_input(
            "Desired Annual Spending in Retirement ($, today's dollars)",
            10000.0,
            step=1000.0,
            format="%.0f",
            value=60000.0,
        )
        withdrawal_rate = st.slider("Withdrawal Rate (%)", 2.0, 10.0, 4.0) / 100

    # ----- BUILD PROJECTION -----
    params = FinancialInputs(
        current_age=current_age,
        current_net_worth=current_net_worth,
        current_gross_income=current_gross_income,
        annual_income_growth_rate=annual_income_growth_rate,
        savings_rate=savings_rate,
        employer_match_rate=employer_match_rate,
        employer_match_cap=employer_match_cap,
        expected_annual_return_rate=expected_annual_return_rate,
        income_tax_rate=income_tax_rate,
        capital_gains_tax_rate=capital_gains_tax_rate,
        inflation_rate=inflation_rate,
        desired_annual_retirement_spending=desired_annual_retirement_spending,
        withdrawal_rate=withdrawal_rate,
    )

    projection, details = build_projection(params)

    # ----- SUMMARY -----
    st.subheader("Retirement Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual Spending Goal (today $)", f"${desired_annual_retirement_spending:,.0f}")
    col2.metric("Withdrawal Rate", f"{withdrawal_rate*100:.1f}%")
    col3.metric("Required Portfolio ($)", f"${details['required_capital']:,.0f}")
    col4.metric("Age Reached", details["age"] or "Not reached")

    if details["age"] is not None:
        st.info(
            f"At age **{details['age']}**, portfolio target is **${details['required_capital']:,.0f}**, supporting about **${details['spending_nominal']:,.0f}** of spending in that year's dollars."
        )

    # ----- CHARTS & TABLE -----
    st.subheader("Net Worth Projection")
    st.line_chart(
        projection.set_index("Age")["Net Worth"],
        height=400,
    )

    st.subheader("Contributions vs Investment Gains")
    st.area_chart(
        projection.set_index("Age")[["Annual Contributions", "Investment Gains"]],
        height=400,
    )

    st.subheader("Data Table (rounded)")
    display_df = projection.copy()
    for col in ["Net Worth", "Annual Contributions", "Investment Gains"]:
        display_df[col] = display_df[col].round(0).apply(lambda x: f"{x:,.0f}")
    st.dataframe(display_df, use_container_width=True)


if __name__ == "__main__":
    app()