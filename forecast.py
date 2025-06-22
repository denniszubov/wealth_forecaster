import streamlit as st
from dataclasses import dataclass
import pandas as pd

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
    target_retirement_net_worth: float
    desired_annual_retirement_spending: float
    withdrawal_rate: float  # decimal

def generate_projection(parameters: FinancialInputs) -> pd.DataFrame:
    age_series = list(range(parameters.current_age, 101))
    net_worth_history, contribution_history, gain_history = [], [], []

    gross_income = parameters.current_gross_income
    net_worth = parameters.current_net_worth
    after_tax_return_rate = parameters.expected_annual_return_rate * (1 - parameters.capital_gains_tax_rate)
    cumulative_contributions = 0.0

    for age in age_series:
        if age > parameters.current_age:
            gross_income *= 1 + parameters.annual_income_growth_rate

        after_tax_income = gross_income * (1 - parameters.income_tax_rate)
        eligible_match_base = min(parameters.employer_match_cap, parameters.savings_rate) * gross_income
        annual_contribution = parameters.savings_rate * after_tax_income + parameters.employer_match_rate * eligible_match_base

        net_worth *= 1 + after_tax_return_rate
        net_worth += annual_contribution

        cumulative_contributions += annual_contribution
        gain = net_worth - parameters.current_net_worth - cumulative_contributions

        net_worth_history.append(net_worth)
        contribution_history.append(annual_contribution)
        gain_history.append(gain)

    return pd.DataFrame(
        {
            "Age": age_series,
            "Net Worth": net_worth_history,
            "Annual Contributions": contribution_history,
            "Investment Gains": gain_history,
        }
    )

def find_retirement_age(parameters: FinancialInputs, projection: pd.DataFrame) -> tuple[int | None, float | None, float | None]:
    """Returns (age, required_capital_nominal, spending_nominal) or (None, None, None)."""
    if parameters.desired_annual_retirement_spending > 0:
        for _, row in projection.iterrows():
            years_from_now = row["Age"] - parameters.current_age
            spending_nominal = parameters.desired_annual_retirement_spending * (
                (1 + parameters.inflation_rate) ** years_from_now
            )
            required_capital = spending_nominal / parameters.withdrawal_rate
            if row["Net Worth"] >= required_capital:
                return int(row["Age"]), required_capital, spending_nominal
        return None, None, None
    else:
        hits = projection[projection["Net Worth"] >= parameters.target_retirement_net_worth]
        if hits.empty:
            return None, None, None
        age_hit = int(hits.iloc[0]["Age"])
        return age_hit, parameters.target_retirement_net_worth, None

def app() -> None:
    st.set_page_config(page_title="Wealth Forecast", layout="wide")
    st.title("Wealth Forecast")

    with st.sidebar:
        st.header("Inputs")
        current_age = st.number_input("Current Age", 18, 80, 30)
        current_net_worth = st.number_input("Current Net Worth ($)", 0.0, step=1000.0)
        current_gross_income = st.number_input("Current Gross Income ($)", 0.0, step=1000.0)
        annual_income_growth_rate = st.slider("Annual Income Growth (%)", 0.0, 15.0, 3.0) / 100
        savings_rate = st.slider("Savings Rate (% of after‑tax income)", 0.0, 100.0, 20.0) / 100
        employer_match_rate = st.slider("Employer Match Rate (%)", 0.0, 100.0, 50.0) / 100
        employer_match_cap = st.slider("Employer Match Cap (% of income)", 0.0, 100.0, 6.0) / 100
        expected_annual_return_rate = st.slider("Expected Annual Return (%)", -5.0, 15.0, 7.0) / 100
        income_tax_rate = st.slider("Marginal Income Tax Rate (%)", 0.0, 50.0, 30.0) / 100
        capital_gains_tax_rate = st.slider("Capital Gains Tax Rate (%)", 0.0, 40.0, 20.0) / 100
        inflation_rate = st.slider("Inflation Rate (%)", 0.0, 10.0, 2.5) / 100

        st.markdown("---")
        st.subheader("Retirement Goal")
        desired_annual_retirement_spending = st.number_input(
            "Desired Annual Spending in Retirement ($, today's dollars)", 0.0, step=1000.0, value=80000.0
        )
        withdrawal_rate = st.slider("Withdrawal Rate (%)", 2.0, 10.0, 4.0) / 100
        st.caption("Set spending to 0 if you prefer to target a fixed net‑worth instead.")
        target_retirement_net_worth = st.number_input(
            "Desired Retirement Wealth ($)", 0.0, step=50000.0, value=0.0
        )

    inputs = FinancialInputs(
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
        target_retirement_net_worth=target_retirement_net_worth,
        desired_annual_retirement_spending=desired_annual_retirement_spending,
        withdrawal_rate=withdrawal_rate,
    )

    projection = generate_projection(inputs)
    retirement_age, required_capital_nominal, spending_nominal = find_retirement_age(inputs, projection)

    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    if inputs.desired_annual_retirement_spending > 0:
        col1.metric("Annual Spending Goal (today $)", f"{inputs.desired_annual_retirement_spending:,.0f}")
        col2.metric("Withdrawal Rate", f"{inputs.withdrawal_rate*100:.1f}%")
        col3.metric("Age When Funding Goal Hit", retirement_age or "Not reached")
        if retirement_age:
            st.write(
                f"At age **{retirement_age}**, required portfolio = **${required_capital_nominal:,.0f}**, supporting annual spending ≈ **${spending_nominal:,.0f}** in that year's dollars."
            )
    else:
        col1.metric("Retirement Target ($)", f"{inputs.target_retirement_net_worth:,.0f}")
        col2.metric("Age When Target Hit", retirement_age or "Not reached")
        col3.metric("—", "—")

    st.subheader("Net Worth Projection")
    st.line_chart(projection.set_index("Age")["Net Worth"])

    st.subheader("Contributions vs Investment Gains")
    st.area_chart(projection.set_index("Age")[["Annual Contributions", "Investment Gains"]])

    st.dataframe(projection, use_container_width=True)

if __name__ == "__main__":
    app()
