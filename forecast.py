import math
from dataclasses import dataclass

import pandas as pd
import streamlit as st

# ----------------------------
# Constants & Default Settings
# ----------------------------
MAX_AGE = 120  # simulation horizon if goal never reached

CURRENCY_SETTINGS = {
    "USD": {
        "label": "USD (United States)",
        "symbol": "$",
        "NET_WORTH": 0.0,
        "INCOME": 60_000.0,
        "INCOME_GROWTH": 0.03,  # 3 %
        "INFLATION": 0.03,  # 3 %
        "SPENDING_GOAL": 60_000.0,
        "INCOME_TAX": 0.35,  # 35 %
        "CAPITAL_GAINS_TAX": 0.20,  # 20 %
    },
    "ZAR": {
        "label": "ZAR (South Africa)",
        "symbol": "R",
        "NET_WORTH": 0.0,
        "INCOME": 320_000.0,  # ≈ R26 600 / month
        "INCOME_GROWTH": 0.04,  # 4 %
        "INFLATION": 0.05,  # 5 %
        "SPENDING_GOAL": 300_000.0,
        "INCOME_TAX": 0.30,  # 30 %
        "CAPITAL_GAINS_TAX": 0.18,  # ≈ 40 % inclusion × 45 % bracket
    },
}

# The remaining parameters seldom change by country, so we keep single defaults
DEFAULT_BASE_SAVINGS_RATE = 0.20  # 20 %
DEFAULT_MAX_SAVINGS_RATE = 0.50  # 50 %
DEFAULT_SAVINGS_ELASTICITY = 1.0
DEFAULT_EMPLOYER_MATCH = 0.0  # 0 %
DEFAULT_EMPLOYER_MATCH_CAP = 0.0  # 0 %
DEFAULT_ANNUAL_RETURN = 0.09  # 9 %
DEFAULT_WITHDRAWAL_RATE = 0.04  # 4 %

# UI formatting helpers
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

ELASTICITY_DESCRIPTION = (
    "**Savings elasticity** controls how fast your savings‑rate glide path rises from the *base* toward the *max* as your income grows."
    "\n• **0.3 – Slow burner:** Halfway to max after ~90 % pay growth."
    "\n• **1.0 – Balanced (default):** Halfway after ~40 % pay bump."
    "\n• **2.0 – Turbo:** Halfway once income is ~20 % above today." 
    "\nPick a smaller value if expenses rise with income; larger if raises go mostly to savings."
)


# ----------------------------
# Dataclasses
# ----------------------------
@dataclass
class CareerPhase:
    """Represents a phase in a person's career with specific income growth characteristics"""

    start_age: int
    end_age: int
    growth_rate: float
    description: str = ""


@dataclass
class FinancialInputs:
    # Core profile
    current_age: int
    current_net_worth: float
    current_gross_income: float

    # Growth & saving dynamics
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
    desired_annual_retirement_spending: float  # today‑currency spending goal
    withdrawal_rate: float  # safe‑withdrawal %

    # Growth model
    career_phases: list[CareerPhase]

    # Optional – income ceiling
    income_ceiling: float = float("inf")


# ----------------------------
# Helper Functions
# ----------------------------

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


# ----------------------------
# Projection Engine
# ----------------------------

def build_projection(params: FinancialInputs):
    rows: list[list[float]] = []

    age = params.current_age
    income = params.current_gross_income
    income_base = income  # reference point for savings‑rate ramp

    net_worth = params.current_net_worth
    after_tax_return = params.expected_annual_return_rate * (1 - params.capital_gains_tax_rate)

    cumulative_contributions = 0.0
    initial_net_worth = net_worth
    annual_contribution = 0.0  # first‑year placeholder

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

        # Update income using phase‑based growth
        current_phase = next(
            (
                p
                for p in params.career_phases
                if p.start_age <= age <= p.end_age
            ),
            CareerPhase(age, age, 0.0, "No phase defined"),
        )
        income *= 1 + current_phase.growth_rate

        # Apply income ceiling
        if params.income_ceiling < float("inf"):
            proximity_factor = max(0.0, 1 - (income / params.income_ceiling))
            income = min(income, params.income_ceiling * (1 - 0.05 * proximity_factor))

        # Dynamic savings rate
        save_rate = effective_savings_rate(
            params.base_savings_rate,
            params.max_savings_rate,
            params.savings_elasticity,
            income,
            income_base,
        )

        # Contributions
        after_tax_income = income * (1 - params.income_tax_rate)
        eligible_match_base = min(params.employer_match_cap, save_rate) * income
        annual_contribution = save_rate * after_tax_income + params.employer_match_rate * eligible_match_base

        # Portfolio growth
        net_worth *= 1 + after_tax_return
        net_worth += annual_contribution
        cumulative_contributions += annual_contribution

    df = pd.DataFrame(rows, columns=YEARLY_COLUMNS)
    return df, {
        "age": target_age,
        "required_capital": target_capital,
        "spending_nominal": target_spending_nominal,
    }


# ----------------------------
# Streamlit UI
# ----------------------------

def app():
    st.set_page_config(page_title="Wealth Forecast", layout="wide")
    st.title("Wealth Forecast")

    # ----- currency selector -----
    with st.sidebar:
        st.header("Global Settings")
        currency_key = st.selectbox(
            "Currency / Country",
            [CURRENCY_SETTINGS[k]["label"] for k in CURRENCY_SETTINGS],
            index=0,
        )

    # Resolve key back from the label
    currency_code = next(
        k for k, v in CURRENCY_SETTINGS.items() if v["label"] == currency_key
    )
    cur = CURRENCY_SETTINGS[currency_code]
    symbol = cur["symbol"]

    # ----- sidebar inputs -----
    with st.sidebar:
        st.markdown("---")
        st.header("Profile & Income")
        current_age = st.number_input("Current Age", 18, 80, 30, step=1)
        st.caption("Your age today. Starting point for the simulation.")

        current_net_worth = st.number_input(
            f"Current Net Worth ({symbol})",
            0.0,
            step=1000.0,
            format=CURRENCY_FORMAT,
            value=cur["NET_WORTH"],
        )
        st.caption("Total assets minus debt — pretax.")

        current_income = st.number_input(
            f"Current Gross Income ({symbol})",
            0.0,
            step=1000.0,
            format=CURRENCY_FORMAT,
            value=cur["INCOME"],
        )
        st.caption("Annual salary + bonus before tax.")

        st.header("Income Growth Model")
        model_type = st.radio("Income Growth Model", ["Simple", "Career Phase"])

        if model_type == "Simple":
            income_growth = (
                st.slider(
                    "Annual Income Growth (%)",
                    0.0,
                    15.0,
                    cur["INCOME_GROWTH"] * 100,
                )
                / 100
            )
            st.caption("Average raise each year.")
            career_phases = [
                CareerPhase(current_age, MAX_AGE, income_growth, "Constant Growth")
            ]
            income_ceiling = float("inf")
        else:
            st.write("Define growth rates for career phases:")
            col1, col2 = st.columns(2)
            with col1:
                early_career_growth = st.slider("Early Career Growth (%)", 0.0, 20.0, 7.0) / 100
                mid_career_growth = st.slider("Mid Career Growth (%)", 0.0, 15.0, 4.0) / 100
                late_career_growth = st.slider("Late Career Growth (%)", 0.0, 10.0, 2.0) / 100
            with col2:
                early_end = st.number_input(
                    "Early Career Ends (age)", current_age + 1, 50, min(current_age + 10, 40)
                )
                mid_end = st.number_input(
                    "Mid Career Ends (age)", early_end + 1, 70, min(early_end + 15, 60)
                )
            career_phases = [
                CareerPhase(current_age, early_end, early_career_growth, "Early Career"),
                CareerPhase(early_end + 1, mid_end, mid_career_growth, "Mid Career"),
                CareerPhase(mid_end + 1, MAX_AGE, late_career_growth, "Late Career"),
            ]

            use_ceiling = st.checkbox("Apply Income Ceiling", value=False)
            income_ceiling = (
                st.number_input(
                    f"Income Ceiling ({symbol})",
                    current_income * 1.5,
                    1_000_000.0,
                    250_000.0,
                    10_000.0,
                )
                if use_ceiling
                else float("inf")
            )

        st.header("Saving Behaviour")
        base_save = (
            st.slider("Base Savings Rate (% of take‑home)", 0.0, 100.0, DEFAULT_BASE_SAVINGS_RATE * 100) / 100
        )
        st.caption("Minimum savings rate today.")

        initial_max_save_value = max(DEFAULT_MAX_SAVINGS_RATE * 100, base_save * 100)
        max_save = st.slider("Max Savings Rate (%)", 0.0, 100.0, initial_max_save_value) / 100
        if max_save < base_save:
            st.warning("Max savings rate cannot be lower than base savings rate. Adjusting to match.")
            max_save = base_save
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
        employer_match_rate = (
            st.slider("Employer Match (%)", 0.0, 100.0, DEFAULT_EMPLOYER_MATCH * 100) / 100
        )
        employer_match_cap = (
            st.slider("Match Cap (% of income)", 0.0, 100.0, DEFAULT_EMPLOYER_MATCH_CAP * 100) / 100
        )
        expected_return = (
            st.slider("Expected Annual Return (%)", -5.0, 15.0, DEFAULT_ANNUAL_RETURN * 100) / 100
        )

        st.header("Taxes & Inflation")
        income_tax = (
            st.slider("Marginal Income Tax (%)", 0.0, 50.0, cur["INCOME_TAX"] * 100) / 100
        )
        cg_tax = (
            st.slider("Capital‑Gains Tax (%)", 0.0, 40.0, cur["CAPITAL_GAINS_TAX"] * 100) / 100
        )
        inflation = (
            st.slider("Inflation (%)", 0.0, 10.0, cur["INFLATION"] * 100) / 100
        )

        st.markdown("---")
        st.subheader("Retirement Goal")
        spend_goal = st.number_input(
            f"Desired Annual Spending (today {symbol})",
            10_000.0,
            step=1_000.0,
            format=CURRENCY_FORMAT,
            value=cur["SPENDING_GOAL"],
        )
        withdrawal = (
            st.slider("Withdrawal Rate (%)", 2.0, 10.0, DEFAULT_WITHDRAWAL_RATE * 100) / 100
        )

    # Build parameters
    params = FinancialInputs(
        current_age=current_age,
        current_net_worth=current_net_worth,
        current_gross_income=current_income,
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
        career_phases=career_phases,
        income_ceiling=income_ceiling,
    )

    projection, details = build_projection(params)

    # ----------------------------
    # Dashboard Output
    # ----------------------------
    st.subheader("Retirement Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Spending Goal (today)", f"{symbol}{spend_goal:,.0f}")
    col2.metric("Withdrawal Rate", f"{withdrawal * 100:.1f}%")
    col3.metric("Required Portfolio", f"{symbol}{details['required_capital']:,.0f}")
    col4.metric("Age Reached", details["age"] or "Not reached")

    if details["age"] is not None:
        st.info(
            f"At age **{details['age']}**, portfolio target is **{symbol}{details['required_capital']:,.0f}**, providing **{symbol}{details['spending_nominal']:,.0f}** per year in that year's currency."
        )

    # Charts
    st.subheader("Gross Income vs Age")
    st.line_chart(projection.set_index("Age")["Gross Income"], height=300)

    if len(params.career_phases) > 1:
        phase_rows = []
        for phase in params.career_phases:
            if phase.start_age <= MAX_AGE and phase.end_age >= params.current_age:
                for age in range(
                    max(phase.start_age, params.current_age),
                    min(phase.end_age + 1, MAX_AGE + 1),
                ):
                    idx = age - params.current_age
                    if idx < len(projection):
                        phase_rows.append(
                            {
                                "Age": age,
                                "Phase": phase.description,
                                "Growth Rate": f"{phase.growth_rate * 100:.1f}%",
                            }
                        )
        phase_df = pd.DataFrame(phase_rows)
        st.caption("Career Phases:")
        st.dataframe(phase_df, hide_index=True, width=600)

    st.subheader("Savings Rate Progression")
    st.line_chart(projection.set_index("Age")["Effective Savings Rate"], height=300)

    st.subheader("Net Worth Projection")
    st.line_chart(projection.set_index("Age")["Net Worth"], height=300)

    st.subheader("Contributions vs Investment Gains")
    st.area_chart(
        projection.set_index("Age")[["Annual Contributions", "Investment Gains"]], height=300
    )

    # Detailed Table
    st.subheader("Detailed Projection (rounded)")
    rounded = projection.copy()
    for col in [
        "Gross Income",
        "Annual Contributions",
        "Net Worth",
        "Investment Gains",
    ]:
        rounded[col] = rounded[col].round(0).apply(lambda x: f"{symbol}{x:,.0f}")
    rounded["Effective Savings Rate"] = (
        rounded["Effective Savings Rate"] * 100
    ).round(1).apply(lambda x: f"{x}%")

    st.dataframe(rounded.set_index("Age"), use_container_width=True)


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app()
