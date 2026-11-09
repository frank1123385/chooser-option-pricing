
from pathlib import Path
import sys
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "src" / "data"))

from two_stage_volatility_chooser import two_stage_chooser_mc
from week7_auto_update import update_jpm, update_treasury, update_vix


# =========================================================
# 1. Page configuration
# =========================================================

st.set_page_config(
    page_title="Chooser Option Pricing Tool",
    page_icon="📈",
    layout="wide",
)

st.title("Chooser Option Pricing Tool")

st.caption(
    "Two-stage volatility Monte Carlo pricing "
    "with market data update functionality"
)

st.markdown("""
This prototype estimates theoretical Chooser Option prices
using two volatility stages and Monte Carlo simulation.
It also supports incremental market data updates.
""")


# =========================================================
# 2. Market data status
# =========================================================

st.header("Market Data Management")

data_files = {
    "JPM": ROOT / "data" / "raw" / "jpm_daily.csv",
    "Treasury": ROOT / "data" / "raw" / "treasury_daily.csv",
    "VIX": ROOT / "data" / "raw" / "vix_daily.csv",
}


def get_data_status():

    records = []

    for name, file_path in data_files.items():

        if file_path.exists():

            data = pd.read_csv(file_path)

            latest_date = pd.to_datetime(
                data["Date"],
                errors="coerce"
            ).max()

            records.append({
                "Data Source": name,
                "Rows": len(data),
                "Latest Date": (
                    latest_date.strftime("%Y-%m-%d")
                    if pd.notna(latest_date)
                    else "Unavailable"
                ),
                "Status": "Available",
            })

        else:

            records.append({
                "Data Source": name,
                "Rows": 0,
                "Latest Date": "N/A",
                "Status": "File Missing",
            })

    return pd.DataFrame(records)


st.subheader("Current Data Status")

st.dataframe(
    get_data_status(),
    use_container_width=True,
    hide_index=True,
)

if st.button("Refresh Market Data", type="primary"):

    tasks = [
        ("JPM", update_jpm),
        ("Treasury", update_treasury),
        ("VIX", update_vix),
    ]

    errors = []
    logs = []

    progress = st.progress(0)

    for i, (name, task) in enumerate(tasks):

        output = io.StringIO()

        try:

            with st.spinner(f"Updating {name} data..."):

                with redirect_stdout(output):
                    task()

            logs.append(output.getvalue())

        except Exception as error:

            errors.append((name, str(error)))
            logs.append(f"{name} update failed: {error}")

        progress.progress((i + 1) / len(tasks))

    st.subheader("Update Log")

    for log in logs:
        if log.strip():
            st.code(log)

    if errors:

        for name, error in errors:
            st.error(f"{name} update failed: {error}")

        st.warning(
            "Some data sources could not be updated. "
            "Check the error messages above."
        )

    else:

        st.success("All market data sources updated successfully!")

    st.subheader("Updated Data Status")

    st.dataframe(
        get_data_status(),
        use_container_width=True,
        hide_index=True,
    )


st.divider()


# =========================================================
# 3. Pricing parameters
# =========================================================

st.header("Chooser Option Pricing")

st.sidebar.header("Market and Contract Parameters")

s0 = st.sidebar.number_input(
    "Underlying Price (S0)",
    min_value=0.01,
    value=231.45,
    step=1.0,
)

strike = st.sidebar.number_input(
    "Strike Price (K)",
    min_value=0.01,
    value=150.0,
    step=1.0,
)

risk_free_rate = st.sidebar.number_input(
    "Risk-free Rate (%)",
    min_value=-10.0,
    max_value=50.0,
    value=4.58,
    step=0.1,
) / 100.0

sigma_1 = st.sidebar.number_input(
    "First-period Volatility (%)",
    min_value=0.1,
    max_value=300.0,
    value=25.0,
    step=0.5,
) / 100.0

sigma_2 = st.sidebar.number_input(
    "Second-period Volatility (%)",
    min_value=0.1,
    max_value=300.0,
    value=28.0,
    step=0.5,
) / 100.0

t1 = st.sidebar.number_input(
    "Choice Time T1 (years)",
    min_value=0.01,
    max_value=5.0,
    value=0.5,
    step=0.1,
)

t2 = st.sidebar.number_input(
    "Expiration Time T2 (years)",
    min_value=0.02,
    max_value=10.0,
    value=1.0,
    step=0.1,
)

n_simulations = st.sidebar.select_slider(
    "Monte Carlo Simulations",
    options=[1000, 5000, 10000, 20000, 50000],
    value=10000,
)


# =========================================================
# 4. Scenario analysis
# =========================================================

st.sidebar.header("Scenario Analysis")

scenario = st.sidebar.selectbox(
    "Select Scenario",
    [
        "Baseline",
        "Volatility Shock (+50%)",
        "Interest Rate Shock (+2 percentage points)",
    ],
)

scenario_sigma_1 = sigma_1
scenario_sigma_2 = sigma_2
scenario_rate = risk_free_rate

if scenario == "Volatility Shock (+50%)":

    scenario_sigma_1 *= 1.5
    scenario_sigma_2 *= 1.5

elif scenario == "Interest Rate Shock (+2 percentage points)":

    scenario_rate += 0.02


if t2 <= t1:

    st.error(
        "Expiration time T2 must be greater than choice time T1."
    )

    st.stop()


# =========================================================
# 5. Pricing calculation
# =========================================================

st.subheader("Pricing Results")

if st.button("Calculate Chooser Option Price"):

    result = two_stage_chooser_mc(
        S0=s0,
        risk_free_rate=scenario_rate,
        sigma_1=scenario_sigma_1,
        sigma_2=scenario_sigma_2,
        strike=strike,
        T1=t1,
        T2=t2,
        n_simulations=n_simulations,
        seed=42,
    )

    price = result["Chooser_Price"]
    standard_error = result["MC_Standard_Error"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Chooser Option Price",
        f"${price:.4f}",
    )

    col2.metric(
        "Monte Carlo Standard Error",
        f"${standard_error:.4f}",
    )

    col3.metric(
        "95% Confidence Interval",
        f"${result['MC_95CI_Lower']:.4f} - "
        f"${result['MC_95CI_Upper']:.4f}",
    )

    # Parameter table

    st.subheader("Scenario Parameters")

    parameter_data = pd.DataFrame({
        "Parameter": [
            "Underlying Price",
            "Strike Price",
            "Risk-free Rate",
            "First-period Volatility",
            "Second-period Volatility",
            "Choice Time",
            "Expiration Time",
        ],
        "Value": [
            f"${s0:.2f}",
            f"${strike:.2f}",
            f"{scenario_rate * 100:.2f}%",
            f"{scenario_sigma_1 * 100:.2f}%",
            f"{scenario_sigma_2 * 100:.2f}%",
            f"{t1:.2f} years",
            f"{t2:.2f} years",
        ],
    })

    st.dataframe(
        parameter_data,
        use_container_width=True,
        hide_index=True,
    )

    # Decision distribution

    st.subheader("Chooser Decision Distribution")

    call_ratio = result["Call_Ratio"]
    put_ratio = result["Put_Ratio"]

    col1, col2 = st.columns(2)

    col1.metric(
        "Call Choice Ratio",
        f"{call_ratio * 100:.2f}%",
    )

    col2.metric(
        "Put Choice Ratio",
        f"{put_ratio * 100:.2f}%",
    )

    fig, ax = plt.subplots(figsize=(7, 3))

    ax.bar(
        ["Call", "Put"],
        [call_ratio, put_ratio],
    )

    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Chooser Decision Distribution")

    st.pyplot(fig)
    plt.close(fig)

    st.success("Pricing calculation completed.")

    st.caption(
        "The output is a theoretical Monte Carlo estimate, "
        "not a market-validated option price."
    )

else:

    st.info(
        "Adjust the parameters in the sidebar "
        "and click Calculate."
    )
