from pathlib import Path
import sys
import io
from contextlib import redirect_stdout
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "src" / "data"))

# Week 7 functions remain available for the original manual-pricing section.
from two_stage_volatility_chooser import two_stage_chooser_mc
from week7_auto_update import update_jpm, update_treasury, update_vix

# Week 8 dual-pricing Monte Carlo function.
from two_stage_volatility_chooser_week8 import price_mc


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
    "Two-stage Chooser option pricing | Historical-volatility benchmark "
    "and Random Forest alternative"
)

st.markdown(
    """
    This application combines the original Week 7 manual Monte Carlo pricing
    prototype with the Week 8 dual-pricing analysis. The Week 8 comparison uses
    maturity-aligned 126D and 252D volatility forecasts.
    """
)

FEATURE_FILE = ROOT / "data" / "processed" / "jpm_features_week8.csv"
PREDICTION_126_FILE = (
    ROOT / "data" / "processed" / "volatility_experiment_week8"
    / "model_results" / "126d" / "test_predictions.csv"
)
PREDICTION_252_FILE = (
    ROOT / "data" / "processed" / "volatility_experiment_week8"
    / "model_results" / "252d" / "test_predictions.csv"
)
PRICING_DETAIL_FILE = (
    ROOT / "data" / "processed" / "volatility_experiment_week8"
    / "two_stage_chooser" / "week8_dual_chooser_pricing.csv"
)
PRICING_SUMMARY_FILE = (
    ROOT / "data" / "processed" / "volatility_experiment_week8"
    / "two_stage_chooser" / "week8_dual_chooser_summary.csv"
)
METRICS_DIR = (
    ROOT / "data" / "processed" / "volatility_experiment_week8"
    / "model_results"
)


# =========================================================
# 2. Helper functions
# =========================================================

def read_csv_if_exists(path):
    if path.exists():
        return pd.read_csv(path)
    return None


def get_data_status():
    data_files = {
        "JPM": ROOT / "data" / "raw" / "jpm_daily.csv",
        "Treasury": ROOT / "data" / "raw" / "treasury_daily.csv",
        "VIX": ROOT / "data" / "raw" / "vix_daily.csv",
    }

    records = []
    for name, file_path in data_files.items():
        if file_path.exists():
            data = pd.read_csv(file_path)
            date_col = "Date" if "Date" in data.columns else (
                "observation_date" if "observation_date" in data.columns else None
            )
            latest_date = (
                pd.to_datetime(data[date_col], errors="coerce").max()
                if date_col else pd.NaT
            )
            records.append({
                "Data Source": name,
                "Rows": len(data),
                "Latest Date": (
                    latest_date.strftime("%Y-%m-%d")
                    if pd.notna(latest_date) else "Unavailable"
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


@st.cache_data
def load_week8_market_and_forecasts():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Missing file: {FEATURE_FILE}")
    if not PREDICTION_126_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PREDICTION_126_FILE}")
    if not PREDICTION_252_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PREDICTION_252_FILE}")

    f126 = pd.read_csv(PREDICTION_126_FILE)
    f252 = pd.read_csv(PREDICTION_252_FILE)
    features = pd.read_csv(FEATURE_FILE)

    for frame in (f126, f252, features):
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")

    required_pred = {
        "Date", "Historical_60D_volatility", "Tuned_Random_Forest"
    }
    for label, frame in (("126D", f126), ("252D", f252)):
        missing = required_pred - set(frame.columns)
        if missing:
            raise ValueError(f"{label} prediction file missing: {sorted(missing)}")

    required_features = {"Date", "Close", "DGS10"}
    missing = required_features - set(features.columns)
    if missing:
        raise ValueError(f"Feature file missing: {sorted(missing)}")

    f126 = f126[[
        "Date", "Historical_60D_volatility", "Tuned_Random_Forest"
    ]].rename(columns={
        "Historical_60D_volatility": "Historical_126D",
        "Tuned_Random_Forest": "RF_126D",
    })
    f252 = f252[[
        "Date", "Historical_60D_volatility", "Tuned_Random_Forest"
    ]].rename(columns={
        "Historical_60D_volatility": "Historical_252D",
        "Tuned_Random_Forest": "RF_252D",
    })

    forecasts = f126.merge(
        f252, on="Date", how="inner", validate="one_to_one"
    )
    if forecasts.empty:
        raise ValueError("No common dates in 126D and 252D forecast files.")

    hist_diff = (
        forecasts["Historical_126D"] - forecasts["Historical_252D"]
    ).abs()
    if hist_diff.max() > 1e-8:
        raise ValueError("Historical 60D volatility differs across horizon files.")

    market = features[["Date", "Close", "DGS10"]].rename(
        columns={"Close": "S0"}
    )
    market["Risk_Free_Rate"] = pd.to_numeric(
        market["DGS10"], errors="coerce"
    ) / 100.0
    market = market[["Date", "S0", "Risk_Free_Rate"]]

    data = forecasts.merge(
        market, on="Date", how="inner", validate="one_to_one"
    )
    numeric_cols = [
        "S0", "Risk_Free_Rate", "Historical_126D",
        "Historical_252D", "RF_126D", "RF_252D",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=numeric_cols).sort_values("Date")

    if data.empty:
        raise ValueError("No valid matched Week 8 observations.")
    if (data[["Historical_126D", "Historical_252D", "RF_126D", "RF_252D"]] <= 0).any().any():
        raise ValueError("Non-positive volatility forecast found.")

    return data.reset_index(drop=True)


def run_week8_price(row, strategy, strike, t1, t2, simulations, seed=42):
    if strategy == "Historical 60D":
        sigma1 = float(row["Historical_126D"])
        sigma2 = float(row["Historical_252D"])
    else:
        sigma1 = float(row["RF_126D"])
        sigma2 = float(row["RF_252D"])

    result = price_mc(
        S0=float(row["S0"]),
        r=float(row["Risk_Free_Rate"]),
        sigma1=sigma1,
        sigma2=sigma2,
        K=float(strike),
        T1=float(t1),
        T2=float(t2),
        n=int(simulations),
        seed=int(seed),
    )
    return sigma1, sigma2, result


def load_metrics():
    frames = []
    for horizon in (20, 126, 252):
        path = METRICS_DIR / f"{horizon}d" / "metrics.csv"
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "Horizon", f"{horizon}D")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


# =========================================================
# 3. Market data management
# =========================================================

st.header("1. Market Data Management")
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
    progress = st.progress(0)

    for i, (name, task) in enumerate(tasks):
        output = io.StringIO()
        try:
            with st.spinner(f"Updating {name} data..."):
                with redirect_stdout(output):
                    task()
            if output.getvalue().strip():
                st.code(output.getvalue())
        except Exception as error:
            errors.append((name, str(error)))
            st.error(f"{name} update failed: {error}")
        progress.progress((i + 1) / len(tasks))

    if errors:
        st.warning("Some data sources could not be updated.")
    else:
        st.success("All market data sources updated successfully.")

    st.dataframe(
        get_data_status(),
        use_container_width=True,
        hide_index=True,
    )

st.divider()


# =========================================================
# 4. Week 7 manual pricing (preserved)
# =========================================================

st.header("2. Manual Chooser Pricing")
st.caption(
    "Original Week 7 calculator. Enter your own two volatility assumptions."
)

with st.sidebar:
    st.header("Manual Pricing Parameters")
    s0_manual = st.number_input(
        "Underlying Price (S0)", min_value=0.01, value=231.45, step=1.0
    )
    strike_manual = st.number_input(
        "Strike Price (K)", min_value=0.01, value=150.0, step=1.0
    )
    rate_manual = st.number_input(
        "Risk-free Rate (%)", min_value=-10.0, max_value=50.0,
        value=4.58, step=0.1
    ) / 100.0
    sigma1_manual = st.number_input(
        "First-period Volatility (%)", min_value=0.1, max_value=300.0,
        value=25.0, step=0.5
    ) / 100.0
    sigma2_manual = st.number_input(
        "Second-period Volatility (%)", min_value=0.1, max_value=300.0,
        value=28.0, step=0.5
    ) / 100.0
    t1_manual = st.number_input(
        "Choice Time T1 (years)", min_value=0.01, max_value=5.0,
        value=0.5, step=0.1
    )
    t2_manual = st.number_input(
        "Expiration Time T2 (years)", min_value=0.02, max_value=10.0,
        value=1.0, step=0.1
    )
    simulations_manual = st.select_slider(
        "Manual Monte Carlo Simulations",
        options=[1000, 5000, 10000, 20000, 50000],
        value=10000,
    )
    scenario_manual = st.selectbox(
        "Manual Scenario",
        [
            "Baseline",
            "Volatility Shock (+50%)",
            "Interest Rate Shock (+2 percentage points)",
        ],
    )

if t2_manual <= t1_manual:
    st.error("Expiration time T2 must be greater than choice time T1.")
else:
    manual_sigma1 = sigma1_manual
    manual_sigma2 = sigma2_manual
    manual_rate = rate_manual
    if scenario_manual == "Volatility Shock (+50%)":
        manual_sigma1 *= 1.5
        manual_sigma2 *= 1.5
    elif scenario_manual == "Interest Rate Shock (+2 percentage points)":
        manual_rate += 0.02

    if st.button("Calculate Manual Chooser Price"):
        manual_result = two_stage_chooser_mc(
            S0=s0_manual,
            risk_free_rate=manual_rate,
            sigma_1=manual_sigma1,
            sigma_2=manual_sigma2,
            strike=strike_manual,
            T1=t1_manual,
            T2=t2_manual,
            n_simulations=simulations_manual,
            seed=42,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Chooser Price", f"${manual_result['Chooser_Price']:.4f}")
        c2.metric("MC Standard Error", f"${manual_result['MC_Standard_Error']:.4f}")
        c3.metric(
            "95% Confidence Interval",
            f"${manual_result['MC_95CI_Lower']:.4f} – "
            f"${manual_result['MC_95CI_Upper']:.4f}",
        )
        c1, c2 = st.columns(2)
        c1.metric("Call Choice Ratio", f"{manual_result['Call_Ratio']:.2%}")
        c2.metric("Put Choice Ratio", f"{manual_result['Put_Ratio']:.2%}")
        st.caption("Theoretical Monte Carlo estimate; not a market-validated price.")

st.divider()


# =========================================================
# 5. Week 8 dual pricing
# =========================================================

st.header("3. Week 8 Dual Model Pricing")
st.markdown(
    """
    The Historical 60D strategy uses the same historical volatility estimate
    in both periods. The Random Forest strategy uses the 126D forecast in the
    first period and the 252D forecast in the second period.
    """
)

try:
    week8_data = load_week8_market_and_forecasts()
    date_options = week8_data["Date"].dt.strftime("%Y-%m-%d").tolist()

    default_date = date_options[-1]
    selected_date = st.selectbox(
        "Select valuation date",
        options=date_options,
        index=len(date_options) - 1,
    )
    selected_row = week8_data.loc[
        week8_data["Date"].dt.strftime("%Y-%m-%d") == selected_date
    ].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Underlying Close", f"${selected_row['S0']:.2f}")
    col2.metric("Risk-free Rate", f"{selected_row['Risk_Free_Rate']:.2%}")
    col3.metric("Matched Valuation Dates", f"{len(week8_data):,}")

    with st.expander("Selected date: volatility inputs", expanded=True):
        st.dataframe(
            pd.DataFrame({
                "Input": ["First period (126D)", "Second period (252D)"],
                "Historical 60D": [
                    selected_row["Historical_126D"],
                    selected_row["Historical_252D"],
                ],
                "Random Forest": [
                    selected_row["RF_126D"],
                    selected_row["RF_252D"],
                ],
            }).assign(
                **{
                    "Historical 60D": lambda x: x["Historical 60D"].map(
                        lambda v: f"{v:.2%}"
                    ),
                    "Random Forest": lambda x: x["Random Forest"].map(
                        lambda v: f"{v:.2%}"
                    ),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        dual_strike = st.number_input(
            "Dual Pricing Strike", min_value=0.01, value=150.0, step=1.0,
            key="dual_strike",
        )
    with c2:
        dual_t1 = st.number_input(
            "Dual Pricing T1 (years)", min_value=0.01, max_value=5.0,
            value=0.5, step=0.1, key="dual_t1",
        )
    with c3:
        dual_t2 = st.number_input(
            "Dual Pricing T2 (years)", min_value=0.02, max_value=10.0,
            value=1.0, step=0.1, key="dual_t2",
        )

    dual_sims = st.select_slider(
        "Dual Pricing Monte Carlo Simulations",
        options=[1000, 5000, 10000, 20000, 50000],
        value=10000,
        key="dual_sims",
    )

    if dual_t2 <= dual_t1:
        st.error("Dual Pricing T2 must be greater than T1.")
    elif st.button("Calculate Historical vs ML Prices", type="primary"):
        h_s1, h_s2, h_result = run_week8_price(
            selected_row, "Historical 60D", dual_strike,
            dual_t1, dual_t2, dual_sims,
        )
        rf_s1, rf_s2, rf_result = run_week8_price(
            selected_row, "Random Forest", dual_strike,
            dual_t1, dual_t2, dual_sims,
        )

        st.subheader(f"Pricing results for {selected_date}")
        a, b = st.columns(2)
        with a:
            st.markdown("**Historical 60D benchmark**")
            st.metric("Chooser Price", f"${h_result['Chooser_Price']:.4f}")
            st.write(
                f"95% CI: ${h_result['MC_95CI_Lower']:.4f} – "
                f"${h_result['MC_95CI_Upper']:.4f}"
            )
            st.write(f"MC standard error: ${h_result['MC_Standard_Error']:.4f}")
            st.write(f"126D / 252D volatility: {h_s1:.2%} / {h_s2:.2%}")
            st.write(
                f"Call / Put choice: {h_result['Call_Ratio']:.2%} / "
                f"{h_result['Put_Ratio']:.2%}"
            )
        with b:
            st.markdown("**Random Forest alternative**")
            st.metric("Chooser Price", f"${rf_result['Chooser_Price']:.4f}")
            st.write(
                f"95% CI: ${rf_result['MC_95CI_Lower']:.4f} – "
                f"${rf_result['MC_95CI_Upper']:.4f}"
            )
            st.write(f"MC standard error: ${rf_result['MC_Standard_Error']:.4f}")
            st.write(f"126D / 252D volatility: {rf_s1:.2%} / {rf_s2:.2%}")
            st.write(
                f"Call / Put choice: {rf_result['Call_Ratio']:.2%} / "
                f"{rf_result['Put_Ratio']:.2%}"
            )

        difference = rf_result["Chooser_Price"] - h_result["Chooser_Price"]
        pct = difference / h_result["Chooser_Price"] * 100 if h_result["Chooser_Price"] else np.nan
        d1, d2 = st.columns(2)
        d1.metric("RF minus Historical Price", f"${difference:.4f}")
        d2.metric("Relative Price Difference", f"{pct:.4f}%")

        comparison = pd.DataFrame({
            "Strategy": ["Historical 60D", "Random Forest"],
            "Price": [h_result["Chooser_Price"], rf_result["Chooser_Price"]],
            "MCSE": [h_result["MC_Standard_Error"], rf_result["MC_Standard_Error"]],
            "CI Lower": [h_result["MC_95CI_Lower"], rf_result["MC_95CI_Lower"]],
            "CI Upper": [h_result["MC_95CI_Upper"], rf_result["MC_95CI_Upper"]],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(comparison["Strategy"], comparison["Price"])
        ax.set_ylabel("Theoretical option price ($)")
        ax.set_title("Historical vs Random Forest Chooser Price")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Historical Price Trend")
    st.caption(
        "This chart shows the saved 10,000-simulation backtest prices across "
        "the 242 common valuation dates."
    )
    pricing_detail = read_csv_if_exists(PRICING_DETAIL_FILE)
    if pricing_detail is not None:
        pricing_detail["Date"] = pd.to_datetime(pricing_detail["Date"])
        pivot = pricing_detail.pivot(
            index="Date", columns="Strategy", values="Chooser_Price"
        ).sort_index()
        st.line_chart(pivot)
    else:
        st.info(
            "Run `python model/two_stage_volatility_chooser_week8.py` "
            "to generate the historical pricing trend data."
        )

    st.subheader("Volatility Sensitivity")
    st.caption(
        "Sensitivity is calculated for the selected valuation date by "
        "multiplying both period volatility inputs by the same shock factor."
    )
    shock_factors = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    sensitivity_rows = []
    for strategy in ("Historical 60D", "Random Forest"):
        base_s1 = (
            float(selected_row["Historical_126D"])
            if strategy == "Historical 60D" else float(selected_row["RF_126D"])
        )
        base_s2 = (
            float(selected_row["Historical_252D"])
            if strategy == "Historical 60D" else float(selected_row["RF_252D"])
        )
        for factor in shock_factors:
            _, _, result = run_week8_price(
                selected_row, strategy, dual_strike,
                dual_t1, dual_t2, min(dual_sims, 10000),
            )
            # Reprice with the requested sensitivity multiplier.
            result = price_mc(
                S0=float(selected_row["S0"]),
                r=float(selected_row["Risk_Free_Rate"]),
                sigma1=base_s1 * factor,
                sigma2=base_s2 * factor,
                K=float(dual_strike),
                T1=float(dual_t1),
                T2=float(dual_t2),
                n=min(int(dual_sims), 10000),
                seed=42,
            )
            sensitivity_rows.append({
                "Volatility multiplier": factor,
                "Strategy": strategy,
                "Chooser Price": result["Chooser_Price"],
            })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_chart = sensitivity.pivot(
        index="Volatility multiplier",
        columns="Strategy",
        values="Chooser Price",
    )
    st.line_chart(sensitivity_chart)
    st.dataframe(
        sensitivity,
        use_container_width=True,
        hide_index=True,
    )

except Exception as error:
    st.error(f"Week 8 dual-pricing section could not load: {error}")
    st.info(
        "Check that the Week 8 feature file and both horizon test-prediction "
        "CSV files exist in the expected project folders."
    )

st.divider()


# =========================================================
# 6. Model performance dashboard
# =========================================================

st.header("4. Volatility Model Performance")
st.markdown(
    """
    Test-set MAE, RMSE and R² are used to compare the Random Forest model
    with the historical 60D volatility benchmark. Lower MAE/RMSE are better;
    higher R² is better. Negative test R² indicates performance worse than
    predicting the test-set mean under the usual R² definition.
    """
)

metrics = load_metrics()
if metrics is not None:
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.caption(
        "The table is read directly from the Week 8 horizon-specific metrics.csv "
        "files. Check the Model column to distinguish RF, historical and "
        "training-mean baselines."
    )
else:
    st.info(
        "No Week 8 metrics.csv files found. Run the Week 8 model-training "
        "script to create the performance metrics."
    )

st.caption(
    "All displayed option prices are theoretical model estimates. "
    "They are not validated against observed market option quotes."
)
