"""
Integrate out-of-sample future-volatility forecasts into the existing
two-stage Chooser Option Monte Carlo pricing model.

Run from the project root:
    python model/volatility_chooser_integration.py

Inputs:
    config/week3_parameters.json
    data/processed/jpm_features.csv
    data/processed/volatility_experiment/window_comparison/
        window_comparison_predictions.csv

Outputs:
    data/processed/volatility_experiment/chooser_integration/
        chooser_volatility_pricing.csv
        chooser_volatility_pricing_summary.csv

Important interpretation:
- Each volatility forecast is annualized and is used as a constant sigma
  over the full original chooser life (T1 to T2), because the existing
  bsm_chooser.py accepts one constant sigma.
- The 20D/126D/252D labels describe the forecast target horizon, not a
  piecewise volatility term structure inside the option-pricing model.
- The predictions are out-of-sample test forecasts from the existing
  window-comparison experiment. They are not forecasts newly generated
  at every historical date using a retrained model.
- No actual exchange-traded Chooser Option prices are supplied, so this
  compares theoretical model prices, not market-price pricing accuracy.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
sys.path.insert(0, str(MODEL_DIR))

from bsm_chooser import simulate_chooser_option  # noqa: E402


CONFIG_FILE = ROOT / "config" / "week3_parameters.json"
FEATURE_FILE = ROOT / "data" / "processed" / "jpm_features.csv"
FORECAST_FILE = (
    ROOT
    / "data"
    / "processed"
    / "volatility_experiment"
    / "window_comparison"
    / "window_comparison_predictions.csv"
)
OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "volatility_experiment"
    / "chooser_integration"
)

SCENARIO_MAP = {
    "Historical_60D_Baseline": "Historical_60D",
    "Expanding_Window": "Expanding_Window",
    "Rolling_Window_252D": "Rolling_Window_252D",
}

HORIZON_LABELS = {20: "20D", 126: "126D", 252: "252D"}


def load_config():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing configuration: {CONFIG_FILE}")

    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    if "updated_jpm" not in config:
        raise KeyError("week3_parameters.json must contain 'updated_jpm'.")
    return config["updated_jpm"]


def load_inputs():
    for path in [FEATURE_FILE, FORECAST_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input: {path}\n"
                "Run the volatility window comparison script first."
            )

    features = pd.read_csv(FEATURE_FILE)
    forecasts = pd.read_csv(FORECAST_FILE)

    features["Date"] = pd.to_datetime(features["Date"])
    forecasts["Date"] = pd.to_datetime(forecasts["Date"])

    required_features = {"Date", "Adj Close", "DGS10"}
    missing = required_features.difference(features.columns)
    if missing:
        raise ValueError(f"Feature file missing columns: {sorted(missing)}")

    required_forecasts = {
        "Date", "Prediction", "Horizon_Days", "Strategy"
    }
    missing = required_forecasts.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Forecast file missing columns: {sorted(missing)}")

    return features, forecasts


def price_one_scenario(
    *,
    s0,
    rate,
    sigma,
    strike,
    t1,
    t2,
    n_simulations,
    seed,
):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Invalid annualized volatility: {sigma}")

    result = simulate_chooser_option(
        S0=float(s0),
        risk_free_rate=float(rate),
        sigma=float(sigma),
        strike=float(strike),
        T1=float(t1),
        T2=float(t2),
        n_simulations=int(n_simulations),
        seed=int(seed),
    )

    discounted = np.asarray(result["Discounted_Payoff"], dtype=float)
    chooser_price = float(np.mean(discounted))
    mc_standard_error = float(
        np.std(discounted, ddof=1) / np.sqrt(len(discounted))
    )

    choices = np.asarray(result["Choice"])
    call_ratio = float(np.mean(choices == "CALL"))
    put_ratio = float(np.mean(choices == "PUT"))

    return {
        "Chooser_Price": chooser_price,
        "MC_Standard_Error": mc_standard_error,
        "MC_95CI_Lower": chooser_price - 1.96 * mc_standard_error,
        "MC_95CI_Upper": chooser_price + 1.96 * mc_standard_error,
        "Call_Ratio": call_ratio,
        "Put_Ratio": put_ratio,
    }


def main():
    params = load_config()
    features, forecasts = load_inputs()

    # Use the same contract and market-input conventions as Week 3/4.
    strike = float(params["strike"])
    t1 = float(params["T1"])
    t2 = float(params["T2"])
    n_simulations = int(params.get("n_simulations", 10000))
    seed = int(params.get("seed", 42))

    # The feature file supplies the contemporaneous underlying and rate.
    market = features[["Date", "Adj Close", "DGS10"]].copy()
    market = market.rename(
        columns={
            "Adj Close": "S0",
            "DGS10": "Risk_Free_Rate_Percent",
        }
    )
    market["risk_free_rate"] = market["Risk_Free_Rate_Percent"] / 100.0

    # One row per date/strategy/horizon; join to contemporaneous market data.
    data = forecasts.merge(market, on="Date", how="inner", validate="many_to_one")

    if data.empty:
        raise RuntimeError("No overlapping dates between forecasts and market data.")

    data["Volatility_Scenario"] = data["Strategy"].map(SCENARIO_MAP)
    data = data[data["Volatility_Scenario"].notna()].copy()

    # Historical baseline rows exist for each horizon in the forecast file.
    # Retain one copy per date/horizon to avoid triple-counting the same baseline.
    baseline_rows = data[
        data["Strategy"] == "Historical_60D_Baseline"
    ].copy()
    baseline_rows = baseline_rows.drop_duplicates(
        subset=["Date", "Horizon_Days"]
    )

    model_rows = data[
        data["Strategy"].isin(["Expanding_Window", "Rolling_Window_252D"])
    ].copy()

    pricing_rows = pd.concat(
        [baseline_rows, model_rows],
        ignore_index=True,
    )

    pricing_rows["Annualized_Volatility_Input"] = (
        pricing_rows["Prediction"].astype(float)
    )

    output_rows = []
    for row in pricing_rows.itertuples(index=False):
        horizon = int(row.Horizon_Days)
        scenario = str(row.Volatility_Scenario)

        metrics = price_one_scenario(
            s0=float(row.S0),
            rate=float(row.risk_free_rate),
            sigma=float(row.Annualized_Volatility_Input),
            strike=strike,
            t1=t1,
            t2=t2,
            n_simulations=n_simulations,
            seed=seed,
        )

        output_rows.append(
            {
                "Date": row.Date.date().isoformat(),
                "Target_Horizon_Days": horizon,
                "Target_Date": getattr(row, "Target_Date", pd.NaT),
                "Scenario": scenario,
                "Underlying_S0": float(row.S0),
                "Risk_Free_Rate": float(row.risk_free_rate),
                "Annualized_Volatility_Input": float(
                    row.Annualized_Volatility_Input
                ),
                "Strike": strike,
                "T1_Years": t1,
                "T2_Years": t2,
                "N_Simulations": n_simulations,
                "Seed": seed,
                **metrics,
            }
        )

    result_df = pd.DataFrame(output_rows)
    result_df["Target_Date"] = pd.to_datetime(
        result_df["Target_Date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Summarize pricing differences within each target horizon.
    summary_rows = []
    for horizon, group in result_df.groupby("Target_Horizon_Days"):
        pivot = group.pivot_table(
            index="Date",
            columns="Scenario",
            values="Chooser_Price",
            aggfunc="first",
        )

        for scenario in [
            "Historical_60D",
            "Expanding_Window",
            "Rolling_Window_252D",
        ]:
            subset = group[group["Scenario"] == scenario]
            if subset.empty:
                continue

            baseline_prices = pivot.get("Historical_60D")
            if baseline_prices is not None:
                scenario_prices = pivot.get(scenario)
                paired = pd.concat(
                    [baseline_prices.rename("Baseline"),
                     scenario_prices.rename("Scenario")],
                    axis=1,
                ).dropna()
                price_diff = paired["Scenario"] - paired["Baseline"]
            else:
                price_diff = pd.Series(dtype=float)

            summary_rows.append(
                {
                    "Target_Horizon_Days": int(horizon),
                    "Scenario": scenario,
                    "Observations": int(len(subset)),
                    "Mean_Annualized_Volatility": float(
                        subset["Annualized_Volatility_Input"].mean()
                    ),
                    "Mean_Chooser_Price": float(
                        subset["Chooser_Price"].mean()
                    ),
                    "Std_Chooser_Price": float(
                        subset["Chooser_Price"].std(ddof=1)
                    ),
                    "Mean_MC_Standard_Error": float(
                        subset["MC_Standard_Error"].mean()
                    ),
                    "Mean_Call_Ratio": float(subset["Call_Ratio"].mean()),
                    "Mean_Put_Ratio": float(subset["Put_Ratio"].mean()),
                    "Mean_Price_Difference_vs_Historical_60D": (
                        float(price_diff.mean()) if len(price_diff) else np.nan
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detailed_path = OUTPUT_DIR / "chooser_volatility_pricing.csv"
    summary_path = OUTPUT_DIR / "chooser_volatility_pricing_summary.csv"

    result_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("=" * 72)
    print("CHOOSER OPTION PRICING WITH OUT-OF-SAMPLE VOLATILITY FORECASTS")
    print("=" * 72)
    print(f"Underlying source: JPM Adjusted Close")
    print(f"Risk-free rate source: DGS10 / 100")
    print(f"Contract: K={strike:.2f}, T1={t1:.2f}, T2={t2:.2f}")
    print(f"Monte Carlo simulations per price: {n_simulations}")
    print(f"Common random seed: {seed}")
    print(
        "Forecast date range: "
        f"{result_df['Date'].min()} to {result_df['Date'].max()}"
    )
    print("\nSummary:")
    display_cols = [
        "Target_Horizon_Days",
        "Scenario",
        "Observations",
        "Mean_Annualized_Volatility",
        "Mean_Chooser_Price",
        "Mean_MC_Standard_Error",
        "Mean_Call_Ratio",
        "Mean_Price_Difference_vs_Historical_60D",
    ]
    print(summary_df[display_cols].round(6).to_string(index=False))

    print("\nInterpretation:")
    print(
        "- Each forecast sigma is held constant over the entire chooser life."
    )
    print(
        "- Horizon labels describe the volatility forecast target horizon, "
        "not separate volatility segments in the option contract."
    )
    print(
        "- Prices are theoretical model outputs; no actual market chooser "
        "prices were provided for pricing-error evaluation."
    )
    print(f"\nDetailed output: {detailed_path}")
    print(f"Summary output:  {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
