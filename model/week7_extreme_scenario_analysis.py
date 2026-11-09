
"""
Week 7: Extreme Scenario Analysis for Two-Stage Chooser Options.

Run from project root:
    python model/week7_extreme_scenario_analysis.py

Scenarios:
1. Baseline
2. Volatility shock: both stage volatilities increase by 50%
3. Interest rate shock: risk-free rate increases by 2 percentage points

The original two-stage model is imported but not modified.
Common random numbers are used across scenarios.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from two_stage_volatility_chooser import (
    load_config,
    prepare_forecasts,
    two_stage_chooser_mc,
    CONFIG_FILE,
    FEATURE_FILE,
    FORECAST_FILE,
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "volatility_experiment"
    / "week7_scenario_analysis"
)

SEED = 42
VOLATILITY_MULTIPLIER = 1.50
RATE_SHOCK = 0.02


def run_scenarios():
    for path in (CONFIG_FILE, FEATURE_FILE, FORECAST_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    cfg = load_config()
    forecasts = prepare_forecasts()

    market = pd.read_csv(FEATURE_FILE)
    market["Date"] = pd.to_datetime(market["Date"])

    market = market[
        ["Date", "Adj Close", "DGS10"]
    ].rename(
        columns={
            "Adj Close": "S0",
            "DGS10": "Risk_Free_Rate_Percent",
        }
    )

    market["Risk_Free_Rate"] = (
        market["Risk_Free_Rate_Percent"] / 100.0
    )

    data = forecasts.merge(
        market,
        on="Date",
        how="inner",
        validate="many_to_one",
    )

    if data.empty:
        raise RuntimeError("No matched forecast and market dates.")

    results = []

    scenario_definitions = [
        "Baseline",
        "Volatility_Shock_50pct",
        "Interest_Rate_Shock_2pp",
    ]

    for row in data.itertuples(index=False):

        base_sigma_1 = float(row.Sigma_1)
        base_sigma_2 = float(row.Sigma_2)
        base_rate = float(row.Risk_Free_Rate)

        scenarios = {
            "Baseline": {
                "sigma_1": base_sigma_1,
                "sigma_2": base_sigma_2,
                "rate": base_rate,
            },
            "Volatility_Shock_50pct": {
                "sigma_1": base_sigma_1 * VOLATILITY_MULTIPLIER,
                "sigma_2": base_sigma_2 * VOLATILITY_MULTIPLIER,
                "rate": base_rate,
            },
            "Interest_Rate_Shock_2pp": {
                "sigma_1": base_sigma_1,
                "sigma_2": base_sigma_2,
                "rate": base_rate + RATE_SHOCK,
            },
        }

        for scenario_name in scenario_definitions:
            params = scenarios[scenario_name]

            mc = two_stage_chooser_mc(
                S0=float(row.S0),
                risk_free_rate=params["rate"],
                sigma_1=params["sigma_1"],
                sigma_2=params["sigma_2"],
                strike=cfg["strike"],
                T1=cfg["T1"],
                T2=cfg["T2"],
                n_simulations=cfg["n_simulations"],
                seed=SEED,
            )

            results.append({
                "Date": row.Date.date().isoformat(),
                "Strategy": row.Strategy,
                "Scenario": scenario_name,
                "Underlying_S0": float(row.S0),
                "Strike": cfg["strike"],
                "Base_Risk_Free_Rate": base_rate,
                "Scenario_Risk_Free_Rate": params["rate"],
                "Base_Sigma_126D": base_sigma_1,
                "Base_Sigma_252D": base_sigma_2,
                "Scenario_Sigma_126D": params["sigma_1"],
                "Scenario_Sigma_252D": params["sigma_2"],
                "N_Simulations": cfg["n_simulations"],
                **mc,
            })

    detail = pd.DataFrame(results)

    # Compare each shocked price with its matching baseline:
    # same valuation date and same forecasting strategy.
    baseline = detail[
        detail["Scenario"] == "Baseline"
    ][
        ["Date", "Strategy", "Chooser_Price"]
    ].rename(
        columns={"Chooser_Price": "Baseline_Price"}
    )

    detail = detail.merge(
        baseline,
        on=["Date", "Strategy"],
        how="left",
        validate="many_to_one",
    )

    detail["Price_Change"] = (
        detail["Chooser_Price"] - detail["Baseline_Price"]
    )

    detail["Price_Change_Percent"] = (
        detail["Price_Change"]
        / detail["Baseline_Price"] * 100.0
    )

    # Summarize results by forecasting strategy and scenario.
    summary = (
        detail.groupby(["Strategy", "Scenario"])
        .agg(
            Observations=("Chooser_Price", "size"),
            Mean_Base_Price=("Baseline_Price", "mean"),
            Mean_Scenario_Price=("Chooser_Price", "mean"),
            Mean_Price_Change=("Price_Change", "mean"),
            Mean_Price_Change_Percent=(
                "Price_Change_Percent", "mean"
            ),
            Mean_Sigma_126D=("Scenario_Sigma_126D", "mean"),
            Mean_Sigma_252D=("Scenario_Sigma_252D", "mean"),
            Mean_Risk_Free_Rate=(
                "Scenario_Risk_Free_Rate", "mean"
            ),
            Mean_MC_Standard_Error=(
                "MC_Standard_Error", "mean"
            ),
        )
        .reset_index()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_path = OUTPUT_DIR / "week7_scenario_detail.csv"
    summary_path = OUTPUT_DIR / "week7_scenario_summary.csv"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("=" * 72)
    print("WEEK 7: EXTREME SCENARIO ANALYSIS")
    print("=" * 72)

    print(f"Matched valuation dates: {detail['Date'].nunique()}")
    print(f"Forecasting strategies: {detail['Strategy'].nunique()}")
    print(f"Monte Carlo simulations: {cfg['n_simulations']}")

    print("\nScenario definitions:")
    print("Baseline: original parameters")
    print("Volatility shock: both volatility forecasts multiplied by 1.50")
    print("Interest rate shock: risk-free rate increased by 0.02")

    print("\nScenario comparison:")
    print(summary.round(6).to_string(index=False))

    print("\nSaved files:")
    print(detail_path)
    print(summary_path)

    print(
        "\nNote: Results are theoretical model prices, "
        "not market-price accuracy measurements."
    )


if __name__ == "__main__":
    run_scenarios()
