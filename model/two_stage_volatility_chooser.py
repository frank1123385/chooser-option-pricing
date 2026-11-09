"""
Two-stage volatility integration for the existing Chooser Option model.

Run from the project root:
    python model/two_stage_volatility_chooser.py

For each matched forecast date:
- First chooser period (0 to T1): use the 126-trading-day volatility forecast.
- Second chooser period (T1 to T2): use the 252-trading-day volatility forecast.
- Historical benchmark: use the same Historical 60D volatility estimate in both periods.
- Compare Expanding and Rolling ML forecasts as separate strategies.

The original model/bsm_chooser.py is not modified. The two-stage simulation
uses the same GBM drift, chooser decision rule, terminal payoff and discounting
as bsm_chooser.py, but allows sigma to differ between the two time intervals.

Only dates with both 126D and 252D out-of-sample forecasts are included.
This avoids using a forecast from a different valuation date.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "week3_parameters.json"
FORECAST_FILE = (
    ROOT / "data" / "processed" / "volatility_experiment"
    / "window_comparison" / "window_comparison_predictions.csv"
)
FEATURE_FILE = ROOT / "data" / "processed" / "jpm_features.csv"
OUTPUT_DIR = (
    ROOT / "data" / "processed" / "volatility_experiment"
    / "two_stage_chooser"
)

N_SIMULATIONS_OVERRIDE = None  # None uses config value; set e.g. 50000 for convergence check.
SEED = 42


def load_config():
    cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))["updated_jpm"]
    return {
        "strike": float(cfg["strike"]),
        "T1": float(cfg["T1"]),
        "T2": float(cfg["T2"]),
        "n_simulations": int(
            N_SIMULATIONS_OVERRIDE or cfg.get("n_simulations", 10000)
        ),
    }


def two_stage_chooser_mc(
    S0, risk_free_rate, sigma_1, sigma_2, strike, T1, T2,
    n_simulations=10000, seed=42
):
    """Two-stage GBM Chooser MC with distinct annualized volatilities."""
    if S0 <= 0 or strike <= 0:
        raise ValueError("S0 and strike must be positive.")
    if T1 <= 0 or T2 <= T1:
        raise ValueError("Require 0 < T1 < T2.")
    if sigma_1 <= 0 or sigma_2 <= 0:
        raise ValueError("Both volatility inputs must be positive.")

    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_simulations)
    z2 = rng.standard_normal(n_simulations)

    dt1 = T1
    dt2 = T2 - T1

    s1 = S0 * np.exp(
        (risk_free_rate - 0.5 * sigma_1**2) * dt1
        + sigma_1 * np.sqrt(dt1) * z1
    )

    choice = np.where(s1 > strike, "CALL", "PUT")

    s2 = s1 * np.exp(
        (risk_free_rate - 0.5 * sigma_2**2) * dt2
        + sigma_2 * np.sqrt(dt2) * z2
    )

    call_payoff = np.maximum(s2 - strike, 0.0)
    put_payoff = np.maximum(strike - s2, 0.0)
    payoff = np.where(choice == "CALL", call_payoff, put_payoff)
    discounted = payoff * np.exp(-risk_free_rate * T2)

    price = float(discounted.mean())
    standard_error = float(discounted.std(ddof=1) / np.sqrt(n_simulations))

    return {
        "Chooser_Price": price,
        "MC_Standard_Error": standard_error,
        "MC_95CI_Lower": price - 1.96 * standard_error,
        "MC_95CI_Upper": price + 1.96 * standard_error,
        "Call_Ratio": float(np.mean(choice == "CALL")),
        "Put_Ratio": float(np.mean(choice == "PUT")),
    }


def prepare_forecasts():
    forecasts = pd.read_csv(FORECAST_FILE)
    forecasts["Date"] = pd.to_datetime(forecasts["Date"])

    # Historical 60D is duplicated once for each horizon. Keep a copy for
    # each horizon, then align the 126D and 252D rows on valuation Date.
    hist = forecasts[
        forecasts["Strategy"] == "Historical_60D_Baseline"
    ].copy()

    expanding = forecasts[
        forecasts["Strategy"] == "Expanding_Window"
    ].copy()
    rolling = forecasts[
        forecasts["Strategy"] == "Rolling_Window_252D"
    ].copy()

    def wide_for(strategy_df, label):
        first = strategy_df[strategy_df["Horizon_Days"] == 126][
            ["Date", "Prediction"]
        ].rename(columns={"Prediction": "Sigma_126D"})
        second = strategy_df[strategy_df["Horizon_Days"] == 252][
            ["Date", "Prediction"]
        ].rename(columns={"Prediction": "Sigma_252D"})
        merged = first.merge(second, on="Date", how="inner", validate="one_to_one")
        merged["Strategy"] = label
        return merged

    hist_wide = wide_for(hist, "Historical_60D")
    expanding_wide = wide_for(expanding, "Expanding_Window")
    rolling_wide = wide_for(rolling, "Rolling_Window_252D")

    # For the baseline, the 126D and 252D copies should represent the same
    # backward-looking 60D volatility on a given date. Use their average only
    # after verifying the two copies agree closely.
    hist_wide["Sigma_Difference_Check"] = (
        hist_wide["Sigma_126D"] - hist_wide["Sigma_252D"]
    ).abs()
    if hist_wide["Sigma_Difference_Check"].max() > 1e-8:
        raise ValueError(
            "Historical baseline differs between 126D and 252D rows; "
            "inspect the baseline alignment before pricing."
        )
    hist_wide["Sigma_1"] = hist_wide["Sigma_126D"]
    hist_wide["Sigma_2"] = hist_wide["Sigma_252D"]

    expanding_wide["Sigma_1"] = expanding_wide["Sigma_126D"]
    expanding_wide["Sigma_2"] = expanding_wide["Sigma_252D"]

    rolling_wide["Sigma_1"] = rolling_wide["Sigma_126D"]
    rolling_wide["Sigma_2"] = rolling_wide["Sigma_252D"]

    all_forecasts = pd.concat(
        [hist_wide, expanding_wide, rolling_wide],
        ignore_index=True,
    )
    return all_forecasts


def main():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(CONFIG_FILE)
    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"{FORECAST_FILE} not found. Run the window comparison first."
        )
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(FEATURE_FILE)

    cfg = load_config()
    forecasts = prepare_forecasts()

    market = pd.read_csv(FEATURE_FILE)
    market["Date"] = pd.to_datetime(market["Date"])
    needed = {"Date", "Adj Close", "DGS10"}
    missing = needed.difference(market.columns)
    if missing:
        raise ValueError(f"Market feature file missing: {sorted(missing)}")

    market = market[["Date", "Adj Close", "DGS10"]].rename(
        columns={"Adj Close": "S0", "DGS10": "Risk_Free_Rate_Percent"}
    )
    market["Risk_Free_Rate"] = market["Risk_Free_Rate_Percent"] / 100.0

    data = forecasts.merge(
        market, on="Date", how="inner", validate="many_to_one"
    )
    if data.empty:
        raise RuntimeError("No overlapping dates found.")

    results = []
    for row in data.itertuples(index=False):
        mc = two_stage_chooser_mc(
            S0=float(row.S0),
            risk_free_rate=float(row.Risk_Free_Rate),
            sigma_1=float(row.Sigma_1),
            sigma_2=float(row.Sigma_2),
            strike=cfg["strike"],
            T1=cfg["T1"],
            T2=cfg["T2"],
            n_simulations=cfg["n_simulations"],
            seed=SEED,
        )
        results.append(
            {
                "Date": row.Date.date().isoformat(),
                "Strategy": row.Strategy,
                "Sigma_First_Period_126D": float(row.Sigma_1),
                "Sigma_Second_Period_252D": float(row.Sigma_2),
                "Underlying_S0": float(row.S0),
                "Risk_Free_Rate": float(row.Risk_Free_Rate),
                "Strike": cfg["strike"],
                "T1_Years": cfg["T1"],
                "T2_Years": cfg["T2"],
                "N_Simulations": cfg["n_simulations"],
                **mc,
            }
        )

    detail = pd.DataFrame(results)

    # Paired price differences by valuation date against Historical 60D.
    baseline = detail[detail["Strategy"] == "Historical_60D"][
        ["Date", "Chooser_Price"]
    ].rename(columns={"Chooser_Price": "Historical_Baseline_Price"})

    detail = detail.merge(baseline, on="Date", how="left", validate="many_to_one")
    detail["Price_Difference_vs_Historical"] = (
        detail["Chooser_Price"] - detail["Historical_Baseline_Price"]
    )
    detail["Price_Difference_Percent_vs_Historical"] = (
        detail["Price_Difference_vs_Historical"]
        / detail["Historical_Baseline_Price"] * 100.0
    )

    summary = (
        detail.groupby("Strategy")
        .agg(
            Observations=("Chooser_Price", "size"),
            Mean_Sigma_First_Period=("Sigma_First_Period_126D", "mean"),
            Mean_Sigma_Second_Period=("Sigma_Second_Period_252D", "mean"),
            Mean_Chooser_Price=("Chooser_Price", "mean"),
            Std_Chooser_Price=("Chooser_Price", "std"),
            Mean_MC_Standard_Error=("MC_Standard_Error", "mean"),
            Mean_Call_Ratio=("Call_Ratio", "mean"),
            Mean_Put_Ratio=("Put_Ratio", "mean"),
            Mean_Price_Difference_vs_Historical=(
                "Price_Difference_vs_Historical", "mean"
            ),
            Mean_Price_Difference_Percent_vs_Historical=(
                "Price_Difference_Percent_vs_Historical", "mean"
            ),
        )
        .reset_index()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTPUT_DIR / "two_stage_chooser_pricing.csv"
    summary_path = OUTPUT_DIR / "two_stage_chooser_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("=" * 72)
    print("TWO-STAGE VOLATILITY CHOOSER OPTION PRICING")
    print("=" * 72)
    print(f"First period: 0 to {cfg['T1']} years, using 126D forecast")
    print(f"Second period: {cfg['T1']} to {cfg['T2']} years, using 252D forecast")
    print(f"Strike: {cfg['strike']:.2f}")
    print(f"Monte Carlo simulations: {cfg['n_simulations']}")
    print(f"Matched valuation dates: {detail['Date'].nunique()}")
    print("\nSummary:")
    print(summary.round(6).to_string(index=False))
    print("\nNote: These are theoretical prices, not market-price accuracy tests.")
    print("The same random seed is used across scenarios for common random numbers.")
    print(f"\nDetailed output: {detail_path}")
    print(f"Summary output: {summary_path}")


if __name__ == "__main__":
    main()
