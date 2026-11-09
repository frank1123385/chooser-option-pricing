"""Week 8 dual two-stage Chooser pricing. Run: python model/two_stage_volatility_chooser_week8.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "week3_parameters.json"
FEATURES = ROOT / "data" / "processed" / "jpm_features_week8.csv"
P126 = ROOT / "data" / "processed" / "volatility_experiment_week8" / "model_results" / "126d" / "test_predictions.csv"
P252 = ROOT / "data" / "processed" / "volatility_experiment_week8" / "model_results" / "252d" / "test_predictions.csv"
OUT = ROOT / "data" / "processed" / "volatility_experiment_week8" / "two_stage_chooser"
N_SIMULATIONS_OVERRIDE = None
SEED = 42


def load_config():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["updated_jpm"]
    return {
        "strike": float(cfg["strike"]),
        "T1": float(cfg["T1"]),
        "T2": float(cfg["T2"]),
        "n": int(N_SIMULATIONS_OVERRIDE or cfg.get("n_simulations", 10000)),
    }


def price_mc(S0, r, sigma1, sigma2, K, T1, T2, n, seed):
    if S0 <= 0 or K <= 0 or not (0 < T1 < T2) or min(sigma1, sigma2) <= 0:
        raise ValueError("Invalid S0, strike, maturity, or volatility.")
    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    s1 = S0 * np.exp((r - 0.5 * sigma1**2) * T1 + sigma1 * np.sqrt(T1) * z1)
    choose_call = s1 > K
    s2 = s1 * np.exp(
        (r - 0.5 * sigma2**2) * (T2 - T1)
        + sigma2 * np.sqrt(T2 - T1) * z2
    )
    payoff = np.where(choose_call, np.maximum(s2 - K, 0),
                      np.maximum(K - s2, 0))
    discounted = payoff * np.exp(-r * T2)
    p = float(discounted.mean())
    se = float(discounted.std(ddof=1) / np.sqrt(n))
    return {
        "Chooser_Price": p,
        "MC_Standard_Error": se,
        "MC_95CI_Lower": p - 1.96 * se,
        "MC_95CI_Upper": p + 1.96 * se,
        "Call_Ratio": float(choose_call.mean()),
        "Put_Ratio": float((~choose_call).mean()),
    }


def read_forecast(path, horizon):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = ["Date", "Historical_60D_volatility", "Tuned_Random_Forest"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"])
    if df["Date"].duplicated().any():
        raise ValueError(f"Duplicate dates in {path}")
    return df[required].rename(columns={
        "Historical_60D_volatility": f"Hist_{horizon}",
        "Tuned_Random_Forest": f"RF_{horizon}",
    })


def prepare_data():
    f126 = read_forecast(P126, 126)
    f252 = read_forecast(P252, 252)
    f = f126.merge(f252, on="Date", how="inner", validate="one_to_one")
    if f.empty:
        raise RuntimeError("No dates shared by 126D and 252D prediction files.")
    diff = (f["Hist_126"] - f["Hist_252"]).abs()
    if diff.max() > 1e-8:
        raise ValueError("Historical 60D volatility differs across horizon files.")
    market = pd.read_csv(FEATURES)
    required = {"Date", "Close", "DGS10"}
    if not required.issubset(market.columns):
        raise ValueError(f"Feature file needs columns: {sorted(required)}")
    market["Date"] = pd.to_datetime(market["Date"])
    if market["Date"].duplicated().any():
        raise ValueError("Duplicate dates in feature file.")
    market = market[["Date", "Close", "DGS10"]].rename(columns={"Close": "S0"})
    market["r"] = pd.to_numeric(market["DGS10"], errors="coerce") / 100
    f = f.merge(market[["Date", "S0", "r"]], on="Date", how="inner",
                validate="one_to_one")
    cols = ["S0", "r", "Hist_126", "Hist_252", "RF_126", "RF_252"]
    for c in cols:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f.dropna(subset=cols).sort_values("Date").reset_index(drop=True)
    if f.empty:
        raise RuntimeError("No valid overlapping forecast and market rows.")
    if (f[["Hist_126", "Hist_252", "RF_126", "RF_252"]] <= 0).any().any():
        raise ValueError("Non-positive volatility found in forecast data.")
    return f


def main():
    cfg = load_config()
    data = prepare_data()
    rows = []
    for x in data.itertuples(index=False):
        common = dict(S0=float(x.S0), r=float(x.r), K=cfg["strike"],
                      T1=cfg["T1"], T2=cfg["T2"], n=cfg["n"], seed=SEED)
        for strategy, s1, s2 in [
            ("Historical_60D", float(x.Hist_126), float(x.Hist_252)),
            ("Random_Forest", float(x.RF_126), float(x.RF_252)),
        ]:
            result = price_mc(**common, sigma1=s1, sigma2=s2)
            rows.append({
                "Date": x.Date.date().isoformat(),
                "Strategy": strategy,
                "Underlying_Close": float(x.S0),
                "Risk_Free_Rate": float(x.r),
                "Sigma_First_Period_126D": s1,
                "Sigma_Second_Period_252D": s2,
                "Strike": cfg["strike"], "T1_Years": cfg["T1"],
                "T2_Years": cfg["T2"], "N_Simulations": cfg["n"],
                **result,
            })
    detail = pd.DataFrame(rows)
    base = detail[detail.Strategy == "Historical_60D"][["Date", "Chooser_Price"]]
    base = base.rename(columns={"Chooser_Price": "Historical_60D_Price"})
    detail = detail.merge(base, on="Date", how="left", validate="many_to_one")
    detail["Price_Difference_vs_Historical"] = (
        detail["Chooser_Price"] - detail["Historical_60D_Price"])
    detail["Price_Difference_Percent_vs_Historical"] = (
        detail["Price_Difference_vs_Historical"]
        / detail["Historical_60D_Price"] * 100)

    summary = detail.groupby("Strategy").agg(
        Observations=("Chooser_Price", "size"),
        Mean_Sigma_126D=("Sigma_First_Period_126D", "mean"),
        Mean_Sigma_252D=("Sigma_Second_Period_252D", "mean"),
        Mean_Chooser_Price=("Chooser_Price", "mean"),
        Median_Chooser_Price=("Chooser_Price", "median"),
        Mean_MC_Standard_Error=("MC_Standard_Error", "mean"),
        Mean_Call_Ratio=("Call_Ratio", "mean"),
        Mean_Put_Ratio=("Put_Ratio", "mean"),
        Mean_Price_Difference=("Price_Difference_vs_Historical", "mean"),
        Mean_Price_Difference_Percent=(
            "Price_Difference_Percent_vs_Historical", "mean"),
    ).reset_index()

    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT / "week8_dual_chooser_pricing.csv", index=False)
    summary.to_csv(OUT / "week8_dual_chooser_summary.csv", index=False)
    print("=" * 68)
    print("WEEK 8: DUAL TWO-STAGE CHOOSER OPTION PRICING")
    print("=" * 68)
    print(f"Matched valuation dates: {detail.Date.nunique()}")
    print(f"First period: 0 to {cfg['T1']} years, 126D volatility")
    print(f"Second period: {cfg['T1']} to {cfg['T2']} years, 252D volatility")
    print(f"Strike: {cfg['strike']:.2f}; simulations: {cfg['n']}")
    print(summary.round(6).to_string(index=False))
    print("\nTheoretical prices only; these are not validated against market option quotes.")
    print(f"Detailed output: {OUT / 'week8_dual_chooser_pricing.csv'}")
    print(f"Summary output: {OUT / 'week8_dual_chooser_summary.csv'}")


if __name__ == "__main__":
    main()
