"""Build 20/126/252-day future annualized realized-volatility targets.
Run from the repository root:
    python src/model/volatility_feature_engineering.py

Input: data/processed/jpm_features.csv
Output: data/processed/volatility_experiment/<horizon>d/{train,validation,test}.csv
Original Week 5 files are not overwritten.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "jpm_features.csv"
OUTPUT_ROOT = ROOT / "data" / "processed" / "volatility_experiment"
HORIZONS = (20, 126, 252)
TRAIN_RATIO = 0.50
VALIDATION_RATIO = 0.25
FEATURE_COLUMNS = [
    "Close", "Adj Close", "Volume", "DGS10", "VIXCLS",
    "Close_outlier", "Adj Close_outlier", "Volume_outlier",
    "DGS10_outlier", "VIXCLS_outlier", "Dividend_Growth",
    "Daily_Return", "Rolling_Volatility_5D", "Rolling_Volatility_20D",
    "Rolling_Volatility_60D", "Price_Momentum_20D", "Volume_Change",
    "VIX_JPM_Correlation_20D", "Interest_Rate_Momentum_20D",
    "Sentiment_Score", "VIX_Change_20D", "Treasury_Yield_Change_20D",
]

def forward_annualized_volatility(returns, horizon):
    # At t, calculate sample SD from returns t+1 through t+horizon.
    future_returns = pd.to_numeric(returns, errors="coerce").shift(-1)
    forward_std = (
        future_returns.iloc[::-1]
        .rolling(window=horizon, min_periods=horizon)
        .std(ddof=1)
        .iloc[::-1]
    )
    return forward_std * np.sqrt(252)

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = (df.dropna(subset=["Date"]).sort_values("Date")
          .drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True))
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if len(df) < 1000:
        raise ValueError(f"Only {len(df)} rows found; check the input history.")
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    n = len(df)
    i1 = int(n * TRAIN_RATIO)
    i2 = int(n * (TRAIN_RATIO + VALIDATION_RATIO))
    blocks = [
        ("train", df.iloc[0]["Date"], df.iloc[i1 - 1]["Date"]),
        ("validation", df.iloc[i1]["Date"], df.iloc[i2 - 1]["Date"]),
        ("test", df.iloc[i2]["Date"], df.iloc[-1]["Date"]),
    ]

    summaries = []
    for h in HORIZONS:
        target = f"Target_Annualized_RV_{h}D"
        work = df.copy()
        work["Target_Date"] = work["Date"].shift(-h)
        work[target] = forward_annualized_volatility(work["Daily_Return"], h)
        work = work.replace([np.inf, -np.inf], np.nan)
        work = work.dropna(subset=["Target_Date", target])

        out_dir = OUTPUT_ROOT / f"{h}d"
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, start, end in blocks:
            # Purge rows whose future target window crosses the next split boundary.
            part = work[
                (work["Date"] >= start) & (work["Date"] <= end)
                & (work["Target_Date"] <= end)
            ].copy()
            if part.empty:
                raise ValueError(f"{split} is empty for horizon {h}D.")
            part.to_csv(out_dir / f"{split}.csv", index=False)
            summaries.append({
                "Horizon_Days": h, "Split": split, "Rows": len(part),
                "Feature_Start": part["Date"].min().date(),
                "Feature_End": part["Date"].max().date(),
                "First_Target_Date": part["Target_Date"].min().date(),
                "Last_Target_Date": part["Target_Date"].max().date(),
                "Target": target,
            })
        print(f"\n{h}-day horizon")
        print(pd.DataFrame([x for x in summaries if x["Horizon_Days"] == h]).to_string(index=False))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUTPUT_ROOT / "split_summary.csv", index=False)
    print(f"\nCompleted. Original Week 5 files were preserved.")
    print(f"Output: {OUTPUT_ROOT}")
    print("Target = sample SD of next H daily returns × sqrt(252).")

if __name__ == "__main__":
    main()
