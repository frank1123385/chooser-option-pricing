"""Build Week 8 future annualized realized-volatility targets.

Input:
    data/processed/jpm_features_week8.csv

Output:
    data/processed/volatility_experiment_week8/
        20d/{train,validation,test}.csv
        126d/{train,validation,test}.csv
        252d/{train,validation,test}.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ==================================
# Paths and configuration
# ==================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "jpm_features_week8.csv"
)

OUTPUT_ROOT = (
    ROOT / "data" / "processed" / "volatility_experiment_week8"
)

HORIZONS = (20, 126, 252)

TRAIN_RATIO = 0.50
VALIDATION_RATIO = 0.25


# ==================================
# Week 8 model predictors: 20
# Adj Close and Adj Close_outlier excluded
# ==================================

FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "DGS10",
    "VIXCLS",
    "Close_outlier",
    "Volume_outlier",
    "DGS10_outlier",
    "VIXCLS_outlier",
    "Dividend_Growth",
    "Daily_Return",
    "Rolling_Volatility_5D",
    "Rolling_Volatility_20D",
    "Rolling_Volatility_60D",
    "Price_Momentum_20D",
    "Volume_Change",
    "VIX_JPM_Correlation_20D",
    "Interest_Rate_Momentum_20D",
    "Sentiment_Score",
    "VIX_Change_20D",
    "Treasury_Yield_Change_20D",
]


# ==================================
# Future annualized volatility
# ==================================

def forward_annualized_volatility(returns, horizon):
    """
    At date t, calculate sample SD using returns:
    t+1, t+2, ..., t+horizon.

    Annualize using sqrt(252).
    """

    future_returns = (
        pd.to_numeric(returns, errors="coerce").shift(-1)
    )

    forward_std = (
        future_returns.iloc[::-1]
        .rolling(window=horizon, min_periods=horizon)
        .std(ddof=1)
        .iloc[::-1]
    )

    return forward_std * np.sqrt(252)


# ==================================
# Main
# ==================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    df = (
        df.dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    # Check required columns
    missing = [
        c for c in FEATURE_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}"
        )

    if len(FEATURE_COLUMNS) != 20:
        raise ValueError(
            "Expected exactly 20 model predictors."
        )

    if len(df) < 1000:
        raise ValueError(
            f"Only {len(df)} rows found. Check input history."
        )

    # Check feature values
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )

    missing_values = df[FEATURE_COLUMNS].isna().sum()

    if missing_values.any():
        raise ValueError(
            "Missing values found in model predictors:\n"
            + missing_values[missing_values > 0].to_string()
        )

    print("========== WEEK 8 TARGET GENERATION ==========")
    print("Input:", INPUT_FILE)
    print("Input rows:", len(df))
    print("Predictor count:", len(FEATURE_COLUMNS))
    print("Date range:", df["Date"].min(), "to", df["Date"].max())

    # ==================================
    # Chronological split boundaries
    # ==================================

    n = len(df)

    i1 = int(n * TRAIN_RATIO)
    i2 = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

    blocks = [
        (
            "train",
            df.iloc[0]["Date"],
            df.iloc[i1 - 1]["Date"],
        ),
        (
            "validation",
            df.iloc[i1]["Date"],
            df.iloc[i2 - 1]["Date"],
        ),
        (
            "test",
            df.iloc[i2]["Date"],
            df.iloc[-1]["Date"],
        ),
    ]

    print("\nInitial chronological boundaries:")

    for split, start, end in blocks:
        print(f"{split}: {start.date()} to {end.date()}")

    summaries = []

    # ==================================
    # Generate each horizon
    # ==================================

    for h in HORIZONS:

        target = f"Target_Annualized_RV_{h}D"

        work = df.copy()

        # Date when the future volatility window ends
        work["Target_Date"] = work["Date"].shift(-h)

        # Future realized volatility
        work[target] = forward_annualized_volatility(
            work["Daily_Return"], h
        )

        work = work.replace(
            [np.inf, -np.inf], np.nan
        )

        work = work.dropna(
            subset=["Target_Date", target]
        )

        out_dir = OUTPUT_ROOT / f"{h}d"
        out_dir.mkdir(parents=True, exist_ok=True)

        horizon_summaries = []

        for split, start, end in blocks:

            # Purge observations whose future target window
            # crosses the current split boundary.
            part = work[
                (work["Date"] >= start)
                & (work["Date"] <= end)
                & (work["Target_Date"] <= end)
            ].copy()

            if part.empty:
                raise ValueError(
                    f"{split} is empty for horizon {h}D."
                )

            # Save only the 20 predictors plus dates and target.
            output_columns = (
                ["Date"]
                + FEATURE_COLUMNS
                + ["Target_Date", target]
            )

            part = part[output_columns]

            output_file = out_dir / f"{split}.csv"

            part.to_csv(output_file, index=False)

            summary = {
                "Horizon_Days": h,
                "Split": split,
                "Rows": len(part),
                "Feature_Start": part["Date"].min().date(),
                "Feature_End": part["Date"].max().date(),
                "First_Target_Date": part["Target_Date"].min().date(),
                "Last_Target_Date": part["Target_Date"].max().date(),
                "Target": target,
            }

            summaries.append(summary)
            horizon_summaries.append(summary)

        print(f"\n========== {h}-DAY HORIZON ==========")

        print(
            pd.DataFrame(horizon_summaries).to_string(
                index=False
            )
        )

    # ==================================
    # Save summary
    # ==================================

    OUTPUT_ROOT.mkdir(
        parents=True, exist_ok=True
    )

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        OUTPUT_ROOT / "split_summary.csv",
        index=False,
    )

    print("\n========== COMPLETED ==========")
    print("Output directory:", OUTPUT_ROOT)
    print("Target horizons:", HORIZONS)
    print("Model predictors:", len(FEATURE_COLUMNS))
    print("Original Week 7 files were preserved.")
    print(
        "Target = future sample SD of H daily returns "
        "multiplied by sqrt(252)."
    )


if __name__ == "__main__":
    main()
