
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "jpm_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "week5"

# Chronological split proportions
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

# Candidate features available in jpm_features.csv
FEATURE_COLUMNS = [
    "Close",
    "Adj Close",
    "Volume",
    "DGS10",
    "VIXCLS",
    "Close_outlier",
    "Adj Close_outlier",
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


def main():
    print("Loading feature dataset...")
    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Target: absolute return on the next trading observation.
    # This is a one-day volatility proxy, not a multi-day realized volatility.
    df["Target_Next_Day_Abs_Return"] = df["Daily_Return"].shift(-1).abs()

    # Remove the last row because its next-day target is unavailable.
    df = df.dropna(subset=["Target_Next_Day_Abs_Return"]).copy()

    # Validate feature columns.
    missing_columns = [
        col for col in FEATURE_COLUMNS if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    # Replace infinite values and remove rows with missing model inputs.
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    # Chronological 70% / 15% / 15% split.
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    validation_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save datasets with dates and target for traceability.
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    # Save a compact metadata summary.
    summary = pd.DataFrame(
        [
            {
                "Dataset": name,
                "Rows": len(part),
                "Start_Date": part["Date"].min().date(),
                "End_Date": part["Date"].max().date(),
            }
            for name, part in [
                ("Train", train_df),
                ("Validation", validation_df),
                ("Test", test_df),
            ]
        ]
    )
    summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    print("\nWeek 5 dataset preparation completed.")
    print(f"Input rows: {len(pd.read_csv(INPUT_FILE))}")
    print(f"Rows after target construction and cleaning: {n}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Target: Target_Next_Day_Abs_Return")
    print("\nChronological split:")
    print(summary.to_string(index=False))
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
