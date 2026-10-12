
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "jpm_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "week5"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

FEATURE_COLUMNS = [
    "Close", "Adj Close", "Volume", "DGS10", "VIXCLS",
    "Close_outlier", "Adj Close_outlier", "Volume_outlier",
    "DGS10_outlier", "VIXCLS_outlier", "Dividend_Growth",
    "Daily_Return", "Rolling_Volatility_5D",
    "Rolling_Volatility_20D", "Rolling_Volatility_60D",
    "Price_Momentum_20D", "Volume_Change",
    "VIX_JPM_Correlation_20D", "Interest_Rate_Momentum_20D",
    "Sentiment_Score", "VIX_Change_20D",
    "Treasury_Yield_Change_20D",
]


def main():
    print("Loading feature dataset...")

    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = (
        df.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Define the next trading date and its return as the target.
    df["Target_Date"] = df["Date"].shift(-1)
    df["Target_Next_Day_Return"] = df["Daily_Return"].shift(-1)
    df["Target_Next_Day_Abs_Return"] = (
        df["Target_Next_Day_Return"].abs()
    )

    required = FEATURE_COLUMNS + [
        "Target_Date",
        "Target_Next_Day_Return",
        "Target_Next_Day_Abs_Return",
    ]

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required).reset_index(drop=True)

    # Create chronological blocks using feature dates.
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    validation_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

    train_start = df.iloc[0]["Date"]
    train_last = df.iloc[train_end - 1]["Date"]

    val_start = df.iloc[train_end]["Date"]
    val_last = df.iloc[validation_end - 1]["Date"]

    test_start = df.iloc[validation_end]["Date"]
    test_last = df.iloc[-1]["Date"]

    # Keep only rows whose target date is within the same block.
    train_df = df[
        (df["Date"] >= train_start)
        & (df["Date"] <= train_last)
        & (df["Target_Date"] <= train_last)
    ].copy()

    validation_df = df[
        (df["Date"] >= val_start)
        & (df["Date"] <= val_last)
        & (df["Target_Date"] <= val_last)
    ].copy()

    test_df = df[
        (df["Date"] >= test_start)
        & (df["Date"] <= test_last)
        & (df["Target_Date"] <= test_last)
    ].copy()

    parts = [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for name, part in parts:
        if part.empty:
            raise ValueError(f"{name} dataset is empty.")

       

        part.to_csv(
            OUTPUT_DIR / f"{name.lower()}.csv",
            index=False,
        )

        summary_rows.append({
            "Dataset": name,
            "Rows": len(part),
            "Start_Date": part["Date"].min().date(),
            "End_Date": part["Date"].max().date(),
            "First_Target_Date": part["Target_Date"].min().date(),
            "Last_Target_Date": part["Target_Date"].max().date(),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    print("\nWeek 5 dataset preparation completed.")
    print(f"Rows after target construction and cleaning: {n}")
    print("\nChronological split:")
    print(summary.to_string(index=False))
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
