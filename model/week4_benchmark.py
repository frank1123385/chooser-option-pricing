import numpy as np
import pandas as pd
from pathlib import Path

from bsm_chooser import simulate_chooser_option


# =========================
# Configuration
# =========================

INPUT_FILE = "data/processed/jpm_features.csv"
OUTPUT_FILE = "data/processed/week4_bsm_predictions.csv"

STRIKE = 150.0
T1 = 0.5
T2 = 1.0

N_SIMULATIONS = 10000
SEED = 42


# =========================
# Load data
# =========================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# =========================
# Check required columns
# =========================

required_columns = [
    "Date",
    "Adj Close",
    "DGS10",
    "Rolling_Volatility_60D",
    "VIXCLS",
    "Sentiment_Score",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =========================
# Prepare model inputs
# =========================

df["S0"] = df["Adj Close"]

df["risk_free_rate"] = df["DGS10"] / 100

df["sigma"] = df["Rolling_Volatility_60D"] * np.sqrt(252)


df = df.dropna(
    subset=[
        "S0",
        "risk_free_rate",
        "sigma"
    ]
).copy()


# =========================
# Run BSM Chooser Model
# =========================

predicted_prices = []
call_ratios = []
put_ratios = []

for _, row in df.iterrows():

    result = simulate_chooser_option(
        S0=float(row["S0"]),
        risk_free_rate=float(row["risk_free_rate"]),
        sigma=float(row["sigma"]),
        strike=STRIKE,
        T1=T1,
        T2=T2,
        n_simulations=N_SIMULATIONS,
        seed=SEED,
    )

    predicted_prices.append(
        result["Chooser_Price"]
    )

    call_count = np.sum(
        result["Choice"] == "CALL"
    )

    put_count = np.sum(
        result["Choice"] == "PUT"
    )

    total_count = call_count + put_count

    call_ratios.append(
        call_count / total_count
    )

    put_ratios.append(
        put_count / total_count
    )


df["BSM_Predicted_Price"] = predicted_prices

df["Call_Ratio"] = call_ratios

df["Put_Ratio"] = put_ratios


# =========================
# Create benchmark dataset
# =========================

benchmark_df = df[
    [
        "Date",
        "S0",
        "risk_free_rate",
        "sigma",
        "VIXCLS",
        "Sentiment_Score",
        "BSM_Predicted_Price",
        "Call_Ratio",
        "Put_Ratio",
    ]
].copy()


# =========================
# Save
# =========================

Path(OUTPUT_FILE).parent.mkdir(
    parents=True,
    exist_ok=True
)

benchmark_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================
# Summary
# =========================

print("=" * 60)
print("Week 4 BSM Benchmark Prediction")
print("=" * 60)

print(f"Number of observations: {len(benchmark_df)}")

print(
    f"Prediction period: "
    f"{benchmark_df['Date'].min().date()} "
    f"to "
    f"{benchmark_df['Date'].max().date()}"
)

print(
    f"Average BSM predicted price: "
    f"${benchmark_df['BSM_Predicted_Price'].mean():.4f}"
)

print(
    f"Minimum BSM predicted price: "
    f"${benchmark_df['BSM_Predicted_Price'].min():.4f}"
)

print(
    f"Maximum BSM predicted price: "
    f"${benchmark_df['BSM_Predicted_Price'].max():.4f}"
)

print("\nFirst 10 observations:")

print(
    benchmark_df.head(10).to_string(
        index=False
    )
)

print("\nOutput file:")
print(OUTPUT_FILE)

print("=" * 60)