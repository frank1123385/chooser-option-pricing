import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/week4_bsm_predictions.csv"
OUTPUT_FILE = "data/processed/week4_volatility_analysis.csv"


# =========================
# Load data
# =========================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])


# =========================
# Define VIX regimes
# =========================

def classify_vix(vix):
    if vix < 20:
        return "Normal"
    elif vix < 30:
        return "High"
    else:
        return "Very High"


df["VIX_Regime"] = df["VIXCLS"].apply(
    classify_vix
)


# =========================
# Calculate regime statistics
# =========================

summary = (
    df.groupby("VIX_Regime")
    .agg(
        Observations=("Date", "count"),
        Average_VIX=("VIXCLS", "mean"),
        Average_Volatility=("sigma", "mean"),
        Average_BSM_Price=("BSM_Predicted_Price", "mean"),
        Average_Call_Ratio=("Call_Ratio", "mean"),
        Average_Put_Ratio=("Put_Ratio", "mean"),
    )
    .reset_index()
)


# =========================
# Sort regimes
# =========================

regime_order = [
    "Normal",
    "High",
    "Very High"
]

summary["VIX_Regime"] = pd.Categorical(
    summary["VIX_Regime"],
    categories=regime_order,
    ordered=True
)

summary = summary.sort_values(
    "VIX_Regime"
)


# =========================
# Save summary
# =========================

summary.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================
# Print results
# =========================

print("=" * 70)
print("Week 4 High-Volatility Analysis")
print("=" * 70)

print("\nVIX Regime Definitions:")
print("Normal: VIX < 20")
print("High: 20 <= VIX < 30")
print("Very High: VIX >= 30")

print("\nRegime Statistics:")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nOutput file:")
print(OUTPUT_FILE)

print("=" * 70)