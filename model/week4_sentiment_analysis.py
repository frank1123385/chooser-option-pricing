import pandas as pd


INPUT_FILE = "data/processed/week4_bsm_predictions.csv"
OUTPUT_FILE = "data/processed/week4_sentiment_analysis.csv"


# =========================
# Load data
# =========================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])


# =========================
# Define sentiment regimes
# =========================

def classify_sentiment(score):

    if score < 0.33:
        return "Low"

    elif score <= 0.67:
        return "Medium"

    else:
        return "High"


df["Sentiment_Regime"] = df["Sentiment_Score"].apply(
    classify_sentiment
)


# =========================
# Calculate statistics
# =========================

summary = (
    df.groupby("Sentiment_Regime")
    .agg(
        Observations=("Date", "count"),
        Average_Sentiment=("Sentiment_Score", "mean"),
        Average_VIX=("VIXCLS", "mean"),
        Average_Volatility=("sigma", "mean"),
        Average_BSM_Price=("BSM_Predicted_Price", "mean"),
        Average_Call_Ratio=("Call_Ratio", "mean"),
        Average_Put_Ratio=("Put_Ratio", "mean"),
    )
    .reset_index()
)


# =========================
# Sort sentiment regimes
# =========================

regime_order = [
    "Low",
    "Medium",
    "High"
]

summary["Sentiment_Regime"] = pd.Categorical(
    summary["Sentiment_Regime"],
    categories=regime_order,
    ordered=True
)

summary = summary.sort_values(
    "Sentiment_Regime"
)


# =========================
# Save results
# =========================

summary.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================
# Print results
# =========================

print("=" * 70)
print("Week 4 Sentiment Impact Analysis")
print("=" * 70)

print("\nSentiment Regime Definitions:")
print("Low:    Sentiment Score < 0.33")
print("Medium: 0.33 <= Sentiment Score <= 0.67")
print("High:   Sentiment Score > 0.67")

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