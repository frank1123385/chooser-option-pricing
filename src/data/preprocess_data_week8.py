from pathlib import Path

import numpy as np
import pandas as pd


# ==================================
# Paths: Week 8 only
# ==================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/jpm_merged_full.csv"
DIVIDEND_FILE = ROOT / "data/raw/jpm_dividends_week8.csv"
OUTPUT_FILE = ROOT / "data/processed/jpm_features_week8.csv"


# ==================================
# Load data
# ==================================

df = pd.read_csv(INPUT_FILE)
dividends = pd.read_csv(DIVIDEND_FILE)

required = [
    "Date", "Close", "Volume", "DGS10", "VIXCLS"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

for col in ["Close", "Volume", "DGS10", "VIXCLS"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = (
    df.dropna(subset=["Date"])
    .sort_values("Date")
    .drop_duplicates(subset="Date", keep="last")
    .reset_index(drop=True)
)

numeric_columns = ["Close", "Volume", "DGS10", "VIXCLS"]

df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf], np.nan
)

df[numeric_columns] = df[numeric_columns].ffill()
df = df.dropna(subset=numeric_columns).reset_index(drop=True)

print("Merged input rows:", len(df))
print("Input date range:", df["Date"].min(), "to", df["Date"].max())


# ==================================
# Historical outlier flags
# ==================================

print("\nCreating historical outlier flags...")

for col in numeric_columns:

    q1 = (
        df[col]
        .expanding(min_periods=60)
        .quantile(0.25)
        .shift(1)
    )

    q3 = (
        df[col]
        .expanding(min_periods=60)
        .quantile(0.75)
        .shift(1)
    )

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    flag = f"{col}_outlier"

    df[flag] = (
        (df[col] < lower) | (df[col] > upper)
    ).fillna(False).astype(int)

    print(f"{flag}: {df[flag].sum()}")


# ==================================
# Dividend growth
# ==================================

dividends["Year"] = pd.to_numeric(
    dividends["Year"], errors="coerce"
)

dividends["Annual_Dividend"] = pd.to_numeric(
    dividends["Annual_Dividend"], errors="coerce"
)

dividends = (
    dividends.dropna(subset=["Year", "Annual_Dividend"])
    .sort_values("Year")
    .drop_duplicates(subset="Year", keep="last")
)

dividends["Dividend_Growth"] = (
    dividends["Annual_Dividend"].pct_change()
)

growth_map = dividends.set_index("Year")["Dividend_Growth"]

df["Year"] = df["Date"].dt.year

# One-calendar-year lag
df["Dividend_Growth"] = (
    (df["Year"] - 1).map(growth_map)
)

print("\nDividend growth by year:")
print(
    df.groupby("Year")["Dividend_Growth"]
    .first()
    .to_string()
)


# ==================================
# Traditional features
# Use Close consistently
# ==================================

df["Daily_Return"] = df["Close"].pct_change()

df["Rolling_Volatility_5D"] = (
    df["Daily_Return"].rolling(5).std()
)

df["Rolling_Volatility_20D"] = (
    df["Daily_Return"].rolling(20).std()
)

df["Rolling_Volatility_60D"] = (
    df["Daily_Return"].rolling(60).std()
)

df["Price_Momentum_20D"] = (
    df["Close"] / df["Close"].shift(20) - 1
)

df["Volume_Change"] = df["Volume"].pct_change()


# ==================================
# Advanced features
# ==================================

df["VIX_JPM_Correlation_20D"] = (
    df["Daily_Return"].rolling(20).corr(df["VIXCLS"])
)

df["Interest_Rate_Momentum_20D"] = (
    df["DGS10"] - df["DGS10"].shift(20)
)

# VIX-derived proxy, not actual news sentiment
df["Sentiment_Score"] = (
    1 - df["VIXCLS"] / 100
).clip(0, 1)

df["VIX_Change_20D"] = (
    df["VIXCLS"] / df["VIXCLS"].shift(20) - 1
)

df["Treasury_Yield_Change_20D"] = (
    df["DGS10"] - df["DGS10"].shift(20)
)


# ==================================
# Final cleaning
# ==================================

feature_columns = [
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

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=feature_columns
).reset_index(drop=True)

# Keep the 20 model predictors.
# Adj Close is intentionally excluded because its price basis
# is inconsistent between historical and newly downloaded data.

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)


# ==================================
# Validation
# ==================================

print("\n========== WEEK 8 FEATURE ENGINEERING ==========")

print("Output file:", OUTPUT_FILE)
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nDate range:")
print(df["Date"].min(), "to", df["Date"].max())

print("\nYearly observations:")
print(df.groupby(df["Date"].dt.year).size().to_string())

print("\nMissing values in model features:")
print(df[feature_columns].isna().sum().to_string())

print("\nModel predictor count:", 20)

print("\n2025-2026 dividend growth:")
print(
    df.loc[df["Date"].dt.year >= 2025,
           ["Date", "Dividend_Growth"]]
    .groupby(df["Date"].dt.year)
    .first()
    .to_string()
)

print("\nFeature engineering completed successfully.")
