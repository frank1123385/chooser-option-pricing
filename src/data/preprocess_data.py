
from pathlib import Path

import numpy as np
import pandas as pd


# ==============================
# Paths
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "jpm_merged.csv"
DIVIDEND_FILE = PROJECT_ROOT / "data" / "raw" / "jpm_dividends.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "jpm_features.csv"


# ==============================
# Load and validate merged data
# ==============================

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "Date", "Close", "Adj Close", "Volume", "DGS10", "VIXCLS"
]

missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

for column in ["Close", "Adj Close", "Volume", "DGS10", "VIXCLS"]:
    df[column] = pd.to_numeric(
        df[column].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

df = (
    df.dropna(subset=["Date"])
    .sort_values("Date")
    .drop_duplicates(subset="Date", keep="last")
    .reset_index(drop=True)
)

numeric_columns = ["Close", "Adj Close", "Volume", "DGS10", "VIXCLS"]

# Treat infinite values as missing.
df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf], np.nan
)

print("Data types:")
print(df.dtypes)

print("\nMissing values before cleaning:")
print(df[required_columns].isna().sum())

print("\nDuplicate dates:", df["Date"].duplicated().sum())


# ==============================
# Missing values
# ==============================

# Forward-fill only: never interpolate using future observations.
# Rows missing values at the beginning cannot be forward-filled and are dropped.
df[numeric_columns] = df[numeric_columns].ffill()
df = df.dropna(subset=numeric_columns).reset_index(drop=True)

print("\nMissing values after forward-fill:")
print(df[numeric_columns].isna().sum())


# ==============================
# Historical outlier flags
# ==============================

# Each day's thresholds use only observations before that day.
# The first 60 observations have no historical threshold and are flagged 0.
print("\nHistorical IQR outlier detection:")

for column in numeric_columns:
    historical_q1 = (
        df[column].expanding(min_periods=60).quantile(0.25).shift(1)
    )
    historical_q3 = (
        df[column].expanding(min_periods=60).quantile(0.75).shift(1)
    )

    historical_iqr = historical_q3 - historical_q1
    lower_bound = historical_q1 - 1.5 * historical_iqr
    upper_bound = historical_q3 + 1.5 * historical_iqr

    flag_column = f"{column}_outlier"

    df[flag_column] = (
        (df[column] < lower_bound) | (df[column] > upper_bound)
    ).fillna(False).astype(int)

    outlier_count = int(df[flag_column].sum())

    print(
        f"{column}: {outlier_count} outliers "
        f"({outlier_count / len(df) * 100:.2f}%)"
    )


# ==============================
# Dividend growth
# ==============================

dividend_df = pd.read_csv(DIVIDEND_FILE)

dividend_required = ["Year", "Annual_Dividend"]
missing_dividend_columns = [
    c for c in dividend_required if c not in dividend_df.columns
]
if missing_dividend_columns:
    raise ValueError(
        f"Missing dividend columns: {missing_dividend_columns}"
    )

dividend_df["Year"] = pd.to_numeric(
    dividend_df["Year"], errors="coerce"
)
dividend_df["Annual_Dividend"] = pd.to_numeric(
    dividend_df["Annual_Dividend"], errors="coerce"
)

dividend_df = (
    dividend_df.dropna(subset=dividend_required)
    .sort_values("Year")
    .drop_duplicates(subset="Year", keep="last")
)

dividend_df["Dividend_Growth"] = (
    dividend_df["Annual_Dividend"].pct_change()
)

df["Year"] = df["Date"].dt.year

# Conservative one-calendar-year lag.
# Annual totals have no announcement/availability dates in the source file.
dividend_growth_by_year = dividend_df.set_index("Year")["Dividend_Growth"]

df["Dividend_Growth"] = (
    (df["Year"] - 1).map(dividend_growth_by_year)
)

df = df.drop(columns=["Year"])

print("\nDividend growth feature created with a one-year lag.")
print(df["Dividend_Growth"].describe())


# ==============================
# Traditional features
# ==============================

df["Daily_Return"] = df["Adj Close"].pct_change()

df["Rolling_Volatility_5D"] = (
    df["Daily_Return"].rolling(window=5).std()
)

df["Rolling_Volatility_20D"] = (
    df["Daily_Return"].rolling(window=20).std()
)

df["Rolling_Volatility_60D"] = (
    df["Daily_Return"].rolling(window=60).std()
)

df["Price_Momentum_20D"] = (
    df["Adj Close"] / df["Adj Close"].shift(20) - 1
)

df["Volume_Change"] = df["Volume"].pct_change()


# ==============================
# Advanced features
# ==============================

df["VIX_JPM_Correlation_20D"] = (
    df["Daily_Return"].rolling(window=20).corr(df["VIXCLS"])
)

df["Interest_Rate_Momentum_20D"] = (
    df["DGS10"] - df["DGS10"].shift(20)
)

# Fixed-scale VIX-derived proxy, not actual news sentiment.
df["Sentiment_Score"] = (
    1 - df["VIXCLS"] / 100
).clip(0, 1)

df["VIX_Change_20D"] = (
    df["VIXCLS"] / df["VIXCLS"].shift(20) - 1
)

df["Treasury_Yield_Change_20D"] = (
    df["DGS10"] - df["DGS10"].shift(20)
)


# ==============================
# Final cleaning and save
# ==============================

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

# Remove rows without enough historical information for features.
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_columns).reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature engineering completed.")
print("Rows after feature cleaning:", len(df))
print("Final missing values in features:")
print(df[feature_columns].isna().sum())
print(f"\nSaved to: {OUTPUT_FILE}")
print("Final dataset shape:", df.shape)
