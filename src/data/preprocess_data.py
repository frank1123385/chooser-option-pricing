import pandas as pd

# Load merged data
input_file = "data/processed/jpm_merged.csv"
df = pd.read_csv(input_file)

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Convert Volume from string to numeric
df["Volume"] = (
    df["Volume"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# Check data types
print(df.dtypes)

# Check missing values
print("\nMissing values:")
print(df.isna().sum())

# Check duplicate dates
print("\nDuplicate dates:", df["Date"].duplicated().sum())

# Detect outliers using IQR method
numeric_columns = ["Close", "Adj Close", "Volume", "DGS10", "VIXCLS"]

print("\nIQR Outlier Detection:")

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(
        f"{column}: {len(outliers)} outliers "
        f"(lower={lower_bound:.2f}, upper={upper_bound:.2f})"
    )

# Time alignment
df = df.sort_values("Date").reset_index(drop=True)

# Remove duplicate dates if any
df = df.drop_duplicates(subset="Date")

# Handle missing values using interpolation
interpolation_columns = [
    "Close",
    "Adj Close",
    "Volume",
    "DGS10",
    "VIXCLS"
]

df[interpolation_columns] = (
    df[interpolation_columns]
    .interpolate(method="linear")
    .ffill()
    .bfill()
)

print("\nMissing values after interpolation:")
print(df[interpolation_columns].isna().sum())

# Create IQR-based outlier flags
for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[f"{column}_outlier"] = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ).astype(int)

print("\nOutlier flags created.")

# ==============================
# Feature Engineering
# Traditional Features
# ==============================

# Daily return
df["Daily_Return"] = df["Adj Close"].pct_change()

# 5-day rolling volatility
df["Rolling_Volatility_5D"] = (
    df["Daily_Return"].rolling(window=5).std()
)

# 20-day rolling volatility
df["Rolling_Volatility_20D"] = (
    df["Daily_Return"].rolling(window=20).std()
)

# 60-day rolling volatility
df["Rolling_Volatility_60D"] = (
    df["Daily_Return"].rolling(window=60).std()
)

# Price momentum
df["Price_Momentum_20D"] = (
    df["Adj Close"] / df["Adj Close"].shift(20) - 1
)

# Volume change
df["Volume_Change"] = df["Volume"].pct_change()

print("\nTraditional features created:")
print([
    "Daily_Return",
    "Rolling_Volatility_5D",
    "Rolling_Volatility_20D",
    "Rolling_Volatility_60D",
    "Price_Momentum_20D",
    "Volume_Change"
])

# ==============================
# Advanced Features
# ==============================

# VIX-JPM rolling correlation
df["VIX_JPM_Correlation_20D"] = (
    df["Daily_Return"]
    .rolling(window=20)
    .corr(df["VIXCLS"])
)

# Interest rate momentum
df["Interest_Rate_Momentum_20D"] = (
    df["DGS10"] - df["DGS10"].shift(20)
)

# VIX-based sentiment proxy
# Higher VIX -> lower sentiment score
vix_min = df["VIXCLS"].min()
vix_max = df["VIXCLS"].max()

df["Sentiment_Score"] = 1 - (
    (df["VIXCLS"] - vix_min) /
    (vix_max - vix_min)
)

# 20-day VIX change
df["VIX_Change_20D"] = (
    df["VIXCLS"] / df["VIXCLS"].shift(20) - 1
)

# 20-day Treasury yield change
df["Treasury_Yield_Change_20D"] = (
    df["DGS10"] - df["DGS10"].shift(20)
)

print("\nAdvanced features created:")
print([
    "VIX_JPM_Correlation_20D",
    "Interest_Rate_Momentum_20D",
    "Sentiment_Score",
    "VIX_Change_20D",
    "Treasury_Yield_Change_20D"
])

# Remove rows with insufficient history for feature calculation
feature_columns = [
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
    "Treasury_Yield_Change_20D"
]

df = df.dropna(subset=feature_columns).reset_index(drop=True)

print("\nRows after feature NaN removal:", len(df))
print("\nFinal missing values:")
print(df.isna().sum())

# Save cleaned and feature-engineered dataset
output_file = "data/processed/jpm_features.csv"
df.to_csv(output_file, index=False)

print(f"\nSaved processed dataset to: {output_file}")
print(f"Final dataset shape: {df.shape}")