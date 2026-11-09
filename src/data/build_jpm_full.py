import pandas as pd
from pathlib import Path

files = [
    "data/raw/jpm_daily.csv",
    "data/raw/jpm_2025_massive.csv",
    "data/raw/jpm_2026_massive.csv"
]

dfs = []

for file in files:
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""),
                errors="coerce"
            )

    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# Massive data is appended later, so keep the latest source on duplicate dates.
combined = combined.drop_duplicates(subset="Date", keep="last")
combined = combined.sort_values("Date")

output = Path("data/raw/jpm_daily_full.csv")
combined.to_csv(output, index=False)

print("Saved:", output)
print("Rows:", len(combined))
print("Date range:", combined["Date"].min(), "to", combined["Date"].max())
print("\nYearly row counts:")
print(combined.groupby(combined["Date"].dt.year).size().to_string())
print("\nMissing values:")
print(combined.isna().sum().to_string())
