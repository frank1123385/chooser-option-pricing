import pandas as pd
from pathlib import Path

# 1. Load data
jpm = pd.read_csv("data/raw/jpm_daily_full.csv")
treasury = pd.read_csv("data/raw/treasury_daily.csv")
vix = pd.read_csv("data/raw/vix_daily.csv")

# 2. Standardize date columns
jpm["Date"] = pd.to_datetime(jpm["Date"]).dt.normalize()

treasury = treasury.rename(
    columns={"observation_date": "Date"}
)
vix = vix.rename(
    columns={"observation_date": "Date"}
)

treasury["Date"] = pd.to_datetime(treasury["Date"]).dt.normalize()
vix["Date"] = pd.to_datetime(vix["Date"]).dt.normalize()

# 3. Convert macro variables to numeric
treasury["DGS10"] = pd.to_numeric(
    treasury["DGS10"], errors="coerce"
)
vix["VIXCLS"] = pd.to_numeric(
    vix["VIXCLS"], errors="coerce"
)

# 4. Keep required columns
treasury = treasury[["Date", "DGS10"]]
vix = vix[["Date", "VIXCLS"]]

# 5. Left join: preserve every JPM trading date
merged = jpm.merge(treasury, on="Date", how="left")
merged = merged.merge(vix, on="Date", how="left")

merged = merged.sort_values("Date").reset_index(drop=True)

# 6. Count missing macro values before filling
print("Missing before forward fill:")
print(merged[["DGS10", "VIXCLS"]].isna().sum().to_string())

# 7. Forward fill macroeconomic variables
merged[["DGS10", "VIXCLS"]] = (
    merged[["DGS10", "VIXCLS"]].ffill()
)

# 8. Validate
print("\nMissing after forward fill:")
print(merged[["DGS10", "VIXCLS"]].isna().sum().to_string())

print("\nRows:", len(merged))
print("Date range:", merged["Date"].min(), "to", merged["Date"].max())

print("\nYearly counts:")
print(merged.groupby(merged["Date"].dt.year).size().to_string())

# 9. Save as a new file
output = Path("data/processed/jpm_merged_full.csv")
output.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(output, index=False)

print("\nSaved:", output)
