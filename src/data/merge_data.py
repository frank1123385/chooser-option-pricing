import pandas as pd

# Load data
jpm = pd.read_csv("data/raw/jpm_daily.csv")
treasury = pd.read_csv("data/raw/treasury_daily.csv")
vix = pd.read_csv("data/raw/vix_daily.csv")

# Convert dates
jpm["Date"] = pd.to_datetime(jpm["Date"])
treasury["Date"] = pd.to_datetime(treasury["Date"])
vix["Date"] = pd.to_datetime(vix["Date"])

# Keep the variables needed for the model
jpm = jpm[["Date", "Close", "Adj Close", "Volume"]]
treasury = treasury[["Date", "DGS10"]]
vix = vix[["Date", "VIXCLS"]]

# Merge on date
data = jpm.merge(treasury, on="Date", how="inner")
data = data.merge(vix, on="Date", how="inner")

# Sort by date
data = data.sort_values("Date")
data["DGS10"] = data["DGS10"].ffill()

# Save merged dataset
output_path = "data/processed"
import os
os.makedirs(output_path, exist_ok=True)

data.to_csv(
    f"{output_path}/jpm_merged.csv",
    index=False,
    date_format="%Y-%m-%d"
)

print("Merge complete.")
print(f"Rows: {len(data)}")
print(f"Columns: {list(data.columns)}")
print(f"Start: {data['Date'].iloc[0].date()}")
print(f"End: {data['Date'].iloc[-1].date()}")
print("\nFirst 5 rows:")
print(data.head())