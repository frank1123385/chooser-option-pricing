import pandas as pd

# File paths
jpm_path = "data/raw/jpm_daily.csv"
treasury_path = "data/raw/treasury_daily.csv"
vix_path = "data/raw/vix_daily.csv"

# Load data
jpm = pd.read_csv(jpm_path)
treasury = pd.read_csv(treasury_path)
vix = pd.read_csv(vix_path)

# Standardize date column names
treasury = treasury.rename(columns={"observation_date": "Date"})
vix = vix.rename(columns={"observation_date": "Date"})

# Convert dates
jpm["Date"] = pd.to_datetime(jpm["Date"])
treasury["Date"] = pd.to_datetime(treasury["Date"])
vix["Date"] = pd.to_datetime(vix["Date"])

# Sort by date
jpm = jpm.sort_values("Date")
treasury = treasury.sort_values("Date")
vix = vix.sort_values("Date")

# Save cleaned raw files
treasury.to_csv(treasury_path, index=False)
vix.to_csv(vix_path, index=False)

print("Data preparation complete.")
print(f"JPM rows: {len(jpm)}")
print(f"Treasury rows: {len(treasury)}")
print(f"VIX rows: {len(vix)}")