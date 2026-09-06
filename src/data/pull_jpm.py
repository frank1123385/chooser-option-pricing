from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import os

load_dotenv("../../.env")

api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

if not api_key:
    raise ValueError("ALPHA_VANTAGE_API_KEY not found.")

ts = TimeSeries(
    key=api_key,
    output_format="pandas"
)

data, metadata = ts.get_daily(
    symbol="JPM",
    outputsize="compact"
)

import os

os.makedirs("../../data/raw", exist_ok=True)

data.to_csv("../../data/raw/jpm_prices.csv")

print(f"Rows downloaded: {len(data)}")
print("Saved to data/raw/jpm_prices.csv")