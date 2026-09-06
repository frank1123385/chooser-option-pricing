from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()

api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

if not api_key:
    raise ValueError("ALPHA_VANTAGE_API_KEY not found.")

# Connect to Alpha Vantage
ts = TimeSeries(
    key=api_key,
    output_format="pandas"
)

# Test JPM daily data
data, metadata = ts.get_daily(
    symbol="JPM",
    outputsize="compact"
)

print(data.head())
print("Alpha Vantage API: PASS")
