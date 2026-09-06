from fredapi import Fred
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()

api_key = os.getenv("FRED_API_KEY")

if not api_key:
    raise ValueError("FRED_API_KEY not found.")

# Connect to FRED
fred = Fred(api_key=api_key)

# Test Treasury data
data = fred.get_series(
    "DGS10",
    observation_start="2024-01-02",
    observation_end="2024-01-10"
)

print(data)
