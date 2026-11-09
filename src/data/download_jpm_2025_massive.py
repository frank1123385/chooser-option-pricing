
from pathlib import Path
import os
import requests
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("MASSIVE_API_KEY")

if not API_KEY:
    raise ValueError("MASSIVE_API_KEY not found in .env")

url = (
    "https://api.massive.com/v2/aggs/ticker/JPM/"
    "range/1/day/2025-01-01/2025-12-31"
)

params = {
    "adjusted": "true",
    "sort": "asc",
    "limit": 50000,
    "apiKey": API_KEY,
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

data = response.json()

if data.get("status") == "ERROR":
    raise RuntimeError(data)

results = data.get("results", [])

if not results:
    raise ValueError(
        f"No data returned. API response: {data}"
    )

df = pd.DataFrame(results)

df["Date"] = pd.to_datetime(
    df["t"], unit="ms", utc=True
).dt.tz_localize(None).dt.normalize()

df = df.rename(columns={
    "o": "Open",
    "h": "High",
    "l": "Low",
    "c": "Close",
    "v": "Volume",
})

# Massive aggregate close is split-adjusted.
# It is not the same as Yahoo dividend-adjusted close.
df["Adj Close"] = df["Close"]

df = df[
    ["Date", "Open", "High", "Low",
     "Close", "Adj Close", "Volume"]
]

df = df.sort_values("Date").drop_duplicates(
    subset="Date", keep="last"
)

output = ROOT / "data" / "raw" / "jpm_2025_massive.csv"
df.to_csv(output, index=False, date_format="%Y-%m-%d")

print("Download completed.")
print("Rows:", len(df))
print("Start:", df["Date"].min())
print("End:", df["Date"].max())
print("Saved:", output)
print(df.head(3).to_string(index=False))
print(df.tail(3).to_string(index=False))
