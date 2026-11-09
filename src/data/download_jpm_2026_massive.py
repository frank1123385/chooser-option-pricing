
from pathlib import Path
import os
import requests
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("MASSIVE_API_KEY")

if not API_KEY:
    raise ValueError("MASSIVE_API_KEY not found")

url = (
    "https://api.massive.com/v2/aggs/ticker/JPM/"
    "range/1/day/2026-01-01/2026-11-07"
)

response = requests.get(
    url,
    params={
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    },
    timeout=30,
)

response.raise_for_status()
data = response.json()
results = data.get("results", [])

if not results:
    raise ValueError(f"No data returned: {data}")

df = pd.DataFrame(results)

df["Date"] = (
    pd.to_datetime(df["t"], unit="ms", utc=True)
    .dt.tz_localize(None)
    .dt.normalize()
)

df = df.rename(columns={
    "o": "Open",
    "h": "High",
    "l": "Low",
    "c": "Close",
    "v": "Volume",
})

df["Adj Close"] = df["Close"]

df = df[
    ["Date", "Open", "High", "Low",
     "Close", "Adj Close", "Volume"]
]

df = df.sort_values("Date").drop_duplicates("Date")

output = ROOT / "data/raw/jpm_2026_massive.csv"
df.to_csv(output, index=False, date_format="%Y-%m-%d")

print("Download completed.")
print("Rows:", len(df))
print("Start:", df["Date"].min())
print("End:", df["Date"].max())
print("Saved:", output)
