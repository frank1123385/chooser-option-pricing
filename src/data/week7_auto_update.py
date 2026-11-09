
from pathlib import Path
from datetime import datetime, timedelta
import os
import pandas as pd

from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
from fredapi import Fred


# =========================================================
# 1. Project paths and API keys
# =========================================================

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

JPM_FILE = RAW_DIR / "jpm_daily.csv"
TREASURY_FILE = RAW_DIR / "treasury_daily.csv"
VIX_FILE = RAW_DIR / "vix_daily.csv"

load_dotenv(ROOT / ".env")

ALPHA_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FRED_KEY = os.getenv("FRED_API_KEY")


# =========================================================
# 2. Common update function
# =========================================================

def merge_and_save(old_file, new_data, date_col="Date"):
    """Merge new observations with existing data without losing history."""

    if old_file.exists():
        old_data = pd.read_csv(old_file)
    else:
        old_data = pd.DataFrame()

    old_rows = len(old_data)

    if new_data.empty:
        print(f"No new data received for {old_file.name}.")
        return

    old_adj_close = None

    if "Adj Close" in old_data.columns:
        old_adj_close = old_data.set_index(date_col)["Adj Close"]

    combined = pd.concat(
        [old_data, new_data],
        ignore_index=True
    )

    combined[date_col] = pd.to_datetime(
        combined[date_col],
        errors="coerce"
    )

    combined = combined.dropna(subset=[date_col])
    combined = combined.sort_values(date_col)
    combined = combined.drop_duplicates(
        subset=[date_col],
        keep="last"
    )

    # Preserve historical adjusted closing prices.
    # Alpha Vantage get_daily does not provide Adj Close.
    if old_adj_close is not None:
        combined["Adj Close"] = combined[date_col].dt.strftime(
            "%Y-%m-%d"
        ).map(old_adj_close)

        combined["Adj Close"] = combined["Adj Close"].fillna(
            combined["Close"]
        )

    combined[date_col] = combined[date_col].dt.strftime("%Y-%m-%d")

    combined.to_csv(old_file, index=False)

    print(f"\nUpdated: {old_file.name}")
    print(f"Rows before: {old_rows}")
    print(f"Rows after:  {len(combined)}")
    print(f"New rows:    {len(combined) - old_rows}")
    print(f"Latest date: {combined[date_col].max()}")


# =========================================================
# 3. Update JPM stock prices
# =========================================================

def update_jpm():

    print("\n========== Updating JPM ==========")

    if not ALPHA_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY not found in .env")

    ts = TimeSeries(
        key=ALPHA_KEY,
        output_format="pandas"
    )

    data, metadata = ts.get_daily(
        symbol="JPM",
        outputsize="compact"
    )

    data = data.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    })

    data.index = pd.to_datetime(data.index)

    data = data.rename_axis("Date").reset_index()

    data["Adj Close"] = data["Close"]

    data = data[
        ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    ]

    merge_and_save(JPM_FILE, data)

    print(
        "Note: Alpha Vantage get_daily does not provide "
        "adjusted close. New observations use Close as a proxy."
    )


# =========================================================
# 4. Update FRED data
# =========================================================

def get_fred_updates(series_id, output_column, output_file):

    if not FRED_KEY:
        raise ValueError("FRED_API_KEY not found in .env")

    fred = Fred(api_key=FRED_KEY)

    if output_file.exists():
        old = pd.read_csv(output_file)

        last_date = pd.to_datetime(old["Date"]).max()

        # Retrieve an overlapping period to avoid missing observations.
        start_date = (
            last_date - timedelta(days=14)
        ).strftime("%Y-%m-%d")
    else:
        start_date = "2018-01-01"

    end_date = datetime.today().strftime("%Y-%m-%d")

    series = fred.get_series(
        series_id,
        observation_start=start_date,
        observation_end=end_date
    )

    new_data = series.rename(output_column).reset_index()
    new_data.columns = ["Date", output_column]

    new_data = new_data.dropna(subset=[output_column])

    return new_data


def update_treasury():

    print("\n========== Updating Treasury ==========")

    data = get_fred_updates(
        "DGS10",
        "DGS10",
        TREASURY_FILE
    )

    merge_and_save(TREASURY_FILE, data)


def update_vix():

    print("\n========== Updating VIX ==========")

    data = get_fred_updates(
        "VIXCLS",
        "VIXCLS",
        VIX_FILE
    )

    merge_and_save(VIX_FILE, data)


# =========================================================
# 5. Main execution
# =========================================================

def main():

    print("=" * 55)
    print("WEEK 7 - AUTOMATIC MARKET DATA UPDATE")
    print("=" * 55)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    errors = []

    tasks = [
        ("JPM", update_jpm),
        ("Treasury", update_treasury),
        ("VIX", update_vix)
    ]

    for name, task in tasks:
        try:
            task()
        except Exception as error:
            errors.append((name, str(error)))
            print(f"\nERROR updating {name}: {error}")

    print("\n========== UPDATE SUMMARY ==========")

    if errors:
        for name, error in errors:
            print(f"{name}: FAILED - {error}")
        print("Some data sources failed. Check the errors above.")
    else:
        print("All three data sources updated successfully.")

    print("\nUpdate process finished.")


if __name__ == "__main__":
    main()
