import pandas as pd

files = [
    "data/raw/treasury_daily.csv",
    "data/raw/vix_daily.csv"
]

for file in files:
    print("\n==========", file, "==========")

    df = pd.read_csv(file)

    date_col = "Date" if "Date" in df.columns else "observation_date"
    df[date_col] = pd.to_datetime(df[date_col])

    print("Rows:", len(df))
    print("Date range:", df[date_col].min(), "to", df[date_col].max())
    print("Duplicate dates:", df[date_col].duplicated().sum())

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print("\nYearly counts:")
    print(df.groupby(df[date_col].dt.year).size().to_string())
