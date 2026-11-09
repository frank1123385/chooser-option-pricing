import pandas as pd

files = {
    "Treasury": ("data/raw/treasury_daily.csv", "DGS10"),
    "VIX": ("data/raw/vix_daily.csv", "VIXCLS")
}

for name, (file, value_col) in files.items():
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    missing = df.loc[df[value_col].isna(), "Date"].sort_values()

    print("\n==========", name, "==========")
    print("Missing dates:", len(missing))
    print("\nFirst 15 missing dates:")
    print(missing.head(15).to_string(index=False))

    # 计算缺失日期之间的间隔
    if len(missing) > 1:
        gaps = missing.diff().dt.days.dropna()
        print("\nLargest gap between missing dates:", gaps.max(), "days")

    # 检查连续自然日缺失的情况
    groups = missing.diff().dt.days.ne(1).cumsum()
    streaks = missing.groupby(groups).agg(["min", "max", "count"])
    print("\nLongest consecutive missing-date groups:")
    print(streaks.sort_values("count", ascending=False).head(10).to_string())
