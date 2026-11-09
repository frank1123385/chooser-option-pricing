import pandas as pd

jpm = pd.read_csv("data/raw/jpm_daily_full.csv")
treasury = pd.read_csv("data/raw/treasury_daily.csv")
vix = pd.read_csv("data/raw/vix_daily.csv")

for df in [jpm, treasury, vix]:
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

jpm_dates = set(jpm["Date"])

for name, df, col in [
    ("Treasury", treasury, "DGS10"),
    ("VIX", vix, "VIXCLS")
]:
    missing = df.loc[df[col].isna(), "Date"]
    affected = sorted(set(missing) & jpm_dates)

    print("\n==========", name, "==========")
    print("Total missing dates:", len(missing))
    print("Missing dates that are JPM trading dates:", len(affected))

    if affected:
        print("\nAffected dates:")
        print(pd.Series(affected).to_string(index=False))
    else:
        print("No JPM trading dates are affected.")
