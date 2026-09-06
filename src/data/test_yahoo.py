import yfinance as yf

# Test JPM data connection
ticker = yf.Ticker("JPM")

data = ticker.history(
    start="2024-01-02",
    end="2024-01-10"
)

print(data)

if not data.empty:
    print("Yahoo Finance: PASS")
else:
    print("Yahoo Finance: NO DATA")
