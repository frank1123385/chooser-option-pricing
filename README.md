# Chooser Option Pricing Model

## 1. Project Overview

This project develops a Monte Carlo-based Chooser Option Pricing Model using real-world financial data and machine learning.

JPMorgan Chase (JPM) is selected as the underlying asset. The project combines historical stock prices, Treasury yields, VIX data, volatility forecasting, and two-stage option pricing.

The final application is built with Streamlit.

## 2. Main Features

- Financial data collection and preprocessing.
- JPM stock price and macroeconomic data integration.
- Financial feature engineering.
- Random Forest volatility forecasting.
- 20-day, 126-day, and 252-day forecasting horizons.
- Historical 60-day volatility benchmark.
- Two-stage Geometric Brownian Motion simulation.
- Monte Carlo Chooser Option pricing.
- Historical volatility versus Random Forest comparison.
- Volatility sensitivity analysis.
- Interactive Streamlit dashboard.

## 3. Data Sources

| Dataset | Source | Description |
|---|---|---|
| JPM | Yahoo Finance / Alpha Vantage | Daily stock prices and volume |
| Treasury | FRED | 10-year Treasury yield (DGS10) |
| VIX | Yahoo Finance / FRED | Market volatility indicator |

The full JPM dataset covers January 2018 to November 2026.

## 4. Installation

Clone the repository:

```bash
git clone https://github.com/frank1123385/chooser-option-pricing.git
cd chooser-option-pricing
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 5. Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the following address in your browser if it does not open automatically:

http://localhost:8501
## 6. Model Methodology

The pricing engine uses a two-stage Monte Carlo simulation.

At the choice date T1, the holder chooses either a call or a put according to the underlying stock price and strike price.

The selected option is then simulated until the final maturity date T2.

Two volatility approaches are compared:
1. Historical 60-day realized volatility.
2. Random Forest forecasts using 126-day and 252-day horizons.
The theoretical option price is estimated from the average discounted simulated payoff.
## 7. Machine Learning Evaluation

Random Forest models forecast future realized volatility over:
- 20 trading days.
- 126 trading days.
- 252 trading days.
Performance is evaluated using MAE, RMSE, and R-squared, with chronological training, validation, and test datasets.
## 8. Main Results

The Week 8 dual-pricing experiment contains 242 common valuation dates.
| Pricing Method | Mean Theoretical Price |
|---|---:|
| Historical 60D | $127.6780 |
| Random Forest | $127.8948 |
The average price difference is approximately $0.2168, or 0.1035%.

Random Forest does not outperform the historical benchmark at every forecasting horizon.
## 9. Limitations

- Theoretical prices have not been validated against matched market Chooser option quotes.
- DGS10 is used as a proxy for the risk-free rate.
- Dividend effects are not explicitly incorporated into the two-stage GBM dynamics.
- Random Forest performs less well than the historical benchmark at the 126-day and 252-day test horizons.
- Sentiment_Score is a VIX-derived proxy, not actual news sentiment.
This application is an experimental research and educational tool, not a market-validated production pricing system.
## 10. Data Updates

Optional market-data updates use Alpha Vantage and FRED API credentials.

Users who want to enable data updates must configure their own API keys in a local .env file.

Never upload API keys or the .env file to GitHub.
## 11. Repository

GitHub: https://github.com/frank1123385/chooser-option-pricing
