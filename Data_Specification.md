# Data Requirement Specification

## 1. Objective

The objective of this data collection plan is to define the data
requirements for the Chooser Option Pricing Model project.

## 2. Data Scope

The initial dataset will cover the period from 2018 to 2024
and will use daily observations.

## 3. Data Requirements

| Data Category | Dataset | Variables | Frequency | Period | Source |
|---|---|---|---|---|---|
| Financial | JPM Stock Price | Open, High, Low, Close, Adjusted Close, Volume | Daily | 2018–2024 | Yahoo Finance |
| Macroeconomic | Treasury Rate | Treasury Yield | Daily | 2018–2024 | FRED |
| Volatility | VIX | VIX Value | Daily | 2018–2024 | Yahoo Finance |

## 4. Expected Raw Data

The raw datasets will be stored in CSV format.

Expected files:

- `jpm_daily.csv`
- `treasury_daily.csv`
- `vix_daily.csv`

## 5. Data Sources

- Yahoo Finance
- FRED

## 6. Week 1 Deliverable

The data specification document defines the required datasets,
variables, sources, frequency, and historical period for the
initial data collection.
