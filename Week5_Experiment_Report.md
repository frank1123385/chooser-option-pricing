# Week 5 Experiment Report: Machine Learning Models

## 1. Overview

This report documents the initial machine learning experiments for the Advanced Chooser Option Pricing Model project.

Two approaches were implemented:

- Approach A: Random Forest prediction of next-day absolute stock returns as a short-term volatility proxy.
- Approach B: Supervised prediction of next-day JPM stock returns, converted into implied stock prices.

The experiments use engineered JPM financial data. They do not constitute direct predictions of actual Chooser option market prices.

## 2. Dataset Preparation

The input dataset is `data/processed/jpm_features.csv`.

The dataset contains 1,701 observations. After constructing the next-day absolute-return target and cleaning the data, 1,700 observations remain.

The data was divided chronologically:

| Dataset | Rows | Date Range |
|---|---:|---|
| Training | 1,190 | 2018-03-29 to 2022-12-16 |
| Validation | 255 | 2022-12-19 to 2023-12-22 |
| Testing | 255 | 2023-12-26 to 2024-12-30 |

The time-series split avoids random shuffling and supports evaluation on later observations.

## 3. Approach A: Random Forest Volatility Proxy

### 3.1 Objective

The objective is to predict the absolute return of the next trading observation. This is used as a proxy for short-term price movement magnitude, not as a direct measure of multi-day realized volatility.

### 3.2 Hyperparameter Selection

Three Random Forest configurations were compared on the validation set.

| Number of Trees | Maximum Depth | Minimum Samples per Leaf | Validation MAE | Validation RMSE |
|---:|---:|---:|---:|---:|
| 100 | 5 | 5 | 0.006440 | 0.009028 |
| 200 | 10 | 5 | 0.006895 | 0.009319 |
| 200 | None | 10 | 0.006689 | 0.009147 |

The first configuration achieved the lowest validation MAE and was selected.

### 3.3 Test Results

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| Training-mean baseline | 0.008481 | 0.011892 |
| Random Forest | 0.006623 | 0.011283 |

The Random Forest achieved approximately 21.9% lower MAE and 5.1% lower RMSE than the training-mean baseline on the test subset.

This is a positive preliminary result for predicting the selected return proxy. Further work is needed before interpreting model outputs as volatility estimates suitable for BSM pricing.

### 3.4 Feature Importance

The five most important features reported by the Random Forest were:

| Feature | Importance |
|---|---:|
| Sentiment_Score | 0.233093 |
| VIX_Change_20D | 0.217754 |
| Price_Momentum_20D | 0.087794 |
| VIXCLS | 0.076791 |
| Treasury_Yield_Change_20D | 0.065697 |

`Sentiment_Score` is a VIX-derived proxy, not an independent news sentiment measure. Feature importance indicates predictive contribution within this fitted model and does not establish causality.

## 4. Approach B: End-to-End Stock Return Prediction

### 4.1 Objective

The objective is to predict the next trading day's JPM stock return and convert the predicted return into an implied stock price.

Three models were compared:

- Zero-return baseline
- Linear Regression
- Random Forest Regression

The zero-return baseline predicts that the next trading day's return is zero, meaning the next closing price equals the current closing price.

### 4.2 Validation Results

| Model | Validation MAE | Validation RMSE |
|---|---:|---:|
| Zero-return baseline | 1.3151 | 1.8078 |
| Linear Regression | 1.5028 | 1.9574 |
| Random Forest | 1.3792 | 1.9032 |

The zero-return baseline achieved the lowest validation MAE and was selected as the best model according to the predefined selection criterion.

### 4.3 Test Results

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| Zero-return baseline | 1.992481 | 3.132873 |
| Linear Regression | 2.276945 | 3.334940 |
| Random Forest | 2.131365 | 3.271747 |

The zero-return baseline outperformed both machine learning models on the test subset.

Relative to the baseline, Linear Regression had approximately 14.3% higher MAE, while Random Forest had approximately 7.0% higher MAE.

These findings indicate that the initial machine learning models did not demonstrate an improvement over the simple baseline for next-day JPM stock price prediction.

## 5. Interpretation

The two approaches produced different preliminary outcomes:

- Approach A: Random Forest outperformed its training-mean baseline for next-day absolute-return prediction.
- Approach B: Neither Linear Regression nor Random Forest outperformed the zero-return baseline for next-day stock price prediction.

These findings do not imply that Approach A can already price Chooser options accurately. The target in Approach A is a return proxy rather than verified future volatility, and the end-to-end experiment predicts JPM stock prices rather than option prices.

## 6. Limitations

1. No verified actual CME Chooser option transaction-price labels are currently available in the project dataset.
2. JPM stock prices cannot be used as substitutes for Chooser option prices.
3. BSM-generated Chooser prices are model outputs rather than observed market labels.
4. The sentiment proxy is derived from VIX and does not represent actual news sentiment.
5. The next-day absolute-return target is not equivalent to multi-day realized volatility.
6. The initial end-to-end machine learning models underperformed the zero-return baseline on the current test subset.
7. The experiments are preliminary and do not establish trading profitability or real-market Chooser pricing accuracy.

## 7. Conclusion and Next Steps

Week 5 established two initial machine learning frameworks and documented their preliminary results.

Approach A showed an improvement over its training-mean baseline. Approach B did not improve on the zero-return baseline.

Recommended next steps include verifying reproducibility, reviewing feature timing and target construction, investigating more appropriate volatility targets, and documenting the limits of the available data.

Evaluation of actual Chooser option pricing accuracy remains pending until reliable market price labels become available.

## 8. Reproducibility

Main scripts:

- `src/model/feature_engineering.py`
- `src/model/random_forest_volatility.py`
- `src/model/end_to_end_model.py`

Generated outputs are stored in `data/processed/week5/`.

The reported metrics correspond to the current scripts and dataset. Results should be regenerated if target construction, feature selection, or data splitting changes.