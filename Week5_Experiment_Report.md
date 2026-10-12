# Week 5 Experiment Report: Machine Learning Models

## 1. Overview

This report documents the initial machine learning experiments for the Advanced Chooser Option Pricing Model project.

Two approaches were implemented:

- **Approach A:** Random Forest prediction of next-day absolute stock returns as a short-term volatility proxy.
- **Approach B:** Supervised prediction of next-day JPM stock returns, converted into implied stock prices.

The experiments use engineered JPM financial data. They do not constitute direct predictions of actual Chooser option market prices.

## 2. Dataset Preparation

The input dataset is `data/processed/jpm_features.csv`.

After constructing the next-trading-day return targets and removing observations with missing or invalid feature or target values, 1,509 observations remained.

The data was divided chronologically into training, validation, and testing subsets. Boundary observations were excluded when their target dates fell outside their respective feature-date intervals.

| Dataset | Rows | Feature Date Range |
|---|---:|---|
| Training | 1,055 | 2019-01-02 to 2023-03-10 |
| Validation | 225 | 2023-03-14 to 2024-02-02 |
| Testing | 226 | 2024-02-06 to 2024-12-27 |

The time-series split avoids random shuffling and supports evaluation on later observations. The generated dataset summary is stored in `data/processed/week5/split_summary.csv`.

## 3. Approach A: Random Forest Volatility Proxy

### 3.1 Objective

The objective is to predict the absolute return of the next trading observation. This is used as a proxy for short-term price movement magnitude, not as a direct measure of multi-day realized volatility or annualized implied volatility.

### 3.2 Hyperparameter Selection

Three Random Forest configurations were compared on the validation set. The configuration with the lowest validation MAE was selected.

| Number of Trees | Maximum Depth | Minimum Samples per Leaf | Validation MAE | Validation RMSE |
|---:|---:|---:|---:|---:|
| 100 | 5 | 5 | 0.006426 | 0.008863 |
| 200 | 10 | 5 | 0.006902 | 0.009269 |
| 200 | None | 10 | 0.006334 | 0.008784 |

The selected configuration used 200 trees, no maximum depth limit, and a minimum of 10 samples per leaf.

### 3.3 Test Results

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| Training-mean baseline | 0.008762 | 0.012369 |
| Random Forest | 0.006992 | 0.011979 |

Relative to the training-mean baseline, Random Forest reduced test MAE by approximately 20.2% and test RMSE by approximately 3.2%.

This is a positive preliminary result for predicting next-day absolute returns. However, the target is a return proxy rather than a directly observed volatility measure. Additional work is required before using these predictions as volatility inputs for BSM-based Chooser pricing.

### 3.4 Feature Importance

The five most important features reported by the selected Random Forest were:

| Feature | Importance |
|---|---:|
| Sentiment_Score | 0.245436 |
| Price_Momentum_20D | 0.156598 |
| VIX_Change_20D | 0.151228 |
| VIXCLS | 0.079950 |
| Volume | 0.063423 |

`Sentiment_Score` is a VIX-derived proxy, not an independent news sentiment measure. Feature importance indicates predictive contribution within this fitted model and does not establish causality.

## 4. Approach B: End-to-End Stock Return Prediction

### 4.1 Objective

The objective is to predict the next trading day's JPM stock return and convert the predicted return into an implied adjusted closing price.

Three models were compared:

- Zero-return baseline
- Linear Regression
- Random Forest Regression

The zero-return baseline predicts that the next trading day's return is zero, meaning the next adjusted closing price equals the current adjusted closing price.

Price metrics are calculated on an adjusted-close basis to match the return target derived from adjusted closing prices.

### 4.2 Validation Results

| Model | Return MAE | Return RMSE | Price MAE | Price RMSE |
|---|---:|---:|---:|---:|
| Zero-return baseline | 0.008573 | 0.012221 | 1.163689 | 1.605649 |
| Linear Regression | 0.009617 | 0.013074 | 1.315864 | 1.753076 |
| Random Forest | 0.008501 | 0.012202 | 1.153667 | 1.603531 |

Random Forest achieved the lowest validation price MAE and was selected as the validation-preferred model.

### 4.3 Test Results

| Model | Return MAE | Return RMSE | Price MAE | Price RMSE |
|---|---:|---:|---:|---:|
| Zero-return baseline | 0.010055 | 0.015445 | 2.018938 | 3.158432 |
| Linear Regression | 0.011753 | 0.016788 | 2.348106 | 3.405407 |
| Random Forest | 0.010108 | 0.015658 | 2.032370 | 3.195705 |

The zero-return baseline achieved the lowest test MAE and RMSE for both return prediction and adjusted-price prediction.

Although Random Forest performed best on the validation set, it did not outperform the zero-return baseline on the test set. Linear Regression also underperformed the baseline.

These findings suggest that the initial end-to-end machine learning models did not demonstrate improved out-of-sample performance over the simple baseline for next-day JPM stock return prediction.

## 5. Interpretation

The two approaches produced different preliminary outcomes:

- **Approach A:** Random Forest outperformed its training-mean baseline for next-day absolute-return prediction.
- **Approach B:** Neither Linear Regression nor Random Forest outperformed the zero-return baseline on the test set.

The MAE and RMSE values from the two approaches should not be compared directly because they predict different targets: Approach A predicts absolute returns, whereas Approach B predicts signed returns and evaluates the resulting adjusted-price estimates.

These findings do not establish that Approach A can accurately price Chooser options. Its target is not verified future volatility, and Approach B predicts JPM stock prices rather than option prices.

## 6. Limitations

1. No verified actual CME Chooser option transaction-price labels are currently available in the project dataset.
2. JPM stock prices cannot be used as substitutes for Chooser option prices.
3. BSM-generated Chooser prices are model outputs rather than observed market labels.
4. The sentiment proxy is derived from VIX and does not represent actual news sentiment.
5. The next-day absolute-return target is not equivalent to multi-day realized volatility or annualized implied volatility.
6. The initial end-to-end machine learning models underperformed the zero-return baseline on the current test subset.
7. The experiments are preliminary and do not establish trading profitability or real-market Chooser pricing accuracy.
8. The reported results depend on the current feature engineering, target construction, chronological split, and model configuration.

## 7. Conclusion and Next Steps

Week 5 established two initial machine learning frameworks and documented their preliminary results.

Approach A showed an improvement over its training-mean baseline for next-day absolute-return prediction. Approach B did not improve on the zero-return baseline for next-day stock return and adjusted-price prediction.

Recommended next steps include verifying reproducibility, reviewing feature timing and target construction, investigating more appropriate volatility targets, and documenting the limits of the available data.

Evaluation of actual Chooser option pricing accuracy remains pending until reliable market price labels become available.

## 8. Reproducibility

Main scripts:

- `src/model/feature_engineering.py`
- `src/model/random_forest_volatility.py`
- `src/model/end_to_end_model.py`

The feature preprocessing script is `src/data/preprocess_data.py`.

Generated outputs are stored in `data/processed/week5/`, including split datasets, model metrics, test predictions, and Random Forest feature importance.

The reported metrics correspond to the latest recorded script outputs and dataset split. Results should be regenerated if target construction, feature selection, data splitting, or model configuration changes.
